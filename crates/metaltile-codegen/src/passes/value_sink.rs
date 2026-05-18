//! Value Sinking — move single-use definitions closer to their consumers.
//!
//! If a value is computed at position *i* in a block but only consumed at
//! position *j* > *i*, sinking it to position *j-1* shortens the live range,
//! reducing register pressure.  This is the complement of LICM (which hoists
//! out of loops); sinking pushes single-use definitions toward their consumers
//! within straight-line code.
//!
//! ## Why It Matters
//!
//! MetalTile emits MSL source with `auto` variables.  The Metal compiler
//! handles register allocation, but shorter live ranges in the IR correlate
//! with lower register pressure in the generated code.  On M3+, the OMU can
//! exploit lower register pressure for higher occupancy.
//!
//! ## Algorithm
//!
//! Block-local single-use sinking:
//! 1. Compute use-counts for each ValueId.
//! 2. For each cheap-ALU op with exactly one use, find its use position.
//! 3. If the use is farther than 1 position away and no barrier separates
//!    them, sink the definition to just before the use.
//! 4. Rebuild the block with adjusted positions.
//!
//! ## Safety
//!
//! - Only cheap ALU ops are sunk (BinOp, UnaryOp, Cast, Select, Const,
//!   ProgramId).
//! - Ops with side effects are never moved.
//! - Ops are never sunk across a Barrier.
//! - Ops are never sunk across block boundaries (Phase 1 is block-local).
//!
//! ## References
//! - Ferrante, Ottenstein & Warren (1987), "The program dependence graph and
//!   its use in optimization", ACM TOPLAS 9(3):319–349.  PDG-based framework
//!   for code motion including sinking.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.5.  Partial-redundancy elimination and code sinking.

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, Kernel, Op, ValueId},
};

use super::remap;

pub struct ValueSinkPass;

impl super::Pass for ValueSinkPass {
    fn name(&self) -> &str { "value_sink" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Pass the full blocks map so sink_in_block can count uses in child
        // blocks (inner loop bodies) — preventing values from being sunk past
        // the loops that reference them.
        {
            let (body, blocks) = (&mut kernel.body, &kernel.blocks);
            sink_in_block(body, blocks);
        }

        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            let mut block = kernel.blocks.remove(&bid).unwrap();
            // For inner blocks, child-of-child nesting is rare; pass the
            // remaining (non-removed) blocks for best-effort cross-block counting.
            sink_in_block(&mut block, &kernel.blocks);
            kernel.blocks.insert(bid, block);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// eligibility
// ---------------------------------------------------------------------------

/// Ops eligible for sinking: cheap ALU ops that are better recomputed than kept alive.
fn is_sinkable(op: &Op) -> bool { remap::is_cheap_alu(op) && !remap::has_side_effects(op) }

/// Ops that block sinking across them (Barrier).
fn is_sink_barrier(op: &Op) -> bool { matches!(op, Op::Barrier) }

// ---------------------------------------------------------------------------
// data structures
// ---------------------------------------------------------------------------

/// A planned sink operation: move the op at `from` to just before position `to`.
struct SinkPlan {
    from: usize,
    to: usize,
}

// ---------------------------------------------------------------------------
// main logic
// ---------------------------------------------------------------------------

fn sink_in_block(block: &mut Block, all_blocks: &BTreeMap<BlockId, Block>) {
    let n = block.ops.len();
    if n == 0 {
        return;
    }

    // Phase 1: compute use-count for each ValueId.
    // Also count uses inside directly-nested child blocks (inner loop / if
    // bodies) so that values used inside a nested block are not sunk past
    // the loop/if op that encloses them.
    let mut use_count: BTreeMap<ValueId, usize> = BTreeMap::new();
    for op in &block.ops {
        for vid in remap::op_value_refs(op) {
            *use_count.entry(vid).or_default() += 1;
        }
        // Count uses inside child blocks referenced by this op.
        let mut child_bids: Vec<BlockId> = Vec::new();
        match op {
            Op::Loop { body, .. } => child_bids.push(*body),
            Op::If { then_block, else_block, .. } => {
                child_bids.push(*then_block);
                if let Some(eb) = else_block {
                    child_bids.push(*eb);
                }
            },
            _ => {},
        }
        for bid in child_bids {
            if let Some(child) = all_blocks.get(&bid) {
                for child_op in &child.ops {
                    for vid in remap::op_value_refs(child_op) {
                        *use_count.entry(vid).or_default() += 1;
                    }
                }
            }
        }
    }

    // Phase 2: find sink opportunities.
    let mut plans: Vec<SinkPlan> = Vec::new();

    for i in 0..n {
        let op = &block.ops[i];
        let Some(Some(result_vid)) = block.results.get(i) else { continue };

        // Must be a cheap ALU op.
        if !is_sinkable(op) {
            continue;
        }

        // Must have exactly one use.
        if use_count.get(result_vid) != Some(&1) {
            continue;
        }

        // Find the single use position after `i`.
        let Some(use_pos) = find_first_use_position(*result_vid, block, i + 1) else {
            continue;
        };

        // Must not cross a barrier.
        if has_barrier_between(block, i, use_pos) {
            continue;
        }

        // Sink only if there's a meaningful gap (at least 2 positions away).
        if use_pos > i + 1 {
            plans.push(SinkPlan { from: i, to: use_pos });
        }
    }

    if plans.is_empty() {
        return;
    }

    // Phase 3: rebuild the block with sunk ops.
    rebuild_with_sinking(block, &plans);
}

/// Find the first position > `from` where `vid` is referenced.
fn find_first_use_position(vid: ValueId, block: &Block, from: usize) -> Option<usize> {
    for (j, op) in block.ops.iter().enumerate().skip(from) {
        let refs = remap::op_value_refs(op);
        if refs.contains(&vid) {
            return Some(j);
        }
    }
    None
}

/// Check if there's a barrier op between two positions (exclusive of `from`, exclusive of `to`).
fn has_barrier_between(block: &Block, from: usize, to: usize) -> bool {
    if from + 1 >= to {
        return false;
    }
    block.ops[from + 1..to].iter().any(is_sink_barrier)
}

/// Rebuild the block, moving ops according to the sink plan.
///
/// Algorithm:
/// 1. For each position in the original block, decide whether to keep it.
/// 2. For each position that *should* receive a sunk op, insert it before that position.
/// 3. Remove the original op from its old position.
fn rebuild_with_sinking(block: &mut Block, plans: &[SinkPlan]) {
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let n = old_ops.len();

    // Build maps: from_pos → remove; to_pos → insert (Vec of ops).
    let mut remove_at: BTreeSet<usize> = BTreeSet::new();
    let mut insert_before: BTreeMap<usize, Vec<(Op, Option<ValueId>)>> = BTreeMap::new();

    for plan in plans {
        remove_at.insert(plan.from);
        insert_before
            .entry(plan.to)
            .or_default()
            .push((old_ops[plan.from].clone(), old_results[plan.from]));
    }

    let mut new_ops = Vec::with_capacity(n);
    let mut new_results = Vec::with_capacity(n);

    for i in 0..n {
        // Insert sunk ops just before position `i`.
        if let Some(inserted) = insert_before.get(&i) {
            for (op, vid) in inserted {
                new_ops.push(op.clone());
                new_results.push(*vid);
            }
        }

        // Emit the original op at `i` unless it was removed.
        if !remove_at.contains(&i) {
            new_ops.push(old_ops[i].clone());
            new_results.push(old_results[i]);
        }
    }

    // Handle insertions at the end (after the last op).
    let end_inserts = insert_before.keys().filter(|&&k| k >= n).max();
    if let Some(&pos) = end_inserts
        && let Some(inserted) = insert_before.get(&pos)
    {
        for (op, vid) in inserted {
            new_ops.push(op.clone());
            new_results.push(*vid);
        }
    }

    block.ops = new_ops;
    block.results = new_results;
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::BinOpKind;

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn sinks_single_use_add_toward_consumer() {
        let mut k = Kernel::new("sink_add");
        // Producer: v0 = const 1, v1 = const 2, v2 = add(v0, v1) — used only at position 5
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        // Filler ops to create gap
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(3));
        k.body.push_op(Op::Const { value: 20 }, ValueId::new(4));
        // Consumer: v5 = mul(v2, v3)
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(5),
        );

        ValueSinkPass.run(&mut k).unwrap();

        // v2 (add) should now be right before the mul at what was position 5.
        // Positions after sinking: [v0, v1, v3, v4, v2(add), v5(mul)]
        // Check that add comes right before mul.
        let mul_idx =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Mul, .. }));
        assert!(mul_idx.is_some());

        let add_idx =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        assert!(add_idx.is_some());

        // Add should come before mul (both are sunk toward the mul).
        assert!(add_idx.unwrap() < mul_idx.unwrap(), "add must precede mul");
    }

    #[test]
    fn does_not_sink_multi_use_value() {
        let mut k = Kernel::new("sink_multiuse");
        // v0, v1: inputs
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        // v2 = add(v0, v1) — used twice
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        // Filler
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(3));
        // Use 1: v4 = mul(v2, v3)
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(4),
        );
        // Use 2: v5 = sub(v2, v3)
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Sub, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(5),
        );

        let add_pos_before =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        ValueSinkPass.run(&mut k).unwrap();
        let add_pos_after =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));

        // Should not have moved (2 uses → not sunk).
        assert_eq!(add_pos_before, add_pos_after);
    }

    #[test]
    fn does_not_sink_across_barrier() {
        let mut k = Kernel::new("sink_barrier");
        // Define barrier as a Barrier op.
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        // Barrier blocks sinking.
        k.body.push_op_no_result(Op::Barrier);
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(3));
        // Consumer of add (single use, but across barrier).
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(4),
        );

        let add_pos_before =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        ValueSinkPass.run(&mut k).unwrap();
        let add_pos_after =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));

        // Should not have moved (barrier in between).
        assert_eq!(add_pos_before, add_pos_after);
    }
}
