//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile_core::ir::{Block, BlockId, Kernel, Op, ValueId};

use super::remap;
use crate::error::{Error, Result};

pub struct ValueSinkPass;

impl super::Pass for ValueSinkPass {
    fn name(&self) -> &str { "value_sink" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Block-local sinking, with use-counts computed across ALL blocks
        // (body + every nested loop / if body) so that a value defined in
        // the outer block but consumed inside an inner block is not treated
        // as single-use and sunk past the op that owns the nested block.
        let global_use_count = compute_global_use_count(kernel);

        sink_in_block(&mut kernel.body, &global_use_count);

        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            let mut block =
                kernel.blocks.remove(&bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            sink_in_block(&mut block, &global_use_count);
            kernel.blocks.insert(bid, block);
        }

        super::dead_value_elim::eliminate_dead_values(kernel)?;
        Ok(())
    }
}

/// Count every use of every `ValueId` across the kernel's entry body and all
/// stored blocks. Sinking is intra-block (it only reorders ops within one
/// block) so this count never goes stale as sinking proceeds.
fn compute_global_use_count(kernel: &Kernel) -> BTreeMap<ValueId, usize> {
    let mut use_count: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut tally = |ops: &[Op]| {
        for op in ops {
            for vid in remap::op_value_refs(op) {
                *use_count.entry(vid).or_default() += 1;
            }
        }
    };
    tally(&kernel.body.ops);
    for block in kernel.blocks.values() {
        tally(&block.ops);
    }
    use_count
}

// ---------------------------------------------------------------------------
// eligibility
// ---------------------------------------------------------------------------

/// Ops eligible for sinking: cheap ALU ops that are better recomputed than kept alive.
fn is_sinkable(op: &Op) -> bool { remap::is_cheap_alu(op) && !remap::has_side_effects(op) }

/// Ops that block sinking across them (Barrier).
fn is_sink_barrier(op: &Op) -> bool { op.is_barrier() }

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

fn sink_in_block(block: &mut Block, global_use_count: &BTreeMap<ValueId, usize>) {
    let n = block.ops.len();
    if n == 0 {
        return;
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

        // Must have exactly one use *across the whole kernel*. A value
        // referenced inside a nested loop / if body must not be sunk past
        // the op that contains the nested block, even if its only
        // straight-line use sits below that op.
        if global_use_count.get(result_vid) != Some(&1) {
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
        // Anchor the Mul result against per-pass DCE (#209/1) — without
        // this, `v4` (Mul) has no consumer and DCE strips both Mul and
        // its upstream Add, masking the "barrier-blocks-sinking"
        // invariant we're trying to assert.
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(4),
            mask: None,
        });

        let add_pos_before =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        ValueSinkPass.run(&mut k).unwrap();
        let add_pos_after =
            k.body.ops.iter().position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));

        // Should not have moved (barrier in between).
        assert_eq!(add_pos_before, add_pos_after);
    }

    /// Regression: a value defined in the outer block and used inside a nested
    /// loop body must not be sunk past the loop op, even if its only
    /// straight-line use in the outer block sits below the loop.
    /// `sdpa_decode` triggered this — `k_base` lives in the outer kv-walk
    /// loop and is consumed both inside the inner dot-product loop and
    /// after it.
    #[test]
    fn does_not_sink_value_used_in_nested_block() {
        use metaltile_core::ir::{Block, BlockId};

        let mut k = Kernel::new("sink_cross_block");

        // Outer block (body):
        //   v0 = const 1
        //   v1 = const 2
        //   v2 = add(v0, v1)        ← defined here
        //   v3 = const 10           ← filler
        //   loop body=inner_block (uses v2)
        //   v4 = mul(v2, v3)        ← straight-line use below loop
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(3));

        // Inner block uses v2 (so v2 has 2 uses total: inside loop + after loop).
        let inner_id = BlockId::new(1);
        let mut inner = Block::new(inner_id);
        inner.push_op(Op::Const { value: 0 }, ValueId::new(100));
        inner.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(2), rhs: ValueId::new(100) },
            ValueId::new(101),
        );
        k.blocks.insert(inner_id, inner);

        k.body.push_op_no_result(Op::Loop {
            var: metaltile_core::ir::VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(3),
            step: ValueId::new(0),
            body: inner_id,
        });
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(4),
        );

        // Snapshot the position of the Add op relative to the Loop op.
        let add_pos_before = k
            .body
            .ops
            .iter()
            .position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }))
            .unwrap();
        let loop_pos_before =
            k.body.ops.iter().position(|op| matches!(op, Op::Loop { .. })).unwrap();
        assert!(add_pos_before < loop_pos_before, "test setup: add must precede loop");

        ValueSinkPass.run(&mut k).unwrap();

        // The Add must STILL sit above the Loop — sinking past the loop
        // would leave the inner-block reference dangling.
        let add_pos_after = k
            .body
            .ops
            .iter()
            .position(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }))
            .unwrap();
        let loop_pos_after =
            k.body.ops.iter().position(|op| matches!(op, Op::Loop { .. })).unwrap();
        assert!(
            add_pos_after < loop_pos_after,
            "value defined in outer block and used inside a nested loop must not be \
             sunk past the loop op (got add at {add_pos_after}, loop at {loop_pos_after})"
        );
    }
}
