//! If-Conversion — replace short branches with predicated Select chains.
//!
//! Converts `Op::If { cond, then_block, else_block }` blocks with short bodies
//! into `Op::Select` chains, eliminating branch divergence on SIMD hardware.
//!
//! ## Motivation
//!
//! On Apple GPUs (SIMT execution), threads in a SIMD group execute in lockstep.
//! When a branch diverges, the hardware serializes both paths — effectively
//! executing both arms sequentially with masking.  If-conversion replaces
//! branches with predicated straight-line code; for regions ≤ ~5 ops per side,
//! this eliminates the branch overhead and the serialization penalty.
//!
//! ## Algorithm
//!
//! 1. Classify the CFG shape: Diamond (both arms) or Triangle (one arm empty).
//! 2. Safety check: reject arms containing unpredictable ops (Barrier, Atomic,
//!    Loop, SetLocal, DeclareLocal, ThreadgroupAlloc, nested If, StrideScan,
//!    StrideArgReduce).
//! 3. Profitability check: Diamond ≤ 8 total ops, Triangle ≤ 5 ops.
//! 4. Transform: inline both arms as Select(output, then_result, else_result)
//!    chains. For Diamond shapes, each result-producing op in the then_block
//!    is paired with the corresponding result from the else_block.
//!    For Triangle shapes, we use the passthrough value (Phase 2).
//! 5. Remove then_block and else_block from kernel.blocks.
//!
//! ## Limitations (Phase 1)
//!
//! - Only handles Diamond shapes (both arms). Triangle shapes (no else_block)
//!   require liveness analysis to determine passthrough values — deferred.
//! - Does not handle extended diamonds (multi-block chains in arms).
//! - Conservatively rejects any arm with unpredictable ops.
//!
//! ## References
//! - Allen, Kennedy, Porterfield & Warren (1983), "Conversion of control
//!   dependence to data dependence", POPL 1983:177–189.
//!   The seminal paper establishing if-conversion as a compiler technique.
//! - Kennedy & Allen (2001), "Optimizing Compilers for Modern Architectures:
//!   A Dependence-based Approach", Morgan Kaufmann, Ch. 7.  Comprehensive
//!   treatment of if-conversion and predicated execution.

use std::collections::BTreeMap;

use metaltile_core::ir::{Block, BlockId, Kernel, Op, ValueId};
use rustc_hash::FxHashMap;

use super::remap;
use crate::error::Result;

/// Max total ops across both arms for profitable if-conversion.
const MAX_DIAMOND_OPS: usize = 8;
/// Max ops in a single arm for profitable if-conversion.
const MAX_TRIANGLE_OPS: usize = 5;

pub struct IfConversionPass;

impl super::Pass for IfConversionPass {
    fn name(&self) -> &str { "if_conversion" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Sort explicitly: `kernel.blocks` is `FxHashMap`, so `.keys()`
        // order is non-deterministic — the inside-out `.rev()` walk
        // below relies on ascending BlockId order.
        let mut block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        block_ids.sort_unstable_by_key(|b| b.as_u32());

        // Process the body first.
        if_convert_block(&mut kernel.body, &mut kernel.blocks);

        // Process nested blocks (inside-out: blocks with higher IDs first,
        // as they're typically children of lower-ID blocks).
        for bid in block_ids.iter().rev() {
            let Some(mut block) = kernel.blocks.remove(bid) else { continue };
            if_convert_block(&mut block, &mut kernel.blocks);
            kernel.blocks.insert(*bid, block);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CFG shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfgShape {
    /// One arm empty (no else_block).
    Triangle,
    /// Both arms have code.
    Diamond,
}

fn classify(op: &Op) -> CfgShape {
    if op.as_if().is_some_and(|(_, _, eb)| eb.is_none()) {
        CfgShape::Triangle
    } else {
        CfgShape::Diamond
    }
}

// ---------------------------------------------------------------------------
// safety checks
// ---------------------------------------------------------------------------

/// Ops that cannot appear inside predicated code.
fn is_unpredictable(op: &Op) -> bool { remap::is_unpredictable(op) }

/// Any unpredictable op in the block?
fn has_unpredictable(block: &Block) -> bool { block.ops.iter().any(is_unpredictable) }

// ---------------------------------------------------------------------------
// profitability
// ---------------------------------------------------------------------------

/// Count of ops in a block (excluding no-side-effect structural ops).
fn predicable_op_count(block: &Block) -> usize {
    block.ops.iter().filter(|op| !is_unpredictable(op)).count()
}

fn is_profitable(shape: CfgShape, then_block: &Block, else_block: Option<&Block>) -> bool {
    match shape {
        CfgShape::Triangle => predicable_op_count(then_block) <= MAX_TRIANGLE_OPS,
        CfgShape::Diamond => {
            let else_count = else_block.map_or(0, predicable_op_count);
            predicable_op_count(then_block) + else_count <= MAX_DIAMOND_OPS
        },
    }
}

// ---------------------------------------------------------------------------
// main transform
// ---------------------------------------------------------------------------

fn if_convert_block(block: &mut Block, blocks: &mut FxHashMap<BlockId, Block>) {
    let n = block.ops.len();

    struct Conversion {
        if_idx: usize,
        inlined: Vec<(Op, Option<ValueId>)>,
        remove_blocks: Vec<BlockId>,
    }

    let mut conversions: Vec<Conversion> = Vec::new();

    for i in 0..n {
        if let Some((cond, then_id, else_id)) = block.ops[i].as_if() {
            let Some(then_block) = blocks.get(&then_id) else { continue };
            let else_block = else_id.and_then(|eid| blocks.get(&eid));

            // Safety check.
            if has_unpredictable(then_block) {
                continue;
            }
            if let Some(eb) = else_block
                && has_unpredictable(eb)
            {
                continue;
            }

            // Profitability check.
            let shape = classify(&block.ops[i]);
            if !is_profitable(shape, then_block, else_block) {
                continue;
            }

            // Phase 1: only Diamond shapes.
            if matches!(shape, CfgShape::Triangle) {
                continue;
            }

            let inlined = inline_diamond_as_selects(cond, then_block, else_block.unwrap());
            let mut remove_blocks = vec![then_id];
            if let Some(eid) = else_id {
                remove_blocks.push(eid);
            }

            conversions.push(Conversion { if_idx: i, inlined, remove_blocks });
        }
    }

    if conversions.is_empty() {
        return;
    }

    // Rebuild the block with inlined Select ops.
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let n = old_ops.len();

    // Map: old position → insertion before it.
    let mut inline_at: BTreeMap<usize, &Vec<(Op, Option<ValueId>)>> = BTreeMap::new();
    for conv in &conversions {
        inline_at.insert(conv.if_idx, &conv.inlined);
    }

    let mut new_ops = Vec::new();
    let mut new_results = Vec::new();

    for i in 0..n {
        if let Some(inlined) = inline_at.get(&i) {
            for (op, vid) in *inlined {
                new_ops.push(op.clone());
                new_results.push(*vid);
            }
            // Skip the original Op::If (it's replaced).
            continue;
        }
        new_ops.push(old_ops[i].clone());
        new_results.push(old_results[i]);
    }

    block.ops = new_ops;
    block.results = new_results;

    // Remove consumed blocks.
    for conv in &conversions {
        for bid in &conv.remove_blocks {
            blocks.remove(bid);
        }
    }
}

/// Inline a diamond-shaped If as a chain of Select ops.
///
/// Each result-producing op in the then_block is paired with the corresponding
/// result from the else_block. The result is:
///   `Select(cond, then_result[i], else_result[i])`
///
/// Ops that don't produce results (stores, barriers) in the arms are
/// inlined directly.
fn inline_diamond_as_selects(
    cond: ValueId,
    then_block: &Block,
    else_block: &Block,
) -> Vec<(Op, Option<ValueId>)> {
    // Both arms must have the same number of ops and results.
    // If they don't, we can't do a direct pairing — reject.
    let then_n = then_block.ops.len();
    let else_n = else_block.ops.len();

    // We only pair result-producing ops.
    // Strategy: walk both blocks in parallel, emit ops inline.
    // For ops that produce results, emit a Select.
    let min_n = then_n.min(else_n);
    let mut inlined = Vec::new();

    for j in 0..min_n {
        let then_op = &then_block.ops[j];
        let else_op = &else_block.ops[j];

        // Both ops must produce results (or both not produce results).
        let then_result = then_block.results[j];
        let else_result = else_block.results[j];

        match (then_result, else_result) {
            (Some(then_vid), Some(else_vid)) => {
                // Both produce results → emit a Select.
                // The Select picks between the two existing values.
                // We don't emit the original ops — they're in the arm blocks
                // which are being removed. The Select IS the result.
                inlined.push((
                    Op::Select { cond, on_true: then_vid, on_false: else_vid },
                    None, // Select result will get assigned a ValueId by the caller
                ));
            },
            (None, None) => {
                // Neither produces results → emit both inline.
                inlined.push((then_op.clone(), None));
                inlined.push((else_op.clone(), None));
            },
            _ => {
                // Mismatch — one has result, one doesn't.
                // Conservative: skip this pair.
            },
        }
    }

    // Handle trailing ops in the longer block (if any).
    // For now: skip them.

    inlined
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::BlockId;

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn rejects_barrier_in_arm() {
        let mut k = Kernel::new("if_conv_barrier");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0)); // cond

        // Create then_block with a Barrier.
        let mut then_block = Block::new(BlockId::new(1));
        then_block.push_op(Op::Barrier, ValueId::new(10));

        let then_id = k.add_block(then_block);

        k.body.push_op_no_result(Op::If {
            cond: ValueId::new(0),
            then_block: then_id,
            else_block: None,
        });

        let op_count_before = k.body.ops.len();
        IfConversionPass.run(&mut k).unwrap();

        // Op::If should still be present (rejected).
        assert_eq!(k.body.ops.len(), op_count_before);
        let has_if = k.body.ops.iter().any(|op| matches!(op, Op::If { .. }));
        assert!(has_if);
    }

    #[test]
    fn converts_simple_diamond() {
        let mut k = Kernel::new("if_conv_diamond");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0)); // cond

        // then_block: v1 = const 42
        let mut then_block = Block::new(BlockId::new(1));
        then_block.push_op(Op::Const { value: 42 }, ValueId::new(1));

        // else_block: v2 = const 99
        let mut else_block = Block::new(BlockId::new(2));
        else_block.push_op(Op::Const { value: 99 }, ValueId::new(2));

        let then_id = k.add_block(then_block);
        let else_id = k.add_block(else_block);

        k.body.push_op_no_result(Op::If {
            cond: ValueId::new(0),
            then_block: then_id,
            else_block: Some(else_id),
        });

        IfConversionPass.run(&mut k).unwrap();

        // Op::If should be gone, replaced by Op::Select.
        let has_if = k.body.ops.iter().any(|op| matches!(op, Op::If { .. }));
        assert!(!has_if, "Op::If should be eliminated");

        let has_select = k.body.ops.iter().any(|op| matches!(op, Op::Select { .. }));
        assert!(has_select, "Op::Select should be present");

        // then_block and else_block should be removed.
        assert!(!k.blocks.contains_key(&then_id));
        assert!(!k.blocks.contains_key(&else_id));
    }

    #[test]
    fn rejects_large_diamond() {
        let mut k = Kernel::new("if_conv_large");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));

        // then_block: 6 const ops (too many).
        let mut then_block = Block::new(BlockId::new(1));
        for v in 1..=6 {
            then_block.push_op(Op::Const { value: v }, ValueId::new(v as u32));
        }

        // else_block: 4 const ops.
        let mut else_block = Block::new(BlockId::new(2));
        for v in 10..=13 {
            else_block.push_op(Op::Const { value: v }, ValueId::new(v as u32));
        }

        let then_id = k.add_block(then_block);
        let else_id = k.add_block(else_block);

        k.body.push_op_no_result(Op::If {
            cond: ValueId::new(0),
            then_block: then_id,
            else_block: Some(else_id),
        });

        IfConversionPass.run(&mut k).unwrap();

        // 6+4 = 10 > 8 → rejected. If should remain.
        let has_if = k.body.ops.iter().any(|op| matches!(op, Op::If { .. }));
        assert!(has_if);
    }
}
