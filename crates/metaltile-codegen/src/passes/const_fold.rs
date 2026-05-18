//! Constant Folding & Dead Code Elimination — evaluate compile-time expressions.
//!
//! Folds literal-literal arithmetic at compile time and propagates identity
//! operations (add/sub 0, mul/div 1).  A conservative dead-code elimination
//! (DCE) sub-pass removes ops whose result is never referenced downstream,
//! including across block boundaries.  Both phases recurse into loop body blocks.
//!
//! This is the first optimization pass in the pipeline.  Running it early
//! reduces IR size for all subsequent passes.
//!
//! ## Patterns folded
//! - `x + 0`, `0 + x`, `x - 0` → x
//! - `x * 1`, `1 * x` → x
//! - `x * 0`, `0 * x` → Const(0)
//! - `Const(a) OP Const(b)` → Const(result)
//!
//! ## Algorithm
//!
//! 1. Block-local fold: scan each op; if all operands are Const, evaluate.
//! 2. Identity elimination: replace identity-binop with its non-trivial operand.
//! 3. Cross-block liveness: collect all ValueId uses across the entire kernel;
//!    any ValueId defined in a block but never used is dead.
//! 4. DCE: rebuild each block omitting dead ops, recurse into loop bodies.
//!
//! ## References
//! - Cocke & Schwartz (1970), "Programming Languages and their Compilers",
//!   Courant Institute.  Earliest description of constant folding as part of
//!   value numbering.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §8.4 (constant folding), §9.1.2 (dead-code elimination).

use std::collections::BTreeSet;

use metaltile_core::{
    error::Result,
    ir::{BinOpKind, Block, BlockId, IndexExpr, Kernel, Op, ValueId},
};

pub struct ConstFoldPass;

impl ConstFoldPass {
    pub fn new() -> Self { ConstFoldPass }
}
impl Default for ConstFoldPass {
    fn default() -> Self { ConstFoldPass }
}

impl super::Pass for ConstFoldPass {
    fn name(&self) -> &str { "const_fold" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                fold_block(block);
            }
        }

        // Collect all ValueIds referenced by any block in the kernel (cross-block uses).
        // These must be treated as "live" in each individual block's DCE pass to prevent
        // eliminating values that are consumed by sibling or child blocks.
        let mut cross_block_refs: BTreeSet<ValueId> = BTreeSet::new();
        for block in kernel.blocks.values() {
            for op in &block.ops {
                collect_uses(op, &mut cross_block_refs);
            }
        }
        for op in &kernel.body.ops {
            collect_uses(op, &mut cross_block_refs);
        }

        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                dce_block(block, &cross_block_refs);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

fn fold_block(block: &mut Block) {
    let mut replacements: Vec<(usize, ValueId)> = Vec::new();
    let mut const_overwrites: Vec<(usize, i64)> = Vec::new();

    for i in 0..block.ops.len() {
        if let Op::BinOp { op, lhs, rhs } = &block.ops[i] {
            let lv = find_const(block, *lhs);
            let rv = find_const(block, *rhs);

            // Literal-literal folding.
            if let (Some(a), Some(b)) = (lv, rv)
                && let Some(result) = eval_binop(*op, a, b)
            {
                const_overwrites.push((i, result));
                continue;
            }

            // Identity / absorbing-element folding.
            match op {
                BinOpKind::Add | BinOpKind::Sub => {
                    if lv == Some(0) && matches!(op, BinOpKind::Add) {
                        replacements.push((i, *rhs));
                    } else if rv == Some(0) {
                        replacements.push((i, *lhs));
                    }
                },
                BinOpKind::Mul =>
                    if lv == Some(0) || rv == Some(0) {
                        const_overwrites.push((i, 0));
                    } else if lv == Some(1) {
                        replacements.push((i, *rhs));
                    } else if rv == Some(1) {
                        replacements.push((i, *lhs));
                    },
                _ => {},
            }
        }
    }

    // Rewrite ops that folded to a constant.
    for (idx, new_val) in const_overwrites {
        block.ops[idx] = Op::Const { value: new_val };
    }

    // Redirect uses of identity-folded ops to their passthrough values.
    for (op_idx, replacement) in replacements {
        let old_vid = match block.results.get(op_idx).and_then(|x| *x) {
            Some(v) => v,
            None => continue,
        };
        for op in block.ops.iter_mut() {
            replace_value_in_op(op, old_vid, replacement);
        }
    }
}

fn eval_binop(op: BinOpKind, a: i64, b: i64) -> Option<i64> {
    match op {
        BinOpKind::Add => Some(a.wrapping_add(b)),
        BinOpKind::Sub => Some(a.wrapping_sub(b)),
        BinOpKind::Mul => Some(a.wrapping_mul(b)),
        BinOpKind::Div =>
            if b != 0 {
                Some(a / b)
            } else {
                None
            },
        BinOpKind::Max => Some(a.max(b)),
        BinOpKind::Min => Some(a.min(b)),
        BinOpKind::And => Some((a != 0 && b != 0) as i64),
        BinOpKind::Or => Some((a != 0 || b != 0) as i64),
        BinOpKind::Xor => Some(a ^ b),
        BinOpKind::CmpLt
        | BinOpKind::CmpGt
        | BinOpKind::CmpLe
        | BinOpKind::CmpGe
        | BinOpKind::CmpEq
        | BinOpKind::CmpNe => None,
        BinOpKind::Pow => None, // floating-point; not folded
        BinOpKind::Shl => Some(a << b),
        BinOpKind::Shr => Some(a >> b),
        BinOpKind::BitAnd => Some(a & b),
        BinOpKind::BitOr => Some(a | b),
        BinOpKind::BitXor => Some(a ^ b),
        BinOpKind::ATan2 | BinOpKind::Rem | BinOpKind::Mod => None,
    }
}

fn find_const(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::Const { value } = op
        {
            return Some(*value);
        }
    }
    None
}

fn replace_value_in_op(op: &mut Op, old: ValueId, new: ValueId) {
    let s = |v: &mut ValueId| {
        if *v == old {
            *v = new;
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. } => s(value),
        Op::Activation { value, .. } => s(value),
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Broadcast { value, .. } => s(value),
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, indices, .. } => {
            s(value);
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            }
        },
        Op::Cast { value, .. } => s(value),
        Op::Reduce { value, .. } => s(value),
        Op::Transpose { value } => s(value),
        Op::Slice { value, .. } => s(value),
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::Load { indices, .. } =>
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs.iter_mut() {
                s(v);
            },
        Op::FlashAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::RmsNorm { x, scale, .. } => {
            s(x);
            s(scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            s(x);
            s(gate_proj);
            s(up_proj);
            s(down_proj);
        },
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. } => {},
        Op::FusedElementwise { ops } =>
            for op in ops.iter_mut() {
                replace_value_in_op(op, old, new);
            },
        Op::VectorLoad { byte_offset, .. } => s(byte_offset),
        Op::VectorStore { byte_offset, value, .. } => {
            s(byte_offset);
            s(value);
        },
        Op::VectorExtract { vec, .. } => s(vec),
        Op::StrideReduce { offset, stride, end, .. } => {
            s(offset);
            s(stride);
            s(end);
        },
        Op::If { cond, .. } => s(cond),
        Op::ExpandDims { value, .. } => s(value),
        Op::Reshape { value, .. } => s(value),
        Op::Cat { values, .. } =>
            for v in values.iter_mut() {
                s(v);
            },
        Op::Gather { indices, .. } => s(indices),
        Op::Scatter { indices, value, .. } => {
            s(indices);
            s(value);
        },
        Op::Atomic { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::Scan { value, .. } => s(value),
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } => s(value),
        Op::SimdShuffleXor { value, mask } => {
            s(value);
            s(mask);
        },
        Op::SimdBroadcast { value, lane } => {
            s(value);
            s(lane);
        },
        Op::ThreadgroupLoad { index, .. } => s(index),
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier | Op::SimdLaneId | Op::SimdGroupId => {},
        Op::SimdgroupAlloc { .. } | Op::SimdgroupMatMul { .. } => {},
        Op::SimdgroupElemLoad { value, .. } => s(value),
        Op::SimdgroupElemStore { value, data, .. } => {
            s(value);
            s(data);
        },
        Op::SimdScan { value, .. } => s(value),
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => s(value),
        Op::ArgReduce { value, .. } => s(value),
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
    }
}

// ---------------------------------------------------------------------------
// Dead Code Elimination
// ---------------------------------------------------------------------------

fn dce_block(block: &mut Block, cross_block_refs: &BTreeSet<ValueId>) {
    let mut used: BTreeSet<ValueId> = BTreeSet::new();
    for op in &block.ops {
        collect_uses(op, &mut used);
    }
    // Values consumed by other blocks must be kept even if unused within this block.
    used.extend(cross_block_refs.iter().copied());

    let n = block.ops.len();
    let mut keep = vec![false; n];
    for (i, keep_entry) in keep.iter_mut().enumerate().take(n) {
        *keep_entry = match block.results.get(i) {
            Some(&Some(vid)) => used.contains(&vid),
            Some(&None) | None => true, // no-result ops always kept
        };
    }

    let mut new_ops = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();
    for (i, op) in block.ops.iter().enumerate() {
        if keep[i] {
            new_ops.push(op.clone());
            new_results.push(block.results[i]);
        }
    }
    block.ops = new_ops;
    block.results = new_results;
}

fn collect_uses(op: &Op, used: &mut BTreeSet<ValueId>) {
    let mut add = |v: ValueId| {
        used.insert(v);
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            add(*lhs);
            add(*rhs);
        },
        Op::UnaryOp { value, .. } => add(*value),
        Op::Activation { value, .. } => add(*value),
        Op::Select { cond, on_true, on_false } => {
            add(*cond);
            add(*on_true);
            add(*on_false);
        },
        Op::Broadcast { value, .. } => add(*value),
        Op::Dot { a, b } => {
            add(*a);
            add(*b);
        },
        Op::Store { value, indices, .. } => {
            add(*value);
            for idx in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    add(*v);
                }
            }
        },
        Op::Cast { value, .. } => add(*value),
        Op::Reduce { value, .. } => add(*value),
        Op::Transpose { value } => add(*value),
        Op::Slice { value, .. } => add(*value),
        Op::Loop { start, end, step, .. } => {
            add(*start);
            add(*end);
            add(*step);
        },
        Op::Load { indices, .. } =>
            for idx in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    add(*v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                add(*v);
            },
        Op::FlashAttention { q, k, v, .. } => {
            add(*q);
            add(*k);
            add(*v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            add(*q);
            add(*k);
            add(*v);
        },
        Op::RmsNorm { x, scale, .. } => {
            add(*x);
            add(*scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            add(*x);
            add(*gate_proj);
            add(*up_proj);
            add(*down_proj);
        },
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. } => {},
        Op::FusedElementwise { ops } =>
            for op in ops {
                collect_uses(op, used);
            },
        Op::VectorLoad { byte_offset, .. } => add(*byte_offset),
        Op::VectorStore { byte_offset, value, .. } => {
            add(*byte_offset);
            add(*value);
        },
        Op::VectorExtract { vec, .. } => add(*vec),
        Op::StrideReduce { offset, stride, end, .. } => {
            add(*offset);
            add(*stride);
            add(*end);
        },
        Op::If { cond, .. } => add(*cond),
        Op::ExpandDims { value, .. } => add(*value),
        Op::Reshape { value, .. } => add(*value),
        Op::Cat { values, .. } =>
            for v in values {
                add(*v);
            },
        Op::Gather { indices, .. } => add(*indices),
        Op::Scatter { indices, value, .. } => {
            add(*indices);
            add(*value);
        },
        Op::Atomic { index, value, .. } => {
            add(*index);
            add(*value);
        },
        Op::Scan { value, .. } => add(*value),
        Op::StrideStore { offset, end, scalar, .. } => {
            add(*offset);
            add(*end);
            add(*scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } => add(*value),
        Op::SimdShuffleXor { value, mask } => {
            add(*value);
            add(*mask);
        },
        Op::SimdBroadcast { value, lane } => {
            add(*value);
            add(*lane);
        },
        Op::ThreadgroupLoad { index, .. } => add(*index),
        Op::ThreadgroupStore { index, value, .. } => {
            add(*index);
            add(*value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier | Op::SimdLaneId | Op::SimdGroupId => {},
        Op::SimdgroupAlloc { .. } | Op::SimdgroupMatMul { .. } => {},
        Op::SimdgroupElemLoad { value, .. } => add(*value),
        Op::SimdgroupElemStore { value, data, .. } => {
            add(*value);
            add(*data);
        },
        Op::SimdScan { value, .. } => add(*value),
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => add(*value),
        Op::ArgReduce { value, .. } => add(*value),
        Op::StrideScan { offset, end, .. } => {
            add(*offset);
            add(*end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            add(*offset);
            add(*end);
        },
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{BinOpKind, VarId};

    use super::*;
    use crate::passes::Pass;

    // Helper: run const_fold on ops in a loop body block and return the block.
    // Use VIDs >= 100 in tests to avoid collision with the body's VIDs 0,1,2.
    fn fold_in_block(
        mut ops: Vec<Op>,
        mut results: Vec<Option<ValueId>>,
        used_vid: ValueId,
    ) -> Block {
        // Append an op that references `used_vid` to prevent DCE from removing it.
        let ref_vid = ValueId::new(
            results.iter().filter_map(|r| r.map(|v| v.as_u32())).max().unwrap_or(99) + 1,
        );
        ops.push(Op::BinOp { op: BinOpKind::Add, lhs: used_vid, rhs: used_vid });
        results.push(Some(ref_vid));

        let mut k = Kernel::new("test");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));
        let mut body = Block::new(BlockId::new(1));
        body.ops = ops;
        body.results = results;
        let body_id = k.add_block(body);
        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });
        ConstFoldPass.run(&mut k).unwrap();
        k.blocks.remove(&body_id).unwrap()
    }

    #[test]
    fn folds_literal_add() {
        let ops = vec![Op::Const { value: 3 }, Op::Const { value: 4 }, Op::BinOp {
            op: BinOpKind::Add,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_const_7 = block.ops.iter().any(|op| matches!(op, Op::Const { value: 7 }));
        assert!(has_const_7, "3+4 should fold to Const(7)");
    }

    #[test]
    fn dce_removes_unused_const() {
        let ops = vec![Op::Const { value: 42 }, Op::Const { value: 99 }, Op::UnaryOp {
            op: metaltile_core::ir::UnaryOpKind::Neg,
            value: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_v42 = block.ops.iter().any(|op| matches!(op, Op::Const { value: 42 }));
        assert!(!has_v42, "unused Const(42) should be DCE'd");
    }

    #[test]
    fn folds_in_loop_body() {
        let ops = vec![Op::Const { value: 0 }, Op::Const { value: 1 }, Op::BinOp {
            op: BinOpKind::Add,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_folded = block.ops.iter().any(|op| matches!(op, Op::Const { value: 1 }));
        assert!(has_folded, "0+1 in loop body should fold to Const(1)");
    }

    #[test]
    fn preserves_nonfoldable_ops() {
        let ops = vec![Op::ProgramId { axis: 0 }, Op::ProgramId { axis: 1 }, Op::BinOp {
            op: BinOpKind::Add,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_add = block.ops.iter().any(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        assert!(has_add, "non-constant add should not be folded");
    }

    #[test]
    fn folds_x_plus_0_to_x() {
        // x + 0 → x: the Add BinOp should be replaced by redirecting uses.
        let ops = vec![Op::ProgramId { axis: 0 }, Op::Const { value: 0 }, Op::BinOp {
            op: BinOpKind::Add,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_add = block.ops.iter().any(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }));
        assert!(!has_add, "x+0 should be folded away");
    }

    #[test]
    fn folds_x_mul_1_to_x() {
        // 7 * 1 → literal fold to Const(7).
        let ops = vec![Op::Const { value: 7 }, Op::Const { value: 1 }, Op::BinOp {
            op: BinOpKind::Mul,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_const_7 = block.ops.iter().any(|op| matches!(op, Op::Const { value: 7 }));
        assert!(has_const_7, "7*1 should fold to Const(7)");
    }

    #[test]
    fn folds_x_mul_0_to_zero() {
        // Any value * 0 → Const(0).
        let ops = vec![Op::ProgramId { axis: 0 }, Op::Const { value: 0 }, Op::BinOp {
            op: BinOpKind::Mul,
            lhs: ValueId::new(100),
            rhs: ValueId::new(101),
        }];
        let results =
            vec![Some(ValueId::new(100)), Some(ValueId::new(101)), Some(ValueId::new(102))];
        let block = fold_in_block(ops, results, ValueId::new(102));
        let has_const_zero = block.ops.iter().any(|op| matches!(op, Op::Const { value: 0 }));
        assert!(has_const_zero, "x*0 should become Const(0)");
    }
}
