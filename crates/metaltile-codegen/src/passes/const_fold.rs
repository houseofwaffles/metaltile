//! Constant Folding & Dead Code Elimination pass.
//!
//! Folds:
//!   - `x + 0`, `0 + x`, `x - 0`         → x
//!   - `x * 1`, `1 * x`                   → x
//!   - `x * 0`, `0 * x`                   → Const(0)
//!   - `Const(a) OP Const(b)`             → Const(result)  (literal-literal arithmetic)
//!
//! DCE removes ops whose result is never used downstream.
//! Both passes recurse into loop body blocks.

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
            if let (Some(a), Some(b)) = (lv, rv) {
                if let Some(result) = eval_binop(*op, a, b) {
                    const_overwrites.push((i, result));
                    continue;
                }
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
    }
}

fn find_const(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid)) {
            if let Op::Const { value } = op {
                return Some(*value);
            }
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
        Op::ThreadgroupLoad { index, .. } => s(index),
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
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
    for i in 0..n {
        keep[i] = match block.results.get(i) {
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
        Op::ThreadgroupLoad { index, .. } => add(*index),
        Op::ThreadgroupStore { index, value, .. } => {
            add(*index);
            add(*value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
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
