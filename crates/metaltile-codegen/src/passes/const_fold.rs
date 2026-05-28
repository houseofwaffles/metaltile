//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile_core::ir::{BinOpKind, Block, BlockId, Kernel, Op, UnaryOpKind, ValueId};

use crate::error::Result;

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
        // Fold `kernel.body` (the entry block) AND every nested block.
        //
        // The previous shape walked `kernel.blocks` only, which skipped
        // the entry block entirely — so a Div / Mul of two Consts that
        // landed in `kernel.body` (e.g. `dims_per_word = 32 / bits`)
        // never got folded.  That hid the trip count from `unroll`,
        // which uses `find_const_in_block` to discover loop bounds: it
        // looks for a direct `Op::Const`, not a still-rolled
        // `BinOp(Div, Const, Const)`.  The result was that small
        // top-level loops in `kernel.body` — for instance the
        // `dims_per_word == 4` loop in `aura_dequant_rotated_int8` —
        // stayed rolled when the rest of the pipeline expected them
        // unrolled, and the loop body fell through downstream passes,
        // leaving an empty `for (...)` header in the emitted MSL.
        // Folding the entry block first closes that gap.
        fold_block(&mut kernel.body);
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

        // DCE limited to nested blocks. The entry block (kernel.body)
        // is intentionally left rolled: tests that hand-build kernels
        // with orphan ops (no Stores, no cross-block uses) rely on the
        // ops surviving to MSL. Folding constants in the entry block
        // is enough to unblock unroll; the harmless dead ops downstream
        // are cleaned up by the per-pass DCE the runtime pipeline
        // already runs after fusion + vectorize.
        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                dce_block(block, &cross_block_refs);
            }
        }
        super::dead_value_elim::eliminate_dead_values(kernel)?;
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
        match &block.ops[i] {
            Op::BinOp { op, lhs, rhs } => {
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
            },
            // Integer-typed `UnaryOp(Const)` folds to a single Const.
            // Float-typed unaries (Exp, Log, Sqrt, Sin, …) intentionally
            // stay rolled — `Op::Const` stores `i64`, so collapsing
            // `exp(Const(2))` to a Const would lose the float-domain
            // semantics the downstream ops expect.  See `eval_unary_op`
            // for the integer-safe set.
            Op::UnaryOp { op: kind, value } => {
                if let Some(a) = find_const(block, *value)
                    && let Some(result) = eval_unary_op(*kind, a)
                {
                    const_overwrites.push((i, result));
                }
            },
            _ => {},
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
        // Integer comparisons fold to 0/1 (matching the convention used
        // by `And`/`Or` above).  Folding these unblocks downstream
        // `Select { cond: Const(0|1), … }` simplification in
        // `algebraic_simplify`, which is the path that retires dead
        // constexpr-param reads in the non-causal variants of
        // `aura_flash_p1_*` (where the macro substitutes `$causal == 0`
        // and the load of `q_position` becomes orphan).
        BinOpKind::CmpLt => Some((a < b) as i64),
        BinOpKind::CmpGt => Some((a > b) as i64),
        BinOpKind::CmpLe => Some((a <= b) as i64),
        BinOpKind::CmpGe => Some((a >= b) as i64),
        BinOpKind::CmpEq => Some((a == b) as i64),
        BinOpKind::CmpNe => Some((a != b) as i64),
        BinOpKind::Pow => None, // floating-point; not folded
        BinOpKind::Shl => Some(a << b),
        BinOpKind::Shr => Some(a >> b),
        BinOpKind::BitAnd => Some(a & b),
        BinOpKind::BitOr => Some(a | b),
        BinOpKind::BitXor => Some(a ^ b),
        BinOpKind::ATan2 | BinOpKind::Rem | BinOpKind::Mod => None,
    }
}

/// Fold `Op::UnaryOp(Const)` for the *integer-safe* unary kinds.
///
/// `Op::Const` stores an `i64` — float-domain unaries like `Exp`,
/// `Log`, `Sqrt`, `Sin`, `Cos`, `Tanh`, `Erf`, `Recip`, etc. don't
/// round-trip through an integer constant pool without changing
/// semantics (`exp(Const(2))` ≠ `Const(7)`), so they intentionally
/// stay rolled and the float computation happens at runtime in MSL.
///
/// The kinds folded here all produce an integer answer from an
/// integer input:
/// - `Neg` — two's-complement negation, wrapping on `i64::MIN`.
/// - `Abs` — absolute value, wrapping on `i64::MIN` (matches
///   `i64::wrapping_abs` semantics).
/// - `Sign` — `-1` / `0` / `+1` based on the input's sign.
/// - `Trunc`, `Floor`, `Ceil`, `Round` — no-ops on integers; preserve
///   the value, drop the float-domain rounding op so downstream code
///   sees a plain `Op::Const`.
fn eval_unary_op(op: UnaryOpKind, a: i64) -> Option<i64> {
    match op {
        UnaryOpKind::Neg => Some(a.wrapping_neg()),
        UnaryOpKind::Abs => Some(a.wrapping_abs()),
        UnaryOpKind::Sign => Some(a.signum()),
        UnaryOpKind::Trunc | UnaryOpKind::Floor | UnaryOpKind::Ceil | UnaryOpKind::Round => Some(a),
        // Float-domain unaries (Exp, Log, Sqrt, Rsqrt, Recip, Sin,
        // Cos, Tan, Erf, ErfInv, Exp2, Log2, Sinh, Cosh, Asin, Acos,
        // Atan, Asinh, Acosh, Atanh, Tanh, Log1p, Expm1, Square) are
        // intentionally not folded — see fn-level docstring.
        _ => None,
    }
}

fn find_const(block: &Block, vid: ValueId) -> Option<i64> {
    block.ops.iter().enumerate().find_map(|(i, op)| {
        if block.results.get(i) == Some(&Some(vid)) { op.as_const() } else { None }
    })
}

fn replace_value_in_op(op: &mut Op, old: ValueId, new: ValueId) {
    op.for_each_value_id_mut(&mut |v| {
        if *v == old {
            *v = new;
        }
    });
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
    for &v in op.value_refs().iter() {
        used.insert(*v);
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
        // Append a Store that consumes `used_vid` so the per-pass DCE
        // postcondition (#209/1) doesn't sweep the ops we want to
        // inspect.  Stores have side effects and are never eliminated;
        // a BinOp ref-op was used pre-#209/1 but is itself DCE-eligible
        // (its result has no consumer), so cascade-removal would strip
        // the whole chain.  Using a Store anchors `used_vid` against
        // the kernel's `out` param.
        ops.push(metaltile_core::ir::Op::Store {
            dst: "out".into(),
            indices: vec![metaltile_core::ir::IndexExpr::Value(used_vid)],
            value: used_vid,
            mask: None,
        });
        results.push(None);

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

    #[test]
    fn folds_unary_neg_of_const() {
        // Neg(Const(7)) → Const(-7).
        let ops = vec![Op::Const { value: 7 }, Op::UnaryOp {
            op: UnaryOpKind::Neg,
            value: ValueId::new(100),
        }];
        let results = vec![Some(ValueId::new(100)), Some(ValueId::new(101))];
        let block = fold_in_block(ops, results, ValueId::new(101));
        // Stand-alone Neg op must be gone — folded into a Const.
        assert!(
            !block.ops.iter().any(|op| matches!(op, Op::UnaryOp { .. })),
            "Neg(Const) should fold away"
        );
        assert!(
            block.ops.iter().any(|op| matches!(op, Op::Const { value: -7 })),
            "Neg(Const(7)) should produce Const(-7): {:?}",
            block.ops
        );
    }

    #[test]
    fn folds_unary_abs_of_const() {
        // Abs(Const(-5)) → Const(5).
        let ops = vec![Op::Const { value: -5 }, Op::UnaryOp {
            op: UnaryOpKind::Abs,
            value: ValueId::new(100),
        }];
        let results = vec![Some(ValueId::new(100)), Some(ValueId::new(101))];
        let block = fold_in_block(ops, results, ValueId::new(101));
        assert!(
            block.ops.iter().any(|op| matches!(op, Op::Const { value: 5 })),
            "Abs(Const(-5)) should produce Const(5): {:?}",
            block.ops
        );
    }

    #[test]
    fn folds_unary_sign_of_const() {
        // Sign(Const(-42)) → Const(-1).  Triple coverage: -42, 0, +42.
        for (input, expected) in [(-42i64, -1i64), (0, 0), (42, 1)] {
            let ops = vec![Op::Const { value: input }, Op::UnaryOp {
                op: UnaryOpKind::Sign,
                value: ValueId::new(100),
            }];
            let results = vec![Some(ValueId::new(100)), Some(ValueId::new(101))];
            let block = fold_in_block(ops, results, ValueId::new(101));
            assert!(
                block.ops.iter().any(|op| matches!(op, Op::Const { value } if *value == expected)),
                "Sign(Const({input})) should produce Const({expected}): {:?}",
                block.ops
            );
        }
    }

    #[test]
    fn float_unary_of_const_not_folded() {
        // Exp(Const(2)) must NOT fold — Op::Const stores i64 and the
        // float-domain semantics of `exp(2.0)` ≠ `Const(7)`.  Stay
        // rolled; the float compute happens at runtime in MSL.
        let ops = vec![Op::Const { value: 2 }, Op::UnaryOp {
            op: UnaryOpKind::Exp,
            value: ValueId::new(100),
        }];
        let results = vec![Some(ValueId::new(100)), Some(ValueId::new(101))];
        let block = fold_in_block(ops, results, ValueId::new(101));
        assert!(
            block.ops.iter().any(|op| matches!(op, Op::UnaryOp { op: UnaryOpKind::Exp, .. })),
            "Exp(Const) must stay unfolded: {:?}",
            block.ops
        );
    }

    #[test]
    fn folds_integer_noop_unary_of_const() {
        // Trunc/Floor/Ceil/Round on integer Const are no-ops on the
        // value — drop the op and keep the Const.  This collapses the
        // `Op::UnaryOp { Trunc | Floor | Ceil | Round, Const(a) }`
        // pattern that's common in code-paths that mix integer index
        // math with float-domain ops (e.g. `(idx as f32).floor()`
        // where the cast-Const-floor sequence reduces away).
        for kind in [UnaryOpKind::Trunc, UnaryOpKind::Floor, UnaryOpKind::Ceil, UnaryOpKind::Round]
        {
            let ops =
                vec![Op::Const { value: 9 }, Op::UnaryOp { op: kind, value: ValueId::new(100) }];
            let results = vec![Some(ValueId::new(100)), Some(ValueId::new(101))];
            let block = fold_in_block(ops, results, ValueId::new(101));
            assert!(
                !block.ops.iter().any(|op| matches!(op, Op::UnaryOp { .. })),
                "{kind:?}(Const(9)) should fold away: {:?}",
                block.ops
            );
            assert!(
                block.ops.iter().any(|op| matches!(op, Op::Const { value: 9 })),
                "{kind:?}(Const(9)) should produce Const(9): {:?}",
                block.ops
            );
        }
    }
}
