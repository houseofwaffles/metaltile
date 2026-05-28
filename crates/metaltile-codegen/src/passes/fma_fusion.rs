//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FMA Fusion — rewrite `Op::Add(Op::Mul(a, b), c)` into a single `Op::Fma { a, b, c }`.
//!
//! Pre-fix, this fusion lived as an *emit-time* peephole in
//! `msl/emit_block.rs` that turned the textual `auto v_add = v_mul + c;`
//! into `auto v_add = fma(a, b, c);` while leaving the upstream
//! `Op::Mul` in the IR.  The standalone Mul then emitted
//! `auto v_mul = a * b;` as a dead variable in MSL, producing one
//! `-Wunused-variable` warning per fusion site.  #207 worked around
//! that by pre-computing a "skip set" of absorbed-Mul VIDs in
//! `compute_fma_absorbed_mul_skips` and threading a `skip_emit`
//! parameter through every `emit_block` call — fragile mirroring of
//! the emit's `fma_ok` predicate.
//!
//! Lifting the fusion into the IR removes both the emit-time peephole
//! and the skip-set workaround:
//! - the Mul orphan is now a real producer-with-no-consumer in the
//!   post-fusion IR, picked up by `DeadValueElimPass` like any other
//!   dead value;
//! - the emit-time peephole collapses to a single
//!   `Op::Fma → fma(a, b, c)` arm in `emit_block.rs`;
//! - the `skip_emit` parameter is deleted from `emit_block`'s
//!   signature.
//!
//! ## Algorithm
//!
//! For each block:
//! 1. Build a use-count map: how many ops in the block reference each
//!    `ValueId` as an operand.  The fusion is safe only when the
//!    candidate Mul has *exactly one* consumer — the absorbing Add —
//!    because rewriting drops the standalone Mul.  (Cross-block uses
//!    are handled by the kernel-wide DCE; we deliberately work
//!    block-local here so we never need a kernel-wide use-count map.)
//! 2. For each `Op::BinOp { Add, lhs, rhs }`, check whether one
//!    operand is a uniquely-consumed `Op::Mul(a, b)` and all three
//!    operand types are floats (Metal's `fma` only has float
//!    overloads — `fma(int, int, int)` is a compile error).  If yes,
//!    rewrite the BinOp to `Op::Fma { a, b, c }`.  The Mul itself is
//!    not removed here — DCE sweeps it on the next pass since nothing
//!    references its result anymore.
//!
//! ## Why only `Add`, not `Sub`?
//!
//! The pre-fix emit-time peephole also handled `a*b - c` →
//! `fma(a, b, -c)`.  Lifting that into the IR would need either:
//! (a) a separate `Op::Fma { … neg_c: bool }` flag, or
//! (b) injecting an `Op::UnaryOp { Neg, c }` before the `Op::Fma`.
//!
//! Both add IR surface area for a relatively rare pattern.  Skipping
//! Sub here means the kernel emits `auto v = (a * b) - c;`; the MSL
//! compiler is free to lower that to an FMA at the AIR level on its
//! own, so the runtime performance impact is nil and the IR stays
//! lean.  If/when a benchmark surfaces a real regression we can
//! revisit with option (b).

use metaltile_core::ir::{BinOpKind, Block, BlockId, Kernel, Op, ValueId};
use rustc_hash::FxHashMap;

use crate::{error::Result, passes::type_check::infer_types};

pub struct FmaFusionPass;

impl super::Pass for FmaFusionPass {
    fn name(&self) -> &str { "fma_fusion" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // FMA is float-only; type inference tells us per-VID dtype.
        let Ok(type_env) = infer_types(kernel) else {
            // Type-check failures are caught elsewhere in the pipeline
            // (TypeCheckPass).  Bail out cleanly if we hit one here —
            // the FMA fusion is a perf optimization, not a correctness
            // requirement.
            return Ok(());
        };
        let is_float = |id: ValueId| -> bool {
            type_env
                .get(&id)
                .map(|tv| {
                    use metaltile_core::dtype::DType;
                    matches!(tv.dtype, DType::F32 | DType::F16 | DType::BF16)
                })
                .unwrap_or(false)
        };

        // Work block by block — the fusion never reaches across block
        // boundaries (the Mul and the Add live in the same block; a
        // Mul whose result is read from a nested block has
        // use_count > 1 here by construction).
        fuse_block(&mut kernel.body, &is_float);
        let ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in ids {
            if let Some(block) = kernel.blocks.get_mut(&bid) {
                fuse_block(block, &is_float);
            }
        }
        // Per-pass DCE postcondition (#209/1): the rewriter leaves the
        // standalone Op::Mul behind (its only consumer, the absorbed
        // Add, was just replaced by Op::Fma).  Sweep it.
        super::dead_value_elim::eliminate_dead_values(kernel)?;
        Ok(())
    }
}

fn fuse_block(block: &mut Block, is_float: &dyn Fn(ValueId) -> bool) {
    // Block-local use count: how many ops in this block reference each
    // result ValueId as an operand?  A Mul we want to absorb must have
    // exactly one consumer (the Add we're rewriting) — otherwise
    // dropping the standalone Mul would break the other consumer.
    let mut use_count: FxHashMap<ValueId, u32> =
        FxHashMap::with_capacity_and_hasher(block.ops.len(), Default::default());
    for op in &block.ops {
        for v in op.value_refs() {
            *use_count.entry(*v).or_insert(0) += 1;
        }
    }

    // Index of the `Op::Mul` that produces a given ValueId, if any.
    let mut mul_index: FxHashMap<ValueId, usize> =
        FxHashMap::with_capacity_and_hasher(block.ops.len(), Default::default());
    for (i, op) in block.ops.iter().enumerate() {
        if let Op::BinOp { op: BinOpKind::Mul, .. } = op
            && let Some(Some(vid)) = block.results.get(i)
        {
            mul_index.insert(*vid, i);
        }
    }

    // Single forward pass: for each `Op::BinOp { Add, … }`, see whether
    // one operand can be absorbed.
    for i in 0..block.ops.len() {
        let Op::BinOp { op: BinOpKind::Add, lhs, rhs } = block.ops[i] else { continue };

        // FMA needs the result AND both Add operands to be float.
        let result_vid = block.results.get(i).and_then(|v| *v);
        let result_is_float = result_vid.map(&is_float).unwrap_or(false);
        if !result_is_float || !is_float(lhs) || !is_float(rhs) {
            continue;
        }

        // Identify the side (if any) the peephole would absorb — must
        // be a Mul defined in this block with exactly one consumer
        // (the current Add).
        let candidate = |vid: ValueId| -> Option<(ValueId, ValueId)> {
            let mul_pos = *mul_index.get(&vid)?;
            if use_count.get(&vid).copied().unwrap_or(0) != 1 {
                return None;
            }
            // Mul operands must also be float — the type-check guard
            // mirrors the result check above.
            let Op::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr } = block.ops[mul_pos] else {
                return None;
            };
            if !is_float(ml) || !is_float(mr) {
                return None;
            }
            Some((ml, mr))
        };

        let (absorbed_vid, ml, mr, other) = if let Some((ml, mr)) = candidate(lhs) {
            (lhs, ml, mr, rhs)
        } else if let Some((ml, mr)) = candidate(rhs) {
            (rhs, ml, mr, lhs)
        } else {
            continue;
        };

        // Rewrite the Add op to an Fma; the standalone Mul becomes
        // dead (its only consumer was this Add, and we just dropped
        // the operand reference).  DCE picks it up on the next pass.
        block.ops[i] = Op::Fma { a: ml, b: mr, c: other };

        // Maintain block-local invariants: the Mul's result is no
        // longer used.  Mark its use count zero so a subsequent FMA
        // candidate in the same block doesn't try to absorb it again.
        if let Some(slot) = use_count.get_mut(&absorbed_vid) {
            *slot = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        ir::{IndexExpr, Param, ParamKind},
        shape::Shape,
    };

    use super::*;
    use crate::passes::Pass;

    /// Construct a kernel that lowers `c[idx] = a[idx] * b[idx] + c[idx]`
    /// at f32.  The body has Op::Mul → Op::Add; FmaFusionPass should
    /// rewrite the Add into Op::Fma and leave the Mul as a dead op
    /// (DCE-eligible — verified separately).
    fn fma_eligible_kernel() -> Kernel {
        let mut k = Kernel::new("fma_smoke");
        for (name, is_output) in [("a", false), ("b", false), ("c", true)] {
            k.params.push(Param {
                name: name.into(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output,
                kind: ParamKind::Tensor,
            });
        }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::Load {
                src: "c".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(3),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(4),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(4), rhs: ValueId::new(3) },
            ValueId::new(5),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(5),
            mask: None,
        });
        k
    }

    #[test]
    fn fuses_mul_add_to_fma() {
        let mut k = fma_eligible_kernel();
        FmaFusionPass.run(&mut k).unwrap();
        let has_fma = k.body.ops.iter().any(|op| matches!(op, Op::Fma { .. }));
        assert!(has_fma, "Mul + Add should fuse into Op::Fma: {:?}", k.body.ops);
        // The Add op is gone — replaced in-place by the Fma.
        let add_count = k
            .body
            .ops
            .iter()
            .filter(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }))
            .count();
        assert_eq!(add_count, 0, "Add should have been rewritten to Fma");
    }

    /// When the Mul has another consumer (e.g. `(a*b) + c` AND `(a*b) * 2`),
    /// fusion must NOT fire — the standalone Mul still needs to emit.
    #[test]
    fn skips_when_mul_has_multiple_consumers() {
        let mut k = fma_eligible_kernel();
        // Add a second use of the Mul result so use_count == 2.
        // (a * b) + c (existing) AND (a * b) + 1.0 (new).
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(10));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(4), rhs: ValueId::new(10) },
            ValueId::new(11),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(11),
            mask: None,
        });

        FmaFusionPass.run(&mut k).unwrap();
        let mul_count = k
            .body
            .ops
            .iter()
            .filter(|op| matches!(op, Op::BinOp { op: BinOpKind::Mul, .. }))
            .count();
        assert_eq!(mul_count, 1, "Mul with multiple consumers must survive: {:?}", k.body.ops);
        let fma_count = k.body.ops.iter().filter(|op| matches!(op, Op::Fma { .. })).count();
        assert_eq!(
            fma_count, 0,
            "FMA must not fire when the Mul has >1 consumer: {:?}",
            k.body.ops
        );
    }

    /// Integer Mul + Add must NOT fuse — Metal's `fma` only has float
    /// overloads (`fma(int, int, int)` is a compile error).
    #[test]
    fn skips_integer_mul_add() {
        let mut k = Kernel::new("int_fma_smoke");
        for (name, dtype, is_output) in [("a", DType::I32, false), ("c", DType::I32, true)] {
            k.params.push(Param {
                name: name.into(),
                dtype,
                shape: Shape::scalar(),
                is_output,
                kind: ParamKind::Tensor,
            });
        }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(2));
        k.body.push_op(Op::Const { value: 5 }, ValueId::new(3));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(4),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(4), rhs: ValueId::new(3) },
            ValueId::new(5),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(5),
            mask: None,
        });
        FmaFusionPass.run(&mut k).unwrap();
        assert!(
            !k.body.ops.iter().any(|op| matches!(op, Op::Fma { .. })),
            "integer Mul+Add must not fuse to Op::Fma: {:?}",
            k.body.ops
        );
    }
}
