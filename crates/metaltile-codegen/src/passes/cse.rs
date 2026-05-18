//! Common Subexpression Elimination — local value numbering.
//!
//! Performs block-local value numbering: when two ops compute the same result
//! (identical opcode and operands), the second is eliminated and all downstream
//! uses are rerouted to the first.  Commutative binary ops (Add, Mul, Max, Min,
//! BitAnd, BitOr, BitXor, CmpEq, CmpNe) are canonicalized to catch `a+b` vs `b+a`.
//!
//! ## CSE-eligible ops
//!
//! | Op | Notes |
//! |----|-------|
//! | `BinOp` | Commutative ops are canonicalized |
//! | `UnaryOp` | |
//! | `Cast` | |
//! | `Activation` | |
//! | `Select` | All three operands must match |
//! | `Load` | Only from read-only (const) params |
//!
//! Never eligible: `Store`, `Reduce`, `StrideReduce`, `Loop`, `Barrier`, `Atomic`,
//! and any other op with side effects.
//!
//! ## References
//! - Cocke & Schwartz (1970), "Programming Languages and their Compilers",
//!   Courant Institute.  The original description of value numbering for CSE.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §8.5 (global common subexpressions).
//! - Briggs, Cooper & Simpson (1997), "Value numbering", Rice University
//!   COMP 512 course notes.  Survey of local, superlocal, and global value
//!   numbering algorithms.

use metaltile_core::{
    dtype::DType,
    error::Result,
    ir::{
        ActKind,
        BinOpKind,
        Block,
        BlockId,
        IndexExpr,
        Kernel,
        Op,
        ParamKind,
        UnaryOpKind,
        ValueId,
    },
};
use rustc_hash::FxHashMap;

/// A structural key for CSE: captures the opcode and operands in a hashable form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OpKey {
    BinOp { op: BinOpKind, lhs: u32, rhs: u32 },
    UnaryOp { op: UnaryOpKind, value: u32 },
    Cast { dtype: DType, value: u32 },
    Activation { kind: ActKind, value: u32 },
    Select { cond: u32, on_true: u32, on_false: u32 },
    Load { src: String, idx0: IndexExpr },
}

pub struct CsePass;

impl super::Pass for CsePass {
    fn name(&self) -> &str { "cse" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Determine which params are read-only.
        let read_only: std::collections::BTreeSet<String> = kernel
            .params
            .iter()
            .filter(|p| !p.is_output && matches!(p.kind, ParamKind::Tensor | ParamKind::Strided))
            .map(|p| p.name.clone())
            .collect();

        // CSE the body, then propagate its remap to every nested block so that
        // ops inside loops / if-branches keep pointing at the surviving SSA
        // value rather than the eliminated duplicate.
        let body_remap = cse_block(&mut kernel.body, &read_only);
        if !body_remap.is_empty() {
            for block in kernel.blocks.values_mut() {
                for op in block.ops.iter_mut() {
                    replace_values(op, &body_remap);
                }
            }
        }

        // CSE each nested block. After each one, propagate its remap to the
        // kernel body and to every other block — a value defined in one
        // nested block may be referenced from a sibling or from the body.
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in &block_ids {
            let Some(mut block) = kernel.blocks.remove(bid) else { continue };
            let remap = cse_block(&mut block, &read_only);
            kernel.blocks.insert(*bid, block);
            if !remap.is_empty() {
                for op in kernel.body.ops.iter_mut() {
                    replace_values(op, &remap);
                }
                for other in kernel.blocks.values_mut() {
                    for op in other.ops.iter_mut() {
                        replace_values(op, &remap);
                    }
                }
            }
        }

        Ok(())
    }
}

/// CSE the given block. Returns the SSA-value remap (`old → new`) that the
/// caller must also apply to any other block that may reference values
/// defined here (loop bodies / if branches / the kernel body).
fn cse_block(
    block: &mut Block,
    read_only: &std::collections::BTreeSet<String>,
) -> FxHashMap<ValueId, ValueId> {
    let n = block.ops.len();

    // Phase 1: find duplicates and build old_vid -> replacement_vid map.
    let mut table: FxHashMap<OpKey, ValueId> = FxHashMap::default();
    let mut old_to_new: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut skip: Vec<bool> = vec![false; n];

    for (i, op) in block.ops.iter().enumerate() {
        let Some(key) = op_key(op, read_only) else {
            continue;
        };
        let Some(&Some(vid)) = block.results.get(i) else {
            continue;
        };

        if let Some(&existing_vid) = table.get(&key) {
            // Duplicate found: remap `vid` to `existing_vid`.
            old_to_new.insert(vid, existing_vid);
            skip[i] = true;
        } else {
            table.insert(key, vid);
        }
    }

    if old_to_new.is_empty() {
        return old_to_new;
    }

    // Phase 2: remap ValueId references in all surviving ops.
    for op in block.ops.iter_mut() {
        replace_values(op, &old_to_new);
    }

    // Phase 3: rebuild the block without skipped ops.
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    for i in 0..n {
        if !skip[i] {
            new_ops.push(old_ops[i].clone());
            new_results.push(old_results[i]);
        }
    }

    block.ops = new_ops;
    block.results = new_results;
    old_to_new
}

/// Build an OpKey for an op if it's CSE-eligible.
fn op_key(op: &Op, read_only: &std::collections::BTreeSet<String>) -> Option<OpKey> {
    match op {
        Op::BinOp { op: kind, lhs, rhs } => {
            let (l, r) = canonicalize_binop(*kind, lhs.as_u32(), rhs.as_u32());
            Some(OpKey::BinOp { op: *kind, lhs: l, rhs: r })
        },
        Op::UnaryOp { op: kind, value } =>
            Some(OpKey::UnaryOp { op: *kind, value: value.as_u32() }),
        Op::Cast { dtype, value } => Some(OpKey::Cast { dtype: *dtype, value: value.as_u32() }),
        Op::Activation { kind, value } =>
            Some(OpKey::Activation { kind: *kind, value: value.as_u32() }),
        Op::Select { cond, on_true, on_false } => Some(OpKey::Select {
            cond: cond.as_u32(),
            on_true: on_true.as_u32(),
            on_false: on_false.as_u32(),
        }),
        Op::Load { src, indices, .. } =>
            if read_only.contains(src.as_str()) && indices.len() == 1 {
                Some(OpKey::Load { src: src.clone(), idx0: indices[0].clone() })
            } else {
                None
            },
        _ => None,
    }
}

/// For commutative binary ops, sort operands so that `a+b` and `b+a` hash identically.
fn canonicalize_binop(op: BinOpKind, lhs: u32, rhs: u32) -> (u32, u32) {
    let is_commutative = matches!(
        op,
        BinOpKind::Add
            | BinOpKind::Mul
            | BinOpKind::Max
            | BinOpKind::Min
            | BinOpKind::And
            | BinOpKind::Or
            | BinOpKind::Xor
            | BinOpKind::BitAnd
            | BinOpKind::BitOr
            | BinOpKind::BitXor
            | BinOpKind::CmpEq
            | BinOpKind::CmpNe
    );
    if is_commutative && lhs > rhs { (rhs, lhs) } else { (lhs, rhs) }
}

/// Replace all ValueId references in `op` using the remapping map.
fn replace_values(op: &mut Op, map: &FxHashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new_v) = map.get(v) {
            *v = new_v;
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
                replace_values(op, map);
            },
        Op::VectorLoad { byte_offset, .. } => s(byte_offset),
        Op::VectorExtract { .. } => {},
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
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
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
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        ir::{BinOpKind, IndexExpr, Param, ParamKind},
        shape::Shape,
    };

    use super::*;
    use crate::passes::Pass;

    fn read_only_param(name: &str) -> Param {
        Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        }
    }

    #[test]
    fn eliminates_duplicate_binop() {
        let mut k = Kernel::new("cse_binop");
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        // First 3+4
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        // Second 3+4 (duplicate)
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(3),
        );
        // Both used in a final store to keep alive.
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(3),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        // Only one BinOp should remain.
        let adds: Vec<_> = k
            .body
            .ops
            .iter()
            .filter(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }))
            .collect();
        assert_eq!(adds.len(), 1, "duplicate BinOp should be CSE'd");
    }

    #[test]
    fn canonicalizes_commutative_binop() {
        let mut k = Kernel::new("cse_commute");
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        // 3+4
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        // 4+3 (commuted duplicate)
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(0) },
            ValueId::new(3),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(3),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        let adds: Vec<_> = k
            .body
            .ops
            .iter()
            .filter(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. }))
            .collect();
        assert_eq!(adds.len(), 1, "commutative duplicate should be CSE'd");
    }

    #[test]
    fn eliminates_duplicate_cast() {
        let mut k = Kernel::new("cse_cast");
        k.body.push_op(Op::Const { value: 5 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        let casts: Vec<_> = k.body.ops.iter().filter(|op| matches!(op, Op::Cast { .. })).collect();
        assert_eq!(casts.len(), 1, "duplicate Cast should be CSE'd");
    }

    #[test]
    fn cse_read_only_load() {
        let mut k = Kernel::new("cse_load");
        k.params.push(read_only_param("weights"));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        let idx = IndexExpr::Value(ValueId::new(0));
        k.body.push_op(
            Op::Load { src: "weights".into(), indices: vec![idx.clone()], mask: None, other: None },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load { src: "weights".into(), indices: vec![idx], mask: None, other: None },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        let loads: Vec<_> = k.body.ops.iter().filter(|op| matches!(op, Op::Load { .. })).collect();
        assert_eq!(loads.len(), 1, "duplicate read-only Load should be CSE'd");
    }

    #[test]
    fn does_not_cse_side_effecting_ops() {
        let mut k = Kernel::new("cse_noside");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        // Two Stores — should NOT be CSE'd (side effects).
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(0),
            mask: None,
        });
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(0),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        assert_eq!(stores.len(), 2, "Stores should not be CSE'd");
    }

    #[test]
    fn different_binops_not_csed() {
        let mut k = Kernel::new("cse_diff");
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(3),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(3),
            mask: None,
        });
        CsePass.run(&mut k).unwrap();
        let binops: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::BinOp { .. })).collect();
        assert_eq!(binops.len(), 2, "different binops should not be CSE'd");
    }

    /// Regression: when CSE eliminates a duplicate in the kernel body, any
    /// reference to the dropped SSA value from inside a nested block must
    /// also be remapped. Triggered by `sdpa_decode`'s `q_off = q_head * head_dim`
    /// duplicating an inline `q_head * head_dim` later consumed inside an
    /// inner loop — pre-fix, the inner-block reference became dangling.
    #[test]
    fn cse_remaps_references_inside_nested_blocks() {
        use metaltile_core::ir::{Block, BlockId, VarId};

        let mut k = Kernel::new("cse_cross_block");
        // Outer body:
        //   v0 = const 1
        //   v1 = const 2
        //   v2 = add(v0, v1)            ← will survive
        //   v3 = add(v0, v1)            ← duplicate; CSE drops this, remaps v3 → v2
        //   v4 = const 5
        //   loop body=inner_block       ← inner block references v3
        //   v6 = mul(v2, v4)            ← post-loop use of survivor
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(3),
        );
        k.body.push_op(Op::Const { value: 5 }, ValueId::new(4));

        let inner_id = BlockId::new(1);
        let mut inner = Block::new(inner_id);
        inner.push_op(Op::Const { value: 0 }, ValueId::new(100));
        // Inner block consumes the duplicate (v3) — must be remapped to v2.
        inner.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(3), rhs: ValueId::new(100) },
            ValueId::new(101),
        );
        k.blocks.insert(inner_id, inner);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(4),
            step: ValueId::new(0),
            body: inner_id,
        });
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(4) },
            ValueId::new(6),
        );

        CsePass.run(&mut k).unwrap();

        // The duplicate v3 must be gone from the body.
        let body_add_count =
            k.body.ops.iter().filter(|op| matches!(op, Op::BinOp { op: BinOpKind::Add, .. })).count();
        assert_eq!(body_add_count, 1, "duplicate add in body must be CSE'd away");

        // The inner block must no longer reference v3 — its add must now use v2.
        let inner = k.blocks.get(&inner_id).expect("inner block survived");
        let inner_add = inner
            .ops
            .iter()
            .find_map(|op| match op {
                Op::BinOp { op: BinOpKind::Add, lhs, rhs } => Some((*lhs, *rhs)),
                _ => None,
            })
            .expect("inner block must still contain its Add op");
        assert!(
            inner_add.0 != ValueId::new(3) && inner_add.1 != ValueId::new(3),
            "inner-block reference to dropped duplicate v3 must be remapped (got {inner_add:?})"
        );
        assert!(
            inner_add.0 == ValueId::new(2) || inner_add.1 == ValueId::new(2),
            "inner-block ref must now point at the survivor v2 (got {inner_add:?})"
        );
    }
}
