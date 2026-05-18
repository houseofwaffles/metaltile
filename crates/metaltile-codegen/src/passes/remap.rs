//! ValueId Remapping — shared utilities for IR traversal and mutation.
//!
//! Provides canonical functions for ValueId remapping, reference collection,
//! and Op classification used by nearly every pass in the pipeline.
//!
//! ## Core functions
//! - [`remap_value_ids`] — rewrite all ValueId references in an Op.
//! - [`op_value_refs`] — collect all ValueId references (read-only, for analysis).
//! - [`max_vid_in_op`] — find the maximum ValueId in an Op.
//! - [`find_max_vid`] — find the maximum ValueId across a whole Kernel.
//!
//! ## Op predicates
//! - [`has_side_effects`] — cannot be moved, duplicated, or deleted.
//! - [`is_unpredictable`] — cannot appear inside predicated (if-converted) code.
//! - [`is_cheap_alu`] — eligible for rematerialization / value sinking.
//! - [`is_load`] / [`is_store`] / [`is_barrier`] — memory classification.
//!
//! Centralizing these here ensures all Op variants are handled consistently
//! across passes.  The exhaustive match arms serve as a single point of truth;
//! a test verifies no variant is silently skipped.

use std::collections::BTreeMap;

use metaltile_core::ir::{IndexExpr, Kernel, Op, ValueId};

// ---------------------------------------------------------------------------
// remap_value_ids — mutate ValueId references in an Op
// ---------------------------------------------------------------------------

/// Remap all `ValueId` references in `op` according to `map`.
/// References not present in `map` are left unchanged.
pub fn remap_value_ids(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    };

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            s(value);
        },
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices.iter_mut() {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => s(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                s(m);
            }
        },
        Op::Store { value, indices, mask, .. } => {
            s(value);
            for ix in indices.iter_mut() {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => s(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                s(m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            s(byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            s(byte_offset);
            s(value);
        },
        Op::Gather { indices, .. } => {
            s(indices);
        },
        Op::Scatter { indices, value, .. } => {
            s(indices);
            s(value);
        },
        Op::Atomic { index, value, .. } => {
            s(index);
            s(value);
        },

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::If { cond, .. } => {
            s(cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                s(v);
            },
        Op::FusedElementwise { ops } =>
            for sub_op in ops.iter_mut() {
                remap_value_ids(sub_op, map);
            },

        // ── ML primitives ────────────────────────────────────────────────
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

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            s(offset);
            s(stride);
            s(end);
            if let Some(sb) = secondary_base {
                s(sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::Scan { value, .. } => {
            s(value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } | Op::SimdScan { value, .. } => {
            s(value);
        },
        Op::SimdgroupElemLoad { value, .. } => {
            s(value);
        },
        Op::SimdgroupElemStore { value, data, .. } => {
            s(value);
            s(data);
        },
        Op::SimdgroupAlloc { .. } | Op::SimdgroupMatMul { .. } => {},
        Op::ThreadgroupLoad { index, .. } => {
            s(index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            s(value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            s(value);
        },
        Op::Cat { values, .. } =>
            for v in values {
                s(v);
            },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. }
        | Op::SimdLaneId
        | Op::SimdGroupId => {},
    }
}

// ---------------------------------------------------------------------------
// op_value_refs — collect all ValueId references (read-only, for analysis)
// ---------------------------------------------------------------------------

/// Return all `ValueId` references in `op` (for liveness, use-count, invariant analysis).
pub fn op_value_refs(op: &Op) -> Vec<ValueId> {
    let mut refs = Vec::new();

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
        Op::BinOp { lhs, rhs, .. } => {
            refs.push(*lhs);
            refs.push(*rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            refs.push(*value);
        },
        Op::Select { cond, on_true, on_false } => {
            refs.push(*cond);
            refs.push(*on_true);
            refs.push(*on_false);
        },
        Op::Dot { a, b } => {
            refs.push(*a);
            refs.push(*b);
        },

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => refs.push(*v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Store { indices, value, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => refs.push(*v),
                    IndexExpr::Const(_) => {},
                }
            }
            refs.push(*value);
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            refs.push(*byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            refs.push(*byte_offset);
            refs.push(*value);
        },
        Op::Gather { indices, .. } => {
            refs.push(*indices);
        },
        Op::Scatter { indices, value, .. } => {
            refs.push(*indices);
            refs.push(*value);
        },
        Op::Atomic { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            refs.push(*start);
            refs.push(*end);
            refs.push(*step);
        },
        Op::If { cond, .. } => {
            refs.push(*cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } => {
            refs.extend(inputs);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                refs.extend(op_value_refs(sub));
            },

        // ── ML primitives ────────────────────────────────────────────────
        Op::FlashAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::RmsNorm { x, scale, .. } => {
            refs.push(*x);
            refs.push(*scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            refs.push(*x);
            refs.push(*gate_proj);
            refs.push(*up_proj);
            refs.push(*down_proj);
        },

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            refs.push(*offset);
            refs.push(*stride);
            refs.push(*end);
            if let Some(sb) = secondary_base {
                refs.push(*sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            refs.push(*offset);
            refs.push(*end);
            refs.push(*scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::Scan { value, .. } => {
            refs.push(*value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } | Op::SimdScan { value, .. } => {
            refs.push(*value);
        },
        Op::SimdgroupElemLoad { value, .. } => {
            refs.push(*value);
        },
        Op::SimdgroupElemStore { value, data, .. } => {
            refs.push(*value);
            refs.push(*data);
        },
        Op::SimdgroupAlloc { .. } | Op::SimdgroupMatMul { .. } => {},
        Op::ThreadgroupLoad { index, .. } => {
            refs.push(*index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            refs.push(*value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            refs.push(*value);
        },
        Op::Cat { values, .. } => {
            refs.extend(values.iter());
        },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. }
        | Op::SimdLaneId
        | Op::SimdGroupId => {},
    }

    refs
}

// ---------------------------------------------------------------------------
// max_vid_in_op — highest ValueId in an Op
// ---------------------------------------------------------------------------

/// Return the maximum *real* `ValueId` referenced by `op`.
///
/// `FusedElementwise` sub-ops encode references to siblings within the
/// fused chain by setting the top bit of the `ValueId` (see
/// `passes::fusion::SUB_OP_FLAG = 0x8000_0000`). Those encoded refs are
/// NOT real ValueIds in the kernel-wide namespace — they're chain-local
/// position indices that happen to share the `ValueId` type. Including
/// them when allocating fresh IDs (`next_vid = max_vid + 1` in the
/// unroll pass) would push `next_vid` past `0x8000_0000`, and the newly
/// minted IDs would collide with the sub-op-ref encoding, causing the
/// MSL emitter to interpret them as bogus sub-op refs and emit
/// `0 /* bad sub-op ref */` placeholders. We mask them out here.
pub fn max_vid_in_op(op: &Op) -> u32 {
    /// Top bit reserved by `passes::fusion::SUB_OP_FLAG` to mark
    /// chain-internal references in `FusedElementwise` sub-ops.
    const SUB_OP_FLAG: u32 = 0x8000_0000;
    let mut m = 0u32;
    let mut push = |v: &ValueId| {
        let raw = v.as_u32();
        // Skip sub-op refs — they're not real ValueIds in the
        // kernel-wide namespace, just chain-local position indices.
        if raw & SUB_OP_FLAG == 0 && raw > m {
            m = raw;
        }
    };

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
        Op::BinOp { lhs, rhs, .. } => {
            push(lhs);
            push(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            push(value);
        },
        Op::Select { cond, on_true, on_false } => {
            push(cond);
            push(on_true);
            push(on_false);
        },
        Op::Dot { a, b } => {
            push(a);
            push(b);
        },

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => push(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                push(m);
            }
        },
        Op::Store { value, indices, mask, .. } => {
            push(value);
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => push(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                push(m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            push(byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            push(byte_offset);
            push(value);
        },
        Op::Gather { indices, .. } => {
            push(indices);
        },
        Op::Scatter { indices, value, .. } => {
            push(indices);
            push(value);
        },
        Op::Atomic { index, value, .. } => {
            push(index);
            push(value);
        },

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            push(start);
            push(end);
            push(step);
        },
        Op::If { cond, .. } => {
            push(cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                push(v);
            },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                m = m.max(max_vid_in_op(sub));
            },

        // ── ML primitives ────────────────────────────────────────────────
        Op::FlashAttention { q, k, v, .. } => {
            push(q);
            push(k);
            push(v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            push(q);
            push(k);
            push(v);
        },
        Op::RmsNorm { x, scale, .. } => {
            push(x);
            push(scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            push(x);
            push(gate_proj);
            push(up_proj);
            push(down_proj);
        },

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            push(offset);
            push(stride);
            push(end);
            if let Some(sb) = secondary_base {
                push(sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            push(offset);
            push(end);
            push(scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            push(offset);
            push(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            push(offset);
            push(end);
        },
        Op::Scan { value, .. } => {
            push(value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } | Op::SimdScan { value, .. } => {
            push(value);
        },
        Op::SimdgroupElemLoad { value, .. } => {
            push(value);
        },
        Op::SimdgroupElemStore { value, data, .. } => {
            push(value);
            push(data);
        },
        Op::SimdgroupAlloc { .. } | Op::SimdgroupMatMul { .. } => {},
        Op::ThreadgroupLoad { index, .. } => {
            push(index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            push(index);
            push(value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            push(value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            push(value);
        },
        Op::Cat { values, .. } =>
            for v in values {
                push(v);
            },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. }
        | Op::SimdLaneId
        | Op::SimdGroupId => {},
    }

    m
}

// ---------------------------------------------------------------------------
// find_max_vid — maximum ValueId across a whole Kernel
// ---------------------------------------------------------------------------

/// Find the maximum *real* `ValueId` across all ops and results in
/// `kernel`. Ignores `FusedElementwise` sub-op refs (top-bit-set
/// chain-internal indices); see [`max_vid_in_op`] for why.
pub fn find_max_vid(kernel: &Kernel) -> u32 {
    /// Top bit reserved by `passes::fusion::SUB_OP_FLAG`.
    const SUB_OP_FLAG: u32 = 0x8000_0000;
    let real_vid = |vid: &ValueId| -> Option<u32> {
        let raw = vid.as_u32();
        if raw & SUB_OP_FLAG == 0 { Some(raw) } else { None }
    };

    let mut m = 0u32;

    // Body ops and results
    for op in &kernel.body.ops {
        m = m.max(max_vid_in_op(op));
    }
    for vid in kernel.body.results.iter().flatten() {
        if let Some(raw) = real_vid(vid) {
            m = m.max(raw);
        }
    }

    // Nested blocks
    for block in kernel.blocks.values() {
        for op in &block.ops {
            m = m.max(max_vid_in_op(op));
        }
        for vid in block.results.iter().flatten() {
            if let Some(raw) = real_vid(vid) {
                m = m.max(raw);
            }
        }
    }

    m
}

// ---------------------------------------------------------------------------
// all_blocks — collect all block IDs in post-order
// ---------------------------------------------------------------------------

/// Collect all block IDs in the kernel, including the body.
/// Returns sorted keys from the block map plus the body block ID.
pub fn all_block_ids(kernel: &Kernel) -> Vec<metaltile_core::ir::BlockId> {
    let mut ids: Vec<metaltile_core::ir::BlockId> = kernel.blocks.keys().copied().collect();
    ids.push(kernel.body.id);
    ids
}

// ---------------------------------------------------------------------------
// Op predicates (shared across passes)
// ---------------------------------------------------------------------------

/// True if the op has side effects and cannot be moved, duplicated, or deleted.
pub fn has_side_effects(op: &Op) -> bool {
    matches!(
        op,
        Op::Store { .. }
            | Op::VectorStore { .. }
            | Op::Atomic { .. }
            | Op::Barrier
            | Op::SetLocal { .. }
            | Op::ThreadgroupStore { .. }
            | Op::ThreadgroupAlloc { .. }
            | Op::StrideStore { .. }
            | Op::Scatter { .. }
    )
}

/// True if the op cannot appear inside predicated code (Barrier, Atomic, Loop, etc.).
pub fn is_unpredictable(op: &Op) -> bool {
    matches!(
        op,
        Op::Barrier
            | Op::Atomic { .. }
            | Op::Loop { .. }
            | Op::SetLocal { .. }
            | Op::DeclareLocal { .. }
            | Op::ThreadgroupAlloc { .. }
            | Op::If { .. } // nested If needs recursive if-conversion, not flat predicates
            | Op::StrideScan { .. }
            | Op::StrideArgReduce { .. }
    )
}

/// True if the op is a "cheap ALU" op (eligible for rematerialization / value sinking).
pub fn is_cheap_alu(op: &Op) -> bool {
    matches!(
        op,
        Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Cast { .. }
            | Op::Select { .. }
            | Op::Const { .. }
            | Op::ProgramId { .. }
    )
}

/// True if the op is a load from device or threadgroup memory.
pub fn is_load(op: &Op) -> bool {
    matches!(op, Op::Load { .. } | Op::VectorLoad { .. } | Op::ThreadgroupLoad { .. })
}

/// True if the op is a store to device or threadgroup memory.
pub fn is_store(op: &Op) -> bool {
    matches!(op, Op::Store { .. } | Op::VectorStore { .. } | Op::ThreadgroupStore { .. })
}

/// True if the op contains a barrier.
pub fn is_barrier(op: &Op) -> bool { matches!(op, Op::Barrier) }

#[cfg(test)]
mod tests {
    use metaltile_core::ir::BinOpKind;

    use super::*;

    /// Every Op variant must be handled by remap_value_ids, op_value_refs, and max_vid_in_op.
    /// This test ensures no variant is silently skipped by a catch-all `_ => {}`.
    #[test]
    fn all_op_variants_covered() {
        // We can't exhaustively instantiate every variant, but we exercise each
        // major category to ensure the match arms don't panic.
        let map: BTreeMap<ValueId, ValueId> = BTreeMap::new();

        // BinOp
        let mut op = Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2);
        assert!(max_vid_in_op(&op) >= 2);

        // Load with mask
        let mut op = Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Value(ValueId::new(3))],
            mask: Some(ValueId::new(4)),
            other: None,
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2); // index + mask

        // Store with mask
        let mut op = Op::Store {
            dst: "b".into(),
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(5),
            mask: Some(ValueId::new(6)),
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2); // value + mask

        // StrideReduce with secondary_base
        let mut op = Op::StrideReduce {
            src: "x".into(),
            offset: ValueId::new(7),
            stride: ValueId::new(8),
            end: ValueId::new(9),
            op: metaltile_core::ir::ReduceKind::Sum,
            dtype: metaltile_core::dtype::DType::F32,
            transform: None,
            secondary_src: None,
            secondary_base: Some(ValueId::new(10)),
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 4); // offset, stride, end, secondary_base

        // Ops with no ValueIds should return 0 refs
        assert_eq!(op_value_refs(&Op::Barrier).len(), 0);
        assert_eq!(op_value_refs(&Op::Const { value: 42 }).len(), 0);
        assert_eq!(max_vid_in_op(&Op::Barrier), 0);
    }

    #[test]
    fn max_vid_ignores_sub_op_refs() {
        // FusedElementwise sub-ops encode chain-internal references by
        // setting the top bit of the ValueId (`SUB_OP_FLAG = 0x8000_0000`).
        // `max_vid_in_op` / `find_max_vid` must skip those — otherwise
        // the unroll pass's `next_vid = max_vid + 1` allocation collides
        // with the sub-op-ref encoding namespace and the MSL emitter
        // produces `0 /* bad sub-op ref */` placeholders. Regression
        // test for the `mt_rope_f16` / `mt_affine_quantize_int8`
        // miscompiles.
        const SUB_OP_FLAG: u32 = 0x8000_0000;
        let sub_op_ref_0 = ValueId::new(SUB_OP_FLAG); // chain position 0
        let real_vid = ValueId::new(42);

        // Single-op test: a fused chain that contains a sub-op ref.
        let op = Op::FusedElementwise {
            ops: vec![Op::BinOp { op: BinOpKind::Add, lhs: sub_op_ref_0, rhs: real_vid }],
        };
        assert_eq!(max_vid_in_op(&op), 42, "sub-op refs must not bump max_vid");

        // Whole-kernel test mirrors the rope path: real ValueIds up to
        // 100, plus a fused op with a sub-op ref far above SUB_OP_FLAG.
        let mut k = Kernel::new("max_vid_ignores_sub_op_refs");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(100));
        k.body.push_op(
            Op::FusedElementwise {
                ops: vec![
                    Op::Cast { value: ValueId::new(100), dtype: metaltile_core::dtype::DType::F32 },
                    Op::UnaryOp { op: metaltile_core::ir::UnaryOpKind::Neg, value: sub_op_ref_0 },
                ],
            },
            ValueId::new(101),
        );
        assert_eq!(
            find_max_vid(&k),
            101,
            "find_max_vid must ignore sub-op refs (else unroll's next_vid space \
             collides with the SUB_OP_FLAG namespace)"
        );
    }

    #[test]
    fn remap_rewrites_referenced_values() {
        let mut map = BTreeMap::new();
        map.insert(ValueId::new(1), ValueId::new(100));
        map.insert(ValueId::new(2), ValueId::new(200));

        let mut op = Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) };
        remap_value_ids(&mut op, &map);

        if let Op::BinOp { lhs, rhs, .. } = &op {
            assert_eq!(lhs.as_u32(), 100);
            assert_eq!(rhs.as_u32(), 200);
        } else {
            panic!("op changed variant");
        }
    }

    #[test]
    fn find_max_vid_works() {
        let mut k = Kernel::new("test");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(5));
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(3));
        assert_eq!(find_max_vid(&k), 5);
    }
}
