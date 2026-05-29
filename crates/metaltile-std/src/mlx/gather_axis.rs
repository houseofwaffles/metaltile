//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gather along an axis — contiguous form of MLX's `gather_axis`.
//!
//! `out[o, a, i] = src[o, indices[o, a, i], i]` — for each output
//! element, the middle (axis) coordinate is looked up from `indices`
//! while the outer/inner coordinates pass through. One thread per
//! output element.
//!
//! Layout (row-contiguous):
//!   src:     [outer, axis_size, inner]  T
//!   indices: [outer, axis_out,  inner]  u32
//!   out:     [outer, axis_out,  inner]  T
//!
//! The general MLX kernel handles arbitrary strides / non-contiguous
//! src+idx via `elem_to_loc`; this port covers the row-contiguous case
//! (the shape `ensureRowContiguous` produces).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per output element over `outer*axis_out*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/gather_axis_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel(
    bench(
        op="indexing",
        subop="gather_axis",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_gather_axis<T>(
    src: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] axis_out: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    // out / indices share shape [outer, axis_out, inner]; `idx` indexes
    // both directly. Only the outer coord `o` and inner coord `i` are
    // needed to re-address `src` (which has `axis_size`, not `axis_out`).
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_out * inner);
    let gathered = load(indices[idx]);
    let src_off = (o * axis_size + gathered) * inner + i;
    store(out[idx], load(src[src_off]));
}
