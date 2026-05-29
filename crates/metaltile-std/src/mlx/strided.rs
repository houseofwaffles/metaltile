//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Strided copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel(
    bench(
        op="strided_copy",
        subop="strided_copy",
        class=StridedCopy,
        m=1024,
        n=4096,
        pad=128,
        tol=0.0,
        mlx="copy_g_nd2{tn}{tn}",
        metal_file="copy.metal",
    )
)]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}

// ─── mt_strided_copy_nd ──────────────────────────────────────────────────
//
// General N-D strided copy — the MLX `copy_g` / `copy_g_nd{1,2,3}`
// counterpart. The 2-D `mt_strided_copy` above only handles a
// row-major-padded `[rows, cols]` source; this kernel copies an
// arbitrary-rank logical tensor out of a source buffer whose physical
// layout is described by per-dimension `shape` + `strides` arrays.
//
// The destination is always contiguous row-major: output element `p`
// (a flat index in `[0, n_out)`) maps to the multi-index obtained by
// unravelling `p` against `out_shape` (== logical `shape`), then the
// source byte offset is `Σ_d coord_d · strides[d]`. This is exactly
// MLX's `elem_to_loc` (`mlx/backend/metal/kernels/utils.h`).
//
// Because the source strides are *arbitrary* (not necessarily a
// padded row-major view), this generalises:
//   - padded copies         (the 2-D `mt_strided_copy` case),
//   - transposes            (strides permuted vs shape),
//   - broadcasts            (a stride of 0 on a broadcast axis),
//   - any slice / dilation  (non-unit innermost stride).
//
// Inputs:
//   src     — source data buffer (raw, physically strided)
//   shape   — [rank]   u32  logical extent of each dimension
//   strides — [rank]   u32  element stride of each source dimension
//   out     — [n_out]  contiguous row-major output
//
// Constexpr:
//   rank    — number of dimensions (logical). Compile-time constant so
//             the unravel loop is fully unrolled — no dynamic trip count.
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D** — one thread per output element, no cross-thread
//   cooperation. `program_id::<0>()` is the flat output index.
// - **Grid: `[n_out, 1, 1]`, TPG: `[1, 1, 1]`** (or any
//   `grid·tpg == n_out` split). `n_out == Π shape[d]`.
// - **`rank >= 1`.** `shape` and `strides` must each hold exactly
//   `rank` u32 entries; a short buffer reads out of bounds.
// - The unravel walks dimensions **last → first**: the running
//   remainder is divided by `shape[d]` from `d = rank-1` down to `0`,
//   so `strides` is interpreted in the same major-to-minor order as
//   `shape` (row-major logical indexing).
#[kernel(
    bench(
        op="strided_copy",
        subop="strided_copy_nd",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_strided_copy_nd<T>(
    src: Tensor<T>,
    shape: Tensor<u32>,
    strides: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] rank: u32,
) {
    let p = program_id::<0>();
    // Unravel the flat output index `p` against `shape`, walking
    // dimensions from the innermost (last) to the outermost (first).
    // `rem` carries the not-yet-consumed portion of `p`; at each step
    // `coord = rem % shape[d]` peels off dimension `d`'s index and
    // `rem /= shape[d]` advances to the next-coarser dimension. The
    // source offset accumulates `coord · strides[d]`.
    let mut rem = p;
    let mut src_off = 0u32;
    for _i in range(0u32, rank, 1u32) {
        // d counts down: rank-1, rank-2, ..., 0.
        let d = rank - 1u32 - _i;
        let extent = load(shape[d]);
        let coord = rem - (rem / extent) * extent; // rem % extent
        rem = rem / extent;
        src_off = src_off + coord * load(strides[d]);
    }
    store(out[p], load(src[src_off]));
}
