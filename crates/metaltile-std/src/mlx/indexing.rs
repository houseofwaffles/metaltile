//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Strided indexing kernels — `gather_front`, `scatter`, `masked_scatter`.
//!
//! The contiguous along-an-axis forms (`gather_axis` / `scatter_axis`)
//! ship in their own modules. This file covers the three remaining
//! `indexing/` ops from MLX's `mlx/backend/metal/kernels/indexing/`:
//!
//! - **`gather_front`** — gather whole rows by a first-axis index:
//!   `out[r, :] = src[indices[r], :]`. The embedding-table-style
//!   row gather where the index selects which source row to copy.
//!   MLX reference: `indexing/gather_front.h`.
//! - **`scatter`** — the inverse: write rows into index-selected slots
//!   of a pre-initialized output, `out[indices[r], :] = updates[r, :]`.
//!   Assignment form (`reduce = None`) — colliding indices race, so the
//!   caller must supply distinct indices for a deterministic result,
//!   matching MLX `scatter` with no reduction.
//! - **`masked_scatter`** — gather with a per-element mask:
//!   `out[i] = mask[i] ? src[scatter_offsets[i]] : out[i]`. The masked
//!   elements pull from a compacted `src` via a precomputed offset
//!   table; unmasked elements keep `out`'s prior value. MLX reference:
//!   `indexing/masked_scatter.h`.
//!
//! All three are one-thread-per-output Grid3D kernels — no cross-thread
//! cooperation, so the reduction-mode dispatch hazards do not apply.
//! Indices / offsets / mask are `u32` tensors (a `0/1` mask rather than
//! a `bool` tensor — `u32` is the dtype the DSL exposes for index
//! buffers, and the caller packs the mask as `0u32` / `1u32`).
//!
//! Codegen-only; correctness pinned by
//! `tests/indexing_gpu_correctness.rs`.

use metaltile::kernel;

/// First-axis row gather — `out[r, i] = src[indices[r], i]`.
///
/// `src` is `[n_src_rows, row_width]`, `indices` is `[n_out_rows]`
/// (u32), `out` is `[n_out_rows, row_width]`. One thread per output
/// element; the output element `idx` decomposes into `(r, i)` and the
/// source row is looked up from `indices[r]`.
///
/// `n_elems = n_out_rows * row_width` is passed as a constexpr so
/// threads past the output (a Grid3D dispatch rounds the thread count
/// up to a multiple of TPG) early-out — they must not read `indices`
/// out of bounds or write a stray `out` slot.
#[kernel(
    bench(
        op="indexing",
        subop="gather_front",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_gather_front<T>(
    src: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] row_width: u32,
    #[constexpr] n_elems: u32,
) {
    let idx = program_id::<0>();
    if idx < n_elems {
        let r = idx / row_width;
        let i = idx - r * row_width;
        let src_row = load(indices[r]);
        store(out[idx], load(src[src_row * row_width + i]));
    }
}

/// First-axis row scatter — `out[indices[r], i] = updates[r, i]`.
///
/// `updates` is `[n_upd_rows, row_width]`, `indices` is `[n_upd_rows]`
/// (u32), `out` is `[n_out_rows, row_width]` and is pre-initialized by
/// the caller (typically a copy of the source). One thread per update
/// element. Assignment (no-reduce) form — distinct `indices` are
/// required for a deterministic result; colliding indices race, the
/// same contract as MLX `scatter` with `reduce = None`.
///
/// `n_elems = n_upd_rows * row_width` is passed as a constexpr so
/// threads past the update count early-out — without the guard a
/// stray thread reads `indices` / `updates` out of bounds and scatters
/// garbage into `out`.
#[kernel(
    bench(
        op="indexing",
        subop="scatter",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_scatter<T>(
    updates: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] row_width: u32,
    #[constexpr] n_elems: u32,
) {
    let idx = program_id::<0>();
    if idx < n_elems {
        let r = idx / row_width;
        let i = idx - r * row_width;
        let out_row = load(indices[r]);
        store(out[out_row * row_width + i], load(updates[idx]));
    }
}

/// Masked gather-scatter — `out[i] = mask[i] ? src[offsets[i]] : out[i]`.
///
/// One thread per output element. `mask` is a `u32` `0/1` buffer the
/// same length as `out`; `offsets` (also `u32`, same length) is the
/// precomputed compacted-`src` index for each masked position. Where
/// the mask is `0` the thread re-reads and re-writes `out`'s prior
/// value (a no-op store rather than a branch — keeps the kernel
/// branch-divergence-free). `out` must be pre-initialized.
///
/// MLX's reference compacts `src` to one batch's worth of rows and
/// derives `batch_idx` from a `mask_batch_size`; this port flattens to
/// the single-batch case (`offsets` already absolute into `src`),
/// which is what the FFAI masked-cache-update path needs.
#[kernel(
    bench(
        op="indexing",
        subop="masked_scatter",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_masked_scatter<T>(
    mask: Tensor<u32>,
    offsets: Tensor<u32>,
    src: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] n_elems: u32,
) {
    let idx = program_id::<0>();
    if idx < n_elems {
        let m = load(mask[idx]);
        let off = load(offsets[idx]);
        let prev = load(out[idx]);
        let picked = load(src[off]);
        // Branchless: select the gathered value when masked, else keep
        // the prior `out` value. `off` is read unconditionally — the
        // caller's offset table must hold an in-bounds index even for
        // unmasked slots (MLX fills them with 0; any valid index works
        // since the value is discarded).
        let chosen = select(m > 0u32, picked, prev);
        store(out[idx], chosen);
    }
}
