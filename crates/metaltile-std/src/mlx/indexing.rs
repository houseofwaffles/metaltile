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
#[kernel]
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
#[kernel]
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
#[kernel]
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

/// New-syntax correctness for the row gather/scatter + masked-scatter index
/// kernels (Grid3D, exact). Oracles replicate the index math on dtype-rounded
/// data; scatter uses distinct indices (no collisions).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_gather_front, mt_masked_scatter, mt_scatter};
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_gather_front(dt: DType) -> TestSetup {
        let (n_src, n_out, w) = (5usize, 3usize, 4usize);
        let src: Vec<f32> = (0..n_src * w).map(|i| i as f32 * 0.1 - 1.0).collect();
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let rows: Vec<u32> = vec![2, 0, 4];
        let n_elems = n_out * w;
        let expected: Vec<f32> =
            (0..n_elems).map(|idx| src_dt[rows[idx / w] as usize * w + idx % w]).collect();
        TestSetup::new(mt_gather_front::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&rows), DType::U32))
            .input(TestBuffer::zeros("out", n_elems, dt))
            .constexpr("row_width", w as u32)
            .constexpr("n_elems", n_elems as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_elems, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_scatter(dt: DType) -> TestSetup {
        let (n_upd, n_out, w) = (3usize, 5usize, 4usize);
        let updates: Vec<f32> = (0..n_upd * w).map(|i| i as f32 * 0.1 - 0.5).collect();
        let upd_dt = unpack_f32(&pack_f32(&updates, dt), dt);
        let rows: Vec<u32> = vec![0, 2, 4]; // distinct → no collisions
        let n_elems = n_upd * w;
        let mut expected = vec![0.0f32; n_out * w];
        for idx in 0..n_elems {
            expected[rows[idx / w] as usize * w + idx % w] = upd_dt[idx];
        }
        TestSetup::new(mt_scatter::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("updates", pack_f32(&updates, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&rows), DType::U32))
            .input(TestBuffer::zeros("out", n_out * w, dt))
            .constexpr("row_width", w as u32)
            .constexpr("n_elems", n_elems as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_elems, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_masked_scatter(dt: DType) -> TestSetup {
        let n_elems = 64usize;
        let src: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.05 - 1.5).collect();
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let mask: Vec<u32> = (0..n_elems).map(|i| (i % 2) as u32).collect();
        let offsets: Vec<u32> = (0..n_elems).map(|i| ((i * 13 + 1) % n_elems) as u32).collect();
        // out is pre-zeroed → unmasked slots keep 0.
        let expected: Vec<f32> = (0..n_elems)
            .map(|i| if mask[i] != 0 { src_dt[offsets[i] as usize] } else { 0.0 })
            .collect();
        TestSetup::new(mt_masked_scatter::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("mask", u32_bytes(&mask), DType::U32))
            .input(TestBuffer::from_vec("offsets", u32_bytes(&offsets), DType::U32))
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::zeros("out", n_elems, dt))
            .constexpr("n_elems", n_elems as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_elems, 256)
    }
}

/// New-syntax benchmarks for the index kernels.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_gather_front, mt_masked_scatter, mt_scatter};

    fn u32_bytes(v: impl Iterator<Item = u32>) -> Vec<u8> {
        v.flat_map(|x| x.to_le_bytes()).collect()
    }

    #[bench(name = "mlx/indexing/gather_front", dtypes = [f32, f16, bf16])]
    fn bench_gather_front(dt: DType) -> BenchSetup {
        let (n_src, n_out, w) = (8192usize, 8192usize, 256usize);
        let n_elems = n_out * w;
        BenchSetup::new(mt_gather_front::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("src", n_src * w, dt))
            .buffer(BenchBuffer::from_vec(
                "indices",
                u32_bytes((0..n_out).map(|r| (r % n_src) as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", n_elems, dt).output())
            .constexpr("row_width", w as u32)
            .constexpr("n_elems", n_elems as u32)
            .grid_1d(n_elems, 256)
            .bytes_moved((2 * n_elems * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/indexing/scatter", dtypes = [f32, f16, bf16])]
    fn bench_scatter(dt: DType) -> BenchSetup {
        let (n_upd, n_out, w) = (8192usize, 8192usize, 256usize);
        let n_elems = n_upd * w;
        BenchSetup::new(mt_scatter::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("updates", n_elems, dt))
            .buffer(BenchBuffer::from_vec(
                "indices",
                u32_bytes((0..n_upd).map(|r| (r % n_out) as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", n_out * w, dt).output())
            .constexpr("row_width", w as u32)
            .constexpr("n_elems", n_elems as u32)
            .grid_1d(n_elems, 256)
            .bytes_moved((2 * n_elems * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/indexing/masked_scatter", dtypes = [f32, f16, bf16])]
    fn bench_masked_scatter(dt: DType) -> BenchSetup {
        let n_elems = 8 * 1024 * 1024usize;
        BenchSetup::new(mt_masked_scatter::kernel_ir_for(dt))
            .buffer(BenchBuffer::from_vec(
                "mask",
                u32_bytes((0..n_elems).map(|i| (i % 2) as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "offsets",
                u32_bytes((0..n_elems).map(|i| (i % n_elems) as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::random("src", n_elems, dt))
            .buffer(BenchBuffer::zeros("out", n_elems, dt).output())
            .constexpr("n_elems", n_elems as u32)
            .grid_1d(n_elems, 256)
            .bytes_moved((3 * n_elems * dt.size_bytes()) as u64)
    }
}
