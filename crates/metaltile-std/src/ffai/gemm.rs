//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-row GEMM — `out[r, :] = weight · input[r, :]` for a block of
//! `n_rows` rows in one dispatch. Generic over T.
//!
//! Used by Nemotron-Labs-Diffusion's block-diffusion / self-speculation
//! `forwardTokens`: a 32-token block runs 7 projections per layer
//! (q/k/v/o/gate/up/down). Done as N single-row `gemv`s the weight is
//! re-streamed once per row — N× the weight bandwidth. This kernel
//! tiles the output into 32×32 blocks and stages a `[32, 16]` weight
//! tile + `[32, 16]` input tile in threadgroup memory, so the weight
//! is read once and reused across all 32 rows of the tile.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel (threadgroup memory + barriers). No
//! `simd_*`-strided walk, so it is not exposed to the n_simd==0 freeze
//! — but it still has a fixed geometry the wrapper must honour:
//!
//! - **TPG = 1024 threads** (BM·BN = 32·32). The 1024 threads
//!   cooperatively load the two tiles (512 weight + 512 input
//!   elements) and then each computes one output element.
//! - **Grid: (out_dim/32) × (n_rows rounded up to /32) threadgroups**,
//!   2-D — `tgid_x` = output-column tile, `tgid_y` = row tile.
//! - **`in_dim % 16 == 0`** — the K loop strides by the 16-wide tile
//!   with no remainder handling.
//! - `weight` is `[out_dim, in_dim]`, `input` is `[n_rows, in_dim]`,
//!   `out` is `[n_rows, out_dim]`, all row-major.
//!
//! Output / row-count edges (`out_dim`, `n_rows` not multiples of 32)
//! are handled in-kernel: out-of-range loads clamp to index 0 and
//! contribute 0, out-of-range stores are skipped.

use metaltile::kernel;

#[kernel]
pub fn ffai_gemm<T>(
    weight: Tensor<T>,
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] n_rows: u32,
) {
    // 32×32 output tile, 16-wide K tile. 1024 threads, one per output.
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32; // output row within the tile (0..31)
    let lo = tid % 32u32; // output col within the tile (0..31)
    // Weight tile [BN=32][BK=16] + input tile [BM=32][BK=16].
    threadgroup_alloc("gemm_w", 512);
    threadgroup_alloc("gemm_x", 512);
    let mut acc = 0.0f32;
    for k0 in range(0u32, in_dim, 16u32) {
        // Cooperative load: threads 0..511 fill the weight tile, threads
        // 512..1023 fill the input tile — one element each.
        if tid < 512u32 {
            let s = tid;
            let w_col = tgid_x * 32u32 + s / 16u32;
            let w_valid = w_col < out_dim;
            let w_col_safe = select(w_valid, w_col, 0u32);
            let w_raw = load(weight[w_col_safe * in_dim + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gemm_w", s, select(w_valid, w_raw, 0.0f32));
        }
        if tid >= 512u32 {
            let s = tid - 512u32;
            let x_row = tgid_y * 32u32 + s / 16u32;
            let x_valid = x_row < n_rows;
            let x_row_safe = select(x_valid, x_row, 0u32);
            let x_raw = load(input[x_row_safe * in_dim + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gemm_x", s, select(x_valid, x_raw, 0.0f32));
        }
        threadgroup_barrier();
        // Each thread accumulates its output element from the tiles.
        for k in range(0u32, 16u32, 1u32) {
            let w = threadgroup_load("gemm_w", lo * 16u32 + k);
            let x = threadgroup_load("gemm_x", lr * 16u32 + k);
            acc = acc + w * x;
        }
        threadgroup_barrier();
    }
    let r = tgid_y * 32u32 + lr;
    let o = tgid_x * 32u32 + lo;
    if r < n_rows {
        if o < out_dim {
            store(out[r * out_dim + o], acc.cast::<T>());
        }
    }
}

/// New-syntax correctness tests for `ffai_gemm` — the multi-row 32×32-tiled
/// GEMM `out[r, :] = weight · input[r, :]`. Reduction-mode (threadgroup-memory
/// tiles + barriers).
///
/// Oracle: straight triple-loop `out[r, o] = Σ_k weight[o, k]·input[r, k]` in
/// f32. Inputs are dtype-rounded so the oracle matches the kernel's load-cast.
/// Covers an aligned shape (dims multiples of 32) and an edge shape (n_rows /
/// out_dim NOT multiples of 32 — exercises the in-kernel load-clamp + store-skip);
/// in_dim stays a multiple of 16 (the K-tile contract).
///
/// Grid: `grid_3d((out_dim+31)/32, (n_rows+31)/32, 1, [1024, 1, 1])` — one TG per
/// 32×32 output tile, 1024 threads (BM·BN) cooperatively load tiles + one output
/// each.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gemm;
    use crate::utils::{pack_f32, unpack_f32};

    /// Triple-loop reference: out[r, o] = Σ_k weight[o, k] · input[r, k].
    fn gemm_oracle(
        weight: &[f32],
        input: &[f32],
        n_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_rows * out_dim];
        for r in 0..n_rows {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += weight[o * in_dim + k] * input[r * in_dim + k];
                }
                out[r * out_dim + o] = acc;
            }
        }
        out
    }

    fn gemm_setup(n_rows: usize, in_dim: usize, out_dim: usize, dt: DType) -> TestSetup {
        let weight_f: Vec<f32> =
            (0..out_dim * in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let input_f: Vec<f32> =
            (0..n_rows * in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
        let w = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = gemm_oracle(&w, &x, n_rows, in_dim, out_dim);
        TestSetup::new(ffai_gemm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_rows * out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("n_rows", n_rows as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(out_dim.div_ceil(32) as u32, n_rows.div_ceil(32) as u32, 1, [1024, 1, 1])
    }

    // Aligned: n_rows / out_dim multiples of 32, in_dim a multiple of 16.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_gemm_aligned(dt: DType) -> TestSetup { gemm_setup(32, 64, 64, dt) }

    // Edge: n_rows / out_dim NOT multiples of 32 (load-clamp + store-skip).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_gemm_edge(dt: DType) -> TestSetup { gemm_setup(20, 48, 100, dt) }
}

/// New-syntax benchmark for `ffai_gemm`. Nemotron-class block-diffusion shape:
/// a 32-row block projected through a `[out_dim, in_dim]` weight (hidden 4096).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gemm;

    #[bench(name = "ffai/gemm/gemm", dtypes = [f32, f16, bf16])]
    fn bench_gemm(dt: DType) -> BenchSetup {
        let (n_rows, in_dim, out_dim) = (32usize, 4096usize, 4096usize);
        let sz = dt.size_bytes();
        let bytes = out_dim * in_dim * sz + n_rows * in_dim * sz + n_rows * out_dim * sz;
        BenchSetup::new(ffai_gemm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", out_dim * in_dim, dt))
            .buffer(BenchBuffer::random("input", n_rows * in_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("n_rows", n_rows as u32)
            .grid_3d(out_dim.div_ceil(32) as u32, n_rows.div_ceil(32) as u32, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
