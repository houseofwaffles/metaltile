//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused RMSNorm + residual add — `out = residual + w * x * inv_rms`.
//!
//! Combines RMS normalization with the residual (skip-connection) add
//! in one dispatch. Saves a kernel launch at every post-attention and
//! post-FFN norm+residual site (≈3 calls/layer).
//!
//! Uses `mt_rms_inv_scalar` (from `mlx/rms_norm.rs`) via cross-kernel
//! call for the shared reduction phase: each thread computes its
//! `partial_ssq`, then calls `mt_rms_inv_scalar(partial_ssq, eps_buf, n)`
//! which inlines the `reduce_sum + rsqrt` body. The second phase applies
//! the residual add and stores the normalized+residual output.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel; the threadgroup geometry is part of its API.
//! Violating it silently miscomputes (best case) or freezes the GPU.
//!
//! - **`N = TPG * 4`.** Each thread owns 4 consecutive elements of the
//!   row; the wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full simdgroup) and
//!   **`TPG ≤ 1024`**. Combined: `n` multiple of 128, `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row** — `program_id::<0>()` = row index.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_residual_gpu_correctness.rs`.

use metaltile::kernel;

/// `out[r, i] = residual[r, i] + w[i] * x[r, i] * rsqrt(mean(x[r]²) + eps)`.
#[kernel]
pub fn ffai_rms_norm_residual<T>(
    x: Tensor<T>,
    residual: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // Each thread owns 4 consecutive elements (N = TPG * 4). OOB lanes
    // re-read row[0..3] (benign — their SSQ contribution is masked to 0)
    // and skip their stores, mirroring `mt_rms_norm`'s freeze-safe guard.
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    let x0 = load(x[safe_base]).cast::<f32>();
    let x1 = load(x[safe_base + 1u32]).cast::<f32>();
    let x2 = load(x[safe_base + 2u32]).cast::<f32>();
    let x3 = load(x[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    // Cross-kernel call: KernelInlinePass splices mt_rms_inv_scalar's body
    // here. partial_ssq is a Value arg (pre-computed f32 scalar, no load);
    // eps_buf and n are Tensor args (renamed in callee's loads transparently).
    let rms = mt_rms_inv_scalar(partial_ssq, eps_buf, n);
    if in_bounds {
        let o0 = load(residual[base]).cast::<f32>() + x0 * rms * load(w[col]).cast::<f32>();
        let o1 = load(residual[base + 1u32]).cast::<f32>()
            + x1 * rms * load(w[col + 1u32]).cast::<f32>();
        let o2 = load(residual[base + 2u32]).cast::<f32>()
            + x2 * rms * load(w[col + 2u32]).cast::<f32>();
        let o3 = load(residual[base + 3u32]).cast::<f32>()
            + x3 * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], o0.cast::<T>());
        store(out[base + 1u32], o1.cast::<T>());
        store(out[base + 2u32], o2.cast::<T>());
        store(out[base + 3u32], o3.cast::<T>());
    }
}

/// New-syntax correctness for `ffai_rms_norm_residual` (Reduction mode, one
/// threadgroup per row, `tpg = n/4` — `n` a multiple of 128, `n ≤ 4096`).
/// Per-row oracle on dtype-rounded inputs: normalize `x` then add `residual`:
/// `out_i = residual_i + x_i / sqrt(mean(x²) + eps) * w_i`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rms_norm_residual;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let mut x = Vec::with_capacity(rows * n);
        let mut residual = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let row: Vec<f32> =
                (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.03).collect();
            let res: Vec<f32> =
                (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.07 + r as f32 * 0.02).collect();
            let xr = unpack_f32(&pack_f32(&row, dt), dt);
            let resr = unpack_f32(&pack_f32(&res, dt), dt);
            let ms: f32 = xr.iter().map(|&v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            expected.extend(
                xr.iter().zip(&w_dt).zip(&resr).map(|((&xi, &wi), &ri)| ri + xi * inv * wi),
            );
            x.extend_from_slice(&row);
            residual.extend_from_slice(&res);
        }
        TestSetup::new(ffai_rms_norm_residual::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("residual", pack_f32(&residual, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_ffai_rms_norm_residual(dt: DType) -> TestSetup { setup(4, 512, dt) }
}

/// New-syntax benchmark for `ffai_rms_norm_residual` (fused RMSNorm + residual
/// add, hidden axis n=4096, tpg=1024 — the Apple TPG cap).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rms_norm_residual;

    #[bench(name = "ffai/rms_norm_residual/rms_norm_residual", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_residual(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(ffai_rms_norm_residual::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * n, dt))
            .buffer(BenchBuffer::random("residual", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            // x + residual read, out write.
            .bytes_moved((3 * rows * n * dt.size_bytes()) as u64)
    }
}
