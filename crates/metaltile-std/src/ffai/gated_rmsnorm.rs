//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused gated RMSNorm — `out = rmsNorm(y) · silu(z)`.
//!
//! The post-step of a Gated-DeltaNet (GDN) layer. After the GDN
//! recurrence (`mt_gated_delta_step` / `_chunk`) produces the linear-
//! attention output `y`, Qwen3.5 / Qwen3.6 apply a *gated* RMSNorm:
//!
//! ```text
//!   out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])
//! ```
//!
//! The distinguishing feature versus the plain `mt_rms_norm` is the
//! **dtype split**: `y` arrives as **fp32** — the GDN recurrence
//! accumulates its state in fp32 and emits `y` in fp32 (a bf16 `y`
//! drifts after a few dozen decode steps, the same reason
//! `gated_delta` / `ssm_step` keep an fp32 accumulator). The gate `z`,
//! the weight `w`, and the output are in the model's activation dtype
//! `T`. No existing GPU norm consumes an fp32 row and writes a `T`
//! row, so without this kernel the GDN post-step runs host-side — one
//! CPU↔GPU sync per GDN layer (≈75 % of Qwen3.5/3.6 layers).
//!
//! `silu(x) = x · sigmoid(x)` is computed in fp32 from the `z` gate
//! (cast up from `T`); the normalized-and-gated result is rounded to
//! `T` at the store.
//!
//! Algorithm-identical reduction to `mlx/rms_norm.rs`'s `mt_rms_norm`
//! — f32 sum-of-squares accumulator, threadgroup-wide `reduce_sum`,
//! `rsqrt(ssq/n + eps)` scaling — with the fp32 `y` input and the
//! extra `silu(z)` gate multiply.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel; the threadgroup geometry is part of its API.
//! Violating it silently miscomputes (best case) or freezes the GPU
//! (worst case — see `docs/developing.md`).
//!
//! - **`N = TPG * 4`.** Each thread owns 4 consecutive elements of the
//!   row; the wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full Apple simdgroup) and
//!   **`TPG ≤ 1024`**. Combined: `n` a multiple of 128, `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row** — `program_id::<0>()` = row index.
//!   Multi-row dispatch uses `grid = (nRows * TPG, 1, 1)`,
//!   `tg = (TPG, 1, 1)`.
//!
//! Codegen-only; correctness pinned by
//! `tests/gated_rmsnorm_gpu_correctness.rs`.

use metaltile::kernel;

/// `out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])`.
///
/// `y` is fp32 (the GDN recurrence output); `z`, `w`, `out` are `T`.
#[kernel(
    bench(
        op="gated_rmsnorm",
        subop="gated_rmsnorm",
        class=GenericEmpty,
        tol=1e-4,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_gated_rmsnorm<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
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
    // `y` is already fp32. The explicit `.cast::<f32>()` is a no-op
    // numerically but forces codegen to bind a *named* scalar for each
    // element — without it the float4-load vectorizer collapses the
    // element names and the post-reduction store references an
    // undeclared identifier (the names must survive across the
    // threadgroup `reduce_sum`).
    let y0 = load(y[safe_base]).cast::<f32>();
    let y1 = load(y[safe_base + 1u32]).cast::<f32>();
    let y2 = load(y[safe_base + 2u32]).cast::<f32>();
    let y3 = load(y[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        // silu(x) = x / (1 + exp(-x)) — inlined in fp32 (same form as
        // mt_swiglu) to keep the gate precise before the round to T.
        let z0 = load(z[base]).cast::<f32>();
        let z1 = load(z[base + 1u32]).cast::<f32>();
        let z2 = load(z[base + 2u32]).cast::<f32>();
        let z3 = load(z[base + 3u32]).cast::<f32>();
        let g0 = z0 / (1.0f32 + exp(0.0f32 - z0));
        let g1 = z1 / (1.0f32 + exp(0.0f32 - z1));
        let g2 = z2 / (1.0f32 + exp(0.0f32 - z2));
        let g3 = z3 / (1.0f32 + exp(0.0f32 - z3));
        let o0 = y0 * rms * load(w[col]).cast::<f32>() * g0;
        let o1 = y1 * rms * load(w[col + 1u32]).cast::<f32>() * g1;
        let o2 = y2 * rms * load(w[col + 2u32]).cast::<f32>() * g2;
        let o3 = y3 * rms * load(w[col + 3u32]).cast::<f32>() * g3;
        store(out[base], o0.cast::<T>());
        store(out[base + 1u32], o1.cast::<T>());
        store(out[base + 2u32], o2.cast::<T>());
        store(out[base + 3u32], o3.cast::<T>());
    }
}

/// New-syntax correctness for `ffai_gated_rmsnorm` (Reduction mode, one
/// threadgroup per row, `tpg = n/4` — `n` a multiple of 128, `n ≤ 4096`).
/// fp32-in / `T`-out split: `y` is always packed f32; `z` / `w` / `out` use
/// `dt`. Per-row oracle: `out_i = w_i · y_i · rsqrt(mean(y²)+eps) · silu(z_i)`,
/// with `silu(z) = z / (1 + exp(-z))`, on dtype-rounded `z` / `w`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gated_rmsnorm;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        // z and w are in the model dtype — round them; y stays full fp32.
        let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let mut y = Vec::with_capacity(rows * n);
        let mut z = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let yr: Vec<f32> =
                (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.03).collect();
            let zr_raw: Vec<f32> =
                (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.07 + r as f32 * 0.02).collect();
            let zr = unpack_f32(&pack_f32(&zr_raw, dt), dt);
            let ms: f32 = yr.iter().map(|&v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..n {
                let silu = zr[i] / (1.0 + (-zr[i]).exp());
                expected.push(yr[i] * inv * w_dt[i] * silu);
            }
            y.extend_from_slice(&yr);
            z.extend_from_slice(&zr_raw);
        }
        TestSetup::new(ffai_gated_rmsnorm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            // `y` is fp32 regardless of T.
            .input(TestBuffer::from_vec("y", pack_f32(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 8e-2])]
    fn test_ffai_gated_rmsnorm(dt: DType) -> TestSetup { setup(4, 512, dt) }
}

/// New-syntax benchmark for `ffai_gated_rmsnorm` (fused GDN post-step, fp32 `y`,
/// `T` gate / weight / output, n=4096, tpg=1024 — the Apple TPG cap).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gated_rmsnorm;

    #[bench(name = "ffai/gated_rmsnorm/gated_rmsnorm", dtypes = [f32, f16, bf16])]
    fn bench_gated_rmsnorm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(ffai_gated_rmsnorm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            // `y` is always fp32 (the GDN recurrence output).
            .buffer(BenchBuffer::random("y", rows * n, DType::F32))
            .buffer(BenchBuffer::random("z", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            // y (f32) + z (T) read, out (T) write.
            .bytes_moved((rows * n * DType::F32.size_bytes() + 2 * rows * n * dt.size_bytes()) as u64)
    }
}
