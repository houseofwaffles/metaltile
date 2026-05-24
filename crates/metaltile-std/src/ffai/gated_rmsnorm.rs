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

use metaltile::{bench_kernel, kernel};

/// `out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])`.
///
/// `y` is fp32 (the GDN recurrence output); `z`, `w`, `out` are `T`.
#[bench_kernel(
    op="gated_rmsnorm",
    subop="gated_rmsnorm",
    class=GenericEmpty,
    tol=1e-4,
    kernel_mode=Reduction,
)]
#[kernel]
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
