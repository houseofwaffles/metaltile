//! YaRN RoPE — per-token decode form, generic over T.
//!
//! YaRN ("Yet another RoPE extensioN") rescales the rotary frequencies
//! to extend a model's usable context. Per dimension it blends between
//! **extrapolation** (the original frequency — kept for high-frequency
//! dimensions) and **interpolation** (the frequency divided by
//! `factor` — applied to low-frequency dimensions), with a linear ramp
//! across a `[low, high]` correction band:
//!
//!   inv_freq_extrap = theta_base^(-2i/head_dim)
//!   inv_freq_interp = inv_freq_extrap / factor
//!   ramp            = clamp((i - low) / (high - low), 0, 1)
//!   inv_freq        = inv_freq_interp*ramp + inv_freq_extrap*(1 - ramp)
//!
//! `low` / `high` are the YaRN correction-range bounds. They derive
//! from `beta_fast` / `beta_slow` via a `floor`/`ceil`/`ln` computation
//! that is constant across the whole dispatch, so the caller computes
//! them once and passes them as constexpr (see `Ops.ropeYaRN`).
//! `attn_factor` is YaRN's mscale attention scaling — `1.0` when the
//! checkpoint's `mscale == mscale_all_dim` (the common case, including
//! Nemotron-Labs-Diffusion).
//!
//! Same Grid3D dispatch shape as `ffai_rope_llama`: one thread per
//! (head, i in 0..head_dim/2), each thread rotating the pair
//! (i, i + half_dim). No reduction, no threadgroup memory.
//!
//! Codegen-only. Validated by `rope_yarn_gpu_correctness` + FFAI
//! integration tests.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rope",
    subop="rope_yarn",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn ffai_rope_yarn<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] factor: f32,
    #[constexpr] low: f32,
    #[constexpr] high: f32,
    #[constexpr] attn_factor: f32,
) {
    let head = program_id::<0>();
    let i = program_id::<1>();

    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();

    // Base (extrapolation) frequency — identical to plain RoPE.
    let inv_freq_extrap = exp2(-i_f * log2(theta_base) / half_f);
    // Interpolation frequency — extended context by `factor`.
    let inv_freq_interp = inv_freq_extrap / factor;

    // Linear ramp over the [low, high] correction band, clamped to
    // [0, 1]. ramp=0 → pure extrapolation; ramp=1 → pure interpolation.
    // The caller guarantees high > low, so the divide is safe.
    let t = (i_f - low) / (high - low);
    let ramp = select(t < 0.0f32, 0.0f32, select(t > 1.0f32, 1.0f32, t));

    let inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0f32 - ramp);

    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta) * attn_factor;
    let sin_t = sin(theta) * attn_factor;

    let base = head * head_dim;
    let i1 = base + i;
    let i2 = base + i + half_dim;

    let x1 = load(qk[i1]).cast::<f32>();
    let x2 = load(qk[i2]).cast::<f32>();
    let o1 = x1 * cos_t - x2 * sin_t;
    let o2 = x1 * sin_t + x2 * cos_t;

    store(out[i1], o1.cast::<T>());
    store(out[i2], o2.cast::<T>());
}
