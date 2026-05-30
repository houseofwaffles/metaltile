//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile::kernel;

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

/// New-syntax correctness + bench for `ffai_rope_yarn` (YaRN context-extension
/// RoPE). Grid3D, grid `[n_heads, half_dim, 1]`, tpg `[1,1,1]`. Oracle replays
/// the extrapolation/interpolation ramp + attn_factor scaling in f32.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rope_yarn;
    use crate::utils::{pack_f32, unpack_f32};

    #[allow(clippy::too_many_arguments)]
    fn rope_yarn_setup(
        dt: DType,
        position: u32,
        factor: f32,
        low: f32,
        high: f32,
        attn: f32,
    ) -> TestSetup {
        let (n_heads, head_dim) = (4usize, 64usize);
        let half = head_dim / 2;
        let theta_base = 10_000.0f32;
        let qk_f: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qk = unpack_f32(&pack_f32(&qk_f, dt), dt);
        let mut exp = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            for i in 0..half {
                let extrap = (-(i as f32) * theta_base.log2() / half as f32).exp2();
                let interp = extrap / factor;
                let t = (i as f32 - low) / (high - low);
                let ramp = t.clamp(0.0, 1.0);
                let invf = interp * ramp + extrap * (1.0 - ramp);
                let th = position as f32 * invf;
                let (c, s) = (th.cos() * attn, th.sin() * attn);
                let base = h * head_dim;
                let (x1, x2) = (qk[base + i], qk[base + i + half]);
                exp[base + i] = x1 * c - x2 * s;
                exp[base + i + half] = x1 * s + x2 * c;
            }
        }
        TestSetup::new(ffai_rope_yarn::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk_f, dt), dt))
            .input(TestBuffer::zeros("out", n_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("position", position)
            .constexpr("theta_base", theta_base)
            .constexpr("factor", factor)
            .constexpr("low", low)
            .constexpr("high", high)
            .constexpr("attn_factor", attn)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n_heads as u32, half as u32, 1, [1, 1, 1])
    }

    // Moderate YaRN: factor 4, attn_factor 1 (the scaling multiply is a no-op).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_yarn(dt: DType) -> TestSetup { rope_yarn_setup(dt, 100, 4.0, 16.0, 24.0, 1.0) }

    // Nemotron-style aggressive YaRN: factor 16 (strong interp band) with a
    // non-unit attn_factor (mscale) — exercises both the deeper extrap/interp
    // ramp and the `cos/sin * attn_factor` scaling the attn=1 config leaves as
    // a no-op. The ramp is selected by the frequency index, not the position,
    // so a small position keeps the cos/sin precision tight.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_yarn_nemotron(dt: DType) -> TestSetup {
        rope_yarn_setup(dt, 100, 16.0, 20.0, 37.0, 1.13)
    }
}

/// New-syntax benchmark for `ffai_rope_yarn`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rope_yarn;

    #[bench(name = "ffai/rope/rope_yarn", dtypes = [f32, f16, bf16])]
    fn bench_rope_yarn(dt: DType) -> BenchSetup {
        let (n_heads, head_dim) = (32usize, 128usize);
        let half = head_dim / 2;
        BenchSetup::new(ffai_rope_yarn::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n_heads * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("position", 1000u32)
            .constexpr("theta_base", 10_000.0f32)
            .constexpr("factor", 4.0f32)
            .constexpr("low", 16.0f32)
            .constexpr("high", 48.0f32)
            .constexpr("attn_factor", 1.0f32)
            .grid_3d(n_heads as u32, half as u32, 1, [1, 1, 1])
            .bytes_moved((2 * n_heads * head_dim * dt.size_bytes()) as u64)
    }
}
