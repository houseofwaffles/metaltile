//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Llama-style RoPE with optional Llama-3 frequency-band scaling.
//! Per-token decode form (single position constexpr), generic over T.
//!
//! Different from `mt_rope` (in `mlx/rope.rs`):
//!   - decode-only (no batch / seq grid)
//!   - generic dtype (mt_rope is f16-only)
//!   - supports Llama-3 wavelength banding (low / high / smoothed)
//!
//! For each (head, i in 0..head_dim/2):
//!
//!   base inv_freq = 1 / theta_base^(2i / head_dim)
//!   wavelen       = 2*pi / inv_freq
//!   if wavelen > low_freq_wavelen:        inv_freq /= scale_factor      (low-freq band)
//!   else if wavelen < high_freq_wavelen:  inv_freq                       (high-freq band)
//!   else (medium band):                   smoothed interpolation
//!
//! To turn scaling OFF, pass scale_factor=1, low_freq_factor=1,
//! high_freq_factor=1, original_max_position=very_large (e.g. 1e9).
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;

#[kernel(
    bench(
        op="rope",
        subop="rope_llama",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn ffai_rope_llama<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] scale_factor: f32,
    #[constexpr] low_freq_factor: f32,
    #[constexpr] high_freq_factor: f32,
    #[constexpr] original_max_position: f32,
) {
    let head = program_id::<0>();
    let i = program_id::<1>();
    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    let inv_freq_base = exp2(-i_f * log2(theta_base) / half_f);
    let two_pi = 6.283185307179586f32;
    let wavelen = two_pi / inv_freq_base;
    let low_freq_wavelen = original_max_position / low_freq_factor;
    let high_freq_wavelen = original_max_position / high_freq_factor;
    let scaled = inv_freq_base / scale_factor;
    let smooth_num = original_max_position / wavelen - low_freq_factor;
    let smooth_den = high_freq_factor - low_freq_factor;
    let s = smooth_num / smooth_den;
    let smoothed = (1.0f32 - s) * scaled + s * inv_freq_base;
    let is_low_freq = wavelen > low_freq_wavelen;
    let is_high_freq = wavelen < high_freq_wavelen;
    let inv_freq = select(is_low_freq, scaled, select(is_high_freq, inv_freq_base, smoothed));
    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
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

/// New-syntax correctness + bench for `ffai_rope_llama` (per-token decode RoPE
/// with Llama-3 banding). Grid3D, grid `[n_heads, half_dim, 1]`, tpg `[1,1,1]`
/// (one thread per (head, i); each writes the rotation pair i / i+half_dim).
/// Oracle replays the exact banded-inv_freq + rotation math in f32.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rope_llama;
    use crate::utils::{pack_f32, unpack_f32};

    /// Llama-3 banded inverse frequency for pair index `i` (mirrors the kernel).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn band_inv_freq(
        i: usize,
        half: usize,
        theta_base: f32,
        scale_factor: f32,
        low_ff: f32,
        high_ff: f32,
        orig_max: f32,
    ) -> f32 {
        let inv_base = (-(i as f32) * theta_base.log2() / half as f32).exp2();
        let two_pi = std::f32::consts::TAU;
        let wavelen = two_pi / inv_base;
        let low_wl = orig_max / low_ff;
        let high_wl = orig_max / high_ff;
        let scaled = inv_base / scale_factor;
        let s = (orig_max / wavelen - low_ff) / (high_ff - low_ff);
        let smoothed = (1.0 - s) * scaled + s * inv_base;
        if wavelen > low_wl {
            scaled
        } else if wavelen < high_wl {
            inv_base
        } else {
            smoothed
        }
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_llama(dt: DType) -> TestSetup {
        let (n_heads, head_dim) = (4usize, 64usize);
        let half = head_dim / 2;
        let (theta_base, position) = (500_000.0f32, 100u32);
        // Scaling OFF (Llama-3 banding disabled): high-freq branch selected.
        let (sf, lf, hf, omp) = (1.0f32, 1.0f32, 1.0f32, 1.0e9f32);
        let qk_f: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qk = unpack_f32(&pack_f32(&qk_f, dt), dt);
        let mut exp = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            for i in 0..half {
                let invf = band_inv_freq(i, half, theta_base, sf, lf, hf, omp);
                let th = position as f32 * invf;
                let (c, s) = (th.cos(), th.sin());
                let base = h * head_dim;
                let (x1, x2) = (qk[base + i], qk[base + i + half]);
                exp[base + i] = x1 * c - x2 * s;
                exp[base + i + half] = x1 * s + x2 * c;
            }
        }
        TestSetup::new(ffai_rope_llama::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk_f, dt), dt))
            .input(TestBuffer::zeros("out", n_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("position", position)
            .constexpr("theta_base", theta_base)
            .constexpr("scale_factor", sf)
            .constexpr("low_freq_factor", lf)
            .constexpr("high_freq_factor", hf)
            .constexpr("original_max_position", omp)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n_heads as u32, half as u32, 1, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_rope_llama` (decode RoPE, head_dim 128).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rope_llama;

    #[bench(name = "ffai/rope/rope_llama", dtypes = [f32, f16, bf16])]
    fn bench_rope_llama(dt: DType) -> BenchSetup {
        let (n_heads, head_dim) = (32usize, 128usize);
        let half = head_dim / 2;
        BenchSetup::new(ffai_rope_llama::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n_heads * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("position", 1000u32)
            .constexpr("theta_base", 500_000.0f32)
            .constexpr("scale_factor", 8.0f32)
            .constexpr("low_freq_factor", 1.0f32)
            .constexpr("high_freq_factor", 4.0f32)
            .constexpr("original_max_position", 8192.0f32)
            .grid_3d(n_heads as u32, half as u32, 1, [1, 1, 1])
            .bytes_moved((2 * n_heads * head_dim * dt.size_bytes()) as u64)
    }
}
