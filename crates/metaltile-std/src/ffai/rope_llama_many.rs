//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched Llama-style RoPE — applies position-dependent rotation across
//! T rows in ONE dispatch.
//!
//! Identical math / banding logic to `rope_llama` (Llama-3 wavelength
//! banding included) but the scalar `position` constexpr is replaced by
//! a per-row `positions: Tensor<u32>` of length T, and a new outer grid
//! axis selects the row index. For Qwen / Llama prefill the caller used
//! to T-loop over `ropePartialTwo` once per token (T dispatches × N
//! attention layers); this kernel collapses that to ONE dispatch per
//! layer per buffer.
//!
//! Layout:
//!
//!   qk        [T, n_heads, head_dim]              T (dtype)
//!   positions [T]                                 u32
//!   out       [T, n_heads, head_dim]              T (dtype)
//!
//! Grid3D: one thread per `(row, head, i)` where `i ∈ [0, half_dim)`.
//!   program_id::<0>() = row r ∈ [0, T)
//!   program_id::<1>() = head ∈ [0, n_heads)
//!   program_id::<2>() = i (half-rotary index)
//!
//! `row_stride` is the stride (in elements) between consecutive rows of
//! `qk` / `out`. For Q this is `n_heads_q * head_dim`; for K it is
//! `n_heads_k * head_dim`. The caller supplies the correct stride per
//! dispatch — this lets a single kernel handle both Q and K with their
//! own KV-grouped head counts.
//!
//! The paired `ropePartialTwo` semantics (rotate Q and K together with a
//! shared positions vector) are intentionally **not** fused into a single
//! kernel here — fusing would either bloat the kernel with an inner
//! `head < n_heads_q ? q_buf : k_buf` select on every thread or duplicate
//! the whole kernel body. Instead the host wraps two `_many` dispatches
//! on a shared command encoder (same PSO, two `setBuffer`s, two
//! `dispatchThreadgroups`) which is the same pipelining win as fused
//! encoders elsewhere in FFAI — see `feedback_ffai_cmd_buffer_pipelining`
//! in vault memory for the relevant pattern.
//!
//! Codegen-only. Correctness validated against `ffai_rope_llama` looped
//! per-row in `tests/rope_llama_many_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn ffai_rope_llama_many<T>(
    qk: Tensor<T>,
    positions: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] row_stride: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] scale_factor: f32,
    #[constexpr] low_freq_factor: f32,
    #[constexpr] high_freq_factor: f32,
    #[constexpr] original_max_position: f32,
) {
    let r = program_id::<0>();
    let head = program_id::<1>();
    let i = program_id::<2>();
    // Per-row position lookup. Each row gets its own scalar `position`
    // taken from the positions vector — this is the only piece of the
    // single-token kernel that becomes data-dependent.
    let position = load(positions[r]);
    // Banded inv_freq calculation — bit-identical to `rope_llama`.
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
    // Row-strided base offset. row_stride may be larger than n_heads *
    // head_dim if the caller is slicing a wider packed tensor (e.g. fused
    // QKV layout) — keep it a separate constexpr rather than recomputing
    // from n_heads so the kernel doesn't need to know n_heads at all.
    let base = r * row_stride + head * head_dim;
    let i1 = base + i;
    let i2 = base + i + half_dim;
    let x1 = load(qk[i1]).cast::<f32>();
    let x2 = load(qk[i2]).cast::<f32>();
    let o1 = x1 * cos_t - x2 * sin_t;
    let o2 = x1 * sin_t + x2 * cos_t;
    store(out[i1], o1.cast::<T>());
    store(out[i2], o2.cast::<T>());
}

/// New-syntax correctness + bench for `ffai_rope_llama_many` (batched per-row
/// Llama RoPE). Grid3D, grid `[T, n_heads, half_dim]`, tpg `[1,1,1]`. Reuses
/// the banded inv_freq from `rope_llama`; each row uses its own position.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rope_llama_many;
    use crate::{
        ffai::rope_llama::kernel_tests::band_inv_freq,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_llama_many(dt: DType) -> TestSetup {
        let (t_len, n_heads, head_dim) = (4usize, 4usize, 64usize);
        let half = head_dim / 2;
        let row_stride = n_heads * head_dim; // contiguous (no fused-QKV slack)
        let theta_base = 500_000.0f32;
        let (sf, lf, hf, omp) = (1.0f32, 1.0f32, 1.0f32, 1.0e9f32);
        let positions: Vec<u32> = (0..t_len).map(|r| (r * 10) as u32).collect();
        let qk_f: Vec<f32> =
            (0..t_len * row_stride).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qk = unpack_f32(&pack_f32(&qk_f, dt), dt);
        let mut exp = vec![0.0f32; t_len * row_stride];
        for (r, &posu) in positions.iter().enumerate() {
            let pos = posu as f32;
            for h in 0..n_heads {
                for i in 0..half {
                    let invf = band_inv_freq(i, half, theta_base, sf, lf, hf, omp);
                    let th = pos * invf;
                    let (c, s) = (th.cos(), th.sin());
                    let base = r * row_stride + h * head_dim;
                    let (x1, x2) = (qk[base + i], qk[base + i + half]);
                    exp[base + i] = x1 * c - x2 * s;
                    exp[base + i + half] = x1 * s + x2 * c;
                }
            }
        }
        TestSetup::new(ffai_rope_llama_many::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk_f, dt), dt))
            .input(TestBuffer::from_vec("positions", u32_bytes(&positions), DType::U32))
            .input(TestBuffer::zeros("out", t_len * row_stride, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("row_stride", row_stride as u32)
            .constexpr("theta_base", theta_base)
            .constexpr("scale_factor", sf)
            .constexpr("low_freq_factor", lf)
            .constexpr("high_freq_factor", hf)
            .constexpr("original_max_position", omp)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(t_len as u32, n_heads as u32, half as u32, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_rope_llama_many` (prefill batched RoPE).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rope_llama_many;

    #[bench(name = "ffai/rope/rope_llama_many", dtypes = [f32, f16, bf16])]
    fn bench_rope_llama_many(dt: DType) -> BenchSetup {
        let (t_len, n_heads, head_dim) = (512usize, 32usize, 128usize);
        let half = head_dim / 2;
        let row_stride = n_heads * head_dim;
        BenchSetup::new(ffai_rope_llama_many::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", t_len * row_stride, dt))
            .buffer(BenchBuffer::random("positions", t_len, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_len * row_stride, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("row_stride", row_stride as u32)
            .constexpr("theta_base", 500_000.0f32)
            .constexpr("scale_factor", 8.0f32)
            .constexpr("low_freq_factor", 1.0f32)
            .constexpr("high_freq_factor", 4.0f32)
            .constexpr("original_max_position", 8192.0f32)
            .with_shape_label(format!(
                "T{t_len} h{n_heads} d{head_dim} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(t_len as u32, n_heads as u32, half as u32, [1, 1, 1])
            .bytes_moved((2 * t_len * row_stride * dt.size_bytes()) as u64)
    }
}
