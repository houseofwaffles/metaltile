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

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rope",
    subop="rope_llama_many",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
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
