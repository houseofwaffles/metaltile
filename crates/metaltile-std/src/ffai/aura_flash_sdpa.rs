//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA fused single-pass SDPA — online-softmax attention over an
//! AURA/TurboQuant-compressed K/V cache, with optional attention sinks
//! and sliding-window causal masking. Port of `turbo_flash_sdpa.h`
//! (spec 041 phase 1.1, GPT-OSS sink-attention family).
//!
//! Unlike the `aura_flash_p1` + `aura_flash_pass2` pair, this does the
//! whole attention in one dispatch — one threadgroup (a single
//! 32-lane simdgroup) per query, iterating every K/V token with a
//! running online softmax, then writing the normalized output. This
//! side-steps the pass2-with-sinks graph-fusion incoherence that the
//! two-pass β-with-sinks drafts hit on GPT-OSS-20B.
//!
//! Layout (matches `aura_flash_p1`):
//!   - q_rot:        [B*nQ, dim] f32   (WHT-rotated + pre-scaled by caller)
//!   - key_packed:   [B*nKV, tokens, key_packed_width]   u32
//!   - key_norms:    [B*nKV, tokens]   f32
//!   - key_codebook: [2^key_bits]      f32
//!   - val_packed:   [B*nKV, tokens, value_packed_width] u32
//!   - val_norms:    [B*nKV, tokens]   f32
//!   - val_codebook: [2^value_bits]    f32
//!   - sinks:        [num_q_heads]     f32  (per-head sink logit)
//!   - out:          [B*nQ, dim]       T    (rotated V space)
//!
//! `has_sinks` (0/1) and `window_size` (0 = full causal) are constexpr.
//! When `has_sinks == 1` the running softmax starts at `(m = sink,
//! l = 1)` — the sink behaves as a virtual key with value 0.
//!
//! Lane `program_id::<0>()` ∈ [0,32) owns dim slots `lane + i*32`;
//! `program_id::<1>()` = query index. The MLX reference fans tokens
//! across 32 simdgroups; this port keeps the simpler single-simdgroup
//! shape of `aura_flash_p1` (correctness-equivalent; token-parallelism
//! is a perf follow-up).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, B*nQ, 1]`, `tg = [32, 1, 1]` — exactly one
//!   simdgroup per query.
//! - `dims_per_lane = ceil(dim / 32)`.
//!
//! Codegen-only; correctness pinned by
//! `tests/aura_flash_sdpa_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

macro_rules! aura_flash_sdpa_kernel {
    (
        $name:ident,
        $key_bits:literal,
        $value_bits:literal,
        $key_levels:literal,
        $value_levels:literal,
        $dims_per_lane:literal,
        $subop:literal
    ) => {
        #[bench_kernel(op="aura", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)]
        #[kernel]
        pub fn $name<T>(
            q_rot: Tensor<f32>,
            key_packed: Tensor<u32>,
            key_norms: Tensor<f32>,
            key_codebook: Tensor<f32>,
            val_packed: Tensor<u32>,
            val_norms: Tensor<f32>,
            val_codebook: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] key_packed_width: u32,
            #[constexpr] value_packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let key_mask = (1u32 << $key_bits) - 1u32;
            let val_mask = (1u32 << $value_bits) - 1u32;

            // Codebook caches in per-thread stack arrays.
            stack_alloc("key_cb", $key_levels, "f32");
            for i in range(0u32, $key_levels, 1u32) {
                stack_store("key_cb", i, load(key_codebook[i]));
            }
            stack_alloc("val_cb", $value_levels, "f32");
            for i in range(0u32, $value_levels, 1u32) {
                stack_store("val_cb", i, load(val_codebook[i]));
            }

            // Per-lane slice of the rotated query, loaded once.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(q_rot[q_idx * dim + d]), 0.0f32);
                stack_store("q_vals", i, v);
            }

            // Online-softmax accumulators. With sinks, the running
            // softmax starts at (m = sink, l = 1): the sink is a virtual
            // key whose value is 0.
            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            // L=1 decode: the query attends K positions [0, tokens).
            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                // Sliding-window mask: keep key `t` when window is off,
                // or when `t` is within `window_size` of the last pos.
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    // Q · K in the compressed domain.
                    // NOTE: row stride is `kv_stride` (cache's `maxSeq`), not
                    // `tokens` (live KV-row count). For caches that aren't
                    // fully populated yet, head 1 starts at offset
                    // `kv_stride`, NOT `tokens` — otherwise we'd read head 0's
                    // tail bytes as if they were head 1's rows.
                    let k_packed_row = (kv_idx * kv_stride + t) * key_packed_width;
                    let k_norm = load(key_norms[kv_idx * kv_stride + t]);
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_offset = d * $key_bits;
                            let word_idx = bit_offset / 32u32;
                            let shift = bit_offset & 31u32;
                            let bits_in_w0 = 32u32 - shift;
                            let lo_bits = select(bits_in_w0 >= $key_bits, $key_bits, bits_in_w0);
                            let spill = $key_bits - lo_bits;
                            let w0 = load(key_packed[k_packed_row + word_idx]);
                            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                            let w1 = load(key_packed[k_packed_row + w1_idx]);
                            let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let value = (lo | hi) & key_mask;
                            let centroid = stack_load("key_cb", value);
                            let qv = stack_load("q_vals", i);
                            dot_partial = dot_partial + qv * centroid;
                        }
                    }
                    let score = simd_sum(dot_partial) * k_norm;

                    // Online-softmax max-shift.
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    // V-side update from compressed centroids.
                    // Same `kv_stride` row stride as the K side above.
                    let v_packed_row = (kv_idx * kv_stride + t) * value_packed_width;
                    let v_norm = load(val_norms[kv_idx * kv_stride + t]);
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_offset = d * $value_bits;
                            let word_idx = bit_offset / 32u32;
                            let shift = bit_offset & 31u32;
                            let bits_in_w0 = 32u32 - shift;
                            let lo_bits =
                                select(bits_in_w0 >= $value_bits, $value_bits, bits_in_w0);
                            let spill = $value_bits - lo_bits;
                            let w0 = load(val_packed[v_packed_row + word_idx]);
                            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                            let w1 = load(val_packed[v_packed_row + w1_idx]);
                            let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let value = (lo | hi) & val_mask;
                            let prev = stack_load("o", i);
                            let centroid = stack_load("val_cb", value);
                            let upd = prev * exp_diff + exp_score * centroid * v_norm;
                            stack_store("o", i, upd);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            // Normalize and write the final attention output.
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}

aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    "flash_sdpa_kb4_vb2_d128"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb4_d128,
    4u32,
    4u32,
    16u32,
    16u32,
    4u32,
    "flash_sdpa_kb4_vb4_d128"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    "flash_sdpa_kb4_vb2_d64"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb4_d64,
    4u32,
    4u32,
    16u32,
    16u32,
    2u32,
    "flash_sdpa_kb4_vb4_d64"
);
