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
//! Layout (matches `aura_flash_p1` / `aura_score` / `aura_value` — all
//! generic over `T` for the auxiliary float buffers; internal math stays
//! f32 via cast-at-load. See header note on the dtype unification):
//!   - q_rot:        [B*nQ, dim] T     (WHT-rotated + pre-scaled by caller)
//!   - key_packed:   [B*nKV, tokens, key_packed_width]   u32
//!   - key_norms:    [B*nKV, tokens]   T
//!   - key_codebook: [2^key_bits]      T
//!   - val_packed:   [B*nKV, tokens, value_packed_width] u32
//!   - val_norms:    [B*nKV, tokens]   T
//!   - val_codebook: [2^value_bits]    T
//!   - sinks:        [num_q_heads]     T    (per-head sink logit)
//!   - out:          [B*nQ, dim]       T    (rotated V space)
//!
//! Norms / codebook were f32-typed in the original port (spec 041
//! phase 1.1), inherited from `mlx-swift-lm`'s TurboQuant codec. The
//! sibling `aura_flash_p1` / `aura_flash_pass2` / `aura_score` /
//! `aura_value` kernels were genericised to `Tensor<T>` during the bf16
//! coverage rollout; this kernel was the last laggard. Unifying the
//! dtype contract lets FFAI's cache store norms+codebook in the
//! activation dtype directly — no per-call cast on the decode hot path.
//! Internal arithmetic still runs in f32 (cast-at-load), matching the
//! precision of the f32 era; the storage narrowing follows the C++
//! `llama.cpp` TQ+ fork's production pattern (fp16-stored norms, f32-at-
//! use, zero PPL impact measured).
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

use metaltile::kernel;

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
        #[kernel]
        pub fn $name<T>(
            q_rot: Tensor<T>,
            key_packed: Tensor<u32>,
            key_norms: Tensor<T>,
            key_codebook: Tensor<T>,
            val_packed: Tensor<u32>,
            val_norms: Tensor<T>,
            val_codebook: Tensor<T>,
            sinks: Tensor<T>,
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

            // Codebook caches in per-thread stack arrays. Loads cast to
            // f32 so the rest of the kernel stays bit-identical to the
            // f32-typed era — only the storage narrowing differs.
            stack_alloc("key_cb", $key_levels, "f32");
            for i in range(0u32, $key_levels, 1u32) {
                stack_store("key_cb", i, load(key_codebook[i]).cast::<f32>());
            }
            stack_alloc("val_cb", $value_levels, "f32");
            for i in range(0u32, $value_levels, 1u32) {
                stack_store("val_cb", i, load(val_codebook[i]).cast::<f32>());
            }

            // Per-lane slice of the rotated query, loaded once.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(q_rot[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v);
            }

            // Online-softmax accumulators. With sinks, the running
            // softmax starts at (m = sink, l = 1): the sink is a virtual
            // key whose value is 0.
            let sink_val = load(sinks[q_idx % num_q_heads]).cast::<f32>();
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
                    let k_norm = load(key_norms[kv_idx * kv_stride + t]).cast::<f32>();
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
                    let v_norm = load(val_norms[kv_idx * kv_stride + t]).cast::<f32>();
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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_flash_sdpa_kb4_vb4_d128;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Bit-pack per-token codebook indices into the spill-aware layout the
    /// kernel unpacks (`bit = d*bits`, possibly crossing a u32 boundary).
    fn pack_int_indices(
        indices: &[u32],
        kv_heads: usize,
        tokens: usize,
        dim: usize,
        bits: usize,
    ) -> Vec<u32> {
        let mask = (1u32 << bits) - 1;
        let pw = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; kv_heads * tokens * pw];
        for kvh in 0..kv_heads {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(kvh * tokens + t) * dim + d] & mask;
                    let bit = d * bits;
                    let word = bit / 32;
                    let shift = bit & 31;
                    packed[(kvh * tokens + t) * pw + word] |= idx << shift;
                    let spill = (shift + bits) as i32 - 32;
                    if spill > 0 {
                        packed[(kvh * tokens + t) * pw + word + 1] |=
                            idx >> (bits as u32 - spill as u32);
                    }
                }
            }
        }
        packed
    }

    /// Dense softmax-attention over the codebook-DECODED K,V. The fused
    /// single-pass AURA flash decode (`kv_stride == tokens`, full
    /// attention, no sinks) must reproduce this. K/V are reconstructed as
    /// `codebook[index] * norm` per token.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        q_rot: &[f32],
        key_idx: &[u32],
        val_idx: &[u32],
        key_norms: &[f32],
        val_norms: &[f32],
        key_cb: &[f32],
        val_cb: &[f32],
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dim: usize,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let mut out = vec![0.0_f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            let mut scores = vec![0.0_f32; tokens];
            for (t, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    let q = key_idx[(kvh * tokens + t) * dim + d];
                    dot += q_rot[qh * dim + d] * key_cb[q as usize];
                }
                *s = dot * key_norms[kvh * tokens + t];
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_w = 0.0_f32;
            let mut acc = vec![0.0_f32; dim];
            for (t, s) in scores.iter().enumerate() {
                let w = (s - m).exp();
                sum_w += w;
                for (d, a) in acc.iter_mut().enumerate() {
                    let v = val_idx[(kvh * tokens + t) * dim + d];
                    *a += w * val_cb[v as usize] * val_norms[kvh * tokens + t];
                }
            }
            let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
            for d in 0..dim {
                out[qh * dim + d] = acc[d] * inv;
            }
        }
        out
    }

    // Representative variant: kb4_vb4_d128 (4-bit key + 4-bit value
    // codebooks, head_dim 128), full attention, no sinks. Codebook
    // decode loosens the half-precision tolerances vs. dense SDPA.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
    fn test_ffai_aura_flash_sdpa_kb4_vb4_d128(dt: DType) -> TestSetup {
        let (q_heads, kv_heads, tokens, dim) = (2usize, 1usize, 8usize, 128usize);
        let (key_bits, value_bits) = (4usize, 4usize);
        let repeat = q_heads / kv_heads;
        let kpw = (dim * key_bits).div_ceil(32);
        let vpw = (dim * value_bits).div_ceil(32);

        // 16-level codebooks for both 4-bit key and 4-bit value.
        let key_cb: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let val_cb: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let key_idx: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
        let val_idx: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 5) % 16) as u32).collect();
        let key_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect();
        let val_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect();
        let q_rot: Vec<f32> =
            (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();
        let sinks = vec![0.0f32; q_heads];

        let key_packed = pack_int_indices(&key_idx, kv_heads, tokens, dim, key_bits);
        let val_packed = pack_int_indices(&val_idx, kv_heads, tokens, dim, value_bits);

        // All float-typed buffers (q_rot / norms / codebook / sinks / out) are
        // `Tensor<T>` now (#212), packed in `dt`. Round them through `dt` so
        // the oracle sees the same cast-at-load values the kernel does.
        let q_rot_r = unpack_f32(&pack_f32(&q_rot, dt), dt);
        let key_norms_r = unpack_f32(&pack_f32(&key_norms, dt), dt);
        let val_norms_r = unpack_f32(&pack_f32(&val_norms, dt), dt);
        let key_cb_r = unpack_f32(&pack_f32(&key_cb, dt), dt);
        let val_cb_r = unpack_f32(&pack_f32(&val_cb, dt), dt);

        let expected = naive(
            &q_rot_r,
            &key_idx,
            &val_idx,
            &key_norms_r,
            &val_norms_r,
            &key_cb_r,
            &val_cb_r,
            q_heads,
            kv_heads,
            tokens,
            dim,
        );

        TestSetup::new(aura_flash_sdpa_kb4_vb4_d128::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("q_rot", pack_f32(&q_rot, dt), dt))
            .input(TestBuffer::from_vec("key_packed", u32_bytes(&key_packed), DType::U32))
            .input(TestBuffer::from_vec("key_norms", pack_f32(&key_norms, dt), dt))
            .input(TestBuffer::from_vec("key_codebook", pack_f32(&key_cb, dt), dt))
            .input(TestBuffer::from_vec("val_packed", u32_bytes(&val_packed), DType::U32))
            .input(TestBuffer::from_vec("val_norms", pack_f32(&val_norms, dt), dt))
            .input(TestBuffer::from_vec("val_codebook", pack_f32(&val_cb, dt), dt))
            .input(TestBuffer::from_vec("sinks", pack_f32(&sinks, dt), dt))
            .input(TestBuffer::zeros("out", q_heads * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("key_packed_width", kpw as u32)
            .constexpr("value_packed_width", vpw as u32)
            .constexpr("tokens", tokens as u32)
            // Fully-populated fixture: stride == live row count.
            .constexpr("kv_stride", tokens as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("num_q_heads", q_heads as u32)
            .constexpr("has_sinks", 0u32)
            .constexpr("window_size", 0u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(1, q_heads as u32, 1, [32, 1, 1])
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    // Decode-class shape: 32 Q heads, GQA fan-out 4, 512-token cache.
    const Q_HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const TOKENS: usize = 512;

    // `key_bits`/`value_bits` set the packed-width and codebook (level)
    // sizes: packed_width = dim / (32 / bits); levels = 2^bits.
    fn setup(
        ir: metaltile::core::ir::Kernel,
        dim: usize,
        key_bits: usize,
        value_bits: usize,
        dt: DType,
    ) -> BenchSetup {
        let key_pw = dim / (32 / key_bits);
        let val_pw = dim / (32 / value_bits);
        let key_levels = 1usize << key_bits;
        let value_levels = 1usize << value_bits;
        let repeat = Q_HEADS / KV_HEADS;
        let kv_stride = TOKENS;
        let kv_rows = KV_HEADS * kv_stride;
        // q_rot + norms now pack in `dt` (T-typed per #212); packed K/V stay
        // u32. Norms are the dominant aux float traffic (2 rows per token).
        let dt_b = dt.size_bytes();
        let bytes = Q_HEADS * dim * dt_b
            + kv_rows * key_pw * 4
            + kv_rows * val_pw * 4
            + kv_rows * 2 * dt_b
            + Q_HEADS * dim * dt_b;
        BenchSetup::new(ir)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("q_rot", Q_HEADS * dim, dt))
            .buffer(BenchBuffer::random("key_packed", kv_rows * key_pw, DType::U32))
            .buffer(BenchBuffer::random("key_norms", kv_rows, dt))
            .buffer(BenchBuffer::random("key_codebook", key_levels, dt))
            .buffer(BenchBuffer::random("val_packed", kv_rows * val_pw, DType::U32))
            .buffer(BenchBuffer::random("val_norms", kv_rows, dt))
            .buffer(BenchBuffer::random("val_codebook", value_levels, dt))
            .buffer(BenchBuffer::random("sinks", Q_HEADS, dt))
            .buffer(BenchBuffer::zeros("out", Q_HEADS * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("key_packed_width", key_pw as u32)
            .constexpr("value_packed_width", val_pw as u32)
            .constexpr("tokens", TOKENS as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("num_q_heads", Q_HEADS as u32)
            .constexpr("has_sinks", 0u32)
            .constexpr("window_size", 0u32)
            .grid_3d(1, Q_HEADS as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/aura_flash_sdpa_kb4_vb2_d128", dtypes = [f32, f16, bf16])]
    fn bench_kb4_vb2_d128(dt: DType) -> BenchSetup {
        setup(super::aura_flash_sdpa_kb4_vb2_d128::kernel_ir_for(dt), 128, 4, 2, dt)
    }

    #[bench(name = "ffai/aura_flash_sdpa_kb4_vb4_d128", dtypes = [f32, f16, bf16])]
    fn bench_kb4_vb4_d128(dt: DType) -> BenchSetup {
        setup(super::aura_flash_sdpa_kb4_vb4_d128::kernel_ir_for(dt), 128, 4, 4, dt)
    }

    #[bench(name = "ffai/aura_flash_sdpa_kb4_vb2_d64", dtypes = [f32, f16, bf16])]
    fn bench_kb4_vb2_d64(dt: DType) -> BenchSetup {
        setup(super::aura_flash_sdpa_kb4_vb2_d64::kernel_ir_for(dt), 64, 4, 2, dt)
    }

    #[bench(name = "ffai/aura_flash_sdpa_kb4_vb4_d64", dtypes = [f32, f16, bf16])]
    fn bench_kb4_vb4_d64(dt: DType) -> BenchSetup {
        setup(super::aura_flash_sdpa_kb4_vb4_d64::kernel_ir_for(dt), 64, 4, 4, dt)
    }
}
