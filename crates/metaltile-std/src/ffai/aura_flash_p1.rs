//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA Flash Pass 1 — per-block online-softmax over the AURA-encoded
//! K and V caches.  The hot path: runs every decode token.
//!
//! Each threadgroup processes one (q_head, k_block) pair across 32
//! lanes.  Per-lane stack arrays cache the rotated query slice and the
//! online-softmax output accumulator across the per-token inner loop;
//! a second pair of stack arrays caches the K-side and V-side codebooks
//! so the inner loop only does a table lookup, not a global memory
//! fetch.
//!
//! Port of `turbo_flash_p1` (non-causal variant) from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_flash.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `q_rot         [q_heads, dim]`                            f32
//! - `key_packed    [kv_heads, tokens, key_packed_width]`      u32
//! - `key_norms     [kv_heads, tokens]`                        f32
//! - `key_codebook  [2**key_bits]`                             f32
//! - `val_packed    [kv_heads, tokens, val_packed_width]`      u32
//! - `val_norms     [kv_heads, tokens]`                        f32
//! - `val_codebook  [2**val_bits]`                             f32
//!
//! Outputs:
//! - `o_partials    [q_heads, num_blocks, dim]`                f32
//! - `m_partials    [q_heads, num_blocks]`                     f32
//! - `l_partials    [q_heads, num_blocks]`                     f32
//!
//! `aura_flash_pass2` later reduces the partials cross-block.
//!
//! ## Dispatch
//!
//! Grid3D: (lane, q_idx, block_idx).  Threadgroup-internal lane
//! grouping (32 lanes) provides the simdgroup that `simd_sum` reduces
//! across for the Q · K dot product.
//!
//! ## Constexpr params
//!
//! - `key_bits`        — AURA K-side bit-width (2 / 3 / 4 / 8).
//! - `value_bits`      — AURA V-side bit-width.
//! - `dim`             — head_dim (64 / 80 / 96 / 128 / 256 / 512).
//! - `key_packed_width / value_packed_width` —
//!   `ceil(dim * bits / 32)`.
//! - `key_levels / value_levels` — `1 << bits`.
//! - `dims_per_lane`   — `ceil(dim / 32)`.
//!
//! Today's instantiation: `(key_bits=4, value_bits=2, dim=128)` — the
//! `aura4v2` scheme on a Qwen3-style head_dim=128.  Extend the
//! invocations at the bottom of the file for new (kb, vb, dim) combos.
//!
//! ## Bounds checking the per-lane dim slots
//!
//! Each inner loop walks dim slots via
//! `for i in 0..dims_per_lane { let d = lane + i*32; … }`.  When dim
//! isn't a multiple of 32 (e.g. dim=80 with `dims_per_lane=3` and
//! `max_d = 31 + 2*32 = 95 > 80`), the trailing lanes must skip the
//! out-of-range dim slots.  An earlier version of this kernel dropped
//! the `if d < dim { … }` guard to work around a metaltile unroll-pass
//! bug (nested `Op::If` bodies weren't being cloned + SSA-remapped
//! per iteration), but that limited us to multiple-of-32 dims.  The
//! unroll-pass fix landed alongside this kernel, so the guards are
//! back in.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! aura_flash_p1_kernel {
    (
        $name:ident,
        $key_bits:literal,
        $value_bits:literal,
        $key_levels:literal,
        $value_levels:literal,
        $dims_per_lane:literal,
        $causal:literal,
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
            mut o_partials: Tensor<T>,
            mut m_partials: Tensor<T>,
            mut l_partials: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] key_packed_width: u32,
            #[constexpr] value_packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] num_blocks: u32,
            #[constexpr] block_size: u32,
            // Global position of this query token in the KV stream. Only
            // consulted by the causal variant (`$causal == 1`): keys at
            // token index `t > q_position` are masked out. The non-causal
            // variant ignores it (constexpr, so the dead branch is folded
            // away — no runtime cost).
            #[constexpr] q_position: u32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let block_idx = program_id::<2>();
            let kv_idx = q_idx / repeat_count;

            let key_mask = (1u32 << $key_bits) - 1u32;
            let val_mask = (1u32 << $value_bits) - 1u32;

            let raw_end = block_idx * block_size + block_size;
            let clamped_end = select(raw_end > tokens, tokens, raw_end);
            // Causal cutoff: tokens strictly after `q_position` contribute
            // nothing, so the inner loop can stop at `q_position + 1`. For
            // the non-causal variant `$causal == 0` makes this a no-op
            // (the macro substitutes the literal at compile time).
            let causal_end = select($causal == 1u32, q_position + 1u32, clamped_end);
            let t_end = select(causal_end < clamped_end, causal_end, clamped_end);
            let t_start = block_idx * block_size;

            // ── Cache codebooks in per-thread stack arrays.  Each lane
            // touches the same codebook; the cache amortises lookups
            // across the inner per-token loop.
            stack_alloc("key_cb", $key_levels, "f32");
            for i in range(0u32, $key_levels, 1u32) {
                stack_store("key_cb", i, load(key_codebook[i]).cast::<f32>());
            }
            stack_alloc("val_cb", $value_levels, "f32");
            for i in range(0u32, $value_levels, 1u32) {
                stack_store("val_cb", i, load(val_codebook[i]).cast::<f32>());
            }

            // ── Per-lane slice of the rotated query vector — held in
            // stack registers, loaded once.  Trailing lanes whose
            // `d >= dim` get zero so the dot product treats them as a
            // no-op. Loaded as T and promoted to f32 for compute.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(q_rot[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v);
            }

            // ── Online-softmax accumulators.  `m` is the running max,
            // `l` the running sum_exp, `o[]` the un-normalised output
            // slice for this lane.
            let mut m_acc = neg_infinity();
            let mut l_acc = 0.0f32;
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            // ── Per-token inner loop ───────────────────────────────────
            for t in range(t_start, t_end, 1u32) {
                // Row stride is `kv_stride` (cache's `maxSeq`), not `tokens`
                // (live KV-row count). When the cache isn't fully populated,
                // head 1 starts at byte offset `kv_stride`, NOT `tokens` —
                // otherwise we'd read head 0's tail bytes as head 1's rows.
                let k_packed_row = (kv_idx * kv_stride + t) * key_packed_width;
                let k_norm = load(key_norms[kv_idx * kv_stride + t]).cast::<f32>();

                // Q · K via compressed-domain dot — bit-extract per dim,
                // lookup centroid in cached key_cb, accumulate against the
                // pre-loaded q_vals slice, simd_sum across the lane group.
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

                // Online-softmax max-shift identity.
                let new_m = select(m_acc > score, m_acc, score);
                let exp_diff = exp(m_acc - new_m);
                let exp_score = exp(score - new_m);

                // V-side update: bit-extract each value, look up in the
                // cached val_cb, scale by exp_score · v_norm, fold into
                // the running output via the standard online-softmax
                // rescale-then-add.
                let v_packed_row = (kv_idx * kv_stride + t) * value_packed_width;
                let v_norm = load(val_norms[kv_idx * kv_stride + t]).cast::<f32>();

                for i in range(0u32, $dims_per_lane, 1u32) {
                    let d = lane + i * 32u32;
                    if d < dim {
                        let bit_offset = d * $value_bits;
                        let word_idx = bit_offset / 32u32;
                        let shift = bit_offset & 31u32;
                        let bits_in_w0 = 32u32 - shift;
                        let lo_bits = select(bits_in_w0 >= $value_bits, $value_bits, bits_in_w0);
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

            // ── Write per-block partials (cast f32 → T on store) ───────
            let partial_base = (q_idx * num_blocks + block_idx) * dim;
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    store(o_partials[partial_base + d], stack_load("o", i).cast::<T>());
                }
            }
            if lane == 0u32 {
                let ml_idx = q_idx * num_blocks + block_idx;
                store(m_partials[ml_idx], m_acc.cast::<T>());
                store(l_partials[ml_idx], l_acc.cast::<T>());
            }
        }
    };
}

// Production (kb, vb, dim) instantiations. The macro is parametric;
// adding a row generates one more dispatchable kernel.
//
//   dims_per_lane = ceil(dim / 32)
//   {kb,vb}_levels = 2^{kb,vb}
//
// Coverage today:
//   - head_dim=128: covers Qwen3, Llama 3.2 3B+, GPT-OSS full-attn layers
//   - head_dim=64:  covers Llama 3.2 1B and GPT-OSS sliding-window layers
//
// Symmetric (kb=vb=4) is the AURAScheme.default (aura4v4) — stability-
// first. Asymmetric kb=4 vb=2 is the production recipe aura4v2 — ~5×
// compression vs fp16 per `papers/aura-compression-algorithm.md` §2.5.
//
// Other dims (80, 96, 192, 256) + other recipes (aura8, aura3) queued
// behind a real consumer — adding more variants now is `make
// emit-all` weight bloat without a use site.
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    0u32,
    "flash_p1_kb4_vb2_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb4_d128,
    4u32,
    4u32,
    16u32,
    16u32,
    4u32,
    0u32,
    "flash_p1_kb4_vb4_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    0u32,
    "flash_p1_kb4_vb2_d64"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb4_d64,
    4u32,
    4u32,
    16u32,
    16u32,
    2u32,
    0u32,
    "flash_p1_kb4_vb4_d64"
);

// ── Causal variants ──────────────────────────────────────────────────────
//
// Same compressed-domain online-softmax as the non-causal kernels, with
// the per-token loop clamped at `q_position + 1` — every key strictly
// after the query token is masked out. This is the prefill / chunked
// form upstream's `turbo_flash_p1` carries as the `causal` template
// flag. The `$causal == 1` literal lets the codegen const-fold the
// `causal_end` selection, so the only runtime difference vs the
// non-causal sibling is the inner-loop trip count.
//
// Production recipe `aura4v2` (kb=4, vb=2) for the two head dims FFAI
// ships today; the symmetric `aura4v4` causal variant follows the same
// macro arm if a consumer needs it.
aura_flash_p1_kernel!(
    aura_flash_p1_causal_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    1u32,
    "flash_p1_causal_kb4_vb2_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_causal_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    1u32,
    "flash_p1_causal_kb4_vb2_d64"
);

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_flash_p1_kb4_vb2_d128;
    use crate::utils::{pack_f32, unpack_f32};

    // ── GPU pass-1 partials, validated directly ─────────────────────────
    //
    // The companion `aura_flash_pass2` test CPU-emulates pass 1 to stage
    // the partials it feeds the GPU reducer — so the GPU pass-1 path itself
    // was never exercised. This test dispatches the real `aura_flash_p1`
    // kernel and validates its three partial outputs against a CPU oracle:
    // the same per-(q_head, block) online-softmax block reduction over the
    // AURA-codebook-decoded K,V that `aura_flash_pass2`'s `emulate_p1`
    // reference computes. All three partials are deterministic per
    // (q_head, block), so all three are checked.

    const DIM: usize = 128;
    const KEY_BITS: usize = 4;
    const VALUE_BITS: usize = 2;

    /// Bit-pack per-dim codebook indices into `[kv_heads, tokens,
    /// packed_width]` u32 words, mirroring the kernel's spill-aware decode
    /// (`bit_offset = d * bits`, split across `word_idx` / `word_idx + 1`).
    /// `kv_stride` is the row stride (cache `maxSeq`); here it equals
    /// `tokens` so the live rows pack contiguously.
    fn pack_indices(
        indices: &[u32],
        kv_heads: usize,
        kv_stride: usize,
        tokens: usize,
        dim: usize,
        bits: usize,
    ) -> Vec<u32> {
        let packed_width = (dim * bits).div_ceil(32);
        let mask = (1u32 << bits) - 1;
        let mut packed = vec![0u32; kv_heads * kv_stride * packed_width];
        for h in 0..kv_heads {
            for t in 0..tokens {
                let row = (h * kv_stride + t) * packed_width;
                for d in 0..dim {
                    let idx = indices[(h * tokens + t) * dim + d] & mask;
                    let bit_offset = d * bits;
                    let word_idx = bit_offset / 32;
                    let shift = bit_offset & 31;
                    let bits_in_w0 = 32 - shift;
                    // Low part lands in word_idx; any spill into word_idx+1.
                    packed[row + word_idx] |= idx << shift;
                    if bits_in_w0 < bits {
                        packed[row + word_idx + 1] |= idx >> bits_in_w0;
                    }
                }
            }
        }
        packed
    }

    /// CPU oracle for `aura_flash_p1` (non-causal): per (q_head, block)
    /// online-softmax over the block's token range, K/V decoded from
    /// `codebook[index] * norm`. Emits the partials exactly as the kernel
    /// stores them — `o` un-normalised accumulator `[q_head, block, dim]`,
    /// `m` block max `[q_head, block]`, `l` block sum_exp `[q_head, block]`.
    /// Mirrors `aura_flash_pass2::kernel_tests::emulate_p1`.
    #[allow(clippy::too_many_arguments)]
    fn emulate_p1(
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
        block_size: usize,
        num_blocks: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let repeat = q_heads / kv_heads;
        let mut o_part = vec![0.0f32; q_heads * num_blocks * dim];
        let mut m_part = vec![f32::NEG_INFINITY; q_heads * num_blocks];
        let mut l_part = vec![0.0f32; q_heads * num_blocks];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            for block in 0..num_blocks {
                let t_start = block * block_size;
                let t_end = ((block + 1) * block_size).min(tokens);
                let mut m_acc = f32::NEG_INFINITY;
                let mut l_acc = 0.0f32;
                let mut acc = vec![0.0f32; dim];
                for t in t_start..t_end {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        let q = key_idx[(kvh * tokens + t) * dim + d];
                        dot += q_rot[qh * dim + d] * key_cb[q as usize];
                    }
                    let score = dot * key_norms[kvh * tokens + t];
                    let new_m = score.max(m_acc);
                    let exp_diff = (m_acc - new_m).exp();
                    let exp_score = (score - new_m).exp();
                    for (d, a) in acc.iter_mut().enumerate() {
                        let v = val_idx[(kvh * tokens + t) * dim + d];
                        *a = *a * exp_diff
                            + exp_score * val_cb[v as usize] * val_norms[kvh * tokens + t];
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
                let base = (qh * num_blocks + block) * dim;
                o_part[base..base + dim].copy_from_slice(&acc);
                m_part[qh * num_blocks + block] = m_acc;
                l_part[qh * num_blocks + block] = l_acc;
            }
        }
        (o_part, m_part, l_part)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
    fn test_ffai_aura_flash_p1_kb4_vb2_d128(dt: DType) -> TestSetup {
        // Small shape matching the bench's representative variant
        // (kb=4, vb=2, dim=128) shrunk to: 2 q-heads / 1 kv-head, 8 tokens,
        // block_size 4 → num_blocks 2. kv_stride == tokens (fully packed).
        let (q_heads, kv_heads, tokens, dim) = (2usize, 1usize, 8usize, DIM);
        let repeat = q_heads / kv_heads;
        let block_size = 4usize;
        let num_blocks = tokens.div_ceil(block_size); // 2
        let kv_stride = tokens;
        let key_pw = (dim * KEY_BITS).div_ceil(32); // 16
        let val_pw = (dim * VALUE_BITS).div_ceil(32); // 8

        let key_cb: Vec<f32> = (0..(1 << KEY_BITS)).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let val_cb: Vec<f32> =
            (0..(1 << VALUE_BITS)).map(|i| -1.0 + 2.0 * i as f32 / 3.0).collect();
        let key_idx: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % (1 << KEY_BITS)) as u32).collect();
        let val_idx: Vec<u32> = (0..kv_heads * tokens * dim)
            .map(|i| ((i * 11 + 5) % (1 << VALUE_BITS)) as u32)
            .collect();
        let key_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect();
        let val_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect();
        let q_rot: Vec<f32> =
            (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();

        // Bit-pack K/V indices into the kernel's packed layout.
        let key_packed = pack_indices(&key_idx, kv_heads, kv_stride, tokens, dim, KEY_BITS);
        let val_packed = pack_indices(&val_idx, kv_heads, kv_stride, tokens, dim, VALUE_BITS);

        // CPU oracle partials, rounded through the storage dtype to match
        // the kernel's cast-on-store.
        let (o_part, m_part, l_part) = emulate_p1(
            &q_rot, &key_idx, &val_idx, &key_norms, &val_norms, &key_cb, &val_cb, q_heads,
            kv_heads, tokens, dim, block_size, num_blocks,
        );
        let o_part = unpack_f32(&pack_f32(&o_part, dt), dt);
        let m_part = unpack_f32(&pack_f32(&m_part, dt), dt);
        let l_part = unpack_f32(&pack_f32(&l_part, dt), dt);

        TestSetup::new(aura_flash_p1_kb4_vb2_d128::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("q_rot", pack_f32(&q_rot, dt), dt))
            .input(TestBuffer::from_vec("key_packed", pack_u32(&key_packed), DType::U32))
            .input(TestBuffer::from_vec("key_norms", pack_f32(&key_norms, dt), dt))
            .input(TestBuffer::from_vec("key_codebook", pack_f32(&key_cb, dt), dt))
            .input(TestBuffer::from_vec("val_packed", pack_u32(&val_packed), DType::U32))
            .input(TestBuffer::from_vec("val_norms", pack_f32(&val_norms, dt), dt))
            .input(TestBuffer::from_vec("val_codebook", pack_f32(&val_cb, dt), dt))
            .input(TestBuffer::zeros("o_partials", q_heads * num_blocks * dim, dt))
            .input(TestBuffer::zeros("m_partials", q_heads * num_blocks, dt))
            .input(TestBuffer::zeros("l_partials", q_heads * num_blocks, dt))
            .constexpr("dim", dim as u32)
            .constexpr("key_packed_width", key_pw as u32)
            .constexpr("value_packed_width", val_pw as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("num_blocks", num_blocks as u32)
            .constexpr("block_size", block_size as u32)
            // Non-causal: q_position is constexpr-folded out of the loop.
            .constexpr("q_position", (tokens - 1) as u32)
            // All three partials are deterministic per (q_head, block).
            .expect(TestBuffer::from_vec("o_partials", pack_f32(&o_part, dt), dt))
            .expect(TestBuffer::from_vec("m_partials", pack_f32(&m_part, dt), dt))
            .expect(TestBuffer::from_vec("l_partials", pack_f32(&l_part, dt), dt))
            // grid_3d args are threadGROUP counts: (lane=1, q_heads,
            // num_blocks) with a 32-lane threadgroup, copied from the bench.
            .grid_3d(1, q_heads as u32, num_blocks as u32, [32, 1, 1])
    }

    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        aura_flash_p1_causal_kb4_vb2_d64,
        aura_flash_p1_causal_kb4_vb2_d128,
        aura_flash_p1_kb4_vb2_d64,
        aura_flash_p1_kb4_vb2_d128,
        aura_flash_p1_kb4_vb4_d64,
        aura_flash_p1_kb4_vb4_d128,
    };

    // Shared builder for every (kb, vb, dim) flash-p1 variant. The grid is
    // (1, q_heads, num_blocks) with a 32-lane threadgroup; only the
    // dim-dependent packed widths and codebook sizes vary per variant.
    // `causal` selects q_position so the causal kernels exercise their
    // clamped inner loop (mid-stream query) vs the non-causal full sweep.
    fn flash_p1(
        s: BenchSetup,
        dt: DType,
        dim: usize,
        key_bits: usize,
        val_bits: usize,
        causal: bool,
    ) -> BenchSetup {
        let (q_heads, kv_heads, tokens) = (32usize, 8usize, 4096usize);
        let repeat = q_heads / kv_heads;
        let block_size = 256usize;
        let num_blocks = tokens.div_ceil(block_size);
        let kv_stride = tokens;
        let key_pw = (dim * key_bits).div_ceil(32);
        let val_pw = (dim * val_bits).div_ceil(32);
        // Causal: place the query mid-stream so roughly half the blocks
        // contribute. Non-causal: ignored by the kernel (constexpr-folded).
        let q_position = if causal { tokens / 2 } else { tokens - 1 };

        s.mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("q_rot", q_heads * dim, dt))
            .buffer(BenchBuffer::random("key_packed", kv_heads * kv_stride * key_pw, DType::U32))
            .buffer(BenchBuffer::random("key_norms", kv_heads * kv_stride, dt))
            .buffer(BenchBuffer::random("key_codebook", 1 << key_bits, dt))
            .buffer(BenchBuffer::random("val_packed", kv_heads * kv_stride * val_pw, DType::U32))
            .buffer(BenchBuffer::random("val_norms", kv_heads * kv_stride, dt))
            .buffer(BenchBuffer::random("val_codebook", 1 << val_bits, dt))
            .buffer(BenchBuffer::zeros("o_partials", q_heads * num_blocks * dim, dt).output())
            .buffer(BenchBuffer::zeros("m_partials", q_heads * num_blocks, dt).output())
            .buffer(BenchBuffer::zeros("l_partials", q_heads * num_blocks, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("key_packed_width", key_pw as u32)
            .constexpr("value_packed_width", val_pw as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("num_blocks", num_blocks as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("q_position", q_position as u32)
            // K+V packed reads dominate.
            .bytes_moved((kv_heads * kv_stride * (key_pw + val_pw) * 4) as u64)
            .grid_3d(1, q_heads as u32, num_blocks as u32, [32, 1, 1])
    }

    #[bench(name = "ffai/aura_flash_p1_kb4_vb2_d128", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1(dt: DType) -> BenchSetup {
        // Production: head_dim 128, kb=4 vb=2, decode-time KV of 4096 tokens.
        flash_p1(
            BenchSetup::new(aura_flash_p1_kb4_vb2_d128::kernel_ir_for(dt)),
            dt,
            128,
            4,
            2,
            false,
        )
    }

    #[bench(name = "ffai/aura_flash_p1_kb4_vb4_d128", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1_kb4_vb4_d128(dt: DType) -> BenchSetup {
        // Symmetric aura4v4 on head_dim 128.
        flash_p1(
            BenchSetup::new(aura_flash_p1_kb4_vb4_d128::kernel_ir_for(dt)),
            dt,
            128,
            4,
            4,
            false,
        )
    }

    #[bench(name = "ffai/aura_flash_p1_kb4_vb2_d64", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1_kb4_vb2_d64(dt: DType) -> BenchSetup {
        // aura4v2 on head_dim 64 (Llama 3.2 1B, GPT-OSS sliding window).
        flash_p1(BenchSetup::new(aura_flash_p1_kb4_vb2_d64::kernel_ir_for(dt)), dt, 64, 4, 2, false)
    }

    #[bench(name = "ffai/aura_flash_p1_kb4_vb4_d64", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1_kb4_vb4_d64(dt: DType) -> BenchSetup {
        // Symmetric aura4v4 on head_dim 64.
        flash_p1(BenchSetup::new(aura_flash_p1_kb4_vb4_d64::kernel_ir_for(dt)), dt, 64, 4, 4, false)
    }

    #[bench(name = "ffai/aura_flash_p1_causal_kb4_vb2_d128", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1_causal_kb4_vb2_d128(dt: DType) -> BenchSetup {
        // Causal prefill/chunked form, aura4v2 on head_dim 128.
        flash_p1(
            BenchSetup::new(aura_flash_p1_causal_kb4_vb2_d128::kernel_ir_for(dt)),
            dt,
            128,
            4,
            2,
            true,
        )
    }

    #[bench(name = "ffai/aura_flash_p1_causal_kb4_vb2_d64", dtypes = [f32, f16, bf16])]
    fn bench_flash_p1_causal_kb4_vb2_d64(dt: DType) -> BenchSetup {
        // Causal prefill/chunked form, aura4v2 on head_dim 64.
        flash_p1(
            BenchSetup::new(aura_flash_p1_causal_kb4_vb2_d64::kernel_ir_for(dt)),
            dt,
            64,
            4,
            2,
            true,
        )
    }
}
