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

use metaltile::{bench_kernel, kernel};

use crate::bench_types::DType;

// Keep `DType` referenced for the inventory submit; `FLOAT_DTYPES`
// supersedes the old `F32_ONLY` shortlist now that the kernel handles
// fp32/fp16/bf16 via the generic `T` I/O dtype.
const _: DType = DType::F32;

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
        #[bench_kernel(op="aura", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Grid3D,)]
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
                let k_packed_row = (kv_idx * tokens + t) * key_packed_width;
                let k_norm = load(key_norms[kv_idx * tokens + t]).cast::<f32>();

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
                let v_packed_row = (kv_idx * tokens + t) * value_packed_width;
                let v_norm = load(val_norms[kv_idx * tokens + t]).cast::<f32>();

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
