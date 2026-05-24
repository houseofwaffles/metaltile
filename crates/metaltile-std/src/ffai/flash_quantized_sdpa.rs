//! Flash quantized SDPA — single-pass online-softmax attention over an
//! affine-quantized K/V cache. Port of `flash_quantized_sdpa.h`
//! (spec 041 phase 1.1/1.2). The affine-quant counterpart of
//! `aura_flash_sdpa`: K and V are dequantized inline per thread from
//! packed-index + per-group scale + bias triples (the layout
//! `quantized` matmul consumes), instead of an AURA codebook.
//!
//! Layout (row-contiguous, N = `tokens`, G = `group_size`):
//!   - queries:  [B*nQ, dim]              T   (caller has *not* pre-scaled)
//!   - k_packed: [B*nKV, N, dim/(32/bits)] u32
//!   - k_scales: [B*nKV, N, dim/G]        T
//!   - k_biases: [B*nKV, N, dim/G]        T
//!   - v_packed / v_scales / v_biases: same shape rule
//!   - sinks:    [num_q_heads]            f32
//!   - out:      [B*nQ, dim]              T
//!
//! `scale` (attention 1/sqrt(d)) multiplies the query once. `has_sinks`
//! (0/1) and `window_size` (0 = full causal) are constexpr. The packed
//! layout is the wasteful pack-strided form (`32/bits` values per u32,
//! no cross-word spill) — bits ∈ {4, 8} divide 32 cleanly.
//!
//! Lane `program_id::<0>()` ∈ [0,32) owns dim slots `lane + i*32`;
//! `program_id::<1>()` = query index. Single-simdgroup shape, matching
//! `aura_flash_sdpa` (token-parallelism is a perf follow-up).
//!
//! ## Mask variants
//!
//! Production attention often requires an explicit attention mask in
//! addition to the built-in causal / sliding-window guard. Two new
//! constexpr-gated kernel variants cover the MLX-upstream mask shapes:
//!
//! - **Bool mask** (`flash_quantized_sdpa_bool_mask_b{4,8}_d{64,128,256}`):
//!   takes a `mask_bool: Tensor<u32>` of shape `[B*nQ, tokens]` (packed
//!   as u32, one bit per token) — or flat byte-per-token; see note below.
//!   When `mask_bool[q_idx * tokens + t] == 0` the key at position `t`
//!   is skipped (softmax weight set to zero). Useful for segment packing
//!   and cross-sequence masking.
//!
//! - **Float mask** (`flash_quantized_sdpa_float_mask_b{4,8}_d{64,128,256}`):
//!   takes a `mask_float: Tensor<T>` of shape `[B*nQ, tokens]`.
//!   The value `mask_float[q_idx * tokens + t]` is added to the raw
//!   attention logit before the online-softmax step, enabling relative-
//!   position biases (ALiBi, T5 bias).
//!
//! Both variants are separate kernel functions (not combined into one)
//! to avoid the cost of loading an unused mask buffer on the common
//! causal-only path. The bool and float masks are composable by chaining
//! their logit modifications inside the token loop.
//!
//! The mask buffers are per-element (one f32/T or one u32 per token per
//! query), row-major `[B*nQ, tokens]`. For the bool mask, each slot is
//! a full `u32` (0 = masked, non-zero = visible) — matching the MLX
//! `mask_t` convention used in `aura_flash_sdpa`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, B*nQ, 1]`, `tg = [32, 1, 1]`.
//! - `dims_per_lane = ceil(dim / 32)`; `dim` a multiple of `32/bits`.
//!
//! Codegen-only; correctness pinned by
//! `tests/flash_quantized_sdpa_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

macro_rules! flash_quantized_sdpa_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[bench_kernel(op="flash_quantized_sdpa", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)]
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            // Per-lane query slice, pre-scaled by the attention scale.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            // Online-softmax accumulators (sink = virtual key, value 0).
            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    let score = simd_sum(dot_partial);

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

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

flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d64, 4u32, 64u32, 2u32, "b4_d64");
// d=96: GPT-NeoX head dim. dims_per_lane = ceil(96/32) = 3.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d96, 4u32, 96u32, 3u32, "b4_d96");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d128, 4u32, 128u32, 4u32, "b4_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d256, 4u32, 256u32, 8u32, "b4_d256");
// d=512: Gemma 4 global-attention head dim. dims_per_lane = 512/32 = 16.
// Register pressure with 16 fp32 accumulators pushes maxTotalThreadsPerThreadgroup
// below 1024; dispatch at 256 threads/TG (8 SG) — same approach as
// ffai_sdpa_decode_d512 which also uses 16 elements/lane.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d512, 4u32, 512u32, 16u32, "b4_d512");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d64, 8u32, 64u32, 2u32, "b8_d64");
// d=96: GPT-NeoX, int8.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d96, 8u32, 96u32, 3u32, "b8_d96");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d128, 8u32, 128u32, 4u32, "b8_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d256, 8u32, 256u32, 8u32, "b8_d256");
// d=512: Gemma 4 global, int8. Same 256-thread/TG constraint as b4_d512.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d512, 8u32, 512u32, 16u32, "b8_d512");

// ── Bool-mask variants ───────────────────────────────────────────────────
//
// `mask_bool: Tensor<u32>` — shape `[B*nQ, tokens]`, one u32 per token.
// When the slot is zero the key at that position is excluded from
// attention (the online-softmax contribution is dropped). Non-zero = visible.
//
// The mask tensor is flat u32 (not bit-packed) for simplicity; one u32
// per token keeps the load a single scalar read with no shift/mask.

macro_rules! flash_quantized_sdpa_bool_mask_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[bench_kernel(op="flash_quantized_sdpa", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)]
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            mask_bool: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                // Causal / sliding-window gate (same as base kernel).
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                // Bool mask gate: skip tokens where the mask slot is 0.
                let mask_pass = load(mask_bool[q_idx * tokens + t]) != 0u32;
                if use_key & mask_pass {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    let score = simd_sum(dot_partial);

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

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

flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d64,
    4u32,
    64u32,
    2u32,
    "bool_mask_b4_d64"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d128,
    4u32,
    128u32,
    4u32,
    "bool_mask_b4_d128"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d256,
    4u32,
    256u32,
    8u32,
    "bool_mask_b4_d256"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d64,
    8u32,
    64u32,
    2u32,
    "bool_mask_b8_d64"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d128,
    8u32,
    128u32,
    4u32,
    "bool_mask_b8_d128"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d256,
    8u32,
    256u32,
    8u32,
    "bool_mask_b8_d256"
);

// ── Float-mask variants ──────────────────────────────────────────────────
//
// `mask_float: Tensor<T>` — shape `[B*nQ, tokens]`, one `T` per token.
// The value is added to the raw attention logit before the softmax step,
// enabling relative-position biases (ALiBi, T5 bias, etc.).

macro_rules! flash_quantized_sdpa_float_mask_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[bench_kernel(op="flash_quantized_sdpa", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)]
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            mask_float: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    // Load the float mask bias and add it to the logit.
                    // The bias is a scalar per (q, t) token — all 32 lanes
                    // in the simdgroup load from the same address and obtain
                    // the same value, so the addition is uniform across lanes.
                    let bias = load(mask_float[q_idx * tokens + t]).cast::<f32>();
                    let score = simd_sum(dot_partial) + bias;

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

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

flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d64,
    4u32,
    64u32,
    2u32,
    "float_mask_b4_d64"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d128,
    4u32,
    128u32,
    4u32,
    "float_mask_b4_d128"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d256,
    4u32,
    256u32,
    8u32,
    "float_mask_b4_d256"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d64,
    8u32,
    64u32,
    2u32,
    "float_mask_b8_d64"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d128,
    8u32,
    128u32,
    4u32,
    "float_mask_b8_d128"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d256,
    8u32,
    256u32,
    8u32,
    "float_mask_b8_d256"
);
