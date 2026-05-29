//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA compressed-domain Q · K dot-product reduction.
//!
//! For each (q_head, k_position) pair, computes the dot product of the
//! rotated query vector against the codebook-quantised key vector at
//! that position, scaled by the per-position norm-correction factor.
//!
//! Port of `turbo_score` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `q_rot     [q_heads, dim]`                       f32
//! - `packed    [kv_heads, tokens, packed_width]`     u32
//! - `norms     [kv_heads, tokens]`                   f32
//! - `codebook  [2**bits]`                            f32
//!
//! Output:
//! - `scores    [q_heads, tokens]`                    f32
//!
//! ## Dispatch
//!
//! Reduction-mode kernel.  Threadgroup = (32, 1, 1); one threadgroup
//! per (q_head, token) pair via tgid_x = q_idx, tgid_y = k_idx.
//! Each of the 32 lanes accumulates a dim-strided slice of the dot
//! product; `simd_sum` reduces across the simdgroup.
//!
//! ## Constexpr params
//!
//! - `bits`            — 2 / 3 / 4 / 8.
//! - `dim`             — vector length.
//! - `packed_width`    — `ceil(dim * bits / 32)`.
//! - `repeat_count`    — GQA repeat factor (`n_q_heads / n_kv_heads`).
//!   When 1 (MHA), `kv_idx == q_idx`.
//!
//! ## Tradeoff vs the MLX upstream
//!
//! MLX caches the codebook in a per-thread stack array
//! (`float cb[LEVELS]`) before the inner loop, amortising LEVELS
//! lookups across `dim/32` iterations.  The DSL doesn't yet expose
//! stack-allocated arrays; we re-read `codebook[value]` per lookup.
//! The codebook is small (≤ 1 KB at bits=8) and Metal L1-caches
//! tightly enough that this is functionally equivalent — re-evaluate
//! if `tile profile` shows codebook reads dominating later.

use metaltile::kernel;

macro_rules! aura_score_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel(
            bench(op="aura", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Reduction,)
        )]
        pub fn $name<T>(
            q_rot: Tensor<T>,
            packed: Tensor<u32>,
            norms: Tensor<T>,
            codebook: Tensor<T>,
            mut scores: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
        ) {
            let lane = tid;
            let q_idx = tgid_x;
            let k_idx = tgid_y;
            let kv_idx = q_idx / repeat_count;

            let mask = (1u32 << $bits) - 1u32;
            let q_off = q_idx * dim;
            let packed_row = (kv_idx * tokens + k_idx) * packed_width;
            let norm_val = load(norms[kv_idx * tokens + k_idx]).cast::<f32>();

            // Lane-strided accumulation over dim.  Each lane handles
            // dims `[lane, lane + 32, lane + 64, …)` so the threadgroup
            // covers the whole vector when reduced via simd_sum.
            let mut acc = 0.0f32;
            let iters = (dim + 31u32) / 32u32;
            for it in range(0u32, iters, 1u32) {
                let d = it * 32u32 + lane;
                if d < dim {
                    // Bit-stream extract.  For bits ∈ {2,4,8} the
                    // window never spills; for {3,6} it can — branch
                    // on `shift + bits > 32` and re-fetch the next
                    // word.  Same shape as dequant_gather_int{3,6}.
                    let bit_offset = d * $bits;
                    let word_idx = bit_offset / 32u32;
                    let shift = bit_offset & 31u32;
                    let bits_in_w0 = 32u32 - shift;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(packed[packed_row + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[packed_row + w1_idx]);

                    let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let value = (lo | hi) & mask;

                    let centroid = load(codebook[value]).cast::<f32>();
                    let qv = load(q_rot[q_off + d]).cast::<f32>();
                    acc = acc + qv * centroid;
                }
            }

            // Reduce across the 32 lanes.  Only lane 0 writes the
            // result back, scaled by the per-position norm correction.
            let total = simd_sum(acc);
            if lane == 0u32 {
                store(scores[q_idx * tokens + k_idx], (total * norm_val).cast::<T>());
            }
        }
    };
}

aura_score_kernel!(aura_score_int2, 2u32, "score_int2");
aura_score_kernel!(aura_score_int3, 3u32, "score_int3");
aura_score_kernel!(aura_score_int4, 4u32, "score_int4");
aura_score_kernel!(aura_score_int6, 6u32, "score_int6");
aura_score_kernel!(aura_score_int8, 8u32, "score_int8");
