//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA Flash Pass 2 — cross-block online-softmax merge.
//!
//! Reduces the `(o_partials, m_partials, l_partials)` tuples emitted
//! by `aura_flash_p1` (one tuple per (q_idx, block_idx) pair) into a
//! single `(o, m, l)` per q_idx, then writes the final attention
//! output `o / l` cast to bf16.
//!
//! Port of `turbo_flash_pass2` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `o_partials  [q_heads, num_blocks, dim]`   f32
//! - `m_partials  [q_heads, num_blocks]`        f32  — per-block max.
//! - `l_partials  [q_heads, num_blocks]`        f32  — per-block sum_exp.
//!
//! Output:
//! - `output      [q_heads, dim]`               bf16
//!
//! ## Dispatch
//!
//! Reduction mode; threadgroup = (32, 1, 1) per q_idx.  Each lane owns
//! `DIMS_PER_LANE = ceil(dim / 32)` output slots (the lane's stride-32
//! slice of `dim`), kept in a per-thread stack array.  Cross-block
//! merge: replay `b_idx ∈ [0, num_blocks)`, rescaling `o[]` and `l`
//! by the standard online-softmax max-shift on each step.
//!
//! ## Output dtype
//!
//! Bf16 directly — matches the MLX upstream's choice.  Accumulators
//! stay fp32; only the final write narrows.  See the note in the
//! upstream file about Qwen3.5-9B `!!!!!` decoding regressions when
//! this was fp32 + caller-side cast.

use metaltile::kernel;

macro_rules! aura_flash_pass2_kernel {
    ($name:ident, $dims_per_lane:literal, $subop:literal) => {
        #[kernel(
            bench(op="aura", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Reduction,)
        )]
        pub fn $name<T>(
            o_partials: Tensor<T>,
            m_partials: Tensor<T>,
            l_partials: Tensor<T>,
            mut output: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] num_blocks: u32,
        ) {
            let lane = tid;
            let q_idx = tgid_x;

            // Per-lane accumulators.  `o` is the running output slice;
            // `m` and `l` are scalars updated each block.  Initialised
            // to (-INF, 0, 0).
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let mut m_acc = neg_infinity();
            let mut l_acc = 0.0f32;

            // Replay every block; rescale on each step using the
            // standard online-softmax max-shift identity. Partials are
            // promoted to f32 for the merge — keeps numerical stability
            // independent of the storage dtype.
            for b in range(0u32, num_blocks, 1u32) {
                let ml_idx = q_idx * num_blocks + b;
                let block_m = load(m_partials[ml_idx]).cast::<f32>();
                let block_l = load(l_partials[ml_idx]).cast::<f32>();
                // Skip empty blocks (causal masking can leave some
                // blocks with l=0).
                if block_l != 0.0f32 {
                    let new_m = select(m_acc > block_m, m_acc, block_m);
                    let exp_old = exp(m_acc - new_m);
                    let exp_block = exp(block_m - new_m);

                    let partial_base = (q_idx * num_blocks + b) * dim;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let prev = stack_load("o", i);
                            let part = load(o_partials[partial_base + d]).cast::<f32>();
                            let scaled = prev * exp_old + part * exp_block;
                            stack_store("o", i, scaled);
                        }
                    }
                    l_acc = l_acc * exp_old + block_l * exp_block;
                    m_acc = new_m;
                }
            }

            // Final normalise + narrow-cast write.
            let inv_l = select(l_acc > 0.0f32, 1.0f32 / l_acc, 0.0f32);
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let v = stack_load("o", i) * inv_l;
                    store(output[q_idx * dim + d], v.cast::<T>());
                }
            }
        }
    };
}

// One instantiation per (dim).  `dims_per_lane = ceil(dim / 32)`.
//
//   dim  64  →  2 dims/lane
//   dim  80  →  3 (3·32 = 96 ≥ 80)
//   dim  96  →  3
//   dim 128  →  4
//   dim 256  →  8
//   dim 512  → 16
aura_flash_pass2_kernel!(aura_flash_pass2_d64, 2u32, "flash_pass2_d64");
aura_flash_pass2_kernel!(aura_flash_pass2_d80, 3u32, "flash_pass2_d80");
aura_flash_pass2_kernel!(aura_flash_pass2_d96, 3u32, "flash_pass2_d96");
aura_flash_pass2_kernel!(aura_flash_pass2_d128, 4u32, "flash_pass2_d128");
aura_flash_pass2_kernel!(aura_flash_pass2_d256, 8u32, "flash_pass2_d256");
aura_flash_pass2_kernel!(aura_flash_pass2_d512, 16u32, "flash_pass2_d512");
