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
        #[kernel]
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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_flash_pass2_d128;
    use crate::utils::{pack_f32, unpack_f32};

    // ── COMBINED AURA flash decode, exercised through pass 2 ─────────────
    //
    // The test runner dispatches a single kernel per `TestSetup`, so we
    // cannot chain p1 → pass2 on the GPU. Instead we emulate the
    // non-causal `aura_flash_p1` per-block online-softmax on the CPU
    // (decoding K/V from `codebook[index] * norm`), producing the
    // `(o_partials, m_partials, l_partials)` staging tuples, then
    // dispatch the real GPU pass-2 reducer. Its output must equal a dense
    // `softmax(QKᵀ)·V` over the whole codebook-decoded cache — proving
    // the p1-emulation → pass2 pipeline reconstructs dense attention.
    // This is the COMBINED result the two-pass AURA decode produces.

    const DIM: usize = 128;
    const KEY_BITS: usize = 4;
    const VALUE_BITS: usize = 4;

    /// Emulate `aura_flash_p1` (non-causal): per (q_head, block)
    /// online-softmax over the block's token range, K/V decoded from
    /// `codebook[index] * norm`. Emits per-block partials exactly as p1
    /// stores them (o is the un-normalised accumulator, m the block max,
    /// l the block sum_exp).
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

    /// Dense softmax-attention over codebook-decoded K,V — the COMBINED
    /// result the two-pass AURA decode must reproduce.
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
        let mut out = vec![0.0f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            let mut scores = vec![0.0f32; tokens];
            for (t, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    let q = key_idx[(kvh * tokens + t) * dim + d];
                    dot += q_rot[qh * dim + d] * key_cb[q as usize];
                }
                *s = dot * key_norms[kvh * tokens + t];
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_w = 0.0f32;
            let mut acc = vec![0.0f32; dim];
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

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
    fn test_ffai_aura_flash_pass2_combined(dt: DType) -> TestSetup {
        let (q_heads, kv_heads, tokens, dim) = (2usize, 1usize, 8usize, DIM);
        let block_size = 4usize;
        let num_blocks = tokens.div_ceil(block_size); // 2

        let key_cb: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let val_cb: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let key_idx: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % (1 << KEY_BITS)) as u32).collect();
        let val_idx: Vec<u32> = (0..kv_heads * tokens * dim)
            .map(|i| ((i * 11 + 5) % (1 << VALUE_BITS)) as u32)
            .collect();
        let key_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect();
        let val_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect();
        let q_rot: Vec<f32> =
            (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();

        // Emulate p1 → partials, rounded through the storage dtype just
        // like the GPU p1 stores them.
        let (o_part, m_part, l_part) = emulate_p1(
            &q_rot, &key_idx, &val_idx, &key_norms, &val_norms, &key_cb, &val_cb, q_heads,
            kv_heads, tokens, dim, block_size, num_blocks,
        );
        let o_part = unpack_f32(&pack_f32(&o_part, dt), dt);
        let m_part = unpack_f32(&pack_f32(&m_part, dt), dt);
        let l_part = unpack_f32(&pack_f32(&l_part, dt), dt);

        let expected = naive(
            &q_rot, &key_idx, &val_idx, &key_norms, &val_norms, &key_cb, &val_cb, q_heads,
            kv_heads, tokens, dim,
        );

        TestSetup::new(aura_flash_pass2_d128::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("o_partials", pack_f32(&o_part, dt), dt))
            .input(TestBuffer::from_vec("m_partials", pack_f32(&m_part, dt), dt))
            .input(TestBuffer::from_vec("l_partials", pack_f32(&l_part, dt), dt))
            .input(TestBuffer::zeros("output", q_heads * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("num_blocks", num_blocks as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d(q_heads as u32, 1, 1, [32, 1, 1])
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        aura_flash_pass2_d64,
        aura_flash_pass2_d80,
        aura_flash_pass2_d96,
        aura_flash_pass2_d128,
        aura_flash_pass2_d256,
        aura_flash_pass2_d512,
    };

    // Shared builder for every pass2 dim. Grid is (q_heads, 1, 1) with a
    // 32-lane reduction threadgroup; only `dim` (and the buffer sizes it
    // drives) varies. KV of 4096 tokens / block 256 = 16 blocks throughout.
    fn flash_pass2(s: BenchSetup, dim: usize, dt: DType) -> BenchSetup {
        let (q_heads, num_blocks) = (32usize, 16usize);
        s.mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("o_partials", q_heads * num_blocks * dim, dt))
            .buffer(BenchBuffer::random("m_partials", q_heads * num_blocks, dt))
            .buffer(BenchBuffer::random("l_partials", q_heads * num_blocks, dt))
            .buffer(BenchBuffer::zeros("output", q_heads * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("num_blocks", num_blocks as u32)
            // o_partials read dominates the merge.
            .bytes_moved((q_heads * num_blocks * dim * dt.size_bytes()) as u64)
            .grid_3d(q_heads as u32, 1, 1, [32, 1, 1])
    }

    #[bench(name = "ffai/aura_flash_pass2_d64", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2_d64(dt: DType) -> BenchSetup {
        flash_pass2(BenchSetup::new(aura_flash_pass2_d64::kernel_ir_for(dt)), 64, dt)
    }

    #[bench(name = "ffai/aura_flash_pass2_d80", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2_d80(dt: DType) -> BenchSetup {
        flash_pass2(BenchSetup::new(aura_flash_pass2_d80::kernel_ir_for(dt)), 80, dt)
    }

    #[bench(name = "ffai/aura_flash_pass2_d96", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2_d96(dt: DType) -> BenchSetup {
        flash_pass2(BenchSetup::new(aura_flash_pass2_d96::kernel_ir_for(dt)), 96, dt)
    }

    #[bench(name = "ffai/aura_flash_pass2_d128", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2(dt: DType) -> BenchSetup {
        // head_dim 128, decode-time KV of 4096 tokens / block 256 = 16 blocks.
        flash_pass2(BenchSetup::new(aura_flash_pass2_d128::kernel_ir_for(dt)), 128, dt)
    }

    #[bench(name = "ffai/aura_flash_pass2_d256", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2_d256(dt: DType) -> BenchSetup {
        flash_pass2(BenchSetup::new(aura_flash_pass2_d256::kernel_ir_for(dt)), 256, dt)
    }

    #[bench(name = "ffai/aura_flash_pass2_d512", dtypes = [f32, f16, bf16])]
    fn bench_flash_pass2_d512(dt: DType) -> BenchSetup {
        flash_pass2(BenchSetup::new(aura_flash_pass2_d512::kernel_ir_for(dt)), 512, dt)
    }
}
