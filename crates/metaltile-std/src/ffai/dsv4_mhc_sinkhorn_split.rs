//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 mHC dynamic-mix split — converts the `hc_fn @ flat` 24-mix
//! output into the three per-token control tensors `(pre, post, comb)`
//! the mHC collapse / expand kernels consume.
//!
//! ```text
//!   mixes [24, n_tokens]   = hc_fn @ flat   (computed upstream)
//!   scale [3]              = (pre_scale, post_scale, comb_scale)
//!   base  [24]             = per-slot bias
//!
//!   pre[t, c]               = sigmoid(mixes[t, c]   * scale[0] + base[c])   + eps   (c ∈ [0, 4))
//!   post[t, c]              = 2 * sigmoid(mixes[t, 4+c] * scale[1] + base[4+c])     (c ∈ [0, 4))
//!   comb_raw[t, dst, src]   = mixes[t, 8 + dst*4 + src] * scale[2] + base[8+...]
//!   comb_softmax[t, dst, src] = softmax over src(comb_raw) + eps
//!   comb[t, dst, src]       = Sinkhorn(comb_softmax, iters)
//! ```
//!
//! Sinkhorn-Knopp normalization: alternate row-normalize (`dst` axis)
//! and col-normalize (`src` axis) for `sinkhorn_iters` iterations.
//! Each normalize divides each slice by `sum + eps`.
//!
//! ## Dispatch
//!
//! 1 thread per token. `n_hc=4` is hardcoded — the 4×4 comb matrix
//! lives in 16 per-thread registers across the Sinkhorn loop. For
//! decode (n_tokens=1) the kernel is single-thread; the work fits in
//! a single dispatch even at prefill scale because per-token work is
//! O(n_hc² · iters) = 16 · iters ≈ <100 FMAs.

use metaltile::kernel;

#[kernel]
pub fn ffai_dsv4_mhc_sinkhorn_split<T>(
    mixes: Tensor<T>,
    scale: Tensor<f32>,
    base: Tensor<f32>,
    mut pre: Tensor<f32>,
    mut post: Tensor<f32>,
    mut comb: Tensor<f32>,
    #[constexpr] n_tokens: u32,
    #[constexpr] eps: f32,
    #[constexpr] sinkhorn_iters: u32,
) {
    let t = tid;
    if t < n_tokens {
        let pre_scale = load(scale[0]);
        let post_scale = load(scale[1]);
        let comb_scale = load(scale[2]);

        // ── Pre: 4 × (sigmoid + eps) ──
        let mix_base = t * 24u32;
        let p0 = load(mixes[mix_base]).cast::<f32>() * pre_scale + load(base[0]);
        let p1 = load(mixes[mix_base + 1u32]).cast::<f32>() * pre_scale + load(base[1]);
        let p2 = load(mixes[mix_base + 2u32]).cast::<f32>() * pre_scale + load(base[2]);
        let p3 = load(mixes[mix_base + 3u32]).cast::<f32>() * pre_scale + load(base[3]);
        let pre_t = t * 4u32;
        store(pre[pre_t], 1.0f32 / (1.0f32 + exp(0.0f32 - p0)) + eps);
        store(pre[pre_t + 1u32], 1.0f32 / (1.0f32 + exp(0.0f32 - p1)) + eps);
        store(pre[pre_t + 2u32], 1.0f32 / (1.0f32 + exp(0.0f32 - p2)) + eps);
        store(pre[pre_t + 3u32], 1.0f32 / (1.0f32 + exp(0.0f32 - p3)) + eps);

        // ── Post: 4 × (2 · sigmoid) ──
        let q0 = load(mixes[mix_base + 4u32]).cast::<f32>() * post_scale + load(base[4]);
        let q1 = load(mixes[mix_base + 5u32]).cast::<f32>() * post_scale + load(base[5]);
        let q2 = load(mixes[mix_base + 6u32]).cast::<f32>() * post_scale + load(base[6]);
        let q3 = load(mixes[mix_base + 7u32]).cast::<f32>() * post_scale + load(base[7]);
        let post_t = t * 4u32;
        store(post[post_t], 2.0f32 / (1.0f32 + exp(0.0f32 - q0)));
        store(post[post_t + 1u32], 2.0f32 / (1.0f32 + exp(0.0f32 - q1)));
        store(post[post_t + 2u32], 2.0f32 / (1.0f32 + exp(0.0f32 - q2)));
        store(post[post_t + 3u32], 2.0f32 / (1.0f32 + exp(0.0f32 - q3)));

        // ── Comb 4×4: scale+bias, softmax-over-src, Sinkhorn ──
        //
        // Layout: comb[dst, src]. mixes[8 + dst*4 + src] supplies the
        // raw value. base[8 + ...] is per-slot bias. Store result into
        // `comb[t * 16 + dst*4 + src]`.
        let r00 = load(mixes[mix_base + 8u32]).cast::<f32>() * comb_scale + load(base[8]);
        let r01 = load(mixes[mix_base + 9u32]).cast::<f32>() * comb_scale + load(base[9]);
        let r02 = load(mixes[mix_base + 10u32]).cast::<f32>() * comb_scale + load(base[10]);
        let r03 = load(mixes[mix_base + 11u32]).cast::<f32>() * comb_scale + load(base[11]);
        let r10 = load(mixes[mix_base + 12u32]).cast::<f32>() * comb_scale + load(base[12]);
        let r11 = load(mixes[mix_base + 13u32]).cast::<f32>() * comb_scale + load(base[13]);
        let r12 = load(mixes[mix_base + 14u32]).cast::<f32>() * comb_scale + load(base[14]);
        let r13 = load(mixes[mix_base + 15u32]).cast::<f32>() * comb_scale + load(base[15]);
        let r20 = load(mixes[mix_base + 16u32]).cast::<f32>() * comb_scale + load(base[16]);
        let r21 = load(mixes[mix_base + 17u32]).cast::<f32>() * comb_scale + load(base[17]);
        let r22 = load(mixes[mix_base + 18u32]).cast::<f32>() * comb_scale + load(base[18]);
        let r23 = load(mixes[mix_base + 19u32]).cast::<f32>() * comb_scale + load(base[19]);
        let r30 = load(mixes[mix_base + 20u32]).cast::<f32>() * comb_scale + load(base[20]);
        let r31 = load(mixes[mix_base + 21u32]).cast::<f32>() * comb_scale + load(base[21]);
        let r32 = load(mixes[mix_base + 22u32]).cast::<f32>() * comb_scale + load(base[22]);
        let r33 = load(mixes[mix_base + 23u32]).cast::<f32>() * comb_scale + load(base[23]);

        // Softmax over src axis (per-row, dst fixed). Numerically stable.
        let m0 = select(r00 > r01, r00, r01);
        let m0 = select(m0 > r02, m0, r02);
        let m0 = select(m0 > r03, m0, r03);
        let e00 = exp(r00 - m0);
        let e01 = exp(r01 - m0);
        let e02 = exp(r02 - m0);
        let e03 = exp(r03 - m0);
        let s0 = e00 + e01 + e02 + e03;
        let mut c00 = e00 / s0 + eps;
        let mut c01 = e01 / s0 + eps;
        let mut c02 = e02 / s0 + eps;
        let mut c03 = e03 / s0 + eps;

        let m1 = select(r10 > r11, r10, r11);
        let m1 = select(m1 > r12, m1, r12);
        let m1 = select(m1 > r13, m1, r13);
        let e10 = exp(r10 - m1);
        let e11 = exp(r11 - m1);
        let e12 = exp(r12 - m1);
        let e13 = exp(r13 - m1);
        let s1 = e10 + e11 + e12 + e13;
        let mut c10 = e10 / s1 + eps;
        let mut c11 = e11 / s1 + eps;
        let mut c12 = e12 / s1 + eps;
        let mut c13 = e13 / s1 + eps;

        let m2 = select(r20 > r21, r20, r21);
        let m2 = select(m2 > r22, m2, r22);
        let m2 = select(m2 > r23, m2, r23);
        let e20 = exp(r20 - m2);
        let e21 = exp(r21 - m2);
        let e22 = exp(r22 - m2);
        let e23 = exp(r23 - m2);
        let s2 = e20 + e21 + e22 + e23;
        let mut c20 = e20 / s2 + eps;
        let mut c21 = e21 / s2 + eps;
        let mut c22 = e22 / s2 + eps;
        let mut c23 = e23 / s2 + eps;

        let m3 = select(r30 > r31, r30, r31);
        let m3 = select(m3 > r32, m3, r32);
        let m3 = select(m3 > r33, m3, r33);
        let e30 = exp(r30 - m3);
        let e31 = exp(r31 - m3);
        let e32 = exp(r32 - m3);
        let e33 = exp(r33 - m3);
        let s3 = e30 + e31 + e32 + e33;
        let mut c30 = e30 / s3 + eps;
        let mut c31 = e31 / s3 + eps;
        let mut c32 = e32 / s3 + eps;
        let mut c33 = e33 / s3 + eps;

        // Sinkhorn-Knopp: alternate col-normalize and row-normalize.
        // (The post-softmax matrix is already row-stochastic, so the
        // first step is col-normalize.) Iterations chained via the
        // 16 c__ live values — no array, no TG mem.
        for _iter in range(0u32, sinkhorn_iters, 1u32) {
            // Col normalize (each col divided by col sum + eps).
            let cs0 = c00 + c10 + c20 + c30 + eps;
            let cs1 = c01 + c11 + c21 + c31 + eps;
            let cs2 = c02 + c12 + c22 + c32 + eps;
            let cs3 = c03 + c13 + c23 + c33 + eps;
            c00 = c00 / cs0;
            c10 = c10 / cs0;
            c20 = c20 / cs0;
            c30 = c30 / cs0;
            c01 = c01 / cs1;
            c11 = c11 / cs1;
            c21 = c21 / cs1;
            c31 = c31 / cs1;
            c02 = c02 / cs2;
            c12 = c12 / cs2;
            c22 = c22 / cs2;
            c32 = c32 / cs2;
            c03 = c03 / cs3;
            c13 = c13 / cs3;
            c23 = c23 / cs3;
            c33 = c33 / cs3;
            // Row normalize.
            let rs0 = c00 + c01 + c02 + c03 + eps;
            let rs1 = c10 + c11 + c12 + c13 + eps;
            let rs2 = c20 + c21 + c22 + c23 + eps;
            let rs3 = c30 + c31 + c32 + c33 + eps;
            c00 = c00 / rs0;
            c01 = c01 / rs0;
            c02 = c02 / rs0;
            c03 = c03 / rs0;
            c10 = c10 / rs1;
            c11 = c11 / rs1;
            c12 = c12 / rs1;
            c13 = c13 / rs1;
            c20 = c20 / rs2;
            c21 = c21 / rs2;
            c22 = c22 / rs2;
            c23 = c23 / rs2;
            c30 = c30 / rs3;
            c31 = c31 / rs3;
            c32 = c32 / rs3;
            c33 = c33 / rs3;
        }

        // Write out [dst, src] layout: comb[t*16 + dst*4 + src].
        let comb_t = t * 16u32;
        store(comb[comb_t], c00);
        store(comb[comb_t + 1u32], c01);
        store(comb[comb_t + 2u32], c02);
        store(comb[comb_t + 3u32], c03);
        store(comb[comb_t + 4u32], c10);
        store(comb[comb_t + 5u32], c11);
        store(comb[comb_t + 6u32], c12);
        store(comb[comb_t + 7u32], c13);
        store(comb[comb_t + 8u32], c20);
        store(comb[comb_t + 9u32], c21);
        store(comb[comb_t + 10u32], c22);
        store(comb[comb_t + 11u32], c23);
        store(comb[comb_t + 12u32], c30);
        store(comb[comb_t + 13u32], c31);
        store(comb[comb_t + 14u32], c32);
        store(comb[comb_t + 15u32], c33);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_mhc_sinkhorn_split;
    use crate::utils::{pack_f32, unpack_f32};

    #[allow(clippy::too_many_arguments)]
    fn cpu_reference(
        mixes: &[f32],
        scale: &[f32],
        base: &[f32],
        n_tokens: usize,
        eps: f32,
        iters: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut pre = vec![0f32; n_tokens * 4];
        let mut post = vec![0f32; n_tokens * 4];
        let mut comb = vec![0f32; n_tokens * 16];
        for t in 0..n_tokens {
            // pre
            for c in 0..4 {
                let raw = mixes[t * 24 + c] * scale[0] + base[c];
                pre[t * 4 + c] = 1.0 / (1.0 + (-raw).exp()) + eps;
            }
            // post
            for c in 0..4 {
                let raw = mixes[t * 24 + 4 + c] * scale[1] + base[4 + c];
                post[t * 4 + c] = 2.0 / (1.0 + (-raw).exp());
            }
            // comb
            let mut c = [[0f32; 4]; 4];
            for dst in 0..4 {
                let mut row = [0f32; 4];
                for src in 0..4 {
                    row[src] =
                        mixes[t * 24 + 8 + dst * 4 + src] * scale[2] + base[8 + dst * 4 + src];
                }
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: [f32; 4] = [
                    (row[0] - m).exp(),
                    (row[1] - m).exp(),
                    (row[2] - m).exp(),
                    (row[3] - m).exp(),
                ];
                let s = exps.iter().sum::<f32>();
                for src in 0..4 {
                    c[dst][src] = exps[src] / s + eps;
                }
            }
            for _ in 0..iters {
                // Col normalize
                for src in 0..4 {
                    let cs = c[0][src] + c[1][src] + c[2][src] + c[3][src] + eps;
                    for dst in 0..4 {
                        c[dst][src] /= cs;
                    }
                }
                // Row normalize
                for dst in 0..4 {
                    let rs = c[dst][0] + c[dst][1] + c[dst][2] + c[dst][3] + eps;
                    for src in 0..4 {
                        c[dst][src] /= rs;
                    }
                }
            }
            for dst in 0..4 {
                for src in 0..4 {
                    comb[t * 16 + dst * 4 + src] = c[dst][src];
                }
            }
        }
        (pre, post, comb)
    }

    fn setup(n_tokens: usize, iters: usize, dt: DType) -> TestSetup {
        let mixes: Vec<f32> =
            (0..n_tokens * 24).map(|i| (i as f32 * 0.013 - 0.4).sin() * 0.8).collect();
        let scale: Vec<f32> = vec![0.5, 0.3, 0.7];
        let base: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.05).collect();
        let mixes_dt = unpack_f32(&pack_f32(&mixes, dt), dt);
        let eps = 1e-6_f32;
        let (pre_exp, post_exp, comb_exp) =
            cpu_reference(&mixes_dt, &scale, &base, n_tokens, eps, iters);
        TestSetup::new(ffai_dsv4_mhc_sinkhorn_split::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("mixes", pack_f32(&mixes, dt), dt))
            .input(TestBuffer::from_vec("scale", pack_f32(&scale, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("base", pack_f32(&base, DType::F32), DType::F32))
            .input(TestBuffer::zeros("pre", n_tokens * 4, DType::F32))
            .input(TestBuffer::zeros("post", n_tokens * 4, DType::F32))
            .input(TestBuffer::zeros("comb", n_tokens * 16, DType::F32))
            .constexpr("n_tokens", n_tokens as u32)
            .constexpr("eps", eps)
            .constexpr("sinkhorn_iters", iters as u32)
            .expect(TestBuffer::from_vec("pre", pack_f32(&pre_exp, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("post", pack_f32(&post_exp, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("comb", pack_f32(&comb_exp, DType::F32), DType::F32))
            .grid_1d(n_tokens, 64)
    }

    /// Single-token decode.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-3, 2e-2])]
    fn test_mhc_sinkhorn_split_decode(dt: DType) -> TestSetup { setup(1, 1, dt) }

    /// Small batch + multi-iter Sinkhorn.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-3, 2e-2])]
    fn test_mhc_sinkhorn_split_batch(dt: DType) -> TestSetup { setup(8, 3, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_mhc_sinkhorn_split;

    #[bench(name = "ffai/dsv4_mhc_sinkhorn_split", dtypes = [f32, f16, bf16])]
    fn bench_split(dt: DType) -> BenchSetup {
        // 2 mHC blocks per layer × 43 layers = 86 dispatches/token. At
        // decode (n_tokens=1) the per-dispatch work is trivial; bench
        // shape covers a prefill chunk of 256 tokens.
        let n_tokens = 256usize;
        let iters = 1usize;
        BenchSetup::new(ffai_dsv4_mhc_sinkhorn_split::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("mixes", n_tokens * 24, dt))
            .buffer(BenchBuffer::random("scale", 3, DType::F32))
            .buffer(BenchBuffer::random("base", 24, DType::F32))
            .buffer(BenchBuffer::zeros("pre", n_tokens * 4, DType::F32).output())
            .buffer(BenchBuffer::zeros("post", n_tokens * 4, DType::F32).output())
            .buffer(BenchBuffer::zeros("comb", n_tokens * 16, DType::F32).output())
            .constexpr("n_tokens", n_tokens as u32)
            .constexpr("eps", 1e-6_f32)
            .constexpr("sinkhorn_iters", iters as u32)
            .grid_1d(n_tokens, 64)
            .bytes_moved((n_tokens * 24 * dt.size_bytes() + 27 * 4 + n_tokens * 24 * 4) as u64)
    }
}
