//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-token SDPA decode for `head_dim == 64`. Parallel of
//! `ffai_sdpa_decode` (head_dim=128) with 2 elements per lane instead
//! of 4 — see that file for the full algorithm walkthrough.
//!
//! Needed for production models in the head_dim=64 bracket:
//!   - Llama 3.2 1B  (32 q-heads × 64 head_dim)
//!   - GPT-OSS-20B   (64 q-heads × 64 head_dim, sliding-window layers)
//!   - Future smaller variants
//!
//! ## DISPATCH INVARIANTS
//!
//! Same shape as `ffai_sdpa_decode`, with the per-lane width halved:
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Each lane
//!   owns 2 consecutive Q/K/V elements (`head_dim / 32 = 64 / 32 = 2`),
//!   loaded unconditionally at `lane * 2 + {0, 1}`.
//! - **`head_dim == 64`.** Wrapper-enforced; other head dims belong
//!   to `ffai_sdpa_decode` (128), `ffai_sdpa_decode_d256` (256, queued),
//!   etc.
//! - **Grid: 1 threadgroup per q_head.** Wrapper uses
//!   `grid = (nQHeads * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`nQHeads % nKVHeads == 0`** so GQA fan-out is integer.
//! - **`n_kv ≤ kv_stride`.** Cache walk stays within pre-allocated
//!   capacity.
//!
//! Wrapping doc: see FFAI/CLAUDE.md §"Wrapping kernels in FFAI".

use metaltile::kernel;

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_decode_d64",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_decode_d64<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // Two tg_out slots instead of four — each lane owns 2 output
    // elements at head_dim=64. Same `+32` bank-conflict padding
    // rationale as the head_dim=128 kernel.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 2u32;
    // Pre-scale this lane's 2-element Q pair once; K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
    }
    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    // ── Cross-simdgroup reduction: outputs ──────────────────────────
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // Transpose-then-reduce with `+1` bank padding — see head_dim=128
    // kernel for the bank-conflict rationale.
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_decode_d64;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_decode_d64::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_decode_d64 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_decode_d64"),
                "MSL for {dt:?} should declare ffai_sdpa_decode_d64:\n{src}",
            );
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_decode_d64;
    use crate::utils::{pack_f32, unpack_f32};

    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv: usize,
        kv_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_q_heads * head_dim];
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = qh * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = kv_slab + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, s) in scores.iter().enumerate() {
                    acc += *s * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d64(dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 64usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected =
            naive_sdpa(&q, &k, &v, n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, scale);

        TestSetup::new(ffai_sdpa_decode_d64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_sdpa_decode_d64` (`class=GenericEmpty`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_decode_d64;

    #[bench(name = "ffai/sdpa_decode_d64", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d64(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 64usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes = (2 * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
