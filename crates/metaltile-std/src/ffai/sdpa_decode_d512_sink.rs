//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-token SDPA decode for `head_dim == 512` with **GPT-OSS-style
//! attention sink** — a per-head learnable scalar that participates in
//! the softmax denominator but contributes no value.
//!
//! ```text
//!   logits[t] = (Q · Kₜ) * scale
//!   M         = max(max_t logits[t], sink_logit[q_head])
//!   denom     = sum_t exp(logits[t] - M) + exp(sink_logit[q_head] - M)
//!   out[d]    = sum_t (exp(logits[t] - M) / denom) · V[t, d]
//! ```
//!
//! Used by DeepSeek V4 HCA (hierarchical-coarse-attention) dense
//! layers — the model's `attn_sink` parameter is a `[n_heads]` fp32
//! tensor learned alongside Q/K/V/O.
//!
//! Clone of [`crate::ffai::sdpa_decode_d512`] with the sink fold
//! applied in the cross-simdgroup reduction (where `g_max` and `g_sum`
//! are finalised). All other dispatch invariants — TPG=512, 16 dims
//! per lane, 4-phase output reduction — are identical.

use metaltile::kernel;

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_decode_d512_sink",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_decode_d512_sink<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    sink_logit: Tensor<f32>,
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
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 16u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let q4 = load(q[q_off + d0 + 4u32]).cast::<f32>() * scale;
    let q5 = load(q[q_off + d0 + 5u32]).cast::<f32>() * scale;
    let q6 = load(q[q_off + d0 + 6u32]).cast::<f32>() * scale;
    let q7 = load(q[q_off + d0 + 7u32]).cast::<f32>() * scale;
    let q8 = load(q[q_off + d0 + 8u32]).cast::<f32>() * scale;
    let q9 = load(q[q_off + d0 + 9u32]).cast::<f32>() * scale;
    let q10 = load(q[q_off + d0 + 10u32]).cast::<f32>() * scale;
    let q11 = load(q[q_off + d0 + 11u32]).cast::<f32>() * scale;
    let q12 = load(q[q_off + d0 + 12u32]).cast::<f32>() * scale;
    let q13 = load(q[q_off + d0 + 13u32]).cast::<f32>() * scale;
    let q14 = load(q[q_off + d0 + 14u32]).cast::<f32>() * scale;
    let q15 = load(q[q_off + d0 + 15u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut o4 = 0.0f32;
    let mut o5 = 0.0f32;
    let mut o6 = 0.0f32;
    let mut o7 = 0.0f32;
    let mut o8 = 0.0f32;
    let mut o9 = 0.0f32;
    let mut o10 = 0.0f32;
    let mut o11 = 0.0f32;
    let mut o12 = 0.0f32;
    let mut o13 = 0.0f32;
    let mut o14 = 0.0f32;
    let mut o15 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv0 + 1u32]).cast::<f32>();
        let k2 = load(k[kv0 + 2u32]).cast::<f32>();
        let k3 = load(k[kv0 + 3u32]).cast::<f32>();
        let k4 = load(k[kv0 + 4u32]).cast::<f32>();
        let k5 = load(k[kv0 + 5u32]).cast::<f32>();
        let k6 = load(k[kv0 + 6u32]).cast::<f32>();
        let k7 = load(k[kv0 + 7u32]).cast::<f32>();
        let k8 = load(k[kv0 + 8u32]).cast::<f32>();
        let k9 = load(k[kv0 + 9u32]).cast::<f32>();
        let k10 = load(k[kv0 + 10u32]).cast::<f32>();
        let k11 = load(k[kv0 + 11u32]).cast::<f32>();
        let k12 = load(k[kv0 + 12u32]).cast::<f32>();
        let k13 = load(k[kv0 + 13u32]).cast::<f32>();
        let k14 = load(k[kv0 + 14u32]).cast::<f32>();
        let k15 = load(k[kv0 + 15u32]).cast::<f32>();
        let partial = q0 * k0
            + q1 * k1
            + q2 * k2
            + q3 * k3
            + q4 * k4
            + q5 * k5
            + q6 * k6
            + q7 * k7
            + q8 * k8
            + q9 * k9
            + q10 * k10
            + q11 * k11
            + q12 * k12
            + q13 * k13
            + q14 * k14
            + q15 * k15;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv0 + 1u32]).cast::<f32>();
        let v2 = load(v[kv0 + 2u32]).cast::<f32>();
        let v3 = load(v[kv0 + 3u32]).cast::<f32>();
        let v4 = load(v[kv0 + 4u32]).cast::<f32>();
        let v5 = load(v[kv0 + 5u32]).cast::<f32>();
        let v6 = load(v[kv0 + 6u32]).cast::<f32>();
        let v7 = load(v[kv0 + 7u32]).cast::<f32>();
        let v8 = load(v[kv0 + 8u32]).cast::<f32>();
        let v9 = load(v[kv0 + 9u32]).cast::<f32>();
        let v10 = load(v[kv0 + 10u32]).cast::<f32>();
        let v11 = load(v[kv0 + 11u32]).cast::<f32>();
        let v12 = load(v[kv0 + 12u32]).cast::<f32>();
        let v13 = load(v[kv0 + 13u32]).cast::<f32>();
        let v14 = load(v[kv0 + 14u32]).cast::<f32>();
        let v15 = load(v[kv0 + 15u32]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
        o6 = o6 * factor + weight * v6;
        o7 = o7 * factor + weight * v7;
        o8 = o8 * factor + weight * v8;
        o9 = o9 * factor + weight * v9;
        o10 = o10 * factor + weight * v10;
        o11 = o11 * factor + weight * v11;
        o12 = o12 * factor + weight * v12;
        o13 = o13 * factor + weight * v13;
        o14 = o14 * factor + weight * v14;
        o15 = o15 * factor + weight * v15;
    }
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
    let g_max0 = threadgroup_load("tg_max", 0);
    let g_sum0 = threadgroup_load("tg_sum", 0);
    // Attention-sink fold: extend the softmax denominator by one
    // "virtual" slot at score = `sink_logit[q_head]`. No value
    // contribution, so the output accumulators rescale unchanged by
    // the new max + sum.
    let sink = load(sink_logit[q_head]);
    let g_max = select(sink > g_max0, sink, g_max0);
    let g_sum = g_sum0 * exp(g_max0 - g_max) + exp(sink - g_max);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
            so3 = so3 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
        store(out[out_off + 3u32], so3.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o4 * rescale);
    threadgroup_store("tg_out1", idx, o5 * rescale);
    threadgroup_store("tg_out2", idx, o6 * rescale);
    threadgroup_store("tg_out3", idx, o7 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so4 = 0.0f32;
        let mut so5 = 0.0f32;
        let mut so6 = 0.0f32;
        let mut so7 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so4 = so4 + threadgroup_load("tg_out0", ri);
            so5 = so5 + threadgroup_load("tg_out1", ri);
            so6 = so6 + threadgroup_load("tg_out2", ri);
            so7 = so7 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 4u32], so4.cast::<T>());
        store(out[out_off + 5u32], so5.cast::<T>());
        store(out[out_off + 6u32], so6.cast::<T>());
        store(out[out_off + 7u32], so7.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o8 * rescale);
    threadgroup_store("tg_out1", idx, o9 * rescale);
    threadgroup_store("tg_out2", idx, o10 * rescale);
    threadgroup_store("tg_out3", idx, o11 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so8 = 0.0f32;
        let mut so9 = 0.0f32;
        let mut so10 = 0.0f32;
        let mut so11 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so8 = so8 + threadgroup_load("tg_out0", ri);
            so9 = so9 + threadgroup_load("tg_out1", ri);
            so10 = so10 + threadgroup_load("tg_out2", ri);
            so11 = so11 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 8u32], so8.cast::<T>());
        store(out[out_off + 9u32], so9.cast::<T>());
        store(out[out_off + 10u32], so10.cast::<T>());
        store(out[out_off + 11u32], so11.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o12 * rescale);
    threadgroup_store("tg_out1", idx, o13 * rescale);
    threadgroup_store("tg_out2", idx, o14 * rescale);
    threadgroup_store("tg_out3", idx, o15 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so12 = 0.0f32;
        let mut so13 = 0.0f32;
        let mut so14 = 0.0f32;
        let mut so15 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so12 = so12 + threadgroup_load("tg_out0", ri);
            so13 = so13 + threadgroup_load("tg_out1", ri);
            so14 = so14 + threadgroup_load("tg_out2", ri);
            so15 = so15 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 12u32], so12.cast::<T>());
        store(out[out_off + 13u32], so13.cast::<T>());
        store(out[out_off + 14u32], so14.cast::<T>());
        store(out[out_off + 15u32], so15.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_decode_d512_sink;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_decode_d512_sink::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_decode_d512_sink codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_decode_d512_sink"),
                "MSL for {dt:?} should declare ffai_sdpa_decode_d512_sink:\n{src}",
            );
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_decode_d512_sink;
    use crate::utils::{pack_f32, unpack_f32};

    /// Dense softmax-attention oracle with a per-head `sink_logit`
    /// scalar folded into the denominator (no value contribution).
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_sink(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sink: &[f32],
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
            let mut m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            m = m.max(sink[qh]);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            sum += (sink[qh] - m).exp();
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

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_ffai_sdpa_decode_d512_sink(dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 512usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q = unpack_f32(&pack_f32(&ramp(n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        // Per-head sink — non-trivial scalars; some larger than expected
        // max-score, some smaller, so the test exercises both branches
        // of the sink-fold max selection.
        let sink: Vec<f32> = (0..n_q_heads).map(|h| (h as f32 - 3.0) * 0.4).collect();
        let expected = naive_sdpa_sink(
            &q, &k, &v, &sink, n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, scale,
        );
        TestSetup::new(ffai_sdpa_decode_d512_sink::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec("sink_logit", pack_f32(&sink, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [512, 1, 1])
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_decode_d512_sink;

    #[bench(name = "ffai/sdpa_decode_d512_sink", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d512_sink(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 512usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes = (2 * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim) * dt.size_bytes()
            + n_q_heads * 4;
        BenchSetup::new(ffai_sdpa_decode_d512_sink::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("sink_logit", n_q_heads, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .grid_3d(n_q_heads as u32, 1, 1, [512, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
