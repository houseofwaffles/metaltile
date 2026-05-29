//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-token SDPA decode for `head_dim == 256`. Parallel of
//! `ffai_sdpa_decode` (head_dim=128) with 8 elements per lane and a
//! **2-phase output reduction** to stay under Apple's 32 KB
//! threadgroup-memory cap.
//!
//! Needed for Gemma 3 (every variant: 1B / 4B / 12B / 27B all use
//! head_dim=256), Gemma 4 (same), and a few research models.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Each lane
//!   owns 8 consecutive Q/K/V elements (`head_dim / 32 = 256 / 32 = 8`),
//!   loaded unconditionally at `lane * 8 + {0..7}`.
//! - **`head_dim == 256`.** Wrapper-enforced.
//! - **Grid: 1 threadgroup per q_head.** Wrapper uses
//!   `grid = (nQHeads * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`nQHeads % nKVHeads == 0`** (GQA fan-out is integer).
//! - **`n_kv ≤ kv_stride`** (cache walk stays within capacity).
//!
//! ## Why 2-phase output reduction
//!
//! d=128 stores all 4 per-lane output dims in 4 tg_outN buffers of
//! `n_lanes * (n_simd + 1) = 32 * 33 = 1056` floats each
//! (`+1` is bank-conflict padding). 4 × 1056 = 4 224 floats = ~16 KB.
//!
//! d=256 has 8 per-lane output dims. 8 × 1056 = 8 448 floats =
//! ~33 KB — over Apple's per-kernel threadgroup-memory cap. We
//! split into two halves (dims 0..3 then 4..7) and reuse the same
//! 4 tg_out buffers across both phases. Two extra barriers, same
//! ~16 KB allocation.
//!
//! Wrapping doc: see FFAI/CLAUDE.md §"Wrapping kernels in FFAI".

use metaltile::kernel;

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_decode_d256",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_decode_d256<T>(
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
    // Two-phase reduction: 4 tg_outN buffers, reused for dims (0..3)
    // then (4..7). 1056 = n_lanes * (n_simd + 1) with bank padding.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 8u32;
    // Pre-scale this lane's 8-element Q stripe once; K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let q4 = load(q[q_off + d0 + 4u32]).cast::<f32>() * scale;
    let q5 = load(q[q_off + d0 + 5u32]).cast::<f32>() * scale;
    let q6 = load(q[q_off + d0 + 6u32]).cast::<f32>() * scale;
    let q7 = load(q[q_off + d0 + 7u32]).cast::<f32>() * scale;
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
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3 + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7;
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
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
        o6 = o6 * factor + weight * v6;
        o7 = o7 * factor + weight * v7;
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
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // ── Cross-simdgroup output reduction — phase 1 (dims 0..3) ─────
    // Transpose layout: idx = lane * stride + sg. After barrier, sg=0
    // lanes read back ns partials each for their own (lane, d) slot.
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
    // ── Cross-simdgroup output reduction — phase 2 (dims 4..7) ─────
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
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_decode_d256;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_decode_d256::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_decode_d256 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_decode_d256"),
                "MSL for {dt:?} should declare ffai_sdpa_decode_d256:\n{src}",
            );
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_decode_d256;
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
    fn test_ffai_sdpa_decode_d256(dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 256usize);
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

        TestSetup::new(ffai_sdpa_decode_d256::kernel_ir_for(dt))
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

/// New-syntax benchmark for `ffai_sdpa_decode_d256` (`class=GenericEmpty`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_decode_d256;

    #[bench(name = "ffai/sdpa_decode_d256", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d256(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 256usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes = (2 * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d256::kernel_ir_for(dt))
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
