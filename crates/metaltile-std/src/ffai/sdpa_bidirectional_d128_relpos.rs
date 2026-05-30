//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Bidirectional multi-query SDPA at `head_dim == 128` with an additive
//! **relative-position bias** on the attention scores.
//!
//! This is the variant needed by the GraniteSpeech (and Conformer-style)
//! audio encoders: bidirectional self-attention over the full acoustic
//! frame sequence, with a learned per-head bias that depends only on the
//! relative offset `key_pos - query_pos`. The score for (query i, key j)
//! is `scale·(Q_i·K_j) + bias[head, (j - i) + rel_zero]` before softmax.
//!
//! It parallels [`super::sdpa_bidirectional`] (which covers the smaller
//! vision-tower head dims 32/64/72/80/96) but at head_dim=128 with 4
//! elements per lane (`128 / 32 = 4`), and adds the relpos bias term —
//! the only structural difference from a plain bidirectional d128.
//!
//! ## Relative-position bias contract
//!
//! `bias` is a per-head vector `[n_q_heads, rel_len]` indexed by relative
//! offset. For query row `i` (absolute position `base_kv + i`) attending
//! key `j` (absolute position `j ∈ [0, n_kv)`), the relative offset is
//! `j - (base_kv + i)`; the bias index is `(j + rel_zero) - (base_kv + i)`.
//! The caller picks `rel_zero` so that relative-offset 0 lands at index
//! `rel_zero` (canonically `rel_zero = base_kv + n_query - 1`, giving a
//! `rel_len = base_kv + 2·n_query - 1` vector that covers every reachable
//! offset). The index is clamped into `[0, rel_len)` for OOB safety; with
//! a correctly-sized bias it never clamps. `bias` is f32 regardless of T.
//!
//! ## DISPATCH INVARIANTS
//!
//! Identical geometry to `sdpa_bidirectional` (Reduction mode, TPG=1024,
//! one threadgroup per `(query, q_head)`, grid
//! `(n_q_heads * n_query * 1024, 1, 1)` via `grid_3d((n_q_heads*n_query),
//! 1, 1, [1024,1,1])`), with `head_dim == 128` (4 elements per lane,
//! every lane participates). Online softmax runs in fp32.
//!
//! Q / `out` layout: `[n_query, n_q_heads, head_dim]` row-major.
//! K / V layout:     `[n_kv_heads, kv_stride, head_dim]` row-major.

use metaltile::kernel;

#[kernel]
pub fn ffai_sdpa_bidirectional_d128_relpos<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    bias: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] rel_zero: u32,
    #[constexpr] rel_len: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    let n_kv = base_kv + n_query;
    // head_dim=128 → 4 elements per lane. Four tg_out slots.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let bias_head_base = q_head * rel_len;
    // Absolute query position + the bias-index numerator. Folding
    // `rel_zero` into the query position once keeps the per-key index a
    // single subtraction. `rel_zero = base_kv + n_query - 1` guarantees
    // `(_t + rel_zero) >= q_abs` for every reachable key, so the u32
    // subtraction never underflows; the select is belt-and-braces.
    let q_abs = base_kv + query_idx;
    let d0 = lane * 4u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let kv2 = kv_idx + 2u32;
        let kv3 = kv_idx + 3u32;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let k3 = load(k[kv3]).cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        // Relative-position bias index: (key + rel_zero) - query_abs,
        // clamped into [0, rel_len). Added to the reduced score (same for
        // every lane — it's a per-(query,key,head) scalar).
        let num = _t + rel_zero;
        let rel = select(num >= q_abs, num - q_abs, 0u32);
        let rel_c = select(rel < rel_len, rel, rel_len - 1u32);
        let bias_val = load(bias[bias_head_base + rel_c]);
        let score = simd_sum(partial) + bias_val;
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
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
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_bidirectional_d128_relpos;
    use crate::utils::{pack_f32, unpack_f32};

    // Per (query, q_head): softmax(Q·Kᵀ·scale + relpos_bias)·V over
    // `[0, base_kv + n_query)`. Q/out `[n_query, n_q_heads, head_dim]`,
    // K/V `[n_kv_heads, kv_stride, head_dim]`, bias `[n_q_heads, rel_len]`
    // indexed `(j + rel_zero) - (base_kv + i)`.
    #[allow(clippy::too_many_arguments)]
    fn naive_relpos(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        bias: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base_kv: usize,
        n_query: usize,
        kv_stride: usize,
        rel_zero: usize,
        rel_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let n_kv = base_kv + n_query;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
            let q_abs = base_kv + r;
            for qh in 0..n_q_heads {
                let kvh = qh / gqa;
                let q_off = (r * n_q_heads + qh) * head_dim;
                let kv_slab = kvh * kv_stride * head_dim;
                let mut scores = vec![0.0f32; n_kv];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_off = kv_slab + t * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    let rel = (t + rel_zero).saturating_sub(q_abs).min(rel_len - 1);
                    *score = dot * scale + bias[qh * rel_len + rel];
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
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    // Conformer encoder self-attention: MHA, base_kv=0, n_query=n_kv=L,
    // head_dim=128. rel_zero = L-1, rel_len = 2L-1 covers every offset.
    fn setup(dt: DType) -> TestSetup {
        let head_dim = 128usize;
        let (n_q_heads, n_kv_heads) = (4usize, 4usize);
        let (base_kv, n_query) = (0usize, 24usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let rel_zero = base_kv + n_query - 1;
        let rel_len = base_kv + 2 * n_query - 1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        // Relative-position bias is f32 in both kernel and oracle.
        let bias: Vec<f32> =
            (0..n_q_heads * rel_len).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let expected = naive_relpos(
            &q, &k, &v, &bias, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride,
            rel_zero, rel_len, scale,
        );

        TestSetup::new(ffai_sdpa_bidirectional_d128_relpos::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n_query * n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("rel_zero", rel_zero as u32)
            .constexpr("rel_len", rel_len as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_bidirectional_d128_relpos(dt: DType) -> TestSetup { setup(dt) }
}

/// New-syntax bench: GraniteSpeech-style encoder block. MHA, base_kv=0,
/// n_query=n_kv=512 acoustic frames, head_dim=128, 8 heads.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_bidirectional_d128_relpos;

    #[bench(name = "ffai/sdpa_bidirectional_d128_relpos", dtypes = [f32, f16, bf16])]
    fn bench_d128_relpos(dt: DType) -> BenchSetup {
        let head_dim = 128usize;
        let (n_q_heads, n_kv_heads) = (8usize, 8usize);
        let (base_kv, n_query) = (0usize, 512usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let rel_zero = base_kv + n_query - 1;
        let rel_len = base_kv + 2 * n_query - 1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv = base_kv + n_query;
        let bytes = (2 * n_query * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_bidirectional_d128_relpos::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("bias", n_q_heads * rel_len, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_query * n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("rel_zero", rel_zero as u32)
            .constexpr("rel_len", rel_len as u32)
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
