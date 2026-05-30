//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-query SDPA — attends `n_query` query rows against a shared
//! K/V cache in a single dispatch. Used by Nemotron-Labs-Diffusion's
//! block-diffusion / self-speculation `forwardTokens`, where a whole
//! block of tokens is forwarded at once instead of one decode step at
//! a time.
//!
//! This is `ffai_sdpa_decode` generalised with a query dimension: one
//! threadgroup per (query, q_head), the same TPG=1024 online-softmax
//! cross-simdgroup reduction. Two attention modes select via the
//! `causal` uniform:
//!
//!   - `causal == 0` — every query attends `[0, base_kv + n_query)`:
//!     full / bidirectional over the cached prefix plus the whole
//!     block (the diffusion-denoise pattern).
//!   - `causal == 1` — query `r` attends `[0, base_kv + r + 1)`:
//!     causal within the block, the prefix always fully visible
//!     (the AR-verify / causal-commit pattern).
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel — STRICT threadgroup geometry, the same
//! machine-freeze hazard as `ffai_sdpa_decode`. Consumers MUST encode
//! these as preconditions in their wrappers.
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Hard. A TPG
//!   below 32 makes `n_simd = TPG / 32 = 0`, turning the K walk
//!   `range(sg, n_kv, 0)` into an infinite GPU loop — the freeze.
//! - **`head_dim == 128`.** Each lane owns 4 consecutive Q/K/V
//!   elements at `lane*4 + {0..3}`, indexed unconditionally.
//! - **Grid: 1 threadgroup per (query, q_head).** `tgid_x` ranges
//!   `[0, n_q_heads * n_query)`; decoded `query = tgid / n_q_heads`,
//!   `q_head = tgid % n_q_heads`. Wrapper dispatches
//!   `grid = (n_q_heads * n_query * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`n_q_heads % heads_per_group == 0`** for integer GQA fan-out.
//! - **`base_kv + n_query <= kv_stride`** — the kernel never walks
//!   past the cache's allocated depth.
//!
//! K/V cache layout `[n_kv_heads, kv_stride, head_dim]`; Q and `out`
//! layout `[n_query, n_q_heads, head_dim]`. Online softmax runs in
//! fp32 throughout (storage stays in T).

use metaltile::kernel;

#[kernel]
pub fn ffai_sdpa_multi<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] causal: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // KV positions this query attends. causal: prefix + the block up to
    // and including this query. full: prefix + the entire block.
    let n_kv = select(causal == 1u32, base_kv + query_idx + 1u32, base_kv + n_query);
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;
    // Pre-scale this lane's 4-element Q quartile once; K/V are streamed.
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
    // Each simdgroup walks every ns-th KV position. simd_sum reduces the
    // per-lane quartile dot product into the full score; online softmax
    // updates the running (max, sum); V accumulates into fp32 registers.
    // Pre-compute the kv VIDs before the loads so vectorize sees 4
    // consecutive Op::Load (same constraint as ffai_sdpa_decode).
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let kv2 = kv_idx + 2u32;
        let kv3 = kv_idx + 3u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
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
    // Transpose-then-reduce with a +1 padded stride so adjacent lanes
    // hit distinct threadgroup-memory banks (see ffai_sdpa_decode).
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

// ─── Tree-causal variant: `ffai_sdpa_multi_tree_mask` ───────────────
//
// Identical to `ffai_sdpa_multi` except the `causal: u32` constexpr is
// replaced by a runtime additive `mask: Tensor<T>` of shape
// `[n_query, n_query]`. The mask is consulted ONLY for in-block KV
// positions (i.e. `_t >= base_kv`); the cached prefix (`_t < base_kv`)
// is always fully attended.
//
// Mask semantics (caller convention):
//   - `mask[q, j] = 0.0`     → query `q` MAY attend to in-block KV `j`
//                              (i.e. `j` is an ancestor of `q` in the
//                              draft tree, or `j == q`).
//   - `mask[q, j] = -inf`    → blocked (sibling / cousin / disjoint
//                              branch — cross-branch attention would
//                              taint the verify forward).
//
// Equivalence: `treeCausalMask` from FFAI's `DraftTreeNode` is the
// canonical mask producer.
//
// Dispatch invariants — IDENTICAL to `ffai_sdpa_multi`. TPG=1024,
// head_dim=128 hard, grid `[n_q_heads * n_query * 1024, 1, 1]`.
#[kernel]
pub fn ffai_sdpa_multi_tree_mask<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    mask: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // n_kv = full prefix + entire in-block region. The tree mask
    // handles intra-block causality, so we always walk all in-block KV.
    let n_kv = base_kv + n_query;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
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
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let raw_score = simd_sum(partial);
        // Tree mask: only consulted for in-block positions. The mask
        // index is `query_idx * n_query + (_t - base_kv)`. Guard against
        // u32 underflow when `_t < base_kv` by clamping the index — the
        // `select` below discards the loaded value anyway.
        let in_block = _t >= base_kv;
        let safe_block_pos = select(in_block, _t - base_kv, 0u32);
        let mask_off = query_idx * n_query + safe_block_pos;
        let mask_val = select(in_block, load(mask[mask_off]).cast::<f32>(), 0.0f32);
        let score = raw_score + mask_val;
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
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

    use super::ffai_sdpa_multi;
    use crate::utils::{pack_f32, unpack_f32};

    // Per (query, q_head): softmax(Q·Kᵀ·scale)·V over the attended KV
    // range. Q/out layout `[n_query, n_q_heads, head_dim]`, K/V layout
    // `[n_kv_heads, kv_stride, head_dim]`. `causal=false` → full range.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_multi(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base_kv: usize,
        n_query: usize,
        kv_stride: usize,
        causal: bool,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
            let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
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
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_multi(dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 128usize);
        let (base_kv, n_query) = (56usize, 8usize);
        let kv_stride = base_kv + n_query; // 64
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa_multi(
            &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, false, scale,
        );

        TestSetup::new(ffai_sdpa_multi::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_query * n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("causal", 0u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_sdpa_multi` (`class=GenericEmpty`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_multi;

    #[bench(name = "ffai/sdpa_multi", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_multi(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 128usize);
        let (base_kv, n_query) = (4096usize, 8usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv = base_kv + n_query;
        let bytes = (2 * n_query * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_multi::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_query * n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("causal", 0u32)
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_multi_tree_mask", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_multi_tree_mask(dt: DType) -> BenchSetup {
        use super::ffai_sdpa_multi_tree_mask;
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 128usize);
        let (base_kv, n_query) = (4096usize, 8usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv = base_kv + n_query;
        let bytes = (2 * n_query * n_q_heads * head_dim
            + 2 * n_kv_heads * n_kv * head_dim
            + n_query * n_query)
            * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_multi_tree_mask::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("mask", n_query * n_query, dt))
            .buffer(BenchBuffer::zeros("out", n_query * n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
