//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Decode-form scaled dot-product attention — `mt_sdpa_vector` family.
//!
//! Faithful port of MLX `sdpa_vector<T, D, V=D>` template. One threadgroup
//! per Q head, 1024 threads = `BN × BD = 32 simdgroups × 32 lanes`. Each
//! simdgroup walks a stride-`BN` slice of `n_kv` positions, then a two-step
//! cross-simdgroup reduction combines the partial online-softmax results.
//!
//! ## Head-dim coverage
//!
//! | Kernel                    | head_dim | elems/lane | phases |
//! |---------------------------|----------|------------|--------|
//! | `mt_sdpa_vector_d64`      | 64       | 2          | 2      |
//! | `mt_sdpa_vector_d96`      | 96       | 3          | 3      |
//! | `mt_sdpa_vector` (d=128)  | 128      | 4          | 4      |
//! | `mt_sdpa_vector_d192`     | 192      | 6          | 6      |
//! | `mt_sdpa_vector_d256`     | 256      | 8          | 8      |
//!
//! All variants use TPG=1024 (32 SG × 32 lanes). The cross-simdgroup output
//! reduction reuses a single 1024-slot `tg_out` buffer, cycling once per
//! element group — identical to the d=128 baseline, just more passes.
//!
//! `ffai/sdpa_decode.rs` is a sibling kernel with the same dispatch +
//! reduction shape but extra FFAI-specific surface
//! (`kv_stride`, `heads_per_group`, `sink_end`, `window_start`). The
//! split is deliberate: this file's charter is a 1:1 MLX port for the
//! `tile bench` head-to-head, so additions that diverge from MLX's
//! `sdpa_vector` template live in `ffai/`. Bandwidth fixes that apply
//! to both should be ported across — see the `tg_out` occupancy fix
//! in PR #43 for the precedent.

use metaltile::kernel;

// ── d=128 baseline (original, benchmarked against MLX) ──────────────────────

#[kernel]
pub fn mt_sdpa_vector<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // 32-slot scalars for the cross-simdgroup max/sum + a 1024-slot output
    // buffer reused 4× in the reduction loop below. Matches MLX's layout:
    // 4 KB tg memory total. On M2 (32 KB tg/SM) that's 7 concurrent TGs/SM
    // vs the 2 we got from the old 16 KB / 4-array layout — the missing
    // occupancy factor that capped bf16 at 62% MT despite vectorized loads.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);
    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    let d0 = lane * 4u32;
    // Each lane pre-scales its 4 query elements once. K/V are streamed.
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
    // Per iter: dot Q with one K row → online-softmax update → accumulate V row.
    // Pre-computing the 4 KV indices and issuing the 4 loads as a single run
    // (no BinOp/Cast interleaved) is what lets the vectorize pass collapse
    // them into one bfloat4 / half4 / float4 load — same shape as
    // `sdpa_decode_2pass_pass1`. Inline'd loads + casts broke the run before.
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let kv2 = kv0 + 2u32;
        let kv3 = kv0 + 3u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3);
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
    // ── Cross-simdgroup reduction: max + sum_exp ───────────────────
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
    // ── Cross-simdgroup reduction: outputs ─────────────────────────
    //
    // Per output element: write per-(lane, sg) partial, barrier, transpose-
    // load (sg*ns + lane) × per-sg `factor_g`, simd_sum across the 32 lanes.
    // lane 0 of each sg then holds the reduced value for output position
    // `sg*4 + i`. Reuses the single 1 KB `tg_out` array for all 4 iters —
    // see the `threadgroup_alloc` comment above for the occupancy rationale.
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);
    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o2);
    threadgroup_barrier();
    let red2 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o3);
    threadgroup_barrier();
    let red3 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    // lane 0 of each simdgroup writes its 4 elements at `q_off + sg*4`.
    // Output assignment is sg-indexed (was lane-indexed pre-occupancy fix),
    // matching MLX. f32→T narrowing is implicit at the MSL Store — adding
    // a `.cast::<T>()` would break the 4-consecutive-Store vectorize window
    // and double-wrap bf16 (`bfloat(bfloat(val))`).
    if lane == 0u32 {
        let out_off = q_off + sg * 4u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
        store(out[out_off + 3u32], red3);
    }
}

// ── Additional head_dim variants ─────────────────────────────────────────────
//
// All use the same BN=32-SG × BD=32-lane = 1024-thread geometry. The output
// reduction reuses the single 1024-slot `tg_out` buffer, cycling once per
// group of 2/3/6/8 output elements. No new threadgroup-memory primitives
// needed: even at d=256 (8 phases × 1024 f32 = 32 KB) Apple's 32 KB cap is
// just met (the d=256 sdpa_decode uses the same approach at 1024 threads).

/// GQA decode SDPA, head_dim=64. Each lane owns 2 elements (`64/32`).
/// Generic `T ∈ {f32, f16, bf16}`. TPG=1024; grid=[n_q_heads,1,1].
#[kernel]
pub fn mt_sdpa_vector_d64<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);
    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    // Each lane owns 2 consecutive elements: d0 = lane * 2.
    let d0 = lane * 2u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1);
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
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);
    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    if lane == 0u32 {
        let out_off = q_off + sg * 2u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
    }
}

/// GQA decode SDPA, head_dim=96. Each lane owns 3 elements (`96/32`).
/// Generic `T ∈ {f32, f16, bf16}`. TPG=1024; grid=[n_q_heads,1,1].
#[kernel]
pub fn mt_sdpa_vector_d96<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);
    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    let d0 = lane * 3u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let kv2 = kv0 + 2u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1 + q2 * k2);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
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
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);
    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o2);
    threadgroup_barrier();
    let red2 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    if lane == 0u32 {
        let out_off = q_off + sg * 3u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
    }
}

/// GQA decode SDPA, head_dim=192. Each lane owns 6 elements (`192/32`).
/// Generic `T ∈ {f32, f16, bf16}`. TPG=1024; grid=[n_q_heads,1,1].
///
/// 6 live K accumulators + 6 V accumulators per lane. Register pressure is
/// higher than d=128 but below the d=256 bound that `ffai_sdpa_decode_d256`
/// confirmed fits in 1024 threads/TG (8 accumulators).
#[kernel]
pub fn mt_sdpa_vector_d192<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);
    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    let d0 = lane * 6u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let q4 = load(q[q_off + d0 + 4u32]).cast::<f32>() * scale;
    let q5 = load(q[q_off + d0 + 5u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut o4 = 0.0f32;
    let mut o5 = 0.0f32;
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let kv2 = kv0 + 2u32;
        let kv3 = kv0 + 3u32;
        let kv4 = kv0 + 4u32;
        let kv5 = kv0 + 5u32;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let k3 = load(k[kv3]).cast::<f32>();
        let k4 = load(k[kv4]).cast::<f32>();
        let k5 = load(k[kv5]).cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3 + q4 * k4 + q5 * k5);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        let v4 = load(v[kv4]).cast::<f32>();
        let v5 = load(v[kv5]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
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
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);
    // 6-phase output reduction — each phase reduces one element.
    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o2);
    threadgroup_barrier();
    let red2 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o3);
    threadgroup_barrier();
    let red3 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o4);
    threadgroup_barrier();
    let red4 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o5);
    threadgroup_barrier();
    let red5 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    if lane == 0u32 {
        let out_off = q_off + sg * 6u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
        store(out[out_off + 3u32], red3);
        store(out[out_off + 4u32], red4);
        store(out[out_off + 5u32], red5);
    }
}

/// GQA decode SDPA, head_dim=256. Each lane owns 8 elements (`256/32`).
/// Generic `T ∈ {f32, f16, bf16}`. TPG=1024; grid=[n_q_heads,1,1].
///
/// 8 live K accumulators + 8 V accumulators per lane. `ffai_sdpa_decode_d256`
/// confirmed that 8-element/lane fits within the pipeline cap at 1024
/// threads/TG. Output reduction uses 8 sequential phases over the same
/// 1024-slot `tg_out` buffer (same ~4 KB TG footprint as d=128).
#[kernel]
pub fn mt_sdpa_vector_d256<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);
    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    let d0 = lane * 8u32;
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
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let kv2 = kv0 + 2u32;
        let kv3 = kv0 + 3u32;
        let kv4 = kv0 + 4u32;
        let kv5 = kv0 + 5u32;
        let kv6 = kv0 + 6u32;
        let kv7 = kv0 + 7u32;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let k3 = load(k[kv3]).cast::<f32>();
        let k4 = load(k[kv4]).cast::<f32>();
        let k5 = load(k[kv5]).cast::<f32>();
        let k6 = load(k[kv6]).cast::<f32>();
        let k7 = load(k[kv7]).cast::<f32>();
        let score =
            simd_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3 + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        let v4 = load(v[kv4]).cast::<f32>();
        let v5 = load(v[kv5]).cast::<f32>();
        let v6 = load(v[kv6]).cast::<f32>();
        let v7 = load(v[kv7]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
        o6 = o6 * factor + weight * v6;
        o7 = o7 * factor + weight * v7;
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
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);
    // 8-phase output reduction — identical single-buffer strategy as d=128.
    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o2);
    threadgroup_barrier();
    let red2 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o3);
    threadgroup_barrier();
    let red3 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o4);
    threadgroup_barrier();
    let red4 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o5);
    threadgroup_barrier();
    let red5 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o6);
    threadgroup_barrier();
    let red6 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    threadgroup_barrier();
    threadgroup_store("tg_out", lane * ns + sg, o7);
    threadgroup_barrier();
    let red7 = simd_sum(threadgroup_load("tg_out", sg * ns + lane) * factor_g) * inv_sum;
    if lane == 0u32 {
        let out_off = q_off + sg * 8u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
        store(out[out_off + 3u32], red3);
        store(out[out_off + 4u32], red4);
        store(out[out_off + 5u32], red5);
        store(out[out_off + 6u32], red6);
        store(out[out_off + 7u32], red7);
    }
}

/// New-syntax correctness + benchmarks for the decode-step attention vector
/// kernels (`mt_sdpa_vector` at head_dim 128 plus the d64/d96/d192/d256
/// variants). Oracle: a straight triple-loop `O = softmax(Q·Kᵀ·scale)·V` per
/// Q head with GQA via `kv_head = q_head / gqa_factor` (the same reference the
/// legacy `sdpa_vector_gpu_correctness.rs` pins). The CPU oracle is reused by
/// `scaled_dot_product_attention`'s `mt_sdpa`. Inputs are dtype-rounded so the
/// oracle sees what the kernel loads after the f32 cast.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::{
        mt_sdpa_vector,
        mt_sdpa_vector_d64,
        mt_sdpa_vector_d96,
        mt_sdpa_vector_d192,
        mt_sdpa_vector_d256,
    };
    use crate::utils::{pack_f32, unpack_f32};

    /// `O = softmax(Q·Kᵀ·scale)·V`, GQA via `kv_head = q_head / gqa`.
    /// `q` is `[n_q_heads, head_dim]`; `k`/`v` are `[n_kv_heads, n_kv, head_dim]`.
    pub(crate) fn cpu_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        n_kv: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; n_q_heads * head_dim];
        for h in 0..n_q_heads {
            let kv_h = h / gqa;
            let mut scores = vec![0.0f32; n_kv];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[h * head_dim + d] * k[kv_h * n_kv * head_dim + j * head_dim + d];
                }
                *sc = dot * scale;
            }
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut e: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for ej in e.iter_mut() {
                *ej *= inv;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &ej) in e.iter().enumerate() {
                    acc += ej * v[kv_h * n_kv * head_dim + j * head_dim + d];
                }
                out[h * head_dim + d] = acc;
            }
        }
        out
    }

    /// Shared setup for the `(q, k, v, out, head_dim, n_kv, gqa_factor, scale)`
    /// vector kernels. Grid `[n_q_heads, 1, 1]`, tpg 1024, Reduction mode.
    pub(crate) fn sdpa_setup(
        kernel: Kernel,
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        dt: DType,
    ) -> TestSetup {
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_f: Vec<f32> =
            (0..n_q_heads * head_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
        let kv_len = n_kv_heads * n_kv * head_dim;
        let k_f: Vec<f32> = (0..kv_len).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
        let v_f: Vec<f32> = (0..kv_len).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
        let q = unpack_f32(&pack_f32(&q_f, dt), dt);
        let k = unpack_f32(&pack_f32(&k_f, dt), dt);
        let v = unpack_f32(&pack_f32(&v_f, dt), dt);
        let expected = cpu_sdpa(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q_f, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k_f, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v_f, dt), dt))
            .input(TestBuffer::zeros("out", n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("gqa_factor", gqa_factor as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_vector(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_vector::kernel_ir_for(dt), 128, 64, 8, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_vector_d64(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_vector_d64::kernel_ir_for(dt), 64, 64, 8, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_vector_d96(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_vector_d96::kernel_ir_for(dt), 96, 64, 8, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_vector_d192(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_vector_d192::kernel_ir_for(dt), 192, 64, 8, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_vector_d256(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_vector_d256::kernel_ir_for(dt), 256, 64, 8, 2, dt)
    }
}

/// New-syntax benchmarks for the decode-step attention vector kernels.
/// Production decode shape: n_q_heads=32, gqa_factor=4, n_kv=4096.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::{
        mt_sdpa_vector,
        mt_sdpa_vector_d64,
        mt_sdpa_vector_d96,
        mt_sdpa_vector_d192,
        mt_sdpa_vector_d256,
    };

    fn sb(kernel: Kernel, head_dim: usize, dt: DType) -> BenchSetup {
        let (n_kv, n_q_heads, gqa_factor) = (4096usize, 32usize, 4usize);
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_len = n_kv_heads * n_kv * head_dim;
        // Dominant traffic: streaming K and V once each.
        let bytes = 2 * kv_len * dt.size_bytes() + 2 * n_q_heads * head_dim * dt.size_bytes();
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", kv_len, dt))
            .buffer(BenchBuffer::random("v", kv_len, dt))
            .buffer(BenchBuffer::zeros("out", n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("gqa_factor", gqa_factor as u32)
            .constexpr("scale", scale)
            .with_shape_label(format!(
                "h{head_dim} kv{n_kv} nq{n_q_heads} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/sdpa/vector_d128", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_vector(dt: DType) -> BenchSetup { sb(mt_sdpa_vector::kernel_ir_for(dt), 128, dt) }
    #[bench(name = "mlx/sdpa/vector_d64", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_vector_d64(dt: DType) -> BenchSetup {
        sb(mt_sdpa_vector_d64::kernel_ir_for(dt), 64, dt)
    }
    #[bench(name = "mlx/sdpa/vector_d96", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_vector_d96(dt: DType) -> BenchSetup {
        sb(mt_sdpa_vector_d96::kernel_ir_for(dt), 96, dt)
    }
    #[bench(name = "mlx/sdpa/vector_d192", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_vector_d192(dt: DType) -> BenchSetup {
        sb(mt_sdpa_vector_d192::kernel_ir_for(dt), 192, dt)
    }
    #[bench(name = "mlx/sdpa/vector_d256", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_vector_d256(dt: DType) -> BenchSetup {
        sb(mt_sdpa_vector_d256::kernel_ir_for(dt), 256, dt)
    }
}
