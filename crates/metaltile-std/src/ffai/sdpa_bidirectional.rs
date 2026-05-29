//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-query bidirectional SDPA — attends `n_query` query rows
//! against a shared K/V cache in a single dispatch, with every query
//! attending the full `[0, base_kv + n_query)` range (no causal mask).
//!
//! This is the variant needed by Vision-Language tower encoders
//! (SigLIP / CLIP / FastViT / PaliGemma SigLIP-So400m) — their
//! attention is bidirectional across all image patches, not causal.
//! `sdpa_multi` already supports the same shape under `causal == 0`,
//! but it is hardcoded to head_dim=128. Common vision tower head_dims
//! are 64 (SigLIP-base/large, CLIP-L), 32 (FastViT-HD), 72
//! (PaliGemma's SigLIP-So400m), 80 (DiNOv2 / DinoV3), and 96
//! (Idefics3, Phi-3.5-vision).
//!
//! ## Naming
//!
//! `ffai_sdpa_bidirectional_dN<T>` — N is the constexpr head_dim,
//! T is the element type (f32 / f16 / bf16).
//!
//! Two per-lane layout families:
//!
//!   - **N ∈ {32, 64, 128, 256, 512}** — `N / 32` elements per lane,
//!     loaded unconditionally at `lane * (N/32) + {0..(N/32 - 1)}`.
//!     Every lane participates in the dot product; no bounds masking.
//!   - **N = 72** — *ragged* layout, `ceil(72 / 32) = 3` elements per
//!     lane at `lane * 3 + {0, 1, 2}`. Lanes 0..23 own the 72 valid
//!     indices (24 × 3 = 72); lanes 24..31 fall entirely out of range
//!     and are bounds-masked into idle (their `q*k` contribution is 0
//!     and their output store is skipped). 25 % lane-occupancy loss.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernels — STRICT threadgroup geometry. Wrappers MUST
//! encode these as preconditions; the same machine-freeze hazard as
//! `ffai_sdpa_decode` / `ffai_sdpa_multi`.
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Hard. A TPG
//!   below 32 makes `n_simd = TPG / 32 = 0`, turning the K walk
//!   `range(sg, n_kv, 0)` into an infinite GPU loop — the freeze.
//! - **`head_dim == N`** (the value baked into the kernel name).
//!   For N ∈ {32, 64} each lane owns N/32 elements unconditionally.
//!   For N = 72 each lane owns 3 elements with per-lane bounds masking
//!   on indices ≥ head_dim.
//! - **Grid: 1 threadgroup per (query, q_head).** `tgid_x` ranges
//!   `[0, n_q_heads * n_query)`; decoded `query = tgid / n_q_heads`,
//!   `q_head = tgid % n_q_heads`. Wrapper dispatches
//!   `grid = (n_q_heads * n_query * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`n_q_heads % heads_per_group == 0`** for integer GQA fan-out.
//! - **`base_kv + n_query <= kv_stride`** — the kernel never walks
//!   past the cache's allocated depth.
//!
//! Q / `out` layout: `[n_query, n_q_heads, head_dim]` row-major.
//! K / V layout:     `[n_kv_heads, kv_stride, head_dim]` row-major.
//! Online softmax runs in fp32 throughout (storage stays in T).

use metaltile::kernel;

// ─── head_dim = 64 (SigLIP base/large, CLIP-L) ─────────────────────

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_bidirectional_d64",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_bidirectional_d64<T>(
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
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // No `causal` branch — every query attends the full block.
    let n_kv = base_kv + n_query;
    // Two tg_out slots at head_dim=64; 2 elements per lane.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
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
    // Same `+1` padded stride as `sdpa_multi` so adjacent lanes hit
    // distinct threadgroup-memory banks.
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

// ─── head_dim = 32 (FastViT-HD) ────────────────────────────────────

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_bidirectional_d32",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_bidirectional_d32<T>(
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
    // Single tg_out slot at head_dim=32; 1 element per lane.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane;
    // Pre-scale this lane's single Q element once; K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let k0_raw = load(k[kv0]);
        let k0 = k0_raw.cast::<f32>();
        let partial = q0 * k0;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v0 = v0_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
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
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
    }
}

// ─── head_dim = 72 (PaliGemma SigLIP-So400m) ──────────────────────
//
// Ragged 3-elements-per-lane layout. Lanes 0..23 cover the 72 valid
// indices (24 * 3 = 72). Lanes 24..31 have all three element indices
// >= head_dim, so every load is mask-zeroed and the per-element store
// at the end is gated by `if d? < head_dim`. The result is that those
// 8 lanes contribute 0 to `simd_sum` (correctly) and skip the output
// store — but they still participate in the simdgroup-collective
// `simd_sum` / `simd_max`, which is what makes the geometry valid on
// a 32-wide simdgroup.

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_bidirectional_d72",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_bidirectional_d72<T>(
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
    // Three tg_out slots at head_dim=72; 3 elements per lane (lanes
    // 24..31 are idle — their per-element bounds checks fail).
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 3u32;
    let d1 = d0 + 1u32;
    let d2 = d0 + 2u32;
    // Per-element bounds masks. Clamp indices to a safe in-buffer
    // location (0) so the load itself never reads OOB; the select on
    // the result zeros the lane's contribution to the dot product.
    let d0s = select(d0 < head_dim, d0, 0u32);
    let d1s = select(d1 < head_dim, d1, 0u32);
    let d2s = select(d2 < head_dim, d2, 0u32);
    // Pre-scale this lane's 3-element Q triplet; mask OOB to 0.
    let q0 = select(d0 < head_dim, load(q[q_off + d0s]).cast::<f32>() * scale, 0.0f32);
    let q1 = select(d1 < head_dim, load(q[q_off + d1s]).cast::<f32>() * scale, 0.0f32);
    let q2 = select(d2 < head_dim, load(q[q_off + d2s]).cast::<f32>() * scale, 0.0f32);
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        // Same clamp-then-mask pattern as Q.
        let k0 = select(d0 < head_dim, load(k[base + d0s]).cast::<f32>(), 0.0f32);
        let k1 = select(d1 < head_dim, load(k[base + d1s]).cast::<f32>(), 0.0f32);
        let k2 = select(d2 < head_dim, load(k[base + d2s]).cast::<f32>(), 0.0f32);
        let partial = q0 * k0 + q1 * k1 + q2 * k2;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = select(d0 < head_dim, load(v[base + d0s]).cast::<f32>(), 0.0f32);
        let v1 = select(d1 < head_dim, load(v[base + d1s]).cast::<f32>(), 0.0f32);
        let v2 = select(d2 < head_dim, load(v[base + d2s]).cast::<f32>(), 0.0f32);
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
    }
    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0u32 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0u32 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0u32 {
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
    // Lanes 24..31 write 0 to their slots — harmless: those slots are
    // only ever read back by sg==0 for the same lane, which gates the
    // final store on `d? < head_dim` so the zeros never reach `out`.
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_barrier();
    if sg == 0u32 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
        }
        let out_off = q_off + d0;
        // Per-element gate prevents OOB writes for lanes 24..31.
        if d0 < head_dim {
            store(out[out_off], so0.cast::<T>());
        }
        if d1 < head_dim {
            store(out[out_off + 1u32], so1.cast::<T>());
        }
        if d2 < head_dim {
            store(out[out_off + 2u32], so2.cast::<T>());
        }
    }
}

// ─── head_dim = 80 (DiNOv2 / DiNOv3) ───────────────────────────────
//
// Same ragged 3-elements-per-lane layout as d72: lanes 0..25 cover
// 78 indices (26 * 3 = 78), lane 26 covers two more (78, 79) and one
// OOB index (80) gated by `d2 < head_dim`, lanes 27..31 are fully
// idle. The kernel body is identical to d72 — only the constexpr
// `head_dim` differs at the call site, and that's already gated by
// the same per-element bounds masks.

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_bidirectional_d80",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_sdpa_bidirectional_d80<T>(
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
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 3u32;
    let d1 = d0 + 1u32;
    let d2 = d0 + 2u32;
    let d0s = select(d0 < head_dim, d0, 0u32);
    let d1s = select(d1 < head_dim, d1, 0u32);
    let d2s = select(d2 < head_dim, d2, 0u32);
    let q0 = select(d0 < head_dim, load(q[q_off + d0s]).cast::<f32>() * scale, 0.0f32);
    let q1 = select(d1 < head_dim, load(q[q_off + d1s]).cast::<f32>() * scale, 0.0f32);
    let q2 = select(d2 < head_dim, load(q[q_off + d2s]).cast::<f32>() * scale, 0.0f32);
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let k0 = select(d0 < head_dim, load(k[base + d0s]).cast::<f32>(), 0.0f32);
        let k1 = select(d1 < head_dim, load(k[base + d1s]).cast::<f32>(), 0.0f32);
        let k2 = select(d2 < head_dim, load(k[base + d2s]).cast::<f32>(), 0.0f32);
        let partial = q0 * k0 + q1 * k1 + q2 * k2;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = select(d0 < head_dim, load(v[base + d0s]).cast::<f32>(), 0.0f32);
        let v1 = select(d1 < head_dim, load(v[base + d1s]).cast::<f32>(), 0.0f32);
        let v2 = select(d2 < head_dim, load(v[base + d2s]).cast::<f32>(), 0.0f32);
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
    }
    if lane == 0u32 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0u32 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0u32 {
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
    threadgroup_barrier();
    if sg == 0u32 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
        }
        let out_off = q_off + d0;
        if d0 < head_dim {
            store(out[out_off], so0.cast::<T>());
        }
        if d1 < head_dim {
            store(out[out_off + 1u32], so1.cast::<T>());
        }
        if d2 < head_dim {
            store(out[out_off + 2u32], so2.cast::<T>());
        }
    }
}

// ─── head_dim = 96 (Idefics3, Phi-3.5-vision) ──────────────────────
//
// 3 elements per lane × 32 lanes = 96 exactly. No bounds masks, no
// idle lanes — same clean layout as d64 just with one extra element
// per lane. Distinct from d72 because that variant has the ragged
// 24-active-lane geometry; here every lane participates.

#[kernel(
    bench(
        op = "sdpa",
        subop = "sdpa_bidirectional_d96",
        class = GenericEmpty,
        tol = 1e-3,
        kernel_mode = Reduction,
    )
)]
pub fn ffai_sdpa_bidirectional_d96<T>(
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
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 3u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let kv2 = kv_idx + 2u32;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
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
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{ffai_sdpa_bidirectional_d64, ffai_sdpa_bidirectional_d96};
    use crate::utils::{pack_f32, unpack_f32};

    // Per (query, q_head): softmax(Q·Kᵀ·scale)·V over `[0, base_kv +
    // n_query)`. Q/out layout `[n_query, n_q_heads, head_dim]`, K/V layout
    // `[n_kv_heads, kv_stride, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_bidirectional(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base_kv: usize,
        n_query: usize,
        kv_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let n_kv = base_kv + n_query;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
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

    // Shared shape (small): base_kv=56, n_query=8 → kv_stride=64.
    fn setup(ir: metaltile::core::ir::Kernel, head_dim: usize, dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads) = (8usize, 4usize);
        let (base_kv, n_query) = (56usize, 8usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa_bidirectional(
            &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
        );

        TestSetup::new(ir)
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
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_bidirectional_d64(dt: DType) -> TestSetup {
        setup(ffai_sdpa_bidirectional_d64::kernel_ir_for(dt), 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_bidirectional_d96(dt: DType) -> TestSetup {
        setup(ffai_sdpa_bidirectional_d96::kernel_ir_for(dt), 96, dt)
    }
}

/// New-syntax benchmarks for the bidirectional SDPA family (all head dims,
/// `class=GenericEmpty`). Vision-tower decode shape: base_kv=4096 prefix,
/// 8-query block, GQA fan-out 4.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        ffai_sdpa_bidirectional_d32,
        ffai_sdpa_bidirectional_d64,
        ffai_sdpa_bidirectional_d72,
        ffai_sdpa_bidirectional_d80,
        ffai_sdpa_bidirectional_d96,
    };

    fn setup(ir: metaltile::core::ir::Kernel, head_dim: usize, dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads) = (32usize, 8usize);
        let (base_kv, n_query) = (4096usize, 8usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv = base_kv + n_query;
        let bytes = (2 * n_query * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ir)
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
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_bidirectional_d32", dtypes = [f32, f16, bf16])]
    fn bench_d32(dt: DType) -> BenchSetup {
        setup(ffai_sdpa_bidirectional_d32::kernel_ir_for(dt), 32, dt)
    }

    #[bench(name = "ffai/sdpa_bidirectional_d64", dtypes = [f32, f16, bf16])]
    fn bench_d64(dt: DType) -> BenchSetup {
        setup(ffai_sdpa_bidirectional_d64::kernel_ir_for(dt), 64, dt)
    }

    #[bench(name = "ffai/sdpa_bidirectional_d72", dtypes = [f32, f16, bf16])]
    fn bench_d72(dt: DType) -> BenchSetup {
        setup(ffai_sdpa_bidirectional_d72::kernel_ir_for(dt), 72, dt)
    }

    #[bench(name = "ffai/sdpa_bidirectional_d80", dtypes = [f32, f16, bf16])]
    fn bench_d80(dt: DType) -> BenchSetup {
        setup(ffai_sdpa_bidirectional_d80::kernel_ir_for(dt), 80, dt)
    }

    #[bench(name = "ffai/sdpa_bidirectional_d96", dtypes = [f32, f16, bf16])]
    fn bench_d96(dt: DType) -> BenchSetup {
        setup(ffai_sdpa_bidirectional_d96::kernel_ir_for(dt), 96, dt)
    }
}
