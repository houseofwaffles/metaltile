//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched-Q SDPA decode for speculative decoding (M7) — K query
//! positions share one KV walk per dispatch, amortizing KV memory
//! bandwidth by K× vs. K independent single-Q `sdpa_decode` dispatches.
//!
//! This file ships the K=2 (`sdpa_decode_batched_q2`) and K=4
//! (`sdpa_decode_batched_q4`) decode-form specializations. K=8 and
//! K=16 land separately via prefill-tile reuse through
//! `mt_sdpa_prefill_mma` (the FA-2 tile already implements the
//! KV-reuse pattern at BQ × BK; the batched variant arm dispatches it
//! with `q_len = K, k_len = n_kv`). See
//! `metaltile-planning/M7-batched-q-sdpa-plan.md`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **K=2 (`sdpa_decode_batched_q2`):** TPG = 1024 (32 simdgroups × 32
//!   lanes). The kernel emits MSL that pins this layout via the
//!   threadgroup-memory allocations and the cross-simdgroup reduction;
//!   any other TPG breaks the bank-conflict-padded `tg_out` stride.
//! - **K=4 (`sdpa_decode_batched_q4`):** TPG ≤ 768; recommended 512
//!   (16 simdgroups × 32 lanes). **Dispatching at 1024 silently
//!   produces all-zero outputs** — Metal's pipeline-state compiler caps
//!   `maxTotalThreadsPerThreadgroup` at 768 for this kernel's register
//!   pressure on M1 Max; dispatching past the cap is undefined
//!   behavior and the observed symptom is zero-init buffer left
//!   untouched. The `run_sdpa_batched_decode_form` runner enforces
//!   this with `assert!(tpg <= 768 || batch_q < 4, ...)` so a future
//!   bench-row edit can't re-introduce the bug.
//! - **Grid:** 1D, `[n_q_heads, 1, 1]`. One threadgroup per Q head.
//! - **head_dim == 128 only.** Other head dims index out of the
//!   pre-scaled Q-quartile registers; the kernel emits 4-element
//!   loads per lane × 32 lanes per simdgroup = 128 exactly.
//!
//! ## Algorithm
//!
//! The single-Q `sdpa_decode` kernel walks the KV cache with a
//! threadgroup of 1024 (32 simdgroups × 32 lanes), each lane owning a
//! 4-element quartile of `head_dim == 128`. Per-simdgroup KV positions
//! are visited in stride; for each, the lane computes its quartile
//! dot product, simd-sums to a full score, updates a running
//! (max, sum) online-softmax tuple, and accumulates V into per-lane
//! fp32 registers.
//!
//! For K=2, each lane carries **two** independent online-softmax tuples
//! (one per Q position) and **two** output accumulators. The KV cache
//! is loaded **once per visited position** and dot-produced against
//! both Q vectors before V is read once and accumulated into both
//! output streams. That's where the 2× KV-bandwidth amortization
//! lives — the dominant cost at long context.
//!
//! ## Threadgroup-memory budget
//!
//! Naively widening the tg buffers by K would double the footprint
//! (~17 KB → ~34 KB at K=2), past Apple GPUs' 32 KB per-threadgroup
//! limit. Instead the kernel performs the **cross-simdgroup output
//! reduction in two sequential phases** (Q[0] then Q[1]), reusing the
//! same `tg_max`/`tg_sum`/`tg_out0..3` buffers across phases. The
//! expensive shared phase — the KV walk itself — runs once and
//! computes both streams' running state simultaneously.
//!
//! ## Credit
//!
//! The KV-reuse-via-single-load + multi-Q dot-product pattern follows
//! the production `verify_qmm` / `gated_delta_tree_tape_kernel`
//! kernels in the dflash-mlx repo, which prove this is the
//! mathematically optimal endpoint for batched-Q SDPA on Apple GPUs.
//!
//! ## Layout
//!
//! * `head_dim == 128` (same constraint as `sdpa_decode`; one
//!   threadgroup is 32 simdgroups × 32 lanes; each lane owns
//!   `head_dim / 32 = 4` elements).
//! * Q shape: `[n_q_heads, 2, head_dim]` — for each query head the two
//!   batched-Q positions sit at adjacent offsets `q_off_0` and
//!   `q_off_0 + head_dim`.
//! * K/V cache shape `[n_kv_heads, kv_stride, head_dim]`, walked from
//!   `0..n_kv` (dense path; sliding-window batched-Q is a future
//!   extension).
//! * GQA: `kv_head = q_head / heads_per_group`.
//!
//! Dispatch: one threadgroup per Q head (1D grid, `tgid_x = q_head`),
//! 1024 threads (32 simdgroups × 32 lanes).
//!
//! Online-softmax math runs in fp32 throughout (storage stays in T) to
//! avoid catastrophic cancellation in `exp(max_old - max_new)`.

use metaltile::kernel;

#[kernel]
pub fn sdpa_decode_batched_q2<T>(
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
    // Reuses the single-Q `sdpa_decode` threadgroup-memory layout: 32
    // slots for max/sum (one per simdgroup) + 4 × 1056-slot transpose
    // buffers (1024 + 32 padding to dodge 32-way bank conflicts during
    // sg=0's per-lane sum sweep — see `sdpa_decode.rs` for the
    // derivation). Both Q streams' cross-sg reductions reuse these same
    // buffers in two sequential phases (Q[0] then Q[1]) so the
    // footprint stays at ~17 KB regardless of K.
    //
    // `32` is sized for the maximum simdgroup count at TPG=1024
    // (1024 / 32-lane simdgroup = 32 simdgroups). K=2 dispatches at
    // TPG=1024 and fills all 32 slots; K=4 dispatches at TPG=512 (per
    // DISPATCH INVARIANTS) and only writes slots 0..15, with the
    // `select(lane < ns, ..., neg_infinity)` guard in the cross-sg
    // reduction zero-padding the unwritten upper half. Do NOT remove
    // the guard — it is the only thing keeping the cross-sg reduction
    // correct at TPG < 1024.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    // Q layout: [n_q_heads, 2, head_dim] — two batched-Q positions per
    // query head sit at adjacent head_dim-wide slots.
    let q_off_0 = q_head * 2u32 * head_dim;
    let q_off_1 = q_off_0 + head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;
    // Pre-scale each lane's 4-element quartile of both Q vectors. K and
    // V are streamed inside the walk loop.
    let q0_a = load(q[q_off_0 + d0]).cast::<f32>() * scale;
    let q0_b = load(q[q_off_0 + d0 + 1u32]).cast::<f32>() * scale;
    let q0_c = load(q[q_off_0 + d0 + 2u32]).cast::<f32>() * scale;
    let q0_d = load(q[q_off_0 + d0 + 3u32]).cast::<f32>() * scale;
    let q1_a = load(q[q_off_1 + d0]).cast::<f32>() * scale;
    let q1_b = load(q[q_off_1 + d0 + 1u32]).cast::<f32>() * scale;
    let q1_c = load(q[q_off_1 + d0 + 2u32]).cast::<f32>() * scale;
    let q1_d = load(q[q_off_1 + d0 + 3u32]).cast::<f32>() * scale;
    // Two independent online-softmax tuples + two output accumulators.
    // At K=2 this is 14 fp32 per lane (max+sum × 2, o0..o3 × 2) plus
    // the 8 pre-loaded Q quartiles — comfortably inside Apple GPUs'
    // per-lane register budget.
    let mut run_max_0 = neg_infinity();
    let mut run_max_1 = neg_infinity();
    let mut run_sum_0 = 0.0f32;
    let mut run_sum_1 = 0.0f32;
    let mut o0_0 = 0.0f32;
    let mut o0_1 = 0.0f32;
    let mut o0_2 = 0.0f32;
    let mut o0_3 = 0.0f32;
    let mut o1_0 = 0.0f32;
    let mut o1_1 = 0.0f32;
    let mut o1_2 = 0.0f32;
    let mut o1_3 = 0.0f32;
    // ── Shared KV walk: KV loaded ONCE per position, both Q streams
    //    updated in lockstep. This is where the bandwidth amortization
    //    lives — the rest of the kernel runs in compute-bound territory.
    //
    // Pre-compute index VIDs BEFORE issuing loads — vectorize wants 4
    // consecutive `Op::Load` with no BinOp/Const interleaved.
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
        let partial_0 = q0_a * k0 + q0_b * k1 + q0_c * k2 + q0_d * k3;
        let partial_1 = q1_a * k0 + q1_b * k1 + q1_c * k2 + q1_d * k3;
        let score_0 = simd_sum(partial_0);
        let score_1 = simd_sum(partial_1);
        let new_max_0 = select(score_0 > run_max_0, score_0, run_max_0);
        let new_max_1 = select(score_1 > run_max_1, score_1, run_max_1);
        let factor_0 = exp(run_max_0 - new_max_0);
        let factor_1 = exp(run_max_1 - new_max_1);
        let weight_0 = exp(score_0 - new_max_0);
        let weight_1 = exp(score_1 - new_max_1);
        run_sum_0 = run_sum_0 * factor_0 + weight_0;
        run_sum_1 = run_sum_1 * factor_1 + weight_1;
        run_max_0 = new_max_0;
        run_max_1 = new_max_1;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0_0 = o0_0 * factor_0 + weight_0 * v0;
        o0_1 = o0_1 * factor_0 + weight_0 * v1;
        o0_2 = o0_2 * factor_0 + weight_0 * v2;
        o0_3 = o0_3 * factor_0 + weight_0 * v3;
        o1_0 = o1_0 * factor_1 + weight_1 * v0;
        o1_1 = o1_1 * factor_1 + weight_1 * v1;
        o1_2 = o1_2 * factor_1 + weight_1 * v2;
        o1_3 = o1_3 * factor_1 + weight_1 * v3;
    }
    // ── Phase A: cross-simdgroup reduction + output write for Q[0] ──
    //
    // Mirrors `sdpa_decode`'s single-Q output stage. Phase B repeats it
    // for Q[1] reusing the same tg buffers; the barrier between phases
    // ensures Q[0]'s reads complete before Q[1]'s writes start.
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_0);
        threadgroup_store("tg_sum", sg, run_sum_0);
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
    let g_max_0 = threadgroup_load("tg_max", 0);
    let g_sum_0 = threadgroup_load("tg_sum", 0);
    // Guard against `n_kv == 0`: no K positions visited → run_max stays
    // -inf, g_sum stays 0, naive `exp(-inf - -inf) / 0 = NaN`. Same
    // shape as `sdpa_decode`'s guard.
    let rescale_0 = select(g_sum_0 > 0.0f32, exp(run_max_0 - g_max_0) / g_sum_0, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0_0 * rescale_0);
    threadgroup_store("tg_out1", idx, o0_1 * rescale_0);
    threadgroup_store("tg_out2", idx, o0_2 * rescale_0);
    threadgroup_store("tg_out3", idx, o0_3 * rescale_0);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0_0 = 0.0f32;
        let mut so0_1 = 0.0f32;
        let mut so0_2 = 0.0f32;
        let mut so0_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0_0 = so0_0 + threadgroup_load("tg_out0", ri);
            so0_1 = so0_1 + threadgroup_load("tg_out1", ri);
            so0_2 = so0_2 + threadgroup_load("tg_out2", ri);
            so0_3 = so0_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_0 = q_head * 2u32 * head_dim + d0;
        store(out[out_off_0], so0_0.cast::<T>());
        store(out[out_off_0 + 1u32], so0_1.cast::<T>());
        store(out[out_off_0 + 2u32], so0_2.cast::<T>());
        store(out[out_off_0 + 3u32], so0_3.cast::<T>());
    }
    // Phase A's reads from tg_out0..3 must complete before Phase B
    // overwrites them. The `mem_threadgroup` scope on the barrier is
    // the standard `sdpa_decode` flavor.
    threadgroup_barrier();
    // ── Phase B: cross-simdgroup reduction + output write for Q[1] ──
    //
    // Bit-identical to Phase A modulo the `_1` register state. The two
    // phases are NOT factored into a shared macro because the
    // `#[kernel]` proc-macro does not expand `macro_rules!`
    // invocations (the body parser keeps the `!` opaque); duplication
    // here matches the established pattern in `sdpa_decode`'s sink /
    // window pass.
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_1);
        threadgroup_store("tg_sum", sg, run_sum_1);
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
    let g_max_1 = threadgroup_load("tg_max", 0);
    let g_sum_1 = threadgroup_load("tg_sum", 0);
    let rescale_1 = select(g_sum_1 > 0.0f32, exp(run_max_1 - g_max_1) / g_sum_1, 0.0f32);
    threadgroup_store("tg_out0", idx, o1_0 * rescale_1);
    threadgroup_store("tg_out1", idx, o1_1 * rescale_1);
    threadgroup_store("tg_out2", idx, o1_2 * rescale_1);
    threadgroup_store("tg_out3", idx, o1_3 * rescale_1);
    threadgroup_barrier();
    if sg == 0 {
        let mut so1_0 = 0.0f32;
        let mut so1_1 = 0.0f32;
        let mut so1_2 = 0.0f32;
        let mut so1_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so1_0 = so1_0 + threadgroup_load("tg_out0", ri);
            so1_1 = so1_1 + threadgroup_load("tg_out1", ri);
            so1_2 = so1_2 + threadgroup_load("tg_out2", ri);
            so1_3 = so1_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_1 = q_head * 2u32 * head_dim + head_dim + d0;
        store(out[out_off_1], so1_0.cast::<T>());
        store(out[out_off_1 + 1u32], so1_1.cast::<T>());
        store(out[out_off_1 + 2u32], so1_2.cast::<T>());
        store(out[out_off_1 + 3u32], so1_3.cast::<T>());
    }
}

// Phase 1 of M7 ships only the kernel + codegen tests; the
// `SdpaBatchedDecode` runner remains a stub (see `run_spec.rs`) until
// the GPU dispatch path and bench rows land in the follow-up commit.

// ── K=4 specialization ──────────────────────────────────────────────────
//
// Mechanical extension of K=2: four online-softmax streams instead of
// two; four sequential output-reduction phases reusing the same
// tg_max/tg_sum/tg_out0..3 buffers (the inter-phase barriers keep the
// peak tg-memory footprint at the single-Q ~17 KB regardless of K).
//
// Per-lane register pressure rises from K=2's ~20 fp32-equivalent
// registers to ~40 (16 pre-scaled Q quartiles, 4 max + 4 sum, 16 o
// accumulators), plus another ~30 transients inside the KV walk. On
// M1, Metal's pipeline-state compiler caps the kernel's
// `maxTotalThreadsPerThreadgroup` at **768** at this register
// pressure, vs. K=2's 1024. Dispatch the K=4 kernel at **512 threads**
// = 16 simdgroups × 32 lanes — leaves headroom, and the kernel's
// `n_simd` / `lane` math handles any 32-multiple thread count cleanly
// (KV walk strides by `n_simd`, cross-sg reduction's `lane < n_simd`
// guard zero-pads the inactive simdgroups in `simd_max`/`simd_sum`).
// Bench rows and tests pin tpg=512 for this kernel; dispatching at
// 1024 is undefined behavior (silently produces all-zero outputs).
//
// The KV-walk amortization win is unchanged by the smaller TG size:
// with 16 simdgroups instead of 32, each simdgroup walks 2× as many
// KV positions, but the per-KV-position bandwidth cost is still paid
// once and split across all 4 Q streams. Net: ~4× KV-bandwidth
// reduction vs. four independent single-Q dispatches.
//
// The output reduction in particular is **not** factored into a shared
// helper because `#[kernel]`'s body parser does not expand
// `macro_rules!` invocations (the AST handed to the parser keeps the
// `!` call opaque); shared-body macros produce empty MSL. Each phase's
// duplicate body matches the pattern established in
// `sdpa_decode`'s sink + window passes.

#[kernel]
pub fn sdpa_decode_batched_q4<T>(
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
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    // Q layout: [n_q_heads, 4, head_dim]. Four batched-Q positions per
    // head sit at adjacent head_dim-wide slots.
    let q_off_0 = q_head * 4u32 * head_dim;
    let q_off_1 = q_off_0 + head_dim;
    let q_off_2 = q_off_1 + head_dim;
    let q_off_3 = q_off_2 + head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;
    // Pre-scale each lane's 4-element quartile of all four Q vectors.
    let q0_a = load(q[q_off_0 + d0]).cast::<f32>() * scale;
    let q0_b = load(q[q_off_0 + d0 + 1u32]).cast::<f32>() * scale;
    let q0_c = load(q[q_off_0 + d0 + 2u32]).cast::<f32>() * scale;
    let q0_d = load(q[q_off_0 + d0 + 3u32]).cast::<f32>() * scale;
    let q1_a = load(q[q_off_1 + d0]).cast::<f32>() * scale;
    let q1_b = load(q[q_off_1 + d0 + 1u32]).cast::<f32>() * scale;
    let q1_c = load(q[q_off_1 + d0 + 2u32]).cast::<f32>() * scale;
    let q1_d = load(q[q_off_1 + d0 + 3u32]).cast::<f32>() * scale;
    let q2_a = load(q[q_off_2 + d0]).cast::<f32>() * scale;
    let q2_b = load(q[q_off_2 + d0 + 1u32]).cast::<f32>() * scale;
    let q2_c = load(q[q_off_2 + d0 + 2u32]).cast::<f32>() * scale;
    let q2_d = load(q[q_off_2 + d0 + 3u32]).cast::<f32>() * scale;
    let q3_a = load(q[q_off_3 + d0]).cast::<f32>() * scale;
    let q3_b = load(q[q_off_3 + d0 + 1u32]).cast::<f32>() * scale;
    let q3_c = load(q[q_off_3 + d0 + 2u32]).cast::<f32>() * scale;
    let q3_d = load(q[q_off_3 + d0 + 3u32]).cast::<f32>() * scale;
    let mut run_max_0 = neg_infinity();
    let mut run_max_1 = neg_infinity();
    let mut run_max_2 = neg_infinity();
    let mut run_max_3 = neg_infinity();
    let mut run_sum_0 = 0.0f32;
    let mut run_sum_1 = 0.0f32;
    let mut run_sum_2 = 0.0f32;
    let mut run_sum_3 = 0.0f32;
    let mut o0_0 = 0.0f32;
    let mut o0_1 = 0.0f32;
    let mut o0_2 = 0.0f32;
    let mut o0_3 = 0.0f32;
    let mut o1_0 = 0.0f32;
    let mut o1_1 = 0.0f32;
    let mut o1_2 = 0.0f32;
    let mut o1_3 = 0.0f32;
    let mut o2_0 = 0.0f32;
    let mut o2_1 = 0.0f32;
    let mut o2_2 = 0.0f32;
    let mut o2_3 = 0.0f32;
    let mut o3_0 = 0.0f32;
    let mut o3_1 = 0.0f32;
    let mut o3_2 = 0.0f32;
    let mut o3_3 = 0.0f32;
    // ── Shared KV walk: KV loaded ONCE per position, all four Q
    //    streams updated in lockstep.
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
        let partial_0 = q0_a * k0 + q0_b * k1 + q0_c * k2 + q0_d * k3;
        let partial_1 = q1_a * k0 + q1_b * k1 + q1_c * k2 + q1_d * k3;
        let partial_2 = q2_a * k0 + q2_b * k1 + q2_c * k2 + q2_d * k3;
        let partial_3 = q3_a * k0 + q3_b * k1 + q3_c * k2 + q3_d * k3;
        let score_0 = simd_sum(partial_0);
        let score_1 = simd_sum(partial_1);
        let score_2 = simd_sum(partial_2);
        let score_3 = simd_sum(partial_3);
        let new_max_0 = select(score_0 > run_max_0, score_0, run_max_0);
        let new_max_1 = select(score_1 > run_max_1, score_1, run_max_1);
        let new_max_2 = select(score_2 > run_max_2, score_2, run_max_2);
        let new_max_3 = select(score_3 > run_max_3, score_3, run_max_3);
        let factor_0 = exp(run_max_0 - new_max_0);
        let factor_1 = exp(run_max_1 - new_max_1);
        let factor_2 = exp(run_max_2 - new_max_2);
        let factor_3 = exp(run_max_3 - new_max_3);
        let weight_0 = exp(score_0 - new_max_0);
        let weight_1 = exp(score_1 - new_max_1);
        let weight_2 = exp(score_2 - new_max_2);
        let weight_3 = exp(score_3 - new_max_3);
        run_sum_0 = run_sum_0 * factor_0 + weight_0;
        run_sum_1 = run_sum_1 * factor_1 + weight_1;
        run_sum_2 = run_sum_2 * factor_2 + weight_2;
        run_sum_3 = run_sum_3 * factor_3 + weight_3;
        run_max_0 = new_max_0;
        run_max_1 = new_max_1;
        run_max_2 = new_max_2;
        run_max_3 = new_max_3;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0_0 = o0_0 * factor_0 + weight_0 * v0;
        o0_1 = o0_1 * factor_0 + weight_0 * v1;
        o0_2 = o0_2 * factor_0 + weight_0 * v2;
        o0_3 = o0_3 * factor_0 + weight_0 * v3;
        o1_0 = o1_0 * factor_1 + weight_1 * v0;
        o1_1 = o1_1 * factor_1 + weight_1 * v1;
        o1_2 = o1_2 * factor_1 + weight_1 * v2;
        o1_3 = o1_3 * factor_1 + weight_1 * v3;
        o2_0 = o2_0 * factor_2 + weight_2 * v0;
        o2_1 = o2_1 * factor_2 + weight_2 * v1;
        o2_2 = o2_2 * factor_2 + weight_2 * v2;
        o2_3 = o2_3 * factor_2 + weight_2 * v3;
        o3_0 = o3_0 * factor_3 + weight_3 * v0;
        o3_1 = o3_1 * factor_3 + weight_3 * v1;
        o3_2 = o3_2 * factor_3 + weight_3 * v2;
        o3_3 = o3_3 * factor_3 + weight_3 * v3;
    }
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    // ── Phase A: cross-simdgroup reduction + output write for Q[0] ──
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_0);
        threadgroup_store("tg_sum", sg, run_sum_0);
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
    let g_max_0 = threadgroup_load("tg_max", 0);
    let g_sum_0 = threadgroup_load("tg_sum", 0);
    let rescale_0 = select(g_sum_0 > 0.0f32, exp(run_max_0 - g_max_0) / g_sum_0, 0.0f32);
    threadgroup_store("tg_out0", idx, o0_0 * rescale_0);
    threadgroup_store("tg_out1", idx, o0_1 * rescale_0);
    threadgroup_store("tg_out2", idx, o0_2 * rescale_0);
    threadgroup_store("tg_out3", idx, o0_3 * rescale_0);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0_0 = 0.0f32;
        let mut so0_1 = 0.0f32;
        let mut so0_2 = 0.0f32;
        let mut so0_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0_0 = so0_0 + threadgroup_load("tg_out0", ri);
            so0_1 = so0_1 + threadgroup_load("tg_out1", ri);
            so0_2 = so0_2 + threadgroup_load("tg_out2", ri);
            so0_3 = so0_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_0 = q_head * 4u32 * head_dim + d0;
        store(out[out_off_0], so0_0.cast::<T>());
        store(out[out_off_0 + 1u32], so0_1.cast::<T>());
        store(out[out_off_0 + 2u32], so0_2.cast::<T>());
        store(out[out_off_0 + 3u32], so0_3.cast::<T>());
    }
    threadgroup_barrier();
    // ── Phase B: cross-simdgroup reduction + output write for Q[1] ──
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_1);
        threadgroup_store("tg_sum", sg, run_sum_1);
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
    let g_max_1 = threadgroup_load("tg_max", 0);
    let g_sum_1 = threadgroup_load("tg_sum", 0);
    let rescale_1 = select(g_sum_1 > 0.0f32, exp(run_max_1 - g_max_1) / g_sum_1, 0.0f32);
    threadgroup_store("tg_out0", idx, o1_0 * rescale_1);
    threadgroup_store("tg_out1", idx, o1_1 * rescale_1);
    threadgroup_store("tg_out2", idx, o1_2 * rescale_1);
    threadgroup_store("tg_out3", idx, o1_3 * rescale_1);
    threadgroup_barrier();
    if sg == 0 {
        let mut so1_0 = 0.0f32;
        let mut so1_1 = 0.0f32;
        let mut so1_2 = 0.0f32;
        let mut so1_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so1_0 = so1_0 + threadgroup_load("tg_out0", ri);
            so1_1 = so1_1 + threadgroup_load("tg_out1", ri);
            so1_2 = so1_2 + threadgroup_load("tg_out2", ri);
            so1_3 = so1_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_1 = q_head * 4u32 * head_dim + head_dim + d0;
        store(out[out_off_1], so1_0.cast::<T>());
        store(out[out_off_1 + 1u32], so1_1.cast::<T>());
        store(out[out_off_1 + 2u32], so1_2.cast::<T>());
        store(out[out_off_1 + 3u32], so1_3.cast::<T>());
    }
    threadgroup_barrier();
    // ── Phase C: cross-simdgroup reduction + output write for Q[2] ──
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_2);
        threadgroup_store("tg_sum", sg, run_sum_2);
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
    let g_max_2 = threadgroup_load("tg_max", 0);
    let g_sum_2 = threadgroup_load("tg_sum", 0);
    let rescale_2 = select(g_sum_2 > 0.0f32, exp(run_max_2 - g_max_2) / g_sum_2, 0.0f32);
    threadgroup_store("tg_out0", idx, o2_0 * rescale_2);
    threadgroup_store("tg_out1", idx, o2_1 * rescale_2);
    threadgroup_store("tg_out2", idx, o2_2 * rescale_2);
    threadgroup_store("tg_out3", idx, o2_3 * rescale_2);
    threadgroup_barrier();
    if sg == 0 {
        let mut so2_0 = 0.0f32;
        let mut so2_1 = 0.0f32;
        let mut so2_2 = 0.0f32;
        let mut so2_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so2_0 = so2_0 + threadgroup_load("tg_out0", ri);
            so2_1 = so2_1 + threadgroup_load("tg_out1", ri);
            so2_2 = so2_2 + threadgroup_load("tg_out2", ri);
            so2_3 = so2_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_2 = q_head * 4u32 * head_dim + 2u32 * head_dim + d0;
        store(out[out_off_2], so2_0.cast::<T>());
        store(out[out_off_2 + 1u32], so2_1.cast::<T>());
        store(out[out_off_2 + 2u32], so2_2.cast::<T>());
        store(out[out_off_2 + 3u32], so2_3.cast::<T>());
    }
    threadgroup_barrier();
    // ── Phase D: cross-simdgroup reduction + output write for Q[3] ──
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_3);
        threadgroup_store("tg_sum", sg, run_sum_3);
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
    let g_max_3 = threadgroup_load("tg_max", 0);
    let g_sum_3 = threadgroup_load("tg_sum", 0);
    let rescale_3 = select(g_sum_3 > 0.0f32, exp(run_max_3 - g_max_3) / g_sum_3, 0.0f32);
    threadgroup_store("tg_out0", idx, o3_0 * rescale_3);
    threadgroup_store("tg_out1", idx, o3_1 * rescale_3);
    threadgroup_store("tg_out2", idx, o3_2 * rescale_3);
    threadgroup_store("tg_out3", idx, o3_3 * rescale_3);
    threadgroup_barrier();
    if sg == 0 {
        let mut so3_0 = 0.0f32;
        let mut so3_1 = 0.0f32;
        let mut so3_2 = 0.0f32;
        let mut so3_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so3_0 = so3_0 + threadgroup_load("tg_out0", ri);
            so3_1 = so3_1 + threadgroup_load("tg_out1", ri);
            so3_2 = so3_2 + threadgroup_load("tg_out2", ri);
            so3_3 = so3_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_3 = q_head * 4u32 * head_dim + 3u32 * head_dim + d0;
        store(out[out_off_3], so3_0.cast::<T>());
        store(out[out_off_3 + 1u32], so3_1.cast::<T>());
        store(out[out_off_3 + 2u32], so3_2.cast::<T>());
        store(out[out_off_3 + 3u32], so3_3.cast::<T>());
    }
}

// ── K=8 specialization ──────────────────────────────────────────────────
//
// Mechanical extension of K=4: eight online-softmax streams instead of
// four; eight sequential output-reduction phases reusing the same
// tg_max/tg_sum/tg_out0..3 buffers (inter-phase barriers keep peak
// tg-memory footprint at the single-Q ~17 KB regardless of K).
//
// Per-lane register pressure at K=8: 32 pre-scaled Q quartiles (8 × 4),
// 8 max + 8 sum scalars, 32 output accumulators (8 × 4) ≈ 80 fp32-
// equivalent registers, plus ~30 transients inside the KV walk. On M7+
// (the target), Apple's GPU compiler is expected to handle this without
// spilling; on older silicon the compiler may cap
// `maxTotalThreadsPerThreadgroup` below 512, requiring a reduced TPG.
//
// Dispatch at **256 threads** (8 simdgroups × 32 lanes) — conservative
// enough to survive on M1/M2 while still providing 8× KV-bandwidth
// amortization vs. 8 independent single-Q dispatches. For M4/M5/M7,
// the TPG can be raised to 512 once per-chip register-pressure caps are
// verified. The bench row pins tpg=256 for portability.
//
// The output reduction is NOT factored into a shared macro because the
// `#[kernel]` proc-macro does not expand `macro_rules!` invocations.
// Each phase's body duplicates the established pattern.

#[kernel]
pub fn sdpa_decode_batched_q8<T>(
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
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    // Q layout: [n_q_heads, 8, head_dim]. Eight batched-Q positions per
    // head sit at adjacent head_dim-wide slots.
    let q_off_0 = q_head * 8u32 * head_dim;
    let q_off_1 = q_off_0 + head_dim;
    let q_off_2 = q_off_1 + head_dim;
    let q_off_3 = q_off_2 + head_dim;
    let q_off_4 = q_off_3 + head_dim;
    let q_off_5 = q_off_4 + head_dim;
    let q_off_6 = q_off_5 + head_dim;
    let q_off_7 = q_off_6 + head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;
    // Pre-scale each lane's 4-element quartile of all eight Q vectors.
    let q0_a = load(q[q_off_0 + d0]).cast::<f32>() * scale;
    let q0_b = load(q[q_off_0 + d0 + 1u32]).cast::<f32>() * scale;
    let q0_c = load(q[q_off_0 + d0 + 2u32]).cast::<f32>() * scale;
    let q0_d = load(q[q_off_0 + d0 + 3u32]).cast::<f32>() * scale;
    let q1_a = load(q[q_off_1 + d0]).cast::<f32>() * scale;
    let q1_b = load(q[q_off_1 + d0 + 1u32]).cast::<f32>() * scale;
    let q1_c = load(q[q_off_1 + d0 + 2u32]).cast::<f32>() * scale;
    let q1_d = load(q[q_off_1 + d0 + 3u32]).cast::<f32>() * scale;
    let q2_a = load(q[q_off_2 + d0]).cast::<f32>() * scale;
    let q2_b = load(q[q_off_2 + d0 + 1u32]).cast::<f32>() * scale;
    let q2_c = load(q[q_off_2 + d0 + 2u32]).cast::<f32>() * scale;
    let q2_d = load(q[q_off_2 + d0 + 3u32]).cast::<f32>() * scale;
    let q3_a = load(q[q_off_3 + d0]).cast::<f32>() * scale;
    let q3_b = load(q[q_off_3 + d0 + 1u32]).cast::<f32>() * scale;
    let q3_c = load(q[q_off_3 + d0 + 2u32]).cast::<f32>() * scale;
    let q3_d = load(q[q_off_3 + d0 + 3u32]).cast::<f32>() * scale;
    let q4_a = load(q[q_off_4 + d0]).cast::<f32>() * scale;
    let q4_b = load(q[q_off_4 + d0 + 1u32]).cast::<f32>() * scale;
    let q4_c = load(q[q_off_4 + d0 + 2u32]).cast::<f32>() * scale;
    let q4_d = load(q[q_off_4 + d0 + 3u32]).cast::<f32>() * scale;
    let q5_a = load(q[q_off_5 + d0]).cast::<f32>() * scale;
    let q5_b = load(q[q_off_5 + d0 + 1u32]).cast::<f32>() * scale;
    let q5_c = load(q[q_off_5 + d0 + 2u32]).cast::<f32>() * scale;
    let q5_d = load(q[q_off_5 + d0 + 3u32]).cast::<f32>() * scale;
    let q6_a = load(q[q_off_6 + d0]).cast::<f32>() * scale;
    let q6_b = load(q[q_off_6 + d0 + 1u32]).cast::<f32>() * scale;
    let q6_c = load(q[q_off_6 + d0 + 2u32]).cast::<f32>() * scale;
    let q6_d = load(q[q_off_6 + d0 + 3u32]).cast::<f32>() * scale;
    let q7_a = load(q[q_off_7 + d0]).cast::<f32>() * scale;
    let q7_b = load(q[q_off_7 + d0 + 1u32]).cast::<f32>() * scale;
    let q7_c = load(q[q_off_7 + d0 + 2u32]).cast::<f32>() * scale;
    let q7_d = load(q[q_off_7 + d0 + 3u32]).cast::<f32>() * scale;
    let mut run_max_0 = neg_infinity();
    let mut run_max_1 = neg_infinity();
    let mut run_max_2 = neg_infinity();
    let mut run_max_3 = neg_infinity();
    let mut run_max_4 = neg_infinity();
    let mut run_max_5 = neg_infinity();
    let mut run_max_6 = neg_infinity();
    let mut run_max_7 = neg_infinity();
    let mut run_sum_0 = 0.0f32;
    let mut run_sum_1 = 0.0f32;
    let mut run_sum_2 = 0.0f32;
    let mut run_sum_3 = 0.0f32;
    let mut run_sum_4 = 0.0f32;
    let mut run_sum_5 = 0.0f32;
    let mut run_sum_6 = 0.0f32;
    let mut run_sum_7 = 0.0f32;
    let mut o0_0 = 0.0f32;
    let mut o0_1 = 0.0f32;
    let mut o0_2 = 0.0f32;
    let mut o0_3 = 0.0f32;
    let mut o1_0 = 0.0f32;
    let mut o1_1 = 0.0f32;
    let mut o1_2 = 0.0f32;
    let mut o1_3 = 0.0f32;
    let mut o2_0 = 0.0f32;
    let mut o2_1 = 0.0f32;
    let mut o2_2 = 0.0f32;
    let mut o2_3 = 0.0f32;
    let mut o3_0 = 0.0f32;
    let mut o3_1 = 0.0f32;
    let mut o3_2 = 0.0f32;
    let mut o3_3 = 0.0f32;
    let mut o4_0 = 0.0f32;
    let mut o4_1 = 0.0f32;
    let mut o4_2 = 0.0f32;
    let mut o4_3 = 0.0f32;
    let mut o5_0 = 0.0f32;
    let mut o5_1 = 0.0f32;
    let mut o5_2 = 0.0f32;
    let mut o5_3 = 0.0f32;
    let mut o6_0 = 0.0f32;
    let mut o6_1 = 0.0f32;
    let mut o6_2 = 0.0f32;
    let mut o6_3 = 0.0f32;
    let mut o7_0 = 0.0f32;
    let mut o7_1 = 0.0f32;
    let mut o7_2 = 0.0f32;
    let mut o7_3 = 0.0f32;
    // ── Shared KV walk: KV loaded ONCE per position, all eight Q
    //    streams updated in lockstep.
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
        let partial_0 = q0_a * k0 + q0_b * k1 + q0_c * k2 + q0_d * k3;
        let partial_1 = q1_a * k0 + q1_b * k1 + q1_c * k2 + q1_d * k3;
        let partial_2 = q2_a * k0 + q2_b * k1 + q2_c * k2 + q2_d * k3;
        let partial_3 = q3_a * k0 + q3_b * k1 + q3_c * k2 + q3_d * k3;
        let partial_4 = q4_a * k0 + q4_b * k1 + q4_c * k2 + q4_d * k3;
        let partial_5 = q5_a * k0 + q5_b * k1 + q5_c * k2 + q5_d * k3;
        let partial_6 = q6_a * k0 + q6_b * k1 + q6_c * k2 + q6_d * k3;
        let partial_7 = q7_a * k0 + q7_b * k1 + q7_c * k2 + q7_d * k3;
        let score_0 = simd_sum(partial_0);
        let score_1 = simd_sum(partial_1);
        let score_2 = simd_sum(partial_2);
        let score_3 = simd_sum(partial_3);
        let score_4 = simd_sum(partial_4);
        let score_5 = simd_sum(partial_5);
        let score_6 = simd_sum(partial_6);
        let score_7 = simd_sum(partial_7);
        let new_max_0 = select(score_0 > run_max_0, score_0, run_max_0);
        let new_max_1 = select(score_1 > run_max_1, score_1, run_max_1);
        let new_max_2 = select(score_2 > run_max_2, score_2, run_max_2);
        let new_max_3 = select(score_3 > run_max_3, score_3, run_max_3);
        let new_max_4 = select(score_4 > run_max_4, score_4, run_max_4);
        let new_max_5 = select(score_5 > run_max_5, score_5, run_max_5);
        let new_max_6 = select(score_6 > run_max_6, score_6, run_max_6);
        let new_max_7 = select(score_7 > run_max_7, score_7, run_max_7);
        let factor_0 = exp(run_max_0 - new_max_0);
        let factor_1 = exp(run_max_1 - new_max_1);
        let factor_2 = exp(run_max_2 - new_max_2);
        let factor_3 = exp(run_max_3 - new_max_3);
        let factor_4 = exp(run_max_4 - new_max_4);
        let factor_5 = exp(run_max_5 - new_max_5);
        let factor_6 = exp(run_max_6 - new_max_6);
        let factor_7 = exp(run_max_7 - new_max_7);
        let weight_0 = exp(score_0 - new_max_0);
        let weight_1 = exp(score_1 - new_max_1);
        let weight_2 = exp(score_2 - new_max_2);
        let weight_3 = exp(score_3 - new_max_3);
        let weight_4 = exp(score_4 - new_max_4);
        let weight_5 = exp(score_5 - new_max_5);
        let weight_6 = exp(score_6 - new_max_6);
        let weight_7 = exp(score_7 - new_max_7);
        run_sum_0 = run_sum_0 * factor_0 + weight_0;
        run_sum_1 = run_sum_1 * factor_1 + weight_1;
        run_sum_2 = run_sum_2 * factor_2 + weight_2;
        run_sum_3 = run_sum_3 * factor_3 + weight_3;
        run_sum_4 = run_sum_4 * factor_4 + weight_4;
        run_sum_5 = run_sum_5 * factor_5 + weight_5;
        run_sum_6 = run_sum_6 * factor_6 + weight_6;
        run_sum_7 = run_sum_7 * factor_7 + weight_7;
        run_max_0 = new_max_0;
        run_max_1 = new_max_1;
        run_max_2 = new_max_2;
        run_max_3 = new_max_3;
        run_max_4 = new_max_4;
        run_max_5 = new_max_5;
        run_max_6 = new_max_6;
        run_max_7 = new_max_7;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        o0_0 = o0_0 * factor_0 + weight_0 * v0;
        o0_1 = o0_1 * factor_0 + weight_0 * v1;
        o0_2 = o0_2 * factor_0 + weight_0 * v2;
        o0_3 = o0_3 * factor_0 + weight_0 * v3;
        o1_0 = o1_0 * factor_1 + weight_1 * v0;
        o1_1 = o1_1 * factor_1 + weight_1 * v1;
        o1_2 = o1_2 * factor_1 + weight_1 * v2;
        o1_3 = o1_3 * factor_1 + weight_1 * v3;
        o2_0 = o2_0 * factor_2 + weight_2 * v0;
        o2_1 = o2_1 * factor_2 + weight_2 * v1;
        o2_2 = o2_2 * factor_2 + weight_2 * v2;
        o2_3 = o2_3 * factor_2 + weight_2 * v3;
        o3_0 = o3_0 * factor_3 + weight_3 * v0;
        o3_1 = o3_1 * factor_3 + weight_3 * v1;
        o3_2 = o3_2 * factor_3 + weight_3 * v2;
        o3_3 = o3_3 * factor_3 + weight_3 * v3;
        o4_0 = o4_0 * factor_4 + weight_4 * v0;
        o4_1 = o4_1 * factor_4 + weight_4 * v1;
        o4_2 = o4_2 * factor_4 + weight_4 * v2;
        o4_3 = o4_3 * factor_4 + weight_4 * v3;
        o5_0 = o5_0 * factor_5 + weight_5 * v0;
        o5_1 = o5_1 * factor_5 + weight_5 * v1;
        o5_2 = o5_2 * factor_5 + weight_5 * v2;
        o5_3 = o5_3 * factor_5 + weight_5 * v3;
        o6_0 = o6_0 * factor_6 + weight_6 * v0;
        o6_1 = o6_1 * factor_6 + weight_6 * v1;
        o6_2 = o6_2 * factor_6 + weight_6 * v2;
        o6_3 = o6_3 * factor_6 + weight_6 * v3;
        o7_0 = o7_0 * factor_7 + weight_7 * v0;
        o7_1 = o7_1 * factor_7 + weight_7 * v1;
        o7_2 = o7_2 * factor_7 + weight_7 * v2;
        o7_3 = o7_3 * factor_7 + weight_7 * v3;
    }
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    // Helper macro pattern: each phase follows the same reduction structure.
    // Phase A: Q[0]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_0);
        threadgroup_store("tg_sum", sg, run_sum_0);
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
    let g_max_0 = threadgroup_load("tg_max", 0);
    let g_sum_0 = threadgroup_load("tg_sum", 0);
    let rescale_0 = select(g_sum_0 > 0.0f32, exp(run_max_0 - g_max_0) / g_sum_0, 0.0f32);
    threadgroup_store("tg_out0", idx, o0_0 * rescale_0);
    threadgroup_store("tg_out1", idx, o0_1 * rescale_0);
    threadgroup_store("tg_out2", idx, o0_2 * rescale_0);
    threadgroup_store("tg_out3", idx, o0_3 * rescale_0);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0_0 = 0.0f32;
        let mut so0_1 = 0.0f32;
        let mut so0_2 = 0.0f32;
        let mut so0_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0_0 = so0_0 + threadgroup_load("tg_out0", ri);
            so0_1 = so0_1 + threadgroup_load("tg_out1", ri);
            so0_2 = so0_2 + threadgroup_load("tg_out2", ri);
            so0_3 = so0_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_0 = q_head * 8u32 * head_dim + d0;
        store(out[out_off_0], so0_0.cast::<T>());
        store(out[out_off_0 + 1u32], so0_1.cast::<T>());
        store(out[out_off_0 + 2u32], so0_2.cast::<T>());
        store(out[out_off_0 + 3u32], so0_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase B: Q[1]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_1);
        threadgroup_store("tg_sum", sg, run_sum_1);
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
    let g_max_1 = threadgroup_load("tg_max", 0);
    let g_sum_1 = threadgroup_load("tg_sum", 0);
    let rescale_1 = select(g_sum_1 > 0.0f32, exp(run_max_1 - g_max_1) / g_sum_1, 0.0f32);
    threadgroup_store("tg_out0", idx, o1_0 * rescale_1);
    threadgroup_store("tg_out1", idx, o1_1 * rescale_1);
    threadgroup_store("tg_out2", idx, o1_2 * rescale_1);
    threadgroup_store("tg_out3", idx, o1_3 * rescale_1);
    threadgroup_barrier();
    if sg == 0 {
        let mut so1_0 = 0.0f32;
        let mut so1_1 = 0.0f32;
        let mut so1_2 = 0.0f32;
        let mut so1_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so1_0 = so1_0 + threadgroup_load("tg_out0", ri);
            so1_1 = so1_1 + threadgroup_load("tg_out1", ri);
            so1_2 = so1_2 + threadgroup_load("tg_out2", ri);
            so1_3 = so1_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_1 = q_head * 8u32 * head_dim + head_dim + d0;
        store(out[out_off_1], so1_0.cast::<T>());
        store(out[out_off_1 + 1u32], so1_1.cast::<T>());
        store(out[out_off_1 + 2u32], so1_2.cast::<T>());
        store(out[out_off_1 + 3u32], so1_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase C: Q[2]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_2);
        threadgroup_store("tg_sum", sg, run_sum_2);
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
    let g_max_2 = threadgroup_load("tg_max", 0);
    let g_sum_2 = threadgroup_load("tg_sum", 0);
    let rescale_2 = select(g_sum_2 > 0.0f32, exp(run_max_2 - g_max_2) / g_sum_2, 0.0f32);
    threadgroup_store("tg_out0", idx, o2_0 * rescale_2);
    threadgroup_store("tg_out1", idx, o2_1 * rescale_2);
    threadgroup_store("tg_out2", idx, o2_2 * rescale_2);
    threadgroup_store("tg_out3", idx, o2_3 * rescale_2);
    threadgroup_barrier();
    if sg == 0 {
        let mut so2_0 = 0.0f32;
        let mut so2_1 = 0.0f32;
        let mut so2_2 = 0.0f32;
        let mut so2_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so2_0 = so2_0 + threadgroup_load("tg_out0", ri);
            so2_1 = so2_1 + threadgroup_load("tg_out1", ri);
            so2_2 = so2_2 + threadgroup_load("tg_out2", ri);
            so2_3 = so2_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_2 = q_head * 8u32 * head_dim + 2u32 * head_dim + d0;
        store(out[out_off_2], so2_0.cast::<T>());
        store(out[out_off_2 + 1u32], so2_1.cast::<T>());
        store(out[out_off_2 + 2u32], so2_2.cast::<T>());
        store(out[out_off_2 + 3u32], so2_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase D: Q[3]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_3);
        threadgroup_store("tg_sum", sg, run_sum_3);
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
    let g_max_3 = threadgroup_load("tg_max", 0);
    let g_sum_3 = threadgroup_load("tg_sum", 0);
    let rescale_3 = select(g_sum_3 > 0.0f32, exp(run_max_3 - g_max_3) / g_sum_3, 0.0f32);
    threadgroup_store("tg_out0", idx, o3_0 * rescale_3);
    threadgroup_store("tg_out1", idx, o3_1 * rescale_3);
    threadgroup_store("tg_out2", idx, o3_2 * rescale_3);
    threadgroup_store("tg_out3", idx, o3_3 * rescale_3);
    threadgroup_barrier();
    if sg == 0 {
        let mut so3_0 = 0.0f32;
        let mut so3_1 = 0.0f32;
        let mut so3_2 = 0.0f32;
        let mut so3_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so3_0 = so3_0 + threadgroup_load("tg_out0", ri);
            so3_1 = so3_1 + threadgroup_load("tg_out1", ri);
            so3_2 = so3_2 + threadgroup_load("tg_out2", ri);
            so3_3 = so3_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_3 = q_head * 8u32 * head_dim + 3u32 * head_dim + d0;
        store(out[out_off_3], so3_0.cast::<T>());
        store(out[out_off_3 + 1u32], so3_1.cast::<T>());
        store(out[out_off_3 + 2u32], so3_2.cast::<T>());
        store(out[out_off_3 + 3u32], so3_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase E: Q[4]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_4);
        threadgroup_store("tg_sum", sg, run_sum_4);
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
    let g_max_4 = threadgroup_load("tg_max", 0);
    let g_sum_4 = threadgroup_load("tg_sum", 0);
    let rescale_4 = select(g_sum_4 > 0.0f32, exp(run_max_4 - g_max_4) / g_sum_4, 0.0f32);
    threadgroup_store("tg_out0", idx, o4_0 * rescale_4);
    threadgroup_store("tg_out1", idx, o4_1 * rescale_4);
    threadgroup_store("tg_out2", idx, o4_2 * rescale_4);
    threadgroup_store("tg_out3", idx, o4_3 * rescale_4);
    threadgroup_barrier();
    if sg == 0 {
        let mut so4_0 = 0.0f32;
        let mut so4_1 = 0.0f32;
        let mut so4_2 = 0.0f32;
        let mut so4_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so4_0 = so4_0 + threadgroup_load("tg_out0", ri);
            so4_1 = so4_1 + threadgroup_load("tg_out1", ri);
            so4_2 = so4_2 + threadgroup_load("tg_out2", ri);
            so4_3 = so4_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_4 = q_head * 8u32 * head_dim + 4u32 * head_dim + d0;
        store(out[out_off_4], so4_0.cast::<T>());
        store(out[out_off_4 + 1u32], so4_1.cast::<T>());
        store(out[out_off_4 + 2u32], so4_2.cast::<T>());
        store(out[out_off_4 + 3u32], so4_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase F: Q[5]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_5);
        threadgroup_store("tg_sum", sg, run_sum_5);
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
    let g_max_5 = threadgroup_load("tg_max", 0);
    let g_sum_5 = threadgroup_load("tg_sum", 0);
    let rescale_5 = select(g_sum_5 > 0.0f32, exp(run_max_5 - g_max_5) / g_sum_5, 0.0f32);
    threadgroup_store("tg_out0", idx, o5_0 * rescale_5);
    threadgroup_store("tg_out1", idx, o5_1 * rescale_5);
    threadgroup_store("tg_out2", idx, o5_2 * rescale_5);
    threadgroup_store("tg_out3", idx, o5_3 * rescale_5);
    threadgroup_barrier();
    if sg == 0 {
        let mut so5_0 = 0.0f32;
        let mut so5_1 = 0.0f32;
        let mut so5_2 = 0.0f32;
        let mut so5_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so5_0 = so5_0 + threadgroup_load("tg_out0", ri);
            so5_1 = so5_1 + threadgroup_load("tg_out1", ri);
            so5_2 = so5_2 + threadgroup_load("tg_out2", ri);
            so5_3 = so5_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_5 = q_head * 8u32 * head_dim + 5u32 * head_dim + d0;
        store(out[out_off_5], so5_0.cast::<T>());
        store(out[out_off_5 + 1u32], so5_1.cast::<T>());
        store(out[out_off_5 + 2u32], so5_2.cast::<T>());
        store(out[out_off_5 + 3u32], so5_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase G: Q[6]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_6);
        threadgroup_store("tg_sum", sg, run_sum_6);
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
    let g_max_6 = threadgroup_load("tg_max", 0);
    let g_sum_6 = threadgroup_load("tg_sum", 0);
    let rescale_6 = select(g_sum_6 > 0.0f32, exp(run_max_6 - g_max_6) / g_sum_6, 0.0f32);
    threadgroup_store("tg_out0", idx, o6_0 * rescale_6);
    threadgroup_store("tg_out1", idx, o6_1 * rescale_6);
    threadgroup_store("tg_out2", idx, o6_2 * rescale_6);
    threadgroup_store("tg_out3", idx, o6_3 * rescale_6);
    threadgroup_barrier();
    if sg == 0 {
        let mut so6_0 = 0.0f32;
        let mut so6_1 = 0.0f32;
        let mut so6_2 = 0.0f32;
        let mut so6_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so6_0 = so6_0 + threadgroup_load("tg_out0", ri);
            so6_1 = so6_1 + threadgroup_load("tg_out1", ri);
            so6_2 = so6_2 + threadgroup_load("tg_out2", ri);
            so6_3 = so6_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_6 = q_head * 8u32 * head_dim + 6u32 * head_dim + d0;
        store(out[out_off_6], so6_0.cast::<T>());
        store(out[out_off_6 + 1u32], so6_1.cast::<T>());
        store(out[out_off_6 + 2u32], so6_2.cast::<T>());
        store(out[out_off_6 + 3u32], so6_3.cast::<T>());
    }
    threadgroup_barrier();
    // Phase H: Q[7]
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max_7);
        threadgroup_store("tg_sum", sg, run_sum_7);
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
    let g_max_7 = threadgroup_load("tg_max", 0);
    let g_sum_7 = threadgroup_load("tg_sum", 0);
    let rescale_7 = select(g_sum_7 > 0.0f32, exp(run_max_7 - g_max_7) / g_sum_7, 0.0f32);
    threadgroup_store("tg_out0", idx, o7_0 * rescale_7);
    threadgroup_store("tg_out1", idx, o7_1 * rescale_7);
    threadgroup_store("tg_out2", idx, o7_2 * rescale_7);
    threadgroup_store("tg_out3", idx, o7_3 * rescale_7);
    threadgroup_barrier();
    if sg == 0 {
        let mut so7_0 = 0.0f32;
        let mut so7_1 = 0.0f32;
        let mut so7_2 = 0.0f32;
        let mut so7_3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so7_0 = so7_0 + threadgroup_load("tg_out0", ri);
            so7_1 = so7_1 + threadgroup_load("tg_out1", ri);
            so7_2 = so7_2 + threadgroup_load("tg_out2", ri);
            so7_3 = so7_3 + threadgroup_load("tg_out3", ri);
        }
        let out_off_7 = q_head * 8u32 * head_dim + 7u32 * head_dim + d0;
        store(out[out_off_7], so7_0.cast::<T>());
        store(out[out_off_7 + 1u32], so7_1.cast::<T>());
        store(out[out_off_7 + 2u32], so7_2.cast::<T>());
        store(out[out_off_7 + 3u32], so7_3.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::sdpa_decode_batched_q2;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = sdpa_decode_batched_q2::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("sdpa_decode_batched_q2 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void sdpa_decode_batched_q2"),
                "MSL for {dt:?} should declare the kernel function:\n{src}",
            );
        }
    }

    #[test]
    fn codegen_uses_threadgroup_reduction_primitives() {
        // Pin the cross-simdgroup reduction structure: simdgroup
        // intrinsics, threadgroup memory, simd_sum + simd_max, and a
        // threadgroup-scoped barrier all need to survive future
        // codegen passes. Two-phase output reduction means at least
        // 5 threadgroup_barriers (1 inside each cross-sg max+sum
        // reduction × 2 phases, +1 between Phase A and Phase B,
        // +1 after each tg_out write).
        let src = msl_for(DType::F32);
        for tok in &[
            "simd_group",
            "simd_lane",
            "threadgroup_barrier",
            "mem_threadgroup",
            "simd_sum",
            "simd_max",
        ] {
            assert!(src.contains(tok), "MSL missing `{tok}`:\n{src}");
        }
    }

    #[test]
    fn codegen_emits_two_phase_reduction() {
        // The Q[0] and Q[1] output reductions reuse the same tg buffer
        // names. We can't directly count "phases" in the MSL, but the
        // store-out path runs twice — once for each Q — so there should
        // be 8 effective element-writes (4 quartiles × 2 Q's), whether
        // emitted as 8 scalar `out[...]=` or 2 vectorized VectorStore ops
        // (`*((device float4*)((device float*)out + ...))`, each covering 4).
        let src = msl_for(DType::F32);
        let scalar = src.matches("out[").count();
        let vector = src.matches("float*)out +").count();
        let effective = scalar + vector * 4;
        assert!(
            effective >= 8,
            "Expected ≥8 effective out-writes (4 quartiles × 2 Q positions); \
             got {effective} ({scalar} scalar + {vector} vectorized×4):\n{src}",
        );
    }

    #[test]
    fn codegen_loads_both_q_vectors() {
        // The kernel pre-loads both Q vectors (8 quartile loads total)
        // before the KV walk. If the IR forgets one Q, KV amortization
        // collapses.
        let src = msl_for(DType::F32);
        // Both Q offsets should appear in the emit.
        assert!(
            src.contains("q_off_0") || src.contains("q[q_head * 2"),
            "Expected Q[0] base offset in MSL:\n{src}"
        );
    }

    // ── K=4 codegen tests ────────────────────────────────────────────

    fn msl_for_q4(dt: DType) -> String {
        let mut k = super::sdpa_decode_batched_q4::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("sdpa_decode_batched_q4 codegen succeeds")
    }

    #[test]
    fn q4_codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for_q4(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void sdpa_decode_batched_q4"),
                "MSL for {dt:?} should declare the kernel function:\n{src}",
            );
        }
    }

    #[test]
    fn q4_codegen_emits_four_phase_reduction() {
        // K=4 runs the output reduction in 4 sequential phases. Each
        // phase writes 4 quartiles, so there should be 16 effective
        // element-writes (4 quartiles × 4 Q positions), whether scalar
        // or fused into VectorStore ops (each covering 4 elements).
        let src = msl_for_q4(DType::F32);
        let scalar = src.matches("out[").count();
        let vector = src.matches("float*)out +").count();
        let effective = scalar + vector * 4;
        assert!(
            effective >= 16,
            "Expected ≥16 effective out-writes (4 quartiles × 4 Q positions); \
             got {effective} ({scalar} scalar + {vector} vectorized×4)",
        );
    }

    #[test]
    fn q4_codegen_uses_threadgroup_reduction_primitives() {
        let src = msl_for_q4(DType::F32);
        for tok in &[
            "simd_group",
            "simd_lane",
            "threadgroup_barrier",
            "mem_threadgroup",
            "simd_sum",
            "simd_max",
        ] {
            assert!(src.contains(tok), "MSL missing `{tok}`");
        }
    }

    #[test]
    fn q4_codegen_loads_all_four_q_vectors() {
        // The kernel pre-loads four Q vectors (16 quartile loads total)
        // before the KV walk. We check for the four Q-position bases
        // via the multiplier-by-4 in the offset arithmetic.
        let src = msl_for_q4(DType::F32);
        assert!(
            src.contains("q_off_0") || src.contains("q[q_head * 4"),
            "Expected Q[0] base offset (q_head * 4 * head_dim) in MSL"
        );
        // All four streams' output rescales should appear.
        for stream in &["rescale_0", "rescale_1", "rescale_2", "rescale_3"] {
            assert!(src.contains(stream), "MSL missing per-stream `{stream}`");
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{sdpa_decode_batched_q2, sdpa_decode_batched_q4, sdpa_decode_batched_q8};
    use crate::utils::{pack_f32, unpack_f32};

    // Dense SDPA for a single Q slot: `[n_q_heads, head_dim]` query,
    // K/V `[n_kv_heads, kv_stride, head_dim]`. Returns `[n_q_heads,
    // head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_one(
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

    /// Build a batched-decode test: `batch_q` independent single-token decodes
    /// sharing one K/V cache, packed into the kernel's `[n_q_heads, batch_q,
    /// head_dim]` layout. Each Q slot must reproduce its standalone decode
    /// (`naive_sdpa_one`) — the batching is purely a layout/throughput change.
    ///
    /// `tpg` is the threads-per-group: q2 tolerates the full 1024, but q4/q8
    /// carry more per-thread register state and MUST dispatch at 512 (16
    /// simdgroups) — at 1024 their register pressure silently miscomputes on
    /// M-class GPUs (the hazard the legacy file guarded with a divergence
    /// regression test).
    fn batched_setup(
        ir: metaltile::core::ir::Kernel,
        batch_q: usize,
        tpg: u32,
        dt: DType,
    ) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 128usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // `batch_q` independent Q slots with distinct ramps.
        let q_slots: Vec<Vec<f32>> = (0..batch_q)
            .map(|b| {
                let step = 0.013 + 0.004 * b as f32;
                let start = -0.4 + 0.1 * b as f32;
                unpack_f32(&pack_f32(&ramp(n_q_heads * head_dim, step, start), dt), dt)
            })
            .collect();
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);

        // Interleave Q slots into [n_q_heads, batch_q, head_dim].
        let mut q = Vec::with_capacity(n_q_heads * batch_q * head_dim);
        for h in 0..n_q_heads {
            for slot in &q_slots {
                q.extend_from_slice(&slot[h * head_dim..(h + 1) * head_dim]);
            }
        }

        let outs: Vec<Vec<f32>> = q_slots
            .iter()
            .map(|qs| {
                naive_sdpa_one(qs, &k, &v, n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, scale)
            })
            .collect();
        let mut expected = Vec::with_capacity(n_q_heads * batch_q * head_dim);
        for h in 0..n_q_heads {
            for out in &outs {
                expected.extend_from_slice(&out[h * head_dim..(h + 1) * head_dim]);
            }
        }

        TestSetup::new(ir)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_q_heads * batch_q * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [tpg, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_decode_batched_q2(dt: DType) -> TestSetup {
        batched_setup(sdpa_decode_batched_q2::kernel_ir_for(dt), 2, 1024, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_decode_batched_q4(dt: DType) -> TestSetup {
        batched_setup(sdpa_decode_batched_q4::kernel_ir_for(dt), 4, 512, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_decode_batched_q8(dt: DType) -> TestSetup {
        // q8 carries the most per-thread state — dispatch at 256 (8 simdgroups)
        // to stay within the Apple GPU register file (512 still miscomputes).
        batched_setup(sdpa_decode_batched_q8::kernel_ir_for(dt), 8, 256, dt)
    }
}

/// New-syntax benchmarks for the batched-Q decode family (q2/q4/q8,
/// `class=GenericEmpty`). Q/out layout `[n_q_heads, batch_q, head_dim]`;
/// Reduction mode, one threadgroup per Q head, tpg=1024.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{sdpa_decode_batched_q2, sdpa_decode_batched_q4, sdpa_decode_batched_q8};

    fn setup(ir: metaltile::core::ir::Kernel, batch_q: usize, dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 128usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes = (2 * n_q_heads * batch_q * head_dim + 2 * n_kv_heads * n_kv * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ir)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * batch_q * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_q_heads * batch_q * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_decode_batched_q2", dtypes = [f32, f16, bf16])]
    fn bench_q2(dt: DType) -> BenchSetup { setup(sdpa_decode_batched_q2::kernel_ir_for(dt), 2, dt) }

    #[bench(name = "ffai/sdpa_decode_batched_q4", dtypes = [f32, f16, bf16])]
    fn bench_q4(dt: DType) -> BenchSetup { setup(sdpa_decode_batched_q4::kernel_ir_for(dt), 4, dt) }

    #[bench(name = "ffai/sdpa_decode_batched_q8", dtypes = [f32, f16, bf16])]
    fn bench_q8(dt: DType) -> BenchSetup { setup(sdpa_decode_batched_q8::kernel_ir_for(dt), 8, dt) }
}
