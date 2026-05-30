//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MoE orchestration kernels — router top-k, permute, unpermute,
//! grouped BGEMM dispatch.
//!
//! Targets Qwen3.6-35B-A3B and Qwen3-Coder-30B-A3B end-to-end serving.
//! The per-expert quantized matmul cell is already served by
//! `mt_qmm_*` (mma / mma_m16 / bm4 / bm2 / v2) — this module adds the
//! routing kernels that go around each expert call.
//!
//! ## Pipeline shape
//!
//! ```text
//!   activations [B*T, hidden]
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ mt_moe_router_topk│   logits  → [B*T, k] (indices + weights)
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │   mt_moe_permute │   [B*T, hidden]  → [k*B*T, hidden] expert-sorted
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ per-expert qmm   │   N × mt_qmm_for() calls — already shipped
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ mt_moe_unpermute │   [k*B*T, hidden] + weights  → [B*T, hidden]
//!   └──────────────────┘
//! ```

use metaltile::kernel;

// ── mt_moe_router_topk ───────────────────────────────────────────────────
//
// Per-token select top-k experts from `router_logits`, plus softmax
// weights over the chosen k.
//
// Inputs:
//   router_logits — [B*T, n_experts]  (any float dtype, computed in f32)
//   indices_out   — [B*T, k]          (u32)
//   weights_out   — [B*T, k]          (same dtype as router_logits, softmax weights)
//
// Constexpr:
//   n_experts   — typical Qwen3.6-A3B: 128.  Must fit one simdgroup
//                 (≤ 32×32 = 1024) — every reasonable MoE topology.
//   k           — typical 6-8 for production MoE.  Hard cap k ≤ 32.
//
// Geometry:
//   tpg=32  (one simdgroup per token row)
//   grid = [B*T, 1, 1]  (Reduction mode)
//
// Algorithm — k iterations of simd-parallel argmax with mask of
// previously-chosen indices stored in TG memory.  After k passes,
// softmax over the chosen k values in-place on lane 0..k-1.
//
// Bench spec uses BenchDispatch::Generic + shapes: &[] so `tile bench`
// skips it; correctness lives in unit tests + downstream MoE
// integration. Same convention as other ffai/ kernels (gather, sampling).
#[kernel]
pub fn mt_moe_router_topk<T>(
    router_logits: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
    // 1 = Qwen3-MoE style (softmax over chosen-k, sum-to-1 — `norm_topk_prob=True`)
    // 0 = Qwen3-Next style (softmax over ALL n_experts, return chosen probs
    //     un-renormalized — `norm_topk_prob=False`)
    // Mathematically equivalent at mode 1: softmax-over-chosen-k is the
    // same as (softmax-over-all → renormalize-over-chosen). Mode 0
    // returns probs that sum to < 1 across the chosen k, matching MLX's
    // qwen3_next.py:334-341.
    //
    // INVARIANT: this kernel pins tpg=32 (one simdgroup per token row).
    // The `simdgroup_barrier_mem_none()` below is correct only at tpg=32.
    // Caller must dispatch with `[n_rows, 1, 1] × [32, 1, 1]`.
    #[constexpr] norm_topk_prob: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;
    // TG scratch: chosen indices + values from each of the k argmax passes.
    // 32 slots covers any reasonable k (typical 6-8). Kernel assumes
    // k ≤ 32 — caller MUST enforce this in the host-side dispatcher
    // (no GPU-side check, would silently scribble into adjacent TG mem).
    threadgroup_alloc("tg_chosen_idx", 32u32);
    threadgroup_alloc("tg_chosen_val", 32u32);
    // Cache the all-experts-softmax sum for Qwen3-Next mode (mode 0).
    // 1 slot, written by lane 0 in the prepass.
    threadgroup_alloc("tg_full_sum", 1u32);
    threadgroup_alloc("tg_full_max", 1u32);
    // ── Pre-pass: compute softmax denominator over ALL n_experts ─────
    // Needed only for norm_topk_prob=0 (Qwen3-Next), but the cost is
    // trivial (one simd_max + simd_sum) and emitting it unconditionally
    // keeps the codegen tight (the codegen DCE will drop the dead path
    // when the constexpr branch is unreachable).
    let mut local_max_all = neg_infinity();
    let n_per_lane_pre = (n_experts + 31u32) / 32u32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            let better = v > local_max_all;
            local_max_all = select(better, v, local_max_all);
        }
    }
    let row_max_all = simd_max(local_max_all);
    let mut local_sum_all = 0.0f32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            local_sum_all = local_sum_all + exp(v - row_max_all);
        }
    }
    let row_sum_all = simd_sum(local_sum_all);
    if lane == 0u32 {
        threadgroup_store("tg_full_max", 0u32, row_max_all);
        threadgroup_store("tg_full_sum", 0u32, row_sum_all);
    }
    simdgroup_barrier_mem_none();
    // ── k argmax passes with chosen-mask ─────────────────────────────
    for it in range(0u32, k, 1u32) {
        // Per-lane local argmax over its slice of n_experts.
        // Each lane covers ceil(n_experts/32) experts.
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                let v = load(router_logits[row_base + j]).cast::<f32>();
                // Mask: was j picked in a previous iter?
                // Scan tg_chosen_idx[0..it] — k ≤ 8 typically so this
                // is fast even without early exit.
                let mut chosen_mask = 0u32;
                for p in range(0u32, it, 1u32) {
                    let cp = threadgroup_load("tg_chosen_idx", p);
                    chosen_mask = chosen_mask | select(j == cp, 1u32, 0u32);
                }
                let candidate = select(chosen_mask > 0u32, neg_infinity(), v);
                let better = candidate > best_val;
                best_val = select(better, candidate, best_val);
                best_idx = select(better, j, best_idx);
            }
        }
        // Cross-lane reduce.  simd_max gives the global best value;
        // ties broken to smaller idx via simd_min on (idx | sentinel).
        let global_best_val = simd_max(best_val);
        let i_have = best_val == global_best_val;
        let my_idx_or_max = select(i_have, best_idx, 4294967295u32); // u32::MAX
        let global_best_idx = simd_min(my_idx_or_max);
        // Lane 0 writes the iter's chosen slot.
        if lane == 0u32 {
            threadgroup_store("tg_chosen_idx", it, global_best_idx);
            threadgroup_store("tg_chosen_val", it, global_best_val);
        }
        simdgroup_barrier_mem_none();
    }
    // ── Softmax / weight emit per `norm_topk_prob` ──────────────────
    // Mode 1 (Qwen3-MoE, default): softmax over chosen-k (sum-to-1).
    //   numerator   = exp(z_i - max_chosen);  divisor = Σ_j∈chosen
    //   == exp(z_i - max_all) · const / Σ_j∈chosen exp(z_j - max_all) · const
    //   so we can use the SAME numerator as mode 0 (exp(z - max_all)) and
    //   just swap the divisor.  Avoids needing a Rust `if`-expression
    //   which the DSL doesn't unify across arms.
    // Mode 0 (Qwen3-Next): un-normalized chosen probs (sum < 1).
    //   weight_i = exp(z_i - max_all) / Σ_j∈all exp(z_j - max_all)
    let my_val = select(lane < k, threadgroup_load("tg_chosen_val", lane), neg_infinity());
    let row_max_full = threadgroup_load("tg_full_max", 0u32);
    let row_sum_full = threadgroup_load("tg_full_sum", 0u32);
    let exp_val = exp(my_val - row_max_full);
    let masked_exp = select(lane < k, exp_val, 0.0f32);
    let sum_chosen = simd_sum(masked_exp);
    // Pick divisor: chosen-k sum for renormalized (mode 1) or all-experts
    // sum for raw probs (mode 0). select() forces both to be live; codegen
    // const-folds when `norm_topk_prob` bakes in.
    let divisor = select(norm_topk_prob == 1u32, sum_chosen, row_sum_full);
    let weight = masked_exp / divisor;
    // ── Write outputs ───────────────────────────────────────────────
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], threadgroup_load("tg_chosen_idx", lane));
        store(weights_out[out_base], weight.cast::<T>());
    }
}

// ── mt_moe_unpermute ─────────────────────────────────────────────────────
//
// Combine k expert outputs back into the original token order with
// top-k softmax weights.
//
// Inputs:
//   expert_outputs  — [k*B*T, hidden]   per-expert dense outputs at the
//                                       expert-sorted positions
//   inv_perm        — [B*T, k]          where (token i, slot j) was placed
//                                       in expert_outputs (computed by
//                                       caller's sort step)
//   top_k_weights   — [B*T, k]          softmax weights from
//                                       mt_moe_router_topk
//   out             — [B*T, hidden]     weighted sum across k experts
//
// Constexpr:
//   hidden — model hidden dim (e.g. 2048 for Qwen3-MoE)
//   k      — top-k expert count (e.g. 8)
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes via 4-wide vectorize)
//   grid=[B*T, 1, 1]
//
// Per-token cost: read k * hidden / 128 = (k * hidden) / 128 expert
// values + k weights, do k FMAs per output column, one store per
// column. At hidden=2048, k=8 → ~1k FMAs per token. Bandwidth-bound,
// not ALU-bound.
#[kernel]
pub fn mt_moe_unpermute<T>(
    expert_outputs: Tensor<T>,
    inv_perm: Tensor<u32>,
    top_k_weights: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] k: u32,
) {
    let token = tgid_x;
    let lane = tid;
    let row_base_inv = token * k;
    let row_base_w = token * k;
    let row_base_out = token * hidden;
    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let mut acc = 0.0f32;
            for j in range(0u32, k, 1u32) {
                let pos = load(inv_perm[row_base_inv + j]);
                let v = load(expert_outputs[pos * hidden + h]).cast::<f32>();
                let w = load(top_k_weights[row_base_w + j]).cast::<f32>();
                acc = acc + w * v;
            }
            store(out[row_base_out + h], acc.cast::<T>());
        }
    }
}

// ── mt_moe_permute ───────────────────────────────────────────────────────
//
// Gather tokens into per-expert contiguous buffers given a pre-computed
// sort permutation. The expensive sort step (argsort over top-k expert
// indices) is done by the caller — typically CPU-side via Rust sort,
// or via a future sort kernel. This kernel is just the data-movement
// half: each output position copies the row indicated by sort_token_idx.
//
// Inputs:
//   tokens          — [B*T, hidden]      activations to gather
//   sort_token_idx  — [k * B*T]          for each permuted position p,
//                                        which original token row sourced it.
//                                        Caller computes via argsort over
//                                        top-k indices flattened to
//                                        (token * k + slot) → token (this is
//                                        the "permute" direction; the inverse
//                                        is `inv_perm` consumed by unpermute).
//   permuted        — [k * B*T, hidden]  expert-sorted output. Each k*B*T
//                                        row corresponds to one (expert, token)
//                                        pair; consecutive rows with the same
//                                        expert form that expert's input slab.
//
// Constexpr:
//   hidden — model hidden dim
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes, ceil(hidden/128) iters)
//   grid=[k*B*T, 1, 1]
//
// Per-permuted-row cost: hidden / 128 = 16 loads + 16 stores (at
// hidden=2048). Bandwidth-bound — no FMAs, just a vector copy.
#[kernel]
pub fn mt_moe_permute<T>(
    tokens: Tensor<T>,
    sort_token_idx: Tensor<u32>,
    mut permuted: Tensor<T>,
    #[constexpr] hidden: u32,
) {
    let permuted_pos = tgid_x;
    let lane = tid;
    let token = load(sort_token_idx[permuted_pos]);
    let src_base = token * hidden;
    let dst_base = permuted_pos * hidden;
    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let v = load(tokens[src_base + h]);
            store(permuted[dst_base + h], v);
        }
    }
}

// ── mt_moe_gather_qmm_int4 ────────────────────────────────────────────────
//
// Grouped quantized matmul for MoE. Matches MLX's `gatherQuantizedMM`
// (called by SwitchLinear → SwitchGLU → Qwen35SparseMoeBlock):
//
//     y[t, m] = Σ_k x[t, k] · W[E(t), m, k]
//
// where E(t) is the expert assigned to row t. Pre-permuted layout (caller
// passes `sortedIndices=true` upstream): consecutive rows share an expert,
// and `expert_offsets` is a CSR row-offset array — expert `e` owns rows
// `[expert_offsets[e] .. expert_offsets[e+1])`.
//
// One dispatch → all experts × M_out × T rows. Vs MLX's N separate qmm
// dispatches (128 experts × 40 layers × 3 projections = 15360 launches at
// Qwen3.6-35B-A3B), folding into one kernel saves ~1.5 s of host-side
// launch overhead per forward. Decode benefits most (every step pays it);
// prefill saves a smaller fraction since each per-expert matmul is fatter.
//
// Inputs:
//   x               — [T, K_in]                 f32/f16/bf16 (sorted-by-expert)
//   weight_packed   — [E, M_out, K_in/8]        uint32 (int4 packed, 8 per uint)
//   scales          — [E, M_out, K_in/group]    T  per-group quant scale
//   biases          — [E, M_out, K_in/group]    T  per-group quant bias
//   expert_offsets  — [E + 1]                   uint32  CSR row offsets
//   out             — [T, M_out]                T
//
// Constexpr:
//   k_in       — fused input dim (e.g. 2048 for Qwen3.6-35B-A3B hidden)
//   m_out      — per-expert output dim (e.g. 256 for moe_intermediate)
//   n_experts  — typically 128
//   group_size — quant group size (typically 64)
//
// DISPATCH INVARIANTS
//   - **Mode: Reduction.** Uses `simd_sum` for the per-row dot product.
//   - **Grid: `[m_out, T, 1]`** — one TG per (output column m, row t).
//   - **TG: `[32, 1, 1]`** — one simdgroup. Each lane handles
//     `k_in / 32` packed uint32s × 8 weights each = `k_in / 32` weights.
//   - `k_in` must be a multiple of 32 (every Qwen3 / Qwen3.6 satisfies).
//   - `group_size` must divide `k_in`.
//   - int4 only (MLX's MoE quantization default). Wider precision is a
//     follow-up.
//
// Algorithm — scalar foundation; MMA tiling lands in a follow-up commit.
//
//   1. Resolve expert: linear walk over `expert_offsets`. With N_experts
//      ≤ 256 this is cheap (~256 reads on lane 0 + broadcast via TG mem).
//
//   2. Per-lane dot product over `k_in / 32` packed uint32s. Each uint32
//      packs 8 int4 weights → unpack 8 weights, dequant per-group, FMA.
//
//   3. `simd_sum` reduces 32 partial sums → one output value per TG.
//
// Mirrors the per-thread pattern in `dequant_gemv_int4`.
#[kernel]
pub fn mt_moe_gather_qmm_int4<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m = tgid_x;
    let row = tgid_y;
    let lane = tid;
    // Resolve expert — linear walk on EVERY lane (cheap, ≤ 256 reads from
    // a small uniform buffer) so the result lives in a per-lane u32 register
    // and never round-trips through float-typed TG memory.
    let mut expert = 0u32;
    let mut found = 0u32;
    for ee in range(0u32, n_experts, 1u32) {
        let end = load(expert_offsets[ee + 1u32]);
        let inside_bool = row < end;
        let inside = select(inside_bool, 1u32, 0u32);
        let take = inside * (1u32 - found);
        expert = select(take == 1u32, ee, expert);
        found = select(take == 1u32, 1u32, found);
    }
    // Stride-by-32 over packs: each lane handles packs at positions
    // lane, lane+32, lane+64, ... up to k_in/8. Correct for both small
    // (k_in=32 → 4 packs, only lanes 0..3 work) and large (k_in=2048 →
    // 256 packs, 8 packs/lane) inputs.
    let total_packs = k_in / 8u32;
    let weight_stride_m = total_packs;
    let weight_row_base = expert * m_out * weight_stride_m + m * weight_stride_m;
    let groups_per_row = k_in / group_size;
    let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
    let x_row_base = row * k_in;
    let mut acc = 0.0f32;
    for pack_idx in range(lane, total_packs, 32u32) {
        let packed = load(weight_packed[weight_row_base + pack_idx]);
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        let scale = load(scales[scale_row_base + g]).cast::<f32>();
        let bias = load(biases[scale_row_base + g]).cast::<f32>();
        let q0 = (packed >> 0u32) & 15u32;
        let q1 = (packed >> 4u32) & 15u32;
        let q2 = (packed >> 8u32) & 15u32;
        let q3 = (packed >> 12u32) & 15u32;
        let q4 = (packed >> 16u32) & 15u32;
        let q5 = (packed >> 20u32) & 15u32;
        let q6 = (packed >> 24u32) & 15u32;
        let q7 = (packed >> 28u32) & 15u32;
        let w0 = q0.cast::<f32>() * scale + bias;
        let w1 = q1.cast::<f32>() * scale + bias;
        let w2 = q2.cast::<f32>() * scale + bias;
        let w3 = q3.cast::<f32>() * scale + bias;
        let w4 = q4.cast::<f32>() * scale + bias;
        let w5 = q5.cast::<f32>() * scale + bias;
        let w6 = q6.cast::<f32>() * scale + bias;
        let w7 = q7.cast::<f32>() * scale + bias;
        let x0 = load(x[x_row_base + k_first + 0u32]).cast::<f32>();
        let x1 = load(x[x_row_base + k_first + 1u32]).cast::<f32>();
        let x2 = load(x[x_row_base + k_first + 2u32]).cast::<f32>();
        let x3 = load(x[x_row_base + k_first + 3u32]).cast::<f32>();
        let x4 = load(x[x_row_base + k_first + 4u32]).cast::<f32>();
        let x5 = load(x[x_row_base + k_first + 5u32]).cast::<f32>();
        let x6 = load(x[x_row_base + k_first + 6u32]).cast::<f32>();
        let x7 = load(x[x_row_base + k_first + 7u32]).cast::<f32>();
        acc = acc + w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row * m_out + m], total.cast::<T>());
    }
}

// ── mt_moe_gather_qmm_b{3,5,6,8} — wider-precision gather matmul ──────────
//
// `mt_moe_gather_qmm_int4` above is int4-only (MLX's MoE quantization
// default). This macro generates the same grouped-gather quantized
// matmul for the remaining MLX bit-widths — int3 / int5 / int6 / int8 —
// so a MoE block quantized at any width has a single-dispatch GPU path.
//
// The body is identical to `mt_moe_gather_qmm_int4` except for the
// weight-code extraction: pow2 widths (8) are pack-aligned (`32/bits`
// codes per u32, simple shift+mask), odd widths (3/5/6) use the
// two-word bit-stream extract (a code may straddle a u32 boundary) —
// the same split as `dequant_gemv.rs` / the `mt_qmv_b*` family. The
// per-output-element routing, CSR `expert_offsets` walk, group-indexed
// scale/bias, and `simd_sum` reduction are unchanged.
//
// `mt_moe_gather_qmv_b*` aliases register the same kernels under a
// `gather_qmv` subop — MLX names the M=1 / single-token decode form
// `gather_qmv`; the per-row-routed body serves both (qmv is the
// `T == n_experts-row-count` case of qmm).
//
// Layouts / constexpr / DISPATCH INVARIANTS — identical to
// `mt_moe_gather_qmm_int4`; see its doc block. `k_in` must be a
// multiple of 32; for pow2 widths it must also be a multiple of
// `32/bits`; `group_size` must divide `k_in`.

/// Grouped-gather quantized matmul — pow2 bit-widths (8).
macro_rules! gather_qmm_pow2 {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            weight_packed: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            expert_offsets: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] k_in: u32,
            #[constexpr] m_out: u32,
            #[constexpr] n_experts: u32,
            #[constexpr] group_size: u32,
        ) {
            let m = tgid_x;
            let row = tgid_y;
            let lane = tid;

            // Resolve expert — linear CSR walk on every lane (cheap).
            let mut expert = 0u32;
            let mut found = 0u32;
            for ee in range(0u32, n_experts, 1u32) {
                let end = load(expert_offsets[ee + 1u32]);
                let inside = select(row < end, 1u32, 0u32);
                let take = inside * (1u32 - found);
                expert = select(take == 1u32, ee, expert);
                found = select(take == 1u32, 1u32, found);
            }

            let vals_per_pack = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let total_packs = k_in / vals_per_pack;
            let weight_row_base = expert * m_out * total_packs + m * total_packs;

            let groups_per_row = k_in / group_size;
            let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
            let x_row_base = row * k_in;

            let mut acc = 0.0f32;
            for pack_idx in range(lane, total_packs, 32u32) {
                let packed = load(weight_packed[weight_row_base + pack_idx]);
                let k_first = pack_idx * vals_per_pack;
                let g = k_first / group_size;
                let scale = load(scales[scale_row_base + g]).cast::<f32>();
                let bias = load(biases[scale_row_base + g]).cast::<f32>();

                // Unpack vals_per_pack codes from this u32, FMA each.
                for i in range(0u32, vals_per_pack, 1u32) {
                    let q = (packed >> (i * $bits)) & mask;
                    let wv = q.cast::<f32>() * scale + bias;
                    let xv = load(x[x_row_base + k_first + i]).cast::<f32>();
                    acc = acc + wv * xv;
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[row * m_out + m], total.cast::<T>());
            }
        }
    };
}

/// Grouped-gather quantized matmul — odd bit-widths (3, 5, 6).
macro_rules! gather_qmm_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            weight_packed: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            expert_offsets: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] k_in: u32,
            #[constexpr] m_out: u32,
            #[constexpr] n_experts: u32,
            #[constexpr] group_size: u32,
        ) {
            let m = tgid_x;
            let row = tgid_y;
            let lane = tid;

            let mut expert = 0u32;
            let mut found = 0u32;
            for ee in range(0u32, n_experts, 1u32) {
                let end = load(expert_offsets[ee + 1u32]);
                let inside = select(row < end, 1u32, 0u32);
                let take = inside * (1u32 - found);
                expert = select(take == 1u32, ee, expert);
                found = select(take == 1u32, 1u32, found);
            }

            // Bit-stream layout: u32_per_row words per weight row.
            let u32_per_row = k_in * $bits / 32u32;
            let weight_row_base = expert * m_out * u32_per_row + m * u32_per_row;

            let groups_per_row = k_in / group_size;
            let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
            let x_row_base = row * k_in;

            // Each lane strides over individual K-elements (odd widths
            // don't pack-align — element-strided, like `dequant_gemv_odd`).
            let mut acc = 0.0f32;
            let n_iters = (k_in + 31u32) / 32u32;
            for _it in range(0u32, n_iters, 1u32) {
                let d = _it * 32u32 + lane;
                if d < k_in {
                    let g = d / group_size;
                    let scale = load(scales[scale_row_base + g]).cast::<f32>();
                    let bias = load(biases[scale_row_base + g]).cast::<f32>();

                    let bit_off = d * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight_packed[weight_row_base + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(weight_packed[weight_row_base + w1idx]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;

                    let wv = q.cast::<f32>() * scale + bias;
                    let xv = load(x[x_row_base + d]).cast::<f32>();
                    acc = acc + wv * xv;
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[row * m_out + m], total.cast::<T>());
            }
        }
    };
}

// gather_qmm — wider-precision grouped-gather matmul (int4 above).
gather_qmm_pow2!(mt_moe_gather_qmm_b8, 8u32, "gather_qmm_b8");
gather_qmm_odd!(mt_moe_gather_qmm_b3, 3u32, "gather_qmm_b3");
gather_qmm_odd!(mt_moe_gather_qmm_b5, 5u32, "gather_qmm_b5");
gather_qmm_odd!(mt_moe_gather_qmm_b6, 6u32, "gather_qmm_b6");
// ── mt_moe_gather_qmm_int4_m8 ────────────────────────────────────────────
//
// Same recurrence as `mt_moe_gather_qmm_int4`, but each TG produces 8
// adjacent `m_out` cells per row. Three wins over the m=1 variant:
//
//   1. 8× fewer TGs → 8× less dispatch + scheduler overhead. At
//      Qwen3.6-A3B down-proj (M=2048, T=8192) the m=1 variant fires 17M
//      TGs; m8 fires 2M.
//   2. `x[row, k]` reads serve 8 dot products instead of 1 → 8× weight-
//      relative-to-x bandwidth ratio.
//   3. Scale/bias loads (already grouped to 1 per `group_size`) are shared
//      across 8 cells too.
//
// DISPATCH:
//   Grid = [m_out / 8, T_rows, 1]   (m_out must be a multiple of 8)
//   TG   = [32, 1, 1]
//
// Per-lane work per TG:
//   - Outer: stride-by-32 over `k_in / 8` packs (same as the m=1 variant).
//   - Inner: 8 m-cells × 8 nibbles = 64 FMAs per pack.
//   - 8 accumulators per lane, simd_sum'd at the end → 8 outputs per TG.
//
// Memory traffic (per TG):
//   - x  : k_in floats (loaded once, used 8 times)
//   - W  : 8 × (k_in / 8) uint32s = k_in uint32s of weight
//   - s/b: 8 × (k_in / group_size) × 2 floats
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_int4_m8<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m_chunk = tgid_x;
    let row = tgid_y;
    let lane = tid;
    let m_base = m_chunk * 8u32;
    // Resolve expert — same linear walk as the m=1 variant.
    let mut expert = 0u32;
    let mut found = 0u32;
    for ee in range(0u32, n_experts, 1u32) {
        let end = load(expert_offsets[ee + 1u32]);
        let inside_bool = row < end;
        let inside = select(inside_bool, 1u32, 0u32);
        let take = inside * (1u32 - found);
        expert = select(take == 1u32, ee, expert);
        found = select(take == 1u32, 1u32, found);
    }
    let total_packs = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let weight_expert_base = expert * m_out * total_packs;
    let scale_expert_base = expert * m_out * groups_per_row;
    let x_row_base = row * k_in;
    // 8 separate accumulators, one per m-cell in the chunk.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    for pack_idx in range(lane, total_packs, 32u32) {
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        // Load 8 input values once — reused across 8 m-cells.
        let x0 = load(x[x_row_base + k_first + 0u32]).cast::<f32>();
        let x1 = load(x[x_row_base + k_first + 1u32]).cast::<f32>();
        let x2 = load(x[x_row_base + k_first + 2u32]).cast::<f32>();
        let x3 = load(x[x_row_base + k_first + 3u32]).cast::<f32>();
        let x4 = load(x[x_row_base + k_first + 4u32]).cast::<f32>();
        let x5 = load(x[x_row_base + k_first + 5u32]).cast::<f32>();
        let x6 = load(x[x_row_base + k_first + 6u32]).cast::<f32>();
        let x7 = load(x[x_row_base + k_first + 7u32]).cast::<f32>();
        // 8 hand-unrolled m-cells: each block computes one dot product and
        // adds directly to its accumulator — no select, no branch.
        //
        // Common per-m work: weight row base, scale row base, packed read,
        // scale + bias, 8-way nibble unpack, FMA into local dot.
        let wrb0 = weight_expert_base + (m_base + 0u32) * total_packs;
        let srb0 = scale_expert_base + (m_base + 0u32) * groups_per_row;
        let p0 = load(weight_packed[wrb0 + pack_idx]);
        let s0 = load(scales[srb0 + g]).cast::<f32>();
        let b0 = load(biases[srb0 + g]).cast::<f32>();
        let dot0 = ((p0 >> 0u32) & 15u32).cast::<f32>() * s0 * x0
            + b0 * x0
            + ((p0 >> 4u32) & 15u32).cast::<f32>() * s0 * x1
            + b0 * x1
            + ((p0 >> 8u32) & 15u32).cast::<f32>() * s0 * x2
            + b0 * x2
            + ((p0 >> 12u32) & 15u32).cast::<f32>() * s0 * x3
            + b0 * x3
            + ((p0 >> 16u32) & 15u32).cast::<f32>() * s0 * x4
            + b0 * x4
            + ((p0 >> 20u32) & 15u32).cast::<f32>() * s0 * x5
            + b0 * x5
            + ((p0 >> 24u32) & 15u32).cast::<f32>() * s0 * x6
            + b0 * x6
            + ((p0 >> 28u32) & 15u32).cast::<f32>() * s0 * x7
            + b0 * x7;
        acc0 = acc0 + dot0;
        let wrb1 = weight_expert_base + (m_base + 1u32) * total_packs;
        let srb1 = scale_expert_base + (m_base + 1u32) * groups_per_row;
        let p1 = load(weight_packed[wrb1 + pack_idx]);
        let s1 = load(scales[srb1 + g]).cast::<f32>();
        let b1 = load(biases[srb1 + g]).cast::<f32>();
        let dot1 = ((p1 >> 0u32) & 15u32).cast::<f32>() * s1 * x0
            + b1 * x0
            + ((p1 >> 4u32) & 15u32).cast::<f32>() * s1 * x1
            + b1 * x1
            + ((p1 >> 8u32) & 15u32).cast::<f32>() * s1 * x2
            + b1 * x2
            + ((p1 >> 12u32) & 15u32).cast::<f32>() * s1 * x3
            + b1 * x3
            + ((p1 >> 16u32) & 15u32).cast::<f32>() * s1 * x4
            + b1 * x4
            + ((p1 >> 20u32) & 15u32).cast::<f32>() * s1 * x5
            + b1 * x5
            + ((p1 >> 24u32) & 15u32).cast::<f32>() * s1 * x6
            + b1 * x6
            + ((p1 >> 28u32) & 15u32).cast::<f32>() * s1 * x7
            + b1 * x7;
        acc1 = acc1 + dot1;
        let wrb2 = weight_expert_base + (m_base + 2u32) * total_packs;
        let srb2 = scale_expert_base + (m_base + 2u32) * groups_per_row;
        let p2 = load(weight_packed[wrb2 + pack_idx]);
        let s2 = load(scales[srb2 + g]).cast::<f32>();
        let b2 = load(biases[srb2 + g]).cast::<f32>();
        let dot2 = ((p2 >> 0u32) & 15u32).cast::<f32>() * s2 * x0
            + b2 * x0
            + ((p2 >> 4u32) & 15u32).cast::<f32>() * s2 * x1
            + b2 * x1
            + ((p2 >> 8u32) & 15u32).cast::<f32>() * s2 * x2
            + b2 * x2
            + ((p2 >> 12u32) & 15u32).cast::<f32>() * s2 * x3
            + b2 * x3
            + ((p2 >> 16u32) & 15u32).cast::<f32>() * s2 * x4
            + b2 * x4
            + ((p2 >> 20u32) & 15u32).cast::<f32>() * s2 * x5
            + b2 * x5
            + ((p2 >> 24u32) & 15u32).cast::<f32>() * s2 * x6
            + b2 * x6
            + ((p2 >> 28u32) & 15u32).cast::<f32>() * s2 * x7
            + b2 * x7;
        acc2 = acc2 + dot2;
        let wrb3 = weight_expert_base + (m_base + 3u32) * total_packs;
        let srb3 = scale_expert_base + (m_base + 3u32) * groups_per_row;
        let p3 = load(weight_packed[wrb3 + pack_idx]);
        let s3 = load(scales[srb3 + g]).cast::<f32>();
        let b3 = load(biases[srb3 + g]).cast::<f32>();
        let dot3 = ((p3 >> 0u32) & 15u32).cast::<f32>() * s3 * x0
            + b3 * x0
            + ((p3 >> 4u32) & 15u32).cast::<f32>() * s3 * x1
            + b3 * x1
            + ((p3 >> 8u32) & 15u32).cast::<f32>() * s3 * x2
            + b3 * x2
            + ((p3 >> 12u32) & 15u32).cast::<f32>() * s3 * x3
            + b3 * x3
            + ((p3 >> 16u32) & 15u32).cast::<f32>() * s3 * x4
            + b3 * x4
            + ((p3 >> 20u32) & 15u32).cast::<f32>() * s3 * x5
            + b3 * x5
            + ((p3 >> 24u32) & 15u32).cast::<f32>() * s3 * x6
            + b3 * x6
            + ((p3 >> 28u32) & 15u32).cast::<f32>() * s3 * x7
            + b3 * x7;
        acc3 = acc3 + dot3;
        let wrb4 = weight_expert_base + (m_base + 4u32) * total_packs;
        let srb4 = scale_expert_base + (m_base + 4u32) * groups_per_row;
        let p4 = load(weight_packed[wrb4 + pack_idx]);
        let s4 = load(scales[srb4 + g]).cast::<f32>();
        let b4 = load(biases[srb4 + g]).cast::<f32>();
        let dot4 = ((p4 >> 0u32) & 15u32).cast::<f32>() * s4 * x0
            + b4 * x0
            + ((p4 >> 4u32) & 15u32).cast::<f32>() * s4 * x1
            + b4 * x1
            + ((p4 >> 8u32) & 15u32).cast::<f32>() * s4 * x2
            + b4 * x2
            + ((p4 >> 12u32) & 15u32).cast::<f32>() * s4 * x3
            + b4 * x3
            + ((p4 >> 16u32) & 15u32).cast::<f32>() * s4 * x4
            + b4 * x4
            + ((p4 >> 20u32) & 15u32).cast::<f32>() * s4 * x5
            + b4 * x5
            + ((p4 >> 24u32) & 15u32).cast::<f32>() * s4 * x6
            + b4 * x6
            + ((p4 >> 28u32) & 15u32).cast::<f32>() * s4 * x7
            + b4 * x7;
        acc4 = acc4 + dot4;
        let wrb5 = weight_expert_base + (m_base + 5u32) * total_packs;
        let srb5 = scale_expert_base + (m_base + 5u32) * groups_per_row;
        let p5 = load(weight_packed[wrb5 + pack_idx]);
        let s5 = load(scales[srb5 + g]).cast::<f32>();
        let b5 = load(biases[srb5 + g]).cast::<f32>();
        let dot5 = ((p5 >> 0u32) & 15u32).cast::<f32>() * s5 * x0
            + b5 * x0
            + ((p5 >> 4u32) & 15u32).cast::<f32>() * s5 * x1
            + b5 * x1
            + ((p5 >> 8u32) & 15u32).cast::<f32>() * s5 * x2
            + b5 * x2
            + ((p5 >> 12u32) & 15u32).cast::<f32>() * s5 * x3
            + b5 * x3
            + ((p5 >> 16u32) & 15u32).cast::<f32>() * s5 * x4
            + b5 * x4
            + ((p5 >> 20u32) & 15u32).cast::<f32>() * s5 * x5
            + b5 * x5
            + ((p5 >> 24u32) & 15u32).cast::<f32>() * s5 * x6
            + b5 * x6
            + ((p5 >> 28u32) & 15u32).cast::<f32>() * s5 * x7
            + b5 * x7;
        acc5 = acc5 + dot5;
        let wrb6 = weight_expert_base + (m_base + 6u32) * total_packs;
        let srb6 = scale_expert_base + (m_base + 6u32) * groups_per_row;
        let p6 = load(weight_packed[wrb6 + pack_idx]);
        let s6 = load(scales[srb6 + g]).cast::<f32>();
        let b6 = load(biases[srb6 + g]).cast::<f32>();
        let dot6 = ((p6 >> 0u32) & 15u32).cast::<f32>() * s6 * x0
            + b6 * x0
            + ((p6 >> 4u32) & 15u32).cast::<f32>() * s6 * x1
            + b6 * x1
            + ((p6 >> 8u32) & 15u32).cast::<f32>() * s6 * x2
            + b6 * x2
            + ((p6 >> 12u32) & 15u32).cast::<f32>() * s6 * x3
            + b6 * x3
            + ((p6 >> 16u32) & 15u32).cast::<f32>() * s6 * x4
            + b6 * x4
            + ((p6 >> 20u32) & 15u32).cast::<f32>() * s6 * x5
            + b6 * x5
            + ((p6 >> 24u32) & 15u32).cast::<f32>() * s6 * x6
            + b6 * x6
            + ((p6 >> 28u32) & 15u32).cast::<f32>() * s6 * x7
            + b6 * x7;
        acc6 = acc6 + dot6;
        let wrb7 = weight_expert_base + (m_base + 7u32) * total_packs;
        let srb7 = scale_expert_base + (m_base + 7u32) * groups_per_row;
        let p7 = load(weight_packed[wrb7 + pack_idx]);
        let s7 = load(scales[srb7 + g]).cast::<f32>();
        let b7 = load(biases[srb7 + g]).cast::<f32>();
        let dot7 = ((p7 >> 0u32) & 15u32).cast::<f32>() * s7 * x0
            + b7 * x0
            + ((p7 >> 4u32) & 15u32).cast::<f32>() * s7 * x1
            + b7 * x1
            + ((p7 >> 8u32) & 15u32).cast::<f32>() * s7 * x2
            + b7 * x2
            + ((p7 >> 12u32) & 15u32).cast::<f32>() * s7 * x3
            + b7 * x3
            + ((p7 >> 16u32) & 15u32).cast::<f32>() * s7 * x4
            + b7 * x4
            + ((p7 >> 20u32) & 15u32).cast::<f32>() * s7 * x5
            + b7 * x5
            + ((p7 >> 24u32) & 15u32).cast::<f32>() * s7 * x6
            + b7 * x6
            + ((p7 >> 28u32) & 15u32).cast::<f32>() * s7 * x7
            + b7 * x7;
        acc7 = acc7 + dot7;
    }
    let t0 = simd_sum(acc0);
    let t1 = simd_sum(acc1);
    let t2 = simd_sum(acc2);
    let t3 = simd_sum(acc3);
    let t4 = simd_sum(acc4);
    let t5 = simd_sum(acc5);
    let t6 = simd_sum(acc6);
    let t7 = simd_sum(acc7);
    if lane == 0u32 {
        store(out[row * m_out + m_base + 0u32], t0.cast::<T>());
        store(out[row * m_out + m_base + 1u32], t1.cast::<T>());
        store(out[row * m_out + m_base + 2u32], t2.cast::<T>());
        store(out[row * m_out + m_base + 3u32], t3.cast::<T>());
        store(out[row * m_out + m_base + 4u32], t4.cast::<T>());
        store(out[row * m_out + m_base + 5u32], t5.cast::<T>());
        store(out[row * m_out + m_base + 6u32], t6.cast::<T>());
        store(out[row * m_out + m_base + 7u32], t7.cast::<T>());
    }
}

// ── mt_moe_gather_qmm_int4_m16 ────────────────────────────────────────────
//
// Same pattern as `mt_moe_gather_qmm_int4_m8`, extended to 16
// adjacent `m_out` cells per row. 16× fewer TGs → 16× less dispatch +
// scheduler overhead, and `x[row, k]` reads serve 16 dot products.
//
// DISPATCH:
//   Grid = [m_out / 16, T_rows, 1]   (m_out must be a multiple of 16)
//   TG   = [32, 1, 1]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_int4_m16<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m_chunk = tgid_x;
    let row = tgid_y;
    let lane = tid;
    let m_base = m_chunk * 16u32;
    // Resolve expert — same linear walk as the m=1 variant.
    let mut expert = 0u32;
    let mut found = 0u32;
    for ee in range(0u32, n_experts, 1u32) {
        let end = load(expert_offsets[ee + 1u32]);
        let inside_bool = row < end;
        let inside = select(inside_bool, 1u32, 0u32);
        let take = inside * (1u32 - found);
        expert = select(take == 1u32, ee, expert);
        found = select(take == 1u32, 1u32, found);
    }
    let total_packs = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let weight_expert_base = expert * m_out * total_packs;
    let scale_expert_base = expert * m_out * groups_per_row;
    let x_row_base = row * k_in;
    // 16 separate accumulators, one per m-cell in the chunk.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut acc8 = 0.0f32;
    let mut acc9 = 0.0f32;
    let mut acc10 = 0.0f32;
    let mut acc11 = 0.0f32;
    let mut acc12 = 0.0f32;
    let mut acc13 = 0.0f32;
    let mut acc14 = 0.0f32;
    let mut acc15 = 0.0f32;
    for pack_idx in range(lane, total_packs, 32u32) {
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        // Load 8 input values once — reused across 16 m-cells.
        let x0 = load(x[x_row_base + k_first + 0u32]).cast::<f32>();
        let x1 = load(x[x_row_base + k_first + 1u32]).cast::<f32>();
        let x2 = load(x[x_row_base + k_first + 2u32]).cast::<f32>();
        let x3 = load(x[x_row_base + k_first + 3u32]).cast::<f32>();
        let x4 = load(x[x_row_base + k_first + 4u32]).cast::<f32>();
        let x5 = load(x[x_row_base + k_first + 5u32]).cast::<f32>();
        let x6 = load(x[x_row_base + k_first + 6u32]).cast::<f32>();
        let x7 = load(x[x_row_base + k_first + 7u32]).cast::<f32>();
        // 16 hand-unrolled m-cells: each block computes one dot product and
        // adds directly to its accumulator — no select, no branch.
        let wrb0 = weight_expert_base + (m_base + 0u32) * total_packs;
        let srb0 = scale_expert_base + (m_base + 0u32) * groups_per_row;
        let p0 = load(weight_packed[wrb0 + pack_idx]);
        let s0 = load(scales[srb0 + g]).cast::<f32>();
        let b0 = load(biases[srb0 + g]).cast::<f32>();
        let dot0 = ((p0 >> 0u32) & 15u32).cast::<f32>() * s0 * x0
            + b0 * x0
            + ((p0 >> 4u32) & 15u32).cast::<f32>() * s0 * x1
            + b0 * x1
            + ((p0 >> 8u32) & 15u32).cast::<f32>() * s0 * x2
            + b0 * x2
            + ((p0 >> 12u32) & 15u32).cast::<f32>() * s0 * x3
            + b0 * x3
            + ((p0 >> 16u32) & 15u32).cast::<f32>() * s0 * x4
            + b0 * x4
            + ((p0 >> 20u32) & 15u32).cast::<f32>() * s0 * x5
            + b0 * x5
            + ((p0 >> 24u32) & 15u32).cast::<f32>() * s0 * x6
            + b0 * x6
            + ((p0 >> 28u32) & 15u32).cast::<f32>() * s0 * x7
            + b0 * x7;
        acc0 = acc0 + dot0;
        let wrb1 = weight_expert_base + (m_base + 1u32) * total_packs;
        let srb1 = scale_expert_base + (m_base + 1u32) * groups_per_row;
        let p1 = load(weight_packed[wrb1 + pack_idx]);
        let s1 = load(scales[srb1 + g]).cast::<f32>();
        let b1 = load(biases[srb1 + g]).cast::<f32>();
        let dot1 = ((p1 >> 0u32) & 15u32).cast::<f32>() * s1 * x0
            + b1 * x0
            + ((p1 >> 4u32) & 15u32).cast::<f32>() * s1 * x1
            + b1 * x1
            + ((p1 >> 8u32) & 15u32).cast::<f32>() * s1 * x2
            + b1 * x2
            + ((p1 >> 12u32) & 15u32).cast::<f32>() * s1 * x3
            + b1 * x3
            + ((p1 >> 16u32) & 15u32).cast::<f32>() * s1 * x4
            + b1 * x4
            + ((p1 >> 20u32) & 15u32).cast::<f32>() * s1 * x5
            + b1 * x5
            + ((p1 >> 24u32) & 15u32).cast::<f32>() * s1 * x6
            + b1 * x6
            + ((p1 >> 28u32) & 15u32).cast::<f32>() * s1 * x7
            + b1 * x7;
        acc1 = acc1 + dot1;
        let wrb2 = weight_expert_base + (m_base + 2u32) * total_packs;
        let srb2 = scale_expert_base + (m_base + 2u32) * groups_per_row;
        let p2 = load(weight_packed[wrb2 + pack_idx]);
        let s2 = load(scales[srb2 + g]).cast::<f32>();
        let b2 = load(biases[srb2 + g]).cast::<f32>();
        let dot2 = ((p2 >> 0u32) & 15u32).cast::<f32>() * s2 * x0
            + b2 * x0
            + ((p2 >> 4u32) & 15u32).cast::<f32>() * s2 * x1
            + b2 * x1
            + ((p2 >> 8u32) & 15u32).cast::<f32>() * s2 * x2
            + b2 * x2
            + ((p2 >> 12u32) & 15u32).cast::<f32>() * s2 * x3
            + b2 * x3
            + ((p2 >> 16u32) & 15u32).cast::<f32>() * s2 * x4
            + b2 * x4
            + ((p2 >> 20u32) & 15u32).cast::<f32>() * s2 * x5
            + b2 * x5
            + ((p2 >> 24u32) & 15u32).cast::<f32>() * s2 * x6
            + b2 * x6
            + ((p2 >> 28u32) & 15u32).cast::<f32>() * s2 * x7
            + b2 * x7;
        acc2 = acc2 + dot2;
        let wrb3 = weight_expert_base + (m_base + 3u32) * total_packs;
        let srb3 = scale_expert_base + (m_base + 3u32) * groups_per_row;
        let p3 = load(weight_packed[wrb3 + pack_idx]);
        let s3 = load(scales[srb3 + g]).cast::<f32>();
        let b3 = load(biases[srb3 + g]).cast::<f32>();
        let dot3 = ((p3 >> 0u32) & 15u32).cast::<f32>() * s3 * x0
            + b3 * x0
            + ((p3 >> 4u32) & 15u32).cast::<f32>() * s3 * x1
            + b3 * x1
            + ((p3 >> 8u32) & 15u32).cast::<f32>() * s3 * x2
            + b3 * x2
            + ((p3 >> 12u32) & 15u32).cast::<f32>() * s3 * x3
            + b3 * x3
            + ((p3 >> 16u32) & 15u32).cast::<f32>() * s3 * x4
            + b3 * x4
            + ((p3 >> 20u32) & 15u32).cast::<f32>() * s3 * x5
            + b3 * x5
            + ((p3 >> 24u32) & 15u32).cast::<f32>() * s3 * x6
            + b3 * x6
            + ((p3 >> 28u32) & 15u32).cast::<f32>() * s3 * x7
            + b3 * x7;
        acc3 = acc3 + dot3;
        let wrb4 = weight_expert_base + (m_base + 4u32) * total_packs;
        let srb4 = scale_expert_base + (m_base + 4u32) * groups_per_row;
        let p4 = load(weight_packed[wrb4 + pack_idx]);
        let s4 = load(scales[srb4 + g]).cast::<f32>();
        let b4 = load(biases[srb4 + g]).cast::<f32>();
        let dot4 = ((p4 >> 0u32) & 15u32).cast::<f32>() * s4 * x0
            + b4 * x0
            + ((p4 >> 4u32) & 15u32).cast::<f32>() * s4 * x1
            + b4 * x1
            + ((p4 >> 8u32) & 15u32).cast::<f32>() * s4 * x2
            + b4 * x2
            + ((p4 >> 12u32) & 15u32).cast::<f32>() * s4 * x3
            + b4 * x3
            + ((p4 >> 16u32) & 15u32).cast::<f32>() * s4 * x4
            + b4 * x4
            + ((p4 >> 20u32) & 15u32).cast::<f32>() * s4 * x5
            + b4 * x5
            + ((p4 >> 24u32) & 15u32).cast::<f32>() * s4 * x6
            + b4 * x6
            + ((p4 >> 28u32) & 15u32).cast::<f32>() * s4 * x7
            + b4 * x7;
        acc4 = acc4 + dot4;
        let wrb5 = weight_expert_base + (m_base + 5u32) * total_packs;
        let srb5 = scale_expert_base + (m_base + 5u32) * groups_per_row;
        let p5 = load(weight_packed[wrb5 + pack_idx]);
        let s5 = load(scales[srb5 + g]).cast::<f32>();
        let b5 = load(biases[srb5 + g]).cast::<f32>();
        let dot5 = ((p5 >> 0u32) & 15u32).cast::<f32>() * s5 * x0
            + b5 * x0
            + ((p5 >> 4u32) & 15u32).cast::<f32>() * s5 * x1
            + b5 * x1
            + ((p5 >> 8u32) & 15u32).cast::<f32>() * s5 * x2
            + b5 * x2
            + ((p5 >> 12u32) & 15u32).cast::<f32>() * s5 * x3
            + b5 * x3
            + ((p5 >> 16u32) & 15u32).cast::<f32>() * s5 * x4
            + b5 * x4
            + ((p5 >> 20u32) & 15u32).cast::<f32>() * s5 * x5
            + b5 * x5
            + ((p5 >> 24u32) & 15u32).cast::<f32>() * s5 * x6
            + b5 * x6
            + ((p5 >> 28u32) & 15u32).cast::<f32>() * s5 * x7
            + b5 * x7;
        acc5 = acc5 + dot5;
        let wrb6 = weight_expert_base + (m_base + 6u32) * total_packs;
        let srb6 = scale_expert_base + (m_base + 6u32) * groups_per_row;
        let p6 = load(weight_packed[wrb6 + pack_idx]);
        let s6 = load(scales[srb6 + g]).cast::<f32>();
        let b6 = load(biases[srb6 + g]).cast::<f32>();
        let dot6 = ((p6 >> 0u32) & 15u32).cast::<f32>() * s6 * x0
            + b6 * x0
            + ((p6 >> 4u32) & 15u32).cast::<f32>() * s6 * x1
            + b6 * x1
            + ((p6 >> 8u32) & 15u32).cast::<f32>() * s6 * x2
            + b6 * x2
            + ((p6 >> 12u32) & 15u32).cast::<f32>() * s6 * x3
            + b6 * x3
            + ((p6 >> 16u32) & 15u32).cast::<f32>() * s6 * x4
            + b6 * x4
            + ((p6 >> 20u32) & 15u32).cast::<f32>() * s6 * x5
            + b6 * x5
            + ((p6 >> 24u32) & 15u32).cast::<f32>() * s6 * x6
            + b6 * x6
            + ((p6 >> 28u32) & 15u32).cast::<f32>() * s6 * x7
            + b6 * x7;
        acc6 = acc6 + dot6;
        let wrb7 = weight_expert_base + (m_base + 7u32) * total_packs;
        let srb7 = scale_expert_base + (m_base + 7u32) * groups_per_row;
        let p7 = load(weight_packed[wrb7 + pack_idx]);
        let s7 = load(scales[srb7 + g]).cast::<f32>();
        let b7 = load(biases[srb7 + g]).cast::<f32>();
        let dot7 = ((p7 >> 0u32) & 15u32).cast::<f32>() * s7 * x0
            + b7 * x0
            + ((p7 >> 4u32) & 15u32).cast::<f32>() * s7 * x1
            + b7 * x1
            + ((p7 >> 8u32) & 15u32).cast::<f32>() * s7 * x2
            + b7 * x2
            + ((p7 >> 12u32) & 15u32).cast::<f32>() * s7 * x3
            + b7 * x3
            + ((p7 >> 16u32) & 15u32).cast::<f32>() * s7 * x4
            + b7 * x4
            + ((p7 >> 20u32) & 15u32).cast::<f32>() * s7 * x5
            + b7 * x5
            + ((p7 >> 24u32) & 15u32).cast::<f32>() * s7 * x6
            + b7 * x6
            + ((p7 >> 28u32) & 15u32).cast::<f32>() * s7 * x7
            + b7 * x7;
        acc7 = acc7 + dot7;
        let wrb8 = weight_expert_base + (m_base + 8u32) * total_packs;
        let srb8 = scale_expert_base + (m_base + 8u32) * groups_per_row;
        let p8 = load(weight_packed[wrb8 + pack_idx]);
        let s8 = load(scales[srb8 + g]).cast::<f32>();
        let b8 = load(biases[srb8 + g]).cast::<f32>();
        let dot8 = ((p8 >> 0u32) & 15u32).cast::<f32>() * s8 * x0
            + b8 * x0
            + ((p8 >> 4u32) & 15u32).cast::<f32>() * s8 * x1
            + b8 * x1
            + ((p8 >> 8u32) & 15u32).cast::<f32>() * s8 * x2
            + b8 * x2
            + ((p8 >> 12u32) & 15u32).cast::<f32>() * s8 * x3
            + b8 * x3
            + ((p8 >> 16u32) & 15u32).cast::<f32>() * s8 * x4
            + b8 * x4
            + ((p8 >> 20u32) & 15u32).cast::<f32>() * s8 * x5
            + b8 * x5
            + ((p8 >> 24u32) & 15u32).cast::<f32>() * s8 * x6
            + b8 * x6
            + ((p8 >> 28u32) & 15u32).cast::<f32>() * s8 * x7
            + b8 * x7;
        acc8 = acc8 + dot8;
        let wrb9 = weight_expert_base + (m_base + 9u32) * total_packs;
        let srb9 = scale_expert_base + (m_base + 9u32) * groups_per_row;
        let p9 = load(weight_packed[wrb9 + pack_idx]);
        let s9 = load(scales[srb9 + g]).cast::<f32>();
        let b9 = load(biases[srb9 + g]).cast::<f32>();
        let dot9 = ((p9 >> 0u32) & 15u32).cast::<f32>() * s9 * x0
            + b9 * x0
            + ((p9 >> 4u32) & 15u32).cast::<f32>() * s9 * x1
            + b9 * x1
            + ((p9 >> 8u32) & 15u32).cast::<f32>() * s9 * x2
            + b9 * x2
            + ((p9 >> 12u32) & 15u32).cast::<f32>() * s9 * x3
            + b9 * x3
            + ((p9 >> 16u32) & 15u32).cast::<f32>() * s9 * x4
            + b9 * x4
            + ((p9 >> 20u32) & 15u32).cast::<f32>() * s9 * x5
            + b9 * x5
            + ((p9 >> 24u32) & 15u32).cast::<f32>() * s9 * x6
            + b9 * x6
            + ((p9 >> 28u32) & 15u32).cast::<f32>() * s9 * x7
            + b9 * x7;
        acc9 = acc9 + dot9;
        let wrb10 = weight_expert_base + (m_base + 10u32) * total_packs;
        let srb10 = scale_expert_base + (m_base + 10u32) * groups_per_row;
        let p10 = load(weight_packed[wrb10 + pack_idx]);
        let s10 = load(scales[srb10 + g]).cast::<f32>();
        let b10 = load(biases[srb10 + g]).cast::<f32>();
        let dot10 = ((p10 >> 0u32) & 15u32).cast::<f32>() * s10 * x0
            + b10 * x0
            + ((p10 >> 4u32) & 15u32).cast::<f32>() * s10 * x1
            + b10 * x1
            + ((p10 >> 8u32) & 15u32).cast::<f32>() * s10 * x2
            + b10 * x2
            + ((p10 >> 12u32) & 15u32).cast::<f32>() * s10 * x3
            + b10 * x3
            + ((p10 >> 16u32) & 15u32).cast::<f32>() * s10 * x4
            + b10 * x4
            + ((p10 >> 20u32) & 15u32).cast::<f32>() * s10 * x5
            + b10 * x5
            + ((p10 >> 24u32) & 15u32).cast::<f32>() * s10 * x6
            + b10 * x6
            + ((p10 >> 28u32) & 15u32).cast::<f32>() * s10 * x7
            + b10 * x7;
        acc10 = acc10 + dot10;
        let wrb11 = weight_expert_base + (m_base + 11u32) * total_packs;
        let srb11 = scale_expert_base + (m_base + 11u32) * groups_per_row;
        let p11 = load(weight_packed[wrb11 + pack_idx]);
        let s11 = load(scales[srb11 + g]).cast::<f32>();
        let b11 = load(biases[srb11 + g]).cast::<f32>();
        let dot11 = ((p11 >> 0u32) & 15u32).cast::<f32>() * s11 * x0
            + b11 * x0
            + ((p11 >> 4u32) & 15u32).cast::<f32>() * s11 * x1
            + b11 * x1
            + ((p11 >> 8u32) & 15u32).cast::<f32>() * s11 * x2
            + b11 * x2
            + ((p11 >> 12u32) & 15u32).cast::<f32>() * s11 * x3
            + b11 * x3
            + ((p11 >> 16u32) & 15u32).cast::<f32>() * s11 * x4
            + b11 * x4
            + ((p11 >> 20u32) & 15u32).cast::<f32>() * s11 * x5
            + b11 * x5
            + ((p11 >> 24u32) & 15u32).cast::<f32>() * s11 * x6
            + b11 * x6
            + ((p11 >> 28u32) & 15u32).cast::<f32>() * s11 * x7
            + b11 * x7;
        acc11 = acc11 + dot11;
        let wrb12 = weight_expert_base + (m_base + 12u32) * total_packs;
        let srb12 = scale_expert_base + (m_base + 12u32) * groups_per_row;
        let p12 = load(weight_packed[wrb12 + pack_idx]);
        let s12 = load(scales[srb12 + g]).cast::<f32>();
        let b12 = load(biases[srb12 + g]).cast::<f32>();
        let dot12 = ((p12 >> 0u32) & 15u32).cast::<f32>() * s12 * x0
            + b12 * x0
            + ((p12 >> 4u32) & 15u32).cast::<f32>() * s12 * x1
            + b12 * x1
            + ((p12 >> 8u32) & 15u32).cast::<f32>() * s12 * x2
            + b12 * x2
            + ((p12 >> 12u32) & 15u32).cast::<f32>() * s12 * x3
            + b12 * x3
            + ((p12 >> 16u32) & 15u32).cast::<f32>() * s12 * x4
            + b12 * x4
            + ((p12 >> 20u32) & 15u32).cast::<f32>() * s12 * x5
            + b12 * x5
            + ((p12 >> 24u32) & 15u32).cast::<f32>() * s12 * x6
            + b12 * x6
            + ((p12 >> 28u32) & 15u32).cast::<f32>() * s12 * x7
            + b12 * x7;
        acc12 = acc12 + dot12;
        let wrb13 = weight_expert_base + (m_base + 13u32) * total_packs;
        let srb13 = scale_expert_base + (m_base + 13u32) * groups_per_row;
        let p13 = load(weight_packed[wrb13 + pack_idx]);
        let s13 = load(scales[srb13 + g]).cast::<f32>();
        let b13 = load(biases[srb13 + g]).cast::<f32>();
        let dot13 = ((p13 >> 0u32) & 15u32).cast::<f32>() * s13 * x0
            + b13 * x0
            + ((p13 >> 4u32) & 15u32).cast::<f32>() * s13 * x1
            + b13 * x1
            + ((p13 >> 8u32) & 15u32).cast::<f32>() * s13 * x2
            + b13 * x2
            + ((p13 >> 12u32) & 15u32).cast::<f32>() * s13 * x3
            + b13 * x3
            + ((p13 >> 16u32) & 15u32).cast::<f32>() * s13 * x4
            + b13 * x4
            + ((p13 >> 20u32) & 15u32).cast::<f32>() * s13 * x5
            + b13 * x5
            + ((p13 >> 24u32) & 15u32).cast::<f32>() * s13 * x6
            + b13 * x6
            + ((p13 >> 28u32) & 15u32).cast::<f32>() * s13 * x7
            + b13 * x7;
        acc13 = acc13 + dot13;
        let wrb14 = weight_expert_base + (m_base + 14u32) * total_packs;
        let srb14 = scale_expert_base + (m_base + 14u32) * groups_per_row;
        let p14 = load(weight_packed[wrb14 + pack_idx]);
        let s14 = load(scales[srb14 + g]).cast::<f32>();
        let b14 = load(biases[srb14 + g]).cast::<f32>();
        let dot14 = ((p14 >> 0u32) & 15u32).cast::<f32>() * s14 * x0
            + b14 * x0
            + ((p14 >> 4u32) & 15u32).cast::<f32>() * s14 * x1
            + b14 * x1
            + ((p14 >> 8u32) & 15u32).cast::<f32>() * s14 * x2
            + b14 * x2
            + ((p14 >> 12u32) & 15u32).cast::<f32>() * s14 * x3
            + b14 * x3
            + ((p14 >> 16u32) & 15u32).cast::<f32>() * s14 * x4
            + b14 * x4
            + ((p14 >> 20u32) & 15u32).cast::<f32>() * s14 * x5
            + b14 * x5
            + ((p14 >> 24u32) & 15u32).cast::<f32>() * s14 * x6
            + b14 * x6
            + ((p14 >> 28u32) & 15u32).cast::<f32>() * s14 * x7
            + b14 * x7;
        acc14 = acc14 + dot14;
        let wrb15 = weight_expert_base + (m_base + 15u32) * total_packs;
        let srb15 = scale_expert_base + (m_base + 15u32) * groups_per_row;
        let p15 = load(weight_packed[wrb15 + pack_idx]);
        let s15 = load(scales[srb15 + g]).cast::<f32>();
        let b15 = load(biases[srb15 + g]).cast::<f32>();
        let dot15 = ((p15 >> 0u32) & 15u32).cast::<f32>() * s15 * x0
            + b15 * x0
            + ((p15 >> 4u32) & 15u32).cast::<f32>() * s15 * x1
            + b15 * x1
            + ((p15 >> 8u32) & 15u32).cast::<f32>() * s15 * x2
            + b15 * x2
            + ((p15 >> 12u32) & 15u32).cast::<f32>() * s15 * x3
            + b15 * x3
            + ((p15 >> 16u32) & 15u32).cast::<f32>() * s15 * x4
            + b15 * x4
            + ((p15 >> 20u32) & 15u32).cast::<f32>() * s15 * x5
            + b15 * x5
            + ((p15 >> 24u32) & 15u32).cast::<f32>() * s15 * x6
            + b15 * x6
            + ((p15 >> 28u32) & 15u32).cast::<f32>() * s15 * x7
            + b15 * x7;
        acc15 = acc15 + dot15;
    }
    let t0 = simd_sum(acc0);
    let t1 = simd_sum(acc1);
    let t2 = simd_sum(acc2);
    let t3 = simd_sum(acc3);
    let t4 = simd_sum(acc4);
    let t5 = simd_sum(acc5);
    let t6 = simd_sum(acc6);
    let t7 = simd_sum(acc7);
    let t8 = simd_sum(acc8);
    let t9 = simd_sum(acc9);
    let t10 = simd_sum(acc10);
    let t11 = simd_sum(acc11);
    let t12 = simd_sum(acc12);
    let t13 = simd_sum(acc13);
    let t14 = simd_sum(acc14);
    let t15 = simd_sum(acc15);
    if lane == 0u32 {
        store(out[row * m_out + m_base + 0u32], t0.cast::<T>());
        store(out[row * m_out + m_base + 1u32], t1.cast::<T>());
        store(out[row * m_out + m_base + 2u32], t2.cast::<T>());
        store(out[row * m_out + m_base + 3u32], t3.cast::<T>());
        store(out[row * m_out + m_base + 4u32], t4.cast::<T>());
        store(out[row * m_out + m_base + 5u32], t5.cast::<T>());
        store(out[row * m_out + m_base + 6u32], t6.cast::<T>());
        store(out[row * m_out + m_base + 7u32], t7.cast::<T>());
        store(out[row * m_out + m_base + 8u32], t8.cast::<T>());
        store(out[row * m_out + m_base + 9u32], t9.cast::<T>());
        store(out[row * m_out + m_base + 10u32], t10.cast::<T>());
        store(out[row * m_out + m_base + 11u32], t11.cast::<T>());
        store(out[row * m_out + m_base + 12u32], t12.cast::<T>());
        store(out[row * m_out + m_base + 13u32], t13.cast::<T>());
        store(out[row * m_out + m_base + 14u32], t14.cast::<T>());
        store(out[row * m_out + m_base + 15u32], t15.cast::<T>());
    }
}

// ── mt_moe_gather_qmm_int4_m32 ────────────────────────────────────────────
//
// Same pattern as `mt_moe_gather_qmm_int4_m8`, extended to 32
// adjacent `m_out` cells per row. 32× fewer TGs → 32× less dispatch +
// scheduler overhead, and `x[row, k]` reads serve 32 dot products.
//
// DISPATCH:
//   Grid = [m_out / 32, T_rows, 1]   (m_out must be a multiple of 32)
//   TG   = [32, 1, 1]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_int4_m32<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m_chunk = tgid_x;
    let row = tgid_y;
    let lane = tid;
    let m_base = m_chunk * 32u32;
    // Resolve expert — same linear walk as the m=1 variant.
    let mut expert = 0u32;
    let mut found = 0u32;
    for ee in range(0u32, n_experts, 1u32) {
        let end = load(expert_offsets[ee + 1u32]);
        let inside_bool = row < end;
        let inside = select(inside_bool, 1u32, 0u32);
        let take = inside * (1u32 - found);
        expert = select(take == 1u32, ee, expert);
        found = select(take == 1u32, 1u32, found);
    }
    let total_packs = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let weight_expert_base = expert * m_out * total_packs;
    let scale_expert_base = expert * m_out * groups_per_row;
    let x_row_base = row * k_in;
    // 32 separate accumulators, one per m-cell in the chunk.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut acc8 = 0.0f32;
    let mut acc9 = 0.0f32;
    let mut acc10 = 0.0f32;
    let mut acc11 = 0.0f32;
    let mut acc12 = 0.0f32;
    let mut acc13 = 0.0f32;
    let mut acc14 = 0.0f32;
    let mut acc15 = 0.0f32;
    let mut acc16 = 0.0f32;
    let mut acc17 = 0.0f32;
    let mut acc18 = 0.0f32;
    let mut acc19 = 0.0f32;
    let mut acc20 = 0.0f32;
    let mut acc21 = 0.0f32;
    let mut acc22 = 0.0f32;
    let mut acc23 = 0.0f32;
    let mut acc24 = 0.0f32;
    let mut acc25 = 0.0f32;
    let mut acc26 = 0.0f32;
    let mut acc27 = 0.0f32;
    let mut acc28 = 0.0f32;
    let mut acc29 = 0.0f32;
    let mut acc30 = 0.0f32;
    let mut acc31 = 0.0f32;
    for pack_idx in range(lane, total_packs, 32u32) {
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        // Load 8 input values once — reused across 32 m-cells.
        let x0 = load(x[x_row_base + k_first + 0u32]).cast::<f32>();
        let x1 = load(x[x_row_base + k_first + 1u32]).cast::<f32>();
        let x2 = load(x[x_row_base + k_first + 2u32]).cast::<f32>();
        let x3 = load(x[x_row_base + k_first + 3u32]).cast::<f32>();
        let x4 = load(x[x_row_base + k_first + 4u32]).cast::<f32>();
        let x5 = load(x[x_row_base + k_first + 5u32]).cast::<f32>();
        let x6 = load(x[x_row_base + k_first + 6u32]).cast::<f32>();
        let x7 = load(x[x_row_base + k_first + 7u32]).cast::<f32>();
        // 32 hand-unrolled m-cells: each block computes one dot product and
        // adds directly to its accumulator — no select, no branch.
        let wrb0 = weight_expert_base + (m_base + 0u32) * total_packs;
        let srb0 = scale_expert_base + (m_base + 0u32) * groups_per_row;
        let p0 = load(weight_packed[wrb0 + pack_idx]);
        let s0 = load(scales[srb0 + g]).cast::<f32>();
        let b0 = load(biases[srb0 + g]).cast::<f32>();
        let dot0 = ((p0 >> 0u32) & 15u32).cast::<f32>() * s0 * x0
            + b0 * x0
            + ((p0 >> 4u32) & 15u32).cast::<f32>() * s0 * x1
            + b0 * x1
            + ((p0 >> 8u32) & 15u32).cast::<f32>() * s0 * x2
            + b0 * x2
            + ((p0 >> 12u32) & 15u32).cast::<f32>() * s0 * x3
            + b0 * x3
            + ((p0 >> 16u32) & 15u32).cast::<f32>() * s0 * x4
            + b0 * x4
            + ((p0 >> 20u32) & 15u32).cast::<f32>() * s0 * x5
            + b0 * x5
            + ((p0 >> 24u32) & 15u32).cast::<f32>() * s0 * x6
            + b0 * x6
            + ((p0 >> 28u32) & 15u32).cast::<f32>() * s0 * x7
            + b0 * x7;
        acc0 = acc0 + dot0;
        let wrb1 = weight_expert_base + (m_base + 1u32) * total_packs;
        let srb1 = scale_expert_base + (m_base + 1u32) * groups_per_row;
        let p1 = load(weight_packed[wrb1 + pack_idx]);
        let s1 = load(scales[srb1 + g]).cast::<f32>();
        let b1 = load(biases[srb1 + g]).cast::<f32>();
        let dot1 = ((p1 >> 0u32) & 15u32).cast::<f32>() * s1 * x0
            + b1 * x0
            + ((p1 >> 4u32) & 15u32).cast::<f32>() * s1 * x1
            + b1 * x1
            + ((p1 >> 8u32) & 15u32).cast::<f32>() * s1 * x2
            + b1 * x2
            + ((p1 >> 12u32) & 15u32).cast::<f32>() * s1 * x3
            + b1 * x3
            + ((p1 >> 16u32) & 15u32).cast::<f32>() * s1 * x4
            + b1 * x4
            + ((p1 >> 20u32) & 15u32).cast::<f32>() * s1 * x5
            + b1 * x5
            + ((p1 >> 24u32) & 15u32).cast::<f32>() * s1 * x6
            + b1 * x6
            + ((p1 >> 28u32) & 15u32).cast::<f32>() * s1 * x7
            + b1 * x7;
        acc1 = acc1 + dot1;
        let wrb2 = weight_expert_base + (m_base + 2u32) * total_packs;
        let srb2 = scale_expert_base + (m_base + 2u32) * groups_per_row;
        let p2 = load(weight_packed[wrb2 + pack_idx]);
        let s2 = load(scales[srb2 + g]).cast::<f32>();
        let b2 = load(biases[srb2 + g]).cast::<f32>();
        let dot2 = ((p2 >> 0u32) & 15u32).cast::<f32>() * s2 * x0
            + b2 * x0
            + ((p2 >> 4u32) & 15u32).cast::<f32>() * s2 * x1
            + b2 * x1
            + ((p2 >> 8u32) & 15u32).cast::<f32>() * s2 * x2
            + b2 * x2
            + ((p2 >> 12u32) & 15u32).cast::<f32>() * s2 * x3
            + b2 * x3
            + ((p2 >> 16u32) & 15u32).cast::<f32>() * s2 * x4
            + b2 * x4
            + ((p2 >> 20u32) & 15u32).cast::<f32>() * s2 * x5
            + b2 * x5
            + ((p2 >> 24u32) & 15u32).cast::<f32>() * s2 * x6
            + b2 * x6
            + ((p2 >> 28u32) & 15u32).cast::<f32>() * s2 * x7
            + b2 * x7;
        acc2 = acc2 + dot2;
        let wrb3 = weight_expert_base + (m_base + 3u32) * total_packs;
        let srb3 = scale_expert_base + (m_base + 3u32) * groups_per_row;
        let p3 = load(weight_packed[wrb3 + pack_idx]);
        let s3 = load(scales[srb3 + g]).cast::<f32>();
        let b3 = load(biases[srb3 + g]).cast::<f32>();
        let dot3 = ((p3 >> 0u32) & 15u32).cast::<f32>() * s3 * x0
            + b3 * x0
            + ((p3 >> 4u32) & 15u32).cast::<f32>() * s3 * x1
            + b3 * x1
            + ((p3 >> 8u32) & 15u32).cast::<f32>() * s3 * x2
            + b3 * x2
            + ((p3 >> 12u32) & 15u32).cast::<f32>() * s3 * x3
            + b3 * x3
            + ((p3 >> 16u32) & 15u32).cast::<f32>() * s3 * x4
            + b3 * x4
            + ((p3 >> 20u32) & 15u32).cast::<f32>() * s3 * x5
            + b3 * x5
            + ((p3 >> 24u32) & 15u32).cast::<f32>() * s3 * x6
            + b3 * x6
            + ((p3 >> 28u32) & 15u32).cast::<f32>() * s3 * x7
            + b3 * x7;
        acc3 = acc3 + dot3;
        let wrb4 = weight_expert_base + (m_base + 4u32) * total_packs;
        let srb4 = scale_expert_base + (m_base + 4u32) * groups_per_row;
        let p4 = load(weight_packed[wrb4 + pack_idx]);
        let s4 = load(scales[srb4 + g]).cast::<f32>();
        let b4 = load(biases[srb4 + g]).cast::<f32>();
        let dot4 = ((p4 >> 0u32) & 15u32).cast::<f32>() * s4 * x0
            + b4 * x0
            + ((p4 >> 4u32) & 15u32).cast::<f32>() * s4 * x1
            + b4 * x1
            + ((p4 >> 8u32) & 15u32).cast::<f32>() * s4 * x2
            + b4 * x2
            + ((p4 >> 12u32) & 15u32).cast::<f32>() * s4 * x3
            + b4 * x3
            + ((p4 >> 16u32) & 15u32).cast::<f32>() * s4 * x4
            + b4 * x4
            + ((p4 >> 20u32) & 15u32).cast::<f32>() * s4 * x5
            + b4 * x5
            + ((p4 >> 24u32) & 15u32).cast::<f32>() * s4 * x6
            + b4 * x6
            + ((p4 >> 28u32) & 15u32).cast::<f32>() * s4 * x7
            + b4 * x7;
        acc4 = acc4 + dot4;
        let wrb5 = weight_expert_base + (m_base + 5u32) * total_packs;
        let srb5 = scale_expert_base + (m_base + 5u32) * groups_per_row;
        let p5 = load(weight_packed[wrb5 + pack_idx]);
        let s5 = load(scales[srb5 + g]).cast::<f32>();
        let b5 = load(biases[srb5 + g]).cast::<f32>();
        let dot5 = ((p5 >> 0u32) & 15u32).cast::<f32>() * s5 * x0
            + b5 * x0
            + ((p5 >> 4u32) & 15u32).cast::<f32>() * s5 * x1
            + b5 * x1
            + ((p5 >> 8u32) & 15u32).cast::<f32>() * s5 * x2
            + b5 * x2
            + ((p5 >> 12u32) & 15u32).cast::<f32>() * s5 * x3
            + b5 * x3
            + ((p5 >> 16u32) & 15u32).cast::<f32>() * s5 * x4
            + b5 * x4
            + ((p5 >> 20u32) & 15u32).cast::<f32>() * s5 * x5
            + b5 * x5
            + ((p5 >> 24u32) & 15u32).cast::<f32>() * s5 * x6
            + b5 * x6
            + ((p5 >> 28u32) & 15u32).cast::<f32>() * s5 * x7
            + b5 * x7;
        acc5 = acc5 + dot5;
        let wrb6 = weight_expert_base + (m_base + 6u32) * total_packs;
        let srb6 = scale_expert_base + (m_base + 6u32) * groups_per_row;
        let p6 = load(weight_packed[wrb6 + pack_idx]);
        let s6 = load(scales[srb6 + g]).cast::<f32>();
        let b6 = load(biases[srb6 + g]).cast::<f32>();
        let dot6 = ((p6 >> 0u32) & 15u32).cast::<f32>() * s6 * x0
            + b6 * x0
            + ((p6 >> 4u32) & 15u32).cast::<f32>() * s6 * x1
            + b6 * x1
            + ((p6 >> 8u32) & 15u32).cast::<f32>() * s6 * x2
            + b6 * x2
            + ((p6 >> 12u32) & 15u32).cast::<f32>() * s6 * x3
            + b6 * x3
            + ((p6 >> 16u32) & 15u32).cast::<f32>() * s6 * x4
            + b6 * x4
            + ((p6 >> 20u32) & 15u32).cast::<f32>() * s6 * x5
            + b6 * x5
            + ((p6 >> 24u32) & 15u32).cast::<f32>() * s6 * x6
            + b6 * x6
            + ((p6 >> 28u32) & 15u32).cast::<f32>() * s6 * x7
            + b6 * x7;
        acc6 = acc6 + dot6;
        let wrb7 = weight_expert_base + (m_base + 7u32) * total_packs;
        let srb7 = scale_expert_base + (m_base + 7u32) * groups_per_row;
        let p7 = load(weight_packed[wrb7 + pack_idx]);
        let s7 = load(scales[srb7 + g]).cast::<f32>();
        let b7 = load(biases[srb7 + g]).cast::<f32>();
        let dot7 = ((p7 >> 0u32) & 15u32).cast::<f32>() * s7 * x0
            + b7 * x0
            + ((p7 >> 4u32) & 15u32).cast::<f32>() * s7 * x1
            + b7 * x1
            + ((p7 >> 8u32) & 15u32).cast::<f32>() * s7 * x2
            + b7 * x2
            + ((p7 >> 12u32) & 15u32).cast::<f32>() * s7 * x3
            + b7 * x3
            + ((p7 >> 16u32) & 15u32).cast::<f32>() * s7 * x4
            + b7 * x4
            + ((p7 >> 20u32) & 15u32).cast::<f32>() * s7 * x5
            + b7 * x5
            + ((p7 >> 24u32) & 15u32).cast::<f32>() * s7 * x6
            + b7 * x6
            + ((p7 >> 28u32) & 15u32).cast::<f32>() * s7 * x7
            + b7 * x7;
        acc7 = acc7 + dot7;
        let wrb8 = weight_expert_base + (m_base + 8u32) * total_packs;
        let srb8 = scale_expert_base + (m_base + 8u32) * groups_per_row;
        let p8 = load(weight_packed[wrb8 + pack_idx]);
        let s8 = load(scales[srb8 + g]).cast::<f32>();
        let b8 = load(biases[srb8 + g]).cast::<f32>();
        let dot8 = ((p8 >> 0u32) & 15u32).cast::<f32>() * s8 * x0
            + b8 * x0
            + ((p8 >> 4u32) & 15u32).cast::<f32>() * s8 * x1
            + b8 * x1
            + ((p8 >> 8u32) & 15u32).cast::<f32>() * s8 * x2
            + b8 * x2
            + ((p8 >> 12u32) & 15u32).cast::<f32>() * s8 * x3
            + b8 * x3
            + ((p8 >> 16u32) & 15u32).cast::<f32>() * s8 * x4
            + b8 * x4
            + ((p8 >> 20u32) & 15u32).cast::<f32>() * s8 * x5
            + b8 * x5
            + ((p8 >> 24u32) & 15u32).cast::<f32>() * s8 * x6
            + b8 * x6
            + ((p8 >> 28u32) & 15u32).cast::<f32>() * s8 * x7
            + b8 * x7;
        acc8 = acc8 + dot8;
        let wrb9 = weight_expert_base + (m_base + 9u32) * total_packs;
        let srb9 = scale_expert_base + (m_base + 9u32) * groups_per_row;
        let p9 = load(weight_packed[wrb9 + pack_idx]);
        let s9 = load(scales[srb9 + g]).cast::<f32>();
        let b9 = load(biases[srb9 + g]).cast::<f32>();
        let dot9 = ((p9 >> 0u32) & 15u32).cast::<f32>() * s9 * x0
            + b9 * x0
            + ((p9 >> 4u32) & 15u32).cast::<f32>() * s9 * x1
            + b9 * x1
            + ((p9 >> 8u32) & 15u32).cast::<f32>() * s9 * x2
            + b9 * x2
            + ((p9 >> 12u32) & 15u32).cast::<f32>() * s9 * x3
            + b9 * x3
            + ((p9 >> 16u32) & 15u32).cast::<f32>() * s9 * x4
            + b9 * x4
            + ((p9 >> 20u32) & 15u32).cast::<f32>() * s9 * x5
            + b9 * x5
            + ((p9 >> 24u32) & 15u32).cast::<f32>() * s9 * x6
            + b9 * x6
            + ((p9 >> 28u32) & 15u32).cast::<f32>() * s9 * x7
            + b9 * x7;
        acc9 = acc9 + dot9;
        let wrb10 = weight_expert_base + (m_base + 10u32) * total_packs;
        let srb10 = scale_expert_base + (m_base + 10u32) * groups_per_row;
        let p10 = load(weight_packed[wrb10 + pack_idx]);
        let s10 = load(scales[srb10 + g]).cast::<f32>();
        let b10 = load(biases[srb10 + g]).cast::<f32>();
        let dot10 = ((p10 >> 0u32) & 15u32).cast::<f32>() * s10 * x0
            + b10 * x0
            + ((p10 >> 4u32) & 15u32).cast::<f32>() * s10 * x1
            + b10 * x1
            + ((p10 >> 8u32) & 15u32).cast::<f32>() * s10 * x2
            + b10 * x2
            + ((p10 >> 12u32) & 15u32).cast::<f32>() * s10 * x3
            + b10 * x3
            + ((p10 >> 16u32) & 15u32).cast::<f32>() * s10 * x4
            + b10 * x4
            + ((p10 >> 20u32) & 15u32).cast::<f32>() * s10 * x5
            + b10 * x5
            + ((p10 >> 24u32) & 15u32).cast::<f32>() * s10 * x6
            + b10 * x6
            + ((p10 >> 28u32) & 15u32).cast::<f32>() * s10 * x7
            + b10 * x7;
        acc10 = acc10 + dot10;
        let wrb11 = weight_expert_base + (m_base + 11u32) * total_packs;
        let srb11 = scale_expert_base + (m_base + 11u32) * groups_per_row;
        let p11 = load(weight_packed[wrb11 + pack_idx]);
        let s11 = load(scales[srb11 + g]).cast::<f32>();
        let b11 = load(biases[srb11 + g]).cast::<f32>();
        let dot11 = ((p11 >> 0u32) & 15u32).cast::<f32>() * s11 * x0
            + b11 * x0
            + ((p11 >> 4u32) & 15u32).cast::<f32>() * s11 * x1
            + b11 * x1
            + ((p11 >> 8u32) & 15u32).cast::<f32>() * s11 * x2
            + b11 * x2
            + ((p11 >> 12u32) & 15u32).cast::<f32>() * s11 * x3
            + b11 * x3
            + ((p11 >> 16u32) & 15u32).cast::<f32>() * s11 * x4
            + b11 * x4
            + ((p11 >> 20u32) & 15u32).cast::<f32>() * s11 * x5
            + b11 * x5
            + ((p11 >> 24u32) & 15u32).cast::<f32>() * s11 * x6
            + b11 * x6
            + ((p11 >> 28u32) & 15u32).cast::<f32>() * s11 * x7
            + b11 * x7;
        acc11 = acc11 + dot11;
        let wrb12 = weight_expert_base + (m_base + 12u32) * total_packs;
        let srb12 = scale_expert_base + (m_base + 12u32) * groups_per_row;
        let p12 = load(weight_packed[wrb12 + pack_idx]);
        let s12 = load(scales[srb12 + g]).cast::<f32>();
        let b12 = load(biases[srb12 + g]).cast::<f32>();
        let dot12 = ((p12 >> 0u32) & 15u32).cast::<f32>() * s12 * x0
            + b12 * x0
            + ((p12 >> 4u32) & 15u32).cast::<f32>() * s12 * x1
            + b12 * x1
            + ((p12 >> 8u32) & 15u32).cast::<f32>() * s12 * x2
            + b12 * x2
            + ((p12 >> 12u32) & 15u32).cast::<f32>() * s12 * x3
            + b12 * x3
            + ((p12 >> 16u32) & 15u32).cast::<f32>() * s12 * x4
            + b12 * x4
            + ((p12 >> 20u32) & 15u32).cast::<f32>() * s12 * x5
            + b12 * x5
            + ((p12 >> 24u32) & 15u32).cast::<f32>() * s12 * x6
            + b12 * x6
            + ((p12 >> 28u32) & 15u32).cast::<f32>() * s12 * x7
            + b12 * x7;
        acc12 = acc12 + dot12;
        let wrb13 = weight_expert_base + (m_base + 13u32) * total_packs;
        let srb13 = scale_expert_base + (m_base + 13u32) * groups_per_row;
        let p13 = load(weight_packed[wrb13 + pack_idx]);
        let s13 = load(scales[srb13 + g]).cast::<f32>();
        let b13 = load(biases[srb13 + g]).cast::<f32>();
        let dot13 = ((p13 >> 0u32) & 15u32).cast::<f32>() * s13 * x0
            + b13 * x0
            + ((p13 >> 4u32) & 15u32).cast::<f32>() * s13 * x1
            + b13 * x1
            + ((p13 >> 8u32) & 15u32).cast::<f32>() * s13 * x2
            + b13 * x2
            + ((p13 >> 12u32) & 15u32).cast::<f32>() * s13 * x3
            + b13 * x3
            + ((p13 >> 16u32) & 15u32).cast::<f32>() * s13 * x4
            + b13 * x4
            + ((p13 >> 20u32) & 15u32).cast::<f32>() * s13 * x5
            + b13 * x5
            + ((p13 >> 24u32) & 15u32).cast::<f32>() * s13 * x6
            + b13 * x6
            + ((p13 >> 28u32) & 15u32).cast::<f32>() * s13 * x7
            + b13 * x7;
        acc13 = acc13 + dot13;
        let wrb14 = weight_expert_base + (m_base + 14u32) * total_packs;
        let srb14 = scale_expert_base + (m_base + 14u32) * groups_per_row;
        let p14 = load(weight_packed[wrb14 + pack_idx]);
        let s14 = load(scales[srb14 + g]).cast::<f32>();
        let b14 = load(biases[srb14 + g]).cast::<f32>();
        let dot14 = ((p14 >> 0u32) & 15u32).cast::<f32>() * s14 * x0
            + b14 * x0
            + ((p14 >> 4u32) & 15u32).cast::<f32>() * s14 * x1
            + b14 * x1
            + ((p14 >> 8u32) & 15u32).cast::<f32>() * s14 * x2
            + b14 * x2
            + ((p14 >> 12u32) & 15u32).cast::<f32>() * s14 * x3
            + b14 * x3
            + ((p14 >> 16u32) & 15u32).cast::<f32>() * s14 * x4
            + b14 * x4
            + ((p14 >> 20u32) & 15u32).cast::<f32>() * s14 * x5
            + b14 * x5
            + ((p14 >> 24u32) & 15u32).cast::<f32>() * s14 * x6
            + b14 * x6
            + ((p14 >> 28u32) & 15u32).cast::<f32>() * s14 * x7
            + b14 * x7;
        acc14 = acc14 + dot14;
        let wrb15 = weight_expert_base + (m_base + 15u32) * total_packs;
        let srb15 = scale_expert_base + (m_base + 15u32) * groups_per_row;
        let p15 = load(weight_packed[wrb15 + pack_idx]);
        let s15 = load(scales[srb15 + g]).cast::<f32>();
        let b15 = load(biases[srb15 + g]).cast::<f32>();
        let dot15 = ((p15 >> 0u32) & 15u32).cast::<f32>() * s15 * x0
            + b15 * x0
            + ((p15 >> 4u32) & 15u32).cast::<f32>() * s15 * x1
            + b15 * x1
            + ((p15 >> 8u32) & 15u32).cast::<f32>() * s15 * x2
            + b15 * x2
            + ((p15 >> 12u32) & 15u32).cast::<f32>() * s15 * x3
            + b15 * x3
            + ((p15 >> 16u32) & 15u32).cast::<f32>() * s15 * x4
            + b15 * x4
            + ((p15 >> 20u32) & 15u32).cast::<f32>() * s15 * x5
            + b15 * x5
            + ((p15 >> 24u32) & 15u32).cast::<f32>() * s15 * x6
            + b15 * x6
            + ((p15 >> 28u32) & 15u32).cast::<f32>() * s15 * x7
            + b15 * x7;
        acc15 = acc15 + dot15;
        let wrb16 = weight_expert_base + (m_base + 16u32) * total_packs;
        let srb16 = scale_expert_base + (m_base + 16u32) * groups_per_row;
        let p16 = load(weight_packed[wrb16 + pack_idx]);
        let s16 = load(scales[srb16 + g]).cast::<f32>();
        let b16 = load(biases[srb16 + g]).cast::<f32>();
        let dot16 = ((p16 >> 0u32) & 15u32).cast::<f32>() * s16 * x0
            + b16 * x0
            + ((p16 >> 4u32) & 15u32).cast::<f32>() * s16 * x1
            + b16 * x1
            + ((p16 >> 8u32) & 15u32).cast::<f32>() * s16 * x2
            + b16 * x2
            + ((p16 >> 12u32) & 15u32).cast::<f32>() * s16 * x3
            + b16 * x3
            + ((p16 >> 16u32) & 15u32).cast::<f32>() * s16 * x4
            + b16 * x4
            + ((p16 >> 20u32) & 15u32).cast::<f32>() * s16 * x5
            + b16 * x5
            + ((p16 >> 24u32) & 15u32).cast::<f32>() * s16 * x6
            + b16 * x6
            + ((p16 >> 28u32) & 15u32).cast::<f32>() * s16 * x7
            + b16 * x7;
        acc16 = acc16 + dot16;
        let wrb17 = weight_expert_base + (m_base + 17u32) * total_packs;
        let srb17 = scale_expert_base + (m_base + 17u32) * groups_per_row;
        let p17 = load(weight_packed[wrb17 + pack_idx]);
        let s17 = load(scales[srb17 + g]).cast::<f32>();
        let b17 = load(biases[srb17 + g]).cast::<f32>();
        let dot17 = ((p17 >> 0u32) & 15u32).cast::<f32>() * s17 * x0
            + b17 * x0
            + ((p17 >> 4u32) & 15u32).cast::<f32>() * s17 * x1
            + b17 * x1
            + ((p17 >> 8u32) & 15u32).cast::<f32>() * s17 * x2
            + b17 * x2
            + ((p17 >> 12u32) & 15u32).cast::<f32>() * s17 * x3
            + b17 * x3
            + ((p17 >> 16u32) & 15u32).cast::<f32>() * s17 * x4
            + b17 * x4
            + ((p17 >> 20u32) & 15u32).cast::<f32>() * s17 * x5
            + b17 * x5
            + ((p17 >> 24u32) & 15u32).cast::<f32>() * s17 * x6
            + b17 * x6
            + ((p17 >> 28u32) & 15u32).cast::<f32>() * s17 * x7
            + b17 * x7;
        acc17 = acc17 + dot17;
        let wrb18 = weight_expert_base + (m_base + 18u32) * total_packs;
        let srb18 = scale_expert_base + (m_base + 18u32) * groups_per_row;
        let p18 = load(weight_packed[wrb18 + pack_idx]);
        let s18 = load(scales[srb18 + g]).cast::<f32>();
        let b18 = load(biases[srb18 + g]).cast::<f32>();
        let dot18 = ((p18 >> 0u32) & 15u32).cast::<f32>() * s18 * x0
            + b18 * x0
            + ((p18 >> 4u32) & 15u32).cast::<f32>() * s18 * x1
            + b18 * x1
            + ((p18 >> 8u32) & 15u32).cast::<f32>() * s18 * x2
            + b18 * x2
            + ((p18 >> 12u32) & 15u32).cast::<f32>() * s18 * x3
            + b18 * x3
            + ((p18 >> 16u32) & 15u32).cast::<f32>() * s18 * x4
            + b18 * x4
            + ((p18 >> 20u32) & 15u32).cast::<f32>() * s18 * x5
            + b18 * x5
            + ((p18 >> 24u32) & 15u32).cast::<f32>() * s18 * x6
            + b18 * x6
            + ((p18 >> 28u32) & 15u32).cast::<f32>() * s18 * x7
            + b18 * x7;
        acc18 = acc18 + dot18;
        let wrb19 = weight_expert_base + (m_base + 19u32) * total_packs;
        let srb19 = scale_expert_base + (m_base + 19u32) * groups_per_row;
        let p19 = load(weight_packed[wrb19 + pack_idx]);
        let s19 = load(scales[srb19 + g]).cast::<f32>();
        let b19 = load(biases[srb19 + g]).cast::<f32>();
        let dot19 = ((p19 >> 0u32) & 15u32).cast::<f32>() * s19 * x0
            + b19 * x0
            + ((p19 >> 4u32) & 15u32).cast::<f32>() * s19 * x1
            + b19 * x1
            + ((p19 >> 8u32) & 15u32).cast::<f32>() * s19 * x2
            + b19 * x2
            + ((p19 >> 12u32) & 15u32).cast::<f32>() * s19 * x3
            + b19 * x3
            + ((p19 >> 16u32) & 15u32).cast::<f32>() * s19 * x4
            + b19 * x4
            + ((p19 >> 20u32) & 15u32).cast::<f32>() * s19 * x5
            + b19 * x5
            + ((p19 >> 24u32) & 15u32).cast::<f32>() * s19 * x6
            + b19 * x6
            + ((p19 >> 28u32) & 15u32).cast::<f32>() * s19 * x7
            + b19 * x7;
        acc19 = acc19 + dot19;
        let wrb20 = weight_expert_base + (m_base + 20u32) * total_packs;
        let srb20 = scale_expert_base + (m_base + 20u32) * groups_per_row;
        let p20 = load(weight_packed[wrb20 + pack_idx]);
        let s20 = load(scales[srb20 + g]).cast::<f32>();
        let b20 = load(biases[srb20 + g]).cast::<f32>();
        let dot20 = ((p20 >> 0u32) & 15u32).cast::<f32>() * s20 * x0
            + b20 * x0
            + ((p20 >> 4u32) & 15u32).cast::<f32>() * s20 * x1
            + b20 * x1
            + ((p20 >> 8u32) & 15u32).cast::<f32>() * s20 * x2
            + b20 * x2
            + ((p20 >> 12u32) & 15u32).cast::<f32>() * s20 * x3
            + b20 * x3
            + ((p20 >> 16u32) & 15u32).cast::<f32>() * s20 * x4
            + b20 * x4
            + ((p20 >> 20u32) & 15u32).cast::<f32>() * s20 * x5
            + b20 * x5
            + ((p20 >> 24u32) & 15u32).cast::<f32>() * s20 * x6
            + b20 * x6
            + ((p20 >> 28u32) & 15u32).cast::<f32>() * s20 * x7
            + b20 * x7;
        acc20 = acc20 + dot20;
        let wrb21 = weight_expert_base + (m_base + 21u32) * total_packs;
        let srb21 = scale_expert_base + (m_base + 21u32) * groups_per_row;
        let p21 = load(weight_packed[wrb21 + pack_idx]);
        let s21 = load(scales[srb21 + g]).cast::<f32>();
        let b21 = load(biases[srb21 + g]).cast::<f32>();
        let dot21 = ((p21 >> 0u32) & 15u32).cast::<f32>() * s21 * x0
            + b21 * x0
            + ((p21 >> 4u32) & 15u32).cast::<f32>() * s21 * x1
            + b21 * x1
            + ((p21 >> 8u32) & 15u32).cast::<f32>() * s21 * x2
            + b21 * x2
            + ((p21 >> 12u32) & 15u32).cast::<f32>() * s21 * x3
            + b21 * x3
            + ((p21 >> 16u32) & 15u32).cast::<f32>() * s21 * x4
            + b21 * x4
            + ((p21 >> 20u32) & 15u32).cast::<f32>() * s21 * x5
            + b21 * x5
            + ((p21 >> 24u32) & 15u32).cast::<f32>() * s21 * x6
            + b21 * x6
            + ((p21 >> 28u32) & 15u32).cast::<f32>() * s21 * x7
            + b21 * x7;
        acc21 = acc21 + dot21;
        let wrb22 = weight_expert_base + (m_base + 22u32) * total_packs;
        let srb22 = scale_expert_base + (m_base + 22u32) * groups_per_row;
        let p22 = load(weight_packed[wrb22 + pack_idx]);
        let s22 = load(scales[srb22 + g]).cast::<f32>();
        let b22 = load(biases[srb22 + g]).cast::<f32>();
        let dot22 = ((p22 >> 0u32) & 15u32).cast::<f32>() * s22 * x0
            + b22 * x0
            + ((p22 >> 4u32) & 15u32).cast::<f32>() * s22 * x1
            + b22 * x1
            + ((p22 >> 8u32) & 15u32).cast::<f32>() * s22 * x2
            + b22 * x2
            + ((p22 >> 12u32) & 15u32).cast::<f32>() * s22 * x3
            + b22 * x3
            + ((p22 >> 16u32) & 15u32).cast::<f32>() * s22 * x4
            + b22 * x4
            + ((p22 >> 20u32) & 15u32).cast::<f32>() * s22 * x5
            + b22 * x5
            + ((p22 >> 24u32) & 15u32).cast::<f32>() * s22 * x6
            + b22 * x6
            + ((p22 >> 28u32) & 15u32).cast::<f32>() * s22 * x7
            + b22 * x7;
        acc22 = acc22 + dot22;
        let wrb23 = weight_expert_base + (m_base + 23u32) * total_packs;
        let srb23 = scale_expert_base + (m_base + 23u32) * groups_per_row;
        let p23 = load(weight_packed[wrb23 + pack_idx]);
        let s23 = load(scales[srb23 + g]).cast::<f32>();
        let b23 = load(biases[srb23 + g]).cast::<f32>();
        let dot23 = ((p23 >> 0u32) & 15u32).cast::<f32>() * s23 * x0
            + b23 * x0
            + ((p23 >> 4u32) & 15u32).cast::<f32>() * s23 * x1
            + b23 * x1
            + ((p23 >> 8u32) & 15u32).cast::<f32>() * s23 * x2
            + b23 * x2
            + ((p23 >> 12u32) & 15u32).cast::<f32>() * s23 * x3
            + b23 * x3
            + ((p23 >> 16u32) & 15u32).cast::<f32>() * s23 * x4
            + b23 * x4
            + ((p23 >> 20u32) & 15u32).cast::<f32>() * s23 * x5
            + b23 * x5
            + ((p23 >> 24u32) & 15u32).cast::<f32>() * s23 * x6
            + b23 * x6
            + ((p23 >> 28u32) & 15u32).cast::<f32>() * s23 * x7
            + b23 * x7;
        acc23 = acc23 + dot23;
        let wrb24 = weight_expert_base + (m_base + 24u32) * total_packs;
        let srb24 = scale_expert_base + (m_base + 24u32) * groups_per_row;
        let p24 = load(weight_packed[wrb24 + pack_idx]);
        let s24 = load(scales[srb24 + g]).cast::<f32>();
        let b24 = load(biases[srb24 + g]).cast::<f32>();
        let dot24 = ((p24 >> 0u32) & 15u32).cast::<f32>() * s24 * x0
            + b24 * x0
            + ((p24 >> 4u32) & 15u32).cast::<f32>() * s24 * x1
            + b24 * x1
            + ((p24 >> 8u32) & 15u32).cast::<f32>() * s24 * x2
            + b24 * x2
            + ((p24 >> 12u32) & 15u32).cast::<f32>() * s24 * x3
            + b24 * x3
            + ((p24 >> 16u32) & 15u32).cast::<f32>() * s24 * x4
            + b24 * x4
            + ((p24 >> 20u32) & 15u32).cast::<f32>() * s24 * x5
            + b24 * x5
            + ((p24 >> 24u32) & 15u32).cast::<f32>() * s24 * x6
            + b24 * x6
            + ((p24 >> 28u32) & 15u32).cast::<f32>() * s24 * x7
            + b24 * x7;
        acc24 = acc24 + dot24;
        let wrb25 = weight_expert_base + (m_base + 25u32) * total_packs;
        let srb25 = scale_expert_base + (m_base + 25u32) * groups_per_row;
        let p25 = load(weight_packed[wrb25 + pack_idx]);
        let s25 = load(scales[srb25 + g]).cast::<f32>();
        let b25 = load(biases[srb25 + g]).cast::<f32>();
        let dot25 = ((p25 >> 0u32) & 15u32).cast::<f32>() * s25 * x0
            + b25 * x0
            + ((p25 >> 4u32) & 15u32).cast::<f32>() * s25 * x1
            + b25 * x1
            + ((p25 >> 8u32) & 15u32).cast::<f32>() * s25 * x2
            + b25 * x2
            + ((p25 >> 12u32) & 15u32).cast::<f32>() * s25 * x3
            + b25 * x3
            + ((p25 >> 16u32) & 15u32).cast::<f32>() * s25 * x4
            + b25 * x4
            + ((p25 >> 20u32) & 15u32).cast::<f32>() * s25 * x5
            + b25 * x5
            + ((p25 >> 24u32) & 15u32).cast::<f32>() * s25 * x6
            + b25 * x6
            + ((p25 >> 28u32) & 15u32).cast::<f32>() * s25 * x7
            + b25 * x7;
        acc25 = acc25 + dot25;
        let wrb26 = weight_expert_base + (m_base + 26u32) * total_packs;
        let srb26 = scale_expert_base + (m_base + 26u32) * groups_per_row;
        let p26 = load(weight_packed[wrb26 + pack_idx]);
        let s26 = load(scales[srb26 + g]).cast::<f32>();
        let b26 = load(biases[srb26 + g]).cast::<f32>();
        let dot26 = ((p26 >> 0u32) & 15u32).cast::<f32>() * s26 * x0
            + b26 * x0
            + ((p26 >> 4u32) & 15u32).cast::<f32>() * s26 * x1
            + b26 * x1
            + ((p26 >> 8u32) & 15u32).cast::<f32>() * s26 * x2
            + b26 * x2
            + ((p26 >> 12u32) & 15u32).cast::<f32>() * s26 * x3
            + b26 * x3
            + ((p26 >> 16u32) & 15u32).cast::<f32>() * s26 * x4
            + b26 * x4
            + ((p26 >> 20u32) & 15u32).cast::<f32>() * s26 * x5
            + b26 * x5
            + ((p26 >> 24u32) & 15u32).cast::<f32>() * s26 * x6
            + b26 * x6
            + ((p26 >> 28u32) & 15u32).cast::<f32>() * s26 * x7
            + b26 * x7;
        acc26 = acc26 + dot26;
        let wrb27 = weight_expert_base + (m_base + 27u32) * total_packs;
        let srb27 = scale_expert_base + (m_base + 27u32) * groups_per_row;
        let p27 = load(weight_packed[wrb27 + pack_idx]);
        let s27 = load(scales[srb27 + g]).cast::<f32>();
        let b27 = load(biases[srb27 + g]).cast::<f32>();
        let dot27 = ((p27 >> 0u32) & 15u32).cast::<f32>() * s27 * x0
            + b27 * x0
            + ((p27 >> 4u32) & 15u32).cast::<f32>() * s27 * x1
            + b27 * x1
            + ((p27 >> 8u32) & 15u32).cast::<f32>() * s27 * x2
            + b27 * x2
            + ((p27 >> 12u32) & 15u32).cast::<f32>() * s27 * x3
            + b27 * x3
            + ((p27 >> 16u32) & 15u32).cast::<f32>() * s27 * x4
            + b27 * x4
            + ((p27 >> 20u32) & 15u32).cast::<f32>() * s27 * x5
            + b27 * x5
            + ((p27 >> 24u32) & 15u32).cast::<f32>() * s27 * x6
            + b27 * x6
            + ((p27 >> 28u32) & 15u32).cast::<f32>() * s27 * x7
            + b27 * x7;
        acc27 = acc27 + dot27;
        let wrb28 = weight_expert_base + (m_base + 28u32) * total_packs;
        let srb28 = scale_expert_base + (m_base + 28u32) * groups_per_row;
        let p28 = load(weight_packed[wrb28 + pack_idx]);
        let s28 = load(scales[srb28 + g]).cast::<f32>();
        let b28 = load(biases[srb28 + g]).cast::<f32>();
        let dot28 = ((p28 >> 0u32) & 15u32).cast::<f32>() * s28 * x0
            + b28 * x0
            + ((p28 >> 4u32) & 15u32).cast::<f32>() * s28 * x1
            + b28 * x1
            + ((p28 >> 8u32) & 15u32).cast::<f32>() * s28 * x2
            + b28 * x2
            + ((p28 >> 12u32) & 15u32).cast::<f32>() * s28 * x3
            + b28 * x3
            + ((p28 >> 16u32) & 15u32).cast::<f32>() * s28 * x4
            + b28 * x4
            + ((p28 >> 20u32) & 15u32).cast::<f32>() * s28 * x5
            + b28 * x5
            + ((p28 >> 24u32) & 15u32).cast::<f32>() * s28 * x6
            + b28 * x6
            + ((p28 >> 28u32) & 15u32).cast::<f32>() * s28 * x7
            + b28 * x7;
        acc28 = acc28 + dot28;
        let wrb29 = weight_expert_base + (m_base + 29u32) * total_packs;
        let srb29 = scale_expert_base + (m_base + 29u32) * groups_per_row;
        let p29 = load(weight_packed[wrb29 + pack_idx]);
        let s29 = load(scales[srb29 + g]).cast::<f32>();
        let b29 = load(biases[srb29 + g]).cast::<f32>();
        let dot29 = ((p29 >> 0u32) & 15u32).cast::<f32>() * s29 * x0
            + b29 * x0
            + ((p29 >> 4u32) & 15u32).cast::<f32>() * s29 * x1
            + b29 * x1
            + ((p29 >> 8u32) & 15u32).cast::<f32>() * s29 * x2
            + b29 * x2
            + ((p29 >> 12u32) & 15u32).cast::<f32>() * s29 * x3
            + b29 * x3
            + ((p29 >> 16u32) & 15u32).cast::<f32>() * s29 * x4
            + b29 * x4
            + ((p29 >> 20u32) & 15u32).cast::<f32>() * s29 * x5
            + b29 * x5
            + ((p29 >> 24u32) & 15u32).cast::<f32>() * s29 * x6
            + b29 * x6
            + ((p29 >> 28u32) & 15u32).cast::<f32>() * s29 * x7
            + b29 * x7;
        acc29 = acc29 + dot29;
        let wrb30 = weight_expert_base + (m_base + 30u32) * total_packs;
        let srb30 = scale_expert_base + (m_base + 30u32) * groups_per_row;
        let p30 = load(weight_packed[wrb30 + pack_idx]);
        let s30 = load(scales[srb30 + g]).cast::<f32>();
        let b30 = load(biases[srb30 + g]).cast::<f32>();
        let dot30 = ((p30 >> 0u32) & 15u32).cast::<f32>() * s30 * x0
            + b30 * x0
            + ((p30 >> 4u32) & 15u32).cast::<f32>() * s30 * x1
            + b30 * x1
            + ((p30 >> 8u32) & 15u32).cast::<f32>() * s30 * x2
            + b30 * x2
            + ((p30 >> 12u32) & 15u32).cast::<f32>() * s30 * x3
            + b30 * x3
            + ((p30 >> 16u32) & 15u32).cast::<f32>() * s30 * x4
            + b30 * x4
            + ((p30 >> 20u32) & 15u32).cast::<f32>() * s30 * x5
            + b30 * x5
            + ((p30 >> 24u32) & 15u32).cast::<f32>() * s30 * x6
            + b30 * x6
            + ((p30 >> 28u32) & 15u32).cast::<f32>() * s30 * x7
            + b30 * x7;
        acc30 = acc30 + dot30;
        let wrb31 = weight_expert_base + (m_base + 31u32) * total_packs;
        let srb31 = scale_expert_base + (m_base + 31u32) * groups_per_row;
        let p31 = load(weight_packed[wrb31 + pack_idx]);
        let s31 = load(scales[srb31 + g]).cast::<f32>();
        let b31 = load(biases[srb31 + g]).cast::<f32>();
        let dot31 = ((p31 >> 0u32) & 15u32).cast::<f32>() * s31 * x0
            + b31 * x0
            + ((p31 >> 4u32) & 15u32).cast::<f32>() * s31 * x1
            + b31 * x1
            + ((p31 >> 8u32) & 15u32).cast::<f32>() * s31 * x2
            + b31 * x2
            + ((p31 >> 12u32) & 15u32).cast::<f32>() * s31 * x3
            + b31 * x3
            + ((p31 >> 16u32) & 15u32).cast::<f32>() * s31 * x4
            + b31 * x4
            + ((p31 >> 20u32) & 15u32).cast::<f32>() * s31 * x5
            + b31 * x5
            + ((p31 >> 24u32) & 15u32).cast::<f32>() * s31 * x6
            + b31 * x6
            + ((p31 >> 28u32) & 15u32).cast::<f32>() * s31 * x7
            + b31 * x7;
        acc31 = acc31 + dot31;
    }
    let t0 = simd_sum(acc0);
    let t1 = simd_sum(acc1);
    let t2 = simd_sum(acc2);
    let t3 = simd_sum(acc3);
    let t4 = simd_sum(acc4);
    let t5 = simd_sum(acc5);
    let t6 = simd_sum(acc6);
    let t7 = simd_sum(acc7);
    let t8 = simd_sum(acc8);
    let t9 = simd_sum(acc9);
    let t10 = simd_sum(acc10);
    let t11 = simd_sum(acc11);
    let t12 = simd_sum(acc12);
    let t13 = simd_sum(acc13);
    let t14 = simd_sum(acc14);
    let t15 = simd_sum(acc15);
    let t16 = simd_sum(acc16);
    let t17 = simd_sum(acc17);
    let t18 = simd_sum(acc18);
    let t19 = simd_sum(acc19);
    let t20 = simd_sum(acc20);
    let t21 = simd_sum(acc21);
    let t22 = simd_sum(acc22);
    let t23 = simd_sum(acc23);
    let t24 = simd_sum(acc24);
    let t25 = simd_sum(acc25);
    let t26 = simd_sum(acc26);
    let t27 = simd_sum(acc27);
    let t28 = simd_sum(acc28);
    let t29 = simd_sum(acc29);
    let t30 = simd_sum(acc30);
    let t31 = simd_sum(acc31);
    if lane == 0u32 {
        store(out[row * m_out + m_base + 0u32], t0.cast::<T>());
        store(out[row * m_out + m_base + 1u32], t1.cast::<T>());
        store(out[row * m_out + m_base + 2u32], t2.cast::<T>());
        store(out[row * m_out + m_base + 3u32], t3.cast::<T>());
        store(out[row * m_out + m_base + 4u32], t4.cast::<T>());
        store(out[row * m_out + m_base + 5u32], t5.cast::<T>());
        store(out[row * m_out + m_base + 6u32], t6.cast::<T>());
        store(out[row * m_out + m_base + 7u32], t7.cast::<T>());
        store(out[row * m_out + m_base + 8u32], t8.cast::<T>());
        store(out[row * m_out + m_base + 9u32], t9.cast::<T>());
        store(out[row * m_out + m_base + 10u32], t10.cast::<T>());
        store(out[row * m_out + m_base + 11u32], t11.cast::<T>());
        store(out[row * m_out + m_base + 12u32], t12.cast::<T>());
        store(out[row * m_out + m_base + 13u32], t13.cast::<T>());
        store(out[row * m_out + m_base + 14u32], t14.cast::<T>());
        store(out[row * m_out + m_base + 15u32], t15.cast::<T>());
        store(out[row * m_out + m_base + 16u32], t16.cast::<T>());
        store(out[row * m_out + m_base + 17u32], t17.cast::<T>());
        store(out[row * m_out + m_base + 18u32], t18.cast::<T>());
        store(out[row * m_out + m_base + 19u32], t19.cast::<T>());
        store(out[row * m_out + m_base + 20u32], t20.cast::<T>());
        store(out[row * m_out + m_base + 21u32], t21.cast::<T>());
        store(out[row * m_out + m_base + 22u32], t22.cast::<T>());
        store(out[row * m_out + m_base + 23u32], t23.cast::<T>());
        store(out[row * m_out + m_base + 24u32], t24.cast::<T>());
        store(out[row * m_out + m_base + 25u32], t25.cast::<T>());
        store(out[row * m_out + m_base + 26u32], t26.cast::<T>());
        store(out[row * m_out + m_base + 27u32], t27.cast::<T>());
        store(out[row * m_out + m_base + 28u32], t28.cast::<T>());
        store(out[row * m_out + m_base + 29u32], t29.cast::<T>());
        store(out[row * m_out + m_base + 30u32], t30.cast::<T>());
        store(out[row * m_out + m_base + 31u32], t31.cast::<T>());
    }
}

// ── mt_moe_gather_qmm_mma_int4 ────────────────────────────────────────────
//
// Tiled-MMA grouped quantized matmul. Mirrors MLX's
// `affine_gather_qmm_rhs_nt` (BM=16, BN=32, BK=32, WM=1, WN=2) — the
// kernel mlx-lm prefill actually dispatches at long context. Designed to
// close the 5.5× prefill gap measured on Qwen3.6-35B-A3B 32K against the
// scalar m8 variant.
//
// Key MLX trick replicated here: when a BM=16 TG-row span crosses
// expert boundaries, the kernel walks `indices[y_row..y_row+16]` and
// emits MULTIPLE matmuls per TG — one per contiguous expert run. No row
// padding required.
//
// Inputs (signature matches MLX gather_qmm semantics):
//   x        — [T, K]            f32/f16/bf16 activations (sorted-by-expert)
//   w        — [E, N, K/8]       uint32 packed int4 (bits=4, transpose=true)
//   scales   — [E, N, K/group]   T
//   biases   — [E, N, K/group]   T
//   indices  — [T]               uint32 per-row expert id (rhs_indices)
//   out      — [T, N]            T
//
// Constexpr:
//   m_total    — T (total rows after permute)
//   n_out      — N (per-expert output dim)
//   k_in       — K
//   group_size — quant group size
//
// DISPATCH INVARIANTS
//   - Mode: Reduction (4 SGs per TG → 128 threads)
//   - Grid: [n_out / 32, ceil(m_total / 16), 1]
//   - TG: [128, 1, 1]
//   - K must be multiple of 32 (Qwen3.6: 2048 / 256, both fit)
//   - N must be multiple of 32 (Qwen3.6: 256 / 2048, both fit)
//   - group_size must divide K
//   - bits = 4 only
//
// We use 4 SGs (vs MLX's 2) because the existing mt_qmm_mma proves the
// 4-SG 2×2 warp grid hits ~95% of MLX throughput on the same 8×8 frag
// path. MoE inherits that geometry.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 4 SGs in 2×2 warp grid (matches mt_qmm_mma).
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // TG memory: X tile [BM=32 × BK=32] and dequant W tile [BN=32 × BK=32].
    // Skew = +4 on BK for bank-conflict avoidance (see mt_qmm_mma).
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG (16×16 sub-tile inside the 32×32 output tile).
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    // Reused per k_inner.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    let w_row_in_tg = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    let m_tile_base = m_tile * 32u32; // first M-row this TG handles
    let n_tile_base = n_tile * 32u32;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    // Walk the TG's BM=32 rows in contiguous expert runs. Up to 32
    // sub-runs (one per row worst case); typically 1-2.
    //
    // No TG broadcast — every lane reads indices[m_tile_base + i] directly.
    // Reads are uniform across lanes (Apple GPU L1 caches small index
    // buffers), so the second-and-later reads are free.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 32u32, 1u32) {
        // Read this run's starting expert with an out-of-range sentinel.
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 32u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end (first row in [sub_offset+1..32) whose expert differs).
        let mut sub_end = 32u32;
        let mut found = 0u32;
        for _ii in range(0u32, 32u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 32u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            // Also stop at sentinel-equivalent: out-of-range rows.
            if (probe < 32u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        // Skip sentinel runs (out-of-range rows) AND past-end iterations.
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 32u32);
        if cur_valid {
            // Per-expert weight + scale/bias base.
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            // For this TG (n_tile), w_n_base is the expert's slab + n column offset.
            let sb_base = sb_expert_base + (n_tile_base + w_row_in_tg) * groups_per_row;
            let w_pack_row_base = w_expert_base + (n_tile_base + w_row_in_tg) * packs_per_row;
            // Reset output frags for this sub-run.
            simdgroup_elem_store(c_f00, 0, 0.0f32);
            simdgroup_elem_store(c_f00, 1, 0.0f32);
            simdgroup_elem_store(c_f01, 0, 0.0f32);
            simdgroup_elem_store(c_f01, 1, 0.0f32);
            simdgroup_elem_store(c_f10, 0, 0.0f32);
            simdgroup_elem_store(c_f10, 1, 0.0f32);
            simdgroup_elem_store(c_f11, 0, 0.0f32);
            simdgroup_elem_store(c_f11, 1, 0.0f32);
            // Inner GEMM over K. Each iteration loads a 32×32 X tile + 32×32 W tile,
            // does 4 k_inner × 4 frags = 16 MMAs.
            for kb in range(0u32, k_in, 32u32) {
                // Coop X load — 128 lanes × 8 contiguous K elements each.
                // x_m_row ∈ 0..32 is the TILE-LOCAL row (NOT sub-run-relative).
                // Only load real x for rows in [sub_offset, sub_end); zero else.
                let tile_row = x_m_row;
                let global_row = m_tile_base + tile_row;
                let x_in_run =
                    (tile_row >= sub_offset) & (tile_row < sub_end) & (global_row < m_total);
                let x_row_dev_base = global_row * k_in + kb + x_k_base;
                let x_ws_base = tile_row * xs_ld + x_k_base;
                let xv0 = select(x_in_run, load(x[x_row_dev_base]).cast::<T>(), 0.0f32.cast::<T>());
                let xv1 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 1u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv2 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 2u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv3 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 3u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv4 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 4u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv5 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 5u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv6 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 6u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv7 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 7u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);
                // Coop W dequant — 128 lanes × 1 pack × 8 nibbles.
                let pack_k_off = kb / 8u32 + pack_in_row;
                let pack = load(w[w_pack_row_base + pack_k_off]);
                let k_off = kb + pack_in_row * 8u32;
                let g = k_off / group_size;
                let s = load(scales[sb_base + g]).cast::<f32>();
                let b = load(biases[sb_base + g]).cast::<f32>();
                let s_16 = 0.0625f32;
                let s_256 = 0.00390625f32;
                let s_4096 = 0.000244140625f32;
                let pack_hi = pack >> 16u32;
                let q0 = (pack & 15u32).cast::<f32>();
                let q1 = (pack & 240u32).cast::<f32>() * s_16;
                let q2 = (pack & 3840u32).cast::<f32>() * s_256;
                let q3 = (pack & 61440u32).cast::<f32>() * s_4096;
                let q4 = (pack_hi & 15u32).cast::<f32>();
                let q5 = (pack_hi & 240u32).cast::<f32>() * s_16;
                let q6 = (pack_hi & 3840u32).cast::<f32>() * s_256;
                let q7 = (pack_hi & 61440u32).cast::<f32>() * s_4096;
                let ws_base = w_row_in_tg * ws_ld + pack_in_row * 8u32;
                threadgroup_store("ws", ws_base, (s * q0 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 1u32, (s * q1 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 2u32, (s * q2 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 3u32, (s * q3 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 4u32, (s * q4 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 5u32, (s * q5 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 6u32, (s * q6 + b).cast::<T>());
                threadgroup_store("ws", ws_base + 7u32, (s * q7 + b).cast::<T>());
                threadgroup_barrier();
                // MMA inner loop — 4 frags × 4 k_inner = 16 MMAs per SG.
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                for k_inner in range(0u32, 4u32, 1u32) {
                    let ki_off = k_inner * 8u32;
                    simdgroup_elem_store(
                        a_f0,
                        0,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f0,
                        1,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        0,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        1,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_elem_store(
                        b_f0,
                        0,
                        threadgroup_load("ws", (col_b0 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f0,
                        1,
                        threadgroup_load("ws", (col_b0 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        0,
                        threadgroup_load("ws", (col_b1 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        1,
                        threadgroup_load("ws", (col_b1 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_matmul(a_f0, b_f0, c_f00);
                    simdgroup_matmul(a_f0, b_f1, c_f01);
                    simdgroup_matmul(a_f1, b_f1, c_f11);
                    simdgroup_matmul(a_f1, b_f0, c_f10);
                    simdgroup_barrier_mem_none();
                }
                threadgroup_barrier();
            }
            // Store the 32×32 output tile back to device memory, masked
            // to [sub_offset, sub_end). 4 lanes per output row (sm * 16 + fm
            // for rows, sn * 16 + fn0/fn1 for cols).
            let out_row_a0 = sm * 16u32 + fm;
            let out_row_a1 = sm * 16u32 + 8u32 + fm;
            let out_col_00 = sn * 16u32 + fn0;
            let out_col_01 = sn * 16u32 + fn1;
            let out_col_10 = sn * 16u32 + 8u32 + fn0;
            let out_col_11 = sn * 16u32 + 8u32 + fn1;
            let r00_0 = simdgroup_elem_load(c_f00, 0);
            let r00_1 = simdgroup_elem_load(c_f00, 1);
            let r01_0 = simdgroup_elem_load(c_f01, 0);
            let r01_1 = simdgroup_elem_load(c_f01, 1);
            let r10_0 = simdgroup_elem_load(c_f10, 0);
            let r10_1 = simdgroup_elem_load(c_f10, 1);
            let r11_0 = simdgroup_elem_load(c_f11, 0);
            let r11_1 = simdgroup_elem_load(c_f11, 1);
            // Output rows are tile-local (0..32). Write only if the frag's row
            // falls in this sub-run AND inside the global m bound.
            let r0_g = m_tile_base + out_row_a0;
            let r0_valid = (out_row_a0 >= sub_offset) & (out_row_a0 < sub_end) & (r0_g < m_total);
            if r0_valid {
                store(out[r0_g * n_out + n_tile_base + out_col_00], r00_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_01], r00_1.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_10], r01_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_11], r01_1.cast::<T>());
            }
            let r1_g = m_tile_base + out_row_a1;
            let r1_valid = (out_row_a1 >= sub_offset) & (out_row_a1 < sub_end) & (r1_g < m_total);
            if r1_valid {
                store(out[r1_g * n_out + n_tile_base + out_col_00], r10_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_01], r10_1.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_10], r11_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_11], r11_1.cast::<T>());
            }
        }
        sub_offset = sub_end;
    }
}

// ── mt_moe_gather_qmm_mma_b{3,5,6,8} — wider-precision MMA gather matmul ──
//
// Bit-width-generalized siblings of `mt_moe_gather_qmm_mma_int4`. Same
// tiled-MMA geometry (BM=BN=BK=32, 4 SGs, 2×2 warp grid, per-TG expert
// sub-runs) and identical signature — the *only* difference is the
// weight coop-dequant: instead of the int4-specific 8-nibble unpack, the
// weight row is treated as a contiguous LSB-first bit-stream and each
// lane extracts 8 codes with the straddle-aware two-word read used by
// `gather_qmm_odd`. That handles every bit-width ≤ 16 — the power-of-2
// widths (8) simply never straddle (`spill == 0`).
//
// `w` layout: `[E, N, k_in*bits/32]` uint32 bit-stream packed.
// `group_size` must divide `k_in`; `pack_in_row*8` group-aligned so the
// per-lane group index is hoistable.
macro_rules! gather_qmm_mma {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            x: Tensor<T>,
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            indices: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] m_total: u32,
            #[constexpr] n_out: u32,
            #[constexpr] k_in: u32,
            #[constexpr] group_size: u32,
        ) {
            let n_tile = tgid_x;
            let m_tile = tgid_y;
            let lane = simd_lane;
            let sg = simd_group_id();
            let sm = sg / 2u32;
            let sn = sg & 1u32;
            let lane_in_tg = sg * 32u32 + lane;

            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;

            threadgroup_alloc("xs", 1152, T);
            threadgroup_alloc("ws", 1152, T);

            let c_f00 = simdgroup_alloc::<f32, 8, 8>();
            let c_f01 = simdgroup_alloc::<f32, 8, 8>();
            let c_f10 = simdgroup_alloc::<f32, 8, 8>();
            let c_f11 = simdgroup_alloc::<f32, 8, 8>();

            let a_f0 = simdgroup_alloc::<T, 8, 8>();
            let a_f1 = simdgroup_alloc::<T, 8, 8>();
            let b_f0 = simdgroup_alloc::<T, 8, 8>();
            let b_f1 = simdgroup_alloc::<T, 8, 8>();

            let w_row_in_tg = lane_in_tg / 4u32;
            let pack_in_row = lane_in_tg & 3u32;
            let x_m_row = lane_in_tg / 4u32;
            let x_k_quad = lane_in_tg & 3u32;
            let x_k_base = x_k_quad * 8u32;

            let xs_ld = 36u32;
            let ws_ld = 36u32;

            let m_tile_base = m_tile * 32u32;
            let n_tile_base = n_tile * 32u32;
            // Bit-stream layout: `k_in*bits/32` uint32 words per weight row.
            let u32_per_row = k_in * $bits / 32u32;
            let groups_per_row = k_in / group_size;

            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 32u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 32u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);

                let mut sub_end = 32u32;
                let mut found = 0u32;
                for _ii in range(0u32, 32u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 32u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 32u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }

                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 32u32);
                if cur_valid {
                    let w_expert_base = cur_expert * n_out * u32_per_row;
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    let sb_base = sb_expert_base + (n_tile_base + w_row_in_tg) * groups_per_row;
                    let w_row_base = w_expert_base + (n_tile_base + w_row_in_tg) * u32_per_row;

                    simdgroup_elem_store(c_f00, 0, 0.0f32);
                    simdgroup_elem_store(c_f00, 1, 0.0f32);
                    simdgroup_elem_store(c_f01, 0, 0.0f32);
                    simdgroup_elem_store(c_f01, 1, 0.0f32);
                    simdgroup_elem_store(c_f10, 0, 0.0f32);
                    simdgroup_elem_store(c_f10, 1, 0.0f32);
                    simdgroup_elem_store(c_f11, 0, 0.0f32);
                    simdgroup_elem_store(c_f11, 1, 0.0f32);

                    for kb in range(0u32, k_in, 32u32) {
                        let tile_row = x_m_row;
                        let global_row = m_tile_base + tile_row;
                        let x_in_run = (tile_row >= sub_offset)
                            & (tile_row < sub_end)
                            & (global_row < m_total);
                        let x_row_dev_base = global_row * k_in + kb + x_k_base;
                        let x_ws_base = tile_row * xs_ld + x_k_base;
                        let xv0 = select(
                            x_in_run,
                            load(x[x_row_dev_base]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv1 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 1u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv2 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 2u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv3 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 3u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv4 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 4u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv5 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 5u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv6 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 6u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        let xv7 = select(
                            x_in_run,
                            load(x[x_row_dev_base + 7u32]).cast::<T>(),
                            0.0f32.cast::<T>(),
                        );
                        threadgroup_store("xs", x_ws_base, xv0);
                        threadgroup_store("xs", x_ws_base + 1u32, xv1);
                        threadgroup_store("xs", x_ws_base + 2u32, xv2);
                        threadgroup_store("xs", x_ws_base + 3u32, xv3);
                        threadgroup_store("xs", x_ws_base + 4u32, xv4);
                        threadgroup_store("xs", x_ws_base + 5u32, xv5);
                        threadgroup_store("xs", x_ws_base + 6u32, xv6);
                        threadgroup_store("xs", x_ws_base + 7u32, xv7);

                        // Coop W dequant — 1 group/lane (the 8-K span
                        // `[pack_in_row*8, +8)` is group-aligned), 8 codes
                        // pulled from the bit-stream with straddle handling.
                        let k0 = kb + pack_in_row * 8u32;
                        let g = k0 / group_size;
                        let s = load(scales[sb_base + g]).cast::<f32>();
                        let b = load(biases[sb_base + g]).cast::<f32>();
                        let ws_base = w_row_in_tg * ws_ld + pack_in_row * 8u32;
                        for _ci in range(0u32, 8u32, 1u32) {
                            let bit_off = (k0 + _ci) * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w[w_row_base + word_idx]);
                            let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                            let w1 = load(w[w_row_base + w1idx]);
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = (lo | hi).cast::<f32>();
                            threadgroup_store("ws", ws_base + _ci, (s * q + b).cast::<T>());
                        }

                        threadgroup_barrier();

                        let row_a0 = sm * 16u32 + fm;
                        let row_a1 = sm * 16u32 + 8u32 + fm;
                        let col_b0 = sn * 16u32;
                        let col_b1 = sn * 16u32 + 8u32;

                        for k_inner in range(0u32, 4u32, 1u32) {
                            let ki_off = k_inner * 8u32;
                            simdgroup_elem_store(
                                a_f0,
                                0,
                                threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn0),
                            );
                            simdgroup_elem_store(
                                a_f0,
                                1,
                                threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn1),
                            );
                            simdgroup_elem_store(
                                a_f1,
                                0,
                                threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn0),
                            );
                            simdgroup_elem_store(
                                a_f1,
                                1,
                                threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn1),
                            );
                            simdgroup_barrier_mem_none();
                            simdgroup_elem_store(
                                b_f0,
                                0,
                                threadgroup_load("ws", (col_b0 + fn0) * ws_ld + ki_off + fm),
                            );
                            simdgroup_elem_store(
                                b_f0,
                                1,
                                threadgroup_load("ws", (col_b0 + fn1) * ws_ld + ki_off + fm),
                            );
                            simdgroup_elem_store(
                                b_f1,
                                0,
                                threadgroup_load("ws", (col_b1 + fn0) * ws_ld + ki_off + fm),
                            );
                            simdgroup_elem_store(
                                b_f1,
                                1,
                                threadgroup_load("ws", (col_b1 + fn1) * ws_ld + ki_off + fm),
                            );
                            simdgroup_barrier_mem_none();
                            simdgroup_matmul(a_f0, b_f0, c_f00);
                            simdgroup_matmul(a_f0, b_f1, c_f01);
                            simdgroup_matmul(a_f1, b_f1, c_f11);
                            simdgroup_matmul(a_f1, b_f0, c_f10);
                            simdgroup_barrier_mem_none();
                        }
                        threadgroup_barrier();
                    }

                    let out_row_a0 = sm * 16u32 + fm;
                    let out_row_a1 = sm * 16u32 + 8u32 + fm;
                    let out_col_00 = sn * 16u32 + fn0;
                    let out_col_01 = sn * 16u32 + fn1;
                    let out_col_10 = sn * 16u32 + 8u32 + fn0;
                    let out_col_11 = sn * 16u32 + 8u32 + fn1;

                    let r00_0 = simdgroup_elem_load(c_f00, 0);
                    let r00_1 = simdgroup_elem_load(c_f00, 1);
                    let r01_0 = simdgroup_elem_load(c_f01, 0);
                    let r01_1 = simdgroup_elem_load(c_f01, 1);
                    let r10_0 = simdgroup_elem_load(c_f10, 0);
                    let r10_1 = simdgroup_elem_load(c_f10, 1);
                    let r11_0 = simdgroup_elem_load(c_f11, 0);
                    let r11_1 = simdgroup_elem_load(c_f11, 1);

                    let r0_g = m_tile_base + out_row_a0;
                    let r0_valid =
                        (out_row_a0 >= sub_offset) & (out_row_a0 < sub_end) & (r0_g < m_total);
                    if r0_valid {
                        store(out[r0_g * n_out + n_tile_base + out_col_00], r00_0.cast::<T>());
                        store(out[r0_g * n_out + n_tile_base + out_col_01], r00_1.cast::<T>());
                        store(out[r0_g * n_out + n_tile_base + out_col_10], r01_0.cast::<T>());
                        store(out[r0_g * n_out + n_tile_base + out_col_11], r01_1.cast::<T>());
                    }
                    let r1_g = m_tile_base + out_row_a1;
                    let r1_valid =
                        (out_row_a1 >= sub_offset) & (out_row_a1 < sub_end) & (r1_g < m_total);
                    if r1_valid {
                        store(out[r1_g * n_out + n_tile_base + out_col_00], r10_0.cast::<T>());
                        store(out[r1_g * n_out + n_tile_base + out_col_01], r10_1.cast::<T>());
                        store(out[r1_g * n_out + n_tile_base + out_col_10], r11_0.cast::<T>());
                        store(out[r1_g * n_out + n_tile_base + out_col_11], r11_1.cast::<T>());
                    }
                }
                sub_offset = sub_end;
            }
        }
    };
}

gather_qmm_mma!(mt_moe_gather_qmm_mma_b3, 3u32, "gather_qmm_mma_b3");
gather_qmm_mma!(mt_moe_gather_qmm_mma_b5, 5u32, "gather_qmm_mma_b5");
gather_qmm_mma!(mt_moe_gather_qmm_mma_b6, 6u32, "gather_qmm_mma_b6");
gather_qmm_mma!(mt_moe_gather_qmm_mma_b8, 8u32, "gather_qmm_mma_b8");

// ── mt_moe_gather_qmm_mma_int4_bm16 ────────────────────────────────────────
//
// Half-height MMA grouped quantized matmul — BM=16 variant of
// `mt_moe_gather_qmm_mma_int4`. Matches MLX's `affine_gather_qmm_rhs_nt`
// at WM=1 WN=2 (2 SGs, 64 tpg).
//
// Rationale: at Qwen3.6-A3B prefill T=1024 × 128 experts, rows_per_expert
// = 8. BM=32 wastes 75% of each MMA tile on zeroed rows (4 sub-runs per
// TG, each padding to BM=32). BM=16 halves the waste to 50% (2 sub-runs
// per TG) AND doubles m-tile parallelism (32→64 m-tiles).
//
// Geometry:
//   tpg = 64 = 2 SG × 32 lanes (WM=1, WN=2 — sm=0, sn=sg)
//   BM = 16, BN = 32, BK = 32 → 16×32 output tile (512 outputs/TG)
//   Grid: [N/32, ceil(M/16), 1]
//   Per SG per K-block: 4 frags × 4 k_inner = 16 MMAs (32 across TG)
//
// Inputs / outputs match the BM=32 sibling — same signature.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4_bm16<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 2 SGs in 1×2 warp grid (WM=1, WN=2).
    let sm = 0u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // TG memory: X tile [BM=16 × (BK=32 +8)] and W tile [BN=32 × (BK=32 +8)].
    // BK_padded = 40 (= BK + 16/sizeof(T) at f16) matches MLX's
    // affine_gather_qmm_rhs_nt padding — breaks the bank-conflict that
    // the +4 skew leaves for fp16 MMA fragment column reads. Cost: a few
    // hundred extra T per TG. Free at our occupancy.
    threadgroup_alloc("xs", 640, T);
    threadgroup_alloc("ws", 1280, T);
    // 4 output frags per SG (16 rows × 16 cols of the 16×32 output tile).
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    // Coop-load lane assignments — X tile is 16×32 = 512 elements,
    // 64 lanes × 8 strides each (= matches mt_qmm_mma_m16).
    // W tile is 32×32 = 1024 elements / 8 nibbles per pack = 128 packs,
    // 64 lanes × 2 packs each.
    let w_row_in_tg = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    // For 2-pack-per-lane W load: pack at lane+64 (second half).
    let w_row_2nd = (64u32 + lane_in_tg) / 4u32;
    let pack_in_row_2nd = (64u32 + lane_in_tg) & 3u32;
    // BK_padded = 40 (= BK + 16/sizeof(T) at f16) matches MLX's
    // affine_gather_qmm_rhs_nt skew — breaks the bank-conflict that the
    // +4 skew leaves for fp16 MMA fragment column reads.
    let xs_ld = 40u32;
    let ws_ld = 40u32;
    let m_tile_base = m_tile * 16u32;
    let n_tile_base = n_tile * 32u32;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    // Walk the TG's BM=16 rows in contiguous expert runs. Up to 16
    // sub-runs (one per row worst case); typical 1-2 at production
    // T*topk shapes.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        // Skip the entire body once we've finished the BM=16 span.
        // Saves 14+ wasted outer iters per TG and their 16×16=256
        // indices probes when only 1-2 sub-runs are needed (production
        // shape T=1024 × 128 experts ≈ 8 rows/expert = 2 sub-runs).
        let mut sub_end = sub_offset;
        let mut cur_expert = 4294967295u32;
        if sub_offset < 16u32 {
            let cur_row = m_tile_base + sub_offset;
            let cur_in_range = cur_row < m_total;
            cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
            sub_end = 16u32;
            let mut found = 0u32;
            for _ii in range(0u32, 16u32, 1u32) {
                let probe = sub_offset + 1u32 + _ii;
                let probe_row = m_tile_base + probe;
                let probe_in_range = (probe < 16u32) & (probe_row < m_total);
                if probe_in_range & (found == 0u32) {
                    let e = load(indices[probe_row]);
                    if e != cur_expert {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                    sub_end = probe;
                    found = 1u32;
                }
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            // Reset 4 frags.
            simdgroup_elem_store(c_f00, 0, 0.0f32);
            simdgroup_elem_store(c_f00, 1, 0.0f32);
            simdgroup_elem_store(c_f01, 0, 0.0f32);
            simdgroup_elem_store(c_f01, 1, 0.0f32);
            simdgroup_elem_store(c_f10, 0, 0.0f32);
            simdgroup_elem_store(c_f10, 1, 0.0f32);
            simdgroup_elem_store(c_f11, 0, 0.0f32);
            simdgroup_elem_store(c_f11, 1, 0.0f32);
            for kb in range(0u32, k_in, 32u32) {
                // X load — 64 lanes × 8 contiguous K elements each (flat
                // index covers all 512 elems of the 16×32 tile).
                let flat0 = lane_in_tg;
                let flat1 = 64u32 + lane_in_tg;
                let flat2 = 128u32 + lane_in_tg;
                let flat3 = 192u32 + lane_in_tg;
                let flat4 = 256u32 + lane_in_tg;
                let flat5 = 320u32 + lane_in_tg;
                let flat6 = 384u32 + lane_in_tg;
                let flat7 = 448u32 + lane_in_tg;
                let mr0 = flat0 / 32u32;
                let mr1 = flat1 / 32u32;
                let mr2 = flat2 / 32u32;
                let mr3 = flat3 / 32u32;
                let mr4 = flat4 / 32u32;
                let mr5 = flat5 / 32u32;
                let mr6 = flat6 / 32u32;
                let mr7 = flat7 / 32u32;
                let kc0 = flat0 & 31u32;
                let kc1 = flat1 & 31u32;
                let kc2 = flat2 & 31u32;
                let kc3 = flat3 & 31u32;
                let kc4 = flat4 & 31u32;
                let kc5 = flat5 & 31u32;
                let kc6 = flat6 & 31u32;
                let kc7 = flat7 & 31u32;
                // Mask-hoist: bare loads in a contiguous run so the
                // Vectorize codegen pass can fuse 4 consecutive Loads
                // into vec4 device loads. Select between {0, load} would
                // insert Cast/Select/Const between Loads and bust the
                // ≤8-op scan window. Mask is applied post-load via
                // multiply (xv * mask_t). The OOB guard uses select on
                // the INDEX (clamp to row 0) so the Load itself is
                // unconditional; row 0 is always in bounds for any
                // m_total >= 1.
                let g0 = select(mr0 + m_tile_base < m_total, mr0 + m_tile_base, 0u32);
                let g1 = select(mr1 + m_tile_base < m_total, mr1 + m_tile_base, 0u32);
                let g2 = select(mr2 + m_tile_base < m_total, mr2 + m_tile_base, 0u32);
                let g3 = select(mr3 + m_tile_base < m_total, mr3 + m_tile_base, 0u32);
                let g4 = select(mr4 + m_tile_base < m_total, mr4 + m_tile_base, 0u32);
                let g5 = select(mr5 + m_tile_base < m_total, mr5 + m_tile_base, 0u32);
                let g6 = select(mr6 + m_tile_base < m_total, mr6 + m_tile_base, 0u32);
                let g7 = select(mr7 + m_tile_base < m_total, mr7 + m_tile_base, 0u32);
                let xv0 = load(x[g0 * k_in + kb + kc0]).cast::<T>();
                let xv1 = load(x[g1 * k_in + kb + kc1]).cast::<T>();
                let xv2 = load(x[g2 * k_in + kb + kc2]).cast::<T>();
                let xv3 = load(x[g3 * k_in + kb + kc3]).cast::<T>();
                let xv4 = load(x[g4 * k_in + kb + kc4]).cast::<T>();
                let xv5 = load(x[g5 * k_in + kb + kc5]).cast::<T>();
                let xv6 = load(x[g6 * k_in + kb + kc6]).cast::<T>();
                let xv7 = load(x[g7 * k_in + kb + kc7]).cast::<T>();
                // Per-row mask: in [sub_offset, sub_end) AND global row valid.
                let g0r = m_tile_base + mr0;
                let g1r = m_tile_base + mr1;
                let g2r = m_tile_base + mr2;
                let g3r = m_tile_base + mr3;
                let g4r = m_tile_base + mr4;
                let g5r = m_tile_base + mr5;
                let g6r = m_tile_base + mr6;
                let g7r = m_tile_base + mr7;
                let m0 =
                    select((mr0 >= sub_offset) & (mr0 < sub_end) & (g0r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m1 =
                    select((mr1 >= sub_offset) & (mr1 < sub_end) & (g1r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m2 =
                    select((mr2 >= sub_offset) & (mr2 < sub_end) & (g2r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m3 =
                    select((mr3 >= sub_offset) & (mr3 < sub_end) & (g3r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m4 =
                    select((mr4 >= sub_offset) & (mr4 < sub_end) & (g4r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m5 =
                    select((mr5 >= sub_offset) & (mr5 < sub_end) & (g5r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m6 =
                    select((mr6 >= sub_offset) & (mr6 < sub_end) & (g6r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                let m7 =
                    select((mr7 >= sub_offset) & (mr7 < sub_end) & (g7r < m_total), 1.0f32, 0.0f32)
                        .cast::<T>();
                threadgroup_store("xs", mr0 * xs_ld + kc0, xv0 * m0);
                threadgroup_store("xs", mr1 * xs_ld + kc1, xv1 * m1);
                threadgroup_store("xs", mr2 * xs_ld + kc2, xv2 * m2);
                threadgroup_store("xs", mr3 * xs_ld + kc3, xv3 * m3);
                threadgroup_store("xs", mr4 * xs_ld + kc4, xv4 * m4);
                threadgroup_store("xs", mr5 * xs_ld + kc5, xv5 * m5);
                threadgroup_store("xs", mr6 * xs_ld + kc6, xv6 * m6);
                threadgroup_store("xs", mr7 * xs_ld + kc7, xv7 * m7);
                // W dequant — 64 lanes × 2 packs each.
                let s_16 = 0.0625f32;
                let s_256 = 0.00390625f32;
                let s_4096 = 0.000244140625f32;
                // Pack 0 — lanes 0..63.
                let pack_row_0 = n_tile_base + w_row_in_tg;
                let pack_dev_0 =
                    w_expert_base + pack_row_0 * packs_per_row + kb / 8u32 + pack_in_row;
                let p0 = load(w[pack_dev_0]);
                let k_off_0 = kb + pack_in_row * 8u32;
                let g_0 = k_off_0 / group_size;
                let sb_base_0 = sb_expert_base + pack_row_0 * groups_per_row;
                let s_0 = load(scales[sb_base_0 + g_0]).cast::<f32>();
                let b_0 = load(biases[sb_base_0 + g_0]).cast::<f32>();
                let hi_0 = p0 >> 16u32;
                let q0_0 = (p0 & 15u32).cast::<f32>();
                let q1_0 = (p0 & 240u32).cast::<f32>() * s_16;
                let q2_0 = (p0 & 3840u32).cast::<f32>() * s_256;
                let q3_0 = (p0 & 61440u32).cast::<f32>() * s_4096;
                let q4_0 = (hi_0 & 15u32).cast::<f32>();
                let q5_0 = (hi_0 & 240u32).cast::<f32>() * s_16;
                let q6_0 = (hi_0 & 3840u32).cast::<f32>() * s_256;
                let q7_0 = (hi_0 & 61440u32).cast::<f32>() * s_4096;
                let wb_0 = w_row_in_tg * ws_ld + pack_in_row * 8u32;
                threadgroup_store("ws", wb_0, (s_0 * q0_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 1u32, (s_0 * q1_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 2u32, (s_0 * q2_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 3u32, (s_0 * q3_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 4u32, (s_0 * q4_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 5u32, (s_0 * q5_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 6u32, (s_0 * q6_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 7u32, (s_0 * q7_0 + b_0).cast::<T>());
                // Pack 1 — lanes 64..127 (second half of 32 rows).
                let pack_row_1 = n_tile_base + w_row_2nd;
                let pack_dev_1 =
                    w_expert_base + pack_row_1 * packs_per_row + kb / 8u32 + pack_in_row_2nd;
                let p1 = load(w[pack_dev_1]);
                let k_off_1 = kb + pack_in_row_2nd * 8u32;
                let g_1 = k_off_1 / group_size;
                let sb_base_1 = sb_expert_base + pack_row_1 * groups_per_row;
                let s_1 = load(scales[sb_base_1 + g_1]).cast::<f32>();
                let b_1 = load(biases[sb_base_1 + g_1]).cast::<f32>();
                let hi_1 = p1 >> 16u32;
                let q0_1 = (p1 & 15u32).cast::<f32>();
                let q1_1 = (p1 & 240u32).cast::<f32>() * s_16;
                let q2_1 = (p1 & 3840u32).cast::<f32>() * s_256;
                let q3_1 = (p1 & 61440u32).cast::<f32>() * s_4096;
                let q4_1 = (hi_1 & 15u32).cast::<f32>();
                let q5_1 = (hi_1 & 240u32).cast::<f32>() * s_16;
                let q6_1 = (hi_1 & 3840u32).cast::<f32>() * s_256;
                let q7_1 = (hi_1 & 61440u32).cast::<f32>() * s_4096;
                let wb_1 = w_row_2nd * ws_ld + pack_in_row_2nd * 8u32;
                threadgroup_store("ws", wb_1, (s_1 * q0_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 1u32, (s_1 * q1_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 2u32, (s_1 * q2_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 3u32, (s_1 * q3_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 4u32, (s_1 * q4_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 5u32, (s_1 * q5_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 6u32, (s_1 * q6_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 7u32, (s_1 * q7_1 + b_1).cast::<T>());
                threadgroup_barrier();
                // MMA inner — 4 frags × 4 k_inner = 16 MMAs per SG.
                // sm=0 (WM=1 → both SGs share rows 0..15).
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                for k_inner in range(0u32, 4u32, 1u32) {
                    let ki_off = k_inner * 8u32;
                    simdgroup_elem_store(
                        a_f0,
                        0,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f0,
                        1,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        0,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        1,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_elem_store(
                        b_f0,
                        0,
                        threadgroup_load("ws", (col_b0 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f0,
                        1,
                        threadgroup_load("ws", (col_b0 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        0,
                        threadgroup_load("ws", (col_b1 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        1,
                        threadgroup_load("ws", (col_b1 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_matmul(a_f0, b_f0, c_f00);
                    simdgroup_matmul(a_f0, b_f1, c_f01);
                    simdgroup_matmul(a_f1, b_f1, c_f11);
                    simdgroup_matmul(a_f1, b_f0, c_f10);
                    simdgroup_barrier_mem_none();
                }
                threadgroup_barrier();
            }
            // Write 4 frags. Mask each row to [sub_offset, sub_end) ∩ m_total.
            let out_row_a0 = sm * 16u32 + fm;
            let out_row_a1 = sm * 16u32 + 8u32 + fm;
            let out_col_00 = sn * 16u32 + fn0;
            let out_col_01 = sn * 16u32 + fn1;
            let out_col_10 = sn * 16u32 + 8u32 + fn0;
            let out_col_11 = sn * 16u32 + 8u32 + fn1;
            let r00_0 = simdgroup_elem_load(c_f00, 0);
            let r00_1 = simdgroup_elem_load(c_f00, 1);
            let r01_0 = simdgroup_elem_load(c_f01, 0);
            let r01_1 = simdgroup_elem_load(c_f01, 1);
            let r10_0 = simdgroup_elem_load(c_f10, 0);
            let r10_1 = simdgroup_elem_load(c_f10, 1);
            let r11_0 = simdgroup_elem_load(c_f11, 0);
            let r11_1 = simdgroup_elem_load(c_f11, 1);
            let r0_g = m_tile_base + out_row_a0;
            let r0_valid = (out_row_a0 >= sub_offset) & (out_row_a0 < sub_end) & (r0_g < m_total);
            if r0_valid {
                store(out[r0_g * n_out + n_tile_base + out_col_00], r00_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_01], r00_1.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_10], r01_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_11], r01_1.cast::<T>());
            }
            let r1_g = m_tile_base + out_row_a1;
            let r1_valid = (out_row_a1 >= sub_offset) & (out_row_a1 < sub_end) & (r1_g < m_total);
            if r1_valid {
                store(out[r1_g * n_out + n_tile_base + out_col_00], r10_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_01], r10_1.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_10], r11_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_11], r11_1.cast::<T>());
            }
        }
        sub_offset = sub_end;
    }
}

// ── mt_moe_gather_qmm_mma_int8 — pack-aligned int8 MoE MMA BGEMM ────────
//
// Simdgroup-matrix MoE BGEMM for int8-quantized weights. Same tiled-MMA
// geometry as `mt_moe_gather_qmm_mma_int4` (BM=BN=BK=32, 4 SGs, 2×2 warp
// grid, per-TG expert sub-runs) — the only difference is the W coop-dequant:
//
//   int4: 128 lanes × 1 pack/lane  × 8 nibbles/pack = 1024 dequant elems
//   int8: 128 lanes × 1 pack/lane  × 4 bytes/pack   = 512 dequant elems
//         → BUT we need BN×BK = 32×32 = 1024 dequant elems per k-tile.
//         So we do 2 passes: lanes 0..127 handle rows 0..31 at pack_col 0,
//         then pack_col 1 handles the second 4 bytes of each row.
//
// `packs_per_row = k_in / 4` (4 bytes/u32 vs 8 nibbles/u32 for int4).
// `w` layout: `[n_experts, n_out, k_in/4]` uint32, each uint32 holding 4
// consecutive signed-byte codes as bytes 0..3 (little-endian bit order).
//
// Dispatch: grid `[N/32, ceil(M/32), 1]`, TG `[128, 1, 1]` (4 SGs).
// Correctness: `tests/moe_gather_qmm_mma_int8_gpu_correctness.rs`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int8<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 4 SGs in 2×2 warp grid (WM=2, WN=2).
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // TG memory: X tile [BM=32 × BK=32+4] and dequant W tile [BN=32 × BK=32+4].
    // Skew +4 for bank-conflict avoidance (same as int4 MMA variant).
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG (16×16 sub-tile inside the 32×32 output tile).
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    // W coop-dequant lane assignments for int8.
    // BN×BK = 32×32 = 1024 elements. Each uint32 holds 4 bytes = 4 dequant vals.
    // → packs_per_BK_slice = BK / 4 = 32 / 4 = 8 packs per BN row.
    // Total packs per k-tile = 32 rows × 8 packs = 256.
    // 128 lanes → each lane handles 256/128 = 2 packs (8 dequant vals each).
    //
    // Lane `i` owns packs at flat indices `i` and `i + 128`:
    //   flat_pack_id = i       → w_row = i / 8,   pack_col = i & 7
    //   flat_pack_id = i + 128 → w_row = (i+128)/8 = i/8 + 16, pack_col same
    //
    // This fills rows 0..15 with pack 0 (via lane 0..127) and
    // rows 16..31 with pack 0 (via lane 0+128..127+128 = effectively the same
    // 128 lanes' second iteration).
    //
    // In practice: first pass covers flat ids 0..127 → rows 0..15, all 8 packs.
    //              second pass covers flat ids 128..255 → rows 16..31, all 8 packs.
    let w_row_0 = lane_in_tg / 8u32; // 0..15 (first BN half)
    let pack_col_0 = lane_in_tg & 7u32; // 0..7 (8 packs × 4 bytes = 32 = BK)
    let w_row_1 = 16u32 + lane_in_tg / 8u32; // 16..31 (second BN half)
    let pack_col_1 = lane_in_tg & 7u32; // same 0..7
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    let m_tile_base = m_tile * 32u32;
    let n_tile_base = n_tile * 32u32;
    // int8: 4 bytes per u32 → packs_per_row = k_in / 4.
    let packs_per_row = k_in / 4u32;
    let groups_per_row = k_in / group_size;
    // Walk the TG's BM=32 rows in contiguous expert sub-runs (identical to int4 MMA).
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 32u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 32u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end (first row in [sub_offset+1..32) whose expert differs).
        let mut sub_end = 32u32;
        let mut found = 0u32;
        for _ii in range(0u32, 32u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 32u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 32u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 32u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            // Per-lane row bases (two rows per lane: rows 0..15 and rows 16..31).
            let sb_base_0 = sb_expert_base + (n_tile_base + w_row_0) * groups_per_row;
            let sb_base_1 = sb_expert_base + (n_tile_base + w_row_1) * groups_per_row;
            let w_pack_row_base_0 = w_expert_base + (n_tile_base + w_row_0) * packs_per_row;
            let w_pack_row_base_1 = w_expert_base + (n_tile_base + w_row_1) * packs_per_row;
            // Reset output frags.
            simdgroup_elem_store(c_f00, 0, 0.0f32);
            simdgroup_elem_store(c_f00, 1, 0.0f32);
            simdgroup_elem_store(c_f01, 0, 0.0f32);
            simdgroup_elem_store(c_f01, 1, 0.0f32);
            simdgroup_elem_store(c_f10, 0, 0.0f32);
            simdgroup_elem_store(c_f10, 1, 0.0f32);
            simdgroup_elem_store(c_f11, 0, 0.0f32);
            simdgroup_elem_store(c_f11, 1, 0.0f32);
            // Inner GEMM over K, BK=32. Each iteration loads 32×32 X tile +
            // 32×32 dequant W tile, then runs 4 k_inner × 4 frags MMAs.
            for kb in range(0u32, k_in, 32u32) {
                // Coop X load — 128 lanes × 8 contiguous K elements each.
                // lane_in_tg covers all 128: 32 rows × 4 quadrants × 8 elems = 1024.
                let tile_row = lane_in_tg / 4u32;
                let global_row = m_tile_base + tile_row;
                let x_k_quad = lane_in_tg & 3u32;
                let x_k_base = x_k_quad * 8u32;
                let x_in_run =
                    (tile_row >= sub_offset) & (tile_row < sub_end) & (global_row < m_total);
                let x_row_dev_base = global_row * k_in + kb + x_k_base;
                let x_ws_base = tile_row * xs_ld + x_k_base;
                let xv0 = select(x_in_run, load(x[x_row_dev_base]).cast::<T>(), 0.0f32.cast::<T>());
                let xv1 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 1u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv2 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 2u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv3 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 3u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv4 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 4u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv5 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 5u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv6 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 6u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                let xv7 = select(
                    x_in_run,
                    load(x[x_row_dev_base + 7u32]).cast::<T>(),
                    0.0f32.cast::<T>(),
                );
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);
                // W int8 dequant — 128 lanes × 2 packs/lane × 4 bytes/pack = 1024 = BN×BK.
                //
                // Pass 0 — lanes 0..127 cover BN rows 0..15, all 8 packs per row:
                //   w_row_0 = lane_in_tg / 8 ∈ 0..15
                //   pack_col_0 = lane_in_tg & 7 ∈ 0..7
                //   k_off_0 = kb + pack_col_0 * 4
                //
                // Pass 1 — same lanes cover BN rows 16..31:
                //   w_row_1 = 16 + lane_in_tg / 8 ∈ 16..31
                //   pack_col_1 = lane_in_tg & 7 ∈ 0..7 (same)
                // Pass 0: rows 0..15.
                let pack_dev_0 = w_pack_row_base_0 + kb / 4u32 + pack_col_0;
                let p0 = load(w[pack_dev_0]);
                let k_off_0 = kb + pack_col_0 * 4u32;
                let g_0 = k_off_0 / group_size;
                let s_0 = load(scales[sb_base_0 + g_0]).cast::<f32>();
                let b_0 = load(biases[sb_base_0 + g_0]).cast::<f32>();
                let q0_0 = (p0 & 255u32).cast::<f32>();
                let q1_0 = ((p0 >> 8u32) & 255u32).cast::<f32>();
                let q2_0 = ((p0 >> 16u32) & 255u32).cast::<f32>();
                let q3_0 = ((p0 >> 24u32) & 255u32).cast::<f32>();
                let wb_0 = w_row_0 * ws_ld + pack_col_0 * 4u32;
                threadgroup_store("ws", wb_0, (s_0 * q0_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 1u32, (s_0 * q1_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 2u32, (s_0 * q2_0 + b_0).cast::<T>());
                threadgroup_store("ws", wb_0 + 3u32, (s_0 * q3_0 + b_0).cast::<T>());
                // Pass 1: rows 16..31.
                let pack_dev_1 = w_pack_row_base_1 + kb / 4u32 + pack_col_1;
                let p1 = load(w[pack_dev_1]);
                let k_off_1 = kb + pack_col_1 * 4u32;
                let g_1 = k_off_1 / group_size;
                let s_1 = load(scales[sb_base_1 + g_1]).cast::<f32>();
                let b_1 = load(biases[sb_base_1 + g_1]).cast::<f32>();
                let q0_1 = (p1 & 255u32).cast::<f32>();
                let q1_1 = ((p1 >> 8u32) & 255u32).cast::<f32>();
                let q2_1 = ((p1 >> 16u32) & 255u32).cast::<f32>();
                let q3_1 = ((p1 >> 24u32) & 255u32).cast::<f32>();
                let wb_1 = w_row_1 * ws_ld + pack_col_1 * 4u32;
                threadgroup_store("ws", wb_1, (s_1 * q0_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 1u32, (s_1 * q1_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 2u32, (s_1 * q2_1 + b_1).cast::<T>());
                threadgroup_store("ws", wb_1 + 3u32, (s_1 * q3_1 + b_1).cast::<T>());
                threadgroup_barrier();
                // MMA inner loop — 4 frags × 4 k_inner = 16 MMAs per SG.
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                for k_inner in range(0u32, 4u32, 1u32) {
                    let ki_off = k_inner * 8u32;
                    simdgroup_elem_store(
                        a_f0,
                        0,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f0,
                        1,
                        threadgroup_load("xs", row_a0 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        0,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn0),
                    );
                    simdgroup_elem_store(
                        a_f1,
                        1,
                        threadgroup_load("xs", row_a1 * xs_ld + ki_off + fn1),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_elem_store(
                        b_f0,
                        0,
                        threadgroup_load("ws", (col_b0 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f0,
                        1,
                        threadgroup_load("ws", (col_b0 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        0,
                        threadgroup_load("ws", (col_b1 + fn0) * ws_ld + ki_off + fm),
                    );
                    simdgroup_elem_store(
                        b_f1,
                        1,
                        threadgroup_load("ws", (col_b1 + fn1) * ws_ld + ki_off + fm),
                    );
                    simdgroup_barrier_mem_none();
                    simdgroup_matmul(a_f0, b_f0, c_f00);
                    simdgroup_matmul(a_f0, b_f1, c_f01);
                    simdgroup_matmul(a_f1, b_f1, c_f11);
                    simdgroup_matmul(a_f1, b_f0, c_f10);
                    simdgroup_barrier_mem_none();
                }
                threadgroup_barrier();
            }
            // Store the 32×32 output tile back to device memory, masked to [sub_offset, sub_end).
            let out_row_a0 = sm * 16u32 + fm;
            let out_row_a1 = sm * 16u32 + 8u32 + fm;
            let out_col_00 = sn * 16u32 + fn0;
            let out_col_01 = sn * 16u32 + fn1;
            let out_col_10 = sn * 16u32 + 8u32 + fn0;
            let out_col_11 = sn * 16u32 + 8u32 + fn1;
            let r00_0 = simdgroup_elem_load(c_f00, 0);
            let r00_1 = simdgroup_elem_load(c_f00, 1);
            let r01_0 = simdgroup_elem_load(c_f01, 0);
            let r01_1 = simdgroup_elem_load(c_f01, 1);
            let r10_0 = simdgroup_elem_load(c_f10, 0);
            let r10_1 = simdgroup_elem_load(c_f10, 1);
            let r11_0 = simdgroup_elem_load(c_f11, 0);
            let r11_1 = simdgroup_elem_load(c_f11, 1);
            let r0_g = m_tile_base + out_row_a0;
            let r0_valid = (out_row_a0 >= sub_offset) & (out_row_a0 < sub_end) & (r0_g < m_total);
            if r0_valid {
                store(out[r0_g * n_out + n_tile_base + out_col_00], r00_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_01], r00_1.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_10], r01_0.cast::<T>());
                store(out[r0_g * n_out + n_tile_base + out_col_11], r01_1.cast::<T>());
            }
            let r1_g = m_tile_base + out_row_a1;
            let r1_valid = (out_row_a1 >= sub_offset) & (out_row_a1 < sub_end) & (r1_g < m_total);
            if r1_valid {
                store(out[r1_g * n_out + n_tile_base + out_col_00], r10_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_01], r10_1.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_10], r11_0.cast::<T>());
                store(out[r1_g * n_out + n_tile_base + out_col_11], r11_1.cast::<T>());
            }
        }
        sub_offset = sub_end;
    }
}

/// New-syntax correctness tests for the MoE grouped-gather quantized matmul
/// family. Only the int4 scalar variants have a clean
/// dequant-then-grouped-matmul oracle keyed by per-row expert routing
/// (`mt_moe_gather_qmm_int4` + the `m8` multi-cell sibling). The wider
/// bit-width variants are covered by the legacy bit-width tests, and the MMA
/// variants are validated against the scalar m1 path via cosine in the legacy
/// GPU tests — both are bench-only here.
///
/// Oracle (mirrors `tests/moe_gather_qmm_gpu_correctness.rs`): resolve each
/// row's expert via the CSR `expert_offsets` array (first `e` where
/// `row < expert_offsets[e+1]`), dequant that expert's int4 weight row
/// (8 nibbles per u32, per-group scale/bias), and dot against the row's input.
/// Inputs are dtype-rounded so the GPU sees exactly what the oracle computes.
///
/// Grid (Reduction mode, one simdgroup per TG):
///   - int4 scalar: `grid_3d(m_out, T, 1, [32,1,1])`
///   - int4 m8:     `grid_3d(m_out/8, T, 1, [32,1,1])`
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Pack a row of int4 codes into u32s (8 nibbles per u32, LSB-first).
    fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
        weights
            .chunks_exact(8)
            .map(|chunk| {
                let mut packed = 0u32;
                for (i, &q) in chunk.iter().enumerate() {
                    packed |= (q & 0xf) << (i * 4);
                }
                packed
            })
            .collect()
    }

    /// CSR-routed dequant-then-matmul reference. `weight_packed` stacks
    /// `[n_experts, m_out, k_in/8]` int4 codes; `scales`/`biases` stack
    /// `[n_experts, m_out, k_in/group_size]`. `expert_offsets` is the
    /// `[n_experts+1]` CSR row-offset array — row `t`'s expert is the first
    /// `e` with `t < expert_offsets[e+1]`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_gather_qmm_int4(
        x: &[f32],
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        expert_offsets: &[u32],
        t_rows: usize,
        k_in: usize,
        m_out: usize,
        n_experts: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let weight_stride_m = k_in / 8;
        let groups_per_row = k_in / group_size;
        let mut out = vec![0.0f32; t_rows * m_out];
        for row in 0..t_rows {
            // Resolve expert: first e where row < expert_offsets[e+1].
            let mut expert = 0usize;
            for e in 0..n_experts {
                if (row as u32) < expert_offsets[e + 1] {
                    expert = e;
                    break;
                }
            }
            for m in 0..m_out {
                let weight_row_base = expert * m_out * weight_stride_m + m * weight_stride_m;
                let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
                let x_row_base = row * k_in;
                let mut acc = 0.0f32;
                for pack_idx in 0..(k_in / 8) {
                    let packed = weight_packed[weight_row_base + pack_idx];
                    let k_first = pack_idx * 8;
                    let g = k_first / group_size;
                    let scale = scales[scale_row_base + g];
                    let bias = biases[scale_row_base + g];
                    for nib in 0..8 {
                        let q = ((packed >> (nib * 4)) & 0xf) as f32;
                        let w = q * scale + bias;
                        acc += w * x[x_row_base + k_first + nib];
                    }
                }
                out[row * m_out + m] = acc;
            }
        }
        out
    }

    /// Shared setup for the int4 scalar / m8 variants. `grid_x` carries each
    /// variant's m-tiling (`m_out` for scalar, `m_out/8` for m8). All share
    /// the same ABI + CSR oracle.
    fn int4_setup(kernel: Kernel, grid_x: u32, dt: DType) -> TestSetup {
        // Small 3-expert case, mirrors the legacy oracle test. m_out=8 is a
        // multiple of 8 so the m8 variant tiles cleanly; k_in=64 a multiple
        // of 32; group_size 32 → 2 groups per row.
        let n_experts = 3usize;
        let k_in = 64usize;
        let m_out = 8usize;
        let group_size = 32usize;
        let t_rows = 6usize;
        // Rows [0..2)→e0, [2..5)→e1, [5..6)→e2.
        let expert_offsets: Vec<u32> = vec![0, 2, 5, 6];

        let mut weight_unpacked = vec![0u32; n_experts * m_out * k_in];
        for (i, w) in weight_unpacked.iter_mut().enumerate() {
            *w = ((i as u32) * 7 + 3) & 0xf;
        }
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

        let n_groups = k_in / group_size;
        let scales_f: Vec<f32> =
            (0..n_experts * m_out * n_groups).map(|i| 0.01 + 0.001 * (i as f32)).collect();
        let biases_f: Vec<f32> =
            (0..n_experts * m_out * n_groups).map(|i| -0.05 + 0.002 * (i as f32)).collect();
        let x_f: Vec<f32> = (0..t_rows * k_in).map(|i| 0.1 * ((i as f32 * 0.17).sin())).collect();

        // Dtype-round the operands the kernel casts through T.
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = cpu_gather_qmm_int4(
            &x,
            &weight_packed,
            &s,
            &b,
            &expert_offsets,
            t_rows,
            k_in,
            m_out,
            n_experts,
            group_size,
        );

        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("weight_packed", u32_bytes(&weight_packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("expert_offsets", u32_bytes(&expert_offsets), DType::U32))
            .input(TestBuffer::zeros("out", t_rows * m_out, dt))
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(grid_x, t_rows as u32, 1, [32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_int4(dt: DType) -> TestSetup {
        int4_setup(mt_moe_gather_qmm_int4::kernel_ir_for(dt), 8, dt)
    }

    // m8: grid_x = m_out / 8 = 8 / 8 = 1 TG of 8 m-cells per x row.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_int4_m8(dt: DType) -> TestSetup {
        int4_setup(mt_moe_gather_qmm_int4_m8::kernel_ir_for(dt), 1, dt)
    }
}

/// New-syntax benchmarks for the full MoE kernel family. Production-ish
/// Qwen3.6-A3B-ish shapes. All Reduction mode; grids mirror each kernel's
/// DISPATCH INVARIANTS (group counts, never total threads).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    // ── Grouped-gather scalar/CSR family (int4 + b{3,5,6,8}) ──────────────
    // ABI: x, weight_packed, scales, biases, expert_offsets, out +
    // {k_in, m_out, n_experts, group_size}. Grid [m_out / m_cells, T, 1],
    // tpg [32,1,1]. `bits` selects the packed-weight word count.
    #[allow(clippy::too_many_arguments)]
    fn csr_bench(
        kernel: Kernel,
        bits: u32,
        m_cells: u32,
        t_rows: usize,
        k_in: usize,
        m_out: usize,
        n_experts: usize,
        group_size: usize,
        dt: DType,
    ) -> BenchSetup {
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in * bits as usize / 32;
        let sz = dt.size_bytes();
        // Active stream: weight slab for the touched experts + scales/biases +
        // x + out. Approximate with the full expert slab (worst case).
        let bytes = n_experts * m_out * words_per_row * 4
            + 2 * n_experts * m_out * groups_per_row * sz
            + t_rows * k_in * sz
            + t_rows * m_out * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", t_rows * k_in, dt))
            .buffer(BenchBuffer::random("weight_packed", n_experts * m_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * m_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * m_out * groups_per_row, dt))
            // CSR offsets must be a sane monotone array; zeros is fine for timing
            // (resolves every row to expert 0) — the walk cost is constexpr-bounded.
            .buffer(BenchBuffer::zeros("expert_offsets", n_experts + 1, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_rows * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "T{t_rows} m{m_out} k{k_in} E{n_experts} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(m_out as u32 / m_cells, t_rows as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/moe/gather_qmm_int4", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_int4(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_int4::kernel_ir_for(dt), 4, 1, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_int4_m8", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_int4_m8(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_int4_m8::kernel_ir_for(dt), 4, 8, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_int4_m16", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_int4_m16(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_int4_m16::kernel_ir_for(dt), 4, 16, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_int4_m32", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_int4_m32(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_int4_m32::kernel_ir_for(dt), 4, 32, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_b8", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_b8(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_b8::kernel_ir_for(dt), 8, 1, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_b3", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_b3(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_b3::kernel_ir_for(dt), 3, 1, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_b5", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_b5(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_b5::kernel_ir_for(dt), 5, 1, 64, 2048, 256, 128, 64, dt)
    }
    #[bench(name = "ffai/moe/gather_qmm_b6", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_b6(dt: DType) -> BenchSetup {
        csr_bench(mt_moe_gather_qmm_b6::kernel_ir_for(dt), 6, 1, 64, 2048, 256, 128, 64, dt)
    }

    // ── Tiled-MMA family (per-row `indices`, no CSR offsets) ──────────────
    // ABI: x, w, scales, biases, indices, out + {m_total, n_out, k_in,
    // group_size}. `bits` selects the packed word count; `bm`/`tpg` carry
    // each variant's tile geometry. Grid [n_out/bn, ceil(m_total/bm), 1].
    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        bits: u32,
        bn: u32,
        bm: u32,
        tpg: u32,
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        group_size: usize,
        dt: DType,
    ) -> BenchSetup {
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in * bits as usize / 32;
        let sz = dt.size_bytes();
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / bn, (m_total as u32).div_ceil(bm), 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    // BM=32 4-SG variants (int4 / b{3,5,6,8} / int8): grid [N/32, ceil(M/32), 1], tpg 128.
    #[bench(name = "ffai/moe/gather_qmm_mma_int4", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int4(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_int4::kernel_ir_for(dt),
            4,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    #[bench(name = "ffai/moe/gather_qmm_mma_b3", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_b3(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_b3::kernel_ir_for(dt),
            3,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    #[bench(name = "ffai/moe/gather_qmm_mma_b5", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_b5(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_b5::kernel_ir_for(dt),
            5,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    #[bench(name = "ffai/moe/gather_qmm_mma_b6", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_b6(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_b6::kernel_ir_for(dt),
            6,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    #[bench(name = "ffai/moe/gather_qmm_mma_b8", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_b8(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_b8::kernel_ir_for(dt),
            8,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    #[bench(name = "ffai/moe/gather_qmm_mma_int8", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int8(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_int8::kernel_ir_for(dt),
            8,
            32,
            32,
            128,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }
    // BM=16 2-SG variant: grid [N/32, ceil(M/16), 1], tpg 64.
    #[bench(name = "ffai/moe/gather_qmm_mma_int4_bm16", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int4_bm16(dt: DType) -> BenchSetup {
        mma_bench(
            mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for(dt),
            4,
            32,
            16,
            64,
            1024,
            256,
            2048,
            128,
            64,
            dt,
        )
    }

    // ── router_topk — data-dependent argmax, bench-only ───────────────────
    // ABI: router_logits, indices_out, weights_out + {n_experts, k,
    // norm_topk_prob}. Grid [B*T, 1, 1], tpg [32,1,1] (pinned in the doc).
    #[bench(name = "ffai/moe/router_topk", dtypes = [f32, f16, bf16])]
    fn bench_moe_router_topk(dt: DType) -> BenchSetup {
        let n_rows = 4096usize; // B*T
        let n_experts = 128usize;
        let k = 8usize;
        let sz = dt.size_bytes();
        let bytes = n_rows * n_experts * sz + n_rows * k * 4 + n_rows * k * sz;
        BenchSetup::new(mt_moe_router_topk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("router_logits", n_rows * n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", n_rows * k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", n_rows * k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .constexpr("norm_topk_prob", 1u32)
            .with_shape_label(format!(
                "BT{n_rows} E{n_experts} k{k} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_rows as u32, 1, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
    }

    // ── permute — pure gather, bench-only ─────────────────────────────────
    // ABI: tokens, sort_token_idx, permuted + {hidden}. Grid [k*B*T, 1, 1],
    // tpg [128,1,1].
    #[bench(name = "ffai/moe/permute", dtypes = [f32, f16, bf16])]
    fn bench_moe_permute(dt: DType) -> BenchSetup {
        let bt = 512usize;
        let k = 8usize;
        let hidden = 2048usize;
        let rows = k * bt;
        let sz = dt.size_bytes();
        // Reads `rows` source rows (worst case all distinct) + writes `rows`.
        let bytes = 2 * rows * hidden * sz;
        BenchSetup::new(mt_moe_permute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("tokens", bt * hidden, dt))
            .buffer(BenchBuffer::zeros("sort_token_idx", rows, DType::U32))
            .buffer(BenchBuffer::zeros("permuted", rows * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .with_shape_label(format!(
                "rows{rows} h{hidden} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(rows as u32, 1, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
    }

    // ── unpermute — weighted scatter-combine, bench-only ──────────────────
    // ABI: expert_outputs, inv_perm, top_k_weights, out + {hidden, k}.
    // Grid [B*T, 1, 1], tpg [128,1,1].
    #[bench(name = "ffai/moe/unpermute", dtypes = [f32, f16, bf16])]
    fn bench_moe_unpermute(dt: DType) -> BenchSetup {
        let bt = 512usize;
        let k = 8usize;
        let hidden = 2048usize;
        let sz = dt.size_bytes();
        let bytes = k * bt * hidden * sz + bt * k * 4 + bt * k * sz + bt * hidden * sz;
        BenchSetup::new(mt_moe_unpermute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("expert_outputs", k * bt * hidden, dt))
            .buffer(BenchBuffer::zeros("inv_perm", bt * k, DType::U32))
            .buffer(BenchBuffer::random("top_k_weights", bt * k, dt))
            .buffer(BenchBuffer::zeros("out", bt * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .with_shape_label(format!(
                "BT{bt} h{hidden} k{k} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(bt as u32, 1, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
