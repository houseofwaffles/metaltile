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
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "router_topk",
        kernel_name: "mt_moe_router_topk",
        kernel_ir: mt_moe_router_topk::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "unpermute",
        kernel_name: "mt_moe_unpermute",
        kernel_ir: mt_moe_unpermute::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "permute",
        kernel_name: "mt_moe_permute",
        kernel_ir: mt_moe_permute::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0, // exact copy — no numerical drift
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_int4",
        kernel_name: "mt_moe_gather_qmm_int4",
        kernel_ir: mt_moe_gather_qmm_int4::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2, // int4 quant — wide tolerance vs full-precision oracle
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_int4_m8",
        kernel_name: "mt_moe_gather_qmm_int4_m8",
        kernel_ir: mt_moe_gather_qmm_int4_m8::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int4",
        kernel_name: "mt_moe_gather_qmm_mma_int4",
        kernel_ir: mt_moe_gather_qmm_mma_int4::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int4_bm16",
        kernel_name: "mt_moe_gather_qmm_mma_int4_bm16",
        kernel_ir: mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
