//! Gated DeltaNet (GDN) — decode + chunked-prefill kernels.
//!
//! GDN is the recurrent linear-attention variant Qwen3.5 / Qwen3.6 / Qwen3.6-MoE
//! use for their `linear_attention` layers (75% of layers in the hybrid
//! architecture). Two kernels:
//!
//!   - `mt_gated_delta_step`  — single-token decode (`T = 1`)
//!   - `mt_gated_delta_chunk` — multi-token chunked prefill (`T > 1`); the
//!     kernel that actually unblocks ctx > 2048 (issue #111). State stays
//!     register-resident across the inner T loop so the recurrence runs
//!     once per dispatch instead of N independent decode calls.
//!
//! Recurrence per step (matches MLX-LM `_gated_delta_step_ops`):
//!
//!   state_decayed = state * g            // forget-gate decay
//!   kv_mem        = (state_decayed * k).sum(dk)   // [Dv]
//!   delta         = (v - kv_mem) * beta           // [Dv]
//!   state_new     = state_decayed + outer(delta, k)
//!   y             = (state_new * q).sum(dk)       // [Dv]
//!
//! Layouts (matching MLX-LM):
//!
//!   q, k     : [B, Hk, Dk]
//!   v, y     : [B, Hv, Dv]
//!   g, beta  : [B, Hv]
//!   state    : [B, Hv, Dv, Dk]
//!
//! Hk / Hv may differ (GQA-style key-sharing): each Hk-group serves
//! `Hv / Hk` Hv-heads. State is allocated per Hv-head.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** Each threadgroup is one simdgroup (32 threads).
//! - **Grid: `[dv, B * Hv, 1]`, TG: `[32, 1, 1]`.** `tgid_x = dv_idx`,
//!   `tgid_y = n` (the flattened batch×Hv index), `tid = dk_idx` within
//!   the simdgroup (0..32).
//! - **`dk % 32 == 0`.** Each lane owns `n_per_t = dk / 32` contiguous
//!   state elements via `s_idx = n_per_t * dk_idx + i`. TPG = 32 is the
//!   minimum valid value per `docs/developing.md`.
//! - **Hv must be divisible by Hk** (`Hv / Hk` is the number of Hv-heads
//!   per shared (q, k) Hk-group). The kernel computes `hk_idx = hv_idx /
//!   (Hv / Hk)` and reads (q, k) from the shared Hk slot.
//!
//! State accumulator runs in **f32**: the `g * state + outer(delta, k)`
//! recurrence in bf16 drifts after a few dozen decode steps, same
//! reasoning as `ssm_step`. Activations stay in T.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gated_delta",
    subop="step",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn mt_gated_delta_step<T>(
    q: Tensor<T>,             // [B, Hk, Dk]   flat: (b * Hk + hk_idx) * Dk + dk_offset
    k: Tensor<T>,             // [B, Hk, Dk]   same layout as q
    v: Tensor<T>,             // [B, Hv, Dv]   flat: n * Dv + dv_idx  where n = b*Hv + hv_idx
    g: Tensor<T>,             // [B, Hv]       flat: n
    beta: Tensor<T>,          // [B, Hv]       flat: n
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]  flat: n * Dv * Dk + dv_idx * Dk + s_idx
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]  same as state_in
    mut y: Tensor<T>,         // [B, Hv, Dv]   same as v
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;

    // GQA decomposition: n = b * Hv + hv_idx; hk_idx = hv_idx / (Hv / Hk)
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;

    let n_per_t = dk / 32u32;

    let g_val = load(g[n]).cast::<f32>();
    let beta_val = load(beta[n]).cast::<f32>();
    let v_val = load(v[n * dv + dv_idx]).cast::<f32>();

    let qk_base = (b * hk + hk_idx) * dk;
    let state_base = n * dv * dk + dv_idx * dk;

    // ─── Phase 1: decay + kv_mem reduction ─────────────────────────────
    //
    // Per-lane register cache for the decayed state (`decayed`) and the
    // key slice (`k_cache`) — Metal places small fixed-size local arrays
    // in registers, so the inner loops in phase 1 + phase 2 read from
    // registers, not global memory. Replaces the prior "re-read state_in
    // and re-load k twice" pattern.
    //
    // Cap = 8 (n_per_t at the max supported Dk = 256). Smaller Dk just
    // under-utilises the upper slots.
    stack_alloc("decayed", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");

    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        let k_val = load(k[qk_base + s_idx]).cast::<f32>();
        stack_store("decayed", i, s_decayed);
        stack_store("k_cache", i, k_val);
        kv_mem = kv_mem + s_decayed * k_val;
    }
    let kv_mem_sum = simd_sum(kv_mem);

    let delta = (v_val - kv_mem_sum) * beta_val;

    // ─── Phase 2: rank-1 update + output projection ────────────────────
    //
    // Read decayed + k from the per-lane register caches (no global
    // load), apply the rank-1 update, store new state, accumulate
    // output against q. Matches MLX-LM's `float state[n_per_t]`
    // register-array pattern from `mlx_lm/models/gated_delta.py`.
    let mut out = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = stack_load("decayed", i);
        let k_val = stack_load("k_cache", i);
        let s_new = s_decayed + k_val * delta;
        store(state_out[state_base + s_idx], s_new.cast::<T>());
        let q_val = load(q[qk_base + s_idx]).cast::<f32>();
        out = out + s_new * q_val;
    }
    let out_sum = simd_sum(out);

    // ─── Phase 3: lane 0 writes the result ────────────────────────────
    if dk_idx == 0u32 {
        store(y[n * dv + dv_idx], out_sum.cast::<T>());
    }
}

// ────────────────────────────────────────────────────────────────────
//  Chunked-prefill form (T > 1)
// ────────────────────────────────────────────────────────────────────

/// `mt_gated_delta_chunk` — multi-token GDN forward over `T` tokens.
///
/// Same recurrence math as `mt_gated_delta_step`, wrapped in an inner
/// `for t in 0..T` loop. The recurrent state stays in per-lane
/// stack-allocated registers across the entire T sweep, so a single
/// dispatch handles a full chunk of `T` tokens with one set of
/// load_state / store_state passes — vs `T` independent decode dispatches
/// which would each re-load + re-write the state.
///
/// This is the kernel that unblocks Qwen3.6 ctx > 2048: the hybrid
/// scheduler in mlx-swift-lm calls a chunked GDN kernel for the
/// `linear_attention` layers during prefill. The bug in issue #111 is
/// the scheduler currently emits a single chunk of 2048 with no T-loop
/// to span longer prefills; this kernel + a scheduler patch fix it.
///
/// Layouts (matching MLX-LM `_make_gated_delta_kernel`):
///
///   q, k     : [B, T, Hk, Dk]
///   v, y     : [B, T, Hv, Dv]
///   g, beta  : [B, T, Hv]
///   state    : [B, Hv, Dv, Dk]   (one state per (b, hv) — NO T dim;
///                                 state persists across t)
///
/// ## DISPATCH INVARIANTS
///
/// Same dispatch geometry as `mt_gated_delta_step`:
///
/// - **Mode: Reduction.** Each threadgroup is one simdgroup (32 threads).
/// - **Grid: `[dv, B * Hv, 1]`, TG: `[32, 1, 1]`.**
/// - **`dk % 32 == 0`.** Each lane owns `n_per_t = dk / 32` state
///   elements in a stack-allocated register array (cap 8 — Qwen3.6's
///   Dk=256 / 32). State survives across the entire `T`-loop.
/// - **`t_len` is a runtime u32** (passed as a scalar buffer, not a
///   constexpr) so a single PSO compiles for all chunk sizes the
///   scheduler picks.
#[bench_kernel(
    op="gated_delta",
    subop="chunk",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn mt_gated_delta_chunk<T>(
    q: Tensor<T>,             // [B, T, Hk, Dk]
    k: Tensor<T>,             // [B, T, Hk, Dk]
    v: Tensor<T>,             // [B, T, Hv, Dv]
    g: Tensor<T>,             // [B, T, Hv]
    beta: Tensor<T>,          // [B, T, Hv]
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, T, Hv, Dv]
    t_len: Tensor<u32>,       // [1] scalar — number of tokens in this chunk
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;

    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;

    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);

    let state_base = n * dv * dk + dv_idx * dk;

    // ─── Load state into per-lane registers once ─────────────────────
    //
    // State persists across all `T` recurrence steps in registers.
    // `k_cache` is reloaded per-token (each token has its own k row);
    // we don't carry it across t.
    stack_alloc("state_reg", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let val = load(state_in[state_base + s_idx]).cast::<f32>();
        stack_store("state_reg", i, val);
    }

    // ─── Inner T-loop: GDN recurrence per token ──────────────────────
    //
    // Pointer arithmetic per t:
    //   q[t], k[t]: (b * T + t) * Hk * Dk + hk_idx * Dk + s_idx
    //   v[t], y[t]: (b * T + t) * Hv * Dv + hv_idx * Dv + dv_idx
    //   g[t], beta[t]: (b * T + t) * Hv + hv_idx
    for t in range(0u32, t_total, 1u32) {
        let bt = b * t_total + t;
        let qk_base = (bt * hk + hk_idx) * dk;
        let vy_base = (bt * hv + hv_idx) * dv;
        let gbeta_idx = bt * hv + hv_idx;

        let g_val = load(g[gbeta_idx]).cast::<f32>();
        let beta_val = load(beta[gbeta_idx]).cast::<f32>();
        let v_val = load(v[vy_base + dv_idx]).cast::<f32>();

        // Phase 1: decay state + accumulate kv_mem; cache k.
        let mut kv_mem = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let s_old = stack_load("state_reg", i);
            let s_decayed = s_old * g_val;
            stack_store("state_reg", i, s_decayed);
            let k_val = load(k[qk_base + s_idx]).cast::<f32>();
            stack_store("k_cache", i, k_val);
            kv_mem = kv_mem + s_decayed * k_val;
        }
        let kv_mem_sum = simd_sum(kv_mem);
        let delta = (v_val - kv_mem_sum) * beta_val;

        // Phase 2: rank-1 update + output projection.
        let mut out = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let s_decayed = stack_load("state_reg", i);
            let k_val = stack_load("k_cache", i);
            let s_new = s_decayed + k_val * delta;
            stack_store("state_reg", i, s_new);
            let q_val = load(q[qk_base + s_idx]).cast::<f32>();
            out = out + s_new * q_val;
        }
        let out_sum = simd_sum(out);

        if dk_idx == 0u32 {
            store(y[vy_base + dv_idx], out_sum.cast::<T>());
        }
    }

    // ─── Write final state once at the end ──────────────────────────
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
    }
}
