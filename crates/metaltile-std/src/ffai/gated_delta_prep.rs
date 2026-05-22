//! Gated DeltaNet — **fused** prep + decode kernel.
//!
//! `mt_gated_delta_prep_step` extends the recurrence-only
//! [`mt_gated_delta_step`](super::gated_delta::mt_gated_delta_step) by
//! absorbing every host-side prep computation Qwen3.6 / Qwen3.5 currently
//! does between the conv1d and the GDN recurrence:
//!
//!   1. **Conv split** — `conv_out = [q (Hk·Dk), k (Hk·Dk), v (Hv·Dv)]`
//!      is split into q / k / v on the GPU instead of via a host
//!      `toFloatArray() + Swift slicing` roundtrip.
//!   2. **Per-head RMSNorm + scale of q, k** — replaces `perHeadRMSNormScale35`
//!      in `Qwen35GDNMixer.forward`. Scale and (optional) per-head_dim
//!      weights are folded into the same simd-sum over Dk that the kernel
//!      already pays for the recurrence.
//!   3. **g = exp(-exp(A_log) · softplus(a_raw + dt_bias))** — fused.
//!   4. **beta = sigmoid(b_raw)** — fused.
//!   5. The existing recurrence (state decay + delta + outer + read).
//!
//! Net effect on Qwen3.6 decode: one fused GDN kernel per layer instead
//! of `commit()/waitUntilCompleted()` → host arithmetic → `makeCommandBuffer()`
//! → `gatedDeltaStep` dispatch. 30 GDN layers per step × ≥2 host-sync
//! gaps per layer = the bandwidth recovery target for Iter FG2.
//!
//! Inputs that are now GPU-resident:
//!   - `conv_out`     : Tensor<T> [B, 2·Hk·Dk + Hv·Dv]
//!   - `a_log`        : Tensor<T> [Hv]   — per-Hv-head learnable
//!   - `dt_bias`      : Tensor<T> [Hv]
//!   - `a_raw`        : Tensor<T> [B, Hv]
//!   - `b_raw`        : Tensor<T> [B, Hv]
//!   - `q_norm_weight`: Tensor<T> [Hk·Dk]  — pass an all-1×scale vector to
//!     recover the unweighted `perHeadRMSNormScale35` path.
//!   - `k_norm_weight`: Tensor<T> [Hk·Dk]
//!   - `state_in`     : Tensor<T> [B, Hv, Dv, Dk]   (recurrence state)
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, Hv, Dv]
//!
//! ## DISPATCH INVARIANTS (identical to `mt_gated_delta_step`)
//!
//! - **Mode: Reduction.** Each TG is one simdgroup (32 threads).
//! - **Grid: `[Dv, B·Hv, 1]`, TG: `[32, 1, 1]`.**
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` contiguous
//!   slots via `s_idx = n_per_t · dk_idx + i`.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//!
//! ## Per-head RMSNorm redundancy
//!
//! Each (Dv_idx, b, hv) TG re-computes the same q_normed / k_normed for
//! its Hk-group. Cost is `O(Dk)` ALU per TG and is already part of the
//! existing per-lane chunked load anyway — every lane reads its `n_per_t`
//! slice of q/k for the recurrence. The fused kernel just folds the
//! ssq + simd_sum + scale into that same pass and stashes the result on
//! the per-lane stack alongside `decayed` / `k_cache`. fp32 throughout.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Fused GDN prep + recurrence step. See module doc for layout and
/// dispatch invariants. Drop-in replacement for the
/// `host-prep + mt_gated_delta_step` pair in `Qwen35GDNMixer.forward`.
#[kernel]
pub fn mt_gated_delta_prep_step<T>(
    conv_out: Tensor<T>,      // [B, 2·Hk·Dk + Hv·Dv]    q | k | v
    a_log: Tensor<T>,         // [Hv]
    dt_bias: Tensor<T>,       // [Hv]
    a_raw: Tensor<T>,         // [B, Hv]
    b_raw: Tensor<T>,         // [B, Hv]
    q_norm_weight: Tensor<T>, // [Hk·Dk]   pass 1.0×invKeyScale²  for unweighted q-scale path
    k_norm_weight: Tensor<T>, // [Hk·Dk]   pass 1.0×invKeyScale   for unweighted k-scale path
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, Hv, Dv]
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;

    // GQA decomposition: n = b · Hv + hv_idx; hk_idx = hv_idx / (Hv/Hk).
    // Mirrors `mt_gated_delta_step` exactly.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;

    let n_per_t = dk / 32u32;

    // Conv-output flat layout for batch `b`:
    //   q_base = b · (2·Hk·Dk + Hv·Dv)
    //   k_base = q_base + Hk·Dk
    //   v_base = q_base + 2·Hk·Dk
    let stride_b = 2u32 * hk * dk + hv * dv;
    let conv_base = b * stride_b;
    let q_off = conv_base + hk_idx * dk;
    let k_off = conv_base + hk * dk + hk_idx * dk;
    let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;

    // Per-head RMSNorm eps = 1e-6 (matches `perHeadRMSNormScale35`).
    let eps = 0.000001f32;
    let dk_f = dk.cast::<f32>();

    // ─── Phase 0a: Per-head RMSNorm of q / k ─────────────────────────────
    //
    // Each lane reads its `n_per_t` chunk of q and k (Dk-wide, per-head),
    // accumulates a partial ssq, then simd_sum to get the per-head total.
    // The same chunk is also weighted by `q_norm_weight` / `k_norm_weight`
    // and stashed on the per-lane stack so phase 1 / phase 2 read register
    // memory (no second load of conv_out).
    //
    // Cap = 8 (n_per_t @ Dk=256 / 32). At Dk=128, n_per_t=4 — upper 4 slots
    // simply go unread. Same convention as `mt_gated_delta_step`.
    stack_alloc("q_raw", 8u32, "f32");
    stack_alloc("k_raw", 8u32, "f32");
    stack_alloc("q_w", 8u32, "f32");
    stack_alloc("k_w", 8u32, "f32");

    let mut q_ssq = 0.0f32;
    let mut k_ssq = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qv = load(conv_out[q_off + s_idx]).cast::<f32>();
        let kv = load(conv_out[k_off + s_idx]).cast::<f32>();
        stack_store("q_raw", i, qv);
        stack_store("k_raw", i, kv);
        q_ssq = q_ssq + qv * qv;
        k_ssq = k_ssq + kv * kv;
        // Weights are layout `[Hk·Dk]` — same hk_idx slot the q/k row reads.
        let qw = load(q_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let kw = load(k_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        stack_store("q_w", i, qw);
        stack_store("k_w", i, kw);
    }
    let q_ssq_sum = simd_sum(q_ssq);
    let k_ssq_sum = simd_sum(k_ssq);
    // rsqrt(ssq/Dk + eps) = 1 / sqrt(mean + eps). Folds the per-head
    // `scale` parameter directly: caller bakes it into `*_norm_weight`.
    let q_inv = rsqrt(q_ssq_sum / dk_f + eps);
    let k_inv = rsqrt(k_ssq_sum / dk_f + eps);

    // ─── Phase 0b: g / beta from a_log / dt_bias / a_raw / b_raw ─────────
    //
    // Math per Hv-head:
    //   dt   = softplus(a_raw + dt_bias)        (log(1 + exp(·)) form)
    //   g    = exp(-exp(a_log) · dt)
    //   beta = sigmoid(b_raw)
    //
    // softplus is not a DSL primitive — emit `log(exp(x) + 1)` directly.
    // Production values of `a_raw + dt_bias` for Qwen3.6 sit in
    // approximately [-6, +2] (see `Qwen3NextGatedDeltaNet` HF config), so
    // the un-clamped formula stays in fp32 dynamic range. The CPU oracle
    // uses the same formula so the GPU↔CPU diff is purely ULP.
    //
    // Every lane redundantly computes g / beta (scalar broadcast across
    // the simdgroup). The scalar load + 4 math ops cost much less than
    // burning a simd_broadcast plus a barrier.
    let a_log_val = load(a_log[hv_idx]).cast::<f32>();
    let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
    let a_raw_val = load(a_raw[n]).cast::<f32>();
    let b_raw_val = load(b_raw[n]).cast::<f32>();

    // softplus(x)   = log(1 + exp(x))  — un-clamped; production magnitudes
    //                  of (a_raw + dt_bias) sit in fp32 safe range.
    // sigmoid(x)    = 1 / (1 + exp(-x))  — inlined rather than using the
    //                  `Activation::Sigmoid` op because the standard pipeline
    //                  folds Activation into `FusedElementwise` and the
    //                  per-kernel feature analyzer (`needs_sigmoid`) does
    //                  not recurse into fused chains. Inlining keeps the
    //                  emitted MSL self-contained — no `mt_sigmoid` helper
    //                  required.
    let pre_softplus = a_raw_val + dt_bias_val;
    let dt_val = log(exp(pre_softplus) + 1.0f32);
    let g_val = exp(0.0f32 - exp(a_log_val) * dt_val);
    let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));

    // v reads once per Dv slot — no normalization, just dtype-cast.
    let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();

    // ─── Phase 1: decay + kv_mem reduction ───────────────────────────────
    //
    // Same shape as `mt_gated_delta_step::phase_1` but reads q/k from the
    // per-lane `*_normed` stash instead of global. `decayed` and `k_cache`
    // stay register-resident across phases 1/2, same convention.
    let state_base = n * dv * dk + dv_idx * dk;

    stack_alloc("decayed", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");

    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        // Normed k = k_raw * q_inv * weight. RMSNorm formula:
        //   x_normed[d] = x[d] · rsqrt(mean(x²) + eps) · w[d]
        let k_normed = stack_load("k_raw", i) * k_inv * stack_load("k_w", i);
        stack_store("decayed", i, s_decayed);
        stack_store("k_cache", i, k_normed);
        kv_mem = kv_mem + s_decayed * k_normed;
    }
    let kv_mem_sum = simd_sum(kv_mem);

    let delta = (v_val - kv_mem_sum) * beta_val;

    // ─── Phase 2: rank-1 update + output projection ──────────────────────
    let mut out_acc = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = stack_load("decayed", i);
        let k_normed = stack_load("k_cache", i);
        let s_new = s_decayed + k_normed * delta;
        store(state_out[state_base + s_idx], s_new.cast::<T>());
        let q_normed = stack_load("q_raw", i) * q_inv * stack_load("q_w", i);
        out_acc = out_acc + s_new * q_normed;
    }
    let out_sum = simd_sum(out_acc);

    // ─── Phase 3: lane 0 writes y[n, dv_idx] ────────────────────────────
    if dk_idx == 0u32 {
        store(y[n * dv + dv_idx], out_sum.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "gated_delta",
        subop: "prep_step",
        kernel_name: "mt_gated_delta_prep_step",
        kernel_ir: mt_gated_delta_prep_step::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p metaltile-std --lib --release -- ffai::gated_delta_prep::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = mt_gated_delta_prep_step::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
