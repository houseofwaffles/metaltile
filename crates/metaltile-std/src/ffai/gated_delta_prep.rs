//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

#[cfg(test)]
mod tests {
    use metaltile_core::ir::KernelMode;

    use super::*;
    use crate::bench_types::DType;

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

/// New-syntax correctness for the fused GDN prep+step kernel
/// (`mt_gated_delta_prep_step`). Oracle is the legacy
/// `gated_delta_prep_step_correctness.rs` reference: CPU prep (conv split →
/// per-head RMSNorm+scale of q/k → `g = exp(-exp(a_log)·softplus(a_raw+dt_bias))`
/// → `beta = sigmoid(b_raw)`) composed with the sequential GDN recurrence. The
/// kernel and oracle use the same un-clamped `softplus = log(exp(x)+1)`, so the
/// only diff is fp32 ULP / dtype rounding. Inputs are dtype-rounded.
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(dv, b*hv, 1, [32,1,1])`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_gated_delta_prep_step;
    use crate::utils::pack_f32;

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// CPU prep: conv_out → (q_normed, k_normed, v_flat, g, beta). Mirrors
    /// `cpu_prep` in the legacy test.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn cpu_prep(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let mut q_normed = vec![0.0_f32; b * hk * dk];
        let mut k_normed = vec![0.0_f32; b * hk * dk];
        let mut v_flat = vec![0.0_f32; b * hv * dv];
        let mut g = vec![0.0_f32; b * hv];
        let mut beta = vec![0.0_f32; b * hv];
        for batch in 0..b {
            let q_base = batch * stride_b;
            let k_base = q_base + hk * dk;
            let v_base = q_base + 2 * hk * dk;
            for hk_idx in 0..hk {
                let row_off = hk_idx * dk;
                let mut q_ssq = 0.0_f32;
                let mut k_ssq = 0.0_f32;
                for d in 0..dk {
                    let qv = conv_out[q_base + row_off + d];
                    let kv = conv_out[k_base + row_off + d];
                    q_ssq += qv * qv;
                    k_ssq += kv * kv;
                }
                let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                for d in 0..dk {
                    let qv = conv_out[q_base + row_off + d];
                    let kv = conv_out[k_base + row_off + d];
                    let qw = q_norm_weight[hk_idx * dk + d];
                    let kw = k_norm_weight[hk_idx * dk + d];
                    q_normed[batch * hk * dk + row_off + d] = qv * q_inv * qw;
                    k_normed[batch * hk * dk + row_off + d] = kv * k_inv * kw;
                }
            }
            for hv_idx in 0..hv {
                for dv_idx in 0..dv {
                    v_flat[(batch * hv + hv_idx) * dv + dv_idx] =
                        conv_out[v_base + hv_idx * dv + dv_idx];
                }
            }
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let dt = softplus_unclamped(a_raw[n] + dt_bias[hv_idx]);
                g[n] = (-a_log[hv_idx].exp() * dt).exp();
                beta[n] = sigmoid(b_raw[n]);
            }
        }
        (q_normed, k_normed, v_flat, g, beta)
    }

    /// CPU GDN recurrence (matches `cpu_step` in the legacy test).
    #[allow(clippy::too_many_arguments)]
    fn cpu_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        let hk_per_hv = hv / hk;
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let g_val = g[n];
                let beta_val = beta[n];
                let qk_base = (batch * hk + hk_idx) * dk;
                for dv_idx in 0..dv {
                    let v_val = v[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state_in[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        kv_mem += s * k[qk_base + s_idx];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k[qk_base + s_idx] * delta;
                        state_out[s_base + s_idx] = s_new;
                        out += s_new * q[qk_base + s_idx];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    /// Small fused GDN-prep shape: dk a multiple of 32, Hv divisible by Hk.
    #[allow(clippy::too_many_arguments)]
    fn setup(
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        weight_scale: f32,
        dt: DType,
    ) -> TestSetup {
        let n_total = b * hv;
        let stride_b = 2 * hk * dk + hv * dv;
        // Bounded magnitudes keep softplus/exp in fp32 range (same fixture
        // shape as the legacy test's `make_fixture`).
        let conv_out: Vec<f32> =
            (0..b * stride_b).map(|i| ((i as f32) * 0.0131).sin() * 0.4).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - (i as f32) * 0.1).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * hv).map(|i| -0.3 + (i as f32) * 0.04).collect();
        let b_raw: Vec<f32> = (0..b * hv).map(|i| -0.2 + (i as f32) * 0.03).collect();
        // Non-identity per-head_dim weights (exercises the scaled path).
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * 0.1).collect();

        // Dtype-round every input so the oracle sees the GPU's load precision.
        let r = |xs: &[f32]| crate::utils::unpack_f32(&pack_f32(xs, dt), dt);
        let (q, k, v, g, beta) = cpu_prep(
            &r(&conv_out),
            &r(&a_log),
            &r(&dt_bias),
            &r(&a_raw),
            &r(&b_raw),
            &r(&q_norm_weight),
            &r(&k_norm_weight),
            b,
            hv,
            hk,
            dv,
            dk,
        );
        let (y_exp, state_exp) = cpu_step(&q, &k, &v, &g, &beta, &r(&state_in), b, hv, hk, dv, dk);

        TestSetup::new(mt_gated_delta_prep_step::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_norm_weight", pack_f32(&q_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("k_norm_weight", pack_f32(&k_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", n_total * dv, dt))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }

    // GQA (Hv = 2·Hk), weighted RMSNorm path. f16/bf16 widen the band: the
    // RMSNorm rsqrt, softplus/exp prep, and the dependent recurrence reduction
    // all compound the mantissa noise.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_gated_delta_prep_step_gqa(dt: DType) -> TestSetup { setup(2, 4, 2, 8, 64, 0.7, dt) }

    // Hv == Hk (no key-sharing) at the minimum dk=32, single batch.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_gated_delta_prep_step_no_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 4, 32, 1.0, dt)
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_gated_delta_prep_step;

    // Grid `[dv, b*hv, 1]`, TG `[32,1,1]`, Reduction — identical geometry to
    // `mt_gated_delta_step`. conv_out is the fused q|k|v slab of width
    // `2·Hk·Dk + Hv·Dv`.
    #[bench(name = "ffai/gated_delta_prep_step", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_step(dt: DType) -> BenchSetup {
        let (b, hv, hk, dv, dk) = (2usize, 4usize, 2usize, 64usize, 64usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(mt_gated_delta_prep_step::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", n_total, dt))
            .buffer(BenchBuffer::random("b_raw", n_total, dt))
            .buffer(BenchBuffer::random("q_norm_weight", hk * dk, dt))
            .buffer(BenchBuffer::random("k_norm_weight", hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", n_total * dv, dt).output())
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
            .bytes_moved((n_total * dv * dk * 2 * dt.size_bytes()) as u64)
    }
}
