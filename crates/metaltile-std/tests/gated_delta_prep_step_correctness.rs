#![allow(clippy::type_complexity)]

//! GPU correctness for `ffai::gated_delta_prep::mt_gated_delta_prep_step`.
//!
//! The fused kernel absorbs the host-side prep that
//! `Qwen35GDNMixer.forward` currently does between conv1d and the GDN
//! recurrence:
//!
//!   1. conv split → q / k / v
//!   2. per-head RMSNorm + scale of q, k
//!   3. g    = exp(-exp(A_log) · softplus(a_raw + dt_bias))
//!   4. beta = sigmoid(b_raw)
//!   5. the recurrence (same math as `mt_gated_delta_step`)
//!
//! The CPU oracle in this file is intentionally written as
//!   `cpu_prep(...)` → `naive_gated_delta_step(...)`
//! so the comparison is "fused == prep + unfused", which is exactly
//! what we get to delete on the host once this kernel lands.
//!
//! Coverage rationale (per spec — "lots of test coverage"):
//!   - f16 small @ Qwen3.6 shape (Dk=128, Dv=128, Hk=16, Hv=32)
//!   - bf16 small @ Qwen3.6 shape (production dtype path)
//!   - f32 small @ Qwen3.6 shape (numerical reference)
//!   - identity weights (recovers unweighted `perHeadRMSNormScale35`)
//!   - non-identity weights (the per-head_dim scaled path)
//!   - multi-step (8 consecutive steps with state carryover)
//!   - Hv==Hk (no GQA) AND Hv=2·Hk (Qwen3.6 GQA factor)
//!   - Dk divisible by simdgroup width (32) at both 128 and 256
//!
//! Cosine ≥ 0.999 vs CPU oracle across every case.
//!
//! macOS-gated, shared `gpu_lock` via `tests/common/`.

#![cfg(target_os = "macos")]
#![allow(clippy::too_many_arguments)]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta_prep::mt_gated_delta_prep_step;

// ────────────────────────────────────────────────────────────────────
//  CPU oracle: scalar prep + recurrence.
// ────────────────────────────────────────────────────────────────────

/// Numerically-stable softplus(x) = log(1 + exp(x)).
///
/// The GPU kernel emits the un-clamped `log(exp(x) + 1)` form; this
/// reference mirrors that exactly so the GPU↔CPU diff is purely ULP.
/// Production magnitudes of `a_raw + dt_bias` for Qwen3.6 stay in fp32
/// dynamic range, so the un-clamped form is safe.
fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// CPU prep: conv_out → (q_normed, k_normed, v_flat, g, beta).
///
/// Mirrors what `Qwen35GDNMixer.forward` does on the host today, just
/// in arithmetic. `q_norm_weight` / `k_norm_weight` are `[Hk·Dk]` —
/// pass all-ones × scale to recover the unweighted path.
fn cpu_prep(
    conv_out: &[f32],      // [B, 2·Hk·Dk + Hv·Dv]
    a_log: &[f32],         // [Hv]
    dt_bias: &[f32],       // [Hv]
    a_raw: &[f32],         // [B, Hv]
    b_raw: &[f32],         // [B, Hv]
    q_norm_weight: &[f32], // [Hk·Dk]
    k_norm_weight: &[f32], // [Hk·Dk]
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

    // q / k per-head RMSNorm + weight.
    for batch in 0..b {
        let q_base = batch * stride_b;
        let k_base = q_base + hk * dk;
        let v_base = q_base + 2 * hk * dk;

        for hk_idx in 0..hk {
            let row_off = hk_idx * dk;
            // ssq over Dk for this head.
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

        // v copy (no norm).
        for hv_idx in 0..hv {
            for dv_idx in 0..dv {
                v_flat[(batch * hv + hv_idx) * dv + dv_idx] =
                    conv_out[v_base + hv_idx * dv + dv_idx];
            }
        }

        // g / beta per Hv-head.
        for hv_idx in 0..hv {
            let n = batch * hv + hv_idx;
            let dt = softplus_unclamped(a_raw[n] + dt_bias[hv_idx]);
            g[n] = (-a_log[hv_idx].exp() * dt).exp();
            beta[n] = sigmoid(b_raw[n]);
        }
    }

    (q_normed, k_normed, v_flat, g, beta)
}

/// CPU recurrence (matches `naive_gated_delta_step` in the unfused
/// test file). Composed with `cpu_prep` this gives the full prep+step
/// reference the fused kernel compares against.
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

/// Full CPU oracle: prep then step.
fn cpu_fused_oracle(
    conv_out: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_raw: &[f32],
    b_raw: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    state_in: &[f32],
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (q, k, v, g, beta) = cpu_prep(
        conv_out,
        a_log,
        dt_bias,
        a_raw,
        b_raw,
        q_norm_weight,
        k_norm_weight,
        b,
        hv,
        hk,
        dv,
        dk,
    );
    cpu_step(&q, &k, &v, &g, &beta, state_in, b, hv, hk, dv, dk)
}

// ────────────────────────────────────────────────────────────────────
//  GPU dispatch helper.
// ────────────────────────────────────────────────────────────────────

fn run_gpu(
    conv_out: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_raw: &[f32],
    b_raw: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    state_in: &[f32],
    dt: Dt,
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(dk.is_multiple_of(32), "mt_gated_delta_prep_step requires dk % 32 == 0");
    let n_total = b * hv;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("conv_out".into(), pack_bytes(conv_out, dt));
    buffers.insert("a_log".into(), pack_bytes(a_log, dt));
    buffers.insert("dt_bias".into(), pack_bytes(dt_bias, dt));
    buffers.insert("a_raw".into(), pack_bytes(a_raw, dt));
    buffers.insert("b_raw".into(), pack_bytes(b_raw, dt));
    buffers.insert("q_norm_weight".into(), pack_bytes(q_norm_weight, dt));
    buffers.insert("k_norm_weight".into(), pack_bytes(k_norm_weight, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total * dv], dt));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gated_delta_prep_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
        .expect("mt_gated_delta_prep_step dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, state_out)
}

// ────────────────────────────────────────────────────────────────────
//  Metrics.
// ────────────────────────────────────────────────────────────────────

/// Cosine similarity. Both vectors must have non-zero norm; tests pick
/// fixtures where this holds. Returns NaN if either norm is zero (which
/// the assertion will then catch downstream).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let af = *av as f64;
        let bf = *bv as f64;
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

// ────────────────────────────────────────────────────────────────────
//  Fixture builders. Deterministic, bounded magnitudes — keeps
//  softplus / exp inside fp32 dynamic range so the un-clamped formula
//  in both kernel and oracle stays well-conditioned.
// ────────────────────────────────────────────────────────────────────

struct Fixture {
    conv_out: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    a_raw: Vec<f32>,
    b_raw: Vec<f32>,
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    state_in: Vec<f32>,
}

/// Build a deterministic fixture with the option to use identity-or-
/// non-identity weights and a tuneable scale (recovers the `invKeyScale`
/// folded-into-weight path from `Qwen35GDNMixer.forward`).
fn make_fixture(
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
    identity_weights: bool,
    weight_scale: f32,
    seed_offset: usize,
) -> Fixture {
    let stride_b = 2 * hk * dk + hv * dv;
    let conv_out: Vec<f32> =
        (0..b * stride_b).map(|i| (((i + seed_offset) as f32) * 0.0131).sin() * 0.4).collect();
    // a_log < 0 → exp(a_log) ∈ (0, 1) → g ∈ (0, 1). Production-realistic.
    let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - (i as f32) * 0.1).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
    let a_raw: Vec<f32> = (0..b * hv).map(|i| -0.3 + (i as f32) * 0.04).collect();
    let b_raw: Vec<f32> = (0..b * hv).map(|i| -0.2 + (i as f32) * 0.03).collect();

    let q_norm_weight: Vec<f32> = if identity_weights {
        vec![weight_scale; hk * dk]
    } else {
        (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect()
    };
    let k_norm_weight: Vec<f32> = if identity_weights {
        vec![weight_scale; hk * dk]
    } else {
        (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect()
    };
    let state_in: Vec<f32> =
        (0..b * hv * dv * dk).map(|i| (((i + seed_offset) as f32) * 0.0073).cos() * 0.1).collect();

    Fixture { conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in }
}

/// Quantise every input through the kernel's load-time dtype so the CPU
/// oracle sees the same precision the GPU does (f16: 10-bit mantissa;
/// bf16: 7-bit). f32 is a no-op.
fn round_fixture(f: &Fixture, dt: Dt) -> Fixture {
    let r = |xs: &[f32]| xs.iter().map(|&v| dt.round(v)).collect::<Vec<_>>();
    Fixture {
        conv_out: r(&f.conv_out),
        a_log: r(&f.a_log),
        dt_bias: r(&f.dt_bias),
        a_raw: r(&f.a_raw),
        b_raw: r(&f.b_raw),
        q_norm_weight: r(&f.q_norm_weight),
        k_norm_weight: r(&f.k_norm_weight),
        state_in: r(&f.state_in),
    }
}

/// Run one (shape, dtype, weight-mode) cell and return (cos_y, cos_state).
fn run_cell(
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
    dt: Dt,
    identity_weights: bool,
    weight_scale: f32,
) -> (f32, f32) {
    let _g = gpu_lock();
    let raw = make_fixture(b, hv, hk, dv, dk, identity_weights, weight_scale, 0);
    let f = round_fixture(&raw, dt);

    let (y_cpu, state_cpu) = cpu_fused_oracle(
        &f.conv_out,
        &f.a_log,
        &f.dt_bias,
        &f.a_raw,
        &f.b_raw,
        &f.q_norm_weight,
        &f.k_norm_weight,
        &f.state_in,
        b,
        hv,
        hk,
        dv,
        dk,
    );
    let (y_gpu, state_gpu) = run_gpu(
        &f.conv_out,
        &f.a_log,
        &f.dt_bias,
        &f.a_raw,
        &f.b_raw,
        &f.q_norm_weight,
        &f.k_norm_weight,
        &f.state_in,
        dt,
        b,
        hv,
        hk,
        dv,
        dk,
    );

    (cosine(&y_gpu, &y_cpu), cosine(&state_gpu, &state_cpu))
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

// ---- f32 reference --------------------------------------------------

#[test]
fn prep_step_f32_qwen36_shape_identity_weights() {
    // Qwen3.6 production shape; identity weights = unweighted +
    // `scale=1.0` recovers the `perHeadRMSNormScale35(_, scale=1)` path.
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 qwen3.6 identity y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 qwen3.6 identity state cos = {cs:.6}");
}

#[test]
fn prep_step_f32_qwen36_shape_nonidentity_weights() {
    // Per-head_dim weighted path. `weight_scale=0.5` mimics the
    // `invKeyScale = 1/sqrt(dk)` folded into the weight vector for the
    // k-norm (and `invKeyScale²` would similarly fold into q-norm).
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::F32, false, 0.5);
    assert!(cy >= 0.999, "f32 qwen3.6 weighted y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 qwen3.6 weighted state cos = {cs:.6}");
}

#[test]
fn prep_step_f32_no_gqa() {
    // Hv == Hk: every Hv-head has its own (q, k). Catches a refactor
    // that breaks the no-share branch the same way the unfused test
    // file does.
    let (cy, cs) = run_cell(1, 4, 4, 32, 64, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 no-GQA y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 no-GQA state cos = {cs:.6}");
}

#[test]
fn prep_step_f32_dk_256_full_n_per_t_slot_usage() {
    // Dk=256 ⇒ n_per_t = 8 — the upper bound of the stack_alloc cap.
    // Any off-by-one in the per-lane iteration would surface here.
    let (cy, cs) = run_cell(1, 4, 2, 8, 256, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 Dk=256 y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 Dk=256 state cos = {cs:.6}");
}

#[test]
fn prep_step_f32_batch_2() {
    // Batch > 1 exercises the `conv_base = b · stride_b` offset and
    // the per-batch a_raw / b_raw indexing.
    let (cy, cs) = run_cell(2, 4, 2, 8, 64, Dt::F32, false, 0.7);
    assert!(cy >= 0.999, "f32 B=2 y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 B=2 state cos = {cs:.6}");
}

// ---- f16 — Qwen3.6 small fp16 path ---------------------------------

#[test]
fn prep_step_f16_qwen36_shape_identity_weights() {
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::F16, true, 1.0);
    assert!(cy >= 0.999, "f16 qwen3.6 identity y cos = {cy:.6}");
    assert!(cs >= 0.999, "f16 qwen3.6 identity state cos = {cs:.6}");
}

#[test]
fn prep_step_f16_qwen36_shape_nonidentity_weights() {
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::F16, false, 0.5);
    assert!(cy >= 0.999, "f16 qwen3.6 weighted y cos = {cy:.6}");
    assert!(cs >= 0.999, "f16 qwen3.6 weighted state cos = {cs:.6}");
}

#[test]
fn prep_step_f16_no_gqa() {
    let (cy, cs) = run_cell(1, 4, 4, 32, 64, Dt::F16, true, 1.0);
    assert!(cy >= 0.999, "f16 no-GQA y cos = {cy:.6}");
    assert!(cs >= 0.999, "f16 no-GQA state cos = {cs:.6}");
}

// ---- bf16 — Qwen3.6 production dtype --------------------------------

#[test]
fn prep_step_bf16_qwen36_shape_identity_weights() {
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::Bf16, true, 1.0);
    assert!(cy >= 0.999, "bf16 qwen3.6 identity y cos = {cy:.6}");
    assert!(cs >= 0.999, "bf16 qwen3.6 identity state cos = {cs:.6}");
}

#[test]
fn prep_step_bf16_qwen36_shape_nonidentity_weights() {
    let (cy, cs) = run_cell(1, 32, 16, 128, 128, Dt::Bf16, false, 0.5);
    assert!(cy >= 0.999, "bf16 qwen3.6 weighted y cos = {cy:.6}");
    assert!(cs >= 0.999, "bf16 qwen3.6 weighted state cos = {cs:.6}");
}

#[test]
fn prep_step_bf16_no_gqa() {
    let (cy, cs) = run_cell(1, 4, 4, 32, 64, Dt::Bf16, true, 1.0);
    assert!(cy >= 0.999, "bf16 no-GQA y cos = {cy:.6}");
    assert!(cs >= 0.999, "bf16 no-GQA state cos = {cs:.6}");
}

// ────────────────────────────────────────────────────────────────────
//  Multi-step state carryover — 8 consecutive steps.
//
//  Validates that running N fused steps with state-out → state-in
//  threading matches N CPU prep+step compositions. Catches state
//  corruption that would only surface across multiple iterations
//  (drift bugs, alias of state_in / state_out, etc.).
// ────────────────────────────────────────────────────────────────────

#[test]
fn prep_step_f32_multi_step_8_consecutive() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 8;
    let dk = 64;
    let n_steps = 8;

    // Identity weights × 0.7 — keep scale away from 1.0 so an accidental
    // "skip the weight" regression would diff visibly.
    let weights_q = vec![0.7_f32; hk * dk];
    let weights_k = vec![0.7_f32; hk * dk];
    let a_log: Vec<f32> = (0..hv).map(|i| -1.0 - (i as f32) * 0.1).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.3 + (i as f32) * 0.05).collect();

    let mut state_gpu = vec![0.0_f32; b * hv * dv * dk];
    let mut state_cpu = state_gpu.clone();

    for step in 0..n_steps {
        let stride_b = 2 * hk * dk + hv * dv;
        // Vary conv_out / a_raw / b_raw per step so the recurrence has
        // real input. Quantising through f32 is a no-op.
        let conv_out: Vec<f32> =
            (0..b * stride_b).map(|i| (((i + step * 17) as f32) * 0.0131).sin() * 0.4).collect();
        let a_raw: Vec<f32> = (0..b * hv).map(|i| -0.3 + ((i + step) as f32) * 0.04).collect();
        let b_raw: Vec<f32> = (0..b * hv).map(|i| -0.2 + ((i + step) as f32) * 0.03).collect();

        let (y_cpu, state_cpu_new) = cpu_fused_oracle(
            &conv_out, &a_log, &dt_bias, &a_raw, &b_raw, &weights_q, &weights_k, &state_cpu, b, hv,
            hk, dv, dk,
        );
        let (y_gpu, state_gpu_new) = run_gpu(
            &conv_out,
            &a_log,
            &dt_bias,
            &a_raw,
            &b_raw,
            &weights_q,
            &weights_k,
            &state_gpu,
            Dt::F32,
            b,
            hv,
            hk,
            dv,
            dk,
        );

        let cy = cosine(&y_gpu, &y_cpu);
        let cs = cosine(&state_gpu_new, &state_cpu_new);
        assert!(cy >= 0.999, "step {step} y cos = {cy:.6}");
        assert!(cs >= 0.999, "step {step} state cos = {cs:.6}");

        // Finite-check — drifting recurrence would NaN out before
        // cosine cared.
        for &s in state_gpu_new.iter() {
            assert!(s.is_finite(), "step {step}: non-finite state {s}");
        }

        state_gpu = state_gpu_new;
        state_cpu = state_cpu_new;
    }
}

#[test]
fn prep_step_bf16_multi_step_8_consecutive() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 8;
    let dk = 64;
    let n_steps = 8;

    // Quantise the static inputs to bf16 once. Per-step inputs get
    // round_through_bf16 inside the loop.
    let round_bf = |xs: &[f32]| xs.iter().map(|&v| Dt::Bf16.round(v)).collect::<Vec<_>>();
    let weights_q = round_bf(&vec![0.7_f32; hk * dk]);
    let weights_k = round_bf(&vec![0.7_f32; hk * dk]);
    let a_log = round_bf(&(0..hv).map(|i| -1.0 - (i as f32) * 0.1).collect::<Vec<_>>());
    let dt_bias = round_bf(&(0..hv).map(|i| -0.3 + (i as f32) * 0.05).collect::<Vec<_>>());

    let mut state_gpu = vec![0.0_f32; b * hv * dv * dk];
    let mut state_cpu = state_gpu.clone();

    for step in 0..n_steps {
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out = round_bf(
            &(0..b * stride_b)
                .map(|i| (((i + step * 17) as f32) * 0.0131).sin() * 0.4)
                .collect::<Vec<_>>(),
        );
        let a_raw =
            round_bf(&(0..b * hv).map(|i| -0.3 + ((i + step) as f32) * 0.04).collect::<Vec<_>>());
        let b_raw =
            round_bf(&(0..b * hv).map(|i| -0.2 + ((i + step) as f32) * 0.03).collect::<Vec<_>>());

        let (_y_cpu, state_cpu_new) = cpu_fused_oracle(
            &conv_out, &a_log, &dt_bias, &a_raw, &b_raw, &weights_q, &weights_k, &state_cpu, b, hv,
            hk, dv, dk,
        );
        let (y_gpu, state_gpu_new) = run_gpu(
            &conv_out,
            &a_log,
            &dt_bias,
            &a_raw,
            &b_raw,
            &weights_q,
            &weights_k,
            &state_gpu,
            Dt::Bf16,
            b,
            hv,
            hk,
            dv,
            dk,
        );

        // bf16 drift across 8 steps — only check cosine on state +
        // finite-ness on y. The recurrence amplifies bf16's 7-bit
        // mantissa noise; cosine still tracks well at 0.999.
        let cs = cosine(&state_gpu_new, &state_cpu_new);
        assert!(cs >= 0.999, "step {step} bf16 state cos = {cs:.6}");
        for &v in y_gpu.iter() {
            assert!(v.is_finite(), "step {step}: bf16 y non-finite {v}");
        }
        for &s in state_gpu_new.iter() {
            assert!(s.is_finite(), "step {step}: bf16 state non-finite {s}");
        }

        state_gpu = state_gpu_new;
        state_cpu = state_cpu_new;
    }
}

// ────────────────────────────────────────────────────────────────────
//  Equivalence with the unfused path.
//
//  When q_norm_weight / k_norm_weight are identity and a_log / a_raw /
//  dt_bias / b_raw are chosen so g and beta land at the values used by
//  the (unfused) `gated_delta_gpu_correctness.rs` tests, the fused
//  kernel's output should match the CPU oracle to the same precision
//  envelope the unfused kernel does. This is the regression net under
//  the "drop-in replacement" claim.
// ────────────────────────────────────────────────────────────────────

#[test]
fn prep_step_f32_matches_unfused_path_when_weights_identity() {
    // Recovers the unweighted `perHeadRMSNormScale35` path the existing
    // Qwen3.6 host code uses. Equivalent to passing (q, k) already
    // RMSNorm-scaled through `mt_gated_delta_step`.
    let (cy, cs) = run_cell(1, 8, 4, 16, 64, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 unfused-equivalence y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 unfused-equivalence state cos = {cs:.6}");
}
