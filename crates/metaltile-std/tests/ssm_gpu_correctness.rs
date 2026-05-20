//! End-to-end GPU correctness for the three Mamba 2 / SSD building blocks
//! in `ffai::ssm`:
//!
//!   - `conv1d_causal_step`  — depthwise causal-conv streaming-decode step
//!   - `ssm_step`            — selective-scan single-token decode (one thread per (h, d))
//!   - `mt_ssm_step`         — MLX-aligned ssm_step port (Reduction mode, simd_sum)
//!
//! Before this file the only validation was inside FFAI integration tests
//! against real Mamba/Nemotron decoding — a wrong index formula or a
//! silent kernel-emptiness regression (the proc-macro class that bit
//! sibling `ffai/` kernels in PR #19) would only surface as decode
//! garbage, not as a unit-test failure. Three CPU oracles + dtype
//! coverage close that gap.
//!
//! Why these matter for serving:
//!   - `conv1d_causal_step` is on Mamba 2's hot decode path (every token,
//!     every layer).
//!   - `ssm_step` is the decode-time selective-scan recurrence that the
//!     hybrid Qwen3.6-35B-A3B port + NemotronH need for non-attention
//!     layers (state-space + GDN blocks).
//!   - `mt_ssm_step` is the faithful MLX-aligned variant — separate
//!     dispatch geometry (Reduction, 32-thread simdgroups, simd_sum
//!     across state dim) so it needs a separate correctness pin.
//!
//! All h/state accumulators run in f32 inside the kernels — the
//! exp(A*dt)*h + dt*B*x recurrence in bf16 drifts in a few dozen
//! steps. Activation tensors stay in T (typically bf16 in real runs).
//! Tolerance bands here reflect that split.
//!
//! macOS-gated. Shared gpu_lock via tests/common/.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::ssm::{conv1d_causal_step, mt_ssm_step, ssm_step};

// ────────────────────────────────────────────────────────────────────
//  conv1d_causal_step
// ────────────────────────────────────────────────────────────────────

/// CPU oracle. `state` is updated in place to match the kernel's
/// post-dispatch state.
fn naive_conv1d_causal_step(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    state: &mut [f32],
    n_channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_channels];
    let k_last = kernel_size - 1;

    for d in 0..n_channels {
        let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
        for k in 0..k_last {
            acc += w[k * n_channels + d] * state[k * n_channels + d];
        }
        y[d] = acc;
    }
    // Shift state up: state[k] = state[k+1], state[K-2] = x.
    for d in 0..n_channels {
        for k in 0..kernel_size.saturating_sub(2) {
            state[k * n_channels + d] = state[(k + 1) * n_channels + d];
        }
        if kernel_size >= 2 {
            state[(kernel_size - 2) * n_channels + d] = x[d];
        }
    }
    y
}

fn run_conv1d_causal_step(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    state: &[f32],
    dt: Dt,
    n_channels: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("state".into(), pack_bytes(state, dt));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_channels], dt));
    buffers.insert("n_channels".into(), (n_channels as u32).to_le_bytes().to_vec());
    buffers.insert("kernel_size".into(), (kernel_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv1d_causal_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id::<0>() = thread index, one thread per channel.
    // n_channels (≤256 in these tests) fits in one TG.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [n_channels, 1, 1])
        .expect("conv1d_causal_step dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state").expect("state"), dt);
    (y, state_out)
}

#[test]
fn conv1d_causal_step_matches_oracle_f32() {
    let _g = gpu_lock();
    // Mamba 2 short-conv: kernel_size=4, conv_dim modest for unit test.
    let n_channels = 128usize;
    let kernel_size = 4usize;
    let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
    let w: Vec<f32> =
        (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
    let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
    let mut state_oracle: Vec<f32> =
        (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();
    let state_initial = state_oracle.clone();

    let y_expected =
        naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
    let (y_actual, state_actual) =
        run_conv1d_causal_step(&x, &w, &b, &state_initial, Dt::F32, n_channels, kernel_size);

    let mut max_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 1e-5, "conv1d y max |diff| = {max_diff:.2e}");

    // State must shift exactly: state[k] = state[k+1], state[K-2] = x.
    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_oracle.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-6, "conv1d state shift max |diff| = {max_state_diff:.2e}");
}

#[test]
fn conv1d_causal_step_matches_oracle_f16() {
    let _g = gpu_lock();
    let n_channels = 64usize;
    let kernel_size = 4usize;
    let x_f32: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
    let w_f32: Vec<f32> =
        (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
    let b_f32: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
    let state_f32: Vec<f32> =
        (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();

    // Round inputs through f16 so CPU oracle uses the same load precision.
    let round = |v: &[f32]| v.iter().map(|&x| Dt::F16.round(x)).collect::<Vec<f32>>();
    let x = round(&x_f32);
    let w = round(&w_f32);
    let b = round(&b_f32);
    let mut state_oracle = round(&state_f32);

    let y_expected =
        naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
    let (y_actual, _state_actual) =
        run_conv1d_causal_step(&x, &w, &b, &round(&state_f32), Dt::F16, n_channels, kernel_size);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // f16 dot-product over kernel_size=4 + bias accumulates ~4 ULPs.
    assert!(max_rel < 5e-3, "conv1d f16 max rel = {max_rel:.2e}");
}

#[test]
fn conv1d_causal_step_matches_oracle_bf16() {
    let _g = gpu_lock();
    let n_channels = 64usize;
    let kernel_size = 4usize;
    let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
    let x_f32: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
    let w_f32: Vec<f32> =
        (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
    let b_f32: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
    let state_f32: Vec<f32> =
        (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();
    let x = round(&x_f32);
    let w = round(&w_f32);
    let b = round(&b_f32);
    let mut state_oracle = round(&state_f32);

    let y_expected =
        naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
    let (y_actual, _state_actual) =
        run_conv1d_causal_step(&x, &w, &b, &round(&state_f32), Dt::Bf16, n_channels, kernel_size);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // bf16 7-bit mantissa — wider tolerance than f16.
    assert!(max_rel < 5e-2, "conv1d bf16 max rel = {max_rel:.2e}");
}

// ────────────────────────────────────────────────────────────────────
//  ssm_step (basic, one thread per (h, d))
// ────────────────────────────────────────────────────────────────────

/// CPU oracle. `h_state` updates in place to match the kernel.
#[allow(clippy::too_many_arguments)]
fn naive_ssm_step(
    x: &[f32],
    a: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    dt: &[f32],
    h_state: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    state_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_heads * head_dim];
    for h in 0..n_heads {
        let decay = (a[h] * dt[h]).exp();
        let h_base = h * state_dim * head_dim;
        for d in 0..head_dim {
            let x_d = x[h * head_dim + d];
            let mut y_d = 0.0_f32;
            for n in 0..state_dim {
                let h_idx = h_base + n * head_dim + d;
                let h_old = h_state[h_idx];
                let new_h = decay * h_old + dt[h] * b_vec[n] * x_d;
                h_state[h_idx] = new_h;
                y_d += c_vec[n] * new_h;
            }
            y[h * head_dim + d] = y_d;
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn run_ssm_step(
    x: &[f32],
    a: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    dt_in: &[f32],
    h_state: &[f32],
    dt: Dt,
    n_heads: usize,
    head_dim: usize,
    state_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b_vec, dt));
    buffers.insert("c".into(), pack_bytes(c_vec, dt));
    buffers.insert("dt".into(), pack_bytes(dt_in, dt));
    // `h` is always f32 in the kernel signature.
    buffers.insert("h".into(), pack_bytes(h_state, Dt::F32));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_heads * head_dim], dt));
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("state_dim".into(), (state_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ssm_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id::<0>() = thread.x; one thread per (h, d).
    // Total = n_heads*head_dim ≤ 1024 in tests, single TG.
    let total = n_heads * head_dim;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [total, 1, 1])
        .expect("ssm_step dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let h_out = unpack_bytes(result.outputs.get("h").expect("h"), Dt::F32);
    (y, h_out)
}

#[test]
fn ssm_step_matches_oracle_f32() {
    let _g = gpu_lock();
    let n_heads = 4usize;
    let head_dim = 16usize;
    let state_dim = 8usize;

    let x: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
    let a: Vec<f32> = (0..n_heads).map(|i| -0.5 - (i as f32) * 0.1).collect();
    let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
    let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
    let mut h_state_oracle: Vec<f32> =
        (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();
    let h_state_initial = h_state_oracle.clone();

    let y_expected = naive_ssm_step(
        &x,
        &a,
        &b_vec,
        &c_vec,
        &dt_in,
        &mut h_state_oracle,
        n_heads,
        head_dim,
        state_dim,
    );
    let (y_actual, h_actual) = run_ssm_step(
        &x,
        &a,
        &b_vec,
        &c_vec,
        &dt_in,
        &h_state_initial,
        Dt::F32,
        n_heads,
        head_dim,
        state_dim,
    );

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 1e-5, "ssm_step y max |diff| = {max_y_diff:.2e}");

    let mut max_h_diff = 0.0_f32;
    for (a, e) in h_actual.iter().zip(h_state_oracle.iter()) {
        max_h_diff = max_h_diff.max((a - e).abs());
    }
    assert!(max_h_diff < 1e-5, "ssm_step h update max |diff| = {max_h_diff:.2e}");
}

#[test]
fn ssm_step_state_decays_when_x_is_zero_f32() {
    let _g = gpu_lock();
    // Invariant: with x=0, the recurrence reduces to h_new = decay*h_old.
    // y = sum(c[n] * decay * h_old[n]). Catches a regression where the
    // db*x term gets wired up wrong (e.g. missing the dt scale).
    let n_heads = 2usize;
    let head_dim = 8usize;
    let state_dim = 4usize;

    let x = vec![0.0_f32; n_heads * head_dim];
    let a: Vec<f32> = vec![-1.0, -2.0];
    let b_vec: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8];
    let c_vec: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let dt_in: Vec<f32> = vec![0.05, 0.05];
    let h_state_initial: Vec<f32> =
        (0..n_heads * state_dim * head_dim).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let mut h_state_oracle = h_state_initial.clone();
    let y_expected = naive_ssm_step(
        &x,
        &a,
        &b_vec,
        &c_vec,
        &dt_in,
        &mut h_state_oracle,
        n_heads,
        head_dim,
        state_dim,
    );
    let (y_actual, _h_actual) = run_ssm_step(
        &x,
        &a,
        &b_vec,
        &c_vec,
        &dt_in,
        &h_state_initial,
        Dt::F32,
        n_heads,
        head_dim,
        state_dim,
    );
    let mut max_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 1e-5, "x=0 decay invariant max |diff| = {max_diff:.2e}");
}

// ────────────────────────────────────────────────────────────────────
//  mt_ssm_step (MLX-aligned, Reduction mode, simd_sum across state)
// ────────────────────────────────────────────────────────────────────

/// CPU oracle for `mt_ssm_step`. Mirrors the kernel exactly — including
/// the `-exp(a_log)` form (always-negative decay base) and the
/// `total + x*d_skip` skip term.
#[allow(clippy::too_many_arguments)]
fn naive_mt_ssm_step(
    x: &[f32],
    a_log: &[f32],
    b_mat: &[f32],
    c_mat: &[f32],
    d_skip: &[f32],
    dt_in: &[f32],
    state_in: &[f32],
    n_total: usize, // n_heads * batch
    dh: usize,
    ds: usize,
    n_heads: usize,
    heads_per_group: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state_out = vec![0.0_f32; state_in.len()];
    let mut out = vec![0.0_f32; n_total * dh];

    for n in 0..n_total {
        let h_idx = n % n_heads;
        let g_idx = n / heads_per_group;
        let dt_val = dt_in[n];
        let a_val = -(a_log[h_idx].exp());
        let da = (a_val * dt_val).exp();

        for d_idx in 0..dh {
            let x_val = x[n * dh + d_idx];
            let mut acc = 0.0_f32;
            for s_idx in 0..ds {
                let idx = n * dh * ds + d_idx * ds + s_idx;
                let bc_idx = g_idx * ds + s_idx;
                let db_by_x = x_val * dt_val * b_mat[bc_idx];
                let new_state = da * state_in[idx] + db_by_x;
                state_out[idx] = new_state;
                acc += new_state * c_mat[bc_idx];
            }
            out[n * dh + d_idx] = acc + x_val * d_skip[h_idx];
        }
    }

    (state_out, out)
}

#[allow(clippy::too_many_arguments)]
fn run_mt_ssm_step(
    x: &[f32],
    a_log: &[f32],
    b_mat: &[f32],
    c_mat: &[f32],
    d_skip: &[f32],
    dt_in: &[f32],
    state_in: &[f32],
    dt: Dt,
    n_total: usize,
    dh: usize,
    ds: usize,
    n_heads: usize,
    heads_per_group: usize,
) -> (Vec<f32>, Vec<f32>) {
    let groups = n_total / heads_per_group;
    let bc_len = groups * ds;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("a_log".into(), pack_bytes(a_log, dt));
    // b_mat/c_mat are shaped [groups, ds] = bc_len long.
    assert_eq!(b_mat.len(), bc_len);
    assert_eq!(c_mat.len(), bc_len);
    buffers.insert("b_mat".into(), pack_bytes(b_mat, dt));
    buffers.insert("c_mat".into(), pack_bytes(c_mat, dt));
    buffers.insert("d_skip".into(), pack_bytes(d_skip, dt));
    buffers.insert("dt".into(), pack_bytes(dt_in, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n_total * dh], dt));
    buffers.insert("dh".into(), (dh as u32).to_le_bytes().to_vec());
    buffers.insert("ds".into(), (ds as u32).to_le_bytes().to_vec());
    buffers.insert("n_heads".into(), (n_heads as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_ssm_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction mode dispatch contract (per docs/developing.md):
    //   - program_id::<0>() = tgid.x = d_idx (0..dh)
    //   - program_id::<1>() = tgid.y = n     (0..n_total)
    //   - tid (= lane within TG) = ds_idx, 0..32
    //   - TPG must be ≥ 32 AND a multiple of 32.
    //   - ds % 32 == 0 (kernel reduces ds/32 elements per lane via simd_sum).
    assert!(ds.is_multiple_of(32), "mt_ssm_step requires ds % 32 == 0");
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dh, n_total, 1], [32, 1, 1])
        .expect("mt_ssm_step dispatch");

    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    let out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    (state_out, out)
}

#[test]
fn mt_ssm_step_matches_oracle_f32() {
    let _g = gpu_lock();
    let n_heads = 4usize;
    let heads_per_group = 2usize;
    let batch = 2usize;
    let n_total = n_heads * batch;
    let dh = 8usize;
    let ds = 32usize; // must be multiple of 32 per kernel contract
    let groups = n_total / heads_per_group;

    let x: Vec<f32> = (0..n_total * dh).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect();
    let a_log: Vec<f32> = (0..n_heads).map(|i| -1.0 + (i as f32) * 0.2).collect();
    let b_mat: Vec<f32> = (0..groups * ds).map(|i| 0.05 + (i as f32) * 0.003).collect();
    let c_mat: Vec<f32> = (0..groups * ds).map(|i| 0.1 - (i as f32) * 0.001).collect();
    let d_skip: Vec<f32> = (0..n_heads).map(|i| 0.05 + (i as f32) * 0.01).collect();
    let dt_in: Vec<f32> = (0..n_total).map(|i| 0.02 + (i as f32) * 0.005).collect();
    let state_in: Vec<f32> =
        (0..n_total * dh * ds).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect();

    let (state_expected, out_expected) = naive_mt_ssm_step(
        &x,
        &a_log,
        &b_mat,
        &c_mat,
        &d_skip,
        &dt_in,
        &state_in,
        n_total,
        dh,
        ds,
        n_heads,
        heads_per_group,
    );
    let (state_actual, out_actual) = run_mt_ssm_step(
        &x,
        &a_log,
        &b_mat,
        &c_mat,
        &d_skip,
        &dt_in,
        &state_in,
        Dt::F32,
        n_total,
        dh,
        ds,
        n_heads,
        heads_per_group,
    );

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "mt_ssm_step state max |diff| = {max_state_diff:.2e}");

    let mut max_out_diff = 0.0_f32;
    for (a, e) in out_actual.iter().zip(out_expected.iter()) {
        // simd_sum across 32 lanes; small FP ordering drift acceptable.
        max_out_diff = max_out_diff.max((a - e).abs());
    }
    assert!(max_out_diff < 5e-5, "mt_ssm_step out max |diff| = {max_out_diff:.2e}");
}

#[test]
fn mt_ssm_step_matches_oracle_bf16() {
    let _g = gpu_lock();
    let n_heads = 4usize;
    let heads_per_group = 2usize;
    let batch = 1usize;
    let n_total = n_heads * batch;
    let dh = 4usize;
    let ds = 32usize;
    let groups = n_total / heads_per_group;

    let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
    let x = round(&(0..n_total * dh).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect::<Vec<_>>());
    let a_log = round(&(0..n_heads).map(|i| -1.0 + (i as f32) * 0.2).collect::<Vec<_>>());
    let b_mat = round(&(0..groups * ds).map(|i| 0.05 + (i as f32) * 0.003).collect::<Vec<_>>());
    let c_mat = round(&(0..groups * ds).map(|i| 0.1 - (i as f32) * 0.001).collect::<Vec<_>>());
    let d_skip = round(&(0..n_heads).map(|i| 0.05 + (i as f32) * 0.01).collect::<Vec<_>>());
    let dt_in = round(&(0..n_total).map(|i| 0.02 + (i as f32) * 0.005).collect::<Vec<_>>());
    let state_in = round(
        &(0..n_total * dh * ds).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect::<Vec<_>>(),
    );

    let (_state_expected, out_expected) = naive_mt_ssm_step(
        &x,
        &a_log,
        &b_mat,
        &c_mat,
        &d_skip,
        &dt_in,
        &state_in,
        n_total,
        dh,
        ds,
        n_heads,
        heads_per_group,
    );
    let (_state_actual, out_actual) = run_mt_ssm_step(
        &x,
        &a_log,
        &b_mat,
        &c_mat,
        &d_skip,
        &dt_in,
        &state_in,
        Dt::Bf16,
        n_total,
        dh,
        ds,
        n_heads,
        heads_per_group,
    );

    let mut max_rel = 0.0_f32;
    for (a, e) in out_actual.iter().zip(out_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // bf16 mantissa 7 bits + simd_sum over 32 lanes.
    assert!(max_rel < 1e-1, "mt_ssm_step bf16 max rel = {max_rel:.2e}");
}
