//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end GPU correctness for
//! `ffai::conv1d_causal_step_silu_cast_many`.
//!
//! `conv1d_causal_step_silu_cast_many` collapses the per-token T-loop of
//! `conv1d_causal_step` + the downstream `mt_silu_cast_to_f32` (or the
//! `silu` + `cast_to_f32` pair when the conv output is already f32) into
//! ONE dispatch with the K-1 conv state held in thread-local registers
//! across the T-sweep.
//!
//! The cleanest oracle is the per-row sequence run on the CPU: roll the
//! K-1=3 state slots one step at a time, compute the depthwise conv +
//! bias, apply SiLU, cast to f32. The CPU reference uses the exact same
//! arithmetic order the GPU kernel does (`b + w3*x + w0*s0 + w1*s1 +
//! w2*s2`, then `acc * (1 / (1 + exp(-acc)))`), and quantises every
//! load through the dtype so f16 / bf16 inputs see the same rounding
//! the GPU sees on load. This pins both:
//!   - that the batched kernel's per-(r, d) outputs match the per-row
//!     conv_step + silu_cast pair for every (T, conv_dim) shape, and
//!   - that the final `state_out` matches the K-1=3 most-recent input
//!     rows from `src`.
//!
//! Dtype coverage: f32 / f16 / bf16. Tolerances tuned by dtype to cover
//! the dtype-specific load/store quantisation between the kernel and
//! the CPU reference; the math itself runs in f32 on both sides.
//!
//! `conv_kernel` is fixed at 4 — the kernel itself hardcodes K=4 in its
//! body (see the docstring on `ffai_conv1d_causal_step_silu_cast_many`
//! for why), so the test mirrors that.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::conv1d_causal_step_silu_cast_many::ffai_conv1d_causal_step_silu_cast_many;

/// CPU reference for the full T-sweep. Mirrors the GPU kernel body
/// op-for-op:
///   - 3 state scalars `s0, s1, s2` rolling forward each step
///   - acc = b + w3*x + w0*s0 + w1*s1 + w2*s2 (same FMA order as kernel)
///   - silu = acc / (1 + exp(-acc)), in f32
///   - cast back to f32 (no-op for the test's final dtype)
///   - state shift: s0=s1; s1=s2; s2=x
///
/// `dt.round` quantises every load to mimic the GPU's `cast::<f32>()`
/// on dtype-T memory (no-op for f32, 10-bit mantissa for f16, 7-bit for
/// bf16) — without this the f16/bf16 oracle would diverge from the
/// GPU by exactly the load quantisation step and look like a kernel
/// bug.
fn cpu_reference(
    src: &[f32],
    w: &[f32],
    b: &[f32],
    state_in: &[f32],
    dt: Dt,
    t_len: usize,
    conv_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0_f32; t_len * conv_dim];
    let mut state_out = vec![0.0_f32; 3 * conv_dim];
    for d in 0..conv_dim {
        let b_d = dt.round(b[d]);
        let w0 = dt.round(w[0 * conv_dim + d]);
        let w1 = dt.round(w[1 * conv_dim + d]);
        let w2 = dt.round(w[2 * conv_dim + d]);
        let w3 = dt.round(w[3 * conv_dim + d]);
        let mut s0 = dt.round(state_in[0 * conv_dim + d]);
        let mut s1 = dt.round(state_in[1 * conv_dim + d]);
        let mut s2 = dt.round(state_in[2 * conv_dim + d]);
        for r in 0..t_len {
            let x_r = dt.round(src[r * conv_dim + d]);
            // Same FMA order as the kernel body so f32 reassociation is
            // a no-op between the two.
            let acc = b_d + w3 * x_r + w0 * s0 + w1 * s1 + w2 * s2;
            let sig = 1.0_f32 / (1.0_f32 + (-acc).exp());
            let y = acc * sig;
            // out_f32 is f32-typed regardless of T; no round on store.
            out[r * conv_dim + d] = y;
            s0 = s1;
            s1 = s2;
            s2 = x_r;
        }
        // state_out stores T-typed values — the GPU rounds the f32
        // scalar back to T on store. Round here so the comparison
        // tolerance is tight against the dtype quantisation.
        state_out[0 * conv_dim + d] = dt.round(s0);
        state_out[1 * conv_dim + d] = dt.round(s1);
        state_out[2 * conv_dim + d] = dt.round(s2);
    }
    (out, state_out)
}

/// Dispatch the batched kernel and return `(out_f32, state_out_f32)`,
/// state_out unpacked through the dtype round-trip.
#[allow(clippy::too_many_arguments)]
fn run_many(
    src: &[f32],
    w: &[f32],
    b: &[f32],
    state_in: &[f32],
    dt: Dt,
    t_len: u32,
    conv_dim: u32,
    conv_kernel: u32,
) -> (Vec<f32>, Vec<f32>) {
    let out_elems = (t_len * conv_dim) as usize;
    let state_elems = ((conv_kernel - 1) * conv_dim) as usize;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(src, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    // `out_f32` is f32 regardless of T — see kernel signature.
    buffers.insert("out_f32".into(), vec![0u8; out_elems * 4]);
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_elems], dt));
    buffers.insert("t_len".into(), t_len.to_le_bytes().to_vec());
    buffers.insert("conv_dim".into(), conv_dim.to_le_bytes().to_vec());
    buffers.insert("conv_kernel".into(), conv_kernel.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_conv1d_causal_step_silu_cast_many::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per channel. Total = conv_dim. Split into
    // groups × tpg with tpg = min(conv_dim, 256). The kernel has no
    // early-return primitive, so we require `groups * tpg == conv_dim`
    // exactly — every conv_dim in the test matrix is a multiple of
    // 64, which we pick as the tpg so groups stays integral for all
    // cases. (64 fits comfortably under the 1024-thread per-TG cap
    // and gives 1/4/32 groups for conv_dim ∈ {64, 256, 2048}.)
    let total_threads = conv_dim as usize;
    let tpg = if total_threads <= 64 { total_threads } else { 64 };
    let groups = total_threads / tpg;
    assert_eq!(
        groups * tpg,
        total_threads,
        "tpg must divide conv_dim exactly (no early-return in DSL)"
    );
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("conv1d_causal_step_silu_cast_many dispatch");

    let out_bytes = result.outputs.get("out_f32").expect("out_f32 buffer");
    let state_bytes = result.outputs.get("state_out").expect("state_out buffer");
    // out_f32 is f32-typed; unpack as f32 unconditionally (not via Dt).
    let out = bytemuck::cast_slice::<u8, f32>(out_bytes).to_vec();
    let state_out = unpack_bytes(state_bytes, dt);
    (out, state_out)
}

/// Deterministic pseudo-random initialiser — same `wrapping_mul +
/// sin/cos` recipe used in `rope_llama_many` / `kv_cache_update_many`
/// tests. Keep values in a small range so SiLU's `exp(-acc)` doesn't
/// underflow into the noise floor of bf16.
fn make_data(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed) as f32;
            ((x * 0.0001).sin() + (x * 0.013).cos()) * 0.5 * scale
        })
        .collect()
}

/// Test cases: (T, conv_dim). conv_kernel = 4 throughout (the kernel
/// hardcodes K=4 in its body).
const CASES: &[(u32, u32)] = &[
    // Smallest — exercises the very-short T-sweep path. conv_dim=64
    // fits in one TG.
    (2, 64),
    // Medium — multi-TG dispatch (conv_dim=256 → 4 TGs of 64).
    (8, 256),
    // Production-shape — Qwen3.6-A3B GDN: conv_dim=2048 → 32 TGs.
    // T=32 keeps the test runtime small (T=512 would be the real
    // prefill case).
    (32, 2048),
];

const CONV_KERNEL: u32 = 4;

#[allow(clippy::too_many_arguments)]
fn check_one_case(
    dt: Dt,
    t_len: u32,
    conv_dim: u32,
    abs_tol: f32,
    rel_tol: f32,
) {
    let src = make_data((t_len * conv_dim) as usize, 0x1234, 1.0);
    let w = make_data((CONV_KERNEL * conv_dim) as usize, 0x5678, 0.3);
    let b = make_data(conv_dim as usize, 0x9ABC, 0.1);
    let state_in = make_data(((CONV_KERNEL - 1) * conv_dim) as usize, 0xDEF0, 1.0);

    let (gpu_out, gpu_state) =
        run_many(&src, &w, &b, &state_in, dt, t_len, conv_dim, CONV_KERNEL);
    let (cpu_out, cpu_state) = cpu_reference(&src, &w, &b, &state_in, dt, t_len as usize, conv_dim as usize);

    assert_eq!(gpu_out.len(), cpu_out.len(), "out length mismatch");
    assert_eq!(gpu_state.len(), cpu_state.len(), "state length mismatch");

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (idx, (a, e)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let d = (a - e).abs();
        let rel = d / e.abs().max(1e-3);
        if d > max_abs {
            max_abs = d;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            d <= abs_tol || rel <= rel_tol,
            "{dt:?} out: shape=(T={t_len}, D={conv_dim}) idx={idx}: \
             gpu={a} cpu={e} abs={d:.3e} rel={rel:.3e}",
        );
    }
    eprintln!(
        "{dt:?} out shape=(T={t_len}, D={conv_dim}): max_abs={max_abs:.2e} max_rel={max_rel:.2e}",
    );

    // state_out is pure data move (no further arithmetic) once the
    // T-sweep is over — both kernel and CPU end at the same K-1 most
    // recent inputs from `src`, then quantise back to T on store.
    // After the `dt.round` on both sides, the values should be
    // bit-equal for f32 and within a single dtype-rounding ULP for
    // f16 / bf16. Use the same abs tolerance for safety.
    let mut max_state_abs = 0.0_f32;
    for (idx, (a, e)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_state_abs {
            max_state_abs = d;
        }
        assert!(
            d <= abs_tol,
            "{dt:?} state: shape=(T={t_len}, D={conv_dim}) idx={idx}: \
             gpu={a} cpu={e} abs={d:.3e}",
        );
    }
    eprintln!(
        "{dt:?} state shape=(T={t_len}, D={conv_dim}): max_abs={max_state_abs:.2e}",
    );
}

fn check_dtype(dt: Dt, abs_tol: f32, rel_tol: f32) {
    for &(t_len, conv_dim) in CASES {
        check_one_case(dt, t_len, conv_dim, abs_tol, rel_tol);
    }
}

#[test]
fn conv1d_causal_step_silu_cast_many_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    // f32 path runs the same arithmetic on CPU and GPU — exp/sin/cos
    // can still drop ULPs across the two implementations, so keep a
    // small absolute cushion. Spec target: ≤ 1e-4.
    check_dtype(Dt::F32, 1e-4, 1e-4);
}

#[test]
fn conv1d_causal_step_silu_cast_many_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    // f16 has a 10-bit mantissa — the 6 weight + state loads quantise
    // to f16 on each side, and after 32 T-steps of accumulated round-
    // off the SiLU output diverges by ~2-3 ULPs. Spec target: ≤ 5e-3.
    check_dtype(Dt::F16, 5e-3, 5e-3);
}

#[test]
fn conv1d_causal_step_silu_cast_many_matches_cpu_reference_bf16() {
    let _g = gpu_lock();
    // bf16 has only a 7-bit mantissa — same drift pattern as f16 but
    // amplified by ~8×. Spec target: ≤ 5e-2.
    check_dtype(Dt::Bf16, 5e-2, 5e-2);
}
