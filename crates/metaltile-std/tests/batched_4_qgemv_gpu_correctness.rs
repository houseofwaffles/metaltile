//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::batched_4_qgemv_fast` — fused 4-output
//! 4-bit quantized GEMV (A, B, C, D projections sharing one `x`).
//!
//! Pins: (1) the `program_id::<2>()` matrix selector routes each
//! z-slice to the right weight set across FOUR matrices (one more than
//! the QKV sibling, the place where a hand-extended `if matrix == 3`
//! branch is most likely to drift); (2) each output lands in its OWN
//! buffer (`a_out`/`b_out`/`c_out`/`d_out`) at index `row*`, no
//! concatenation offset; (3) asymmetric out dims route correctly when
//! `out_a > out_b == out_c == out_d` (production GDN shape).
//!
//! Oracle: two passes through `ffai_batched_qkv_qgemv_fast`. Pass 1
//! dispatches with weights (A, B, C) and reads the concatenated
//! `[A | B | C]` output; pass 2 dispatches with (D, D, D) and reads the
//! first slice for D. Both oracle and kernel use the exact same inner
//! loop (mask-without-shift + algebraic-split accumulator), so any
//! divergence isolates a matrix-branch / output-buffer offset bug.
//!
//! Constraints (same as the 3-output sibling):
//!   * `in_dim % 512 == 0`
//!   * `out_a`, `out_b`, `out_c`, `out_d` each a multiple of 8
//!   * `group_size == 64`
//!
//! macOS-gated. Shared `gpu_lock`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::batched_4_qgemv::ffai_batched_4_qgemv_fast;
use metaltile_std::ffai::batched_qkv_qgemv::ffai_batched_qkv_qgemv_fast;

/// Affine per-group int4 quantize of one weight row, nibble-packed.
fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    let n_groups = in_dim / group_size;
    let mut packed = vec![0u32; in_dim / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let gs = &row[g * group_size..(g + 1) * group_size];
        let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in gs.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
            let d = g * group_size + i;
            packed[d / 8] |= q << ((d % 8) * 4);
        }
    }
    (packed, scales, biases)
}

/// Quantize a whole `[out_dim, in_dim]` weight matrix.
fn quantize_matrix(
    rows: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut w = Vec::new();
    let mut s = Vec::new();
    let mut b = Vec::new();
    for row in 0..out_dim {
        let (pw, ps, pb) = quantize_int4_row(&rows[row * in_dim..(row + 1) * in_dim], group_size);
        w.extend(pw);
        s.extend(ps);
        b.extend(pb);
    }
    (w, s, b)
}

fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
        })
        .collect()
}

/// Dispatch the 3-output fast variant. Returns the concatenated
/// `[out_q | out_k | out_v]` output. Used as oracle: pass (A, B, C) to
/// get reference values for the first three projections, then pass
/// (D, D, D) to get the reference for D (read first slice).
#[allow(clippy::too_many_arguments)]
fn run_qkv_oracle(
    x: &[f32],
    wq_p: &[u32],
    sq: &[f32],
    bq: &[f32],
    wk_p: &[u32],
    sk: &[f32],
    bk: &[f32],
    wv_p: &[u32],
    sv: &[f32],
    bv: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_q: usize,
    out_k: usize,
    out_v: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w_q".into(), pack_u32_bytes(wq_p));
    buffers.insert("scales_q".into(), pack_bytes(sq, dt));
    buffers.insert("biases_q".into(), pack_bytes(bq, dt));
    buffers.insert("w_k".into(), pack_u32_bytes(wk_p));
    buffers.insert("scales_k".into(), pack_bytes(sk, dt));
    buffers.insert("biases_k".into(), pack_bytes(bk, dt));
    buffers.insert("w_v".into(), pack_u32_bytes(wv_p));
    buffers.insert("scales_v".into(), pack_bytes(sv, dt));
    buffers.insert("biases_v".into(), pack_bytes(bv, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; out_q + out_k + out_v], dt));
    buffers.insert("out_q".into(), (out_q as u32).to_le_bytes().to_vec());
    buffers.insert("out_k".into(), (out_k as u32).to_le_bytes().to_vec());
    buffers.insert("out_v".into(), (out_v as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_batched_qkv_qgemv_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let max_rows = out_q.max(out_k).max(out_v);
    let n_tgs = max_rows.div_ceil(8);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tgs, 1, 3], [64, 1, 1])
        .expect("batched_qkv_qgemv_fast oracle dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[allow(clippy::too_many_arguments)]
fn run_4_fast(
    x: &[f32],
    wa_p: &[u32],
    sa: &[f32],
    ba: &[f32],
    wb_p: &[u32],
    sb_: &[f32],
    bb: &[f32],
    wc_p: &[u32],
    sc: &[f32],
    bc: &[f32],
    wd_p: &[u32],
    sd: &[f32],
    bd: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_a: usize,
    out_b: usize,
    out_c: usize,
    out_d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w_a".into(), pack_u32_bytes(wa_p));
    buffers.insert("scales_a".into(), pack_bytes(sa, dt));
    buffers.insert("biases_a".into(), pack_bytes(ba, dt));
    buffers.insert("w_b".into(), pack_u32_bytes(wb_p));
    buffers.insert("scales_b".into(), pack_bytes(sb_, dt));
    buffers.insert("biases_b".into(), pack_bytes(bb, dt));
    buffers.insert("w_c".into(), pack_u32_bytes(wc_p));
    buffers.insert("scales_c".into(), pack_bytes(sc, dt));
    buffers.insert("biases_c".into(), pack_bytes(bc, dt));
    buffers.insert("w_d".into(), pack_u32_bytes(wd_p));
    buffers.insert("scales_d".into(), pack_bytes(sd, dt));
    buffers.insert("biases_d".into(), pack_bytes(bd, dt));
    buffers.insert("a_out".into(), pack_bytes(&vec![0.0_f32; out_a], dt));
    buffers.insert("b_out".into(), pack_bytes(&vec![0.0_f32; out_b], dt));
    buffers.insert("c_out".into(), pack_bytes(&vec![0.0_f32; out_c], dt));
    buffers.insert("d_out".into(), pack_bytes(&vec![0.0_f32; out_d], dt));
    buffers.insert("out_a".into(), (out_a as u32).to_le_bytes().to_vec());
    buffers.insert("out_b".into(), (out_b as u32).to_le_bytes().to_vec());
    buffers.insert("out_c".into(), (out_c as u32).to_le_bytes().to_vec());
    buffers.insert("out_d".into(), (out_d as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_batched_4_qgemv_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let max_rows = out_a.max(out_b).max(out_c).max(out_d);
    let n_tgs = max_rows.div_ceil(8);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tgs, 1, 4], [64, 1, 1])
        .expect("batched_4_qgemv_fast dispatch");
    let a = unpack_bytes(result.outputs.get("a_out").expect("a_out"), dt);
    let b = unpack_bytes(result.outputs.get("b_out").expect("b_out"), dt);
    let c = unpack_bytes(result.outputs.get("c_out").expect("c_out"), dt);
    let d = unpack_bytes(result.outputs.get("d_out").expect("d_out"), dt);
    (a, b, c, d)
}

#[allow(clippy::too_many_arguments)]
fn run_case_fast_4(
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_a: usize,
    out_b: usize,
    out_c: usize,
    out_d: usize,
    tol: f32,
) {
    assert_eq!(in_dim % 512, 0, "fast variant requires in_dim % 512 == 0");
    assert_eq!(out_a % 8, 0, "fast variant requires out_a % 8 == 0");
    assert_eq!(out_b % 8, 0, "fast variant requires out_b % 8 == 0");
    assert_eq!(out_c % 8, 0, "fast variant requires out_c % 8 == 0");
    assert_eq!(out_d % 8, 0, "fast variant requires out_d % 8 == 0");
    assert_eq!(group_size, 64, "fast variant requires group_size == 64");

    let _g = gpu_lock();
    // Distinct seeds per matrix so any cross-write between A/B/C/D
    // slices flips the magnitude of the comparison.
    let x: Vec<f32> = source(in_dim, 0x11, 2.0, 0.05).iter().map(|&v| dt.round(v)).collect();
    let wa = source(out_a * in_dim, 0x22, 3.0, 0.0);
    let wb = source(out_b * in_dim, 0x33, 3.0, 0.0);
    let wc = source(out_c * in_dim, 0x44, 3.0, 0.0);
    let wd = source(out_d * in_dim, 0x55, 3.0, 0.0);

    let (wa_p, sa, ba) = quantize_matrix(&wa, out_a, in_dim, group_size);
    let (wb_p, sb_, bb) = quantize_matrix(&wb, out_b, in_dim, group_size);
    let (wc_p, sc, bc) = quantize_matrix(&wc, out_c, in_dim, group_size);
    let (wd_p, sd, bd) = quantize_matrix(&wd, out_d, in_dim, group_size);

    // Oracle pass 1: run the 3-output sibling on (A, B, C). Output is
    // the concatenated [A | B | C] slice.
    let abc_oracle = run_qkv_oracle(
        &x, &wa_p, &sa, &ba, &wb_p, &sb_, &bb, &wc_p, &sc, &bc, dt, in_dim, group_size, out_a,
        out_b, out_c,
    );
    let expected_a: Vec<f32> = abc_oracle[..out_a].to_vec();
    let expected_b: Vec<f32> = abc_oracle[out_a..out_a + out_b].to_vec();
    let expected_c: Vec<f32> = abc_oracle[out_a + out_b..out_a + out_b + out_c].to_vec();

    // Oracle pass 2: D is reached by passing (D, D, D) to the 3-output
    // sibling and reading the first slice. Same inner loop math, just
    // different matrix selector.
    let ddd_oracle = run_qkv_oracle(
        &x, &wd_p, &sd, &bd, &wd_p, &sd, &bd, &wd_p, &sd, &bd, dt, in_dim, group_size, out_d, out_d,
        out_d,
    );
    let expected_d: Vec<f32> = ddd_oracle[..out_d].to_vec();

    let (actual_a, actual_b, actual_c, actual_d) = run_4_fast(
        &x, &wa_p, &sa, &ba, &wb_p, &sb_, &bb, &wc_p, &sc, &bc, &wd_p, &sd, &bd, dt, in_dim,
        group_size, out_a, out_b, out_c, out_d,
    );

    assert_eq!(actual_a.len(), out_a);
    assert_eq!(actual_b.len(), out_b);
    assert_eq!(actual_c.len(), out_c);
    assert_eq!(actual_d.len(), out_d);
    assert!(actual_a.iter().any(|&v| v != 0.0), "a_out is all zeros");
    assert!(actual_b.iter().any(|&v| v != 0.0), "b_out is all zeros");
    assert!(actual_c.iter().any(|&v| v != 0.0), "c_out is all zeros");
    assert!(actual_d.iter().any(|&v| v != 0.0), "d_out is all zeros");

    let max_rel = |actual: &[f32], expected: &[f32]| -> f32 {
        actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
            .fold(0.0_f32, f32::max)
    };
    let ra = max_rel(&actual_a, &expected_a);
    let rb = max_rel(&actual_b, &expected_b);
    let rc = max_rel(&actual_c, &expected_c);
    let rd = max_rel(&actual_d, &expected_d);
    assert!(ra <= tol, "A dt={:?}: max rel = {ra:.3e} > {tol:.3e}", dt as u32);
    assert!(rb <= tol, "B dt={:?}: max rel = {rb:.3e} > {tol:.3e}", dt as u32);
    assert!(rc <= tol, "C dt={:?}: max rel = {rc:.3e} > {tol:.3e}", dt as u32);
    assert!(rd <= tol, "D dt={:?}: max rel = {rd:.3e} > {tol:.3e}", dt as u32);
}

// Production-like GDN input-projection shape from Qwen35: in_dim = 2048
// (hidden), out_a = conv_dim (768), out_b = value_dim (1024),
// out_c = out_d = num_value_heads aligned to 8 (64). Tolerances mirror
// the 3-output sibling tests.

#[test]
fn batched_4_qgemv_fast_f32_gdn() {
    run_case_fast_4(Dt::F32, 2048, 64, 768, 1024, 64, 64, 5e-3);
}

#[test]
fn batched_4_qgemv_fast_f16_gdn() {
    run_case_fast_4(Dt::F16, 2048, 64, 768, 1024, 64, 64, 2e-2);
}

#[test]
fn batched_4_qgemv_fast_bf16_gdn() {
    run_case_fast_4(Dt::Bf16, 2048, 64, 768, 1024, 64, 64, 5e-2);
}

// Smaller-shape smoke for fast iteration when isolating geometry bugs.
#[test]
fn batched_4_qgemv_fast_f32_small() {
    run_case_fast_4(Dt::F32, 512, 64, 16, 8, 8, 8, 5e-3);
}
