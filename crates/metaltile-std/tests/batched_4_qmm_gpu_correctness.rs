//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::batched_4_qmm_fast` — fused 4-output int4
//! quantized QMM at M>1 (prefill path).
//!
//! Pins: (1) row-m indexing of `x` and the four output buffers is
//! correct so projections for different m's don't bleed into each
//! other; (2) each output lands in its OWN buffer (`a_buf`/`b_buf`/
//! `c_buf`/`d_buf`) at index `m*out_* + row`, no concatenation offset;
//! (3) the `program_id::<2>()` matrix selector routes each z-slice to
//! the right weight set across FOUR matrices (the place where a
//! hand-extended `if matrix == 3` branch is most likely to drift);
//! (4) asymmetric out dims route correctly when `out_a == out_b >>
//! out_c == out_d` (production GDN shape: qkv/z big, b/a small).
//!
//! Oracle is `ffai_batched_4_qgemv_fast` dispatched M times — one per
//! row of `x` — with per-row outputs concatenated. Both kernels share
//! the same inner-loop math (mask-without-shift + algebraic-split
//! accumulator), so any divergence isolates a row-offset / matrix-
//! branch bug rather than a numerical drift.
//!
//! Constraints (same as the GEMV-fast 4-output sibling):
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
use metaltile_std::ffai::batched_4_qmm::ffai_batched_4_qmm_fast;

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

/// Dispatch the 4-output GEMV-fast variant on a single row of x.
/// Returns the four per-row outputs as separate vectors. Oracle for
/// the M>1 kernel: run this M times and concatenate.
#[allow(clippy::too_many_arguments)]
fn run_4_gemv_fast(
    x_row: &[f32],
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
    buffers.insert("x".into(), pack_bytes(x_row, dt));
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
        .expect("batched_4_qgemv_fast oracle dispatch");
    let a = unpack_bytes(result.outputs.get("a_out").expect("a_out"), dt);
    let b = unpack_bytes(result.outputs.get("b_out").expect("b_out"), dt);
    let c = unpack_bytes(result.outputs.get("c_out").expect("c_out"), dt);
    let d = unpack_bytes(result.outputs.get("d_out").expect("d_out"), dt);
    (a, b, c, d)
}

#[allow(clippy::too_many_arguments)]
fn run_4_qmm_fast(
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
    m: usize,
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
    buffers.insert("a_buf".into(), pack_bytes(&vec![0.0_f32; m * out_a], dt));
    buffers.insert("b_buf".into(), pack_bytes(&vec![0.0_f32; m * out_b], dt));
    buffers.insert("c_buf".into(), pack_bytes(&vec![0.0_f32; m * out_c], dt));
    buffers.insert("d_buf".into(), pack_bytes(&vec![0.0_f32; m * out_d], dt));
    buffers.insert("out_a".into(), (out_a as u32).to_le_bytes().to_vec());
    buffers.insert("out_b".into(), (out_b as u32).to_le_bytes().to_vec());
    buffers.insert("out_c".into(), (out_c as u32).to_le_bytes().to_vec());
    buffers.insert("out_d".into(), (out_d as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_batched_4_qmm_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let max_rows = out_a.max(out_b).max(out_c).max(out_d);
    let n_tgs = max_rows.div_ceil(8);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tgs, m, 4], [64, 1, 1])
        .expect("batched_4_qmm_fast dispatch");
    let a = unpack_bytes(result.outputs.get("a_buf").expect("a_buf"), dt);
    let b = unpack_bytes(result.outputs.get("b_buf").expect("b_buf"), dt);
    let c = unpack_bytes(result.outputs.get("c_buf").expect("c_buf"), dt);
    let d = unpack_bytes(result.outputs.get("d_buf").expect("d_buf"), dt);
    (a, b, c, d)
}

#[allow(clippy::too_many_arguments)]
fn run_case_qmm_4(
    dt: Dt,
    m: usize,
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
    // slices flips the magnitude of the comparison. Distinct rows per m
    // so row-offset bugs aren't masked.
    let x: Vec<f32> = source(m * in_dim, 0x11, 2.0, 0.05).iter().map(|&v| dt.round(v)).collect();
    let wa = source(out_a * in_dim, 0x22, 3.0, 0.0);
    let wb = source(out_b * in_dim, 0x33, 3.0, 0.0);
    let wc = source(out_c * in_dim, 0x44, 3.0, 0.0);
    let wd = source(out_d * in_dim, 0x55, 3.0, 0.0);

    let (wa_p, sa, ba) = quantize_matrix(&wa, out_a, in_dim, group_size);
    let (wb_p, sb_, bb) = quantize_matrix(&wb, out_b, in_dim, group_size);
    let (wc_p, sc, bc) = quantize_matrix(&wc, out_c, in_dim, group_size);
    let (wd_p, sd, bd) = quantize_matrix(&wd, out_d, in_dim, group_size);

    // Oracle: run the 4-output GEMV-fast variant M times, concatenate
    // per-row outputs row-major into the four `[M, out_*]` reference
    // buffers.
    let mut expected_a: Vec<f32> = Vec::with_capacity(m * out_a);
    let mut expected_b: Vec<f32> = Vec::with_capacity(m * out_b);
    let mut expected_c: Vec<f32> = Vec::with_capacity(m * out_c);
    let mut expected_d: Vec<f32> = Vec::with_capacity(m * out_d);
    for row in 0..m {
        let x_row = &x[row * in_dim..(row + 1) * in_dim];
        let (ea, eb, ec, ed) = run_4_gemv_fast(
            x_row, &wa_p, &sa, &ba, &wb_p, &sb_, &bb, &wc_p, &sc, &bc, &wd_p, &sd, &bd, dt, in_dim,
            group_size, out_a, out_b, out_c, out_d,
        );
        expected_a.extend(ea);
        expected_b.extend(eb);
        expected_c.extend(ec);
        expected_d.extend(ed);
    }

    let (actual_a, actual_b, actual_c, actual_d) = run_4_qmm_fast(
        &x, &wa_p, &sa, &ba, &wb_p, &sb_, &bb, &wc_p, &sc, &bc, &wd_p, &sd, &bd, dt, m, in_dim,
        group_size, out_a, out_b, out_c, out_d,
    );

    assert_eq!(actual_a.len(), m * out_a);
    assert_eq!(actual_b.len(), m * out_b);
    assert_eq!(actual_c.len(), m * out_c);
    assert_eq!(actual_d.len(), m * out_d);
    assert!(actual_a.iter().any(|&v| v != 0.0), "a_buf is all zeros");
    assert!(actual_b.iter().any(|&v| v != 0.0), "b_buf is all zeros");
    assert!(actual_c.iter().any(|&v| v != 0.0), "c_buf is all zeros");
    assert!(actual_d.iter().any(|&v| v != 0.0), "d_buf is all zeros");

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
    assert!(ra <= tol, "A dt={:?} m={m}: max rel = {ra:.3e} > {tol:.3e}", dt as u32);
    assert!(rb <= tol, "B dt={:?} m={m}: max rel = {rb:.3e} > {tol:.3e}", dt as u32);
    assert!(rc <= tol, "C dt={:?} m={m}: max rel = {rc:.3e} > {tol:.3e}", dt as u32);
    assert!(rd <= tol, "D dt={:?} m={m}: max rel = {rd:.3e} > {tol:.3e}", dt as u32);
}

// ── ffai_batched_4_qmm_fast ───────────────────────────────────────────
// Production-like GDN input-projection shape: in_dim = 2048 (hidden),
// out_a = out_b = 2048 (qkv + z), out_c = out_d = 16 (b + a; aligned
// up to 16 = next multiple of 8). Tolerances mirror the 4-output GEMV
// sibling tests.

// ── M=2 (smallest M>1) ────────────────────────────────────────────────

#[test]
fn batched_4_qmm_fast_f32_gdn_m2() {
    run_case_qmm_4(Dt::F32, 2, 2048, 64, 2048, 2048, 16, 16, 5e-3);
}

#[test]
fn batched_4_qmm_fast_f16_gdn_m2() {
    run_case_qmm_4(Dt::F16, 2, 2048, 64, 2048, 2048, 16, 16, 2e-2);
}

#[test]
fn batched_4_qmm_fast_bf16_gdn_m2() {
    run_case_qmm_4(Dt::Bf16, 2, 2048, 64, 2048, 2048, 16, 16, 5e-2);
}

// ── M=8 (common prefill chunk granularity) ────────────────────────────

#[test]
fn batched_4_qmm_fast_f32_mid_m8() {
    run_case_qmm_4(Dt::F32, 8, 1024, 64, 512, 256, 8, 8, 5e-3);
}

#[test]
fn batched_4_qmm_fast_f16_mid_m8() {
    run_case_qmm_4(Dt::F16, 8, 1024, 64, 512, 256, 8, 8, 2e-2);
}

#[test]
fn batched_4_qmm_fast_bf16_mid_m8() {
    run_case_qmm_4(Dt::Bf16, 8, 1024, 64, 512, 256, 8, 8, 5e-2);
}

// ── M=32 (larger prefill chunk) ───────────────────────────────────────

#[test]
fn batched_4_qmm_fast_f32_small_m32() {
    run_case_qmm_4(Dt::F32, 32, 512, 64, 8, 8, 8, 8, 5e-3);
}

#[test]
fn batched_4_qmm_fast_f16_small_m32() {
    run_case_qmm_4(Dt::F16, 32, 512, 64, 8, 8, 8, 8, 2e-2);
}

#[test]
fn batched_4_qmm_fast_bf16_small_m32() {
    run_case_qmm_4(Dt::Bf16, 32, 512, 64, 8, 8, 8, 8, 5e-2);
}
