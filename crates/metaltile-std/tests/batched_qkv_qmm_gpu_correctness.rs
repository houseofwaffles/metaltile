//! GPU correctness for `ffai::batched_qkv_qmm_fast` — fused Q/K/V int4
//! quantized QMM at M>1 (prefill path).
//!
//! Pins: (1) row-m indexing of `x` and `out` is correct so projections
//! for different m's don't bleed into each other; (2) the
//! `[Q | K | V]` per-row layout matches the GEMV-fast variant (callers
//! slice the output the same way at prefill and decode); (3) the GQA
//! asymmetric out dims (out_k = out_v < out_q) route correctly.
//!
//! Oracle is `ffai_batched_qkv_qgemv_fast` dispatched M times — one per
//! row of `x` — with per-row outputs concatenated. Both kernels share
//! the same inner-loop math (mask-without-shift + algebraic-split
//! accumulator), so any divergence isolates a row-offset bug rather than
//! a numerical drift.
//!
//! Constraints (same as the GEMV-fast variant):
//!   * `in_dim % 512 == 0`
//!   * `out_q`, `out_k`, `out_v` each a multiple of 8
//!   * `group_size == 64`
//!
//! macOS-gated. Shared `gpu_lock`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::batched_qkv_qgemv::ffai_batched_qkv_qgemv_fast;
use metaltile_std::ffai::batched_qkv_qmm::ffai_batched_qkv_qmm_fast;

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

/// Dispatch the GEMV-fast variant for a single row of x. Returns the
/// `out_q + out_k + out_v` concatenated output.
#[allow(clippy::too_many_arguments)]
fn run_gemv_fast(
    x_row: &[f32],
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
    buffers.insert("x".into(), pack_bytes(x_row, dt));
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
        .expect("batched_qkv_qgemv_fast dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[allow(clippy::too_many_arguments)]
fn run_qmm_fast(
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
    m: usize,
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
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; m * (out_q + out_k + out_v)], dt));
    buffers.insert("out_q".into(), (out_q as u32).to_le_bytes().to_vec());
    buffers.insert("out_k".into(), (out_k as u32).to_le_bytes().to_vec());
    buffers.insert("out_v".into(), (out_v as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_batched_qkv_qmm_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let max_rows = out_q.max(out_k).max(out_v);
    let n_tgs = max_rows.div_ceil(8);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tgs, m, 3], [64, 1, 1])
        .expect("batched_qkv_qmm_fast dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

fn run_case_qmm(
    dt: Dt,
    m: usize,
    in_dim: usize,
    group_size: usize,
    out_q: usize,
    out_k: usize,
    tol: f32,
) {
    assert_eq!(in_dim % 512, 0, "fast variant requires in_dim % 512 == 0");
    assert_eq!(out_q % 8, 0, "fast variant requires out_q % 8 == 0");
    assert_eq!(out_k % 8, 0, "fast variant requires out_k % 8 == 0");
    assert_eq!(group_size, 64, "fast variant requires group_size == 64");

    let _g = gpu_lock();
    let out_v = out_k;
    // Distinct seed per row so we catch row-offset bugs (would otherwise
    // be invisible if every row had the same x).
    let x: Vec<f32> = source(m * in_dim, 0x11, 2.0, 0.05).iter().map(|&v| dt.round(v)).collect();
    let wq = source(out_q * in_dim, 0x22, 3.0, 0.0);
    let wk = source(out_k * in_dim, 0x33, 3.0, 0.0);
    let wv = source(out_v * in_dim, 0x44, 3.0, 0.0);

    let (wq_p, sq, bq) = quantize_matrix(&wq, out_q, in_dim, group_size);
    let (wk_p, sk, bk) = quantize_matrix(&wk, out_k, in_dim, group_size);
    let (wv_p, sv, bv) = quantize_matrix(&wv, out_v, in_dim, group_size);

    // Oracle: run the GEMV-fast variant M times and concatenate.
    let mut expected: Vec<f32> = Vec::with_capacity(m * (out_q + out_k + out_v));
    for row in 0..m {
        let x_row = &x[row * in_dim..(row + 1) * in_dim];
        let row_out = run_gemv_fast(
            x_row, &wq_p, &sq, &bq, &wk_p, &sk, &bk, &wv_p, &sv, &bv, dt, in_dim, group_size,
            out_q, out_k, out_v,
        );
        expected.extend(row_out);
    }

    let actual = run_qmm_fast(
        &x, &wq_p, &sq, &bq, &wk_p, &sk, &bk, &wv_p, &sv, &bv, dt, m, in_dim, group_size, out_q,
        out_k, out_v,
    );

    assert_eq!(actual.len(), m * (out_q + out_k + out_v));
    assert!(actual.iter().any(|&v| v != 0.0), "qmm output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(max_rel <= tol, "qmm dt={:?} m={m}: max rel = {max_rel:.3e} > {tol:.3e}", dt as u32);
}

// ── ffai_batched_qkv_qmm_fast ────────────────────────────────────────────

#[test]
fn batched_qkv_qmm_fast_f32_gqa_m2() {
    // Smallest M>1 case: GQA shape out_q=16, out_k=out_v=8; M=2.
    run_case_qmm(Dt::F32, 2, 512, 64, 16, 8, 5e-3);
}

#[test]
fn batched_qkv_qmm_fast_f32_gqa_m8() {
    // M=8 — common prefill chunk granularity.
    run_case_qmm(Dt::F32, 8, 512, 64, 16, 8, 5e-3);
}

#[test]
fn batched_qkv_qmm_fast_f16_gqa_m4() { run_case_qmm(Dt::F16, 4, 512, 64, 16, 8, 2e-2); }

#[test]
fn batched_qkv_qmm_fast_bf16_gqa_m4() { run_case_qmm(Dt::Bf16, 4, 512, 64, 16, 8, 5e-2); }
