//! GPU correctness for `ffai::batched_qkv_qgemv` and
//! `ffai::batched_qkv_qgemv_fast` — fused Q/K/V 4-bit quantized GEMV.
//!
//! Pins: (1) the `program_id::<2>()` matrix selector routes each
//! z-slice to the right weight set; (2) the concatenated output layout
//! `[Q | K | V]`; (3) GQA-style asymmetric out dims (out_k = out_v <
//! out_q). A regression in the selector or the offset cross-writes one
//! projection's rows into another's slice.
//!
//! The fast variant (`ffai_batched_qkv_qgemv_fast`) requires `in_dim`
//! a multiple of 512, `group_size == 64`, and out_q / out_k / out_v each
//! a multiple of 8.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::batched_qkv_qgemv::{ffai_batched_qkv_qgemv, ffai_batched_qkv_qgemv_fast};

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

/// CPU oracle: per-row dequant dot, for one projection matrix.
fn naive_gemv(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
) -> Vec<f32> {
    let u32_per_row = in_dim / 8;
    let n_groups = in_dim / group_size;
    (0..out_dim)
        .map(|row| {
            let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
            let rs = &scales[row * n_groups..(row + 1) * n_groups];
            let rb = &biases[row * n_groups..(row + 1) * n_groups];
            let mut acc = 0.0_f32;
            for (d, &x_d) in x.iter().enumerate().take(in_dim) {
                let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                let g = d / group_size;
                acc += (q as f32 * rs[g] + rb[g]) * x_d;
            }
            acc
        })
        .collect()
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

fn run_case(dt: Dt, in_dim: usize, group_size: usize, out_q: usize, out_k: usize, tol: f32) {
    let _g = gpu_lock();
    let out_v = out_k; // K and V share dims (GQA)
    let x: Vec<f32> = source(in_dim, 0x11, 2.0, 0.05).iter().map(|&v| dt.round(v)).collect();
    let wq = source(out_q * in_dim, 0x22, 3.0, 0.0);
    let wk = source(out_k * in_dim, 0x33, 3.0, 0.0);
    let wv = source(out_v * in_dim, 0x44, 3.0, 0.0);

    let (wq_p, sq, bq) = quantize_matrix(&wq, out_q, in_dim, group_size);
    let (wk_p, sk, bk) = quantize_matrix(&wk, out_k, in_dim, group_size);
    let (wv_p, sv, bv) = quantize_matrix(&wv, out_v, in_dim, group_size);

    let round = |v: &[f32]| v.iter().map(|&x| dt.round(x)).collect::<Vec<_>>();
    let mut expected = naive_gemv(&wq_p, &round(&sq), &round(&bq), &x, in_dim, group_size, out_q);
    expected.extend(naive_gemv(&wk_p, &round(&sk), &round(&bk), &x, in_dim, group_size, out_k));
    expected.extend(naive_gemv(&wv_p, &round(&sv), &round(&bv), &x, in_dim, group_size, out_v));

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, dt));
    buffers.insert("w_q".into(), pack_u32_bytes(&wq_p));
    buffers.insert("scales_q".into(), pack_bytes(&sq, dt));
    buffers.insert("biases_q".into(), pack_bytes(&bq, dt));
    buffers.insert("w_k".into(), pack_u32_bytes(&wk_p));
    buffers.insert("scales_k".into(), pack_bytes(&sk, dt));
    buffers.insert("biases_k".into(), pack_bytes(&bk, dt));
    buffers.insert("w_v".into(), pack_u32_bytes(&wv_p));
    buffers.insert("scales_v".into(), pack_bytes(&sv, dt));
    buffers.insert("biases_v".into(), pack_bytes(&bv, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; out_q + out_k + out_v], dt));
    buffers.insert("out_q".into(), (out_q as u32).to_le_bytes().to_vec());
    buffers.insert("out_k".into(), (out_k as u32).to_le_bytes().to_vec());
    buffers.insert("out_v".into(), (out_v as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_batched_qkv_qgemv::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let max_rows = out_q.max(out_k).max(out_v);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [max_rows, 1, 3], [128, 1, 1])
        .expect("batched_qkv_qgemv dispatch");
    let actual = unpack_bytes(result.outputs.get("out").expect("out"), dt);

    assert_eq!(actual.len(), out_q + out_k + out_v);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(max_rel <= tol, "dt={:?}: max rel = {max_rel:.3e} > {tol:.3e}", dt as u32);
}

#[test]
fn batched_qkv_qgemv_f32_gqa() {
    // GQA shape: 16 query rows, 4 K/V rows each.
    run_case(Dt::F32, 256, 64, 16, 4, 5e-3);
}

#[test]
fn batched_qkv_qgemv_f32_gs128() { run_case(Dt::F32, 512, 128, 12, 6, 5e-3); }

#[test]
fn batched_qkv_qgemv_f16_gqa() { run_case(Dt::F16, 256, 64, 16, 4, 2e-2); }

#[test]
fn batched_qkv_qgemv_bf16_gqa() { run_case(Dt::Bf16, 256, 64, 16, 4, 5e-2); }

// ── ffai_batched_qkv_qgemv_fast ──────────────────────────────────────────
//
// Fast 8-row-per-TG variant: `in_dim` must be a multiple of 512;
// `out_q`, `out_k`, `out_v` must each be a multiple of 8;
// `group_size` must be 64.
// Grid: [ceil(max(out_q,out_k,out_v)/8), 1, 3]; TPG = 64.

#[allow(clippy::too_many_arguments)]
fn run_fast_qkv(
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
        .expect("batched_qkv_qgemv_fast dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

fn run_case_fast_qkv(
    dt: Dt,
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
    let x: Vec<f32> = source(in_dim, 0x11, 2.0, 0.05).iter().map(|&v| dt.round(v)).collect();
    let wq = source(out_q * in_dim, 0x22, 3.0, 0.0);
    let wk = source(out_k * in_dim, 0x33, 3.0, 0.0);
    let wv = source(out_v * in_dim, 0x44, 3.0, 0.0);

    let (wq_p, sq, bq) = quantize_matrix(&wq, out_q, in_dim, group_size);
    let (wk_p, sk, bk) = quantize_matrix(&wk, out_k, in_dim, group_size);
    let (wv_p, sv, bv) = quantize_matrix(&wv, out_v, in_dim, group_size);

    let round = |v: &[f32]| v.iter().map(|&x| dt.round(x)).collect::<Vec<_>>();
    let mut expected = naive_gemv(&wq_p, &round(&sq), &round(&bq), &x, in_dim, group_size, out_q);
    expected.extend(naive_gemv(&wk_p, &round(&sk), &round(&bk), &x, in_dim, group_size, out_k));
    expected.extend(naive_gemv(&wv_p, &round(&sv), &round(&bv), &x, in_dim, group_size, out_v));

    let actual = run_fast_qkv(
        &x, &wq_p, &sq, &bq, &wk_p, &sk, &bk, &wv_p, &sv, &bv, dt, in_dim, group_size, out_q,
        out_k, out_v,
    );

    assert_eq!(actual.len(), out_q + out_k + out_v);
    assert!(actual.iter().any(|&v| v != 0.0), "fast output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(max_rel <= tol, "fast dt={:?}: max rel = {max_rel:.3e} > {tol:.3e}", dt as u32);
}

#[test]
fn batched_qkv_qgemv_fast_f32_gqa() {
    // GQA shape: out_q=16, out_k=out_v=8; in_dim=512, gs=64.
    run_case_fast_qkv(Dt::F32, 512, 64, 16, 8, 5e-3);
}

#[test]
fn batched_qkv_qgemv_fast_f16_gqa() { run_case_fast_qkv(Dt::F16, 512, 64, 16, 8, 2e-2); }

#[test]
fn batched_qkv_qgemv_fast_bf16_gqa() { run_case_fast_qkv(Dt::Bf16, 512, 64, 16, 8, 5e-2); }
