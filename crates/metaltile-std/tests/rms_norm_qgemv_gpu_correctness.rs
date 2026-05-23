//! GPU correctness for `ffai::rms_norm_qgemv` and
//! `ffai::rms_norm_qgemv_fast` — fused RMSNorm + 4-bit quantized GEMV for
//! decode.
//!
//! Pins that both kernels compute
//! `y = qmatmul(rms_norm(x) * norm_weight, W_q)` — i.e. the RMSNorm scale
//! (reduced over the whole input vector) is applied to the activation
//! *before* the quantized dot, and the norm weight multiplies in too.
//! A regression that drops the norm, or folds `norm_weight` into the SSQ,
//! only shows as drifting logits in FFAI integration; the naive f32 oracle
//! pins it.
//!
//! The fast variant (`ffai_rms_norm_qgemv_fast`) uses the `mt_qmv`
//! 8-row-per-TG geometry and requires `in_dim` a multiple of 512 and
//! `out_dim` a multiple of 8. Its tests use shapes that satisfy those
//! constraints.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rms_norm_qgemv::{ffai_rms_norm_qgemv, ffai_rms_norm_qgemv_fast};

/// Affine per-group int4 quantize of one weight row, nibble-packed
/// 8 values per u32 (the pack-strided layout the kernel decodes).
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

#[allow(clippy::too_many_arguments)]
fn naive(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    norm_weight: &[f32],
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let ssq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (ssq / in_dim as f32 + eps).sqrt();
    let u32_per_row = in_dim / 8;
    let n_groups = in_dim / group_size;
    (0..out_dim)
        .map(|row| {
            let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
            let rs = &scales[row * n_groups..(row + 1) * n_groups];
            let rb = &biases[row * n_groups..(row + 1) * n_groups];
            let mut acc = 0.0_f32;
            for d in 0..in_dim {
                let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                let g = d / group_size;
                let w_real = q as f32 * rs[g] + rb[g];
                let normed = x[d] * norm_weight[d] * inv_rms;
                acc += w_real * normed;
            }
            acc
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    norm_weight: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("norm_weight".into(), pack_bytes(norm_weight, dt));
    buffers.insert("weight".into(), pack_u32_bytes(weight));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("output".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_rms_norm_qgemv::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim, 1, 1], [128, 1, 1])
        .expect("rms_norm_qgemv dispatch");
    unpack_bytes(result.outputs.get("output").expect("output"), dt)
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

fn run_case(dt: Dt, in_dim: usize, group_size: usize, out_dim: usize, tol: f32) {
    let _g = gpu_lock();
    let eps = 1e-5_f32;
    let x: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1).iter().map(|&v| dt.round(v)).collect();
    let norm_weight: Vec<f32> =
        source(in_dim, 0xB2, 0.4, 1.0).iter().map(|&v| dt.round(v)).collect();
    let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);

    let u32_per_row = in_dim / 8;
    let n_groups = in_dim / group_size;
    let mut weight = Vec::with_capacity(u32_per_row * out_dim);
    let mut scales = Vec::with_capacity(n_groups * out_dim);
    let mut biases = Vec::with_capacity(n_groups * out_dim);
    for row in 0..out_dim {
        let (w, s, b) = quantize_int4_row(&w_rows[row * in_dim..(row + 1) * in_dim], group_size);
        weight.extend(w);
        scales.extend(s);
        biases.extend(b);
    }
    let scales_r: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_r: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected =
        naive(&weight, &scales_r, &biases_r, &x, &norm_weight, in_dim, group_size, out_dim, eps);
    let actual =
        run(&weight, &scales, &biases, &x, &norm_weight, dt, in_dim, group_size, out_dim, eps);

    assert_eq!(actual.len(), out_dim);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel <= tol,
        "dt={:?} in_dim={in_dim}: max rel = {max_rel:.3e} > {tol:.3e}",
        dt as u32
    );
}

#[test]
fn rms_norm_qgemv_f32_gs64() { run_case(Dt::F32, 256, 64, 8, 5e-3); }

#[test]
fn rms_norm_qgemv_f32_gs128() { run_case(Dt::F32, 512, 128, 6, 5e-3); }

#[test]
fn rms_norm_qgemv_f16_gs64() { run_case(Dt::F16, 256, 64, 8, 2e-2); }

#[test]
fn rms_norm_qgemv_bf16_gs64() { run_case(Dt::Bf16, 256, 64, 8, 5e-2); }

// ── ffai_rms_norm_qgemv_fast ─────────────────────────────────────────────
//
// Fast 8-row-per-TG variant: `in_dim` must be a multiple of 512;
// `out_dim` must be a multiple of 8; `group_size` must be 64.
// Use `tpg=64` and `grid=[out_dim/8, 1, 1]`.

#[allow(clippy::too_many_arguments)]
fn run_fast(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    norm_weight: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("norm_weight".into(), pack_bytes(norm_weight, dt));
    buffers.insert("weight".into(), pack_u32_bytes(weight));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("output".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_rms_norm_qgemv_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [out_dim/8, 1, 1]; TPG = 64 (2 SG × 32 lanes).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim / 8, 1, 1], [64, 1, 1])
        .expect("rms_norm_qgemv_fast dispatch");
    unpack_bytes(result.outputs.get("output").expect("output"), dt)
}

fn run_case_fast(dt: Dt, in_dim: usize, group_size: usize, out_dim: usize, tol: f32) {
    // Constraints: in_dim % 512 == 0, out_dim % 8 == 0, group_size == 64.
    assert_eq!(in_dim % 512, 0, "fast variant requires in_dim % 512 == 0");
    assert_eq!(out_dim % 8, 0, "fast variant requires out_dim % 8 == 0");
    assert_eq!(group_size, 64, "fast variant requires group_size == 64");

    let _g = gpu_lock();
    let eps = 1e-5_f32;
    let x: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1).iter().map(|&v| dt.round(v)).collect();
    let norm_weight: Vec<f32> =
        source(in_dim, 0xB2, 0.4, 1.0).iter().map(|&v| dt.round(v)).collect();
    let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);

    let u32_per_row = in_dim / 8;
    let n_groups = in_dim / group_size;
    let mut weight = Vec::with_capacity(u32_per_row * out_dim);
    let mut scales = Vec::with_capacity(n_groups * out_dim);
    let mut biases = Vec::with_capacity(n_groups * out_dim);
    for row in 0..out_dim {
        let (w, s, b) = quantize_int4_row(&w_rows[row * in_dim..(row + 1) * in_dim], group_size);
        weight.extend(w);
        scales.extend(s);
        biases.extend(b);
    }
    let scales_r: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_r: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected =
        naive(&weight, &scales_r, &biases_r, &x, &norm_weight, in_dim, group_size, out_dim, eps);
    let actual =
        run_fast(&weight, &scales, &biases, &x, &norm_weight, dt, in_dim, group_size, out_dim, eps);

    assert_eq!(actual.len(), out_dim);
    assert!(actual.iter().any(|&v| v != 0.0), "fast output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel <= tol,
        "fast dt={:?} in_dim={in_dim}: max rel = {max_rel:.3e} > {tol:.3e}",
        dt as u32
    );
}

#[test]
fn rms_norm_qgemv_fast_f32_gs64() {
    // in_dim=512 (= 1×512), out_dim=16 (= 2×8), gs=64.
    run_case_fast(Dt::F32, 512, 64, 16, 5e-3);
}

#[test]
fn rms_norm_qgemv_fast_f16_gs64() { run_case_fast(Dt::F16, 512, 64, 16, 2e-2); }

#[test]
fn rms_norm_qgemv_fast_bf16_gs64() { run_case_fast(Dt::Bf16, 512, 64, 16, 5e-2); }

#[test]
fn rms_norm_qgemv_fast_f32_large() {
    // Larger shape: in_dim=1024, out_dim=32.
    run_case_fast(Dt::F32, 1024, 64, 32, 5e-3);
}
