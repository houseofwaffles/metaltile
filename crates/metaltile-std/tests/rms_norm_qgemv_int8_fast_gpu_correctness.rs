//! GPU correctness for `ffai_rms_norm_qgemv_int8_fast` — fused RMSNorm +
//! int8-quantized 8-row-per-TG GEMV.
//!
//! CPU oracle: same fused formula as the int4 fast test —
//!   `y = qmatmul(rms_norm(x) * norm_weight, W_q)`
//! but weight codes are 8-bit (0-255) packed 4 per u32.
//!
//! Constraints (inherited from the kernel):
//!   `in_dim % 512 == 0` (TG covers full vector),
//!   `out_dim % 8 == 0` (8 rows per TG),
//!   `group_size == 64`.
//!
//! Grid: `[out_dim/8, 1, 1]`, TPG = 64 (2 SG × 32 lanes).
//!
//! Tolerances (relative): f32 ≤ 5e-3, f16 ≤ 2e-2, bf16 ≤ 5e-2.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rms_norm_qgemv::ffai_rms_norm_qgemv_int8_fast;

/// Affine per-group int8 quantize of one weight row, byte-packed
/// 4 values per u32. Returns (packed, scales, biases).
fn quantize_int8_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    let n_groups = in_dim / group_size;
    // 4 bytes per u32.
    let mut packed = vec![0u32; in_dim / 4];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let gs = &row[g * group_size..(g + 1) * group_size];
        let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / 255.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in gs.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 255.0) as u32;
            let d = g * group_size + i;
            // 4 bytes per pack: byte d%4 is shifted d%4 * 8 bits.
            packed[d / 4] |= q << ((d % 4) * 8);
        }
    }
    (packed, scales, biases)
}

/// CPU oracle: fused RMSNorm + int8 GEMV.
#[allow(clippy::too_many_arguments)]
fn naive_int8(
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
    // packs_per_row: int8 has 4 bytes/u32.
    let packs_per_row = in_dim / 4;
    let n_groups = in_dim / group_size;
    (0..out_dim)
        .map(|row| {
            let rw = &weight[row * packs_per_row..(row + 1) * packs_per_row];
            let rs = &scales[row * n_groups..(row + 1) * n_groups];
            let rb = &biases[row * n_groups..(row + 1) * n_groups];
            let mut acc = 0.0_f32;
            for d in 0..in_dim {
                let q = (rw[d / 4] >> ((d % 4) * 8)) & 0xFF;
                let g = d / group_size;
                let w_real = q as f32 * rs[g] + rb[g];
                let normed = x[d] * norm_weight[d] * inv_rms;
                acc += w_real * normed;
            }
            acc
        })
        .collect()
}

/// Dispatch `ffai_rms_norm_qgemv_int8_fast` and return f32-decoded output.
#[allow(clippy::too_many_arguments)]
fn run_int8_fast(
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
    assert!(in_dim.is_multiple_of(512), "int8_fast: in_dim must be multiple of 512");
    assert!(out_dim.is_multiple_of(8), "int8_fast: out_dim must be multiple of 8");
    assert_eq!(group_size, 64, "int8_fast: group_size must be 64");

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
    let mut kernel = ffai_rms_norm_qgemv_int8_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [out_dim/8, 1, 1]; TPG = 64 (2 SG × 32 lanes).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim / 8, 1, 1], [64, 1, 1])
        .expect("rms_norm_qgemv_int8_fast dispatch");
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
    // Constraints: in_dim % 512 == 0, out_dim % 8 == 0, group_size == 64.
    assert_eq!(in_dim % 512, 0);
    assert_eq!(out_dim % 8, 0);
    assert_eq!(group_size, 64);

    let _g = gpu_lock();
    let eps = 1e-5_f32;
    let x: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1).iter().map(|&v| dt.round(v)).collect();
    let norm_weight: Vec<f32> =
        source(in_dim, 0xB2, 0.4, 1.0).iter().map(|&v| dt.round(v)).collect();
    let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);

    let packs_per_row = in_dim / 4;
    let n_groups = in_dim / group_size;
    let mut weight = Vec::with_capacity(packs_per_row * out_dim);
    let mut scales = Vec::with_capacity(n_groups * out_dim);
    let mut biases = Vec::with_capacity(n_groups * out_dim);
    for row in 0..out_dim {
        let (w, s, b) = quantize_int8_row(&w_rows[row * in_dim..(row + 1) * in_dim], group_size);
        weight.extend(w);
        scales.extend(s);
        biases.extend(b);
    }
    let scales_r: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_r: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected = naive_int8(
        &weight,
        &scales_r,
        &biases_r,
        &x,
        &norm_weight,
        in_dim,
        group_size,
        out_dim,
        eps,
    );
    let actual = run_int8_fast(
        &weight,
        &scales,
        &biases,
        &x,
        &norm_weight,
        dt,
        in_dim,
        group_size,
        out_dim,
        eps,
    );

    assert_eq!(actual.len(), out_dim);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    eprintln!("[int8_fast {dt:?} in={in_dim} out={out_dim}] max_rel={max_rel:.3e} (tol {tol:.3e})");
    assert!(
        max_rel <= tol,
        "int8_fast dt={dt:?} in_dim={in_dim}: max rel = {max_rel:.3e} > {tol:.3e}",
    );
}

#[test]
fn rms_norm_qgemv_int8_fast_f32_gs64_small() {
    // in_dim=512, out_dim=8 (smallest valid shape).
    run_case(Dt::F32, 512, 64, 8, 5e-3);
}

#[test]
fn rms_norm_qgemv_int8_fast_f32_gs64_large() {
    // in_dim=1024, out_dim=32.
    run_case(Dt::F32, 1024, 64, 32, 5e-3);
}

#[test]
fn rms_norm_qgemv_int8_fast_f16_gs64() { run_case(Dt::F16, 512, 64, 16, 2e-2); }

#[test]
fn rms_norm_qgemv_int8_fast_bf16_gs64() { run_case(Dt::Bf16, 512, 64, 16, 5e-2); }
