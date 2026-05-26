//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::gated_rms_norm_qgemv_int4_fast` - the
//! fused gated-RMSNorm + int4 GEMV that closes every Qwen3.5 / Qwen3.6
//! GDN layer.
//!
//! The kernel collapses the two-stage chain
//!
//! ```text
//!   inner[r, d] = w[d] * y[r, d] * rsqrt(mean(y[r]^2) + eps) * silu(z[r, d])
//!   out[o]      = sum_i (q[o, i] * scale + bias) * inner_flat[i]
//! ```
//!
//! where `y: [Hv, Dv]` is fp32 (the GDN recurrence output), `z`, `w`,
//! and `out` are model dtype `T`, and `q`, `scale`, `bias` are the int4
//! out-projection. Per-row RMS (not full Hv*Dv) and post-norm `silu(z)`
//! gating are both load-bearing - a regression that folds them into a
//! single global RMS, or moves the gate before the norm, only drifts
//! logits in FFAI integration. The oracle pins them.
//!
//! Fast 8-row-per-TG variant: `in_dim = Hv * Dv` a multiple of 512,
//! `out_dim` a multiple of 8, `group_size = 64`, `dv` a multiple of 32,
//! `hv` even.
//!
//! macOS-gated. Shared `gpu_lock`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_rms_norm_qgemv::ffai_gated_rms_norm_qgemv_int4_fast;

/// Per-row affine int4 quantize, nibble-packed 8-per-u32 - same packing
/// the kernel decodes. Group size and the row are both passed in.
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
    y: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    hv: usize,
    dv: usize,
    out_dim: usize,
    group_size: usize,
    eps: f32,
) -> Vec<f32> {
    let in_dim = hv * dv;
    // Per-row gated RMSNorm into a flat [hv * dv] vector.
    let mut inner = vec![0.0_f32; in_dim];
    for r in 0..hv {
        let base = r * dv;
        let row = &y[base..base + dv];
        let ssq: f32 = row.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ssq / dv as f32 + eps).sqrt();
        for d in 0..dv {
            let g = z[base + d] / (1.0 + (-z[base + d]).exp());
            inner[base + d] = y[base + d] * inv_rms * norm_weight[d] * g;
        }
    }
    // Out-projection: int4 GEMV over the flattened gated-norm output.
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
                acc += w_real * inner[d];
            }
            acc
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    y: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    dt: Dt,
    hv: usize,
    dv: usize,
    out_dim: usize,
    group_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // `y` is always fp32 (the GDN recurrence output stays in fp32 across
    // the kernel boundary).
    buffers.insert("y".into(), pack_bytes(y, Dt::F32));
    buffers.insert("z".into(), pack_bytes(z, dt));
    buffers.insert("norm_weight".into(), pack_bytes(norm_weight, dt));
    buffers.insert("q_weight".into(), pack_u32_bytes(weight));
    buffers.insert("q_scales".into(), pack_bytes(scales, dt));
    buffers.insert("q_biases".into(), pack_bytes(biases, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_gated_rms_norm_qgemv_int4_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [out_dim / 8, 1, 1]; TPG = 64 (2 simdgroups x 32 lanes).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim / 8, 1, 1], [64, 1, 1])
        .expect("gated_rms_norm_qgemv_int4_fast dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
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

fn run_case(dt: Dt, hv: usize, dv: usize, out_dim: usize, group_size: usize, tol: f32) {
    // Fast-variant constraints (mirrored from the kernel doc).
    let in_dim = hv * dv;
    assert_eq!(in_dim % 512, 0, "in_dim (= hv*dv) must be a multiple of 512");
    assert_eq!(out_dim % 8, 0, "out_dim must be a multiple of 8");
    assert_eq!(group_size, 64, "group_size must be 64");
    assert_eq!(dv % 32, 0, "dv must be a multiple of 32");
    assert!(hv.is_multiple_of(2), "hv must be even");
    assert!(in_dim <= 8192, "in_dim must fit in the 8192-element tg_inner buffer");

    let _g = gpu_lock();
    let eps = 1e-5_f32;
    // `y` stays fp32 - no per-dtype rounding.
    let y: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1);
    let z: Vec<f32> = source(in_dim, 0xD4, 1.5, 0.0).iter().map(|&v| dt.round(v)).collect();
    let norm_weight: Vec<f32> = source(dv, 0xB2, 0.4, 1.0).iter().map(|&v| dt.round(v)).collect();
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

    let expected = naive(
        &y,
        &z,
        &norm_weight,
        &weight,
        &scales_r,
        &biases_r,
        hv,
        dv,
        out_dim,
        group_size,
        eps,
    );
    let actual =
        run(&y, &z, &norm_weight, &weight, &scales, &biases, dt, hv, dv, out_dim, group_size, eps);

    assert_eq!(actual.len(), out_dim);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel <= tol,
        "dt={:?} hv={hv} dv={dv} out_dim={out_dim}: max rel = {max_rel:.3e} > {tol:.3e}",
        dt as u32
    );
}

// ── Smaller f32 shape: hv=4, dv=128, in_dim=512, out_dim=512 ───────────
#[test]
fn gated_rms_norm_qgemv_int4_fast_f32_small() { run_case(Dt::F32, 4, 128, 512, 64, 5e-3); }

// ── Qwen3.6-A3B production shape: hv=16, dv=128, in_dim=2048, out_dim=2048 ──
#[test]
fn gated_rms_norm_qgemv_int4_fast_f32_qwen36() { run_case(Dt::F32, 16, 128, 2048, 64, 5e-3); }

#[test]
fn gated_rms_norm_qgemv_int4_fast_f16_qwen36() { run_case(Dt::F16, 16, 128, 2048, 64, 3e-2); }

#[test]
fn gated_rms_norm_qgemv_int4_fast_bf16_qwen36() { run_case(Dt::Bf16, 16, 128, 2048, 64, 6e-2); }
