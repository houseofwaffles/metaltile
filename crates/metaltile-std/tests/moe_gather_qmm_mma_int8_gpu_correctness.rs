#![allow(clippy::manual_is_multiple_of)]

//! GPU correctness for `ffai::moe::mt_moe_gather_qmm_mma_int8`.
//!
//! Pack-aligned int8 MoE simdgroup-matrix BGEMM. Same tiled-MMA geometry as
//! `mt_moe_gather_qmm_mma_int4` (BM=BN=BK=32, 4 SGs, 2×2 warp grid,
//! per-TG expert sub-runs), but the W coop-dequant uses 4-byte packs instead
//! of 4-nibble packs.
//!
//! Validated against a naive CPU gather-matmul oracle with unsigned-byte codes
//! and affine dequant `wv = q * scale + bias`. Shape: n_experts=4, T=64,
//! N=64, K=64, group_size=32 — every dim a clean multiple of BM=BN=BK=32.
//! Cosine ≥ 0.999 vs the CPU oracle.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::mt_moe_gather_qmm_mma_int8;

// ── helpers ────────────────────────────────────────────────────────────────

/// Pack a row of `k_in` unsigned-byte codes into uint32s (4 bytes/u32, LSB-first).
fn pack_int8_row(codes: &[u32]) -> Vec<u32> {
    assert!(codes.len() % 4 == 0, "k_in must be divisible by 4 for int8 packing");
    codes
        .chunks_exact(4)
        .map(|chunk| {
            (chunk[0] & 0xff)
                | ((chunk[1] & 0xff) << 8)
                | ((chunk[2] & 0xff) << 16)
                | ((chunk[3] & 0xff) << 24)
        })
        .collect()
}

/// CPU oracle: per-row, look up expert via per-row `indices`, dequantize
/// weight row with `wv = q * scale + bias`, dot against activation row.
#[allow(clippy::too_many_arguments)]
fn cpu_oracle_int8(
    x: &[f32],
    codes: &[u32], // [n_experts * n_out * k_in] unpacked byte codes
    scales: &[f32],
    biases: &[f32],
    indices: &[u32], // per-row expert id
    t_rows: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
) -> Vec<f32> {
    let groups = k_in / group_size;
    let mut out = vec![0.0f32; t_rows * n_out];
    for t in 0..t_rows {
        let e = indices[t] as usize;
        for n in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                let code = codes[(e * n_out + n) * k_in + k] as f32;
                let g = (e * n_out + n) * groups + k / group_size;
                acc += (scales[g] * code + biases[g]) * x[t * k_in + k];
            }
            out[t * n_out + n] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

// ── dispatch helper ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_int8_mma(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    indices: &[u32],
    t_rows: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
    dt: Dt,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], dt));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new");
    let mut k = mt_moe_gather_qmm_mma_int8::kernel_ir_for(dt.to_dtype());
    k.mode = KernelMode::Reduction;
    // Grid: [N/BN=32, ceil(T/BM=32), 1], TG: [128, 1, 1] (4 SGs).
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(32), 1], [
            128, 1, 1,
        ])
        .expect("dispatch");
    unpack_bytes(r.outputs.get("out").expect("out"), dt)
}

// ── test shapes ────────────────────────────────────────────────────────────

/// Test data factory: n_experts=4, T=64, N=64, K=64, group_size=32.
/// All dims are clean multiples of BM=BN=BK=32.
#[allow(clippy::type_complexity)]
fn make_test_data(dt: Dt) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let n_experts = 4usize;
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;

    // Sorted per-row indices: rows 0..16 → e0, 16..32 → e1, etc.
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total = n_experts * n_out * k_in;
    // Generate codes in 0..255 (unsigned byte range).
    let codes: Vec<u32> = (0..total).map(|i| (i as u32).wrapping_mul(2654435761) & 0xff).collect();
    let weight_packed: Vec<u32> = codes.chunks_exact(k_in).flat_map(pack_int8_row).collect();

    let groups_total = n_experts * n_out * (k_in / group_size);
    // Small scales/biases; round through dtype so oracle matches kernel loads.
    let scales: Vec<f32> =
        (0..groups_total).map(|i| dt.round(0.002 + 0.0005 * (i as f32 * 0.03).sin())).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| dt.round(-0.05 + 0.01 * (i as f32 * 0.07).cos())).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|i| dt.round(0.05 * (i as f32 * 0.013).sin())).collect();

    (codes, weight_packed, scales, biases, x, indices)
}

// ── f32 ────────────────────────────────────────────────────────────────────

#[test]
fn moe_gather_qmm_mma_int8_matches_cpu_oracle_f32() {
    let _g = gpu_lock();
    let dt = Dt::F32;
    let (codes, weight_packed, scales, biases, x, indices) = make_test_data(dt);
    let n_experts = 4usize;
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;

    let expected =
        cpu_oracle_int8(&x, &codes, &scales, &biases, &indices, t_rows, n_out, k_in, group_size);
    let actual = run_int8_mma(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &indices,
        t_rows,
        n_out,
        k_in,
        group_size,
        dt,
    );

    let cos = cosine(&expected, &actual);
    eprintln!(
        "[int8 MMA f32] cos={cos:.6}  exp[0..4]={:?} got[0..4]={:?}",
        &expected[..4],
        &actual[..4]
    );
    assert!(actual.iter().any(|&v| v != 0.0), "all-zero output (kernel body not reached?)");
    assert!(cos >= 0.999, "int8 MMA f32 vs CPU oracle cosine = {cos:.6} (want ≥ 0.999)");
    let _ = n_experts;
}

// ── f16 ────────────────────────────────────────────────────────────────────

#[test]
fn moe_gather_qmm_mma_int8_matches_cpu_oracle_f16() {
    let _g = gpu_lock();
    let dt = Dt::F16;
    let (codes, weight_packed, scales, biases, x, indices) = make_test_data(dt);
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;

    let expected =
        cpu_oracle_int8(&x, &codes, &scales, &biases, &indices, t_rows, n_out, k_in, group_size);
    let actual = run_int8_mma(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &indices,
        t_rows,
        n_out,
        k_in,
        group_size,
        dt,
    );

    let cos = cosine(&expected, &actual);
    eprintln!("[int8 MMA f16] cos={cos:.6}");
    assert!(cos >= 0.999, "int8 MMA f16 vs CPU oracle cosine = {cos:.6} (want ≥ 0.999)");
}

// ── bf16 ───────────────────────────────────────────────────────────────────

#[test]
fn moe_gather_qmm_mma_int8_matches_cpu_oracle_bf16() {
    let _g = gpu_lock();
    let dt = Dt::Bf16;
    let (codes, weight_packed, scales, biases, x, indices) = make_test_data(dt);
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;

    let expected =
        cpu_oracle_int8(&x, &codes, &scales, &biases, &indices, t_rows, n_out, k_in, group_size);
    let actual = run_int8_mma(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &indices,
        t_rows,
        n_out,
        k_in,
        group_size,
        dt,
    );

    let cos = cosine(&expected, &actual);
    eprintln!("[int8 MMA bf16] cos={cos:.6}");
    // bf16 has 7-bit mantissa; looser bar than f32/f16.
    assert!(cos >= 0.997, "int8 MMA bf16 vs CPU oracle cosine = {cos:.6} (want ≥ 0.997)");
}
