//! Correctness tests for `mt_qmv_int8_fast` — 8-row-per-TG int8 decode GEMV.
//!
//! Weight layout: `w[m, k/4]` u32 (4 int8 codes per word, LE byte order).
//! Scales/biases: `[m, k/group_size]`. X: `[k]`. Out: `[m]`.
//!
//! Dispatch: grid = `[m/8, 1, 1]`, tpg = `[64, 1, 1]`.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::mt_qmv_int8_fast;

// ── CPU oracle ───────────────────────────────────────────────────────────

/// Quantize `row` (f32) into int8 per-group using a symmetric affine
/// quantizer: `q = round((v - min) / range * 255).clamp(0, 255)`.
/// Returns packed u32 (4 bytes/word, LE), scales, biases.
fn quantize_int8_per_group(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let k = row.len();
    let n_groups = k / group_size;
    // 4 int8 codes per u32 word.
    let mut packed = vec![0u32; k / 4];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let g_off = g * group_size;
        let g_slice = &row[g_off..g_off + group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Avoid divide-by-zero when all values are equal.
        let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / 255.0 };
        let bias = mn;
        scales[g] = scale;
        biases[g] = bias;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - bias) / scale).round().clamp(0.0, 255.0) as u32;
            let abs_idx = g_off + i;
            let word = abs_idx / 4;
            // LE byte order: element 0 → bits [7:0], element 1 → bits [15:8], etc.
            let shift = (abs_idx % 4) * 8;
            packed[word] |= q << shift;
        }
    }
    (packed, scales, biases)
}

/// CPU reference for int8 GEMV:
///   `out[i] = Σ_j ((q[i,j] * scale[i,g] + bias[i,g]) * x[j])`
/// where `g = j / group_size`.
fn cpu_qmv_int8_reference(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let n_groups = k / group_size;
    let packs_per_row = k / 4;
    let mut out = vec![0.0_f32; m];
    for i in 0..m {
        let mut acc = 0.0_f32;
        for j in 0..k {
            let g = j / group_size;
            let scale = scales[i * n_groups + g];
            let bias = biases[i * n_groups + g];
            let word = packed[i * packs_per_row + j / 4];
            let shift = (j % 4) * 8;
            let q = ((word >> shift) & 0xFF) as f32;
            acc += (q * scale + bias) * x[j];
        }
        out[i] = acc;
    }
    out
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 1.0 } else { dot / (na * nb) }
}

// ── Dispatch helper ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_qmv_int8_fast(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    k: usize,
    gs_per_row: usize,
    out_bpe: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(8), "mt_qmv_int8_fast requires m % 8 == 0");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(w));
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * out_bpe]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmv_int8_fast::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    // Grid: m/8 TGs (8-row tile per TG), 64 threads each (2 SG × 32 lanes).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m / 8, 1, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

// ── Data generator ────────────────────────────────────────────────────────

/// Build packed weights, scales, biases for an `m × k` weight matrix,
/// round-tripped through `dt` precision for scales/biases/x.
fn build_int8_qmv_inputs(
    m: usize,
    k: usize,
    group_size: usize,
    dt: Dt,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_groups = k / group_size;
    let mut all_packed: Vec<u32> = Vec::with_capacity(m * k / 4);
    let mut all_scales: Vec<f32> = Vec::with_capacity(m * n_groups);
    let mut all_biases: Vec<f32> = Vec::with_capacity(m * n_groups);
    for i in 0..m {
        // Pseudo-random row values in [-1.0, 1.0] range.
        let row: Vec<f32> = (0..k)
            .map(|j| {
                let v = (((i * 7 + j * 13 + 3) % 37) as f32 - 18.0) * 0.05;
                dt.round(v)
            })
            .collect();
        let (pk, sc, bs) = quantize_int8_per_group(&row, group_size);
        all_packed.extend(pk);
        all_scales.extend(sc.iter().map(|&v| dt.round(v)));
        all_biases.extend(bs.iter().map(|&v| dt.round(v)));
    }
    let x: Vec<f32> = (0..k).map(|j| dt.round((((j * 11 + 5) % 19) as f32 - 9.0) * 0.1)).collect();
    (all_packed, all_scales, all_biases, x)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn mt_qmv_int8_fast_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    let m = 32usize;
    let k = 256usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let dt = Dt::F32;

    let (packed, scales, biases, x) = build_int8_qmv_inputs(m, k, group_size, dt);
    let expected = cpu_qmv_int8_reference(&packed, &scales, &biases, &x, m, k, group_size);

    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmv_int8_fast(
        &ctx,
        DType::F32,
        &packed,
        &pack_bytes(&scales, dt),
        &pack_bytes(&biases, dt),
        &pack_bytes(&x, dt),
        m,
        k,
        gs_per_row,
        4,
    );
    let actual = unpack_bytes(&out_bytes, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmv_int8_fast f32: cosine = {cos:.6} (need >= 0.999)",);
}

#[test]
fn mt_qmv_int8_fast_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    let m = 32usize;
    let k = 256usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let dt = Dt::F16;

    let (packed, scales, biases, x) = build_int8_qmv_inputs(m, k, group_size, dt);
    let expected = cpu_qmv_int8_reference(&packed, &scales, &biases, &x, m, k, group_size);

    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmv_int8_fast(
        &ctx,
        DType::F16,
        &packed,
        &pack_bytes(&scales, dt),
        &pack_bytes(&biases, dt),
        &pack_bytes(&x, dt),
        m,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out_bytes, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmv_int8_fast f16: cosine = {cos:.6} (need >= 0.999)",);
}

#[test]
fn mt_qmv_int8_fast_matches_cpu_reference_bf16() {
    let _g = gpu_lock();
    let m = 32usize;
    let k = 256usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let dt = Dt::Bf16;

    let (packed, scales, biases, x) = build_int8_qmv_inputs(m, k, group_size, dt);
    let expected = cpu_qmv_int8_reference(&packed, &scales, &biases, &x, m, k, group_size);

    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmv_int8_fast(
        &ctx,
        DType::BF16,
        &packed,
        &pack_bytes(&scales, dt),
        &pack_bytes(&biases, dt),
        &pack_bytes(&x, dt),
        m,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out_bytes, dt);
    let cos = cosine_similarity(&expected, &actual);
    // bf16 has 7-bit mantissa: allow slightly looser threshold.
    assert!(cos >= 0.997, "mt_qmv_int8_fast bf16: cosine = {cos:.6} (need >= 0.997)",);
}

#[test]
fn mt_qmv_int8_fast_larger_k_f16() {
    // k=512, m=16 — two K-blocks to exercise the loop boundary.
    let _g = gpu_lock();
    let m = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let dt = Dt::F16;

    let (packed, scales, biases, x) = build_int8_qmv_inputs(m, k, group_size, dt);
    let expected = cpu_qmv_int8_reference(&packed, &scales, &biases, &x, m, k, group_size);

    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmv_int8_fast(
        &ctx,
        DType::F16,
        &packed,
        &pack_bytes(&scales, dt),
        &pack_bytes(&biases, dt),
        &pack_bytes(&x, dt),
        m,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out_bytes, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmv_int8_fast f16 k=512: cosine = {cos:.6} (need >= 0.999)",);
}
