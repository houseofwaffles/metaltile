//! Correctness tests for the int8 matmul family:
//!   - `mt_qmm_int8_fast`      — BM=1 (one M-row per TG)
//!   - `mt_qmm_bm2_int8_fast`  — BM=2 (two M-rows per TG)
//!   - `mt_qmm_bm4_int8_fast`  — BM=4 (four M-rows per TG)
//!
//! Weight layout: `w[n, k/4]` u32 (4 int8 codes per word, LE byte order).
//! Scales/biases: `[n, k/group_size]`. X: `[m, k]`. Out: `[m, n]`.
//!
//! Dispatch grids:
//!   qmm:      `[n/8, m, 1]`,   tpg=64
//!   qmm_bm2:  `[n/8, m/2, 1]`, tpg=64
//!   qmm_bm4:  `[n/8, m/4, 1]`, tpg=64
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{mt_qmm_bm2_int8_fast, mt_qmm_bm4_int8_fast, mt_qmm_int8_fast};

// ── CPU oracle ───────────────────────────────────────────────────────────

/// Quantize `row` (f32, length k) into int8 per-group:
///   `q = round((v - min) / (max - min) * 255).clamp(0, 255)`.
/// Returns packed u32 (4 bytes/word, LE), scales, biases.
fn quantize_int8_per_group(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let k = row.len();
    let n_groups = k / group_size;
    let mut packed = vec![0u32; k / 4];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let g_off = g * group_size;
        let g_slice = &row[g_off..g_off + group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / 255.0 };
        let bias = mn;
        scales[g] = scale;
        biases[g] = bias;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - bias) / scale).round().clamp(0.0, 255.0) as u32;
            let abs_idx = g_off + i;
            let word = abs_idx / 4;
            let shift = (abs_idx % 4) * 8;
            packed[word] |= q << shift;
        }
    }
    (packed, scales, biases)
}

/// CPU reference for int8 QMM:
///   `out[m_row, n_col] = Σ_j ((q[n_col,j]*scale[n_col,g] + bias[n_col,g]) * x[m_row,j])`
#[allow(clippy::too_many_arguments)]
fn cpu_qmm_int8_reference(
    w: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    group_size: usize,
) -> Vec<f32> {
    let packs_per_row = k / 4;
    let mut out = vec![0.0_f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0_f32;
            for j in 0..k {
                let g = j / group_size;
                let scale = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let word = w[n_col * packs_per_row + j / 4];
                let shift = (j % 4) * 8;
                let q = ((word >> shift) & 0xFF) as f32;
                acc += (q * scale + bias) * x[m_row * k + j];
            }
            out[m_row * n + n_col] = acc;
        }
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

// ── Data generator ────────────────────────────────────────────────────────

/// Generate deterministic int8 weight matrix `[n, k]` plus X `[m, k]`.
/// All inputs are round-tripped through `dt` precision.
fn build_inputs(
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    dt: Dt,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let gs_per_row = k / group_size;
    let mut all_packed = Vec::with_capacity(n * k / 4);
    let mut all_scales = Vec::with_capacity(n * gs_per_row);
    let mut all_biases = Vec::with_capacity(n * gs_per_row);
    // Weight rows correspond to N-axis (output features).
    for ni in 0..n {
        let row: Vec<f32> = (0..k)
            .map(|j| {
                let v = (((ni * 11 + j * 7 + 1) % 29) as f32 - 14.0) * 0.04;
                dt.round(v)
            })
            .collect();
        let (pk, sc, bs) = quantize_int8_per_group(&row, group_size);
        all_packed.extend(pk);
        all_scales.extend(sc.iter().map(|&v| dt.round(v)));
        all_biases.extend(bs.iter().map(|&v| dt.round(v)));
    }
    let x: Vec<f32> =
        (0..m * k).map(|i| dt.round((((i * 13 + 3) % 23) as f32 - 11.0) * 0.05)).collect();
    (all_packed, all_scales, all_biases, x)
}

// ── Dispatch helpers ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_qmm_int8(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bpe: usize,
) -> Vec<u8> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(w));
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bpe]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_int8_fast::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    // Grid: (n/8 N-tiles, m M-rows, 1). 64 threads per TG.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[allow(clippy::too_many_arguments)]
fn run_qmm_bm2_int8(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bpe: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(2), "mt_qmm_bm2_int8_fast requires m % 2 == 0");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(w));
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bpe]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_bm2_int8_fast::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    // Grid: (n/8, m/2, 1).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m / 2, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[allow(clippy::too_many_arguments)]
fn run_qmm_bm4_int8(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bpe: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(4), "mt_qmm_bm4_int8_fast requires m % 4 == 0");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(w));
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bpe]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_bm4_int8_fast::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    // Grid: (n/8, m/4, 1).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m / 4, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

// ── mt_qmm_int8_fast tests ────────────────────────────────────────────────

#[test]
fn mt_qmm_int8_fast_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F32;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_int8_fast f32: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_int8_fast_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_int8(
        &ctx,
        DType::F16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_int8_fast f16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_int8_fast_matches_cpu_reference_bf16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::Bf16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_int8(
        &ctx,
        DType::BF16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.997, "mt_qmm_int8_fast bf16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_int8_fast_larger_shape_f16() {
    // m=16, n=32, k=256 — exercises multiple K-blocks and a larger grid.
    let _g = gpu_lock();
    let (m, n, k, gs) = (16usize, 32usize, 256usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_int8(
        &ctx,
        DType::F16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_int8_fast f16 larger: cosine = {cos:.6}");
}

// ── mt_qmm_bm2_int8_fast tests ────────────────────────────────────────────

#[test]
fn mt_qmm_bm2_int8_fast_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F32;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm2_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_bm2_int8_fast f32: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm2_int8_fast_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm2_int8(
        &ctx,
        DType::F16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_bm2_int8_fast f16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm2_int8_fast_matches_cpu_reference_bf16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::Bf16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm2_int8(
        &ctx,
        DType::BF16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.997, "mt_qmm_bm2_int8_fast bf16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm2_int8_fast_agrees_with_bm1_f32() {
    // Cross-check: bm2 should match bm1 at same shape (fp32, no rounding noise from dtype).
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F32;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);

    let ctx = Context::new().expect("Context::new");
    let out_bm1 = run_qmm_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let out_bm2 = run_qmm_bm2_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let a1 = unpack_bytes(&out_bm1, dt);
    let a2 = unpack_bytes(&out_bm2, dt);
    let cos = cosine_similarity(&a1, &a2);
    assert!(cos >= 0.9999, "bm2 vs bm1 diverge: cosine = {cos:.6}");
}

// ── mt_qmm_bm4_int8_fast tests ────────────────────────────────────────────

#[test]
fn mt_qmm_bm4_int8_fast_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F32;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm4_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_bm4_int8_fast f32: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm4_int8_fast_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm4_int8(
        &ctx,
        DType::F16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_bm4_int8_fast f16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm4_int8_fast_matches_cpu_reference_bf16() {
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::Bf16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm4_int8(
        &ctx,
        DType::BF16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.997, "mt_qmm_bm4_int8_fast bf16: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm4_int8_fast_agrees_with_bm1_f32() {
    // Cross-check: bm4 should match bm1 at same shape (fp32).
    let _g = gpu_lock();
    let (m, n, k, gs) = (8usize, 16usize, 128usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F32;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);

    let ctx = Context::new().expect("Context::new");
    let out_bm1 = run_qmm_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let out_bm4 = run_qmm_bm4_int8(
        &ctx,
        DType::F32,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let a1 = unpack_bytes(&out_bm1, dt);
    let a4 = unpack_bytes(&out_bm4, dt);
    let cos = cosine_similarity(&a1, &a4);
    assert!(cos >= 0.9999, "bm4 vs bm1 diverge: cosine = {cos:.6}");
}

#[test]
fn mt_qmm_bm4_int8_fast_larger_shape_f16() {
    // m=32, n=32, k=256 — exercises 8 BM=4 tiles in Y and 4 K-blocks.
    let _g = gpu_lock();
    let (m, n, k, gs) = (32usize, 32usize, 256usize, 64usize);
    let gs_per_row = k / gs;
    let dt = Dt::F16;
    let (w, sc, bs, x) = build_inputs(m, n, k, gs, dt);
    let expected = cpu_qmm_int8_reference(&w, &sc, &bs, &x, m, n, k, gs_per_row, gs);

    let ctx = Context::new().expect("Context::new");
    let out = run_qmm_bm4_int8(
        &ctx,
        DType::F16,
        &w,
        &pack_bytes(&sc, dt),
        &pack_bytes(&bs, dt),
        &pack_bytes(&x, dt),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = unpack_bytes(&out, dt);
    let cos = cosine_similarity(&expected, &actual);
    assert!(cos >= 0.999, "mt_qmm_bm4_int8_fast f16 m=32: cosine = {cos:.6}");
}
