//! GPU correctness for `mt_fp4_qmm_mma` (fp4 E2M1) and `mt_fp8_e4m3_qmm_mma`
//! (fp8 E4M3) — simdgroup-matrix MMA prefill kernels for floating-point
//! quantized weights.
//!
//! CPU oracle for fp4: decode each 4-bit nibble via the `two_m_int` trick
//! (E2M1 format, scale-only, group_size=32), compute `Out = X @ W^T`.
//!
//! CPU oracle for fp8 E4M3: decode each 8-bit code via biased-exponent
//! reconstruction (E4M3, bias=7, scale-only, group_size=32), compute
//! `Out = X @ W^T`.
//!
//! Constraints:
//!   m, n, k must be multiples of 32.
//!   group_size = 32 (one scale per BK=32 block per N-row), no bias.
//!   gs_per_row = k / group_size.
//!
//! Tolerances:
//!   f32:  cosine ≥ 0.999
//!   f16:  cosine ≥ 0.999
//!   bf16: cosine ≥ 0.997

#![cfg(target_os = "macos")]
#![allow(clippy::manual_is_multiple_of)]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized_mma::{mt_fp4_qmm_mma, mt_fp8_e4m3_qmm_mma};

// ── Numeric helpers ──────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

// ── fp4 E2M1 helpers ─────────────────────────────────────────────────────────

/// Decode one 4-bit nibble to f32 using the E2M1 two_m_int trick.
/// Mirrors the kernel's dequant body exactly.
fn fp4_decode(nibble: u32) -> f32 {
    let sign = 1.0f32 - 2.0f32 * ((nibble >> 3) & 1) as f32;
    let code3 = nibble & 7;
    let exp = code3 >> 1;
    let mantissa = code3 & 1;
    let two_m_int = if exp > 0 { (mantissa + 2) << (exp - 1) } else { mantissa };
    sign * two_m_int as f32 * 0.5
}

/// Pack fp4 codes row-by-row into u32 words (8 codes per word).
fn pack_fp4_row(codes: &[u32]) -> Vec<u32> {
    assert!(codes.len() % 8 == 0, "fp4 row length must be multiple of 8");
    codes
        .chunks_exact(8)
        .map(|ch| ch.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | ((c & 0xF) << (i * 4))))
        .collect()
}

/// Build deterministic fp4-quantized weight inputs.
/// Returns (packed_w [n, k/8], scales [n, gs_per_row], x [m, k], codes_flat [n*k]).
fn build_fp4_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    // Generate codes in 0..15 range (all valid fp4 nibbles).
    let codes_flat: Vec<u32> =
        (0..n * k).map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(12) & 0xF).collect();

    let packed: Vec<u32> = codes_flat.chunks_exact(k).flat_map(pack_fp4_row).collect();

    // Scales: positive, group-varying (no bias for fp4).
    let scales: Vec<f32> =
        (0..n * gs_per_row).map(|i| 0.5 + 0.1 * (i as f32 * 0.07).sin().abs()).collect();

    let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i as f32 * 0.017).sin()).collect();

    (packed, scales, x, codes_flat)
}

/// CPU oracle for fp4 MMA: `Out = X @ dequant(W)^T`, scale-only.
#[allow(clippy::too_many_arguments)]
fn cpu_fp4_qmm_reference(
    codes_flat: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    let group_size = k / gs_per_row;
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                let code = codes_flat[n_col * k + d];
                let g = d / group_size;
                let s = scales[n_col * gs_per_row + g];
                let wv = s * fp4_decode(code);
                acc += wv * x[m_row * k + d];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

// ── fp8 E4M3 helpers ─────────────────────────────────────────────────────────

/// Decode one 8-bit fp8 E4M3 code to f32.
/// Matches the kernel's decode path: bias=7, subnormal when e_raw=0.
fn fp8_e4m3_decode(code: u32) -> f32 {
    let sign = 1.0f32 - 2.0f32 * (code >> 7) as f32;
    let code7 = code & 0x7F;
    let e_raw = code7 >> 3;
    let m = code7 & 7;
    let mag = if e_raw > 0 {
        // Normal: 2^(e_raw-7) * (1 + m/8)
        let exp_f = e_raw as f32 - 7.0;
        exp_f.exp2() * (1.0 + m as f32 * 0.125)
    } else {
        // Subnormal: 2^(-6) * m/8
        (-6.0f32).exp2() * m as f32 * 0.125
    };
    sign * mag
}

/// Pack fp8 codes into u32 words (4 codes per word, LSB first).
fn pack_fp8_row(codes: &[u32]) -> Vec<u32> {
    assert!(codes.len() % 4 == 0, "fp8 row length must be multiple of 4");
    codes
        .chunks_exact(4)
        .map(|ch| ch.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | ((c & 0xFF) << (i * 8))))
        .collect()
}

/// Build deterministic fp8 E4M3-quantized weight inputs.
/// Returns (packed_w [n, k/4], scales [n, gs_per_row], x [m, k], codes_flat [n*k]).
fn build_fp8_e4m3_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    // Generate codes in 1..127 range (normal fp8 E4M3, positive, no NaN/inf).
    // E4M3 has NaN at 0x7F; avoid by capping at 0x7E.
    let codes_flat: Vec<u32> = (0..n * k)
        .map(|i| {
            let c = (i as u32).wrapping_mul(2654435761).wrapping_shr(11) & 0x7F;
            // Ensure normal range: e_raw ≥ 1 so m/8 is meaningful.
            let e = ((c >> 3) & 0xF).max(1);
            let m = c & 7;
            (e << 3) | m
        })
        .collect();

    let packed: Vec<u32> = codes_flat.chunks_exact(k).flat_map(pack_fp8_row).collect();

    // Scales: positive, group-varying.
    let scales: Vec<f32> =
        (0..n * gs_per_row).map(|i| 0.1 + 0.05 * (i as f32 * 0.11).sin().abs()).collect();

    let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.019).cos()).collect();

    (packed, scales, x, codes_flat)
}

/// CPU oracle for fp8 E4M3 MMA: `Out = X @ dequant(W)^T`, scale-only.
#[allow(clippy::too_many_arguments)]
fn cpu_fp8_e4m3_qmm_reference(
    codes_flat: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    let group_size = k / gs_per_row;
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                let code = codes_flat[n_col * k + d];
                let g = d / group_size;
                let s = scales[n_col * gs_per_row + g];
                let wv = s * fp8_e4m3_decode(code);
                acc += wv * x[m_row * k + d];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

// ── Dispatch helpers ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_fp4_mma(
    ctx: &Context,
    dt: Dt,
    packed: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), packed.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_fp4_qmm_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_fp4_qmm_mma");
    unpack_bytes(result.outputs.get("out").expect("`out` buffer"), dt)
}

#[allow(clippy::too_many_arguments)]
fn run_fp8_e4m3_mma(
    ctx: &Context,
    dt: Dt,
    packed: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), packed.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_fp8_e4m3_qmm_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_fp8_e4m3_qmm_mma");
    unpack_bytes(result.outputs.get("out").expect("`out` buffer"), dt)
}

// ── Generic runners ──────────────────────────────────────────────────────────

fn run_fp4_case(dt: Dt, m: usize, n: usize, k: usize, gs_per_row: usize, tol: f32, label: &str) {
    assert!(m % 32 == 0 && n % 32 == 0 && k % 32 == 0, "dims must be multiples of 32");
    // fp4: 8 codes/u32 → k must be multiple of 8.
    assert!(k % 8 == 0, "fp4: k must be multiple of 8");

    let (packed, scales_f32, x_f32, codes_flat) = build_fp4_inputs(m, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| dt.round(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| dt.round(v)).collect();

    let expected = cpu_fp4_qmm_reference(&codes_flat, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let actual = run_fp4_mma(&ctx, dt, &packed, &scales_f32, &x_f32, m, n, k, gs_per_row);

    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    assert!(actual.iter().any(|&v| v != 0.0), "{label}: all-zero output");
    let cos = cosine(&expected, &actual);
    eprintln!(
        "[{label}] cos={cos:.6}  exp[0..4]={:?} got[0..4]={:?}",
        &expected[..4.min(expected.len())],
        &actual[..4.min(actual.len())]
    );
    assert!(cos >= tol, "{label}: cosine {cos:.6} < {tol}");
}

fn run_fp8_e4m3_case(
    dt: Dt,
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    tol: f32,
    label: &str,
) {
    assert!(m % 32 == 0 && n % 32 == 0 && k % 32 == 0, "dims must be multiples of 32");
    // fp8: 4 codes/u32 → k must be multiple of 4.
    assert!(k % 4 == 0, "fp8: k must be multiple of 4");

    let (packed, scales_f32, x_f32, codes_flat) = build_fp8_e4m3_inputs(m, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| dt.round(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| dt.round(v)).collect();

    let expected = cpu_fp8_e4m3_qmm_reference(&codes_flat, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let actual = run_fp8_e4m3_mma(&ctx, dt, &packed, &scales_f32, &x_f32, m, n, k, gs_per_row);

    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    assert!(actual.iter().any(|&v| v != 0.0), "{label}: all-zero output");
    let cos = cosine(&expected, &actual);
    eprintln!(
        "[{label}] cos={cos:.6}  exp[0..4]={:?} got[0..4]={:?}",
        &expected[..4.min(expected.len())],
        &actual[..4.min(actual.len())]
    );
    assert!(cos >= tol, "{label}: cosine {cos:.6} < {tol}");
}

// ── mt_fp4_qmm_mma tests ─────────────────────────────────────────────────────

#[test]
fn fp4_qmm_mma_f32_small() {
    // group_size=32 → gs_per_row = k/32 = 1.
    run_fp4_case(Dt::F32, 32, 32, 32, 1, 0.999, "fp4_mma f32 small");
}

#[test]
fn fp4_qmm_mma_f32_multi_k() {
    // k=128, gs_per_row=4 (group_size=32).
    run_fp4_case(Dt::F32, 32, 32, 128, 4, 0.999, "fp4_mma f32 multi-k");
}

#[test]
fn fp4_qmm_mma_f32_multi_tile() {
    run_fp4_case(Dt::F32, 64, 64, 128, 4, 0.999, "fp4_mma f32 multi-tile");
}

#[test]
fn fp4_qmm_mma_f16_small() { run_fp4_case(Dt::F16, 32, 32, 128, 4, 0.999, "fp4_mma f16 small"); }

#[test]
fn fp4_qmm_mma_bf16_small() { run_fp4_case(Dt::Bf16, 32, 32, 128, 4, 0.997, "fp4_mma bf16 small"); }

// ── mt_fp8_e4m3_qmm_mma tests ────────────────────────────────────────────────

#[test]
fn fp8_e4m3_qmm_mma_f32_small() {
    // group_size=32 → gs_per_row = k/32 = 1.
    run_fp8_e4m3_case(Dt::F32, 32, 32, 32, 1, 0.999, "fp8_e4m3_mma f32 small");
}

#[test]
fn fp8_e4m3_qmm_mma_f32_multi_k() {
    // k=128, gs_per_row=4.
    run_fp8_e4m3_case(Dt::F32, 32, 32, 128, 4, 0.999, "fp8_e4m3_mma f32 multi-k");
}

#[test]
fn fp8_e4m3_qmm_mma_f32_multi_tile() {
    run_fp8_e4m3_case(Dt::F32, 64, 64, 128, 4, 0.999, "fp8_e4m3_mma f32 multi-tile");
}

#[test]
fn fp8_e4m3_qmm_mma_f16_small() {
    run_fp8_e4m3_case(Dt::F16, 32, 32, 128, 4, 0.999, "fp8_e4m3_mma f16 small");
}

#[test]
fn fp8_e4m3_qmm_mma_bf16_small() {
    run_fp8_e4m3_case(Dt::Bf16, 32, 32, 128, 4, 0.997, "fp8_e4m3_mma bf16 small");
}
