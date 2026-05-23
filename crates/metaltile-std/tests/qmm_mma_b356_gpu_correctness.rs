//! GPU correctness for `mt_qmm_mma_b{3,5,6}` — bit-stream MMA prefill
//! kernels for odd-bit-width quantized weights.
//!
//! CPU oracle: triple-loop with straddle-aware bit-stream extract (LSB-first
//! contiguous packing), matching the kernel's `qmm_mma_bitwidth!` macro.
//! `wv = q * scale + bias` where q is the code extracted from the
//! bit-stream, scale/bias looked up by group index.
//!
//! Constraints (inherited from `mt_qmm_mma`):
//!   m, n, k must be multiples of 32.
//!   group_size must divide k (group_size = k / gs_per_row).
//!   `k * bits` must be divisible by 32 so bit-stream rows are word-aligned.
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
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{mt_qmm_mma_b3, mt_qmm_mma_b5, mt_qmm_mma_b6};

// ── CPU oracle ───────────────────────────────────────────────────────────────

/// Pack one weight row into a contiguous LSB-first bit-stream.
/// Mirrors the `qmm_mma_bitwidth!` kernel's decode path.
fn pack_bitstream_row(codes: &[u32], bits: u32, max_code: u32) -> Vec<u32> {
    let n_words = codes.len() * bits as usize / 32;
    let mut words = vec![0u32; n_words];
    for (c, &code) in codes.iter().enumerate() {
        let masked = code & max_code;
        let bo = c * bits as usize;
        for bi in 0..bits as usize {
            if (masked >> bi) & 1 == 1 {
                let abs = bo + bi;
                words[abs / 32] |= 1u32 << (abs % 32);
            }
        }
    }
    words
}

/// Triple-loop CPU oracle for bit-stream quantized matmul.
/// W is [N, K] float codes (unpacked), packed as a contiguous bit-stream.
/// Out = X @ W^T (W row-major [N, K], X row-major [M, K]).
#[allow(clippy::too_many_arguments)]
fn cpu_qmm_bitstream_reference(
    codes_flat: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                let code = codes_flat[n_col * k + d] as f32;
                let g = d / group_size;
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let wv = code * s + bias;
                acc += wv * x[m_row * k + d];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

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

// ── Input builder ────────────────────────────────────────────────────────────

/// `(packed_words, scales, biases, x, codes_flat)` returned by `build_bitstream_inputs`.
type BitstreamInputs = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>);

/// Build deterministic bit-stream-quantized weight inputs.
/// Returns (packed_words, scales, biases, x, codes_flat).
fn build_bitstream_inputs(
    m: usize,
    n: usize,
    k: usize,
    bits: u32,
    gs_per_row: usize,
) -> BitstreamInputs {
    let max_code = (1u32 << bits) - 1;
    // Generate deterministic codes in [1, max_code] range.
    let codes_flat: Vec<u32> = (0..n * k)
        .map(|i| {
            let c = (i as u32).wrapping_mul(2654435761).wrapping_shr(13) & max_code;
            if c == 0 { 1 } else { c }
        })
        .collect();

    // Pack each N-row into a bit-stream.
    let packed: Vec<u32> = codes_flat
        .chunks_exact(k)
        .flat_map(|row| pack_bitstream_row(row, bits, max_code))
        .collect();

    // Scales: small positive, varying per group.
    let scales: Vec<f32> =
        (0..n * gs_per_row).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin().abs()).collect();
    // Biases: near-zero.
    let biases: Vec<f32> =
        (0..n * gs_per_row).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    // Activations: bounded, non-trivial.
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    (packed, scales, biases, x, codes_flat)
}

// ── Dispatch helper ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_bitstream_kernel(
    ctx: &Context,
    kernel_ir: fn(DType) -> metaltile_core::ir::Kernel,
    dt: Dt,
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    assert!(m % 32 == 0, "m must be multiple of 32");
    assert!(n % 32 == 0, "n must be multiple of 32");
    assert!(k % 32 == 0, "k must be multiple of 32");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), packed.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch bitstream qmm_mma");
    unpack_bytes(result.outputs.get("out").expect("`out` buffer"), dt)
}

// ── Generic test runner ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_case(
    bits: u32,
    kernel_ir: fn(DType) -> metaltile_core::ir::Kernel,
    dt: Dt,
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    tol: f32,
    label: &str,
) {
    let group_size = k / gs_per_row;
    let (packed, scales_f32, biases_f32, x_f32, codes_flat) =
        build_bitstream_inputs(m, n, k, bits, gs_per_row);

    // Round inputs through dtype so oracle matches GPU precision.
    let scales: Vec<f32> = scales_f32.iter().map(|&v| dt.round(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| dt.round(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| dt.round(v)).collect();

    let expected = cpu_qmm_bitstream_reference(
        &codes_flat,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        gs_per_row,
        group_size,
    );

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let actual = run_bitstream_kernel(
        &ctx,
        kernel_ir,
        dt,
        &packed,
        &scales_f32,
        &biases_f32,
        &x_f32,
        m,
        n,
        k,
        gs_per_row,
    );

    assert_eq!(actual.len(), expected.len(), "{label}: output length mismatch");
    assert!(actual.iter().any(|&v| v != 0.0), "{label}: all-zero output");

    let cos = cosine(&expected, &actual);
    eprintln!(
        "[{label}] cos={cos:.6}  exp[0..4]={:?} got[0..4]={:?}",
        &expected[..4.min(expected.len())],
        &actual[..4.min(actual.len())]
    );
    assert!(cos >= tol, "{label}: cosine {cos:.6} < {tol}");
}

// ── mt_qmm_mma_b3 tests ──────────────────────────────────────────────────────

#[test]
fn mt_qmm_mma_b3_f32_small() {
    // k=32*4=128; 128*3=384 bits = 12 words per row → word-aligned.
    // group_size=32 → gs_per_row=4.
    run_case(3, mt_qmm_mma_b3::kernel_ir_for, Dt::F32, 32, 32, 128, 4, 0.999, "b3 f32 small");
}

#[test]
fn mt_qmm_mma_b3_f32_multi_tile() {
    run_case(3, mt_qmm_mma_b3::kernel_ir_for, Dt::F32, 64, 64, 128, 4, 0.999, "b3 f32 multi");
}

#[test]
fn mt_qmm_mma_b3_f16_small() {
    run_case(3, mt_qmm_mma_b3::kernel_ir_for, Dt::F16, 32, 32, 128, 4, 0.999, "b3 f16 small");
}

#[test]
fn mt_qmm_mma_b3_bf16_small() {
    run_case(3, mt_qmm_mma_b3::kernel_ir_for, Dt::Bf16, 32, 32, 128, 4, 0.997, "b3 bf16 small");
}

// ── mt_qmm_mma_b5 tests ──────────────────────────────────────────────────────

#[test]
fn mt_qmm_mma_b5_f32_small() {
    // k=32*8=256; 256*5=1280 bits = 40 words per row → word-aligned.
    // group_size=32 → gs_per_row=8.
    run_case(5, mt_qmm_mma_b5::kernel_ir_for, Dt::F32, 32, 32, 256, 8, 0.999, "b5 f32 small");
}

#[test]
fn mt_qmm_mma_b5_f32_multi_tile() {
    run_case(5, mt_qmm_mma_b5::kernel_ir_for, Dt::F32, 64, 64, 256, 8, 0.999, "b5 f32 multi");
}

#[test]
fn mt_qmm_mma_b5_f16_small() {
    run_case(5, mt_qmm_mma_b5::kernel_ir_for, Dt::F16, 32, 32, 256, 8, 0.999, "b5 f16 small");
}

#[test]
fn mt_qmm_mma_b5_bf16_small() {
    run_case(5, mt_qmm_mma_b5::kernel_ir_for, Dt::Bf16, 32, 32, 256, 8, 0.997, "b5 bf16 small");
}

// ── mt_qmm_mma_b6 tests ──────────────────────────────────────────────────────

#[test]
fn mt_qmm_mma_b6_f32_small() {
    // k=32*2=64; 64*6=384 bits = 12 words per row → word-aligned.
    // group_size=32 → gs_per_row=2.
    run_case(6, mt_qmm_mma_b6::kernel_ir_for, Dt::F32, 32, 32, 64, 2, 0.999, "b6 f32 small");
}

#[test]
fn mt_qmm_mma_b6_f32_multi_tile() {
    run_case(6, mt_qmm_mma_b6::kernel_ir_for, Dt::F32, 64, 64, 64, 2, 0.999, "b6 f32 multi");
}

#[test]
fn mt_qmm_mma_b6_f16_small() {
    run_case(6, mt_qmm_mma_b6::kernel_ir_for, Dt::F16, 32, 32, 64, 2, 0.999, "b6 f16 small");
}

#[test]
fn mt_qmm_mma_b6_bf16_small() {
    run_case(6, mt_qmm_mma_b6::kernel_ir_for, Dt::Bf16, 32, 32, 64, 2, 0.997, "b6 bf16 small");
}
