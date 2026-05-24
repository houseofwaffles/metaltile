//! GPU correctness oracle for `mt_fp_qmm_nax` — fp4 (E2M1) quantized
//! matmul backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Dispatches `mt_fp_qmm_nax_{f32,f16}` over a small set of shapes
//! (single 32×32 tile + multi-tile / multi-K-block) and validates
//! against a triple-loop CPU oracle that dequantizes the fp4 weights
//! with the same E2M1 codebook the kernel uses. Requires macOS 26+ /
//! Metal 4 — the kernel includes
//! `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls
//! `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`. On
//! pre-Metal-4 toolchains the kernel emits a single-scalar fallback so
//! the metallib still links; this test then fails the correctness
//! check, which is the intended signal.
//!
//!
//! Run:
//!   cargo test --release -p metaltile-std --test fp_quantized_nax_gpu_correctness -- --nocapture

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized_nax::mt_fp_qmm_nax;

/// fp4 quantization group size — must match `GROUP_SIZE` in the kernel.
const GROUP_SIZE: usize = 32;

/// fp4 E2M1 magnitude codebook — the nvfp4 levels (MLX `fp4.h`).
const FP4_LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Dequantize one 4-bit fp4 E2M1 code: `[sign : 1][magnitude : 3]`.
fn fp4_dequant(code: u32) -> f32 {
    let mag = FP4_LUT[(code & 7) as usize];
    if code & 8 != 0 { -mag } else { mag }
}

/// Triple-loop CPU oracle — fp4 quantized matmul `C = X · dequant(W)ᵀ`.
/// `w` is `[N, K/8]` packed (8 fp4 codes per `u32`); `scales` is
/// `[N, K/GROUP_SIZE]` row-major. Scale-only (no bias).
fn cpu_fp_qmm_reference(
    w: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let pack = w[n_col * (k / 8) + kk / 8];
                let code = (pack >> ((kk % 8) as u32 * 4)) & 15;
                let g = kk / GROUP_SIZE;
                let s = scales[n_col * gs_per_row + g];
                let wv = s * fp4_dequant(code);
                acc += x[m_row * k + kk] * wv;
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_fp_qmm_nax(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(32), "mt_fp_qmm_nax requires m % 32 == 0");
    assert!(n.is_multiple_of(32), "mt_fp_qmm_nax requires n % 32 == 0");
    assert!(k.is_multiple_of(32), "mt_fp_qmm_nax requires k % 32 == 0");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_fp_qmm_nax::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_fp_qmm_nax");

    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }

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

/// Deterministic fp4-packed weights + per-group scales + X inputs.
fn build_fp_quant_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    // 8 fp4 codes per u32; each code is a deterministic 4-bit value.
    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for nibble in 0..8u32 {
                v |= ((i as u32 + nibble) & 0xF) << (nibble * 4);
            }
            v
        })
        .collect();
    // Small scales keep the f16 path well inside dynamic range.
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.05 + (i as f32) * 0.001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 + (i as f32 % 19.0) * 0.011).collect();
    (w, scales, x)
}

// ── Shape 1 : smallest valid tile (1 TG, 1 K-block / 1 group) ──────────────

#[test]
fn mt_fp_qmm_nax_matches_cpu_reference_f32_small() {
    let (m, n, k) = (32usize, 32usize, 32usize);
    let gs_per_row = k / GROUP_SIZE;
    let (w, scales, x) = build_fp_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_fp_qmm_reference(&w, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_fp_qmm_nax(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[f32 small] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 small)");
}

// ── Shape 2 : multi-K-block fp32 (multiple groups) ─────────────────────────

#[test]
fn mt_fp_qmm_nax_matches_cpu_reference_f32_multi_k() {
    let (m, n, k) = (32usize, 32usize, 256usize);
    let gs_per_row = k / GROUP_SIZE;
    let (w, scales, x) = build_fp_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_fp_qmm_reference(&w, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_fp_qmm_nax(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-k k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 multi-k)");
}

// ── Shape 3 : multi-tile fp32 (M=64, N=64) ─────────────────────────────────

#[test]
fn mt_fp_qmm_nax_matches_cpu_reference_f32_multi_tile() {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let gs_per_row = k / GROUP_SIZE;
    let (w, scales, x) = build_fp_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_fp_qmm_reference(&w, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_fp_qmm_nax(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 multi-tile)");
}

// ── Shape 4 : fp16 multi-tile ──────────────────────────────────────────────

#[test]
fn mt_fp_qmm_nax_matches_cpu_reference_f16_multi_tile() {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let gs_per_row = k / GROUP_SIZE;
    let (w, scales_f32, x_f32) = build_fp_quant_inputs(m, n, k, gs_per_row);
    // Round inputs through fp16 BEFORE the oracle so the reference
    // matches the kernel's load-cast quantization.
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_fp_qmm_reference(&w, &scales, &x, m, n, k, gs_per_row);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_fp_qmm_nax(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let cos = cosine(&expected, &actual);
    println!("[f16 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 multi-tile)");
}
