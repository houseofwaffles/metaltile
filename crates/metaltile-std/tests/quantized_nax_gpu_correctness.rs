//! GPU correctness oracle for `mt_qmm_nax` — the production int4
//! quantized matmul backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Dispatches `mt_qmm_nax_{f32,f16}` over a small set of shapes
//! (single 32×32 tile + multi-tile / multi-K-block) and validates against
//! the same triple-loop CPU oracle used by `qmm_gpu_correctness.rs` for
//! `mt_qmm_mma`. Requires macOS 26+ / Metal 4 — the kernel includes
//! `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls
//! `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`. The
//! kernel emits a pre-Metal-4 fallback that writes a single scalar so
//! the metallib still links on older toolchains; on such toolchains this
//! test fails the correctness check, which is the intended signal.
//!
//!
//! Run:
//!   cargo test --release -p metaltile-std --test quantized_nax_gpu_correctness -- --nocapture
//!
//! This file is correctness only — no bench harness wired in.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized_nax::mt_qmm_nax;

/// Triple-loop CPU oracle — bit-identical algorithm to `cpu_qmm_reference`
/// in `qmm_gpu_correctness.rs`. Replicated here to keep the test file
/// self-contained (integration tests can't share helpers across files
/// without a `mod common`).
#[allow(clippy::too_many_arguments)]
fn cpu_qmm_reference(
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
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gs_per_row {
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let mut q_dot = 0.0f32;
                let mut x_sum = 0.0f32;
                for p in 0..8usize {
                    let packed = w[n_col * k / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let q = ((packed >> (bit * 4)) & 0xF) as f32;
                        let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                        q_dot += q * xv;
                        x_sum += xv;
                    }
                }
                acc += s * q_dot + bias * x_sum;
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_qmm_mma_mpp(
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
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(32), "mt_qmm_nax requires m %% 32 == 0 (BM=32 tile)");
    assert!(n.is_multiple_of(32), "mt_qmm_nax requires n %% 32 == 0 (BN=32 tile)");
    assert!(k.is_multiple_of(32), "mt_qmm_nax requires k %% 32 == 0 (BK=32 step)");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_nax::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_qmm_nax");

    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

/// Pack a fp32 vector → fp16 bytes (little-endian).
fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

/// Pack a fp32 vector → fp32 bytes (little-endian).
fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }

/// Cosine similarity between two equal-length fp32 vectors.
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

/// Deterministic q4 weights — same per-pack pattern as `mt_qmm` /
/// `mt_qmm_mma` correctness tests so we exercise an identical bit layout.
fn build_quant_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    (w, scales, biases, x)
}

// ── Shape 1 : smallest valid tile (1 TG, 2 K-blocks) ───────────────────────

#[test]
fn mt_qmm_nax_matches_cpu_reference_f32_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmm_mma_mpp(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&biases),
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
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    println!(
        "[f32 small] cos={cos:.6} max|Δ|={max_diff:.3e} at idx {max_at} (expected {:.4}, got {:.4})",
        expected[max_at], actual[max_at]
    );
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 small)");
}

// ── Shape 2 : multi-K-block fp32 (still single TG in M/N) ─────────────────

#[test]
fn mt_qmm_nax_matches_cpu_reference_f32_multi_k() {
    let m = 32usize;
    let n = 32usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmm_mma_mpp(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&biases),
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
fn mt_qmm_nax_matches_cpu_reference_f32_multi_tile() {
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_quant_inputs(m, n, k, gs_per_row);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmm_mma_mpp(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&biases),
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

// ── Shape 4 : fp16 small ───────────────────────────────────────────────────

#[test]
fn mt_qmm_nax_matches_cpu_reference_f16_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    // For f16, round each input through fp16 BEFORE the CPU oracle so the
    // reference matches the kernel's load-cast quantisation.
    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();

    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmm_mma_mpp(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
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
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    let mut max_rel = 0.0f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    println!(
        "[f16 small] cos={cos:.6} max_rel={max_rel:.3e} at idx {max_at} (expected {:.4}, got {:.4})",
        expected[max_at], actual[max_at]
    );
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 small)");
}

// ── Shape 5 : fp16 multi-tile ──────────────────────────────────────────────

#[test]
fn mt_qmm_nax_matches_cpu_reference_f16_multi_tile() {
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();

    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_qmm_mma_mpp(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
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
