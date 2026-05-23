//! GPU correctness oracle for `mt_steel_gemm_fused_nax` — plain fused
//! GEMM `C = A · B` backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Dispatches `mt_steel_gemm_fused_nax_{f32,f16}` over a small set of
//! shapes (single 32×32 tile + multi-tile / multi-K-block) and validates
//! against a naive triple-loop CPU oracle. Requires macOS 26+ / Metal 4 —
//! the kernel includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
//! and calls `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`.
//! On pre-Metal-4 toolchains the kernel emits a single-scalar fallback so
//! the metallib still links; this test then fails the correctness check,
//! which is the intended signal.
//!
//! Gated behind the `nax` Cargo feature.
//!
//! Run:
//!   cargo test --release -p metaltile-std --features nax \
//!     --test steel_gemm_fused_nax_gpu_correctness -- --nocapture

#![cfg(all(target_os = "macos", feature = "nax"))]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::gemm::steel_gemm_fused_nax::mt_steel_gemm_fused_nax;

/// Naive triple-loop GEMM oracle — `A[M,K] · B[K,N] = C[M,N]`, all
/// row-major. fp32 accumulation, matching the kernel's `AccumType=float`.
fn cpu_gemm_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[m_row * k + kk] * b[kk * n + n_col];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_fused_nax(
    ctx: &Context,
    dtype: DType,
    a_bytes: &[u8],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(32), "mt_steel_gemm_fused_nax requires m % 32 == 0");
    assert!(n.is_multiple_of(32), "mt_steel_gemm_fused_nax requires n % 32 == 0");
    assert!(k.is_multiple_of(32), "mt_steel_gemm_fused_nax requires k % 32 == 0");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), a_bytes.to_vec());
    buffers.insert("b".into(), b_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let mut kernel = mt_steel_gemm_fused_nax::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_steel_gemm_fused_nax");

    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect()
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

/// Deterministic small-magnitude A / B inputs — keep values small so the
/// f16 path stays well inside dynamic range.
fn build_gemm_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| 0.01 + (i as f32 % 17.0) * 0.013).collect();
    let b: Vec<f32> = (0..k * n).map(|i| -0.02 + (i as f32 % 13.0) * 0.011).collect();
    (a, b)
}

// ── Shape 1 : smallest valid tile (1 TG, 1 K-block) ────────────────────────

#[test]
fn mt_steel_gemm_fused_nax_matches_cpu_reference_f32_small() {
    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gemm_fused_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[f32 small] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 small)");
}

// ── Shape 2 : multi-K-block fp32 (single TG in M/N) ────────────────────────

#[test]
fn mt_steel_gemm_fused_nax_matches_cpu_reference_f32_multi_k() {
    let (m, n, k) = (32usize, 32usize, 256usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gemm_fused_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
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
fn mt_steel_gemm_fused_nax_matches_cpu_reference_f32_multi_tile() {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gemm_fused_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
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
fn mt_steel_gemm_fused_nax_matches_cpu_reference_f16_multi_tile() {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let (a_f32, b_f32) = build_gemm_inputs(m, n, k);
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let a: Vec<f32> = a_f32.iter().map(|&v| round_f16(v)).collect();
    let b: Vec<f32> = b_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gemm_fused_nax(
        &ctx,
        DType::F16,
        &f32_to_f16_bytes(&a),
        &f32_to_f16_bytes(&b),
        m,
        n,
        k,
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

// ── Shape 5 : bf16 multi-tile ──────────────────────────────────────────────
//
// bf16 activations stage through `half` via the DSL `coop_stage(T)` form
// (Apple `matmul2d` mishandles `bfloat` cooperative tensors). bf16 has
// only 7 mantissa bits, so the cosine bar is slightly relaxed (≥ 0.997).

#[test]
fn mt_steel_gemm_fused_nax_matches_cpu_reference_bf16_multi_tile() {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let (a_f32, b_f32) = build_gemm_inputs(m, n, k);
    let round_bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
    let a: Vec<f32> = a_f32.iter().map(|&v| round_bf16(v)).collect();
    let b: Vec<f32> = b_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gemm_fused_nax(
        &ctx,
        DType::BF16,
        &f32_to_bf16_bytes(&a),
        &f32_to_bf16_bytes(&b),
        m,
        n,
        k,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let cos = cosine(&expected, &actual);
    println!("[bf16 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.997, "cosine {cos:.6} < 0.997 (bf16 multi-tile)");
}
