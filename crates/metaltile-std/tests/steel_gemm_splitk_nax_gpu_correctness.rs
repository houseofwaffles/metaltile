//! GPU correctness oracle for `mt_steel_gemm_splitk_nax` — split-K GEMM
//! `C = A · B` backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Split-K GEMM is a two-pass dispatch:
//!   1. `mt_steel_gemm_splitk_nax_{f32,f16}` — each K-split computes a
//!      partial `[M, N]` product over its K-slice into an
//!      `[n_splits, M, N]` fp32 partials buffer.
//!   2. `mt_steel_gemm_splitk_accum_nax_{f32,f16}` — reduces the
//!      partials into the final `[M, N]` output (plain sum).
//!
//! This file dispatches both passes back-to-back and validates against
//! a naive triple-loop CPU oracle. Requires macOS 26+ / Metal 4 — the
//! kernel includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
//! and calls `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`.
//! On pre-Metal-4 toolchains the kernel emits a single-scalar fallback so
//! the metallib still links; this test then fails the correctness check,
//! which is the intended signal.
//!
//! Gated behind the `nax` Cargo feature.
//!
//! Run:
//!   cargo test --release -p metaltile-std --features nax \
//!     --test steel_gemm_splitk_nax_gpu_correctness -- --nocapture

#![cfg(all(target_os = "macos", feature = "nax"))]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::gemm::steel_gemm_splitk_nax::{
    mt_steel_gemm_splitk_accum_nax,
    mt_steel_gemm_splitk_nax,
};

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

/// Dispatch both split-K passes back-to-back. `n_splits * k_per_split`
/// must equal `k`; `k_per_split % 32 == 0`.
#[allow(clippy::too_many_arguments)]
fn run_splitk_nax(
    ctx: &Context,
    dtype: DType,
    a_bytes: &[u8],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    n_splits: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(32), "mt_steel_gemm_splitk_nax requires m % 32 == 0");
    assert!(n.is_multiple_of(32), "mt_steel_gemm_splitk_nax requires n % 32 == 0");
    assert!(k.is_multiple_of(32), "mt_steel_gemm_splitk_nax requires k % 32 == 0");
    let k_per_split = k / n_splits;
    assert!(k_per_split.is_multiple_of(32), "k_per_split must be a multiple of 32");
    assert_eq!(n_splits * k_per_split, k, "n_splits * k_per_split must equal k");

    // ── Pass 1 — split-K partial GEMM ──
    let mut p1_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p1_buffers.insert("a".into(), a_bytes.to_vec());
    p1_buffers.insert("b".into(), b_bytes.to_vec());
    // partials: [n_splits, M, N] fp32.
    p1_buffers.insert("partials".into(), vec![0u8; n_splits * m * n * 4]);
    p1_buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    p1_buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    p1_buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    p1_buffers.insert("k_per_split".into(), (k_per_split as u32).to_le_bytes().to_vec());

    let mut p1 = mt_steel_gemm_splitk_nax::kernel_ir_for(dtype);
    p1.mode = KernelMode::Reduction;
    let p1_res = ctx
        .dispatch_with_grid(&p1, &p1_buffers, &BTreeMap::new(), [n / 32, m / 32, n_splits], [
            128, 1, 1,
        ])
        .expect("dispatch mt_steel_gemm_splitk_nax");
    let partials = p1_res.outputs.get("partials").expect("`partials` buffer").clone();

    // ── Pass 2 — partial-sum reduction ──
    let mut p2_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p2_buffers.insert("partials".into(), partials);
    p2_buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    p2_buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    p2_buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    p2_buffers.insert("n_splits".into(), (n_splits as u32).to_le_bytes().to_vec());

    let mut p2 = mt_steel_gemm_splitk_accum_nax::kernel_ir_for(dtype);
    p2.mode = KernelMode::Reduction;
    let p2_res = ctx
        .dispatch_with_grid(&p2, &p2_buffers, &BTreeMap::new(), [m * n, 1, 1], [1, 1, 1])
        .expect("dispatch mt_steel_gemm_splitk_accum_nax");

    p2_res.outputs.get("out").expect("`out` buffer in dispatch result").clone()
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

/// Deterministic small-magnitude A / B inputs.
fn build_gemm_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| 0.01 + (i as f32 % 17.0) * 0.013).collect();
    let b: Vec<f32> = (0..k * n).map(|i| -0.02 + (i as f32 % 13.0) * 0.011).collect();
    (a, b)
}

// ── Shape 1 : 2-way split, single tile ─────────────────────────────────────

#[test]
fn mt_steel_gemm_splitk_nax_matches_cpu_reference_f32_2way() {
    let (m, n, k, n_splits) = (32usize, 32usize, 256usize, 2usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_splitk_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
        n_splits,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[f32 2-way k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 2-way)");
}

// ── Shape 2 : 3-way split (uneven last split), single tile ─────────────────

#[test]
fn mt_steel_gemm_splitk_nax_matches_cpu_reference_f32_3way() {
    // 3-way split of k=384 → k_per_split=128 each, even.
    let (m, n, k, n_splits) = (32usize, 32usize, 384usize, 3usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_splitk_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
        n_splits,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 3-way k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 3-way)");
}

// ── Shape 3 : multi-tile, 2-way split ──────────────────────────────────────

#[test]
fn mt_steel_gemm_splitk_nax_matches_cpu_reference_f32_multi_tile() {
    let (m, n, k, n_splits) = (64usize, 64usize, 256usize, 2usize);
    let (a, b) = build_gemm_inputs(m, n, k);
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_splitk_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&a),
        &f32_to_f32_bytes(&b),
        m,
        n,
        k,
        n_splits,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 multi-tile)");
}

// ── Shape 4 : fp16 multi-tile, 2-way split ─────────────────────────────────

#[test]
fn mt_steel_gemm_splitk_nax_matches_cpu_reference_f16_multi_tile() {
    let (m, n, k, n_splits) = (64usize, 64usize, 256usize, 2usize);
    let (a_f32, b_f32) = build_gemm_inputs(m, n, k);
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let a: Vec<f32> = a_f32.iter().map(|&v| round_f16(v)).collect();
    let b: Vec<f32> = b_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_splitk_nax(
        &ctx,
        DType::F16,
        &f32_to_f16_bytes(&a),
        &f32_to_f16_bytes(&b),
        m,
        n,
        k,
        n_splits,
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

// ── Shape 5 : bf16 multi-tile, 2-way split ─────────────────────────────────
//
// bf16 activations stage through `half` via the DSL `coop_stage(T)` form
// (Apple `matmul2d` mishandles `bfloat` cooperative tensors). The fp32
// partials buffer keeps cross-split sums full precision; bf16 only
// affects the operands. 7-bit mantissa → 0.997 cosine bar.

#[test]
fn mt_steel_gemm_splitk_nax_matches_cpu_reference_bf16_multi_tile() {
    let (m, n, k, n_splits) = (64usize, 64usize, 256usize, 2usize);
    let (a_f32, b_f32) = build_gemm_inputs(m, n, k);
    let round_bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
    let a: Vec<f32> = a_f32.iter().map(|&v| round_bf16(v)).collect();
    let b: Vec<f32> = b_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_gemm_reference(&a, &b, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_splitk_nax(
        &ctx,
        DType::BF16,
        &f32_to_bf16_bytes(&a),
        &f32_to_bf16_bytes(&b),
        m,
        n,
        k,
        n_splits,
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
