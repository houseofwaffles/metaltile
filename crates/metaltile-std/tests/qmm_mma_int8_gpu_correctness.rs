//! GPU correctness tests for `mt_qmm_mma_int8` and `mt_qmm_mma_m16_int8` —
//! simdgroup-matrix MMA prefill kernels for int8-quantized weights.
//!
//! CPU oracle: triple-loop with `wv = q * scale + bias` where q is the 8-bit
//! code extracted from the packed u32 and scale/bias are looked up by
//! group index `(n_col, byte_offset / group_size)`.
//!
//! Tolerances:
//!   f32:  cosine ≥ 0.999
//!   f16:  cosine ≥ 0.999
//!   bf16: cosine ≥ 0.997
//!
//! Run:
//!   cargo test --release -p metaltile-std \
//!     --test qmm_mma_int8_gpu_correctness -- --nocapture

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{mt_qmm_mma_int8, mt_qmm_mma_m16_int8};

// ── CPU oracle ──────────────────────────────────────────────────────────────

/// Triple-loop CPU oracle for int8-quantized matmul (W is row-major [N × K],
/// packed as 4 bytes per u32; group_size=64). Out = X @ W^T.
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
    // packs_per_row = k / 4 (4 bytes per u32)
    let packs_per_row = k / 4;
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            // Iterate over all K positions
            for d in 0..k {
                let g = d / group_size;
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];

                // Pack index and byte slot inside the pack
                let pack_idx = d / 4;
                let byte_slot = d % 4;
                let word = w[n_col * packs_per_row + pack_idx];
                let q = ((word >> (byte_slot * 8)) & 0xFF) as f32;

                let wv = q * s + bias;
                acc += wv * x[m_row * k + d];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

// ── Numeric helpers ──────────────────────────────────────────────────────────

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

fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn bytes_to_f16_as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn bytes_to_bf16_as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

// ── Weight builder ───────────────────────────────────────────────────────────

/// Build deterministic int8-quantized weight inputs.
/// Weights packed as 4 bytes per u32: w[n_col * packs_per_row + pack_idx].
fn build_int8_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let packs_per_row = k / 4;
    // Each pack has 4 bytes (8-bit codes 0–255).
    let w: Vec<u32> = (0..n * packs_per_row)
        .map(|i| {
            let mut v = 0u32;
            for byte_idx in 0..4u32 {
                // Deterministic but non-trivial; keep codes in 1–200 range
                // to avoid zero-scale degenerate cases.
                let code = ((i as u32 * 7 + byte_idx * 13 + 1) % 200 + 1) & 0xFF;
                v |= code << (byte_idx * 8);
            }
            v
        })
        .collect();
    // Scales: small positive, varying per group
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.01 + (i as f32 % 64.0) * 0.0005).collect();
    // Biases: near-zero, slowly varying
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32 % 32.0) * 0.0001).collect();
    // Activations: positive, bounded
    let x: Vec<f32> = (0..m * k).map(|i| 0.5 + (i as f32 % 64.0) * 0.01).collect();
    (w, scales, biases, x)
}

// ── Dispatch helpers ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_mt_qmm_mma_int8(
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
    assert!(m.is_multiple_of(32), "mt_qmm_mma_int8: m must be multiple of 32");
    assert!(n.is_multiple_of(32), "mt_qmm_mma_int8: n must be multiple of 32");
    assert!(k.is_multiple_of(32), "mt_qmm_mma_int8: k must be multiple of 32");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_mma_int8::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_qmm_mma_int8");
    result.outputs.get("out").expect("`out` buffer").clone()
}

#[allow(clippy::too_many_arguments)]
fn run_mt_qmm_mma_m16_int8(
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
    assert!(m.is_multiple_of(16), "mt_qmm_mma_m16_int8: m must be multiple of 16");
    assert!(n.is_multiple_of(32), "mt_qmm_mma_m16_int8: n must be multiple of 32");
    assert!(k.is_multiple_of(32), "mt_qmm_mma_m16_int8: k must be multiple of 32");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_mma_m16_int8::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 16, 1], [64, 1, 1])
        .expect("dispatch mt_qmm_mma_m16_int8");
    result.outputs.get("out").expect("`out` buffer").clone()
}

// ── mt_qmm_mma_int8 tests ────────────────────────────────────────────────────

#[test]
fn mt_qmm_mma_int8_f32_small() {
    // Smallest valid tile: m=32 (1 TG in M), n=32 (1 TG in N), k=64 (2 K-blocks).
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
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
    let actual = bytes_to_f32(&out_bytes);
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
    println!("[f32 small m={m} n={n} k={k}] cos={cos:.6} max|Δ|={max_diff:.3e} at {max_at}",);
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_int8_f32_multi_k() {
    // Multiple K-blocks to exercise the loop body.
    let m = 32usize;
    let n = 32usize;
    let k = 256usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
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
    let actual = bytes_to_f32(&out_bytes);

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-k m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_int8_f32_multi_tile() {
    // Multiple tiles in both M and N dimensions.
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
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
    let actual = bytes_to_f32(&out_bytes);

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-tile m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_int8_f16_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_inputs(m, n, k, gs_per_row);
    // Round-trip through f16 so oracle matches GPU loads.
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
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
    let actual = bytes_to_f16_as_f32(&out_bytes);
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[f16 small m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16)");
}

#[test]
fn mt_qmm_mma_int8_f16_multi_tile() {
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
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
    let actual = bytes_to_f16_as_f32(&out_bytes);

    let cos = cosine(&expected, &actual);
    println!("[f16 multi-tile m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 multi-tile)");
}

#[test]
fn mt_qmm_mma_int8_bf16_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_inputs(m, n, k, gs_per_row);
    let round_bf16 = |v: f32| half::bf16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_int8(
        &ctx,
        DType::BF16,
        &w,
        &f32_to_bf16_bytes(&scales),
        &f32_to_bf16_bytes(&biases),
        &f32_to_bf16_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = bytes_to_bf16_as_f32(&out_bytes);
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[bf16 small m={m} n={n} k={k}] cos={cos:.6}");
    // bf16 has 7-bit mantissa — looser tolerance.
    assert!(cos >= 0.997, "cosine {cos:.6} < 0.997 (bf16)");
}

// ── mt_qmm_mma_m16_int8 tests ────────────────────────────────────────────────

#[test]
fn mt_qmm_mma_m16_int8_f32_small() {
    // M=16 single tile: m=16, n=32, k=64.
    let m = 16usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_m16_int8(
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
    let actual = bytes_to_f32(&out_bytes);
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
    println!("[m16 f32 small m={m} n={n} k={k}] cos={cos:.6} max|Δ|={max_diff:.3e} at {max_at}",);
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_m16_int8_f32_multi_k() {
    let m = 16usize;
    let n = 32usize;
    let k = 256usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_m16_int8(
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
    let actual = bytes_to_f32(&out_bytes);

    let cos = cosine(&expected, &actual);
    println!("[m16 f32 multi-k m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_m16_int8_f32_multi_tile() {
    let m = 32usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_m16_int8(
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
    let actual = bytes_to_f32(&out_bytes);

    let cos = cosine(&expected, &actual);
    println!("[m16 f32 multi-tile m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn mt_qmm_mma_m16_int8_f16_small() {
    let m = 16usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_m16_int8(
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
    let actual = bytes_to_f16_as_f32(&out_bytes);
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[m16 f16 small m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16)");
}

#[test]
fn mt_qmm_mma_m16_int8_bf16_small() {
    let m = 16usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_inputs(m, n, k, gs_per_row);
    let round_bf16 = |v: f32| half::bf16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_mt_qmm_mma_m16_int8(
        &ctx,
        DType::BF16,
        &w,
        &f32_to_bf16_bytes(&scales),
        &f32_to_bf16_bytes(&biases),
        &f32_to_bf16_bytes(&x),
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual = bytes_to_bf16_as_f32(&out_bytes);
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[m16 bf16 small m={m} n={n} k={k}] cos={cos:.6}");
    assert!(cos >= 0.997, "cosine {cos:.6} < 0.997 (bf16)");
}
