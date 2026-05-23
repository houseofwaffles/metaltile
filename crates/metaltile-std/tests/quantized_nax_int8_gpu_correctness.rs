//! GPU correctness oracle for `mt_qmm_nax_int8` — the int8 quantized
//! matmul backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Dispatches `mt_qmm_nax_int8_{f32,f16,bf16}` over a small set of
//! shapes and validates against a CPU triple-loop oracle.
//!
//! ## Key differences from the int4 test (`quantized_nax_gpu_correctness.rs`)
//!
//! - W is `[n, k/4]` — 4 bytes per u32 instead of 8 nibbles.
//! - CPU oracle extracts full bytes: `(packed >> (b*8)) & 0xFF`.
//! - group_size = 32 (natural int8 group; int4 used 64).
//! - Weight codes span 0..255 (uint8), not 0..15 (uint4).
//!
//! Gated behind the `nax` Cargo feature.
//!
//! Run:
//!   cargo test --release -p metaltile-std --features nax \
//!     --test quantized_nax_int8_gpu_correctness -- --nocapture

#![cfg(all(target_os = "macos", feature = "nax"))]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized_nax_int8::mt_qmm_nax_int8;

/// NAX `tensor_ops::matmul2d` requires Apple10 (gen-17) + macOS 26.2+.
/// Returns `None` when the device doesn't qualify so the caller can skip.
fn ctx_or_skip(test_name: &str) -> Option<Context> {
    let ctx = Context::new().expect("Context::new");
    let family = ctx.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip {test_name}: needs Apple10+ GPU (chip_family={family:?})");
        return None;
    }
    Some(ctx)
}

/// Triple-loop CPU oracle for int8 quantized matmul (same as mpp_int8 test).
///
/// W layout: `[n, k/4]` — each u32 packs 4 consecutive uint8 codes.
/// scales/biases: `[n, gs_per_row]`. group_size = 32.
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
    let mut out = vec![0.0f32; m * n];
    let packs_per_row = k / 4;
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..packs_per_row {
                let packed = w[n_col * packs_per_row + p];
                for b in 0..4usize {
                    let byte_val = ((packed >> (b * 8)) & 0xFF) as f32;
                    let k_idx = p * 4 + b;
                    let g = k_idx / group_size;
                    let scale = scales[n_col * gs_per_row + g];
                    let bias = biases[n_col * gs_per_row + g];
                    let xv = x[m_row * k + k_idx];
                    acc += (byte_val * scale + bias) * xv;
                }
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_qmm_nax_int8(
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
    assert!(m.is_multiple_of(32), "mt_qmm_nax_int8 requires m %% 32 == 0");
    assert!(n.is_multiple_of(32), "mt_qmm_nax_int8 requires n %% 32 == 0");
    assert!(k.is_multiple_of(32), "mt_qmm_nax_int8 requires k %% 32 == 0");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_nax_int8::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_qmm_nax_int8");

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

/// Build deterministic int8 quantized weight inputs.
fn build_int8_quant_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let packs_per_row = k / 4;
    let w: Vec<u32> = (0..n * packs_per_row)
        .map(|i| {
            let mut v = 0u32;
            for b in 0..4u32 {
                let code = (i as u32 * 4 + b) % 256;
                v |= code << (b * 8);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.01 + (i as f32) * 0.0001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.00001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    (w, scales, biases, x)
}

// ── Shape 1: smallest valid tile, f32 ───────────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_f32_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("f32_small") else { return };
    let out_bytes = run_qmm_nax_int8(
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
        "[nax int8 f32 small] cos={cos:.6} max|Δ|={max_diff:.3e} at idx {max_at} \
         (expected {:.4}, got {:.4})",
        expected[max_at], actual[max_at]
    );
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (nax int8 f32 small)");
}

// ── Shape 2: multi-K-block f32 ───────────────────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_f32_multi_k() {
    let m = 32usize;
    let n = 32usize;
    let k = 512usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("f32_multi_k") else { return };
    let out_bytes = run_qmm_nax_int8(
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
    println!("[nax int8 f32 multi-k k={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (nax int8 f32 multi-k)");
}

// ── Shape 3: multi-tile f32 (M=64, N=64) ────────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_f32_multi_tile() {
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("f32_multi_tile") else { return };
    let out_bytes = run_qmm_nax_int8(
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
    println!("[nax int8 f32 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (nax int8 f32 multi-tile)");
}

// ── Shape 4: f16 small ───────────────────────────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_f16_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();

    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("f16_small") else { return };
    let out_bytes = run_qmm_nax_int8(
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
    println!("[nax int8 f16 small] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (nax int8 f16 small)");
}

// ── Shape 5: f16 multi-tile ──────────────────────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_f16_multi_tile() {
    let m = 64usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();

    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("f16_multi_tile") else { return };
    let out_bytes = run_qmm_nax_int8(
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
    println!("[nax int8 f16 multi-tile m={m} n={n}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (nax int8 f16 multi-tile)");
}

// ── Shape 6: bf16 small (lower tolerance) ───────────────────────────────────

#[test]
fn mt_qmm_nax_int8_matches_cpu_reference_bf16_small() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
    let group_size = 32usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_int8_quant_inputs(m, n, k, gs_per_row);
    let round_bf16 = |v: f32| half::bf16::from_f32(v).to_f32();
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();

    let expected =
        cpu_qmm_int8_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let Some(ctx) = ctx_or_skip("bf16_small") else { return };
    let out_bytes = run_qmm_nax_int8(
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
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[nax int8 bf16 small] cos={cos:.6}");
    assert!(cos >= 0.997, "cosine {cos:.6} < 0.997 (nax int8 bf16 small)");
}
