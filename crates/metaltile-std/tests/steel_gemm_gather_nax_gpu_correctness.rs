//! GPU correctness oracle for `mt_steel_gemm_gather_nax` — gather GEMM
//! `C = A_gathered · B_gathered` backed by `mpp::tensor_ops::matmul2d`.
//!
//! `lhs_indices[out_row]` redirects each output row to an `A` source row;
//! `rhs_indices[n_block]` selects which stacked `[K, N]` `B` matrix a
//! 32-wide N-block multiplies against. Validated against a naive CPU
//! oracle that applies the same two redirections.
//!
//! Requires macOS 26+ / Metal 4. Gated behind the `nax` Cargo feature.
//!
//! Run:
//!   cargo test --release -p metaltile-std --features nax \
//!     --test steel_gemm_gather_nax_gpu_correctness -- --nocapture

#![cfg(all(target_os = "macos", feature = "nax"))]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::gemm::steel_gemm_gather_nax::mt_steel_gemm_gather_nax;

/// Gather-GEMM oracle. `a` is `[n_a_rows, k]`, `b` is the stacked
/// `[n_b_mats, k, n]`. `out[mr, nc] = Σ_k a[lhs[mr], k] · b[rhs[nc/32], k, nc]`.
fn cpu_gather_gemm_reference(
    a: &[f32],
    b: &[f32],
    lhs: &[u32],
    rhs: &[u32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        let a_row = lhs[m_row] as usize;
        for n_col in 0..n {
            let b_mat = rhs[n_col / 32] as usize;
            let b_base = b_mat * k * n;
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[a_row * k + kk] * b[b_base + kk * n + n_col];
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gather_nax(
    ctx: &Context,
    dtype: DType,
    a_bytes: &[u8],
    b_bytes: &[u8],
    lhs: &[u32],
    rhs: &[u32],
    m: usize,
    n: usize,
    k: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(32));

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), a_bytes.to_vec());
    buffers.insert("b".into(), b_bytes.to_vec());
    buffers.insert("lhs_indices".into(), lhs.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("rhs_indices".into(), rhs.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let mut kernel = mt_steel_gemm_gather_nax::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch mt_steel_gemm_gather_nax");
    result.outputs.get("out").expect("`out` buffer").clone()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}
fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (xf, yf) = (*x as f64, *y as f64);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
}

/// `m`-row gather permutation + per-N-block B-matrix selection.
fn build_indices(m: usize, n: usize, n_a_rows: usize, n_b_mats: usize) -> (Vec<u32>, Vec<u32>) {
    // Stride-7 walk over the A rows — a non-identity, non-contiguous map.
    let lhs: Vec<u32> = (0..m).map(|i| ((i * 7 + 3) % n_a_rows) as u32).collect();
    let rhs: Vec<u32> = (0..n / 32).map(|i| ((i * 2 + 1) % n_b_mats) as u32).collect();
    (lhs, rhs)
}

fn build_inputs(n_a_rows: usize, n_b_mats: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n_a_rows * k).map(|i| 0.01 + (i as f32 % 17.0) * 0.013).collect();
    let b: Vec<f32> = (0..n_b_mats * k * n).map(|i| -0.02 + (i as f32 % 13.0) * 0.011).collect();
    (a, b)
}

fn run_case(dtype: DType, label: &str, cos_min: f32) {
    let (m, n, k) = (64usize, 64usize, 128usize);
    let (n_a_rows, n_b_mats) = (96usize, 3usize);
    let (lhs, rhs) = build_indices(m, n, n_a_rows, n_b_mats);
    let (a_f32, b_f32) = build_inputs(n_a_rows, n_b_mats, n, k);

    // Round inputs to the kernel dtype so the oracle and GPU agree.
    let (a, b, abytes, bbytes, obpe): (Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>, usize) = match dtype {
        DType::F16 => {
            let r = |v: f32| half::f16::from_f32(v).to_f32();
            let a: Vec<f32> = a_f32.iter().map(|&v| r(v)).collect();
            let b: Vec<f32> = b_f32.iter().map(|&v| r(v)).collect();
            let (ab, bb) = (f16_bytes(&a), f16_bytes(&b));
            (a, b, ab, bb, 2)
        },
        DType::BF16 => {
            let r = |v: f32| half::bf16::from_f32(v).to_f32();
            let a: Vec<f32> = a_f32.iter().map(|&v| r(v)).collect();
            let b: Vec<f32> = b_f32.iter().map(|&v| r(v)).collect();
            let (ab, bb) = (bf16_bytes(&a), bf16_bytes(&b));
            (a, b, ab, bb, 2)
        },
        _ => {
            let (ab, bb) = (f32_bytes(&a_f32), f32_bytes(&b_f32));
            (a_f32, b_f32, ab, bb, 4)
        },
    };

    let expected = cpu_gather_gemm_reference(&a, &b, &lhs, &rhs, m, n, k);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_gather_nax(&ctx, dtype, &abytes, &bbytes, &lhs, &rhs, m, n, k, obpe);
    let actual: Vec<f32> = match dtype {
        DType::F16 => out_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DType::BF16 => out_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    };
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[{label}] cos={cos:.6}");
    assert!(cos >= cos_min, "cosine {cos:.6} < {cos_min} ({label})");
}

#[test]
fn mt_steel_gemm_gather_nax_matches_cpu_reference_f32() {
    run_case(DType::F32, "gather f32 multi-tile", 0.999);
}

#[test]
fn mt_steel_gemm_gather_nax_matches_cpu_reference_f16() {
    run_case(DType::F16, "gather f16 multi-tile", 0.999);
}

#[test]
fn mt_steel_gemm_gather_nax_matches_cpu_reference_bf16() {
    // bf16 staged through `half` via `coop_stage(T)`. 7-bit mantissa → 0.997 bar.
    run_case(DType::BF16, "gather bf16 multi-tile", 0.997);
}
