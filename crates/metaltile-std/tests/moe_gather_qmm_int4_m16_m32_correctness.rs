//! GPU correctness for `mt_moe_gather_qmm_int4_m16` and
//! `mt_moe_gather_qmm_int4_m32` — short-prefill MoE GEMV with int4
//! quantized weights and 16/32-row-per-TG batching.
//!
//! Both variants are correctness-compared against `mt_moe_gather_qmm_int4_m8`
//! (which is already tested): same expert routing + int4 dequant, just
//! handling more output rows per TG. Any difference in routing or dequant
//! logic would surface as a divergence between m8 and m16/m32.
//!
//! Grid:
//!   `mt_moe_gather_qmm_int4_m16`: `[m_out/16, t_rows, 1]`, TPG = 32 (1 SG).
//!   `mt_moe_gather_qmm_int4_m32`: `[m_out/32, t_rows, 1]`, TPG = 32 (1 SG).
//!
//! Tolerances: cosine ≥ 0.999 vs CPU oracle; max abs Δ < 5e-4 vs m8.

#![cfg(target_os = "macos")]
#![allow(clippy::manual_is_multiple_of)]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{
    mt_moe_gather_qmm_int4_m8,
    mt_moe_gather_qmm_int4_m16,
    mt_moe_gather_qmm_int4_m32,
};

// ── CPU oracle ───────────────────────────────────────────────────────────────

/// Pack a row of int4 weight codes into u32 words (8 codes per u32, LSB-first).
fn pack_int4_row(codes: &[u32]) -> Vec<u32> {
    assert!(codes.len() % 8 == 0, "pack_int4_row: length must be multiple of 8");
    codes
        .chunks_exact(8)
        .map(|ch| ch.iter().enumerate().fold(0u32, |acc, (i, &q)| acc | ((q & 0xF) << (i * 4))))
        .collect()
}

/// CPU oracle: per-row expert routing + int4-quantized dot product.
#[allow(clippy::too_many_arguments)]
fn cpu_oracle(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    expert_offsets: &[u32],
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    let weight_stride_m = k_in / 8;
    let groups_per_row = k_in / group_size;
    let mut out = vec![0.0_f32; t_rows * m_out];
    for row in 0..t_rows {
        // Resolve expert: first e where row < expert_offsets[e+1].
        let mut expert = 0usize;
        for e in 0..n_experts {
            if (row as u32) < expert_offsets[e + 1] {
                expert = e;
                break;
            }
        }
        for m in 0..m_out {
            let w_base = expert * m_out * weight_stride_m + m * weight_stride_m;
            let s_base = expert * m_out * groups_per_row + m * groups_per_row;
            let mut acc = 0.0_f32;
            for pack_idx in 0..weight_stride_m {
                let k_first = pack_idx * 8;
                let g = k_first / group_size;
                let s = scales[s_base + g];
                let b = biases[s_base + g];
                let p = weight_packed[w_base + pack_idx];
                for i in 0..8usize {
                    let code = ((p >> (i * 4)) & 0xF) as f32;
                    let x_val = x[row * k_in + k_first + i];
                    acc += (code * s + b) * x_val;
                }
            }
            out[row * m_out + m] = acc;
        }
    }
    let _ = n_experts;
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

// ── Input builder ────────────────────────────────────────────────────────────

/// `(weight_packed, scales, biases, x, expert_offsets)` from `build_inputs`.
type MoeInputs = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>);

/// Build deterministic int4-quantized MoE inputs.
/// Returns (weight_packed, scales, biases, x, expert_offsets).
fn build_inputs(
    n_experts: usize,
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    group_size: usize,
) -> MoeInputs {
    let groups_per_row = k_in / group_size;
    let weight_stride_m = k_in / 8;

    // Generate codes in [1, 14] to avoid degenerate zero/all-ones.
    let total_codes = n_experts * m_out * k_in;
    let codes_flat: Vec<u32> = (0..total_codes)
        .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(12) % 14 + 1)
        .collect();

    let weight_packed: Vec<u32> = codes_flat.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    assert_eq!(weight_packed.len(), n_experts * m_out * weight_stride_m);

    let scales: Vec<f32> = (0..n_experts * m_out * groups_per_row)
        .map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin().abs())
        .collect();
    let biases: Vec<f32> = (0..n_experts * m_out * groups_per_row)
        .map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos())
        .collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    // Evenly distribute rows across experts.
    let rows_per_expert = t_rows / n_experts;
    let mut expert_offsets = vec![0u32; n_experts + 1];
    for e in 0..n_experts {
        expert_offsets[e + 1] = expert_offsets[e] + rows_per_expert as u32;
    }
    // Absorb any remainder into the last expert.
    expert_offsets[n_experts] = t_rows as u32;

    (weight_packed, scales, biases, x, expert_offsets)
}

// ── Dispatch helper ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    kernel_ir: fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
    grid_x: usize,
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    expert_offsets: &[u32],
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    let ctx = Context::new().expect("Context::new on macOS");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, Dt::F32));
    buffers.insert(
        "weight_packed".into(),
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
    );
    buffers.insert("scales".into(), pack_bytes(scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(biases, Dt::F32));
    buffers.insert(
        "expert_offsets".into(),
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
    );
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let mut kernel = kernel_ir(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, t_rows, 1], [32, 1, 1])
        .expect("dispatch gather_qmm_int4_m_batch");
    unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32)
}

// ── Test runner ──────────────────────────────────────────────────────────────

fn run_case_m16(n_experts: usize, t_rows: usize, k_in: usize, m_out: usize, group_size: usize) {
    assert!(m_out % 16 == 0, "m16: m_out must be multiple of 16");
    assert!(m_out % 8 == 0, "m8: m_out must be multiple of 8");

    let (weight_packed, scales, biases, x, expert_offsets) =
        build_inputs(n_experts, t_rows, k_in, m_out, group_size);

    let expected = cpu_oracle(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let _g = gpu_lock();
    let y_m8 = run_kernel(
        mt_moe_gather_qmm_int4_m8::kernel_ir_for,
        m_out / 8,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );
    let y_m16 = run_kernel(
        mt_moe_gather_qmm_int4_m16::kernel_ir_for,
        m_out / 16,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    assert_eq!(y_m16.len(), expected.len());
    assert!(y_m16.iter().any(|&v| v != 0.0), "m16: all-zero output");

    let cos = cosine(&expected, &y_m16);
    let max_diff_vs_m8 =
        y_m8.iter().zip(&y_m16).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "[m16 n_experts={n_experts} t={t_rows} k={k_in} m={m_out}] cos={cos:.6} max|Δ_m8|={max_diff_vs_m8:.2e}"
    );
    assert!(cos >= 0.999, "m16: cosine {cos:.6} < 0.999");
    assert!(max_diff_vs_m8 < 5e-4, "m16 vs m8: max diff = {max_diff_vs_m8:.2e}");
}

fn run_case_m32(n_experts: usize, t_rows: usize, k_in: usize, m_out: usize, group_size: usize) {
    assert!(m_out % 32 == 0, "m32: m_out must be multiple of 32");
    assert!(m_out % 8 == 0, "m8: m_out must be multiple of 8");

    let (weight_packed, scales, biases, x, expert_offsets) =
        build_inputs(n_experts, t_rows, k_in, m_out, group_size);

    let expected = cpu_oracle(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let _g = gpu_lock();
    let y_m8 = run_kernel(
        mt_moe_gather_qmm_int4_m8::kernel_ir_for,
        m_out / 8,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );
    let y_m32 = run_kernel(
        mt_moe_gather_qmm_int4_m32::kernel_ir_for,
        m_out / 32,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    assert_eq!(y_m32.len(), expected.len());
    assert!(y_m32.iter().any(|&v| v != 0.0), "m32: all-zero output");

    let cos = cosine(&expected, &y_m32);
    let max_diff_vs_m8 =
        y_m8.iter().zip(&y_m32).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "[m32 n_experts={n_experts} t={t_rows} k={k_in} m={m_out}] cos={cos:.6} max|Δ_m8|={max_diff_vs_m8:.2e}"
    );
    assert!(cos >= 0.999, "m32: cosine {cos:.6} < 0.999");
    assert!(max_diff_vs_m8 < 5e-4, "m32 vs m8: max diff = {max_diff_vs_m8:.2e}");
}

// ── mt_moe_gather_qmm_int4_m16 tests ─────────────────────────────────────────

#[test]
fn moe_gather_qmm_int4_m16_f32_small() {
    // 4 experts, 4 T-rows per expert (16 total), k=256, m_out=64, gs=64.
    run_case_m16(4, 16, 256, 64, 64);
}

#[test]
fn moe_gather_qmm_int4_m16_f32_multi_expert() {
    // 8 experts, 4 T-rows per expert (32 total), k=512, m_out=64, gs=64.
    run_case_m16(8, 32, 512, 64, 64);
}

#[test]
fn moe_gather_qmm_int4_m16_f32_wide_m() {
    // Wider m_out: 128 = 8 chunks of 16.
    run_case_m16(4, 16, 256, 128, 64);
}

// ── mt_moe_gather_qmm_int4_m32 tests ─────────────────────────────────────────

#[test]
fn moe_gather_qmm_int4_m32_f32_small() {
    // 4 experts, 4 T-rows per expert (16 total), k=256, m_out=64, gs=64.
    run_case_m32(4, 16, 256, 64, 64);
}

#[test]
fn moe_gather_qmm_int4_m32_f32_multi_expert() {
    // 8 experts, 4 T-rows per expert (32 total), k=512, m_out=64, gs=64.
    run_case_m32(8, 32, 512, 64, 64);
}

#[test]
fn moe_gather_qmm_int4_m32_f32_wide_m() {
    // Wider m_out: 128 = 4 chunks of 32.
    run_case_m32(4, 16, 256, 128, 64);
}
