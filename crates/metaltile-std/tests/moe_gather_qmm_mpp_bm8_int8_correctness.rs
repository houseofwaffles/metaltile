#![allow(clippy::manual_is_multiple_of)]

//! GPU correctness for `ffai::moe_mpp_bm8_int8::mt_moe_gather_qmm_mma_int8_bm8_mpp`.
//!
//! BM=8 MPP MoE int8 kernel — same output semantics as the int4 BM=8 sibling
//! but the weight layout changes from 8 nibbles/u32 to 4 bytes/u32.
//! Validated against the scalar `mt_moe_gather_qmm_b8` oracle (pow2-width
//! family from `moe.rs`) on a battery of shapes covering f32 / f16 / bf16
//! dtypes, single-tile + multi-tile + ragged-T, and a production-like tile.
//!
//! Requires macOS 26+ / Metal 4 for the MPP header to be available. On older
//! silicon the kernel falls through to a zero-write stub and the Apple10 gate
//! below skips rather than fails — that is the intended signal.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{moe::mt_moe_gather_qmm_b8, moe_mpp_bm8_int8};

/// Pack a row of int8 weight codes into uint32s (4 codes per uint, LE byte
/// order). Code values must be in 0..=255.
fn pack_int8_row(weights: &[u32]) -> Vec<u32> {
    assert!(weights.len() % 4 == 0);
    weights
        .chunks_exact(4)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xff) << (i * 8);
            }
            packed
        })
        .collect()
}

/// Skip the test unless the GPU is Apple10+ (gen-17). Returns a live Context
/// when the hardware qualifies, None to skip.
fn skip_unless_apple10(test_name: &str) -> Option<Context> {
    let ctx = Context::new().expect("Context::new");
    let family = ctx.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip {test_name}: needs Apple10+ GPU (chip_family={family:?})");
        return None;
    }
    Some(ctx)
}

/// Workload spec for a single bm8-int8 vs scalar-b8 comparison.
struct Case {
    n_experts: usize,
    /// Post-permute row count (= T × topK after MoE gather).
    t_rows: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
    dt: Dt,
    /// Optional override for the per-row expert indices vector.
    indices: Option<Vec<u32>>,
    /// Cosine similarity threshold. Defaults to 0.999.
    min_cos: f64,
    label: &'static str,
}

impl Case {
    fn new(
        label: &'static str,
        n_experts: usize,
        t_rows: usize,
        n_out: usize,
        k_in: usize,
        group_size: usize,
        dt: Dt,
    ) -> Self {
        Self {
            n_experts,
            t_rows,
            n_out,
            k_in,
            group_size,
            dt,
            indices: None,
            min_cos: 0.999,
            label,
        }
    }
}

/// Run one bm8-int8 vs scalar-b8 comparison. Asserts cosine ≥ `case.min_cos`
/// and zero NaN/Inf elements.
fn run_case(case: &Case) {
    let _g = gpu_lock();
    let Some(_ctx) = skip_unless_apple10(case.label) else { return };

    let Case { n_experts, t_rows, n_out, k_in, group_size, dt, .. } = *case;

    // Default index layout: rows distributed evenly across experts in sorted
    // order — the post-permute layout the MoE pipeline produces.
    let indices: Vec<u32> = case
        .indices
        .clone()
        .unwrap_or_else(|| (0..t_rows).map(|r| ((r * n_experts) / t_rows) as u32).collect());

    // Deterministic int8 weight codes in 0..=255.
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32).wrapping_mul(13).wrapping_add(7)) & 0xff).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int8_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.041).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.083).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.019).sin()).collect();

    // Build expert_offsets for the scalar b8 reference (first-row-of-expert).
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // ── Reference: scalar b8 oracle (always f32) ──────────────────────────
    let y_ref: Vec<f32> = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_b8::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // ── Under test: BM=8 MPP int8 kernel at the case's dtype ─────────────
    let y_mpp: Vec<f32> = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, dt));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, dt));
        buffers.insert("biases".into(), pack_bytes(&biases, dt));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], dt));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k =
            moe_mpp_bm8_int8::mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(dt.to_dtype());
        k.mode = KernelMode::Reduction;
        // Grid: [ceil(N/32), ceil(T/8), 1]. TG: 32 lanes = 1 SG.
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out.div_ceil(32), t_rows.div_ceil(8), 1],
                [32, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), dt)
    };

    // Cosine similarity vs the scalar oracle.
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut nan_count = 0usize;
    for (a, b) in y_ref.iter().zip(&y_mpp) {
        if !a.is_finite() || !b.is_finite() {
            nan_count += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("[{}] y_ref[0..8]  = {:?}", case.label, &y_ref[..y_ref.len().min(8)]);
    eprintln!("[{}] y_mpp[0..8] = {:?}", case.label, &y_mpp[..y_mpp.len().min(8)]);
    eprintln!("[{}] nan_count   = {} / {}", case.label, nan_count, t_rows * n_out);
    eprintln!("[{}] cosine      = {:.6}", case.label, cos);
    assert_eq!(nan_count, 0, "[{}] MPP BM=8 int8 produced non-finite values", case.label);
    assert!(
        cos >= case.min_cos,
        "[{}] MPP MoE BM=8 int8 vs b8 cosine = {:.6} (want ≥ {:.3})",
        case.label,
        cos,
        case.min_cos
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 1: f32 small — canonical T=8 decode shape (topK=8). Single TG,
// N=64 (2 BN=32 tiles). 4 experts × 2 rows each.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_f32_small_t8() { run_case(&Case::new("f32_small_t8", 4, 8, 64, 64, 32, Dt::F32)); }

// ─────────────────────────────────────────────────────────────────────────────
// Case 2: f16 small — same shape as case 1, f16 dtype.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_f16_small_t8() { run_case(&Case::new("f16_small_t8", 4, 8, 64, 64, 32, Dt::F16)); }

// ─────────────────────────────────────────────────────────────────────────────
// Case 3: bf16 small — same shape, bf16 dtype. 7-bit mantissa means
// a slightly looser cosine bar (0.997).
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_bf16_small_t8() {
    let mut case = Case::new("bf16_small_t8", 4, 8, 64, 64, 32, Dt::Bf16);
    case.min_cos = 0.997;
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 4: f16 multi-tile — T=16 forces 2 BM=8 tiles in M, N=128 forces
// 4 BN=32 tiles. 8 experts × 2 rows each. Tests cross-tile correctness.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_f16_multi_tile() {
    run_case(&Case::new("f16_multi_tile", 8, 16, 128, 128, 64, Dt::F16));
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 5: f16 ragged T (T=5) — not a multiple of BM=8. Verifies the
// `gr < m_total` mask and sub_end clamp don't write past m_total.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_f16_ragged_t5() {
    let mut case = Case::new("f16_ragged_t5", 3, 5, 64, 64, 32, Dt::F16);
    // Rows 0-1 → e0, rows 2-3 → e1, row 4 → e2.
    case.indices = Some(vec![0, 0, 1, 1, 2]);
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 6: f16 production shape — Qwen3.6-A3B-like decode tile. 128 experts,
// T=8 (T=1 × topK=8), N=512, K=2048, group_size=64. One row per expert.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_f16_production_shape() {
    let mut case = Case::new("f16_production_shape", 128, 8, 512, 2048, 64, Dt::F16);
    case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 7: bf16 production shape — same as case 6 but bf16 dtype.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_int8_bf16_production_shape() {
    let mut case = Case::new("bf16_production_shape", 128, 8, 512, 2048, 64, Dt::Bf16);
    case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
    case.min_cos = 0.997;
    run_case(&case);
}
