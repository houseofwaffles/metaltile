#![allow(clippy::manual_is_multiple_of)]

//! GPU correctness for `ffai::moe_mpp_bm8::mt_moe_gather_qmm_mma_int4_bm8_mpp`.
//!
//! BM=8 MPP MoE kernel — same output semantics as the BM=16 / BM=64 siblings
//! but the per-TG row tile shrinks to 8 to match decode-time MoE shapes
//! (T=1, topK=8 → m_total=8 after gather/permute). Validated against the
//! scalar `mt_moe_gather_qmm_int4` oracle on a battery of shapes covering
//! the f32 / f16 / bf16 dtypes, single-tile + multi-tile + ragged-T, and
//! the Qwen3.6-A3B production tile.
//!
//! Requires macOS 26+ / Metal 4 for the MPP header to be available. On
//! older toolchains the kernel falls through to a zero-write stub and the
//! tests fail loudly — that's the intended signal.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{moe::mt_moe_gather_qmm_int4, moe_mpp_bm8};

/// Pack a row of int4 weights into uint32s (8 per uint, LSB-first per
/// nibble). Same helper used by the bm16_mpp / bm64_mpp test files —
/// duplicated so this test stays self-contained.
fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    assert!(weights.len() % 8 == 0);
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

/// Shared workload spec for a single bm8 vs scalar-m1 comparison.
struct Case {
    n_experts: usize,
    /// Number of rows in the gathered/permuted input (post-MoE-permute).
    t_rows: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
    /// Activation/scale/bias/output dtype used on both reference + kernel.
    dt: Dt,
    /// Override the indices vector; defaults to a sorted-per-expert layout
    /// (rows split evenly across experts). Used by `ragged_T` to test
    /// odd row counts.
    indices: Option<Vec<u32>>,
    /// Cosine threshold. Defaults to 0.999.
    min_cos: f64,
    /// Label for error messages.
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

/// Run one bm8 vs m1 comparison. Returns nothing — asserts cosine ≥
/// `case.min_cos` and zero NaN/Inf elements. Centralising the boilerplate
/// keeps the actual tests tiny: each case is ~5 lines.
fn run_case(case: &Case) {
    let _g = gpu_lock();

    // MPP `tensor_ops::matmul2d` needs Apple10 (gen-17) + macOS 26.2+. On
    // older silicon or virtualised CI runners (chip_family = None or < 10)
    // the kernel hits its pre-Metal-4 stub branch and writes zeros — the
    // cosine assertion would then fail. Skip rather than fail so CI's
    // hosted Mac runner stays green. Same gate pattern used on the bm64
    // / qmm / smoke tests.
    let ctx_probe = Context::new().expect("Context::new");
    let family = ctx_probe.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip {}: needs Apple10+ GPU (chip_family={family:?})", case.label);
        return;
    }
    drop(ctx_probe);

    let Case { n_experts, t_rows, n_out, k_in, group_size, dt, .. } = *case;

    // Default index layout: rows split as evenly as possible across
    // experts in sorted order — the post-permute layout the MoE pipeline
    // produces. Each row hashes to floor(row * n_experts / t_rows).
    let indices: Vec<u32> = case
        .indices
        .clone()
        .unwrap_or_else(|| (0..t_rows).map(|r| ((r * n_experts) / t_rows) as u32).collect());

    // Deterministic but non-uniform weights / scales / biases / x.
    // Same generator family used by the bm16 / bm64 tests, just
    // different coefficients so each case hits a different mix of
    // dequant bits.
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32).wrapping_mul(13).wrapping_add(7)) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.041).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.083).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.019).sin()).collect();

    // expert_offsets for the m1 reference. The scalar kernel walks
    // `expert_offsets[e]..expert_offsets[e+1]` for each expert and skips
    // experts with empty spans. Build from the (sorted) indices.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // ── Reference: scalar m1 always in F32 (it's the spec oracle) ────────
    let y_m1: Vec<f32> = {
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
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // ── Under test: BM=8 MPP MoE kernel at the case's dtype ──────────────
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
        let mut k = moe_mpp_bm8::kernel_ir_for(dt.to_dtype());
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

    // Cosine vs the m1 oracle. fp accumulation order differs (cooperative
    // tensor reduction inside the MMA frag) so cosine is the right metric
    // — same convention the bm16 / bm64 siblings use.
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut nan_count = 0usize;
    for (a, b) in y_m1.iter().zip(&y_mpp) {
        if !a.is_finite() || !b.is_finite() {
            nan_count += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("[{}] y_m1[0..8]  = {:?}", case.label, &y_m1[..y_m1.len().min(8)]);
    eprintln!("[{}] y_mpp[0..8] = {:?}", case.label, &y_mpp[..y_mpp.len().min(8)]);
    eprintln!("[{}] nan_count   = {} / {}", case.label, nan_count, t_rows * n_out);
    eprintln!("[{}] cosine      = {:.6}", case.label, cos);
    assert_eq!(nan_count, 0, "[{}] MPP BM=8 produced non-finite values", case.label);
    assert!(
        cos >= case.min_cos,
        "[{}] MPP MoE BM=8 vs m1 cosine = {:.6} (want ≥ {:.3})",
        case.label,
        cos,
        case.min_cos
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Case 1: f32 small — the canonical T=8 decode shape (topK=8). Single TG,
// single n-tile (N=64 = 2 × BN=32). Covers the "exactly one BM=8 tile"
// happy path with multiple sub-runs (sorted indices split rows across
// 4 experts at 2 rows/expert).
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_f32_small_t8() { run_case(&Case::new("f32_small_t8", 4, 8, 64, 64, 32, Dt::F32)); }

// ─────────────────────────────────────────────────────────────────────────
// Case 2: f16 small — same shape as case 1, f16 dtype. f16 quant
// accumulation lands within the cosine envelope easily.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_f16_small_t8() { run_case(&Case::new("f16_small_t8", 4, 8, 64, 64, 32, Dt::F16)); }

// ─────────────────────────────────────────────────────────────────────────
// Case 3: bf16 small — same shape, bf16 dtype. bf16 has a 7-bit mantissa
// vs f16's 10-bit, so the cosine is looser; 0.997 still cleanly catches
// any structural bug while tolerating the dequant precision loss.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_bf16_small_t8() {
    let mut case = Case::new("bf16_small_t8", 4, 8, 64, 64, 32, Dt::Bf16);
    case.min_cos = 0.997;
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────
// Case 4: f16 multi-tile — T=16 forces 2 BM=8 tiles in the M dim,
// N=128 forces 4 BN=32 tiles. 8 experts × 2 rows/expert means each TG
// covers ~4 sub-runs. Tests cross-tile correctness (different (tgid_x,
// tgid_y) → different output regions, all must match m1).
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_f16_multi_tile() { run_case(&Case::new("f16_multi_tile", 8, 16, 128, 128, 64, Dt::F16)); }

// ─────────────────────────────────────────────────────────────────────────
// Case 5: f16 ragged T (T=5) — not a multiple of BM=8. The last TG covers
// rows 0..5 with rows 5..7 masked off by the in-kernel `gr < m_total`
// check. Custom indices: 5 rows split across 3 experts. Verifies the row
// mask + sub_end clamp don't write past `m_total`.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_f16_ragged_t5() {
    let mut case = Case::new("f16_ragged_t5", 3, 5, 64, 64, 32, Dt::F16);
    // Rows 0-1 → e0, rows 2-3 → e1, row 4 → e2. Tests both
    // boundary-between-experts and single-row trailing expert.
    case.indices = Some(vec![0, 0, 1, 1, 2]);
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────
// Case 6: f16 production shape — Qwen3.6-A3B-like decode tile.
// n_experts=128 (full Qwen3.6 expert count), T=8 (T=1 batch × topK=8),
// N=512 (modest down_proj fragment — full is 2048 but tests run faster on
// a 16-tile output), K=2048 (down_proj input dim), group_size=64. Single
// row per expert because each token routes to its own expert at decode.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_f16_production_shape() {
    // Custom indices: 8 distinct experts (one per row) drawn from the
    // 128-expert space, monotonic so expert_offsets stays correct.
    let mut case = Case::new("f16_production_shape", 128, 8, 512, 2048, 64, Dt::F16);
    case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
    run_case(&case);
}

// ─────────────────────────────────────────────────────────────────────────
// Case 7: bf16 production shape — same as case 6 but bf16 dtype, the
// production activation precision for Qwen3.6-A3B-bf16 builds.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn bm8_bf16_production_shape() {
    let mut case = Case::new("bf16_production_shape", 128, 8, 512, 2048, 64, Dt::Bf16);
    case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
    case.min_cos = 0.997;
    run_case(&case);
}
