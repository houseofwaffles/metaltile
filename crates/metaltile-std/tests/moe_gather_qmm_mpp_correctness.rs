#![allow(clippy::manual_is_multiple_of)]

//! GPU correctness for `ffai::moe_mpp::mt_moe_gather_qmm_mma_int4_bm16_mpp`.
//!
//! This is the MPP (MetalPerformancePrimitives) MoE BGEMM — same algorithm
//! and output as `mt_moe_gather_qmm_mma_int4_bm16` but routes the inner
//! 16×32×16 tile through `mpp::tensor_ops::matmul2d` (Apple's NAX-tapping
//! cooperative-tensor API). Validated against the scalar `mt_moe_gather_qmm_int4`
//! oracle on the same "clean tile" shape the simdgroup-matrix bm16 variant
//! is tested at (n_experts=4, T=64, N=64, K=64, group_size=32).
//!
//! Requires macOS 26+ / Metal 4 for the MPP header to be available. On
//! older toolchains the kernel falls through to a zero-write stub and this
//! test is expected to fail loudly — that's the intended signal.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{moe::mt_moe_gather_qmm_int4, moe_mpp};

/// Pack a row of int4 weights into uint32s (8 per uint, LSB-first per nibble).
/// Identical to the helper used by `moe_gather_qmm_gpu_correctness.rs` —
/// duplicated to keep this test file self-contained.
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

/// Clean-tile correctness: MPP MoE kernel matches the scalar m1 reference
/// at n_experts=4, T=64, N=64, K=64, group_size=32. Cosine ≥ 0.999.
///
/// The "clean tile" name comes from all dims being multiples of the BM=16
/// × BN=32 × BK=16 descriptor — no per-row mask edge cases, no K-remainder.
/// Same shape the simdgroup-matrix BM=16 variant validates on, which lets
/// us inherit the m1-vs-bm16 cosine sanity from the sibling test.
#[test]
fn moe_gather_qmm_mma_int4_bm16_mpp_matches_m1_clean_tile() {
    let _g = gpu_lock();

    // MPP `tensor_ops::matmul2d` needs Apple10 (gen-17) + macOS 26.2+.
    // On older silicon or virtualised CI runners the kernel hits its
    // pre-Metal-4 stub branch and writes zeros — skip rather than fail.
    let probe = Context::new().expect("Context::new");
    let family = probe.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bm16_mpp_clean_tile: needs Apple10+ GPU (chip_family={family:?})");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 64usize; // multiple of 32 (and 16 = BK)
    let n_out = 64usize; // BN=32 → 2 n-tiles
    let group_size = 32usize;
    let t_rows = 64usize; // BM=16 → 4 m-tiles

    // Per-row expert indices, sorted: rows 0..16 → e0, 16..32 → e1, etc.
    // This is the post-permute layout the MoE pipeline produces.
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    // m1 reference uses expert_offsets (first-row-of-each-expert), not the
    // per-row indices — build it from the sorted indices.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // ── Reference: scalar m1 ─────────────────────────────────────────────
    let y_m1 = {
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

    // ── Under test: MPP MoE kernel ───────────────────────────────────────
    let y_mpp = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = moe_mpp::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        // Grid: [N/BN=32, ceil(T/BM=16), 1]. TG: 32 lanes = 1 SG (MPP's
        // matmul2d uses `execution_simdgroup`).
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out / 32, t_rows.div_ceil(16), 1],
                [32, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // MPP cooperative-tensor accumulator vs scalar reduction — fp
    // accumulation order differs, so cosine is the right metric (same
    // criterion the bm16 simdgroup-matrix variant uses).
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
    eprintln!("y_m1[0..8]  = {:?}", &y_m1[..8]);
    eprintln!("y_mpp[0..8] = {:?}", &y_mpp[..8]);
    eprintln!("nan_count = {nan_count} / {}", t_rows * n_out);
    assert_eq!(nan_count, 0, "MPP kernel produced non-finite values");
    assert!(cos >= 0.999, "MPP MoE vs m1 cosine = {cos:.6} (want ≥ 0.999)");
}

/// bf16 activations. `mpp::tensor_ops::matmul2d` mishandles `bfloat`
/// cooperative tensors, so the bf16 kernel reads device `bfloat`, stages
/// through `half` threadgroup tiles + half coop tensors, and accumulates
/// in fp32. This cell guards that path: cosine ≥ 0.997 (looser than the
/// f32 cell's 0.999 only because `x`/`scales`/`biases` are themselves
/// bf16-rounded — matches the bm8 bf16 cells' bar) — and never the
/// garbage a broken `bfloat` matmul produced (cosine ≈ 0). Same clean-tile
/// shape as `..._matches_m1_clean_tile`; m1 oracle runs in f32.
#[test]
fn moe_gather_qmm_mma_int4_bm16_mpp_bf16_matches_m1_clean_tile() {
    let _g = gpu_lock();

    let probe = Context::new().expect("Context::new");
    let family = probe.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bm16_mpp_bf16_clean_tile: needs Apple10+ GPU (chip_family={family:?})");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // ── Reference: scalar m1 in f32 ──────────────────────────────────────
    let y_m1 = {
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

    // ── Under test: MPP MoE kernel, bf16 activations ─────────────────────
    let y_mpp = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::Bf16));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::Bf16));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::Bf16));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::Bf16));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = moe_mpp::kernel_ir_for(Dt::Bf16.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out / 32, t_rows.div_ceil(16), 1],
                [32, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::Bf16)
    };

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
    eprintln!("bf16 y_m1[0..8]  = {:?}", &y_m1[..8]);
    eprintln!("bf16 y_mpp[0..8] = {:?}", &y_mpp[..8]);
    eprintln!("bf16 cosine      = {cos:.6}");
    assert_eq!(nan_count, 0, "MPP bf16 kernel produced non-finite values");
    assert!(cos >= 0.997, "MPP MoE bf16 vs m1 cosine = {cos:.6} (want ≥ 0.997)");
}
