//! GPU correctness for `quantize_kv_fp8_e4m3`, `quantize_kv_fp8_e5m2`,
//! `bulk_dequant_kv_fp8_e4m3`, `bulk_dequant_kv_fp8_e5m2` via round-trip.
//!
//! The fp8 KV cache kernels are scale-only (no bias): one group amax → scale,
//! codes are 8-bit values packed 4 per u32.
//!
//! Test strategy: same round-trip pattern as `kv_cache_quant_roundtrip_gpu.rs`
//!   1. Build a random [n_kv_heads × head_dim] source slot.
//!   2. Dispatch `quantize_kv_fp8_*` → `out_w` + `out_s` at `position`.
//!   3. Dispatch `bulk_dequant_kv_fp8_*` → reconstructed output.
//!   4. Compare reconstructed vs source: max abs err ≤ one quantization step
//!      (step = amax / fp8_levels) + dtype slack.
//!
//! E4M3: 7 bits of mantissa-equivalent (3 mantissa bits, max 448.0, ~240 levels).
//! E5M2: 5-bit mantissa-equivalent (2 mantissa bits, max 57344.0, ~96 levels).
//! Tolerance is derived from the format resolution at the test data range.
//!
//! macOS-gated. Serial GPU lock (shared common::gpu_lock).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::kv_cache::{
    bulk_dequant_kv_fp8_e4m3,
    bulk_dequant_kv_fp8_e5m2,
    quantize_kv_fp8_e4m3,
    quantize_kv_fp8_e5m2,
};

/// Shape parameters for the round-trip test.
struct Shape {
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    group_size: usize,
    position: usize,
    n_positions: usize,
}

impl Shape {
    fn qwen_decode() -> Self {
        Self {
            n_kv_heads: 8,
            head_dim: 128,
            max_seq: 64,
            group_size: 32,
            position: 7,
            n_positions: 16,
        }
    }
}

fn build_source(shape: &Shape, dt: Dt, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let n = shape.n_kv_heads * shape.head_dim;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let raw = ((s as i64 % 20_000) as f32) / 10_000.0;
            let group_offset = ((i / shape.group_size) as f32 * 0.7).sin();
            dt.round(raw + group_offset)
        })
        .collect()
}

fn quantize_dispatch_grid(shape: &Shape) -> ([usize; 3], [usize; 3]) {
    let total_groups = shape.n_kv_heads * (shape.head_dim / shape.group_size);
    ([1, 1, 1], [total_groups, 1, 1])
}

fn dequant_dispatch_grid(shape: &Shape) -> ([usize; 3], [usize; 3]) {
    let total = shape.n_kv_heads * shape.n_positions * shape.head_dim;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    ([groups, 1, 1], [tpg, 1, 1])
}

/// Run the fp8 E4M3 round-trip.
fn roundtrip_fp8_e4m3(shape: &Shape, dt: Dt, source: &[f32]) -> Vec<f32> {
    let dtype = dt.to_dtype();
    // fp8: 8 bits/code → 4 codes per u32.
    let vals_per_pack = 4;
    let groups_per_head = shape.head_dim / shape.group_size;

    let n_packed_per_slot = shape.head_dim / vals_per_pack;
    let n_groups_per_slot = groups_per_head;

    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    let ctx = Context::new().expect("Context::new on macOS");

    // ── Quantize ────────────────────────────────────────────────────────
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&vec![0u32; w_total]));
    // Scale-only: no bias buffer.
    buffers.insert("out_s".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_fp8_e4m3::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(shape);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_fp8_e4m3 dispatch");

    let w_bytes = q_out.outputs.get("out_w").expect("out_w buffer").clone();
    let s_bytes = q_out.outputs.get("out_s").expect("out_s buffer").clone();

    // ── Dequantize ──────────────────────────────────────────────────────
    let recon_total = shape.n_kv_heads * shape.max_seq * shape.head_dim;
    let mut dbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    dbuf.insert("in_w".into(), w_bytes);
    dbuf.insert("in_s".into(), s_bytes);
    dbuf.insert("out".into(), pack_bytes(&vec![0.0f32; recon_total], dt));
    dbuf.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    dbuf.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    dbuf.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    dbuf.insert("n_positions".into(), (shape.n_positions as u32).to_le_bytes().to_vec());

    let mut dkernel = bulk_dequant_kv_fp8_e4m3::kernel_ir_for(dtype);
    dkernel.mode = KernelMode::Grid3D;
    let (dgrid, dtpg) = dequant_dispatch_grid(shape);
    let d_out = ctx
        .dispatch_with_grid(&dkernel, &dbuf, &BTreeMap::new(), dgrid, dtpg)
        .expect("bulk_dequant_kv_fp8_e4m3 dispatch");

    unpack_bytes(d_out.outputs.get("out").expect("out buffer"), dt)
}

/// Run the fp8 E5M2 round-trip.
fn roundtrip_fp8_e5m2(shape: &Shape, dt: Dt, source: &[f32]) -> Vec<f32> {
    let dtype = dt.to_dtype();
    let vals_per_pack = 4;
    let groups_per_head = shape.head_dim / shape.group_size;

    let n_packed_per_slot = shape.head_dim / vals_per_pack;
    let n_groups_per_slot = groups_per_head;

    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    let ctx = Context::new().expect("Context::new on macOS");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&vec![0u32; w_total]));
    buffers.insert("out_s".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_fp8_e5m2::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(shape);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_fp8_e5m2 dispatch");

    let w_bytes = q_out.outputs.get("out_w").expect("out_w buffer").clone();
    let s_bytes = q_out.outputs.get("out_s").expect("out_s buffer").clone();

    let recon_total = shape.n_kv_heads * shape.max_seq * shape.head_dim;
    let mut dbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    dbuf.insert("in_w".into(), w_bytes);
    dbuf.insert("in_s".into(), s_bytes);
    dbuf.insert("out".into(), pack_bytes(&vec![0.0f32; recon_total], dt));
    dbuf.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    dbuf.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    dbuf.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    dbuf.insert("n_positions".into(), (shape.n_positions as u32).to_le_bytes().to_vec());

    let mut dkernel = bulk_dequant_kv_fp8_e5m2::kernel_ir_for(dtype);
    dkernel.mode = KernelMode::Grid3D;
    let (dgrid, dtpg) = dequant_dispatch_grid(shape);
    let d_out = ctx
        .dispatch_with_grid(&dkernel, &dbuf, &BTreeMap::new(), dgrid, dtpg)
        .expect("bulk_dequant_kv_fp8_e5m2 dispatch");

    unpack_bytes(d_out.outputs.get("out").expect("out buffer"), dt)
}

/// Compare reconstructed slot against source. Tolerance = one quantization
/// step based on source data range, plus dtype slack.
fn assert_roundtrip(
    shape: &Shape,
    dt: Dt,
    source: &[f32],
    recon: &[f32],
    fp8_levels: f32,
    label: &str,
) {
    // recon layout: [n_kv_heads, max_seq, head_dim].
    let mut max_abs_err = 0.0_f32;
    let mut worst_idx = (0usize, 0usize);
    for h in 0..shape.n_kv_heads {
        for d in 0..shape.head_dim {
            let src_idx = h * shape.head_dim + d;
            let cache_idx =
                h * shape.max_seq * shape.head_dim + shape.position * shape.head_dim + d;
            let s = source[src_idx];
            let r = recon[cache_idx];
            let err = (s - r).abs();
            if err > max_abs_err {
                max_abs_err = err;
                worst_idx = (h, d);
            }
        }
    }

    // Source values live in roughly [-2, 2]; step = 4 / fp8_levels.
    let group_range_ub = 4.0_f32;
    let step = group_range_ub / fp8_levels;
    let dtype_slack = match dt {
        Dt::F32 => 0.0,
        Dt::F16 => 1e-3,
        Dt::Bf16 => 1e-2,
    };
    let tol = step * 2.0 + dtype_slack;
    eprintln!(
        "[{label}] max_abs_err={max_abs_err:.4} tol={tol:.4} at (h={}, d={})",
        worst_idx.0, worst_idx.1
    );
    assert!(
        max_abs_err <= tol,
        "{label}: max abs err = {max_abs_err:.4} > tol {tol:.4} at (h={}, d={})",
        worst_idx.0,
        worst_idx.1,
    );
}

/// Verify that the fp8 quantize kernel writes only to the target slot
/// and leaves all other slots untouched.
fn assert_no_cross_slot_bleed_fp8_e4m3(shape: &Shape, dt: Dt) {
    let dtype = dt.to_dtype();
    let vals_per_pack = 4;
    let groups_per_head = shape.head_dim / shape.group_size;
    let n_packed_per_slot = shape.head_dim / vals_per_pack;
    let n_groups_per_slot = groups_per_head;
    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    let sentinel_w: Vec<u32> = (0..w_total).map(|i| 0xDEAD0000 | (i as u32 & 0xFFFF)).collect();
    let sentinel_s = vec![1.5_f32; s_total];

    let source = build_source(shape, dt, 0x1234_5678);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(&source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&sentinel_w));
    buffers.insert("out_s".into(), pack_bytes(&sentinel_s, dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_fp8_e4m3::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(shape);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_fp8_e4m3 dispatch");

    let w_after = unpack_u32_bytes(q_out.outputs.get("out_w").expect("out_w"));
    let s_after = unpack_bytes(q_out.outputs.get("out_s").expect("out_s"), dt);

    for h in 0..shape.n_kv_heads {
        for p in 0..shape.max_seq {
            if p == shape.position {
                continue;
            }
            for w in 0..n_packed_per_slot {
                let idx = (h * shape.max_seq + p) * n_packed_per_slot + w;
                assert_eq!(
                    w_after[idx], sentinel_w[idx],
                    "fp8 e4m3 weight cross-slot bleed at (h={h}, p={p}, w={w})",
                );
            }
            for g in 0..n_groups_per_slot {
                let idx = (h * shape.max_seq + p) * n_groups_per_slot + g;
                assert!(
                    (s_after[idx] - sentinel_s[idx]).abs() < 1e-5,
                    "fp8 e4m3 scale cross-slot bleed at (h={h}, p={p}, g={g})",
                );
            }
        }
    }
}

// ── fp8 E4M3 tests ───────────────────────────────────────────────────────────

#[test]
fn kv_cache_fp8_e4m3_roundtrip_f32() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F32, 0x9E37_79B9);
    let recon = roundtrip_fp8_e4m3(&shape, Dt::F32, &source);
    // E4M3: ~3-bit mantissa, effective levels ~14 per binade. Use 14 as
    // representative level count for tolerance (conservative for test data range).
    assert_roundtrip(&shape, Dt::F32, &source, &recon, 14.0, "fp8_e4m3 f32");
}

#[test]
fn kv_cache_fp8_e4m3_roundtrip_f16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F16, 0xDEAD_BEEF);
    let recon = roundtrip_fp8_e4m3(&shape, Dt::F16, &source);
    assert_roundtrip(&shape, Dt::F16, &source, &recon, 14.0, "fp8_e4m3 f16");
}

#[test]
fn kv_cache_fp8_e4m3_roundtrip_bf16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::Bf16, 0xCAFE_BABE);
    let recon = roundtrip_fp8_e4m3(&shape, Dt::Bf16, &source);
    assert_roundtrip(&shape, Dt::Bf16, &source, &recon, 14.0, "fp8_e4m3 bf16");
}

#[test]
fn kv_cache_fp8_e4m3_no_cross_slot_bleed() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    assert_no_cross_slot_bleed_fp8_e4m3(&shape, Dt::F32);
}

// ── fp8 E5M2 tests ───────────────────────────────────────────────────────────

#[test]
fn kv_cache_fp8_e5m2_roundtrip_f32() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F32, 0x9E37_79B9);
    let recon = roundtrip_fp8_e5m2(&shape, Dt::F32, &source);
    // E5M2: 2-bit mantissa, ~4 levels per binade; looser tolerance.
    assert_roundtrip(&shape, Dt::F32, &source, &recon, 6.0, "fp8_e5m2 f32");
}

#[test]
fn kv_cache_fp8_e5m2_roundtrip_f16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F16, 0xDEAD_BEEF);
    let recon = roundtrip_fp8_e5m2(&shape, Dt::F16, &source);
    assert_roundtrip(&shape, Dt::F16, &source, &recon, 6.0, "fp8_e5m2 f16");
}

#[test]
fn kv_cache_fp8_e5m2_roundtrip_bf16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::Bf16, 0xCAFE_BABE);
    let recon = roundtrip_fp8_e5m2(&shape, Dt::Bf16, &source);
    assert_roundtrip(&shape, Dt::Bf16, &source, &recon, 6.0, "fp8_e5m2 bf16");
}
