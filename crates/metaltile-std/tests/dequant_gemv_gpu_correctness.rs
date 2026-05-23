//! GPU correctness for `ffai::dequant_gemv_int{4,8,6}` — dequantizing GEMV
//! kernels used at decode-time output / LM-head projections.
//!
//! Layout (per dtype, with N = `in_dim`, G = `group_size`):
//!
//!   weight  [out_dim, N * bits / 32]   uint32  (bit-packed)
//!   scales  [out_dim, N / G]           T
//!   biases  [out_dim, N / G]           T
//!   input   [N]                        T
//!   output  [out_dim]                  T
//!
//! Per output row: dequantize the row's packed weights via
//! `q * scale[g] + bias[g]` (one (scale, bias) per group), then dot
//! with `input` to produce `output[row]`. Reduction-mode dispatch:
//! one threadgroup per output row, threads cooperate via `reduce_sum`.
//!
//! Coverage gap: before this file the five `dequant_gemv_int{3,4,5,6,8}`
//! kernels (~205 LOC of source) had zero in-tree GPU coverage — like
//! the kv_cache quant kernels, they emit from `macro_rules!` shells
//! (the `#[kernel]` proc-macro doesn't expand inner declarative
//! macros). An empty kernel body or a wrong index formula would
//! produce all-zeros / cross-row corruption that only surfaces as
//! garbage decode in FFAI integration.
//!
//! This file pins int4 (pack-strided, nibble-aligned) + int8 (pack-
//! strided, byte-aligned) + int6 (element-strided, exercises the
//! odd-bit-width spill path) across f32 / f16 / bf16. int3 / int5
//! share the int6 codepath shape; covering int6 catches the same
//! word-spill regression class.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::dequant_gemv::{
    dequant_gemv_int3,
    dequant_gemv_int4,
    dequant_gemv_int4_fast,
    dequant_gemv_int5,
    dequant_gemv_int6,
    dequant_gemv_int8,
};

// ── Quantize helpers ──────────────────────────────────────────────────────

/// Per-group affine quantize a row to `bits`-wide values, packed as a u32
/// bit-stream. For `bits ∈ {4, 8}` (nibble / byte aligned) this is
/// equivalent to the int4/int8 pack-strided format. For `bits = 6`
/// (odd width) values span u32 boundaries — matches the kernel's
/// two-word bit-stream decode.
fn quantize_row(row: &[f32], group_size: usize, bits: u32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    assert_eq!(in_dim % group_size, 0, "in_dim must be a multiple of group_size");
    assert_eq!(
        (in_dim * bits as usize) % 32,
        0,
        "in_dim * bits must be a multiple of 32 (one packed-row u32 boundary)",
    );
    let n_groups = in_dim / group_size;
    let n_u32 = in_dim * bits as usize / 32;
    let mut packed = vec![0u32; n_u32];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    let max_q = (1u32 << bits) - 1;

    for g in 0..n_groups {
        let g_slice = &row[g * group_size..(g + 1) * group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / max_q as f32 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, max_q as f32) as u32;
            let bit_off = ((g * group_size + i) * bits as usize) as u32;
            let word = (bit_off / 32) as usize;
            let in_w = bit_off & 31;
            // Lower fragment lives in `word`; spill (if any) in `word+1`.
            let bits_in_w0 = 32 - in_w;
            if bits_in_w0 >= bits {
                packed[word] |= q << in_w;
            } else {
                packed[word] |= q << in_w;
                packed[word + 1] |= q >> bits_in_w0;
            }
        }
    }
    (packed, scales, biases)
}

/// CPU oracle: per-row dequant + dot with input.
#[allow(clippy::too_many_arguments)]
fn naive_dequant_gemv(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    input: &[f32],
    in_dim: usize,
    group_size: usize,
    bits: u32,
    out_dim: usize,
) -> Vec<f32> {
    let u32_per_row = in_dim * bits as usize / 32;
    let n_groups = in_dim / group_size;
    let max_q_mask: u64 = (1u64 << bits) - 1;
    let mut out = vec![0.0_f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0.0_f32;
        let row_w = &weight[row * u32_per_row..(row + 1) * u32_per_row];
        let row_s = &scales[row * n_groups..(row + 1) * n_groups];
        let row_b = &biases[row * n_groups..(row + 1) * n_groups];
        for (d, &x_d) in input.iter().enumerate().take(in_dim) {
            let g = d / group_size;
            let bit_off = (d * bits as usize) as u32;
            let word = (bit_off / 32) as usize;
            let in_w = bit_off & 31;
            let bits_in_w0 = 32 - in_w;
            let q = if bits_in_w0 >= bits {
                ((row_w[word] as u64) >> in_w) & max_q_mask
            } else {
                let lo_bits = bits_in_w0;
                let spill = bits - lo_bits;
                let lo_mask: u64 = (1u64 << lo_bits) - 1;
                let spill_mask: u64 = (1u64 << spill) - 1;
                let lo = ((row_w[word] as u64) >> in_w) & lo_mask;
                let hi = ((row_w[word + 1] as u64) & spill_mask) << lo_bits;
                lo | hi
            };
            let w_real = (q as f32) * row_s[g] + row_b[g];
            acc += w_real * x_d;
        }
        out[row] = acc;
    }
    out
}

// ── Dispatch helpers ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_dequant_gemv(
    kernel_kind: u32, // 4, 8, or 6
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    input: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_u32_bytes(weight));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("output".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = match kernel_kind {
        3 => dequant_gemv_int3::kernel_ir_for(dt.to_dtype()),
        4 => dequant_gemv_int4::kernel_ir_for(dt.to_dtype()),
        5 => dequant_gemv_int5::kernel_ir_for(dt.to_dtype()),
        6 => dequant_gemv_int6::kernel_ir_for(dt.to_dtype()),
        8 => dequant_gemv_int8::kernel_ir_for(dt.to_dtype()),
        _ => unreachable!("test covers int3 / int4 / int5 / int6 / int8"),
    };
    kernel.mode = KernelMode::Reduction;

    // Reduction dispatch contract (docs/developing.md):
    //   grid=[rows, 1, 1] tg=[TPG, 1, 1], TPG must be ≥ 32 + multiple of 32.
    // 128 lanes per row is a good fit for in_dim=128/256 and provides
    // a healthy `reduce_sum` factor.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim, 1, 1], [128, 1, 1])
        .expect("dequant_gemv dispatch");

    unpack_bytes(result.outputs.get("output").expect("output"), dt)
}

// ── Source generator ──────────────────────────────────────────────────────

/// Per-row source values with non-trivial per-group range so the affine
/// quant has signal to compress.
fn build_source(out_dim: usize, in_dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..out_dim * in_dim)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let raw = ((s as i64 % 20_000) as f32) / 10_000.0;
            let group_offset = (((i / 32) as f32) * 0.7).sin();
            raw + group_offset
        })
        .collect()
}

fn build_input(in_dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..in_dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as i64 % 10_000) as f32) / 10_000.0 - 0.5
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

fn dequantize_full(
    rows: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    bits: u32,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let u32_per_row = in_dim * bits as usize / 32;
    let n_groups = in_dim / group_size;
    let mut weight = Vec::with_capacity(u32_per_row * out_dim);
    let mut scales = Vec::with_capacity(n_groups * out_dim);
    let mut biases = Vec::with_capacity(n_groups * out_dim);
    for row in 0..out_dim {
        let r = &rows[row * in_dim..(row + 1) * in_dim];
        let (w, s, b) = quantize_row(r, group_size, bits);
        weight.extend(w);
        scales.extend(s);
        biases.extend(b);
    }
    (weight, scales, biases)
}

#[allow(clippy::too_many_arguments)]
fn run_one_test(bits: u32, dt: Dt, in_dim: usize, group_size: usize, out_dim: usize, tol: f32) {
    let _g = gpu_lock();
    let seed_w = 0x9E37_79B9 ^ ((bits as u64) << 16);
    let seed_x = 0xDEAD_BEEF ^ ((bits as u64) << 16);
    let rows = build_source(out_dim, in_dim, seed_w);
    // Round inputs through dt so the CPU oracle sees the same precision
    // the kernel does at its load-cast.
    let input_raw = build_input(in_dim, seed_x);
    let input: Vec<f32> = input_raw.iter().map(|&v| dt.round(v)).collect();

    let (weight, scales, biases) = dequantize_full(&rows, out_dim, in_dim, group_size, bits);

    // Round scales/biases through dt for the CPU oracle too — the kernel
    // loads them in T precision.
    let scales_rounded: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_rounded: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected = naive_dequant_gemv(
        &weight,
        &scales_rounded,
        &biases_rounded,
        &input,
        in_dim,
        group_size,
        bits,
        out_dim,
    );
    let actual =
        run_dequant_gemv(bits, &weight, &scales, &biases, &input, dt, in_dim, group_size, out_dim);

    assert_eq!(actual.len(), out_dim, "output length mismatch");
    let mut max_rel = 0.0_f32;
    let mut worst_row = 0usize;
    for (row, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
            worst_row = row;
        }
    }
    assert!(
        max_rel <= tol,
        "bits={bits} dt={:?} in_dim={in_dim} out_dim={out_dim}: max rel = {max_rel:.3e} > {tol:.3e} at row {worst_row}",
        dt as u32,
    );
}

#[test]
fn dequant_gemv_int4_qwen_shape_f32() {
    // Qwen-class in_dim=5120/4096 truncated to fit single TG; group_size 32.
    run_one_test(4, Dt::F32, 256, 32, 4, 5e-3);
}

#[test]
fn dequant_gemv_int4_qwen_shape_f16() { run_one_test(4, Dt::F16, 256, 32, 4, 1e-2); }

#[test]
fn dequant_gemv_int4_qwen_shape_bf16() { run_one_test(4, Dt::Bf16, 256, 32, 4, 3e-2); }

#[test]
fn dequant_gemv_int8_qwen_shape_f32() { run_one_test(8, Dt::F32, 256, 32, 4, 5e-4); }

#[test]
fn dequant_gemv_int8_qwen_shape_f16() { run_one_test(8, Dt::F16, 256, 32, 4, 5e-3); }

#[test]
fn dequant_gemv_int8_qwen_shape_bf16() { run_one_test(8, Dt::Bf16, 256, 32, 4, 3e-2); }

#[test]
fn dequant_gemv_int6_word_spill_path_f32() {
    // int6 is the odd-bit-width family — values straddle u32 boundaries.
    // in_dim * 6 must be a multiple of 32 so the packed-row stays u32-
    // aligned; 64 * 6 = 384 = 12 u32. Catches the bit-stream `lo | hi`
    // decode regression class.
    run_one_test(6, Dt::F32, 64, 32, 4, 5e-3);
}

#[test]
fn dequant_gemv_int6_word_spill_path_f16() { run_one_test(6, Dt::F16, 64, 32, 4, 1e-2); }

// ── int3 / int5 odd-width pin (BenchSpec registration + bit-stream decode) ──
//
// int3 and int5 share the same element-strided word-spill codepath as int6,
// but each registers its own `BenchSpec` (kernel_name, kernel_ir reference).
// A registration regression — e.g. a typo in `kernel_name` or pointing
// `kernel_ir` at the wrong function — wouldn't surface from the int6 test.
// One cell each pins the registration surface and exercises the same
// bit-stream `lo | hi` decode at a different bit-width parameter.
//
// Shape constraints:
//   - int3: in_dim * 3 must be a u32-aligned bit count → 64 * 3 = 192 = 6 u32
//   - int5: in_dim * 5 must be a u32-aligned bit count → 64 * 5 = 320 = 10 u32
// Both use group_size=32 (in_dim / group_size = 2 groups per row).

#[test]
fn dequant_gemv_int3_word_spill_path_f32() {
    // int3: quant step = 4 / 7 ≈ 0.57; in_dim=64 dot → tolerance widens
    // vs higher bit-widths (only 7 levels of quantization).
    run_one_test(3, Dt::F32, 64, 32, 4, 2e-2);
}

#[test]
fn dequant_gemv_int5_word_spill_path_f32() {
    // int5: quant step = 4 / 31 ≈ 0.129; tighter than int3 + int6.
    run_one_test(5, Dt::F32, 64, 32, 4, 8e-3);
}

// ── dequant_gemv_int4_fast ─────────────────────────────────────────────────
//
// 8-row-per-TG fast variant: `in_dim` must be a multiple of 512;
// `out_dim` must be a multiple of 8; `group_size` must be 64.
// Grid: [out_dim/8, 1, 1]; TPG = 64.

#[allow(clippy::too_many_arguments)]
fn run_dequant_gemv_int4_fast(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    input: &[f32],
    dt: Dt,
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_u32_bytes(weight));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("output".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = dequant_gemv_int4_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Fast variant: grid=[out_dim/8, 1, 1], TPG=64.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim / 8, 1, 1], [64, 1, 1])
        .expect("dequant_gemv_int4_fast dispatch");
    unpack_bytes(result.outputs.get("output").expect("output"), dt)
}

fn run_one_test_fast(dt: Dt, in_dim: usize, group_size: usize, out_dim: usize, tol: f32) {
    assert_eq!(in_dim % 512, 0, "fast variant requires in_dim % 512 == 0");
    assert_eq!(out_dim % 8, 0, "fast variant requires out_dim % 8 == 0");
    assert_eq!(group_size, 64, "fast variant requires group_size == 64");

    let _g = gpu_lock();
    let rows = build_source(out_dim, in_dim, 0x9E37_79B9 ^ (4u64 << 16));
    let input_raw = build_input(in_dim, 0xDEAD_BEEF ^ (4u64 << 16));
    let input: Vec<f32> = input_raw.iter().map(|&v| dt.round(v)).collect();

    let (weight, scales, biases) = dequantize_full(&rows, out_dim, in_dim, group_size, 4);

    let scales_rounded: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_rounded: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected = naive_dequant_gemv(
        &weight,
        &scales_rounded,
        &biases_rounded,
        &input,
        in_dim,
        group_size,
        4,
        out_dim,
    );
    let actual = run_dequant_gemv_int4_fast(
        &weight, &scales, &biases, &input, dt, in_dim, group_size, out_dim,
    );

    assert_eq!(actual.len(), out_dim, "output length mismatch");
    let mut max_rel = 0.0_f32;
    let mut worst_row = 0usize;
    for (row, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
            worst_row = row;
        }
    }
    assert!(
        max_rel <= tol,
        "int4_fast dt={:?} in_dim={in_dim} out_dim={out_dim}: max rel = {max_rel:.3e} > {tol:.3e} at row {worst_row}",
        dt as u32,
    );
}

#[test]
fn dequant_gemv_int4_fast_f32() { run_one_test_fast(Dt::F32, 512, 64, 8, 5e-3); }

#[test]
fn dequant_gemv_int4_fast_f16() { run_one_test_fast(Dt::F16, 512, 64, 8, 1e-2); }

#[test]
fn dequant_gemv_int4_fast_bf16() { run_one_test_fast(Dt::Bf16, 512, 64, 8, 3e-2); }

#[test]
fn dequant_gemv_int4_fast_f32_large() {
    // Larger shape exercises multiple blocks per row.
    run_one_test_fast(Dt::F32, 1024, 64, 16, 5e-3);
}
