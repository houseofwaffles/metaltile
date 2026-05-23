//! GPU correctness for `ffai::flash_quantized_sdpa` — single-pass SDPA
//! over an affine-quantized K/V cache.
//!
//! Pins, against a naive f32 oracle: (1) per-group affine dequant of K
//! and V (`scale·q + bias`); (2) the query pre-scale; (3) attention
//! sinks and (4) the sliding-window mask. Covers bits ∈ {4, 8}.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::ffai::flash_quantized_sdpa::{
    flash_quantized_sdpa_b4_d96,
    flash_quantized_sdpa_b4_d128,
    flash_quantized_sdpa_b4_d512,
    flash_quantized_sdpa_b8_d96,
    flash_quantized_sdpa_b8_d128,
    flash_quantized_sdpa_b8_d512,
};

/// Affine per-group quantize of a `[rows, dim]` tensor. Returns packed
/// u32 (pack-strided, `32/bits` values per word), scales, biases, and
/// the dequantized float values (what the kernel effectively sees).
fn quantize(
    vals: &[f32],
    rows: usize,
    dim: usize,
    group_size: usize,
    bits: u32,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let pack_factor = 32 / bits as usize;
    let n_groups = dim / group_size;
    let max_q = ((1u32 << bits) - 1) as f32;
    let mut packed = vec![0u32; rows * dim / pack_factor];
    let mut scales = vec![0.0_f32; rows * n_groups];
    let mut biases = vec![0.0_f32; rows * n_groups];
    let mut deq = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        for g in 0..n_groups {
            let slice = &vals[r * dim + g * group_size..r * dim + (g + 1) * group_size];
            let mn = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / max_q };
            scales[r * n_groups + g] = scale;
            biases[r * n_groups + g] = mn;
            for (i, &v) in slice.iter().enumerate() {
                let d = g * group_size + i;
                let q = ((v - mn) / scale).round().clamp(0.0, max_q) as u32;
                packed[(r * dim + d) / pack_factor] |= q << ((d % pack_factor) * bits as usize);
                deq[r * dim + d] = scale * q as f32 + mn;
            }
        }
    }
    (packed, scales, biases, deq)
}

#[allow(clippy::too_many_arguments)]
fn naive(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
    sinks: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    scale: f32,
    has_sinks: bool,
    window: usize,
    num_q_heads: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let causal_upper = tokens - 1;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for t in 0..tokens {
            if window == 0 || t + window > causal_upper {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += scale * q[qh * dim + d] * k_deq[(kvh * tokens + t) * dim + d];
                }
                kept.push((t, dot));
            }
        }
        let mut m = if has_sinks { sinks[qh % num_q_heads] } else { f32::NEG_INFINITY };
        for &(_, s) in &kept {
            m = m.max(s);
        }
        let mut sum_w = if has_sinks { (sinks[qh % num_q_heads] - m).exp() } else { 0.0 };
        let mut acc = vec![0.0_f32; dim];
        for &(t, s) in &kept {
            let w = (s - m).exp();
            sum_w += w;
            for (d, a) in acc.iter_mut().enumerate() {
                *a += w * v_deq[(kvh * tokens + t) * dim + d];
            }
        }
        let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
        for d in 0..dim {
            out[qh * dim + d] = acc[d] * inv;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run(
    kernel_ir: fn(DType) -> Kernel,
    q: &[f32],
    k_packed: &[u32],
    k_scales: &[f32],
    k_biases: &[f32],
    v_packed: &[u32],
    v_scales: &[f32],
    v_biases: &[f32],
    sinks: &[f32],
    dt: Dt,
    q_heads: usize,
    dim: usize,
    tokens: usize,
    repeat: usize,
    group_size: usize,
    num_q_heads: usize,
    has_sinks: u32,
    window: u32,
    scale: f32,
) -> Vec<f32> {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("queries".into(), pack_bytes(q, dt));
    b.insert("k_packed".into(), pack_u32_bytes(k_packed));
    b.insert("k_scales".into(), pack_bytes(k_scales, dt));
    b.insert("k_biases".into(), pack_bytes(k_biases, dt));
    b.insert("v_packed".into(), pack_u32_bytes(v_packed));
    b.insert("v_scales".into(), pack_bytes(v_scales, dt));
    b.insert("v_biases".into(), pack_bytes(v_biases, dt));
    b.insert("sinks".into(), pack_bytes(sinks, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0_f32; q_heads * dim], dt));
    for (k, v) in [
        ("dim", dim as u32),
        ("tokens", tokens as u32),
        ("repeat_count", repeat as u32),
        ("group_size", group_size as u32),
        ("num_q_heads", num_q_heads as u32),
        ("has_sinks", has_sinks),
        ("window_size", window),
    ] {
        b.insert(k.into(), v.to_le_bytes().to_vec());
    }
    b.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, q_heads, 1], [32, 1, 1])
        .expect("flash_quantized_sdpa dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(q_heads * dim);
    out
}

fn source(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_000) as f32 / 20_000.0 * scale - scale * 0.5
        })
        .collect()
}

fn check_dim(
    kernel_ir: fn(DType) -> Kernel,
    bits: u32,
    dt: Dt,
    has_sinks: bool,
    window: usize,
    dim: usize,
    tol: f32,
) {
    let _g = gpu_lock();
    let (q_heads, kv_heads, tokens) = (2usize, 1usize, 8usize);
    // group_size must divide dim; use 32 for d=96 (96 % 32 == 0) and
    // d=512 (512 % 32 == 0), 64 for the original d=128 / d=256.
    let group_size = if dim.is_multiple_of(64) { 64usize } else { 32usize };
    let attn_scale = 1.0_f32 / (dim as f32).sqrt();
    let q: Vec<f32> = source(q_heads * dim, 0x51, 2.0).iter().map(|&v| dt.round(v)).collect();
    let k_raw = source(kv_heads * tokens * dim, 0x62, 3.0);
    let v_raw = source(kv_heads * tokens * dim, 0x73, 3.0);
    let sinks: Vec<f32> = (0..q_heads).map(|i| -0.4 + 0.3 * i as f32).collect();

    let (kp, ks, kb, k_deq) = quantize(&k_raw, kv_heads * tokens, dim, group_size, bits);
    let (vp, vs, vb, v_deq) = quantize(&v_raw, kv_heads * tokens, dim, group_size, bits);
    let round = |x: &[f32]| x.iter().map(|&v| dt.round(v)).collect::<Vec<_>>();

    let expected = naive(
        &round(&q),
        &k_deq,
        &v_deq,
        &sinks,
        q_heads,
        kv_heads,
        tokens,
        dim,
        attn_scale,
        has_sinks,
        window,
        q_heads,
    );
    let actual = run(
        kernel_ir,
        &q,
        &kp,
        &ks,
        &kb,
        &vp,
        &vs,
        &vb,
        &sinks,
        dt,
        q_heads,
        dim,
        tokens,
        q_heads / kv_heads,
        group_size,
        q_heads,
        u32::from(has_sinks),
        window as u32,
        attn_scale,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "bits={bits} dt={:?} sinks={has_sinks} window={window}: max |diff| = {diff:.2e}",
        dt as u32
    );
}

/// Convenience wrapper: run `check_dim` with the original d=128.
fn check(
    kernel_ir: fn(DType) -> Kernel,
    bits: u32,
    dt: Dt,
    has_sinks: bool,
    window: usize,
    tol: f32,
) {
    check_dim(kernel_ir, bits, dt, has_sinks, window, 128, tol);
}

#[test]
fn flash_quantized_sdpa_b4_plain_f32() {
    check(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, Dt::F32, false, 0, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_sinks_f32() {
    check(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, Dt::F32, true, 0, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_window_f32() {
    check(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, Dt::F32, false, 4, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b8_plain_f32() {
    check(flash_quantized_sdpa_b8_d128::kernel_ir_for, 8, Dt::F32, false, 0, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_plain_bf16() {
    check(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, Dt::Bf16, false, 0, 5e-2);
}

#[test]
fn flash_quantized_sdpa_b8_sinks_bf16() {
    check(flash_quantized_sdpa_b8_d128::kernel_ir_for, 8, Dt::Bf16, true, 0, 5e-2);
}

// ── d=96 (GPT-NeoX) ────────────────────────────────────────────────────────

#[test]
fn flash_quantized_sdpa_b4_d96_plain_f32() {
    check_dim(flash_quantized_sdpa_b4_d96::kernel_ir_for, 4, Dt::F32, false, 0, 96, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_d96_sinks_f32() {
    check_dim(flash_quantized_sdpa_b4_d96::kernel_ir_for, 4, Dt::F32, true, 0, 96, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b8_d96_plain_f32() {
    check_dim(flash_quantized_sdpa_b8_d96::kernel_ir_for, 8, Dt::F32, false, 0, 96, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_d96_plain_f16() {
    check_dim(flash_quantized_sdpa_b4_d96::kernel_ir_for, 4, Dt::F16, false, 0, 96, 5e-3);
}

#[test]
fn flash_quantized_sdpa_b8_d96_plain_bf16() {
    check_dim(flash_quantized_sdpa_b8_d96::kernel_ir_for, 8, Dt::Bf16, false, 0, 96, 5e-2);
}

// ── d=512 (Gemma 4 global attention) ─────────────────────────────────────────
// Dispatch uses 256 threads/TG (see kernel comments). The grid shape is the
// same as d=64/d=128 — [1, q_heads, 1] — but tg=[32,1,1]. The `run` helper
// handles this correctly since it uses `[32, 1, 1]` threads_per_group.

#[test]
fn flash_quantized_sdpa_b4_d512_plain_f32() {
    check_dim(flash_quantized_sdpa_b4_d512::kernel_ir_for, 4, Dt::F32, false, 0, 512, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_d512_sinks_f32() {
    check_dim(flash_quantized_sdpa_b4_d512::kernel_ir_for, 4, Dt::F32, true, 0, 512, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b8_d512_plain_f32() {
    check_dim(flash_quantized_sdpa_b8_d512::kernel_ir_for, 8, Dt::F32, false, 0, 512, 1e-4);
}

#[test]
fn flash_quantized_sdpa_b4_d512_plain_f16() {
    check_dim(flash_quantized_sdpa_b4_d512::kernel_ir_for, 4, Dt::F16, false, 0, 512, 5e-3);
}

#[test]
fn flash_quantized_sdpa_b8_d512_plain_bf16() {
    check_dim(flash_quantized_sdpa_b8_d512::kernel_ir_for, 8, Dt::Bf16, false, 0, 512, 5e-2);
}
