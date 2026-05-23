//! GPU correctness for `flash_quantized_sdpa` bool-mask and float-mask variants.
//!
//! Strategy:
//!   * **Bool mask** — build a binary mask over `[q_heads, tokens]`. The CPU
//!     oracle skips any token whose mask slot is 0. The GPU kernel must produce
//!     the same output as running the base kernel over only the visible tokens.
//!
//!   * **Float mask** — build a per-token logit bias over `[q_heads, tokens]`.
//!     The CPU oracle adds `bias[q_head * tokens + t]` to each dot-product
//!     before the softmax step. The GPU kernel must match within f32 noise.
//!
//! Covers bits ∈ {4, 8} × dtypes ∈ {F32, BF16} for the representative d=128
//! shapes. Window and sinks disabled to isolate the mask path.
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
    flash_quantized_sdpa_bool_mask_b4_d128,
    flash_quantized_sdpa_bool_mask_b8_d128,
    flash_quantized_sdpa_float_mask_b4_d128,
    flash_quantized_sdpa_float_mask_b8_d128,
};

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Affine per-group quantize of a `[rows, dim]` tensor. Returns packed u32
/// (pack-strided, `32/bits` values per word), scales, biases, and the
/// dequantized float values that the kernel will effectively see.
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

/// Simple LCG-based float source with configurable seed and amplitude.
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

// ── Bool-mask CPU oracle ──────────────────────────────────────────────────────

/// CPU oracle for the bool-mask kernel. Each `(q_head, token)` pair is
/// included only when `mask[q_head * tokens + token] != 0`. The causal and
/// window guards are disabled (window == 0 means all tokens pass the built-in
/// gate, which is always `t < tokens` — i.e. all tokens).
#[allow(clippy::too_many_arguments)]
fn naive_bool_mask(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
    sinks: &[f32],
    mask: &[u32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    scale: f32,
    has_sinks: bool,
    num_q_heads: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        // Collect (token_index, dot_product) for visible tokens.
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for t in 0..tokens {
            // window == 0 means use_key is always true (t < tokens is always true).
            let mask_pass = mask[qh * tokens + t] != 0;
            if mask_pass {
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

// ── Float-mask CPU oracle ─────────────────────────────────────────────────────

/// CPU oracle for the float-mask kernel. The bias `mask_float[q_head * tokens + t]`
/// is added to the raw dot-product score before the softmax step. No bool
/// filtering — all tokens are visible (window == 0).
#[allow(clippy::too_many_arguments)]
fn naive_float_mask(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
    sinks: &[f32],
    mask_float: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    scale: f32,
    has_sinks: bool,
    num_q_heads: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for t in 0..tokens {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                dot += scale * q[qh * dim + d] * k_deq[(kvh * tokens + t) * dim + d];
            }
            // Add per-token logit bias.
            let biased = dot + mask_float[qh * tokens + t];
            kept.push((t, biased));
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

// ── GPU dispatch helpers ──────────────────────────────────────────────────────

/// Shared buffer builder for the base flash_quantized_sdpa fields.
#[allow(clippy::too_many_arguments)]
fn base_buffers(
    q: &[f32],
    k_packed: &[u32],
    k_scales: &[f32],
    k_biases: &[f32],
    v_packed: &[u32],
    v_scales: &[f32],
    v_biases: &[f32],
    sinks: &[f32],
    q_heads: usize,
    dim: usize,
    tokens: usize,
    repeat: usize,
    group_size: usize,
    num_q_heads: usize,
    has_sinks: u32,
    window: u32,
    scale: f32,
    dt: Dt,
) -> BTreeMap<String, Vec<u8>> {
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
    b
}

/// Dispatch the bool-mask variant and return f32 outputs.
#[allow(clippy::too_many_arguments)]
fn run_bool_mask(
    kernel_ir: fn(DType) -> Kernel,
    q: &[f32],
    k_packed: &[u32],
    k_scales: &[f32],
    k_biases: &[f32],
    v_packed: &[u32],
    v_scales: &[f32],
    v_biases: &[f32],
    sinks: &[f32],
    mask_bool: &[u32],
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
    let mut b = base_buffers(
        q,
        k_packed,
        k_scales,
        k_biases,
        v_packed,
        v_scales,
        v_biases,
        sinks,
        q_heads,
        dim,
        tokens,
        repeat,
        group_size,
        num_q_heads,
        has_sinks,
        window,
        scale,
        dt,
    );
    b.insert("mask_bool".into(), pack_u32_bytes(mask_bool));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, q_heads, 1], [32, 1, 1])
        .expect("flash_quantized_sdpa_bool_mask dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(q_heads * dim);
    out
}

/// Dispatch the float-mask variant and return f32 outputs.
#[allow(clippy::too_many_arguments)]
fn run_float_mask(
    kernel_ir: fn(DType) -> Kernel,
    q: &[f32],
    k_packed: &[u32],
    k_scales: &[f32],
    k_biases: &[f32],
    v_packed: &[u32],
    v_scales: &[f32],
    v_biases: &[f32],
    sinks: &[f32],
    mask_float: &[f32],
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
    let mut b = base_buffers(
        q,
        k_packed,
        k_scales,
        k_biases,
        v_packed,
        v_scales,
        v_biases,
        sinks,
        q_heads,
        dim,
        tokens,
        repeat,
        group_size,
        num_q_heads,
        has_sinks,
        window,
        scale,
        dt,
    );
    b.insert("mask_float".into(), pack_bytes(mask_float, dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, q_heads, 1], [32, 1, 1])
        .expect("flash_quantized_sdpa_float_mask dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(q_heads * dim);
    out
}

// ── Test harness ──────────────────────────────────────────────────────────────

/// Shared test data used by all bool-mask and float-mask tests below.
struct TestData {
    q: Vec<f32>,
    k_packed: Vec<u32>,
    k_scales: Vec<f32>,
    k_biases: Vec<f32>,
    v_packed: Vec<u32>,
    v_scales: Vec<f32>,
    v_biases: Vec<f32>,
    k_deq: Vec<f32>,
    v_deq: Vec<f32>,
    sinks: Vec<f32>,
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    group_size: usize,
    attn_scale: f32,
    num_q_heads: usize,
}

impl TestData {
    fn build(bits: u32, dt: Dt) -> Self {
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let tokens = 16usize;
        let dim = 128usize;
        let group_size = 64usize;
        let attn_scale = 0.125_f32;
        let num_q_heads = q_heads;

        let q_raw = source(q_heads * dim, 0xA1, 2.0);
        let k_raw = source(kv_heads * tokens * dim, 0xB2, 3.0);
        let v_raw = source(kv_heads * tokens * dim, 0xC3, 3.0);
        let sinks: Vec<f32> = (0..q_heads).map(|i| -0.4 + 0.3 * i as f32).collect();

        let q: Vec<f32> = q_raw.iter().map(|&v| dt.round(v)).collect();
        let (k_packed, k_scales, k_biases, k_deq) =
            quantize(&k_raw, kv_heads * tokens, dim, group_size, bits);
        let (v_packed, v_scales, v_biases, v_deq) =
            quantize(&v_raw, kv_heads * tokens, dim, group_size, bits);

        Self {
            q,
            k_packed,
            k_scales,
            k_biases,
            v_packed,
            v_scales,
            v_biases,
            k_deq,
            v_deq,
            sinks,
            q_heads,
            kv_heads,
            tokens,
            dim,
            group_size,
            attn_scale,
            num_q_heads,
        }
    }
}

// ── Bool-mask tests ───────────────────────────────────────────────────────────

/// Run a single bool-mask correctness check.
fn check_bool_mask(kernel_ir: fn(DType) -> Kernel, bits: u32, dt: Dt, tol: f32) {
    let _g = gpu_lock();
    let td = TestData::build(bits, dt);

    // Build a checkerboard mask: even tokens visible, odd tokens hidden.
    let mask_bool: Vec<u32> =
        (0..td.q_heads * td.tokens).map(|idx| if idx % 2 == 0 { 1u32 } else { 0u32 }).collect();

    let expected = naive_bool_mask(
        &td.q,
        &td.k_deq,
        &td.v_deq,
        &td.sinks,
        &mask_bool,
        td.q_heads,
        td.kv_heads,
        td.tokens,
        td.dim,
        td.attn_scale,
        false, // has_sinks disabled to isolate the mask
        td.num_q_heads,
    );

    let actual = run_bool_mask(
        kernel_ir,
        &td.q,
        &td.k_packed,
        &td.k_scales,
        &td.k_biases,
        &td.v_packed,
        &td.v_scales,
        &td.v_biases,
        &td.sinks,
        &mask_bool,
        dt,
        td.q_heads,
        td.dim,
        td.tokens,
        td.q_heads / td.kv_heads,
        td.group_size,
        td.num_q_heads,
        0, // has_sinks = 0
        0, // window_size = 0 (full causal — all pass use_key)
        td.attn_scale,
    );

    assert!(actual.iter().any(|&v| v != 0.0), "bool-mask output is all zeros");
    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "bool_mask bits={bits} dt={:?}: max |diff| = {diff:.2e} (tol {tol:.2e})",
        dt as u32,
    );
}

#[test]
fn flash_quantized_sdpa_bool_mask_b4_f32() {
    check_bool_mask(flash_quantized_sdpa_bool_mask_b4_d128::kernel_ir_for, 4, Dt::F32, 1e-4);
}

#[test]
fn flash_quantized_sdpa_bool_mask_b8_f32() {
    check_bool_mask(flash_quantized_sdpa_bool_mask_b8_d128::kernel_ir_for, 8, Dt::F32, 1e-4);
}

#[test]
fn flash_quantized_sdpa_bool_mask_b4_bf16() {
    check_bool_mask(flash_quantized_sdpa_bool_mask_b4_d128::kernel_ir_for, 4, Dt::Bf16, 5e-2);
}

#[test]
fn flash_quantized_sdpa_bool_mask_b8_bf16() {
    check_bool_mask(flash_quantized_sdpa_bool_mask_b8_d128::kernel_ir_for, 8, Dt::Bf16, 5e-2);
}

/// Verify that a fully-open bool mask (all 1s) produces the same output as the
/// base kernel (no masking) — regression guard for the mask gate logic.
#[test]
fn flash_quantized_sdpa_bool_mask_all_visible_matches_base() {
    let _g = gpu_lock();
    let bits = 8u32;
    let dt = Dt::F32;
    let td = TestData::build(bits, dt);
    // All-ones mask — every token should be visible.
    let mask_bool_all: Vec<u32> = vec![1u32; td.q_heads * td.tokens];
    // All-tokens-visible oracle (no bool filter).
    let expected = naive_bool_mask(
        &td.q,
        &td.k_deq,
        &td.v_deq,
        &td.sinks,
        &mask_bool_all,
        td.q_heads,
        td.kv_heads,
        td.tokens,
        td.dim,
        td.attn_scale,
        false,
        td.num_q_heads,
    );
    let actual = run_bool_mask(
        flash_quantized_sdpa_bool_mask_b8_d128::kernel_ir_for,
        &td.q,
        &td.k_packed,
        &td.k_scales,
        &td.k_biases,
        &td.v_packed,
        &td.v_scales,
        &td.v_biases,
        &td.sinks,
        &mask_bool_all,
        dt,
        td.q_heads,
        td.dim,
        td.tokens,
        td.q_heads / td.kv_heads,
        td.group_size,
        td.num_q_heads,
        0,
        0,
        td.attn_scale,
    );
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "all-visible bool mask: max |diff| = {diff:.2e}");
}

// ── Float-mask tests ──────────────────────────────────────────────────────────

/// Run a single float-mask correctness check.
fn check_float_mask(kernel_ir: fn(DType) -> Kernel, bits: u32, dt: Dt, tol: f32) {
    let _g = gpu_lock();
    let td = TestData::build(bits, dt);

    // Build a float bias: alternating small positive/negative values so the
    // mask non-trivially shifts the softmax distribution.
    let mask_float_raw: Vec<f32> = (0..td.q_heads * td.tokens)
        .map(|idx| {
            let v = (idx as f32 * 0.13 - 1.0).sin() * 2.0;
            dt.round(v)
        })
        .collect();

    let expected = naive_float_mask(
        &td.q,
        &td.k_deq,
        &td.v_deq,
        &td.sinks,
        &mask_float_raw,
        td.q_heads,
        td.kv_heads,
        td.tokens,
        td.dim,
        td.attn_scale,
        false,
        td.num_q_heads,
    );

    let actual = run_float_mask(
        kernel_ir,
        &td.q,
        &td.k_packed,
        &td.k_scales,
        &td.k_biases,
        &td.v_packed,
        &td.v_scales,
        &td.v_biases,
        &td.sinks,
        &mask_float_raw,
        dt,
        td.q_heads,
        td.dim,
        td.tokens,
        td.q_heads / td.kv_heads,
        td.group_size,
        td.num_q_heads,
        0,
        0,
        td.attn_scale,
    );

    assert!(actual.iter().any(|&v| v != 0.0), "float-mask output is all zeros");
    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "float_mask bits={bits} dt={:?}: max |diff| = {diff:.2e} (tol {tol:.2e})",
        dt as u32,
    );
}

#[test]
fn flash_quantized_sdpa_float_mask_b4_f32() {
    check_float_mask(flash_quantized_sdpa_float_mask_b4_d128::kernel_ir_for, 4, Dt::F32, 1e-4);
}

#[test]
fn flash_quantized_sdpa_float_mask_b8_f32() {
    check_float_mask(flash_quantized_sdpa_float_mask_b8_d128::kernel_ir_for, 8, Dt::F32, 1e-4);
}

#[test]
fn flash_quantized_sdpa_float_mask_b4_bf16() {
    check_float_mask(flash_quantized_sdpa_float_mask_b4_d128::kernel_ir_for, 4, Dt::Bf16, 5e-2);
}

#[test]
fn flash_quantized_sdpa_float_mask_b8_bf16() {
    check_float_mask(flash_quantized_sdpa_float_mask_b8_d128::kernel_ir_for, 8, Dt::Bf16, 5e-2);
}

/// Verify that a zero-bias float mask produces the same output as no mask.
/// This guards against the float bias being accumulated incorrectly (e.g.
/// applied multiple times or on wrong lanes).
#[test]
fn flash_quantized_sdpa_float_mask_zero_bias_matches_base() {
    let _g = gpu_lock();
    let bits = 8u32;
    let dt = Dt::F32;
    let td = TestData::build(bits, dt);

    let mask_float_zero: Vec<f32> = vec![0.0_f32; td.q_heads * td.tokens];

    let expected = naive_float_mask(
        &td.q,
        &td.k_deq,
        &td.v_deq,
        &td.sinks,
        &mask_float_zero,
        td.q_heads,
        td.kv_heads,
        td.tokens,
        td.dim,
        td.attn_scale,
        false,
        td.num_q_heads,
    );

    let actual = run_float_mask(
        flash_quantized_sdpa_float_mask_b8_d128::kernel_ir_for,
        &td.q,
        &td.k_packed,
        &td.k_scales,
        &td.k_biases,
        &td.v_packed,
        &td.v_scales,
        &td.v_biases,
        &td.sinks,
        &mask_float_zero,
        dt,
        td.q_heads,
        td.dim,
        td.tokens,
        td.q_heads / td.kv_heads,
        td.group_size,
        td.num_q_heads,
        0,
        0,
        td.attn_scale,
    );

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "zero-bias float mask: max |diff| = {diff:.2e}");
}
