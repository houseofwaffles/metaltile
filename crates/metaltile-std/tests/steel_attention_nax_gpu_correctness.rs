//! GPU correctness oracle for `mt_sdpa_prefill_nax` — flash attention
//! backed by `mpp::tensor_ops::matmul2d` (NAX path).
//!
//! Dispatches `mt_sdpa_prefill_nax_{f32,f16}` over a small set of
//! shapes (single Q-tile + multi-tile / multi-head) and validates
//! against a naive causal-softmax CPU oracle. Requires macOS 26+ /
//! Metal 4 — the kernel includes
//! `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls
//! `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`. On
//! pre-Metal-4 toolchains the kernel emits a single-scalar fallback so
//! the metallib still links; this test then fails the correctness
//! check, which is the intended signal.
//!
//! `head_dim` is fixed at 32 —
//! the NAX QK descriptor's K-dim.
//!
//! Run:
//!   cargo test --release -p metaltile-std --test steel_attention_nax_gpu_correctness -- --nocapture

#![cfg(target_os = "macos")]
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::attn::steel_attention_nax;

/// Head dimension — fixed by the NAX QK descriptor (K-dim = 32).
const HEAD_DIM: usize = 32;

/// Naive causal SDPA oracle — single batch.
///   q, out : [n_q_heads,  q_len, head_dim]
///   k, v   : [n_kv_heads, k_len, head_dim]
/// Causal mask: query at absolute position `q_off + qi` attends keys
/// `0..=q_abs` where `q_abs = (k_len - q_len) + qi`.
#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    k_len: usize,
    gqa_factor: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let hd = HEAD_DIM;
    let q_len_off = k_len - q_len;
    let mut out = vec![0.0f32; n_q_heads * q_len * hd];
    for qh in 0..n_q_heads {
        let kvh = qh / gqa_factor;
        assert!(kvh < n_kv_heads);
        for qi in 0..q_len {
            let q_abs = q_len_off + qi;
            let q_base = qh * q_len * hd + qi * hd;
            // Scores over all keys, causally masked.
            let mut scores = vec![f32::NEG_INFINITY; k_len];
            let mut row_max = f32::NEG_INFINITY;
            for ki in 0..=q_abs.min(k_len - 1) {
                let k_base = kvh * k_len * hd + ki * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_base + d] * k[k_base + d];
                }
                let s = dot * scale;
                scores[ki] = s;
                if s > row_max {
                    row_max = s;
                }
            }
            // Softmax + weighted V.
            let mut sum = 0.0f32;
            for ki in 0..k_len {
                if scores[ki] > f32::NEG_INFINITY {
                    let e = (scores[ki] - row_max).exp();
                    scores[ki] = e;
                    sum += e;
                } else {
                    scores[ki] = 0.0;
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..k_len {
                    if scores[ki] != 0.0 {
                        let k_base = kvh * k_len * hd + ki * hd;
                        acc += scores[ki] * v[k_base + d];
                    }
                }
                out[q_base + d] = acc * inv;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_nax(
    ctx: &Context,
    dtype: DType,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    q_len: usize,
    k_len: usize,
    gqa_factor: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    scale: f32,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(q_len.is_multiple_of(16), "mt_sdpa_prefill_nax requires q_len % 16 == 0");
    assert!(k_len.is_multiple_of(16), "mt_sdpa_prefill_nax requires k_len % 16 == 0");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), q_bytes.to_vec());
    buffers.insert("k".into(), k_bytes.to_vec());
    buffers.insert("v".into(), v_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; n_q_heads * q_len * HEAD_DIM * out_bytes_per_elem]);
    buffers.insert("q_len".into(), (q_len as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (k_len as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let mut kernel = steel_attention_nax::mt_sdpa_prefill_nax::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [q_len / 16, n_q_heads, 1], [
            32, 1, 1,
        ])
        .expect("dispatch mt_sdpa_prefill_nax");

    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
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

/// Deterministic small-magnitude Q / K / V inputs — keep values small so
/// the f16 path stays well inside dynamic range and the softmax is
/// numerically tame.
fn build_attn_inputs(
    q_len: usize,
    k_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = HEAD_DIM;
    let q: Vec<f32> =
        (0..n_q_heads * q_len * hd).map(|i| 0.01 + (i as f32 % 23.0) * 0.007).collect();
    let k: Vec<f32> =
        (0..n_kv_heads * k_len * hd).map(|i| -0.02 + (i as f32 % 19.0) * 0.006).collect();
    let v: Vec<f32> =
        (0..n_kv_heads * k_len * hd).map(|i| 0.03 + (i as f32 % 29.0) * 0.005).collect();
    (q, k, v)
}

// ── Shape 1 : single Q-tile, single head ───────────────────────────────────

#[test]
fn mt_sdpa_prefill_nax_matches_cpu_reference_f32_single_tile() {
    let (q_len, k_len, gqa, nqh, nkvh) = (16usize, 16usize, 1usize, 1usize, 1usize);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = build_attn_inputs(q_len, k_len, nqh, nkvh);
    let expected = cpu_sdpa_reference(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());

    let cos = cosine(&expected, &actual);
    println!("[f32 single-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 single-tile)");
}

// ── Shape 2 : multi-tile causal (q_len = k_len = 64) ───────────────────────

#[test]
fn mt_sdpa_prefill_nax_matches_cpu_reference_f32_multi_tile() {
    let (q_len, k_len, gqa, nqh, nkvh) = (64usize, 64usize, 1usize, 1usize, 1usize);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = build_attn_inputs(q_len, k_len, nqh, nkvh);
    let expected = cpu_sdpa_reference(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 multi-tile q_len={q_len}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 multi-tile)");
}

// ── Shape 3 : multi-head GQA (4 q-heads, 2 kv-heads) ───────────────────────

#[test]
fn mt_sdpa_prefill_nax_matches_cpu_reference_f32_gqa() {
    let (q_len, k_len, gqa, nqh, nkvh) = (32usize, 32usize, 2usize, 4usize, 2usize);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = build_attn_inputs(q_len, k_len, nqh, nkvh);
    let expected = cpu_sdpa_reference(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax(
        &ctx,
        DType::F32,
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos = cosine(&expected, &actual);
    println!("[f32 gqa nqh={nqh} nkvh={nkvh}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 gqa)");
}

// ── Shape 4 : fp16 multi-tile ──────────────────────────────────────────────

#[test]
fn mt_sdpa_prefill_nax_matches_cpu_reference_f16_multi_tile() {
    let (q_len, k_len, gqa, nqh, nkvh) = (64usize, 64usize, 1usize, 1usize, 1usize);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (q_f32, k_f32, v_f32) = build_attn_inputs(q_len, k_len, nqh, nkvh);
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let q: Vec<f32> = q_f32.iter().map(|&v| round_f16(v)).collect();
    let k: Vec<f32> = k_f32.iter().map(|&v| round_f16(v)).collect();
    let v: Vec<f32> = v_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_sdpa_reference(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax(
        &ctx,
        DType::F16,
        &f32_to_f16_bytes(&q),
        &f32_to_f16_bytes(&k),
        &f32_to_f16_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        scale,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let cos = cosine(&expected, &actual);
    println!("[f16 multi-tile q_len={q_len}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 multi-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_matches_cpu_reference_bf16_multi_tile() {
    // bf16 stages through `half` inside the kernel (`coop_stage(T)`) —
    // Apple's `matmul2d` mishandles `bfloat` cooperative tensors. The
    // oracle rounds inputs through bf16 so the comparison is apples-to-
    // apples; bf16's 8-bit significand widens the bar slightly.
    let (q_len, k_len, gqa, nqh, nkvh) = (64usize, 64usize, 1usize, 1usize, 1usize);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (q_f32, k_f32, v_f32) = build_attn_inputs(q_len, k_len, nqh, nkvh);
    let round_bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
    let q: Vec<f32> = q_f32.iter().map(|&v| round_bf16(v)).collect();
    let k: Vec<f32> = k_f32.iter().map(|&v| round_bf16(v)).collect();
    let v: Vec<f32> = v_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_sdpa_reference(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, scale);

    let to_bf16 = |xs: &[f32]| -> Vec<u8> {
        xs.iter().flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes()).collect()
    };

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax(
        &ctx,
        DType::BF16,
        &to_bf16(&q),
        &to_bf16(&k),
        &to_bf16(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        scale,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let cos = cosine(&expected, &actual);
    println!("[bf16 multi-tile q_len={q_len}] cos={cos:.6}");
    assert!(cos >= 0.997, "cosine {cos:.6} < 0.997 (bf16 multi-tile)");
}

// ── Wide head_dim variants: d={64,128,256} ─────────────────────────────────
//
// These tests validate the D-chunk loop in `mt_sdpa_prefill_nax_d{64,128,256}`.
// The CPU oracle `cpu_sdpa_reference_hd` is head_dim-parameterized.

use metaltile_std::mlx::steel::attn::steel_attention_nax::{
    mt_sdpa_prefill_nax_d64,
    mt_sdpa_prefill_nax_d128,
    mt_sdpa_prefill_nax_d256,
};

/// Parameterized causal SDPA oracle (same math as `cpu_sdpa_reference` but
/// `head_dim` is passed as an argument).
#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_reference_hd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    k_len: usize,
    gqa_factor: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let q_len_off = k_len - q_len;
    let mut out = vec![0.0f32; n_q_heads * q_len * head_dim];
    for qh in 0..n_q_heads {
        let kvh = qh / gqa_factor;
        assert!(kvh < n_kv_heads);
        for qi in 0..q_len {
            let q_abs = q_len_off + qi;
            let q_base = qh * q_len * head_dim + qi * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; k_len];
            let mut row_max = f32::NEG_INFINITY;
            for ki in 0..=q_abs.min(k_len - 1) {
                let k_base = kvh * k_len * head_dim + ki * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + d] * k[k_base + d];
                }
                let s = dot * scale;
                scores[ki] = s;
                if s > row_max {
                    row_max = s;
                }
            }
            let mut sum = 0.0f32;
            for ki in 0..k_len {
                if scores[ki] > f32::NEG_INFINITY {
                    let e = (scores[ki] - row_max).exp();
                    scores[ki] = e;
                    sum += e;
                } else {
                    scores[ki] = 0.0;
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for ki in 0..k_len {
                    if scores[ki] != 0.0 {
                        let k_base = kvh * k_len * head_dim + ki * head_dim;
                        acc += scores[ki] * v[k_base + d];
                    }
                }
                out[q_base + d] = acc * inv;
            }
        }
    }
    out
}

/// Build test inputs for a given head_dim.
fn build_attn_inputs_hd(
    q_len: usize,
    k_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> =
        (0..n_q_heads * q_len * head_dim).map(|i| 0.01 + (i as f32 % 23.0) * 0.007).collect();
    let k: Vec<f32> =
        (0..n_kv_heads * k_len * head_dim).map(|i| -0.02 + (i as f32 % 19.0) * 0.006).collect();
    let v: Vec<f32> =
        (0..n_kv_heads * k_len * head_dim).map(|i| 0.03 + (i as f32 % 29.0) * 0.005).collect();
    (q, k, v)
}

/// Dispatch a wide nax kernel and return the output bytes.
#[allow(clippy::too_many_arguments)]
fn run_sdpa_nax_wide(
    ctx: &Context,
    _dtype: DType,
    kernel_ir: metaltile_core::ir::Kernel,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    q_len: usize,
    k_len: usize,
    gqa_factor: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(q_len.is_multiple_of(16), "q_len % 16 must == 0");
    assert!(k_len.is_multiple_of(16), "k_len % 16 must == 0");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), q_bytes.to_vec());
    buffers.insert("k".into(), k_bytes.to_vec());
    buffers.insert("v".into(), v_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; n_q_heads * q_len * head_dim * out_bytes_per_elem]);
    buffers.insert("q_len".into(), (q_len as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (k_len as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let mut kernel = kernel_ir;
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [q_len / 16, n_q_heads, 1], [
            32, 1, 1,
        ])
        .expect("dispatch wide nax kernel");

    result.outputs.get("out").expect("`out` buffer").clone()
}

// ── d=64 tests ────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_prefill_nax_d64_matches_cpu_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (16usize, 16usize, 1usize, 1usize, 1usize, 64usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d64::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());
    let cos = cosine(&expected, &actual);
    println!("[d64 f32 single-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d64 f32 single-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d64_multi_tile_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 64usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d64::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cos = cosine(&expected, &actual);
    println!("[d64 f32 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d64 f32 multi-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d64_f16_multi_tile() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 64usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q_f32, k_f32, v_f32) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let q: Vec<f32> = q_f32.iter().map(|&v| round_f16(v)).collect();
    let k: Vec<f32> = k_f32.iter().map(|&v| round_f16(v)).collect();
    let v: Vec<f32> = v_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F16,
        mt_sdpa_prefill_nax_d64::kernel_ir_for(DType::F16),
        &f32_to_f16_bytes(&q),
        &f32_to_f16_bytes(&k),
        &f32_to_f16_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[d64 f16 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d64 f16 multi-tile)");
}

// ── d=128 tests ────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_prefill_nax_d128_matches_cpu_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (16usize, 16usize, 1usize, 1usize, 1usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d128::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());
    let cos = cosine(&expected, &actual);
    println!("[d128 f32 single-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d128 f32 single-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d128_multi_tile_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d128::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cos = cosine(&expected, &actual);
    println!("[d128 f32 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d128 f32 multi-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d128_gqa_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (32usize, 32usize, 2usize, 4usize, 2usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d128::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cos = cosine(&expected, &actual);
    println!("[d128 f32 gqa nqh={nqh} nkvh={nkvh}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d128 f32 gqa)");
}

#[test]
fn mt_sdpa_prefill_nax_d128_f16_multi_tile() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q_f32, k_f32, v_f32) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let q: Vec<f32> = q_f32.iter().map(|&v| round_f16(v)).collect();
    let k: Vec<f32> = k_f32.iter().map(|&v| round_f16(v)).collect();
    let v: Vec<f32> = v_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F16,
        mt_sdpa_prefill_nax_d128::kernel_ir_for(DType::F16),
        &f32_to_f16_bytes(&q),
        &f32_to_f16_bytes(&k),
        &f32_to_f16_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[d128 f16 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d128 f16 multi-tile)");
}

// ── d=256 tests ────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_prefill_nax_d256_matches_cpu_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (16usize, 16usize, 1usize, 1usize, 1usize, 256usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d256::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());
    let cos = cosine(&expected, &actual);
    println!("[d256 f32 single-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d256 f32 single-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d256_multi_tile_f32() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 256usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F32,
        mt_sdpa_prefill_nax_d256::kernel_ir_for(DType::F32),
        &f32_to_f32_bytes(&q),
        &f32_to_f32_bytes(&k),
        &f32_to_f32_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cos = cosine(&expected, &actual);
    println!("[d256 f32 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d256 f32 multi-tile)");
}

#[test]
fn mt_sdpa_prefill_nax_d256_f16_multi_tile() {
    let (q_len, k_len, gqa, nqh, nkvh, hd) = (64usize, 64usize, 1usize, 1usize, 1usize, 256usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q_f32, k_f32, v_f32) = build_attn_inputs_hd(q_len, k_len, nqh, nkvh, hd);
    let round_f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let q: Vec<f32> = q_f32.iter().map(|&v| round_f16(v)).collect();
    let k: Vec<f32> = k_f32.iter().map(|&v| round_f16(v)).collect();
    let v: Vec<f32> = v_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_sdpa_reference_hd(&q, &k, &v, q_len, k_len, gqa, nqh, nkvh, hd, scale);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_sdpa_nax_wide(
        &ctx,
        DType::F16,
        mt_sdpa_prefill_nax_d256::kernel_ir_for(DType::F16),
        &f32_to_f16_bytes(&q),
        &f32_to_f16_bytes(&k),
        &f32_to_f16_bytes(&v),
        q_len,
        k_len,
        gqa,
        nqh,
        nkvh,
        hd,
        scale,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[d256 f16 multi-tile] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (d256 f16 multi-tile)");
}
