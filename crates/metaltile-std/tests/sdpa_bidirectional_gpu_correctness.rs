//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end GPU correctness for `ffai::sdpa_bidirectional_d{32,64,72}`
//! — multi-query bidirectional SDPA at head_dim 32 (FastViT-HD),
//! head_dim 64 (SigLIP / CLIP), and head_dim 72 (PaliGemma's
//! SigLIP-So400m, ragged 3-elems-per-lane layout).
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a straight-translation CPU reference. Covers:
//!   - full bidirectional attention (every query attends every key)
//!   - a non-zero `base_kv` prefix (cached context before the block)
//!   - GQA fan-out (`n_q_heads > n_kv_heads`)
//!   - f32 / f16 / bf16
//!
//! Shapes stay small so the CPU reference is instant and eyeball-able.
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_bidirectional::{
    ffai_sdpa_bidirectional_d32,
    ffai_sdpa_bidirectional_d64,
    ffai_sdpa_bidirectional_d72,
    ffai_sdpa_bidirectional_d80,
    ffai_sdpa_bidirectional_d96,
};

/// CPU reference: per (query, q_head), softmax(Q·Kᵀ·scale)·V over the
/// full `[0, base_kv + n_query)` range. fp32 throughout.
#[allow(clippy::too_many_arguments)]
fn naive_sdpa_bidirectional(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    scale: f32,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let n_kv = base_kv + n_query;
    let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
    for r in 0..n_query {
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = (r * n_q_heads + qh) * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = kv_slab + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - m).exp();
                sum += *score;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, score) in scores.iter().enumerate() {
                    acc += *score * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

/// Dispatch flavour: which variant + which Dt.
#[derive(Copy, Clone)]
enum Variant {
    D32,
    D64,
    D72,
    D80,
    D96,
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_bidirectional(
    variant: Variant,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dt: Dt,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    scale: f32,
) -> Vec<f32> {
    let heads_per_group = n_q_heads / n_kv_heads;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_query * n_q_heads * head_dim], dt));
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("base_kv".into(), (base_kv as u32).to_le_bytes().to_vec());
    buffers.insert("n_query".into(), (n_query as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = match variant {
        Variant::D32 => ffai_sdpa_bidirectional_d32::kernel_ir_for(dt.to_dtype()),
        Variant::D64 => ffai_sdpa_bidirectional_d64::kernel_ir_for(dt.to_dtype()),
        Variant::D72 => ffai_sdpa_bidirectional_d72::kernel_ir_for(dt.to_dtype()),
        Variant::D80 => ffai_sdpa_bidirectional_d80::kernel_ir_for(dt.to_dtype()),
        Variant::D96 => ffai_sdpa_bidirectional_d96::kernel_ir_for(dt.to_dtype()),
    };
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per (query, q_head); TPG = 1024 (kernel invariant —
    // a smaller TPG would make n_simd=0 and freeze the GPU).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads * n_query, 1, 1], [
            1024, 1, 1,
        ])
        .expect("dispatch_with_grid");
    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at]
    );
}

// ─── head_dim = 64 (SigLIP / CLIP) ────────────────────────────────

#[test]
fn sdpa_bidirectional_d64_no_prefix_matches_cpu_f32() {
    let _g = gpu_lock();
    // No prefix, 8-query block (the small-image SigLIP shape).
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 64usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D64,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d64 no-prefix f32");
}

#[test]
fn sdpa_bidirectional_d64_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    // Non-zero cached prefix + GQA fan-out — exercises the same code
    // path as a Whisper-style cross-attention with KV cached.
    let (n_q_heads, n_kv_heads, head_dim) = (8usize, 2usize, 64usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D64,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d64 prefix+GQA f32");
}

#[test]
fn sdpa_bidirectional_d64_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 64usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D64,
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_bidirectional_d64 f16");
}

#[test]
fn sdpa_bidirectional_d64_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 64usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D64,
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_bidirectional_d64 bf16");
}

// ─── head_dim = 32 (FastViT-HD) ───────────────────────────────────

#[test]
fn sdpa_bidirectional_d32_no_prefix_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 32usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D32,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d32 no-prefix f32");
}

#[test]
fn sdpa_bidirectional_d32_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (8usize, 2usize, 32usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D32,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d32 prefix+GQA f32");
}

#[test]
fn sdpa_bidirectional_d32_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 32usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D32,
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_bidirectional_d32 f16");
}

#[test]
fn sdpa_bidirectional_d32_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 32usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D32,
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_bidirectional_d32 bf16");
}

// ─── head_dim = 72 (PaliGemma SigLIP-So400m, ragged layout) ───────
//
// These cases specifically exercise the bounds-masking path: head_dim
// is not a multiple of 32, so lanes 24..31 of every simdgroup must
// contribute 0 to the dot product and skip the per-element output
// store. The CPU reference is the same naive softmax(Q·Kᵀ·scale)·V —
// it implicitly handles "no ragged layout" by just summing over all
// 72 elements. The GPU result must agree.

#[test]
fn sdpa_bidirectional_d72_no_prefix_matches_cpu_f32() {
    let _g = gpu_lock();
    // SigLIP-So400m-style heads (16 heads × 72 head_dim = 1152 hidden).
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 72usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D72,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d72 no-prefix f32");
}

#[test]
fn sdpa_bidirectional_d72_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (8usize, 2usize, 72usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D72,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d72 prefix+GQA f32");
}

#[test]
fn sdpa_bidirectional_d72_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 72usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D72,
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_bidirectional_d72 f16");
}

#[test]
fn sdpa_bidirectional_d72_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 72usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D72,
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_bidirectional_d72 bf16");
}

// ─── head_dim = 80 (Qwen2.5-VL vision tower, ragged) ──────────────
//
// Like d72, 3 elements per lane with bounds masking — but with the
// added subtlety that lane 26 is PARTIALLY in range (one of its three
// indices is OOB at d=80). Exercises the per-element mask vs the
// per-lane mask.

#[test]
fn sdpa_bidirectional_d80_no_prefix_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 80usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D80,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d80 no-prefix f32");
}

#[test]
fn sdpa_bidirectional_d80_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (8usize, 2usize, 80usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D80,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d80 prefix+GQA f32");
}

#[test]
fn sdpa_bidirectional_d80_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 80usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D80,
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_bidirectional_d80 f16");
}

#[test]
fn sdpa_bidirectional_d80_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 80usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D80,
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_bidirectional_d80 bf16");
}

// ─── head_dim = 96 (Qwen2-VL vision tower, clean fit) ─────────────
//
// 32 lanes × 3 = 96 exactly; no bounds masking. Validates the
// no-mask branch.

#[test]
fn sdpa_bidirectional_d96_no_prefix_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 96usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D96,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d96 no-prefix f32");
}

#[test]
fn sdpa_bidirectional_d96_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (8usize, 2usize, 96usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_bidirectional(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D96,
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_bidirectional_d96 prefix+GQA f32");
}

#[test]
fn sdpa_bidirectional_d96_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 96usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D96,
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_bidirectional_d96 f16");
}

#[test]
fn sdpa_bidirectional_d96_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 96usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_bidirectional(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    let actual = run_sdpa_bidirectional(
        Variant::D96,
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_bidirectional_d96 bf16");
}
