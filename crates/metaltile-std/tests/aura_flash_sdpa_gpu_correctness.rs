//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::aura_flash_sdpa` — fused single-pass SDPA
//! over an AURA-compressed K/V cache, with attention sinks and
//! sliding-window causal masking.
//!
//! Pins three things against a naive f32 oracle: (1) the plain
//! online-softmax attention matches the two-pass `aura_flash` reference;
//! (2) `has_sinks` injects a virtual zero-value key at logit `sink`,
//! shifting the softmax denominator; (3) `window_size` drops keys
//! outside the sliding window.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_flash_sdpa::aura_flash_sdpa_kb4_vb2_d128;

fn pack_int_indices(
    indices: &[u32],
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    bits: usize,
) -> Vec<u32> {
    let mask = (1u32 << bits) - 1;
    let pw = (dim * bits).div_ceil(32);
    let mut packed = vec![0u32; kv_heads * tokens * pw];
    for kvh in 0..kv_heads {
        for t in 0..tokens {
            for d in 0..dim {
                let idx = indices[(kvh * tokens + t) * dim + d] & mask;
                let bit = d * bits;
                let word = bit / 32;
                let shift = bit & 31;
                packed[(kvh * tokens + t) * pw + word] |= idx << shift;
                let spill = (shift + bits) as i32 - 32;
                if spill > 0 {
                    packed[(kvh * tokens + t) * pw + word + 1] |=
                        idx >> (bits as u32 - spill as u32);
                }
            }
        }
    }
    packed
}

#[allow(clippy::too_many_arguments)]
fn naive(
    q_rot: &[f32],
    key_idx: &[u32],
    val_idx: &[u32],
    key_norms: &[f32],
    val_norms: &[f32],
    key_cb: &[f32],
    val_cb: &[f32],
    sinks: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    has_sinks: bool,
    window_size: usize,
    num_q_heads: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let causal_upper = tokens - 1;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        // Scores for participating (windowed) tokens.
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for t in 0..tokens {
            if window_size == 0 || t + window_size > causal_upper {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    let q = key_idx[(kvh * tokens + t) * dim + d];
                    dot += q_rot[qh * dim + d] * key_cb[q as usize];
                }
                kept.push((t, dot * key_norms[kvh * tokens + t]));
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
                let v = val_idx[(kvh * tokens + t) * dim + d];
                *a += w * val_cb[v as usize] * val_norms[kvh * tokens + t];
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
    q_rot: &[f32],
    key_packed: &[u32],
    key_norms: &[f32],
    key_cb: &[f32],
    val_packed: &[u32],
    val_norms: &[f32],
    val_cb: &[f32],
    sinks: &[f32],
    dt: Dt,
    q_heads: usize,
    dim: usize,
    tokens: usize,
    repeat: usize,
    num_q_heads: usize,
    has_sinks: u32,
    window_size: u32,
) -> Vec<f32> {
    let kpw = (dim * 4).div_ceil(32);
    let vpw = (dim * 2).div_ceil(32);
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q_rot".into(), pack_bytes(q_rot, Dt::F32));
    b.insert("key_packed".into(), pack_u32_bytes(key_packed));
    b.insert("key_norms".into(), pack_bytes(key_norms, Dt::F32));
    b.insert("key_codebook".into(), pack_bytes(key_cb, Dt::F32));
    b.insert("val_packed".into(), pack_u32_bytes(val_packed));
    b.insert("val_norms".into(), pack_bytes(val_norms, Dt::F32));
    b.insert("val_codebook".into(), pack_bytes(val_cb, Dt::F32));
    b.insert("sinks".into(), pack_bytes(sinks, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0_f32; q_heads * dim], dt));
    for (k, v) in [
        ("dim", dim as u32),
        ("key_packed_width", kpw as u32),
        ("value_packed_width", vpw as u32),
        ("tokens", tokens as u32),
        // Fully-populated fixture: stride == live row count.
        ("kv_stride", tokens as u32),
        ("repeat_count", repeat as u32),
        ("num_q_heads", num_q_heads as u32),
        ("has_sinks", has_sinks),
        ("window_size", window_size),
    ] {
        b.insert(k.into(), v.to_le_bytes().to_vec());
    }

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = aura_flash_sdpa_kb4_vb2_d128::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, q_heads, 1], [32, 1, 1])
        .expect("aura_flash_sdpa dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(q_heads * dim);
    out
}

struct Fixture {
    q_rot: Vec<f32>,
    key_idx: Vec<u32>,
    val_idx: Vec<u32>,
    key_packed: Vec<u32>,
    val_packed: Vec<u32>,
    key_norms: Vec<f32>,
    val_norms: Vec<f32>,
    key_cb: Vec<f32>,
    val_cb: Vec<f32>,
    sinks: Vec<f32>,
}

fn fixture(q_heads: usize, kv_heads: usize, tokens: usize, dim: usize) -> Fixture {
    let key_cb: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
    let val_cb: Vec<f32> = (0..4).map(|i| -1.0 + 2.0 * i as f32 / 3.0).collect();
    let key_idx: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
    let val_idx: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 5) % 4) as u32).collect();
    Fixture {
        key_packed: pack_int_indices(&key_idx, kv_heads, tokens, dim, 4),
        val_packed: pack_int_indices(&val_idx, kv_heads, tokens, dim, 2),
        key_norms: (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect(),
        val_norms: (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect(),
        q_rot: (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect(),
        sinks: (0..q_heads).map(|i| -0.5 + 0.2 * i as f32).collect(),
        key_idx,
        val_idx,
        key_cb,
        val_cb,
    }
}

fn check(dt: Dt, has_sinks: bool, window: usize, tol: f32) {
    let _g = gpu_lock();
    let (q_heads, kv_heads, tokens, dim) = (2usize, 1usize, 8usize, 128usize);
    let f = fixture(q_heads, kv_heads, tokens, dim);
    let expected = naive(
        &f.q_rot,
        &f.key_idx,
        &f.val_idx,
        &f.key_norms,
        &f.val_norms,
        &f.key_cb,
        &f.val_cb,
        &f.sinks,
        q_heads,
        kv_heads,
        tokens,
        dim,
        has_sinks,
        window,
        q_heads,
    );
    let actual = run(
        &f.q_rot,
        &f.key_packed,
        &f.key_norms,
        &f.key_cb,
        &f.val_packed,
        &f.val_norms,
        &f.val_cb,
        &f.sinks,
        dt,
        q_heads,
        dim,
        tokens,
        q_heads / kv_heads,
        q_heads,
        u32::from(has_sinks),
        window as u32,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "dt={:?} sinks={has_sinks} window={window}: max |diff| = {diff:.2e}",
        dt as u32
    );
}

#[test]
fn aura_flash_sdpa_plain_f32() { check(Dt::F32, false, 0, 1e-4); }

#[test]
fn aura_flash_sdpa_with_sinks_f32() { check(Dt::F32, true, 0, 1e-4); }

#[test]
fn aura_flash_sdpa_sliding_window_f32() { check(Dt::F32, false, 4, 1e-4); }

#[test]
fn aura_flash_sdpa_sinks_and_window_f32() { check(Dt::F32, true, 4, 1e-4); }

#[test]
fn aura_flash_sdpa_plain_bf16() { check(Dt::Bf16, false, 0, 5e-2); }

#[test]
fn aura_flash_sdpa_with_sinks_bf16() { check(Dt::Bf16, true, 0, 5e-2); }
