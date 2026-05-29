//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end correctness for `ffai_sdpa_decode_d96` — the
//! head_dim=96 specialization needed for Phi-3-mini / Phi-3.5-mini.
//!
//! Same shape as `sdpa_decode_d64_gpu_correctness.rs` but with
//! head_dim=96. Pins the 3-elt-per-lane Q/K/V layout + the
//! cross-simdgroup output reduction. macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, SdpaShape, naive_sdpa_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_d96::ffai_sdpa_decode_d96;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

#[test]
fn sdpa_decode_d96_matches_naive_cpu_reference_f32() {
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 96usize;
    let n_kv = 5usize;
    let kv_stride = 5usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = naive_sdpa_f32(&q, &k, &v, &shape);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(&q));
    buffers.insert("k".into(), f32_slice_to_bytes(&k));
    buffers.insert("v".into(), f32_slice_to_bytes(&v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = ffai_sdpa_decode_d96::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // One TG per Q head, 1024 threads per TG (32 simdgroups × 32 lanes
    // — same design TPG as the head_dim=128 kernel).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer");
    let actual = bytes_to_f32_vec(out_bytes);

    assert_eq!(actual.len(), expected.len(), "output element count");

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }

    assert!(
        max_diff < 1e-4,
        "sdpa_decode_d96 diverges from CPU reference: max |diff| = {max_diff:.2e} at {max_at}",
    );
}
