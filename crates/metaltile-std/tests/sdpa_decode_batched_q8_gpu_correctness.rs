//! GPU correctness for `sdpa_decode_batched_q8` (M7 branching-factor-8).
//!
//! Strategy: build eight independent Q vectors per head, interleave them
//! into the batched layout `[n_q_heads, 8, head_dim]`, dispatch the
//! batched kernel once, then assert that each output slot matches a
//! naive per-Q scalar SDPA oracle.
//!
//! This catches:
//!   * Wrong Q indexing in `q_off_{0..7}` — a wrong multiplier
//!     (e.g. still 4 instead of 8) shifts all eight Q bases.
//!   * Phase aliasing — Phase B reading Phase A's residual tg state.
//!   * Wrong output offsets — `q_head * 8 * head_dim` base.
//!   * KV-walk interleaving — factor/weight applied to the wrong stream.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, SdpaShape, gpu_lock, naive_sdpa_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_batched::sdpa_decode_batched_q8;

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Interleave eight `[n_q_heads, head_dim]` arrays into
/// `[n_q_heads, 8, head_dim]`.
fn interleave_q8(qs: [&[f32]; 8], n_q_heads: usize, head_dim: usize) -> Vec<f32> {
    for q in &qs {
        assert_eq!(q.len(), n_q_heads * head_dim);
    }
    let mut out = Vec::with_capacity(n_q_heads * 8 * head_dim);
    for h in 0..n_q_heads {
        for &q in &qs {
            out.extend_from_slice(&q[h * head_dim..(h + 1) * head_dim]);
        }
    }
    out
}

/// Split `[n_q_heads, 8, head_dim]` output into eight `[n_q_heads, head_dim]` views.
fn split_batched_out_q8(batched: &[f32], n_q_heads: usize, head_dim: usize) -> [Vec<f32>; 8] {
    assert_eq!(batched.len(), n_q_heads * 8 * head_dim);
    let mut outs: [Vec<f32>; 8] = Default::default();
    for o in &mut outs {
        o.reserve(n_q_heads * head_dim);
    }
    for h in 0..n_q_heads {
        let base = h * 8 * head_dim;
        for (i, o) in outs.iter_mut().enumerate() {
            o.extend_from_slice(&batched[base + i * head_dim..base + (i + 1) * head_dim]);
        }
    }
    outs
}

#[allow(clippy::too_many_arguments)]
fn run_q8_f32(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    q_batched: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_to_bytes(q_batched));
    buffers.insert("k".into(), f32_to_bytes(k));
    buffers.insert("v".into(), f32_to_bytes(v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * 8 * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    // K=8 dispatched at 256 threads (8 simdgroups × 32 lanes) — conservative
    // TPG to stay within Apple GPU register-file limits on M1/M2/M3.
    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [256, 1, 1])
        .expect("dispatch sdpa_decode_batched_q8");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32(out_bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
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
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at [{max_at}] (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

/// Small shape (n_kv=4) — catches layout / indexing bugs without
/// floating-point reduction noise drowning the signal.
#[test]
fn sdpa_decode_batched_q8_matches_eight_independent_decodes_f32_small() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 4usize;
    let kv_stride = 4usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Eight visibly-different Q tensors.
    let qs: [Vec<f32>; 8] = [
        ramp(n_q_heads * head_dim, 17, 8.0),
        ramp(n_q_heads * head_dim, 11, 5.0),
        ramp(n_q_heads * head_dim, 23, 11.0),
        ramp(n_q_heads * head_dim, 29, 14.0),
        ramp(n_q_heads * head_dim, 31, 15.0),
        ramp(n_q_heads * head_dim, 37, 18.0),
        ramp(n_q_heads * head_dim, 41, 20.0),
        ramp(n_q_heads * head_dim, 43, 21.0),
    ];
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected: [Vec<f32>; 8] = std::array::from_fn(|i| naive_sdpa_f32(&qs[i], &k, &v, &shape));

    let q_batched = interleave_q8(qs.each_ref().map(|q| q.as_slice()), n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = sdpa_decode_batched_q8::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_q8_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q8(&batched_out, n_q_heads, head_dim);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, 1e-4, &format!("sdpa_decode_batched_q8 Q[{i}] n_kv=4"));
    }
}

/// Larger shape (n_kv=1024, GQA) — exercises the cross-simdgroup
/// reduction at realistic context length.
#[test]
fn sdpa_decode_batched_q8_matches_eight_independent_decodes_f32_large() {
    let _g = gpu_lock();
    let n_q_heads = 8usize;
    let n_kv_heads = 2usize; // gqa_factor = 4
    let head_dim = 128usize;
    let n_kv = 1024usize;
    let kv_stride = 1024usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let qs: [Vec<f32>; 8] = [
        ramp(n_q_heads * head_dim, 17, 8.0),
        ramp(n_q_heads * head_dim, 11, 5.0),
        ramp(n_q_heads * head_dim, 23, 11.0),
        ramp(n_q_heads * head_dim, 29, 14.0),
        ramp(n_q_heads * head_dim, 31, 15.0),
        ramp(n_q_heads * head_dim, 37, 18.0),
        ramp(n_q_heads * head_dim, 41, 20.0),
        ramp(n_q_heads * head_dim, 43, 21.0),
    ];
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected: [Vec<f32>; 8] = std::array::from_fn(|i| naive_sdpa_f32(&qs[i], &k, &v, &shape));

    let q_batched = interleave_q8(qs.each_ref().map(|q| q.as_slice()), n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = sdpa_decode_batched_q8::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_q8_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q8(&batched_out, n_q_heads, head_dim);

    // 5e-4 tolerance: 1024 KV positions stack up simd_sum reorder noise.
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, 5e-4, &format!("sdpa_decode_batched_q8 Q[{i}] n_kv=1024"));
    }
}

/// Sanity: when all 8 Q vectors are identical the outputs must be
/// bit-identical (within fp32 noise). Catches any phase-aliasing bug.
#[test]
fn sdpa_decode_batched_q8_identical_qs_produce_identical_outputs_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 8usize;
    let kv_stride = 8usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let q_refs = [q.as_slice(); 8];
    let q_batched = interleave_q8(q_refs, n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = sdpa_decode_batched_q8::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_q8_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q8(&batched_out, n_q_heads, head_dim);

    for i in 1..8 {
        assert_close(
            &actual[i],
            &actual[0],
            1e-6,
            &format!("Q[{i}] vs Q[0] when all Q vectors are equal"),
        );
    }
}
