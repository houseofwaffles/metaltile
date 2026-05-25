//! End-to-end GPU correctness for `ffai::sdpa_multi_d256`, the
//! head_dim=256 multi-query SDPA kernel.
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a per-row oracle built by dispatching `ffai_sdpa_decode_d256` once
//! per query row with the row's effective `n_kv`. This is the
//! strongest oracle we can build, the d=256 single-query kernel is
//! already independently validated against a naive CPU reference, so
//! agreement with it across all query rows proves the multi-query
//! kernel preserves both the online-softmax math and the d=256
//! 2-phase output reduction.
//!
//! Covers:
//!   - causal mode (`causal == 1`), query `r` attends `[0, base_kv + r + 1)`
//!   - full mode (`causal == 0`), every query attends `[0, base_kv + n_query)`
//!   - non-zero `base_kv` prefix (cached context before the block)
//!   - GQA fan-out (`n_q_heads > n_kv_heads`)
//!   - f32 / f16 / bf16
//!
//! Shapes stay small so the per-row decode oracle is cheap.
//! macOS-gated, needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_d256::ffai_sdpa_decode_d256;
use metaltile_std::ffai::sdpa_multi_d256::ffai_sdpa_multi_d256;

const HEAD_DIM: usize = 256;

/// Dispatch the multi-query d=256 kernel for one (Q, K, V) triple.
#[allow(clippy::too_many_arguments)]
fn run_multi_d256(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dt: Dt,
    n_q_heads: usize,
    n_kv_heads: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let heads_per_group = n_q_heads / n_kv_heads;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_query * n_q_heads * HEAD_DIM], dt));
    buffers.insert("head_dim".into(), (HEAD_DIM as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("base_kv".into(), (base_kv as u32).to_le_bytes().to_vec());
    buffers.insert("n_query".into(), (n_query as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("causal".into(), (causal as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_sdpa_multi_d256::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per (query, q_head), TPG = 1024.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads * n_query, 1, 1], [
            1024, 1, 1,
        ])
        .expect("dispatch_with_grid sdpa_multi_d256");
    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
}

/// Per-row oracle built by dispatching `ffai_sdpa_decode_d256` once
/// for each query row. The K/V cache is shared (full `kv_stride`
/// depth), and the row's `n_kv` selects how far into the cache the
/// decode attends. For full mode every row uses `base_kv + n_query`,
/// for causal mode row `r` uses `base_kv + r + 1`.
#[allow(clippy::too_many_arguments)]
fn oracle_per_row_decode_d256(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dt: Dt,
    n_q_heads: usize,
    n_kv_heads: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let heads_per_group = n_q_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_query * n_q_heads * HEAD_DIM];

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_sdpa_decode_d256::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    for r in 0..n_query {
        let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
        // Slice this query row out of the (n_query, n_q_heads, head_dim)
        // tensor, the d=256 decode kernel expects (n_q_heads, head_dim).
        let row_off = r * n_q_heads * HEAD_DIM;
        let q_row = &q[row_off..row_off + n_q_heads * HEAD_DIM];

        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("q".into(), pack_bytes(q_row, dt));
        buffers.insert("k".into(), pack_bytes(k, dt));
        buffers.insert("v".into(), pack_bytes(v, dt));
        buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_q_heads * HEAD_DIM], dt));
        buffers.insert("head_dim".into(), (HEAD_DIM as u32).to_le_bytes().to_vec());
        buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
        buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
        buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
        buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [
                1024, 1, 1,
            ])
            .expect("dispatch_with_grid sdpa_decode_d256");
        let row_out = unpack_bytes(result.outputs.get("out").expect("out buffer"), dt);
        out[row_off..row_off + n_q_heads * HEAD_DIM].copy_from_slice(&row_out);
    }
    out
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

#[test]
fn sdpa_multi_d256_full_mode_matches_decode_oracle_f32() {
    let _g = gpu_lock();
    // No prefix, 8-query block, full attention (bidirectional within block).
    let (n_q_heads, n_kv_heads) = (4usize, 1usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q = ramp(n_query * n_q_heads * HEAD_DIM, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 11, 5.0);

    let expected = oracle_per_row_decode_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, false, scale,
    );
    let actual = run_multi_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, false, scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi_d256 full f32");
}

#[test]
fn sdpa_multi_d256_causal_mode_matches_decode_oracle_f32() {
    let _g = gpu_lock();
    // Causal within the block, query r attends [0, r+1).
    let (n_q_heads, n_kv_heads) = (4usize, 1usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q = ramp(n_query * n_q_heads * HEAD_DIM, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 11, 5.0);

    let expected = oracle_per_row_decode_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    let actual = run_multi_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi_d256 causal f32");
}

#[test]
fn sdpa_multi_d256_with_prefix_and_gqa_matches_decode_oracle_f32() {
    let _g = gpu_lock();
    // Non-zero cached prefix + GQA fan-out (Qwen3.6-A3B full-attention
    // shape: 32 q-heads over 4 kv-heads at head_dim=256). Causal mode.
    let (n_q_heads, n_kv_heads) = (32usize, 4usize);
    let (base_kv, n_query) = (16usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q = ramp(n_query * n_q_heads * HEAD_DIM, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 11, 5.0);

    let expected = oracle_per_row_decode_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    let actual = run_multi_d256(
        &q, &k, &v, Dt::F32, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi_d256 prefix+GQA causal f32");
}

#[test]
fn sdpa_multi_d256_full_mode_matches_decode_oracle_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads) = (4usize, 2usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q = ramp(n_query * n_q_heads * HEAD_DIM, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 11, 5.0);

    // Both kernels see the same f16-quantised inputs (oracle is also
    // run at Dt::F16), so the only allowed drift comes from f16 storage
    // round-trips in K/V loads — tolerance can stay tight.
    let expected = oracle_per_row_decode_d256(
        &q, &k, &v, Dt::F16, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, false, scale,
    );
    let actual = run_multi_d256(
        &q, &k, &v, Dt::F16, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, false, scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_multi_d256 full f16");
}

#[test]
fn sdpa_multi_d256_causal_mode_matches_decode_oracle_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads) = (4usize, 2usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q = ramp(n_query * n_q_heads * HEAD_DIM, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 11, 5.0);

    let expected = oracle_per_row_decode_d256(
        &q, &k, &v, Dt::Bf16, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    let actual = run_multi_d256(
        &q, &k, &v, Dt::Bf16, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, true, scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_multi_d256 causal bf16");
}
