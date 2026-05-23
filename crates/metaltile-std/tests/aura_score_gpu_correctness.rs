//! End-to-end correctness test for `ffai::aura_score_int4` on real Metal.
//!
//! Reduction-mode kernel. Threadgroup geometry: 32 threads/TG, one TG
//! per `(q_head, k_token)` pair. Each lane handles `dim/32` slices of
//! the dot product; `simd_sum` reduces across the simdgroup.
//!
//! Computes `scores[qh, t] = (Σ_d q_rot[qh, d] * codebook[q[kvh, t, d]]) * norms[kvh, t]`
//! where `kvh = qh / repeat_count` (GQA fan-out).
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_score::aura_score_int4;

fn pack_int4_indices(indices: &[u32], kv_heads: usize, tokens: usize, dim: usize) -> Vec<u32> {
    let bits = 4;
    let packed_width = (dim * bits).div_ceil(32);
    let mut packed = vec![0u32; kv_heads * tokens * packed_width];
    for kvh in 0..kv_heads {
        for t in 0..tokens {
            for d in 0..dim {
                let idx = indices[(kvh * tokens + t) * dim + d];
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                packed[(kvh * tokens + t) * packed_width + word_idx] |= (idx & 0xf) << shift;
            }
        }
    }
    packed
}

#[allow(clippy::too_many_arguments)]
fn naive_aura_score(
    q_rot: &[f32],
    indices: &[u32],
    norms: &[f32],
    codebook: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let mut scores = vec![0.0_f32; q_heads * tokens];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        for t in 0..tokens {
            let norm_val = norms[kvh * tokens + t];
            let mut acc = 0.0_f32;
            for d in 0..dim {
                let q = indices[(kvh * tokens + t) * dim + d];
                let centroid = codebook[q as usize];
                acc += q_rot[qh * dim + d] * centroid;
            }
            scores[qh * tokens + t] = acc * norm_val;
        }
    }
    scores
}

fn run_aura_score_dtype(dt: Dt, tol: f32, label: &str) {
    let dim = 128usize;
    let bits = 4usize;
    let packed_width = (dim * bits).div_ceil(32);
    let q_heads = 4usize;
    let kv_heads = 2usize;
    let tokens = 8usize;
    let repeat = q_heads / kv_heads;

    let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
    let indices: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 7) % 16) as u32).collect();
    let packed = pack_int4_indices(&indices, kv_heads, tokens, dim);
    let norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.05 * i as f32).collect();
    let q_rot: Vec<f32> =
        (0..q_heads * dim).map(|i| (((i * 13) % 21) as f32 - 10.0) * 0.02).collect();

    // Round inputs through the kernel dtype so the CPU oracle matches the
    // load-cast quantisation the kernel applies.
    let round_in = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| dt.round(x)).collect() };
    let codebook_r = round_in(&codebook);
    let norms_r = round_in(&norms);
    let q_rot_r = round_in(&q_rot);

    let expected =
        naive_aura_score(&q_rot_r, &indices, &norms_r, &codebook_r, q_heads, kv_heads, tokens, dim);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q_rot".into(), pack_bytes(&q_rot_r, dt));
    buffers.insert("packed".into(), pack_u32_bytes(&packed));
    buffers.insert("norms".into(), pack_bytes(&norms_r, dt));
    buffers.insert("codebook".into(), pack_bytes(&codebook_r, dt));
    buffers.insert("scores".into(), pack_bytes(&vec![0.0_f32; q_heads * tokens], dt));
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    buffers.insert("packed_width".into(), (packed_width as u32).to_le_bytes().to_vec());
    buffers.insert("tokens".into(), (tokens as u32).to_le_bytes().to_vec());
    buffers.insert("repeat_count".into(), (repeat as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = aura_score_int4::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // One TG per (q_head, k_token) pair, 32 threads per TG.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [q_heads, tokens, 1], [32, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("scores").expect("`scores` buffer");
    let actual = unpack_bytes(out_bytes, dt);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < tol, "aura_score int4 {label}: max |diff| = {diff:.2e} > {tol:.0e}");
}

#[test]
fn aura_score_int4_matches_naive_reference_f32() { run_aura_score_dtype(Dt::F32, 1e-3, "f32"); }

#[test]
fn aura_score_int4_matches_naive_reference_f16() { run_aura_score_dtype(Dt::F16, 1e-2, "f16"); }

#[test]
fn aura_score_int4_matches_naive_reference_bf16() { run_aura_score_dtype(Dt::Bf16, 5e-2, "bf16"); }

// Unused after the parameterized rewrite — kept so the `metaltile-core::ir`
// path stays in the `use` list if a future test needs it directly.
#[allow(dead_code)]
fn _unused_dtype() -> DType { DType::F32 }
