//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end correctness test for the AURA flash pair —
//! `ffai::aura_flash_p1_kb4_vb2_d128` (block-level online softmax over
//! compressed K/V) + `ffai::aura_flash_pass2_d128` (cross-block merge
//! and narrow-cast to bf16). Two-pass SDPA over the AURA-encoded cache.
//!
//! Test approach: deterministic codebook indices, run a CPU reference
//! that mirrors the kernel math (Q · codebook[k_idx[t]] dot, online
//! softmax over compressed scores, V centroid accumulation), dispatch
//! both kernels, compare bf16 output within tolerance.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    aura_flash_p1::aura_flash_p1_kb4_vb2_d128,
    aura_flash_pass2::aura_flash_pass2_d128,
};

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }
// `bytes_to_bf16_vec` was used by the old f32-partials / bf16-output
// path; the kernels are generic over `T` now, so the pair test runs
// f32 end-to-end (the bf16 narrow-write is exercised by aura_flash_sdpa).
#[allow(dead_code)]
fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::Bf16) }

fn pack_int_indices(
    indices: &[u32],
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    bits: usize,
) -> Vec<u32> {
    let mask = (1u32 << bits) - 1;
    let packed_width = (dim * bits).div_ceil(32);
    let mut packed = vec![0u32; kv_heads * tokens * packed_width];
    for kvh in 0..kv_heads {
        for t in 0..tokens {
            for d in 0..dim {
                let idx = indices[(kvh * tokens + t) * dim + d];
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                let masked = idx & mask;
                packed[(kvh * tokens + t) * packed_width + word_idx] |= masked << shift;
                let spill = (shift + bits) as i32 - 32;
                if spill > 0 {
                    let s = spill as u32;
                    packed[(kvh * tokens + t) * packed_width + word_idx + 1] |=
                        masked >> (bits as u32 - s);
                }
            }
        }
    }
    packed
}

/// End-to-end CPU reference for the AURA flash pair: dot scores in the
/// compressed domain, softmax, then accumulate V centroids weighted by
/// the softmax weights + per-token V norm. Final divide by Σ weights.
#[allow(clippy::too_many_arguments)]
fn naive_aura_flash(
    q_rot: &[f32],
    key_indices: &[u32],
    val_indices: &[u32],
    key_norms: &[f32],
    val_norms: &[f32],
    key_codebook: &[f32],
    val_codebook: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        let mut scores = vec![0.0_f32; tokens];
        for t in 0..tokens {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                let q = key_indices[(kvh * tokens + t) * dim + d];
                dot += q_rot[qh * dim + d] * key_codebook[q as usize];
            }
            scores[t] = dot * key_norms[kvh * tokens + t];
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum_w: f32 = weights.iter().sum();
        let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
        for d in 0..dim {
            let mut acc = 0.0_f32;
            for t in 0..tokens {
                let v = val_indices[(kvh * tokens + t) * dim + d];
                let centroid = val_codebook[v as usize] * val_norms[kvh * tokens + t];
                acc += weights[t] * centroid;
            }
            out[qh * dim + d] = acc * inv;
        }
    }
    out
}

#[test]
fn aura_flash_pair_matches_naive_reference_kb4_vb2_d128() {
    let dim = 128usize;
    let key_bits = 4usize;
    let value_bits = 2usize;
    let key_packed_width = (dim * key_bits).div_ceil(32); // 16
    let value_packed_width = (dim * value_bits).div_ceil(32); // 8
    let q_heads = 2usize;
    let kv_heads = 1usize;
    let repeat = q_heads / kv_heads;
    let tokens = 8usize;
    let block_size = 4usize;
    let num_blocks = tokens.div_ceil(block_size); // 2

    // Codebooks.
    let key_codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
    let val_codebook: Vec<f32> = (0..4).map(|i| -1.0 + 2.0 * i as f32 / 3.0).collect();

    // Pseudo-random indices.
    let key_indices: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
    let val_indices: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 5) % 4) as u32).collect();
    let key_packed = pack_int_indices(&key_indices, kv_heads, tokens, dim, key_bits);
    let val_packed = pack_int_indices(&val_indices, kv_heads, tokens, dim, value_bits);

    let key_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect();
    let val_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect();

    let q_rot: Vec<f32> =
        (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();

    let expected = naive_aura_flash(
        &q_rot,
        &key_indices,
        &val_indices,
        &key_norms,
        &val_norms,
        &key_codebook,
        &val_codebook,
        q_heads,
        kv_heads,
        tokens,
        dim,
    );

    let ctx = Context::new().expect("Context::new should succeed on macOS");

    // ── Pass 1: build (o, m, l) partials per (q_idx, block_idx) ────
    let mut p1_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p1_buffers.insert("q_rot".into(), f32_slice_to_bytes(&q_rot));
    p1_buffers.insert("key_packed".into(), pack_u32_bytes(&key_packed));
    p1_buffers.insert("key_norms".into(), f32_slice_to_bytes(&key_norms));
    p1_buffers.insert("key_codebook".into(), f32_slice_to_bytes(&key_codebook));
    p1_buffers.insert("val_packed".into(), pack_u32_bytes(&val_packed));
    p1_buffers.insert("val_norms".into(), f32_slice_to_bytes(&val_norms));
    p1_buffers.insert("val_codebook".into(), f32_slice_to_bytes(&val_codebook));
    p1_buffers.insert("o_partials".into(), vec![0u8; q_heads * num_blocks * dim * 4]);
    p1_buffers.insert("m_partials".into(), vec![0u8; q_heads * num_blocks * 4]);
    p1_buffers.insert("l_partials".into(), vec![0u8; q_heads * num_blocks * 4]);
    p1_buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    p1_buffers.insert("key_packed_width".into(), (key_packed_width as u32).to_le_bytes().to_vec());
    p1_buffers
        .insert("value_packed_width".into(), (value_packed_width as u32).to_le_bytes().to_vec());
    p1_buffers.insert("tokens".into(), (tokens as u32).to_le_bytes().to_vec());
    // Fully-populated fixture: stride == live row count.
    p1_buffers.insert("kv_stride".into(), (tokens as u32).to_le_bytes().to_vec());
    p1_buffers.insert("repeat_count".into(), (repeat as u32).to_le_bytes().to_vec());
    p1_buffers.insert("num_blocks".into(), (num_blocks as u32).to_le_bytes().to_vec());
    p1_buffers.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    // q_position is consulted only by the causal variant; the non-causal
    // kernel ignores it, but the buffer must still be bound. tokens-1 =
    // every token visible (a no-op cutoff).
    p1_buffers.insert("q_position".into(), ((tokens - 1) as u32).to_le_bytes().to_vec());

    let mut p1_kernel = aura_flash_p1_kb4_vb2_d128::kernel_ir_for(DType::F32);
    p1_kernel.mode = KernelMode::Grid3D;

    // Grid3D: gid.x = thread_position_in_grid.x = tg_id.x * tg.x + tid.x.
    // The kernel wants `lane = program_id::<0>() ∈ [0, 32)` — one
    // simdgroup per (q_head, k_block) pair. So for the x-axis we want
    // grid_groups.x=1 and tg.x=32 (32 threads = 1 simdgroup, matching
    // the kernel's `simd_sum(dot_partial)` reduction). y/z axes carry
    // the (q_head, block_idx) extent via grid_groups since tg.y/z=1.
    let p1_result = ctx
        .dispatch_with_grid(&p1_kernel, &p1_buffers, &BTreeMap::new(), [1, q_heads, num_blocks], [
            32, 1, 1,
        ])
        .expect("flash_p1 dispatch should succeed");

    let o_partials_bytes = p1_result.outputs.get("o_partials").expect("`o_partials` buffer");
    let m_partials_bytes = p1_result.outputs.get("m_partials").expect("`m_partials` buffer");
    let l_partials_bytes = p1_result.outputs.get("l_partials").expect("`l_partials` buffer");
    let _o_p = bytes_to_f32_vec(o_partials_bytes);
    let _m_p = bytes_to_f32_vec(m_partials_bytes);
    let _l_p = bytes_to_f32_vec(l_partials_bytes);

    // ── Pass 2: merge partials, normalise, write f32 output ──────
    // Both p1 and pass2 are generic over T; for this end-to-end
    // numerical-stability check we keep T = F32 throughout. The bf16
    // narrow-write path is exercised separately by the typed
    // `aura_flash_sdpa` correctness test.
    let mut p2_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p2_buffers.insert("o_partials".into(), o_partials_bytes.clone());
    p2_buffers.insert("m_partials".into(), m_partials_bytes.clone());
    p2_buffers.insert("l_partials".into(), l_partials_bytes.clone());
    p2_buffers.insert("output".into(), vec![0u8; q_heads * dim * 4]); // f32
    p2_buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    p2_buffers.insert("num_blocks".into(), (num_blocks as u32).to_le_bytes().to_vec());

    let mut p2_kernel = aura_flash_pass2_d128::kernel_ir_for(DType::F32);
    p2_kernel.mode = KernelMode::Reduction;

    // One TG per q_idx, 32 threads/TG (1 simdgroup).
    let p2_result = ctx
        .dispatch_with_grid(&p2_kernel, &p2_buffers, &BTreeMap::new(), [q_heads, 1, 1], [32, 1, 1])
        .expect("flash_pass2 dispatch should succeed");

    let output_bytes = p2_result.outputs.get("output").expect("`output` buffer");
    let actual = bytes_to_f32_vec(output_bytes);

    // f32 output: tight tolerance (the only rounding is from the
    // compressed-domain dequant + per-token softmax accumulation).
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "aura_flash kb4 vb2 d128: max |diff| = {diff:.2e} (expected < 1e-4)",);
}
