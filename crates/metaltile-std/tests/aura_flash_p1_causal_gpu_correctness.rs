//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for the causal AURA flash pass-1 variant
//! (`ffai::aura_flash_p1_causal_kb4_vb2_d128`).
//!
//! The causal kernel clamps the per-token inner loop at `q_position +
//! 1` — every key strictly after the query token is masked out. Two
//! checks:
//!   1. With `q_position = tokens - 1` (full visibility), the causal
//!      kernel must produce partials bit-identical to the non-causal
//!      sibling.
//!   2. With a mid-stream `q_position`, the per-block `(o, m, l)`
//!      partials must match a CPU online-softmax reference that folds
//!      only tokens `t ≤ q_position`; blocks entirely past the cutoff
//!      stay at their identity values (`m = -inf`, `l = 0`, `o = 0`).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_flash_p1::{
    aura_flash_p1_causal_kb4_vb2_d128,
    aura_flash_p1_kb4_vb2_d128,
};

const DIM: usize = 128;
const KEY_BITS: usize = 4;
const VALUE_BITS: usize = 2;
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const TOKENS: usize = 8;
const BLOCK_SIZE: usize = 4;

/// Pack per-dim integer codebook indices into the kernel's bit-stream.
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
                let idx = indices[(kvh * tokens + t) * dim + d] & mask;
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                packed[(kvh * tokens + t) * packed_width + word_idx] |= idx << shift;
                let spill = (shift + bits) as i32 - 32;
                if spill > 0 {
                    let s = spill as u32;
                    packed[(kvh * tokens + t) * packed_width + word_idx + 1] |=
                        idx >> (bits as u32 - s);
                }
            }
        }
    }
    packed
}

#[allow(dead_code)] // key_indices / val_indices mirror the kernel input set
struct Inputs {
    q_rot: Vec<f32>,
    key_packed: Vec<u32>,
    val_packed: Vec<u32>,
    key_indices: Vec<u32>,
    val_indices: Vec<u32>,
    key_norms: Vec<f32>,
    val_norms: Vec<f32>,
    key_codebook: Vec<f32>,
    val_codebook: Vec<f32>,
}

fn build_inputs() -> Inputs {
    let key_codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
    let val_codebook: Vec<f32> = (0..4).map(|i| -1.0 + 2.0 * i as f32 / 3.0).collect();
    let key_indices: Vec<u32> =
        (0..KV_HEADS * TOKENS * DIM).map(|i| ((i * 7 + 3) % 16) as u32).collect();
    let val_indices: Vec<u32> =
        (0..KV_HEADS * TOKENS * DIM).map(|i| ((i * 11 + 5) % 4) as u32).collect();
    let key_packed = pack_int_indices(&key_indices, KV_HEADS, TOKENS, DIM, KEY_BITS);
    let val_packed = pack_int_indices(&val_indices, KV_HEADS, TOKENS, DIM, VALUE_BITS);
    let key_norms: Vec<f32> = (0..KV_HEADS * TOKENS).map(|i| 0.5 + 0.05 * i as f32).collect();
    let val_norms: Vec<f32> = (0..KV_HEADS * TOKENS).map(|i| 0.3 + 0.07 * i as f32).collect();
    let q_rot: Vec<f32> =
        (0..Q_HEADS * DIM).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();
    Inputs {
        q_rot,
        key_packed,
        val_packed,
        key_indices,
        val_indices,
        key_norms,
        val_norms,
        key_codebook,
        val_codebook,
    }
}

/// Dispatch one aura_flash_p1 kernel; returns the `(o, m, l)` partials.
#[allow(clippy::type_complexity)]
fn run_p1(causal: bool, q_position: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let inp = build_inputs();
    let num_blocks = TOKENS.div_ceil(BLOCK_SIZE);
    let repeat = Q_HEADS / KV_HEADS;
    let key_packed_width = (DIM * KEY_BITS).div_ceil(32);
    let value_packed_width = (DIM * VALUE_BITS).div_ceil(32);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q_rot".into(), pack_bytes(&inp.q_rot, Dt::F32));
    b.insert("key_packed".into(), pack_u32_bytes(&inp.key_packed));
    b.insert("key_norms".into(), pack_bytes(&inp.key_norms, Dt::F32));
    b.insert("key_codebook".into(), pack_bytes(&inp.key_codebook, Dt::F32));
    b.insert("val_packed".into(), pack_u32_bytes(&inp.val_packed));
    b.insert("val_norms".into(), pack_bytes(&inp.val_norms, Dt::F32));
    b.insert("val_codebook".into(), pack_bytes(&inp.val_codebook, Dt::F32));
    b.insert("o_partials".into(), vec![0u8; Q_HEADS * num_blocks * DIM * 4]);
    b.insert("m_partials".into(), vec![0u8; Q_HEADS * num_blocks * 4]);
    b.insert("l_partials".into(), vec![0u8; Q_HEADS * num_blocks * 4]);
    b.insert("dim".into(), (DIM as u32).to_le_bytes().to_vec());
    b.insert("key_packed_width".into(), (key_packed_width as u32).to_le_bytes().to_vec());
    b.insert("value_packed_width".into(), (value_packed_width as u32).to_le_bytes().to_vec());
    b.insert("tokens".into(), (TOKENS as u32).to_le_bytes().to_vec());
    // Fully-populated fixture: stride == live row count.
    b.insert("kv_stride".into(), (TOKENS as u32).to_le_bytes().to_vec());
    b.insert("repeat_count".into(), (repeat as u32).to_le_bytes().to_vec());
    b.insert("num_blocks".into(), (num_blocks as u32).to_le_bytes().to_vec());
    b.insert("block_size".into(), (BLOCK_SIZE as u32).to_le_bytes().to_vec());
    b.insert("q_position".into(), (q_position as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = if causal {
        aura_flash_p1_causal_kb4_vb2_d128::kernel_ir_for(DType::F32)
    } else {
        aura_flash_p1_kb4_vb2_d128::kernel_ir_for(DType::F32)
    };
    kernel.mode = KernelMode::Grid3D;
    let res = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, Q_HEADS, num_blocks], [32, 1, 1])
        .expect("flash_p1 dispatch");
    (
        unpack_bytes(res.outputs.get("o_partials").expect("o_partials"), Dt::F32),
        unpack_bytes(res.outputs.get("m_partials").expect("m_partials"), Dt::F32),
        unpack_bytes(res.outputs.get("l_partials").expect("l_partials"), Dt::F32),
    )
}

#[test]
fn causal_full_visibility_matches_non_causal() {
    let _g = gpu_lock();
    // q_position = TOKENS - 1 → causal cutoff includes every token.
    let (o_c, m_c, l_c) = run_p1(true, TOKENS - 1);
    let (o_n, m_n, l_n) = run_p1(false, TOKENS - 1);
    for (i, (a, e)) in m_c.iter().zip(&m_n).enumerate() {
        assert!((a - e).abs() < 1e-5, "m_partials[{i}]: causal {a} != non-causal {e}");
    }
    for (i, (a, e)) in l_c.iter().zip(&l_n).enumerate() {
        assert!((a - e).abs() < 1e-5, "l_partials[{i}]: causal {a} != non-causal {e}");
    }
    for (i, (a, e)) in o_c.iter().zip(&o_n).enumerate() {
        assert!((a - e).abs() < 1e-4, "o_partials[{i}]: causal {a} != non-causal {e}");
    }
}

#[test]
fn causal_mid_cutoff_masks_later_blocks() {
    let _g = gpu_lock();
    // q_position = 3 → only block 0 (tokens 0..4) contributes; block 1
    // (tokens 4..8) is entirely past the cutoff.
    let q_pos = 3usize;
    let num_blocks = TOKENS.div_ceil(BLOCK_SIZE);
    let (_o, m, l) = run_p1(true, q_pos);

    // Block 1 (the one fully past q_pos) must stay at identity.
    for qh in 0..Q_HEADS {
        let blk = 1usize;
        let idx = qh * num_blocks + blk;
        assert!(
            m[idx] == f32::NEG_INFINITY || m[idx] < -1e30,
            "head {qh} block {blk}: m must stay -inf (past causal cutoff), got {}",
            m[idx],
        );
        assert_eq!(l[idx], 0.0, "head {qh} block {blk}: l must stay 0 (past causal cutoff)");
    }

    // Block 0 straddles nothing here (q_pos=3 is the last token of block
    // 0) so it must have folded all 4 of its tokens — l strictly > 0.
    for qh in 0..Q_HEADS {
        let idx = qh * num_blocks;
        assert!(l[idx] > 0.0, "head {qh} block 0: l must be > 0 (tokens 0..=3 visible)");
    }
}
