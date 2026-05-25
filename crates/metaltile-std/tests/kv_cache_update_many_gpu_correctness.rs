//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end GPU correctness for `ffai::kv_cache_update_many`.
//!
//! `kv_cache_update_many` collapses the per-token T-loop of
//! `kv_cache_update` into ONE dispatch by lifting `position` from a
//! constexpr to a `Tensor<u32>` of length T and adding a row axis to the
//! source layout. Because the kernel is a pure data move (no arithmetic
//! at all), the cleanest oracle is `kv_cache_update` itself looped
//! row-by-row — that pins that the batched kernel writes to bit-equal
//! cache slots as the per-row primitive.
//!
//! For each `(T, n_kv_heads, head_dim)` shape we:
//!   1. Pre-fill the cache with a sentinel so untouched slots can be
//!      audited.
//!   2. Generate random source data of shape [T, n_kv_heads, head_dim]
//!      and pick T non-overlapping positions in [0, max_seq).
//!   3. Run `kv_cache_update_many` in one dispatch.
//!   4. Run `kv_cache_update` once per row over a fresh sentinel-filled
//!      cache as the reference oracle.
//!   5. Assert byte-identical output (tol=0.0 — same as the single-token
//!      kernel's bench spec).
//!
//! Non-overlapping positions matter: if two rows write the same slot,
//! the "last write wins" ordering of the parallel kernel becomes a race
//! the per-row loop can't model. Real callers always write distinct
//! positions (one per token in the prefill batch), so this restriction
//! matches production.
//!
//! Dtype coverage: f32 / f16 / bf16. Tolerance 0.0 across the board
//! because no float arithmetic happens — the kernel issues `load` /
//! `store` of T-typed values without any cast through `f32`.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::kv_cache::kv_cache_update;
use metaltile_std::ffai::kv_cache_update_many::kv_cache_update_many;

/// Dispatch the batched `kv_cache_update_many` over T rows and read back
/// the full cache as f32.
///
/// `init_cache` is the starting cache contents (`[n_kv_heads, max_seq,
/// head_dim]`) — the test pre-fills it with a sentinel so untouched
/// slots can be verified to still hold it. The kernel only writes
/// `[*, positions[r], *]` for each r, so every other slot must come
/// through verbatim.
#[allow(clippy::too_many_arguments)]
fn run_kv_cache_update_many(
    src: &[f32],
    positions: &[u32],
    init_cache: &[f32],
    dt: Dt,
    n_tokens: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(src, dt));
    buffers.insert("positions".into(), pack_u32_bytes(positions));
    buffers.insert("out".into(), pack_bytes(init_cache, dt));
    buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), max_seq.to_le_bytes().to_vec());
    buffers
        .insert("n_kv_heads_x_head_dim".into(), (n_kv_heads * head_dim).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kv_cache_update_many::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per source element. Total = T * n_kv_heads *
    // head_dim. We dispatch over axis 0 only — flat program_id::<0>()
    // recovers `(r, h, d)` inside the kernel. Picking tpg = head_dim
    // makes `groups * tpg == total` exactly, so there are no
    // out-of-bounds threads (the DSL has no early-return primitive,
    // and head_dim ∈ [64, 256] in our test cases fits comfortably under
    // the 1024-thread per-threadgroup cap).
    let total_threads = (n_tokens * n_kv_heads * head_dim) as usize;
    let tpg = head_dim as usize;
    let groups = total_threads / tpg;
    assert_eq!(groups * tpg, total_threads, "tpg must divide total threads exactly");
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("kv_cache_update_many dispatch");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
}

/// Per-row oracle: invoke `kv_cache_update` once per row with that row's
/// scalar position constexpr. Same single-token data move, dispatched T
/// times. This is the "before" state of the prefill T-loop the batched
/// kernel is replacing — exactly what FFAI's `KVCache.appendRangeOnGPU`
/// currently calls in a loop.
fn run_kv_cache_update_per_row(
    src: &[f32],
    positions: &[u32],
    init_cache: &[f32],
    dt: Dt,
    n_tokens: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
) -> Vec<f32> {
    let row_elems = (n_kv_heads * head_dim) as usize;
    let cache_elems = (n_kv_heads * max_seq * head_dim) as usize;

    // Start from the same initial state the batched dispatch saw — but
    // because each per-row dispatch returns the FULL output cache, we
    // must thread the post-write cache into the next iteration's input.
    let mut cache = init_cache.to_vec();
    assert_eq!(cache.len(), cache_elems);

    let ctx = Context::new().expect("Context::new on macOS");

    for r in 0..n_tokens as usize {
        let row_slice = &src[r * row_elems..(r + 1) * row_elems];
        let position = positions[r];

        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("src".into(), pack_bytes(row_slice, dt));
        buffers.insert("out".into(), pack_bytes(&cache, dt));
        buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
        buffers.insert("max_seq".into(), max_seq.to_le_bytes().to_vec());
        buffers.insert("position".into(), position.to_le_bytes().to_vec());

        let mut kernel = kv_cache_update::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Grid3D;

        // Single-token primitive's grid: total threads = n_kv_heads *
        // head_dim. Match the existing `kv_cache_update_writes_to_correct_slot_f32`
        // dispatch shape: groups=1, tpg=total along axis 0. Always
        // ≤ 8*256 = 2048 — split with the same tpg=head_dim recipe so
        // we never blow the per-threadgroup cap.
        let total_threads = row_elems;
        let tpg = head_dim as usize;
        let groups = total_threads / tpg;
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("kv_cache_update per-row dispatch");

        let out = unpack_bytes(result.outputs.get("out").expect("out buffer"), dt);
        cache = out;
    }

    cache
}

/// Deterministic pseudo-random source data — same pattern shape as the
/// rope_llama_many test so dtype round-trips look familiar.
fn make_src(n_tokens: u32, n_kv_heads: u32, head_dim: u32, seed: u32) -> Vec<f32> {
    let n = (n_tokens * n_kv_heads * head_dim) as usize;
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed) as f32;
            ((x * 0.0001).sin() + (x * 0.013).cos()) * 0.5
        })
        .collect()
}

/// T non-overlapping positions in [0, max_seq). Builds [0..max_seq),
/// shuffles deterministically via a small LCG, takes the first T. Pins
/// non-overlap by construction — see the file docstring on why duplicates
/// would race against the per-row oracle.
fn make_positions(n_tokens: u32, max_seq: u32, seed: u32) -> Vec<u32> {
    assert!(n_tokens <= max_seq, "need at least T distinct positions in [0, max_seq)");
    let mut pool: Vec<u32> = (0..max_seq).collect();
    let mut state = (seed as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    // Fisher-Yates with the LCG. Same hash recipe as the SRHT rotation
    // helper in common/mod.rs — keeps the test reproducible.
    for i in (1..pool.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        pool.swap(i, j);
    }
    pool.truncate(n_tokens as usize);
    pool
}

/// (T, n_kv_heads, head_dim) cases requested in the issue.
const CASES: &[(u32, u32, u32)] = &[
    (2, 4, 64),
    (8, 8, 128),
    (64, 2, 256),
];

const MAX_SEQ: u32 = 1024;
const SENTINEL: f32 = 999.0;

fn check_dtype(dt: Dt) {
    for &(n_tokens, n_kv_heads, head_dim) in CASES {
        let src = make_src(n_tokens, n_kv_heads, head_dim, 0x1234);
        let positions = make_positions(n_tokens, MAX_SEQ, 0x5678);
        let cache_elems = (n_kv_heads * MAX_SEQ * head_dim) as usize;
        let init_cache = vec![SENTINEL; cache_elems];

        let many = run_kv_cache_update_many(
            &src,
            &positions,
            &init_cache,
            dt,
            n_tokens,
            n_kv_heads,
            head_dim,
            MAX_SEQ,
        );
        let oracle = run_kv_cache_update_per_row(
            &src,
            &positions,
            &init_cache,
            dt,
            n_tokens,
            n_kv_heads,
            head_dim,
            MAX_SEQ,
        );

        assert_eq!(many.len(), oracle.len(), "length mismatch");
        // Bit-identical: pure data move, no float arithmetic. Both
        // kernels read the same source bytes and write to the same
        // cache slots — any divergence is an indexing bug.
        for (idx, (a, e)) in many.iter().zip(oracle.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{dt:?}: shape=(T={n_tokens}, H={n_kv_heads}, D={head_dim}) idx={idx}: \
                 many={a} oracle={e}",
            );
        }

        // Spot-check: untouched slots really do still hold the sentinel
        // in the batched output (i.e. we didn't accidentally write past
        // the intended slice). The oracle path's per-row dispatches
        // would also leak any such bug, but checking the batched
        // output directly localises it to the new kernel.
        //
        // Round-trip the sentinel through the test dtype — bf16 / f16
        // load+store through the GPU buffer come back quantised, so
        // comparing against the raw `f32` constant would false-positive
        // on the lower-precision dtypes (bf16 turns 999.0 into 1000.0
        // because its 7-bit mantissa can't represent 999 exactly).
        let sentinel_rt = dt.round(SENTINEL);
        let pos_set: std::collections::BTreeSet<u32> = positions.iter().copied().collect();
        for h in 0..n_kv_heads as usize {
            for p in 0..MAX_SEQ as usize {
                if pos_set.contains(&(p as u32)) {
                    continue;
                }
                for d in 0..head_dim as usize {
                    let cache_idx =
                        h * (MAX_SEQ as usize) * (head_dim as usize) + p * (head_dim as usize) + d;
                    assert_eq!(
                        many[cache_idx].to_bits(),
                        sentinel_rt.to_bits(),
                        "{dt:?}: untouched slot (h={h}, pos={p}, d={d}) = {} (sentinel {sentinel_rt})",
                        many[cache_idx],
                    );
                }
            }
        }
    }
}

#[test]
fn kv_cache_update_many_matches_per_row_oracle_f32() {
    let _g = gpu_lock();
    check_dtype(Dt::F32);
}

#[test]
fn kv_cache_update_many_matches_per_row_oracle_f16() {
    let _g = gpu_lock();
    check_dtype(Dt::F16);
}

#[test]
fn kv_cache_update_many_matches_per_row_oracle_bf16() {
    let _g = gpu_lock();
    check_dtype(Dt::Bf16);
}
