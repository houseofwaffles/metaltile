//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 Lightning Indexer — top-k index selection over per-position
//! aggregate scores.
//!
//! Sits downstream of [`ffai_dsv4_indexer_score`]. Takes the
//! `score[n_kv]` produced by the indexer score kernel and returns the
//! indices of the K largest entries (DSv4 production: `K = 512`). The
//! returned indices feed CSA's sparse-gather SDPA inner loop.
//!
//! ## Single-block bitonic top-k
//!
//! For `n_kv <= 1024` the entire score array fits in one threadgroup's
//! shared memory. We do a parallel bitonic sort **descending** over
//! `(score, original_index)` pairs and emit the first `k` indices.
//! Same Batcher-pattern compare-and-swap as
//! [`crate::mlx::sort::mt_sort`] but with parallel storage for an
//! `u32` index tag so the original cache-position survives the swaps.
//!
//! Scores below `n_kv` are padded with `-INFINITY` so the sort
//! relegates them to the back regardless of K — caller does NOT need
//! a power-of-two `n_kv`, only an upper bound of 1024.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG = 256** (each thread owns 4 sort slots → 1024 total).
//! - **Grid: 1 threadgroup** — single-block sort. For `n_kv > 1024`
//!   the multi-block-merge variant is the follow-up; the cleanest
//!   path is `mt_sort` per chunk + a `mt_merge`-style co-rank merge
//!   that keeps the index tag through the merge.
//! - `n_kv <= 1024` enforced at the call site.
//! - `k <= 1024` enforced at the call site.

use metaltile::kernel;

#[kernel]
pub fn ffai_dsv4_indexer_topk_block(
    score: Tensor<f32>,
    mut out_indices: Tensor<u32>,
    #[constexpr] n_kv: u32,
    #[constexpr] k: u32,
) {
    let t = tid;
    // Parallel TG buffers — sort the (score, index) pair in lockstep so
    // each compare-and-swap moves both halves together.
    threadgroup_alloc("tg_scores", 1024, f32);
    threadgroup_alloc("tg_idx", 1024, u32);

    // Load 4 slots per thread. Beyond `n_kv` pads with `-INFINITY` so
    // the descending sort sinks them to the tail and the first `k`
    // slots are guaranteed to be real cache positions (when `k <=
    // n_kv`).
    let neg_inf = neg_infinity();
    for _e in range(0u32, 4u32, 1u32) {
        let gi = t * 4u32 + _e;
        let valid = gi < n_kv;
        let raw_score = select(valid, load(score[gi]), neg_inf);
        threadgroup_store("tg_scores", gi, raw_score);
        threadgroup_store("tg_idx", gi, gi);
    }
    threadgroup_barrier();

    // Bitonic sort DESCENDING. 10 outer stages (log2(1024)). The
    // mt_sort kernel's barrier discipline (`if flip >= 7`) is reused:
    // strides ≤ 64 stay within one simdgroup so the implicit lane
    // ordering is enough; strides > 64 need a TG barrier.
    for _k_stage in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k_stage, 1u32) {
            let flip = _k_stage - _jb - 1u32;
            if flip >= 7u32 {
                threadgroup_barrier();
            }
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a_score = threadgroup_load("tg_scores", gi);
                    let b_score = threadgroup_load("tg_scores", partner);
                    let a_idx = threadgroup_load("tg_idx", gi);
                    let b_idx = threadgroup_load("tg_idx", partner);
                    let dir = (gi >> _k_stage) & 1u32;
                    // Descending: dir=0 keeps larger first → swap if `a < b`.
                    // dir=1 flips the sub-block (bitonic property).
                    let want_swap = select(dir == 0u32, a_score < b_score, a_score > b_score);
                    threadgroup_store("tg_scores", gi, select(want_swap, b_score, a_score));
                    threadgroup_store("tg_scores", partner, select(want_swap, a_score, b_score));
                    threadgroup_store("tg_idx", gi, select(want_swap, b_idx, a_idx));
                    threadgroup_store("tg_idx", partner, select(want_swap, a_idx, b_idx));
                }
            }
        }
    }
    threadgroup_barrier();

    // Emit the first `k` indices. Threads whose slots fall past `k`
    // skip — the rest of `tg_idx` is sorted but unused.
    for _e in range(0u32, 4u32, 1u32) {
        let gi = t * 4u32 + _e;
        if gi < k {
            store(out_indices[gi], threadgroup_load("tg_idx", gi));
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_indexer_topk_block;
    use crate::utils::pack_f32;

    /// CPU reference: argsort scores descending, take first `k`
    /// original indices. Ties broken by ascending index (stable).
    fn cpu_topk_indices(scores: &[f32], n_kv: usize, k: usize) -> Vec<u32> {
        let mut paired: Vec<(f32, u32)> =
            scores[..n_kv].iter().enumerate().map(|(i, &s)| (s, i as u32)).collect();
        paired.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
        });
        paired.iter().take(k).map(|(_, i)| *i).collect()
    }

    fn setup(n_kv: usize, k: usize) -> TestSetup {
        // Generate distinct scores so the top-k set is well-defined
        // (no ties at the K-th boundary that could swap places between
        // CPU and GPU stable-sort traversals).
        let scores: Vec<f32> = (0..n_kv).map(|i| (i as f32 * 0.0379 - 1.7).sin() * 2.3).collect();
        let expected = cpu_topk_indices(&scores, n_kv, k);
        // Pack u32 expected as raw little-endian bytes for the test
        // framework — same path mxfp4_dequant uses for u32 outputs.
        let expected_bytes: Vec<u8> = expected.iter().flat_map(|i| i.to_le_bytes()).collect();
        TestSetup::new(ffai_dsv4_indexer_topk_block::kernel_ir())
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("score", pack_f32(&scores, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out_indices", k, DType::U32))
            .constexpr("n_kv", n_kv as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out_indices", expected_bytes, DType::U32))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    /// Small shape — 64 cache positions, top-8 — sanity check.
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_topk_small(_dt: DType) -> TestSetup { setup(64, 8) }

    /// Mid shape — 256 positions, top-32.
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_topk_mid(_dt: DType) -> TestSetup { setup(256, 32) }

    /// Full-block — 1024 positions, top-512 (DSv4 production `K`).
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_topk_dsv4(_dt: DType) -> TestSetup { setup(1024, 512) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_indexer_topk_block;

    #[bench(name = "ffai/dsv4_indexer_topk_block", dtypes = [f32])]
    fn bench_topk(_dt: DType) -> BenchSetup {
        let (n_kv, k) = (1024usize, 512usize);
        BenchSetup::new(ffai_dsv4_indexer_topk_block::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("score", n_kv, DType::F32))
            .buffer(BenchBuffer::zeros("out_indices", k, DType::U32).output())
            .constexpr("n_kv", n_kv as u32)
            .constexpr("k", k as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n_kv * 4 + k * 4) as u64)
    }
}
