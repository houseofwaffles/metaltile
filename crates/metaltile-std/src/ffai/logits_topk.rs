//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Top-K filter — masking variant.
//!
//! The full top-K filter pipeline is:
//!
//!   1. Find the K-th largest logit value: `threshold = argpartition(logits, -K)`
//!   2. For every logit, if `logit >= threshold` keep it; else set to `-inf`
//!
//! Step 1 is a selection / partial-sort. On GPU at typical serving K (50, 100)
//! and Qwen-scale vocab (152K) the per-call cost is dominated by Metal command-
//! buffer overhead, not arithmetic — a CPU argpartition + threshold-pass is
//! roughly the same wall-clock as a GPU select kernel and one less dispatch.
//! This file ships the GPU mask kernel and leaves threshold computation to
//! the caller. A future PR can add a GPU-side selection kernel when serving
//! batch sizes make a single fused dispatch pull ahead.
//!
//! Caller contract:
//!   - Compute `threshold` = the K-th largest value (descending) on the host.
//!   - Pass it as the constexpr `threshold` parameter.
//!   - Logits below `threshold` are replaced with `-INFINITY` (this is the
//!     standard sentinel — downstream softmax sees `exp(-inf) = 0` and the
//!     filtered tokens contribute zero probability).
//!
//! Generic over T. Grid3D one-thread-per-vocab-position.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Grid3D.** One thread per vocab position.
//! - **Grid: `[ceil(n / TPG), 1, 1]`, TG: `[TPG, 1, 1]`** (TPG = 256 is the
//!   tested geometry; the kernel is pure elementwise so any TPG works).
//! - **`n = grid.x * tg.x`** — caller sizes the dispatch so the total
//!   thread count exactly matches the vocab length. Threads past `n`
//!   would read/write out of bounds; the runtime should not overshoot.
//! - **No `threadgroup_*` / `simd_*` cooperation** — every thread is
//!   independent. The only invariant is the threshold semantic above.

use metaltile::kernel;

#[kernel]
pub fn logits_topk_mask<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] threshold: f32) {
    let i = program_id::<0>();
    let v = load(inp[i]).cast::<f32>();
    // `select(cond, lhs, rhs)` returns lhs when cond is true.
    // Keep value when v >= threshold; otherwise sentinel to -inf.
    let neg_inf = neg_infinity();
    let masked = select(v >= threshold, v, neg_inf);
    store(out[i], masked.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::logits_topk_mask;
    use crate::utils::{pack_f32, unpack_f32};

    /// K-th largest value (descending) — how callers pre-compute the cutoff.
    fn kth_largest(logits: &[f32], k: usize) -> f32 {
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted[k - 1]
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_logits_topk_mask(dt: DType) -> TestSetup {
        let (n, k) = (1024usize, 50usize);
        // `sin` produces distinct floats so there are no ties at the
        // threshold; after dtype rounding the threshold is recomputed on
        // the rounded values so the keep test stays exact.
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 5.0).collect();
        let rounded = unpack_f32(&pack_f32(&logits, dt), dt);
        let threshold = kth_largest(&rounded, k);
        let expected: Vec<f32> =
            rounded.iter().map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY }).collect();
        TestSetup::new(logits_topk_mask::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("inp", pack_f32(&logits, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("threshold", threshold)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}

/// New-syntax benchmark for `logits_topk_mask` at Qwen3 vocab scale
/// (Grid3D, one thread per vocab position).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::logits_topk_mask;

    #[bench(name = "ffai/logits_processors/topk_mask", dtypes = [f32, f16, bf16])]
    fn bench_logits_topk_mask(dt: DType) -> BenchSetup {
        let n = 152_064usize;
        BenchSetup::new(logits_topk_mask::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("threshold", 0.0f32)
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }
}
