//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Min-p (minimum-probability) logits filter for the sampling pipeline.
//!
//! Min-p sampling keeps every token whose probability is at least
//! `min_p` times the probability of the most-likely token, and masks
//! the rest:
//!
//!   keep token i  ⇔  P(i) ≥ min_p · P_max
//!
//! Working in logit space avoids a full softmax. For any shift `C`,
//! `P(i) / P_max = exp(logit_i − logit_max)`, so the keep test is
//! simply `exp(logit_i − logit_max) ≥ min_p`. The kernel finds the
//! row max with one threadgroup reduction, then masks every logit
//! below the cutoff to `-INFINITY` in a second pass. Downstream
//! `softmax_categorical_sample` sees `exp(-inf) = 0`, so masked tokens
//! contribute zero probability.
//!
//! This is the reduction-mode sibling of `logits_topk_mask`: top-K
//! needs a host-computed K-th-largest threshold, but min-p's cutoff is
//! defined purely by the row max, so the whole filter fits in one
//! self-contained GPU kernel — no host round-trip, no sort.
//!
//! Reduction-mode, generic over T; the max and the ratio are computed
//! in f32 so f16/bf16 logits don't drift. One threadgroup per row;
//! `n` is the vocab length, looped so any `n` works at any
//! (multiple-of-32) threadgroup size.
//!
//! Caller contract: `0 < min_p < 1`. As `min_p → 0` nothing is masked;
//! as `min_p → 1` only the argmax (and exact ties) survive. A typical
//! serving value is 0.05–0.1.

use metaltile::kernel;

#[kernel]
pub fn logits_min_p_mask<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] min_p: f32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    // Pass 1: threadgroup-wide max of the row's logits.
    let mut lm = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        lm = max(lm, load(inp[_i]).cast::<f32>());
    }
    let row_max = reduce_max(lm);
    // Pass 2: keep a logit iff exp(logit - row_max) >= min_p, else -inf.
    let neg_inf = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        store(out[_i], select(exp(v - row_max) >= min_p, v, neg_inf).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::logits_min_p_mask;
    use crate::utils::{pack_f32, unpack_f32};

    /// CPU oracle: per row, keep `v` iff `exp(v − row_max) ≥ min_p`.
    fn cpu_min_p_mask(logits: &[f32], n: usize, rows: usize, min_p: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let base = r * n;
            let row = &logits[base..base + n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (i, &v) in row.iter().enumerate() {
                out[base + i] = if (v - m).exp() >= min_p { v } else { f32::NEG_INFINITY };
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_logits_min_p_mask(dt: DType) -> TestSetup {
        let (n, rows, min_p) = (320usize, 4usize, 0.1f32);
        // Wide-spread ramp so the cutoff lands in a gap between distinct
        // weights — no token sits on the keep/mask boundary where a GPU
        // vs CPU `exp` ULP could flip the result.
        let logits: Vec<f32> = (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect();
        let rounded = unpack_f32(&pack_f32(&logits, dt), dt);
        let expected = cpu_min_p_mask(&rounded, n, rows, min_p);
        TestSetup::new(logits_min_p_mask::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&logits, dt), dt))
            .input(TestBuffer::zeros("out", n * rows, dt))
            .constexpr("n", n as u32)
            .constexpr("min_p", min_p)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }
}

/// New-syntax benchmark for `logits_min_p_mask` at Qwen3 vocab scale
/// (Reduction, one TG per row).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::logits_min_p_mask;

    #[bench(name = "ffai/logits_processors/min_p_mask", dtypes = [f32, f16, bf16])]
    fn bench_logits_min_p_mask(dt: DType) -> BenchSetup {
        let (n, rows) = (152_064usize, 2usize);
        BenchSetup::new(logits_min_p_mask::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n * rows, dt))
            .buffer(BenchBuffer::zeros("out", n * rows, dt).output())
            .constexpr("n", n as u32)
            .constexpr("min_p", 0.1f32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * n * rows * dt.size_bytes()) as u64)
    }
}
