//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 Lightning Indexer — per-position aggregate score.
//!
//! For each KV cache position `t`, compute one aggregate score the
//! top-k selector then sorts on:
//!
//! ```text
//!   score[t] = sum_h  w[h] * ReLU( q_idx[h] · k_idx[t, h] )
//! ```
//!
//! Where:
//!   - `q_idx       [n_heads, d_idx]`  — projected query (one token).
//!   - `k_idx       [n_kv, n_heads, d_idx]` — projected key cache.
//!   - `w           [n_heads]`          — per-head learnable scalar.
//!   - `score       [n_kv]`             — output (f32; the top-k pass
//!     downstream needs full mantissa for stable bitonic ordering).
//!
//! DSv4 production shape: `n_heads = 64`, `d_idx = 128`. The
//! indexer's job is to pick 512 cache positions per CSA / HCA layer
//! per decode step; this kernel produces the aggregate score, and a
//! follow-up `ffai_dsv4_indexer_topk` does the bitonic top-512.
//!
//! ## Dispatch
//!
//! 1 thread per cache position. Grid: `n_kv` threads in 1D. Each
//! thread walks `n_heads × d_idx = 8192` MADs on a hot 8 KB per-head
//! row of `k_idx`. Memory-bound (n_kv × 8 KB reads per call), so the
//! arithmetic cost is hidden under DRAM streaming on M5 Max.
//!
//! A fused per-simdgroup variant (1 SG per cache position, lanes
//! split d_idx, simd_sum across the dot) is the perf follow-up — but
//! "correct first" per the playbook §"Testing — what to ALWAYS
//! check" → "End-to-end GPU dispatch test, not just MSL-emit smoke
//! test".

use metaltile::kernel;

#[kernel]
pub fn ffai_dsv4_indexer_score<T>(
    q_idx: Tensor<T>,
    k_idx: Tensor<T>,
    w: Tensor<f32>,
    mut score: Tensor<f32>,
    #[constexpr] n_heads: u32,
    #[constexpr] d_idx: u32,
    #[constexpr] n_kv: u32,
) {
    let t = tid;
    if t < n_kv {
        let mut total = 0.0f32;
        let kv_base = t * n_heads * d_idx;
        for _h in range(0u32, n_heads, 1u32) {
            let q_base = _h * d_idx;
            let k_base = kv_base + _h * d_idx;
            let mut dot = 0.0f32;
            for _d in range(0u32, d_idx, 1u32) {
                let q_val = load(q_idx[q_base + _d]).cast::<f32>();
                let k_val = load(k_idx[k_base + _d]).cast::<f32>();
                dot = dot + q_val * k_val;
            }
            // ReLU-clamp the per-head dot, then weight + accumulate.
            let clamped = select(dot > 0.0f32, dot, 0.0f32);
            let head_w = load(w[_h]);
            total = total + head_w * clamped;
        }
        store(score[t], total);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_indexer_score;
    use crate::utils::{pack_f32, unpack_f32};

    fn cpu_reference(
        q_idx: &[f32],
        k_idx: &[f32],
        w: &[f32],
        n_heads: usize,
        d_idx: usize,
        n_kv: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n_kv];
        for (t, slot) in out.iter_mut().enumerate() {
            let mut total = 0f32;
            for h in 0..n_heads {
                let q_base = h * d_idx;
                let k_base = t * n_heads * d_idx + h * d_idx;
                let mut dot = 0f32;
                for d in 0..d_idx {
                    dot += q_idx[q_base + d] * k_idx[k_base + d];
                }
                let clamped = if dot > 0.0 { dot } else { 0.0 };
                total += w[h] * clamped;
            }
            *slot = total;
        }
        out
    }

    fn setup(n_heads: usize, d_idx: usize, n_kv: usize, dt: DType) -> TestSetup {
        let q_idx: Vec<f32> =
            (0..n_heads * d_idx).map(|i| (i as f32 * 0.013 - 0.4).sin() * 0.8).collect();
        let k_idx: Vec<f32> =
            (0..n_kv * n_heads * d_idx).map(|i| (i as f32 * 0.0073 + 0.2).cos() * 0.7).collect();
        // Mix of positive and negative head weights to exercise both
        // sign branches of the per-head sum (a uniformly-positive w
        // wouldn't catch a sign-flip bug in the per-head accumulator).
        let w: Vec<f32> = (0..n_heads).map(|h| (h as f32 - n_heads as f32 / 2.0) * 0.05).collect();
        let q_dt = unpack_f32(&pack_f32(&q_idx, dt), dt);
        let k_dt = unpack_f32(&pack_f32(&k_idx, dt), dt);
        let expected = cpu_reference(&q_dt, &k_dt, &w, n_heads, d_idx, n_kv);
        TestSetup::new(ffai_dsv4_indexer_score::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("q_idx", pack_f32(&q_idx, dt), dt))
            .input(TestBuffer::from_vec("k_idx", pack_f32(&k_idx, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, DType::F32), DType::F32))
            .input(TestBuffer::zeros("score", n_kv, DType::F32))
            .constexpr("n_heads", n_heads as u32)
            .constexpr("d_idx", d_idx as u32)
            .constexpr("n_kv", n_kv as u32)
            .expect(TestBuffer::from_vec("score", pack_f32(&expected, DType::F32), DType::F32))
            .grid_1d(n_kv, 256)
    }

    /// Small shape sanity — 8 heads × 32 dim × 64 cache positions.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-2, 2e-1])]
    fn test_indexer_score_small(dt: DType) -> TestSetup { setup(8, 32, 64, dt) }

    /// DSv4 production shape — 64 heads × 128 dim × 256 cache
    /// positions (small `n_kv` to keep the test runtime bounded).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 4e-1])]
    fn test_indexer_score_dsv4(dt: DType) -> TestSetup { setup(64, 128, 256, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_indexer_score;

    #[bench(name = "ffai/dsv4_indexer_score", dtypes = [f32, f16, bf16])]
    fn bench_indexer(dt: DType) -> BenchSetup {
        let (n_heads, d_idx, n_kv) = (64usize, 128usize, 4096usize);
        let bytes =
            (n_heads * d_idx + n_kv * n_heads * d_idx) * dt.size_bytes() + n_heads * 4 + n_kv * 4;
        BenchSetup::new(ffai_dsv4_indexer_score::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("q_idx", n_heads * d_idx, dt))
            .buffer(BenchBuffer::random("k_idx", n_kv * n_heads * d_idx, dt))
            .buffer(BenchBuffer::random("w", n_heads, DType::F32))
            .buffer(BenchBuffer::zeros("score", n_kv, DType::F32).output())
            .constexpr("n_heads", n_heads as u32)
            .constexpr("d_idx", d_idx as u32)
            .constexpr("n_kv", n_kv as u32)
            .grid_1d(n_kv, 256)
            .bytes_moved(bytes as u64)
    }
}
