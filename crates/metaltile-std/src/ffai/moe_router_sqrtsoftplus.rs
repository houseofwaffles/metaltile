//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MoE router — sqrt(softplus(·)) + bias-correction scoring.
//!
//! The DeepSeek V4 (Flash + Pro) routing pattern, replacing V3.2's
//! sigmoid+bias gate per the V4 paper's `scoring_func: sqrtsoftplus`.
//!
//! ```text
//!     score_unbiased[e]  =  sqrt(softplus(logits[e]))     // routing signal
//!     score_biased[e]    =  score_unbiased[e] + bias[e]   // selection signal
//! ```
//!
//! Top-k selection uses `score_biased`; the gather weight downstream
//! is `score_unbiased * routed_scaling_factor` (the bias is for
//! aux-loss-free load balancing, not weight magnitude — V3.2 / V4
//! `noaux_tc` mechanism). Both outputs ship side-by-side so the
//! downstream top-k + normalize + scale chain consumes whichever it
//! needs without re-running the scoring math.
//!
//! Compared to the sister `ffai_moe_router_sigmoid_bias`:
//!   - sigmoid+bias: `s = sigmoid(x)`; bounded in (0, 1).
//!   - sqrtsoftplus: `s = sqrt(log(1 + exp(x)))`; unbounded, larger
//!     dynamic range — paired with the bias-correction for selection.
//!
//! Pitfall: softplus(x) overflows in fp32 for large x. We use the
//! numerically-stable form `softplus(x) = max(x, 0) + log(1 + exp(-|x|))`
//! to avoid `exp` overflow on positive tails.
//!
//! ## ABI
//!
//! ```text
//!   logits           [n_experts] f32  — pre-scoring router output (`W_router · x`)
//!   bias             [n_experts] f32  — per-expert routing-bias (noaux_tc)
//!   score_unbiased   [n_experts] f32  — out: `sqrt(softplus(logits))`
//!   score_biased     [n_experts] f32  — out: `score_unbiased + bias`
//! ```
//!
//! Grid is 1D elementwise — one thread per expert. Modal n_experts
//! across the production checkpoints is 288 (V4-Flash, ~256K-ctx
//! Pareto point); the kernel scales fine to V4-Pro's larger counts.

use metaltile::kernel;

// Bare `#[kernel]` — non-generic, all-f32 kernel doesn't fit the
// legacy `bench(...)` shape; declarative `#[bench]` below registers
// for `tile bench`.
#[kernel]
pub fn ffai_moe_router_sqrtsoftplus(
    logits: Tensor<f32>,
    bias: Tensor<f32>,
    mut score_unbiased: Tensor<f32>,
    mut score_biased: Tensor<f32>,
) {
    let idx = tid;
    let x = load(logits[idx]);
    // Numerically stable softplus: `max(x, 0) + log(1 + exp(-|x|))`.
    // For x >> 0:  ≈ x + log(1 + tiny)   ≈ x
    // For x << 0:  ≈ 0 + log(1 + e^x)    ≈ e^x
    let ax = select(x >= 0.0f32, x, -x); // |x|
    let sp = select(x >= 0.0f32, x, 0.0f32) + (1.0f32 + (-ax).exp()).ln();
    let s = sp.sqrt();
    store(score_unbiased[idx], s);
    let b = load(bias[idx]);
    store(score_biased[idx], s + b);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_moe_router_sqrtsoftplus;
    use crate::utils::pack_f32;

    /// CPU reference. Mirrors the GPU kernel for tight-tolerance check.
    fn cpu_reference(n_experts: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let dt = DType::F32;
        // Logits cover the full numerically-interesting range: large
        // positive (overflow tail of naive `exp(x)`), large negative
        // (underflow tail of `log(1 + exp(x))` for naive `softplus`),
        // and the dense-around-zero region where routing actually
        // discriminates.
        let logits: Vec<f32> =
            (0..n_experts).map(|i| (i as f32 - n_experts as f32 / 2.0) * 0.25).collect();
        let bias: Vec<f32> = (0..n_experts).map(|i| (i % 13) as f32 * 0.02 - 0.13).collect();
        // Stable softplus matches the kernel exactly.
        let score_unbiased: Vec<f32> = logits
            .iter()
            .map(|&x| {
                let ax = x.abs();
                let pos = if x >= 0.0 { x } else { 0.0 };
                (pos + (1.0 + (-ax).exp()).ln()).sqrt()
            })
            .collect();
        let score_biased: Vec<f32> = score_unbiased.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let _ = dt; // dtype implied by the test_kernel decl
        (logits, bias, score_unbiased, score_biased)
    }

    fn setup(n_experts: usize) -> TestSetup {
        let dt = DType::F32;
        let (logits, bias, score_unbiased, score_biased) = cpu_reference(n_experts);
        TestSetup::new(ffai_moe_router_sqrtsoftplus::kernel_ir())
            .input(TestBuffer::from_vec("logits", pack_f32(&logits, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("score_unbiased", n_experts, dt))
            .input(TestBuffer::zeros("score_biased", n_experts, dt))
            .expect(TestBuffer::from_vec("score_unbiased", pack_f32(&score_unbiased, dt), dt))
            .expect(TestBuffer::from_vec("score_biased", pack_f32(&score_biased, dt), dt))
            .grid_1d(n_experts, 64)
    }

    /// V4-Flash production shape — 288 experts.
    #[test_kernel(dtypes = [f32], tol = [1e-5])]
    fn test_router_sqrtsoftplus_v4_flash(_dt: DType) -> TestSetup { setup(288) }

    /// V4-Pro / future-larger shape sanity check.
    #[test_kernel(dtypes = [f32], tol = [1e-5])]
    fn test_router_sqrtsoftplus_large(_dt: DType) -> TestSetup { setup(512) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_router_sqrtsoftplus;

    #[bench(name = "ffai/moe_router_sqrtsoftplus", dtypes = [f32])]
    fn bench_router(_dt: DType) -> BenchSetup {
        let dt = DType::F32;
        let n_experts = 288usize;
        BenchSetup::new(ffai_moe_router_sqrtsoftplus::kernel_ir())
            .buffer(BenchBuffer::random("logits", n_experts, dt))
            .buffer(BenchBuffer::random("bias", n_experts, dt))
            .buffer(BenchBuffer::zeros("score_unbiased", n_experts, dt).output())
            .buffer(BenchBuffer::zeros("score_biased", n_experts, dt).output())
            .grid_1d(n_experts, 64)
            .bytes_moved((4 * n_experts * dt.size_bytes()) as u64)
    }
}
