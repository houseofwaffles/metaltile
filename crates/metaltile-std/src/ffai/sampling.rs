//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU sampling kernels — softmax + categorical inverse-CDF walk used
//! by FFAI's `gpu-categorical` decode path (T > 0, no filters). The
//! greedy fast path uses `argmax` instead.
//!
//! Codegen-only. End-to-end sampling correctness lives in FFAI's
//! harness.

use metaltile::kernel;

// Tree reductions for the max-pass and sum-pass each fold 256 threadgroup
// slots → 1 value across 8 power-of-two halving stages.  Originally
// hand-unrolled via `tg_max_step!` / `tg_sum_step!` declarative macros;
// the proc-macro does not expand inner `macro_rules!` so the unrolled
// expansions silently produced no IR.  Replaced with DSL `for` loops
// that yield the same Metal output and survive the proc-macro intact.

// Softmax + categorical sample over a 1D logits tensor. Cooperative
// reduction (256 threads) for max-pass; combined chunked sum-exp +
// inclusive scan + parallel-prefix CDF walk for the categorical pick.
//
// Inputs:
//   inp            — logits [n]
//   out            — token id [1] (u32)
//   temperature_in — temperature [1] (f32, must be > 0)
//   uniform_in     — uniform draw in [0, 1) [1] (f32)
//
// Output is the smallest index `i` such that the cumulative softmax
// (in fp32) up to and including `i` is ≥ `uniform_in * sum_exp`.
//
// Cost: vocab=152K on M5 Max ~563µs median (down from ~8370µs in the
// single-thread CDF walk version, measured via the 1000-iter dispatch
// loop in `tests/softmax_categorical_sample_perf.rs`). ~15× speedup
// dominated by collapsing pass 3's O(n) walk. Lane lid owns a contiguous
// chunk = ceil(n/lsize) ≈ 594 positions; Hillis-Steele inclusive scan
// turns per-lane chunk-partials into per-lane cumulative bounds; the
// lane whose chunk contains `u * total` walks its own chunk serially
// to find the exact index. The full-vocab serial walk (152K ops) is
// replaced by 1 × n/lsize chunk-traverse per lane + an 8-stage scan +
// 1 × n/lsize finalizing walk on the winning lane.
#[kernel]
pub fn softmax_categorical_sample<T>(
    inp: Tensor<T>,
    out: Tensor<u32>,
    temperature_in: Tensor<f32>,
    uniform_in: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let lid = tid;
    let inv_t = 1.0f32 / load(temperature_in[0]);
    // ─── Pass 1: cooperative max reduce (strided) ───────────────────
    let mut local_max = neg_infinity();
    threadgroup_alloc("tg_max", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>() * inv_t;
            local_max = select(v > local_max, v, local_max);
        }
    }
    threadgroup_store("tg_max", lid, local_max);
    threadgroup_barrier();
    // 8-stage power-of-two halving max-reduction (stride 128 → 1).
    for _stage in range(0u32, 8u32, 1u32) {
        let stride = 128u32 >> _stage;
        if lid < stride {
            let ov = threadgroup_load("tg_max", lid + stride);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
    }
    let max_val = threadgroup_load("tg_max", 0u32);
    // ─── Combined pass 2+3: chunk-partial sum-exp → inclusive scan
    //                       → parallel-prefix CDF walk ─────────────
    //
    // Lane lid covers contiguous chunk [lo, hi); `total = tg_cdf[lsize-1]`
    // after the scan replaces the previous standalone sum-exp reduce.
    let chunk = (n + lsize - 1u32) / lsize;
    let lo = lid * chunk;
    let hi_raw = lo + chunk;
    let hi = select(hi_raw > n, n, hi_raw);
    let mut local_partial = 0.0f32;
    for j in range(lo, hi, 1u32) {
        if j < n {
            let v = load(inp[j]).cast::<f32>() * inv_t;
            local_partial = local_partial + exp(v - max_val);
        }
    }
    threadgroup_alloc("tg_cdf", 256);
    threadgroup_store("tg_cdf", lid, local_partial);
    threadgroup_barrier();
    // Hillis-Steele inclusive scan: 8 stages (stride 1 → 128).
    // Underflow-safe: lanes with lid < stride contribute 0 instead of
    // reading from negative indices.
    for _stage in range(0u32, 8u32, 1u32) {
        let stride = 1u32 << _stage;
        let safe_neighbor = select(lid >= stride, lid - stride, lid);
        let raw = threadgroup_load("tg_cdf", safe_neighbor);
        let neighbor_val = select(lid >= stride, raw, 0.0f32);
        threadgroup_barrier();
        let cur = threadgroup_load("tg_cdf", lid);
        threadgroup_store("tg_cdf", lid, cur + neighbor_val);
        threadgroup_barrier();
    }
    let total = threadgroup_load("tg_cdf", lsize - 1u32);
    let target = load(uniform_in[0]) * total;
    let my_cum_end = threadgroup_load("tg_cdf", lid);
    let prev_cum = select(
        lid == 0u32,
        0.0f32,
        threadgroup_load("tg_cdf", select(lid > 0u32, lid - 1u32, lid)),
    );
    // Hit lane: target sits in (prev_cum, my_cum_end]. The strict
    // lower bound means exactly one lane fires at a boundary value.
    let is_hit = (prev_cum < target) & (target <= my_cum_end) & (lo < n);
    if is_hit {
        let mut cum = prev_cum;
        let mut found_idx = hi - 1u32; // fallback: last position in chunk
        let mut done = 0u32;
        for i in range(lo, hi, 1u32) {
            if i < n {
                let v = load(inp[i]).cast::<f32>() * inv_t;
                cum = cum + exp(v - max_val);
                let hit_i = (cum >= target) & (done == 0u32);
                found_idx = select(hit_i, i, found_idx);
                done = select(hit_i, 1u32, done);
            }
        }
        store(out[0], found_idx);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::softmax_categorical_sample;
    use crate::utils::pack_f32;

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.5)]
    fn test_softmax_categorical_sample(dt: DType) -> TestSetup {
        // A dominant spike: exp(spike) ≫ Σ of the rest, so the CDF jumps
        // from ~0 to ~total at `idx`. The smallest index whose cumulative
        // softmax ≥ uniform·total is therefore `idx` for any uniform in
        // (0, 1) — pick 0.5 as the deterministic draw.
        let (n, idx, spike) = (1024usize, 813usize, 30.0f32);
        let mut logits = vec![0.0f32; n];
        logits[idx] = spike;
        TestSetup::new(softmax_categorical_sample::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&logits, dt), dt))
            .input(TestBuffer::zeros("out", 1, DType::U32))
            .input(TestBuffer::from_vec("temperature_in", pack_f32(&[1.0], DType::F32), DType::F32))
            .input(TestBuffer::from_vec("uniform_in", pack_f32(&[0.5], DType::F32), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&[idx as f32], DType::U32), DType::U32))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }
}

/// New-syntax benchmark for `softmax_categorical_sample` at Qwen3 vocab
/// scale (Reduction, single threadgroup; cooperative max + scan + CDF
/// walk). Random logits, fixed temperature and uniform draw.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::softmax_categorical_sample;
    use crate::utils::pack_f32;

    #[bench(name = "ffai/sampling/softmax_categorical_sample", dtypes = [f32, f16, bf16])]
    fn bench_softmax_categorical_sample(dt: DType) -> BenchSetup {
        let n = 152_064usize;
        BenchSetup::new(softmax_categorical_sample::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", 1, DType::U32).output())
            .buffer(BenchBuffer::from_vec(
                "temperature_in",
                pack_f32(&[1.0], DType::F32),
                DType::F32,
            ))
            .buffer(BenchBuffer::from_vec("uniform_in", pack_f32(&[0.5], DType::F32), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes()) as u64)
    }
}
