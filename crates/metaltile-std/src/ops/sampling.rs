//! GPU sampling kernels — softmax + categorical inverse-CDF walk used
//! by FFAI's `gpu-categorical` decode path (T > 0, no filters). The
//! greedy fast path uses `ffai_argmax` instead.
//!
//! Registered via `inventory::submit!` with empty `shapes`, so
//! `tile bench` and `tile test` skip it — the kernel ships for
//! codegen-only consumption today. Add a `ShapeSpec` entry + matching
//! runner if/when we want a perf baseline against MLX.

use metaltile::kernel;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// Softmax + categorical sample over a 1D logits tensor. Cooperative
// reduction (256 threads) for max + sum-exp; single-thread inverse
// CDF walk for the categorical pick.
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
// Cost: ~150µs at vocab=152K on M-class GPU. The cooperative max +
// sum-exp passes are fast; the single-thread CDF walk is the
// bottleneck. Parallel-prefix CDF walk is the next perf step.
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

    // ─── Pass 1: cooperative max reduce ─────────────────────────────
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

    if lid < 128u32 {
        let ov = threadgroup_load("tg_max", lid + 128u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 64u32 {
        let ov = threadgroup_load("tg_max", lid + 64u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 32u32 {
        let ov = threadgroup_load("tg_max", lid + 32u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 16u32 {
        let ov = threadgroup_load("tg_max", lid + 16u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 8u32 {
        let ov = threadgroup_load("tg_max", lid + 8u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 4u32 {
        let ov = threadgroup_load("tg_max", lid + 4u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 2u32 {
        let ov = threadgroup_load("tg_max", lid + 2u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid == 0u32 {
        let ov = threadgroup_load("tg_max", 1u32);
        let tv = threadgroup_load("tg_max", 0u32);
        threadgroup_store("tg_max", 0u32, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    let max_val = threadgroup_load("tg_max", 0u32);

    // ─── Pass 2: cooperative sum-exp reduce ─────────────────────────
    let mut local_sum = 0.0f32;
    threadgroup_alloc("tg_sum", 256);
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>() * inv_t;
            local_sum = local_sum + exp(v - max_val);
        }
    }
    threadgroup_store("tg_sum", lid, local_sum);
    threadgroup_barrier();

    if lid < 128u32 {
        let ov = threadgroup_load("tg_sum", lid + 128u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 64u32 {
        let ov = threadgroup_load("tg_sum", lid + 64u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 32u32 {
        let ov = threadgroup_load("tg_sum", lid + 32u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 16u32 {
        let ov = threadgroup_load("tg_sum", lid + 16u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 8u32 {
        let ov = threadgroup_load("tg_sum", lid + 8u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 4u32 {
        let ov = threadgroup_load("tg_sum", lid + 4u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 2u32 {
        let ov = threadgroup_load("tg_sum", lid + 2u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid == 0u32 {
        let ov = threadgroup_load("tg_sum", 1u32);
        let tv = threadgroup_load("tg_sum", 0u32);
        threadgroup_store("tg_sum", 0u32, ov + tv);
    }
    threadgroup_barrier();
    let total = threadgroup_load("tg_sum", 0u32);

    // ─── Pass 3: single-thread inverse CDF walk ─────────────────────
    if lid == 0u32 {
        let target = load(uniform_in[0]) * total;
        let mut cum = 0.0f32;
        let mut found_idx = n - 1u32;   // fallback to last index
        let mut done = 0u32;
        for i in range(0u32, n, 1u32) {
            let v = load(inp[i]).cast::<f32>() * inv_t;
            cum = cum + exp(v - max_val);
            let hit = (cum >= target) & (done == 0u32);
            found_idx = select(hit, i, found_idx);
            done = select(hit, 1u32, done);
        }
        store(out[0], found_idx);
    }
}

inventory::submit! {
    BenchSpec {
        op: "sampling",
        subop: "softmax_categorical_sample",
        kernel_name: "softmax_categorical_sample",
        kernel_ir: softmax_categorical_sample::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}
