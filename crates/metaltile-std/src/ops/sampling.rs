//! GPU sampling kernels — softmax + categorical inverse-CDF walk used
//! by FFAI's `gpu-categorical` decode path (T > 0, no filters). The
//! greedy fast path uses `mt_argmax` instead.
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
// reduction (256 threads) for max + sum-exp; single-threaded inverse
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
// Cost: ~150µs at vocab=152K on M-class GPU (1% overhead per token at
// 60 tok/s decode). The cooperative max + sum-exp passes are fast; the
// single-thread CDF walk is the bottleneck. Parallel-prefix CDF walk
// is the next perf step.
#[kernel]
pub fn softmax_categorical_sample<T>(
    inp: Tensor<T>,
    out: Tensor<u32>,
    temperature_in: Tensor<f32>,
    uniform_in: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let lid = tid;
    let temperature = load(temperature_in[0u32]);
    let uniform = load(uniform_in[0u32]);

    // ─── Pass 1: cooperative max across threads ──────────────────────
    let mut my_max = neg_infinity();
    let mut i = lid;
    while i < n {
        let v = load(inp[i]).cast::<f32>();
        if v > my_max {
            my_max = v;
        }
        i = i + lsize;
    }

    threadgroup_alloc("tg_max", 256);
    threadgroup_store("tg_max", lid, my_max);
    threadgroup_barrier();

    if lid < 128u32 {
        let o = threadgroup_load("tg_max", lid + 128u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 64u32 {
        let o = threadgroup_load("tg_max", lid + 64u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 32u32 {
        let o = threadgroup_load("tg_max", lid + 32u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 16u32 {
        let o = threadgroup_load("tg_max", lid + 16u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 8u32 {
        let o = threadgroup_load("tg_max", lid + 8u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 4u32 {
        let o = threadgroup_load("tg_max", lid + 4u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid < 2u32 {
        let o = threadgroup_load("tg_max", lid + 2u32);
        let t = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(o > t, o, t));
    }
    threadgroup_barrier();
    if lid == 0u32 {
        let o = threadgroup_load("tg_max", 1u32);
        let t = threadgroup_load("tg_max", 0u32);
        threadgroup_store("tg_max", 0u32, select(o > t, o, t));
    }
    threadgroup_barrier();
    let global_max = threadgroup_load("tg_max", 0u32);

    // ─── Pass 2: cooperative sum of exp((x - max) / T) ───────────────
    let mut my_sum = 0.0f32;
    let mut j = lid;
    while j < n {
        let v = load(inp[j]).cast::<f32>();
        my_sum = my_sum + exp((v - global_max) / temperature);
        j = j + lsize;
    }
    threadgroup_alloc("tg_sum", 256);
    threadgroup_store("tg_sum", lid, my_sum);
    threadgroup_barrier();

    if lid < 128u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 128u32),
        );
    }
    threadgroup_barrier();
    if lid < 64u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 64u32),
        );
    }
    threadgroup_barrier();
    if lid < 32u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 32u32),
        );
    }
    threadgroup_barrier();
    if lid < 16u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 16u32),
        );
    }
    threadgroup_barrier();
    if lid < 8u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 8u32),
        );
    }
    threadgroup_barrier();
    if lid < 4u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 4u32),
        );
    }
    threadgroup_barrier();
    if lid < 2u32 {
        threadgroup_store(
            "tg_sum",
            lid,
            threadgroup_load("tg_sum", lid) + threadgroup_load("tg_sum", lid + 2u32),
        );
    }
    threadgroup_barrier();
    if lid == 0u32 {
        threadgroup_store(
            "tg_sum",
            0u32,
            threadgroup_load("tg_sum", 0u32) + threadgroup_load("tg_sum", 1u32),
        );
    }
    threadgroup_barrier();
    let sum_exp = threadgroup_load("tg_sum", 0u32);

    // ─── Pass 3: single-threaded inverse-CDF walk ────────────────────
    if lid == 0u32 {
        let target = uniform * sum_exp;
        let mut cum = 0.0f32;
        let mut chosen = n - 1u32;
        let mut found = false;
        for k in range(0u32, n, 1u32) {
            if !found {
                let v = load(inp[k]).cast::<f32>();
                cum = cum + exp((v - global_max) / temperature);
                if cum >= target {
                    chosen = k;
                    found = true;
                }
            }
        }
        store(out[0u32], chosen);
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
