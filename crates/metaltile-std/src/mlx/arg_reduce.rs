//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! ArgReduce benchmarks — #[kernel] DSL vs MLX metal/arg_reduce.metal
//!
//! Two reductions over a flat input: `argmax` and `argmin`. Both
//! emit the winning index as a `u32` — MLX's `arg_reduce_general`
//! writes `out[out_idx] = best.index` (a `uint32_t`) regardless of
//! the input dtype, so a `u32` index buffer round-trips every index
//! exactly (a `T`-cast would lose large indices in f16 / bf16).
//!
//! Tie-breaking: strict comparison on values, smallest index on ties
//! (NumPy / PyTorch / MLX `arg_reduce` semantics).
//!
//! Both kernels are generic over `T` — f32 / f16 / bf16 all flow
//! through the same `#[kernel] fn`; values are widened to f32 for the
//! comparison so f16 / bf16 inputs reduce without precision loss.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [1, 1, 1]`, `tg = [256, 1, 1]`.
//! - TPG is fixed at 256 (8 simdgroups × 32 lanes) — the 7-stage
//!   halving tree below assumes exactly 256 threads.
//!
//! Correctness pinned by `tests/arg_reduce_gpu_correctness.rs`.

use metaltile::kernel;

// Tree-reduction strides: 128 → 64 → 32 → 16 → 8 → 4 → 2, then a final
// inline stride-1 merge. Each iteration merges the upper half into the
// lower half: take the winning value; on ties take the smaller index.
//
// Expressed as a DSL `for` loop over the seven stages rather than a
// hand-unrolled `macro_rules!` chain — the proc-macro does not expand
// inner declarative macros, so an unrolled inner macro would silently
// emit no IR (see docs/developing.md kernel-authoring hazards). The
// `for` loop yields identical MSL and survives the proc-macro intact.

#[kernel]
pub fn mt_argmax<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
    let lid = tid;
    let mut best_val = neg_infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>();
            let better = v > best_val;
            if better {
                best_val = v;
                best_idx = pos;
            }
        }
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();
    // 7-stage power-of-two halving reduction over the 256-thread group.
    for _stage in range(0u32, 7u32, 1u32) {
        let stride = 128u32 >> _stage;
        if lid < stride {
            let ov = threadgroup_load("tg_vals", lid + stride);
            let oi = threadgroup_load("tg_idxs", lid + stride);
            let tv = threadgroup_load("tg_vals", lid);
            let ti = threadgroup_load("tg_idxs", lid);
            // argmax: take higher value; on ties take smaller index.
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }
    // Final stride-1 merge writes the result directly to output.
    if lid == 0u32 {
        let ov = threadgroup_load("tg_vals", 1u32);
        let oi = threadgroup_load("tg_idxs", 1u32);
        let tv = threadgroup_load("tg_vals", 0u32);
        let ti = threadgroup_load("tg_idxs", 0u32);
        let bet = (ov > tv) | ((ov == tv) & (oi < ti));
        let final_idx = select(bet, oi, ti);
        store(out[0], final_idx);
    }
}

#[kernel]
pub fn mt_argmin<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
    let lid = tid;
    // argmin seeds with +infinity so any finite value wins.
    let mut best_val = infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>();
            let better = v < best_val;
            if better {
                best_val = v;
                best_idx = pos;
            }
        }
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();
    // 7-stage power-of-two halving reduction over the 256-thread group.
    for _stage in range(0u32, 7u32, 1u32) {
        let stride = 128u32 >> _stage;
        if lid < stride {
            let ov = threadgroup_load("tg_vals", lid + stride);
            let oi = threadgroup_load("tg_idxs", lid + stride);
            let tv = threadgroup_load("tg_vals", lid);
            let ti = threadgroup_load("tg_idxs", lid);
            // argmin: take lower value; on ties take smaller index.
            let bet = (ov < tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }
    // Final stride-1 merge writes the result directly to output.
    if lid == 0u32 {
        let ov = threadgroup_load("tg_vals", 1u32);
        let oi = threadgroup_load("tg_idxs", 1u32);
        let tv = threadgroup_load("tg_vals", 0u32);
        let ti = threadgroup_load("tg_idxs", 0u32);
        let bet = (ov < tv) | ((ov == tv) & (oi < ti));
        let final_idx = select(bet, oi, ti);
        store(out[0], final_idx);
    }
}

/// New-syntax correctness for the mlx arg-reduce kernels (Reduction mode, one
/// 256-lane threadgroup, u32 index output). A lone spike/dip in a flat field
/// gives an unambiguous winner in every dtype.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_argmax, mt_argmin};
    use crate::utils::pack_f32;

    fn setup(
        kernel: metaltile::core::ir::Kernel,
        n: usize,
        idx: usize,
        spike: f32,
        dt: DType,
    ) -> TestSetup {
        let mut inp = vec![0.0f32; n];
        inp[idx] = spike; // +2 for argmax, -2 for argmin
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", 1, DType::U32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&[idx as f32], DType::U32), DType::U32))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.5)]
    fn test_mt_argmax(dt: DType) -> TestSetup {
        setup(mt_argmax::kernel_ir_for(dt), 1000, 813, 2.0, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.5)]
    fn test_mt_argmin(dt: DType) -> TestSetup {
        setup(mt_argmin::kernel_ir_for(dt), 1000, 271, -2.0, dt)
    }
}

/// New-syntax benchmarks for the mlx arg-reduce kernels (vs MLX
/// `metal/arg_reduce.metal`). Vocab-sized, read-dominated.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_argmax, mt_argmin};

    fn ab(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let n = 256 * 1024usize;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", 1, DType::U32).output())
            .constexpr("n", n as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/arg_reduce/argmax", dtypes = [f32, f16, bf16])]
    fn bench_argmax(dt: DType) -> BenchSetup { ab(mt_argmax::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/arg_reduce/argmin", dtypes = [f32, f16, bf16])]
    fn bench_argmin(dt: DType) -> BenchSetup { ab(mt_argmin::kernel_ir_for(dt), dt) }
}
