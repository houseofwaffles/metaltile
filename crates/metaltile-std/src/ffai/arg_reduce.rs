//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Generic `argmax<T>` with u32 index output — FFAI's decode-form
//! greedy-sampler workhorse.
//!
//! Adapted from `mt_argmax_f32` (in `mlx/arg_reduce.rs`) but generic
//! over input dtype and emitting a `u32` index rather than a float-cast
//! version. Decode-form samplers (greedy token pick) need an integer
//! token id; the f32-output upstream variant doesn't fit that contract.
//!
//! Tie-breaking: strict `>` on values, smallest index on ties — matches
//! NumPy / PyTorch / MLX `argmax` semantics.
//!
//! Codegen-only — there's no MLX argmax template with the same
//! u32-output signature. Correctness validated in FFAI integration
//! tests against reference decoder output.

use metaltile::kernel;

// Tree-reduction strides: 128 → 64 → 32 → 16 → 8 → 4 → 2.
// Each iteration: threads with `lid < stride` merge the upper half into
// the lower half (take higher value; on ties take smaller index — NumPy
// argmax semantics).  Final stride-1 merge writes the result directly
// to `out[0]` and is kept inline below.
//
// Originally hand-unrolled via a `macro_rules! argmax_step!` invoked
// 7×; the proc-macro does not expand inner declarative macros, so the
// expansion silently produced no IR.  A DSL `for` loop over the seven
// stages yields identical MSL and survives the proc-macro intact.

#[kernel]
pub fn ffai_argmax<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
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
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }
    // Final stride-1 merge writes result directly to output.
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

/// New-syntax correctness for `ffai_argmax`.
///
/// This is a **reduction-mode, MLX-less (ffai)** kernel — exactly the case the
/// legacy harness left unbenched via `class=GenericEmpty`. The new builders
/// express it with `.mode(KernelMode::Reduction)` (so codegen emits the
/// threadgroup reduction) and a one-threadgroup grid; the u32 index output is
/// compared exactly through the integer-aware `pack_f32`/`unpack_f32` path.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_argmax;
    use crate::utils::pack_f32;

    /// One 256-lane threadgroup reduces all `n` elements (the kernel loops over
    /// `n` in chunks of `lsize`). `max_idx` carries a lone `2.0` spike in a
    /// field of `1.0` — an unambiguous argmax in every dtype (no rounding ties).
    fn setup(n: usize, max_idx: usize, dt: DType) -> TestSetup {
        let mut inp = vec![1.0f32; n];
        inp[max_idx] = 2.0;
        TestSetup::new(ffai_argmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", 1, DType::U32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec(
                "out",
                pack_f32(&[max_idx as f32], DType::U32),
                DType::U32,
            ))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    // Integer index output — tol < 1 means the index must match exactly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.5)]
    fn test_ffai_argmax_interior(dt: DType) -> TestSetup { setup(256, 37, dt) }

    // n > lsize exercises the chunked outer loop (4 chunks of 256).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.5)]
    fn test_ffai_argmax_multi_chunk(dt: DType) -> TestSetup { setup(1000, 813, dt) }
}

/// New-syntax benchmark for `ffai_argmax` — an MLX-less reduction kernel that
/// previously produced no bench rows (`class=GenericEmpty`). It now reports
/// real GB/s in `tile bench` with `Ref(GB/s)` blank (no MLX counterpart).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_argmax;

    // Vocab-sized argmax (greedy decode): one threadgroup reduces 256K elements.
    // Read-dominated, so bytes_moved counts the input.
    #[bench(name = "ffai/argmax", dtypes = [f32, f16, bf16])]
    fn bench_argmax(dt: DType) -> BenchSetup {
        let n = 256 * 1024usize;
        BenchSetup::new(ffai_argmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", 1, DType::U32).output())
            .constexpr("n", n as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes()) as u64)
    }
}
