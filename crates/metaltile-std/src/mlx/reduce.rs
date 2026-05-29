//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Reduce benchmarks — #[kernel] DSL vs MLX metal/reduce.metal
//!
//! Covers four reduction shapes:
//!   - **all-reduce** — `mt_all_reduce*`: one threadgroup folds the
//!     whole input to a scalar (Reduction mode).
//!   - **row-reduce** — `mt_row_reduce*`: one threadgroup per row of a
//!     `[rows, n]` input (Reduction mode).
//!   - **column-reduce** — `mt_col_reduce*`: one thread per column of a
//!     `[rows, cols]` input; each thread walks its column with a
//!     `cols`-strided `strided_reduce` (Grid3D, no threadgroup
//!     cooperation). Mirrors MLX's `col_reduce_*` family.
//!   - **segmented-reduce** — `mt_seg_reduce*`: one thread per segment
//!     of a flat input split into `n_segments` fixed-length contiguous
//!     runs; each thread contiguously folds its `seg_len`-element run
//!     (Grid3D). Suits many short segments where the row-reduce
//!     threadgroup-per-row layout would under-occupy the GPU.

use metaltile::kernel;

#[kernel(
    bench(
        op="all_reduce",
        subop="sum",
        class=AllReduce,
        // tol=256.0 — summing 64M signed bf16 values, MT and MLX accumulate
        // in slightly different orders. With bf16 precision (~7-bit
            // mantissa, ~1% relative) the result drifts by up to ~192 absolute
        // between the two reduction trees. f32 stays comfortably below 1e-3.
        tol=256.0,
        mlx="all_reduce_sum{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_all_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, sum);
    let result = reduce_sum(acc);
    store(out[0], result);
}

#[kernel(
    bench(
        op="all_reduce",
        subop="prod",
        class=AllReduce,
        // tol=1024.0 — product grows exponentially; 64M bf16 values compound
        // ~1% relative error per multiply, leading to large absolute divergence.
        tol=1024.0,
        mlx="all_reduce_prod{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_all_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let off = 0;
    let acc = strided_reduce(inp, off, n, product);
    let result = reduce_product(acc);
    store(out[0], result);
}

#[kernel(
    bench(
        op="all_reduce",
        subop="max",
        class=AllReduce,
        tol=0.0,
        mlx="all_reduce_max{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_all_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, max);
    let result = reduce_max(acc);
    store(out[0], result);
}

#[kernel(
    bench(
        op="all_reduce",
        subop="min",
        class=AllReduce,
        tol=0.0,
        mlx="all_reduce_min{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_all_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, min);
    let result = reduce_min(acc);
    store(out[0], result);
}

#[kernel(
    bench(
        op="row_reduce",
        subop="sum",
        class=RowReduce,
        tol=128.0,
        mlx="row_reduce_simple_sum{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_row_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, sum);
    let result = reduce_sum(acc);
    store(out[row], result);
}

#[kernel(
    bench(
        op="row_reduce",
        subop="prod",
        class=RowReduce,
        tol=32.0,
        mlx="row_reduce_simple_prod{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_row_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, product);
    let result = reduce_product(acc);
    store(out[row], result);
}

#[kernel(
    bench(
        op="row_reduce",
        subop="max",
        class=RowReduce,
        tol=0.0,
        mlx="row_reduce_simple_max{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_row_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, max);
    let result = reduce_max(acc);
    store(out[row], result);
}

#[kernel(
    bench(
        op="row_reduce",
        subop="min",
        class=RowReduce,
        tol=0.0,
        mlx="row_reduce_simple_min{tn}",
        metal_file="reduce.metal",
    )
)]
pub fn mt_row_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, min);
    let result = reduce_min(acc);
    store(out[row], result);
}

// ── Column reduce ────────────────────────────────────────────────────────
//
// `inp` is a row-major `[rows, cols]` matrix; `out` is `[cols]` with
// `out[c] = reduce over r of inp[r * cols + c]`. One thread per output
// column (Grid3D). Each thread folds its column with a `cols`-strided
// `strided_reduce`: offset = c, stride = cols, end = rows * cols.
//
// Grid3D mode emits the `for (_i = off; _i < end; _i += stride)` form
// (see codegen `emit_block.rs` — the `stride` field is honoured only
// outside Reduction mode), so the strided walk is correct here.
//
// Unlike the Reduction-mode `mt_row_reduce`, NO `reduce_*(acc)`
// finishing step is applied: in Grid3D the `strided_reduce` loop is
// run by a single thread and already folds the whole column. A
// `reduce_sum` here would lower to `simd_sum` and wrongly sum 32
// independent columns together.
//
// The four ops share one body; the outer `macro_rules!` wraps the
// whole `#[kernel]` declaration so the proc-macro sees concrete tokens
// (an inner macro inside the body would silently emit no IR — see
// docs/developing.md kernel-authoring hazards).

#[rustfmt::skip]
macro_rules! col_reduce_kernel {
    ($name:ident, $reduce_op:ident, $subop:literal) => {
        #[kernel(
            bench(
                op="col_reduce",
                subop=$subop,
                class=GenericEmpty,
                tol=128.0,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            inp: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] rows: u32,
            #[constexpr] cols: u32,
        ) {
            let col = program_id::<0>();
            if col < cols {
                let end = rows * cols;
                let acc = strided_reduce(inp, col, cols, end, $reduce_op);
                store(out[col], acc.cast::<T>());
            }
        }
    };
}

col_reduce_kernel!(mt_col_reduce, sum, "sum");
col_reduce_kernel!(mt_col_reduce_prod, product, "prod");
col_reduce_kernel!(mt_col_reduce_max, max, "max");
col_reduce_kernel!(mt_col_reduce_min, min, "min");

// ── Segmented reduce ─────────────────────────────────────────────────────
//
// `inp` is a flat buffer split into `n_segments` contiguous runs of
// `seg_len` elements; `out` is `[n_segments]` with
// `out[s] = reduce(inp[s * seg_len .. (s + 1) * seg_len])`. One thread
// per segment (Grid3D), each folding its run contiguously
// (stride = 1).
//
// This is the one-thread-per-segment counterpart to `mt_row_reduce`'s
// one-threadgroup-per-row layout: for many short segments the
// threadgroup-per-row form under-occupies the GPU (most lanes idle),
// whereas one thread per segment keeps every lane busy.

#[rustfmt::skip]
macro_rules! seg_reduce_kernel {
    ($name:ident, $reduce_op:ident, $subop:literal) => {
        #[kernel(
            bench(
                op="seg_reduce",
                subop=$subop,
                class=GenericEmpty,
                tol=128.0,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            inp: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] n_segments: u32,
            #[constexpr] seg_len: u32,
        ) {
            let seg = program_id::<0>();
            if seg < n_segments {
                let start = seg * seg_len;
                let end = start + seg_len;
                // Grid3D: one thread folds the whole segment — no
                // `reduce_*` finishing step (see col-reduce note above).
                let acc = strided_reduce(inp, start, 1u32, end, $reduce_op);
                store(out[seg], acc.cast::<T>());
            }
        }
    };
}

seg_reduce_kernel!(mt_seg_reduce, sum, "sum");
seg_reduce_kernel!(mt_seg_reduce_prod, product, "prod");
seg_reduce_kernel!(mt_seg_reduce_max, max, "max");
seg_reduce_kernel!(mt_seg_reduce_min, min, "min");

/// New-syntax correctness for the reduce family.
///
/// all/row reduce are Reduction-mode (`.mode(Reduction)`, one threadgroup per
/// row); col/seg reduce are Grid3D (one thread per output). Oracles fold the
/// dtype-rounded inputs in f32. max/min are exact; sum/prod widen per dtype to
/// cover f16/bf16 accumulation order vs the f32 oracle. Inputs are kept small
/// (prod stays near 1) so the accumulation drift is bounded.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn fold(init: f32, xs: impl Iterator<Item = f32>, op: fn(f32, f32) -> f32) -> f32 {
        xs.fold(init, op)
    }

    // ── all-reduce: one threadgroup folds `n` elements → out[0] ───────────
    fn all_setup_for(
        kernel: Kernel,
        n: usize,
        vals: &[f32],
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let v = unpack_f32(&pack_f32(vals, dt), dt);
        let expected = fold(init, v.into_iter(), op);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(vals, dt), dt))
            .input(TestBuffer::zeros("out", 1, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&[expected], dt), dt))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    fn sum_vals(n: usize) -> Vec<f32> { (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect() }
    fn prod_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.001).collect()
    }
    fn ext_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 2.0, 16.0])]
    fn test_all_reduce_sum(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce::kernel_ir_for(dt),
            2048,
            &sum_vals(2048),
            0.0,
            |a, b| a + b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 1.0])]
    fn test_all_reduce_prod(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_prod::kernel_ir_for(dt),
            512,
            &prod_vals(512),
            1.0,
            |a, b| a * b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_all_reduce_max(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_max::kernel_ir_for(dt),
            2048,
            &ext_vals(2048),
            f32::NEG_INFINITY,
            f32::max,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_all_reduce_min(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_min::kernel_ir_for(dt),
            2048,
            &ext_vals(2048),
            f32::INFINITY,
            f32::min,
            dt,
        )
    }

    // ── row-reduce: one threadgroup per row of [rows, n] → out[row] ────────
    fn row_setup_for(
        kernel: Kernel,
        rows: usize,
        n: usize,
        per_row: &dyn Fn(usize) -> Vec<f32>,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let mut inp = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = per_row(r);
            let rd = unpack_f32(&pack_f32(&row, dt), dt);
            expected.push(fold(init, rd.into_iter(), op));
            inp.extend_from_slice(&row);
        }
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", rows, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1.0, 8.0])]
    fn test_row_reduce_sum(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i % 17) as f32 - 8.0) * 0.01 + r as f32 * 0.001).collect(),
            0.0,
            |a, b| a + b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_row_reduce_max(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce_max::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0 + r as f32).collect(),
            f32::NEG_INFINITY,
            f32::max,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_row_reduce_min(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce_min::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0 + r as f32).collect(),
            f32::INFINITY,
            f32::min,
            dt,
        )
    }

    // ── col-reduce: Grid3D, one thread per column of [rows, cols] ─────────
    fn col_setup_for(
        kernel: Kernel,
        rows: usize,
        cols: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let inp: Vec<f32> = (0..rows * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> =
            (0..cols).map(|c| fold(init, (0..rows).map(|r| id[r * cols + c]), op)).collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", cols, dt))
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(cols, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_col_reduce_sum(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce::kernel_ir_for(dt), 37, 100, 0.0, |a, b| a + b, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_col_reduce_max(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce_max::kernel_ir_for(dt), 50, 70, f32::NEG_INFINITY, f32::max, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_col_reduce_min(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce_min::kernel_ir_for(dt), 50, 70, f32::INFINITY, f32::min, dt)
    }

    // ── seg-reduce: Grid3D, one thread per contiguous segment ─────────────
    fn seg_setup_for(
        kernel: Kernel,
        n_segments: usize,
        seg_len: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let inp: Vec<f32> =
            (0..n_segments * seg_len).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> = (0..n_segments)
            .map(|s| fold(init, (0..seg_len).map(|j| id[s * seg_len + j]), op))
            .collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n_segments, dt))
            .constexpr("n_segments", n_segments as u32)
            .constexpr("seg_len", seg_len as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_segments, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_seg_reduce_sum(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce::kernel_ir_for(dt), 64, 48, 0.0, |a, b| a + b, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_seg_reduce_max(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce_max::kernel_ir_for(dt), 64, 48, f32::NEG_INFINITY, f32::max, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_seg_reduce_min(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce_min::kernel_ir_for(dt), 64, 48, f32::INFINITY, f32::min, dt)
    }
}

/// New-syntax benchmarks for the reduce family (vs MLX `metal/reduce.metal`).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    // all-reduce: one threadgroup folds N elements to a scalar.
    fn all_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let n = 16 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", 1, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/all_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_all_sum(dt: DType) -> BenchSetup { all_b(mt_all_reduce::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/all_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_all_prod(dt: DType) -> BenchSetup { all_b(mt_all_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/all_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_all_max(dt: DType) -> BenchSetup { all_b(mt_all_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/all_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_all_min(dt: DType) -> BenchSetup { all_b(mt_all_reduce_min::kernel_ir_for(dt), dt) }

    // row-reduce: one threadgroup per row.
    fn row_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", rows * n, dt))
            .buffer(BenchBuffer::zeros("out", rows, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((rows * n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/row_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_row_sum(dt: DType) -> BenchSetup { row_b(mt_row_reduce::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/row_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_row_prod(dt: DType) -> BenchSetup { row_b(mt_row_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/row_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_row_max(dt: DType) -> BenchSetup { row_b(mt_row_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/row_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_row_min(dt: DType) -> BenchSetup { row_b(mt_row_reduce_min::kernel_ir_for(dt), dt) }

    // col-reduce: Grid3D, one thread per output column of a [rows, cols] matrix.
    fn col_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let (rows, cols) = (4096usize, 4096usize);
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", rows * cols, dt))
            .buffer(BenchBuffer::zeros("out", cols, dt).output())
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .grid_1d(cols, 256)
            .bytes_moved((rows * cols * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/col_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_col_sum(dt: DType) -> BenchSetup { col_b(mt_col_reduce::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/col_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_col_prod(dt: DType) -> BenchSetup { col_b(mt_col_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/col_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_col_max(dt: DType) -> BenchSetup { col_b(mt_col_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/col_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_col_min(dt: DType) -> BenchSetup { col_b(mt_col_reduce_min::kernel_ir_for(dt), dt) }

    // seg-reduce: Grid3D, one thread per contiguous segment.
    fn seg_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let (n_segments, seg_len) = (65536usize, 256usize);
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n_segments * seg_len, dt))
            .buffer(BenchBuffer::zeros("out", n_segments, dt).output())
            .constexpr("n_segments", n_segments as u32)
            .constexpr("seg_len", seg_len as u32)
            .grid_1d(n_segments, 256)
            .bytes_moved((n_segments * seg_len * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/seg_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_seg_sum(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/seg_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_seg_prod(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/seg_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_seg_max(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/seg_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_seg_min(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_min::kernel_ir_for(dt), dt) }
}
