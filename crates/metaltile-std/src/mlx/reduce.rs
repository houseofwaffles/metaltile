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
