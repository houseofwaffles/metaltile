//! Reduce benchmarks — #[kernel] DSL vs MLX metal/reduce.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
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
)]
#[kernel]
pub fn mt_all_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, sum);
    let result = reduce_sum(acc);
    store(out[0], result);
}

#[bench_kernel(
    op="all_reduce",
    subop="prod",
    class=AllReduce,
    // tol=1024.0 — product grows exponentially; 64M bf16 values compound
    // ~1% relative error per multiply, leading to large absolute divergence.
    tol=1024.0,
    mlx="all_reduce_prod{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_all_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let off = 0;
    let acc = strided_reduce(inp, off, n, product);
    let result = reduce_product(acc);
    store(out[0], result);
}

#[bench_kernel(
    op="all_reduce",
    subop="max",
    class=AllReduce,
    tol=0.0,
    mlx="all_reduce_max{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_all_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, max);
    let result = reduce_max(acc);
    store(out[0], result);
}

#[bench_kernel(
    op="all_reduce",
    subop="min",
    class=AllReduce,
    tol=0.0,
    mlx="all_reduce_min{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_all_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, min);
    let result = reduce_min(acc);
    store(out[0], result);
}

#[bench_kernel(
    op="row_reduce",
    subop="sum",
    class=RowReduce,
    tol=128.0,
    mlx="row_reduce_simple_sum{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_row_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, sum);
    let result = reduce_sum(acc);
    store(out[row], result);
}

#[bench_kernel(
    op="row_reduce",
    subop="prod",
    class=RowReduce,
    tol=32.0,
    mlx="row_reduce_simple_prod{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_row_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, product);
    let result = reduce_product(acc);
    store(out[row], result);
}

#[bench_kernel(
    op="row_reduce",
    subop="max",
    class=RowReduce,
    tol=0.0,
    mlx="row_reduce_simple_max{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_row_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, max);
    let result = reduce_max(acc);
    store(out[row], result);
}

#[bench_kernel(
    op="row_reduce",
    subop="min",
    class=RowReduce,
    tol=0.0,
    mlx="row_reduce_simple_min{tn}",
    metal_file="reduce.metal",
)]
#[kernel]
pub fn mt_row_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, min);
    let result = reduce_min(acc);
    store(out[row], result);
}
