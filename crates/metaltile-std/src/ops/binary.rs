//! Elementwise binary ops — #[kernel] DSL vs MLX metal/binary.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="binary",
    subop="add",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Add{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn vector_add<T>(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>) {
    let idx = program_id(0);
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[bench_kernel(
    op="binary",
    subop="mul",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Multiply{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) * load(b[idx]));
}

#[bench_kernel(
    op="binary",
    subop="sub",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Subtract{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_sub<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) - load(b[idx]));
}

#[bench_kernel(
    op="binary",
    subop="div",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Divide{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_div<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) / load(b[idx]));
}

#[bench_kernel(
    op="binary",
    subop="maximum",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Maximum{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_max_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], max(load(a[idx]), load(b[idx])));
}

#[bench_kernel(
    op="binary",
    subop="minimum",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-6,
    mlx="vvn_Minimum{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_min_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], min(load(a[idx]), load(b[idx])));
}

#[bench_kernel(
    op="binary",
    subop="pow",
    class=Binary,
    input_a=Unit,
    input_b=Half,
    tol=1e-4,
    mlx="vvn_Power{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_pow<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], pow(load(a[idx]), load(b[idx])));
}

#[bench_kernel(
    op="binary",
    subop="logaddexp",
    class=Binary,
    input_a=Signed,
    input_b=Signed,
    tol=1e-4,
    mlx="vvn_LogAddExp{tn}",
    metal_file="binary.metal",
)]
#[kernel]
pub fn mt_logaddexp<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(exp(load(a[idx])) + exp(load(b[idx]))));
}
