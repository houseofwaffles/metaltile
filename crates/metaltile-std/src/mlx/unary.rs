//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="unary",
    subop="exp",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Exp{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="log",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sqrt",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Sqrt{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="rsqrt",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Rsqrt{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="abs",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Abs{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="silu",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="gelu",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="relu",
    class=Unary,
    input=Signed,
    tol=1e-6,
)]
#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="cos",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Cos{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sin",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Sin{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="ceil",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Ceil{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="floor",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Floor{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="erf",
    class=Unary,
    input=Signed,
    tol=1e-3,
    mlx="v_Erf{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="exp2",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="log2",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log2{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sign",
    class=Unary,
    input=Signed,
    tol=0.0,
    mlx="v_Sign{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="round",
    class=Unary,
    input=Signed,
    tol=0.0,
    mlx="v_Round{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="neg",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Negative{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[bench_kernel(
    op="unary",
    subop="recip",
    class=Unary,
    input=Positive,
    tol=1e-4,
)]
#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[bench_kernel(
    op="unary",
    subop="square",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Square{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[bench_kernel(
    op="unary",
    subop="sigmoid",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Sigmoid{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], 1.0f32.cast::<T>() / (1.0f32.cast::<T>() + exp(-x)));
}

#[bench_kernel(
    op="unary",
    subop="log1p",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log1p{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}
