//! binary_two benchmark — #[kernel] DSL fused two-output elementwise

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="binary_two",
    subop="add_mul",
    class=BinaryTwo,
    input_a=Signed,
    input_b=Half,
    tol=1e-3,
)]
#[kernel]
pub fn mt_binary_two<T>(a: Tensor<T>, b: Tensor<T>, mut c: Tensor<T>, mut d: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    let y = load(b[idx]);
    store(c[idx], x + y);
    store(d[idx], x * y);
}
