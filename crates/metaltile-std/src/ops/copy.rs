//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="copy",
    subop="copy",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_copy{tn}{tn}",
    metal_file="copy.metal",
)]
#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}
