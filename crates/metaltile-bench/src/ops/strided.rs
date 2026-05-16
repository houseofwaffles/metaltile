//! Strided copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="strided_copy",
    subop="strided_copy",
    class=StridedCopy,
    m=1024,
    n=4096,
    pad=128,
    tol=0.0,
    mlx="copy_g_nd2{tn}{tn}",
    metal_file="copy.metal",
)]
#[kernel]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}
