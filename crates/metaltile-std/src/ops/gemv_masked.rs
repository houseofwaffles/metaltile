//! Masked GEMV benchmark — #[kernel] DSL (no MLX reference)

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gemv_masked",
    subop="gemv_masked",
    class=MatVecMasked,
    b=4096,
    n=4096,
    tpg=256,
    tol=1e-2,
)]
#[kernel]
pub fn mt_gemv_masked<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    mask: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let mut acc = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let col = _i - rs;
        let m_val = load(mask[col]).cast::<f32>();
        acc = acc + load(mat[_i]).cast::<f32>() * load(vec[col]).cast::<f32>() * m_val;
    }
    let result = reduce_sum(acc);
    store(out[row], result.cast::<T>());
}
