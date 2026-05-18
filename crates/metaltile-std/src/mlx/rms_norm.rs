//! RMS normalization benchmark — #[kernel] DSL vs MLX metal/rms_norm.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rms_norm",
    subop="rms_norm",
    class=RowNorm,
    b=1024,
    n=4096,
    tpg=1024,
    reads=2,
    pre_weight=1.0,
    post_eps=1e-5,
    tol=1e-4,
    mlx="rms{tn}",
    metal_file="rms_norm.metal",
)]
#[kernel]
pub fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // Each thread owns exactly 4 consecutive elements (N = TPG * 4).
    // Read x once, cache in registers, reuse for both ssq and output — 3 reads total.
    let base = rs + tid * 4u32;
    let col = tid * 4u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let x2 = load(x[base + 2u32]).cast::<f32>();
    let x3 = load(x[base + 3u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
    store(out[base + 2u32], (x2 * rms * load(w[col + 2u32]).cast::<f32>()).cast::<T>());
    store(out[base + 3u32], (x3 * rms * load(w[col + 3u32]).cast::<f32>()).cast::<T>());
}
