//! Layer normalization benchmark — #[kernel] DSL vs MLX metal/layer_norm.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="layer_norm",
    subop="layer_norm",
    class=RowNorm,
    b=1024,
    n=4096,
    tpg=1024,
    reads=2,
    pre_weight=1.0,
    pre_bias=0.0,
    post_eps=1e-5,
    tol=1e-4,
    mlx="layer_norm_looped{tn}",
    metal_file="layer_norm.metal",
)]
#[kernel]
pub fn mt_layer_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(x[base]).cast::<f32>();
        let v1 = load(x[base + 1u32]).cast::<f32>();
        let v2 = load(x[base + 2u32]).cast::<f32>();
        let v3 = load(x[base + 3u32]).cast::<f32>();
        s = s + v0 + v1 + v2 + v3;
        sq = sq + v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        s = s + xi;
        sq = sq + xi * xi;
    }
    let st = reduce_sum(s);
    let sqt = reduce_sum(sq);
    let mean = st / n;
    let var = sqt / n - mean * mean;
    let eps = load(eps_buf[0]);
    let is = rsqrt(var + eps);
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = (load(x[base]).cast::<f32>() - mean) * is * load(w[col]).cast::<f32>()
            + load(b[col]).cast::<f32>();
        let n1 =
            (load(x[base + 1u32]).cast::<f32>() - mean) * is * load(w[col + 1u32]).cast::<f32>()
                + load(b[col + 1u32]).cast::<f32>();
        let n2 =
            (load(x[base + 2u32]).cast::<f32>() - mean) * is * load(w[col + 2u32]).cast::<f32>()
                + load(b[col + 2u32]).cast::<f32>();
        let n3 =
            (load(x[base + 3u32]).cast::<f32>() - mean) * is * load(w[col + 3u32]).cast::<f32>()
                + load(b[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        let ci = _i - rs;
        let norm = (xi - mean) * is * load(w[ci]).cast::<f32>() + load(b[ci]).cast::<f32>();
        store(out[_i], norm.cast::<T>());
    }
}
