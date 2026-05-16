//! LogSumExp benchmark — #[kernel] DSL vs MLX metal/logsumexp.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="logsumexp",
    subop="logsumexp",
    class=RowNorm,
    b=1024,
    n=4096,
    tpg=256,
    reads=1,
    out_elements=1,
    tol=1e-4,
    mlx="looped_logsumexp_{tn}",
    metal_file="logsumexp.metal",
)]
#[kernel]
pub fn mt_logsumexp<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut lm = neg_infinity();
    let mut nz = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(inp[base]).cast::<f32>();
        let v1 = load(inp[base + 1u32]).cast::<f32>();
        let v2 = load(inp[base + 2u32]).cast::<f32>();
        let v3 = load(inp[base + 3u32]).cast::<f32>();
        let cm = max(max(v0, v1), max(v2, v3));
        let pm = lm;
        let nm = max(pm, cm);
        nz = nz * exp(pm - nm) + exp(v0 - nm) + exp(v1 - nm) + exp(v2 - nm) + exp(v3 - nm);
        lm = nm;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(inp[_i]).cast::<f32>();
        let pm = lm;
        let nm = max(pm, xi);
        nz = nz * exp(pm - nm) + exp(xi - nm);
        lm = nm;
    }
    let gm = reduce_max(lm);
    let rscl = nz * exp(lm - gm);
    let gs = reduce_sum(rscl);
    if tid == 0 {
        store(out[row], (gm + log(gs)).cast::<T>());
    }
}
