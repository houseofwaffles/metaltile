//! Quantized MatVec benchmark — #[kernel] DSL vs MLX metal/quantized.metal

use metaltile::{bench_kernel, kernel};
static QUANTIZED_SHAPES: &[(usize, usize)] = &[(4096, 4096)];

#[bench_kernel(
    op="quantized",
    subop="qmv",
    class=QuantizedMatVec,
    shapes=&QUANTIZED_SHAPES,
    group_size=64,
    tpg=64,
    tol=1e-3,
    mlx="affine_qmv_fast_float16_t_gs_64_b_4_batch_0",
    metal_file="quantized.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_qmv_f32(
    w: Tensor<u32>,
    scales: Tensor<f32>,
    biases: Tensor<f32>,
    x: Tensor<f32>,
    out: Tensor<f32>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    let row = program_id::<0>();
    let packs_per_row = k / 8u32;
    let w_base = row * packs_per_row;
    let sb_base = row * gs_per_row;
    let mut acc = 0.0f32;
    for _g in range(tid, gs_per_row, lsize) {
        let s = load(scales[sb_base + _g]);
        let bias = load(biases[sb_base + _g]);
        let g_w_base = w_base + _g * 8u32;
        let g_x_base = _g * 64u32;
        for _p in range(0u32, 8u32, 1u32) {
            let packed = load(w[g_w_base + _p]);
            let xb = g_x_base + _p * 8u32;
            for _b in range(0u32, 8u32, 1u32) {
                let shift = _b * 4u32;
                let int4_val = (packed >> shift) & 15u32;
                let xi = load(x[xb + _b]);
                acc = acc + (s * (int4_val * 1.0f32) + bias) * xi;
            }
        }
    }
    let result = reduce_sum(acc);
    store(out[row], result);
}
