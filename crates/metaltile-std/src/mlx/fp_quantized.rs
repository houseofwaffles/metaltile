//! FP quantized benchmark — #[kernel] DSL vs MLX metal/fp_quantized.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="fp_quantized",
    subop="fp4_quant_dequant",
    class=FpQuantized,
    n=1048576,
    tpg=32,
    tol=0.5,
    mlx="nvfp4_quantize_dequantize_float_gs_16_b_4",
    metal_file="fp_quantized.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_fp4_quant_dequant(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let x = load(inp[gid]);
    let ax = abs(x);
    let group_max = simd_max(ax);
    let inv_scale = select(group_max > 0.0f32, 6.0f32 / group_max, 0.0f32);
    let norm = ax * inv_scale;
    let q = select(
        norm < 0.25f32,
        0.0f32,
        select(
            norm < 0.75f32,
            0.5f32,
            select(
                norm < 1.25f32,
                1.0f32,
                select(
                    norm < 1.75f32,
                    1.5f32,
                    select(
                        norm < 2.5f32,
                        2.0f32,
                        select(norm < 3.5f32, 3.0f32, select(norm < 5.0f32, 4.0f32, 6.0f32)),
                    ),
                ),
            ),
        ),
    );
    let sign = select(x < 0.0f32, -1.0f32, 1.0f32);
    let result = sign * q * (group_max / 6.0f32);
    store(out[gid], result);
}
