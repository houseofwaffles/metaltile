//! RoPE benchmark — #[kernel] DSL vs MLX metal/rope.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rope",
    subop="rope",
    class=Rope,
    b=1,
    h=32,
    l=512,
    d=128,
    n_per_group=4,
    tol=0.01,
    metal_file="rope.metal",
    dtypes=crate::spec::F16_ONLY,
)]
#[kernel]
pub fn mt_rope_f16(
    inp: Tensor<f16>,
    out: Tensor<f16>,
    #[constexpr] h_stride: u32,
    #[constexpr] seq_stride: u32,
    #[constexpr] grid_x: u32,
    #[constexpr] base: f32,
) {
    let px = program_id::<0>();
    let py = program_id::<1>();
    let pz = program_id::<2>();
    let px_f = px.cast::<f32>();
    let gx_f = grid_x.cast::<f32>();
    let d_norm = px_f / gx_f;
    let inv_freq = exp2(-(d_norm * base));
    let theta = py.cast::<f32>() * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let head_base = pz * 4;
    for i in range(0, 4, 1) {
        let head = head_base + i;
        let idx1 = py * seq_stride + head * h_stride + px;
        let idx2 = idx1 + grid_x;
        let x1 = load(inp[idx1]).cast::<f32>();
        let x2 = load(inp[idx2]).cast::<f32>();
        let rx1 = x1 * cos_t - x2 * sin_t;
        let rx2 = x1 * sin_t + x2 * cos_t;
        store(out[idx1], rx1.cast::<f16>());
        store(out[idx2], rx2.cast::<f16>());
    }
}
