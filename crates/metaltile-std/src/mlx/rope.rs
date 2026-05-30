//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! RoPE benchmark — #[kernel] DSL vs MLX metal/rope.metal

use metaltile::kernel;

#[kernel]
pub fn mt_rope<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
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
        store(out[idx1], rx1.cast::<T>());
        store(out[idx2], rx2.cast::<T>());
    }
}

/// New-syntax correctness for `mt_rope` (Grid3D, single threadgroup with
/// `tpg = [grid_x, seq_len, n_heads/4]`). Oracle = rotate-half RoPE on
/// dtype-rounded input; constexprs derived as in the legacy test.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_rope;
    use crate::utils::{pack_f32, unpack_f32};

    fn naive_rope(
        inp: &[f32],
        n_heads: u32,
        seq_len: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> Vec<f32> {
        let grid_x = head_dim / 2;
        let h_stride = seq_len * head_dim;
        let seq_stride = head_dim;
        let base = theta_base.log2();
        let mut out = inp.to_vec();
        for pz in 0..n_heads / 4 {
            for py in 0..seq_len {
                for px in 0..grid_x {
                    let inv_freq = (-(px as f32 / grid_x as f32 * base)).exp2();
                    let theta = py as f32 * inv_freq;
                    let (c, s) = (theta.cos(), theta.sin());
                    for i in 0..4 {
                        let head = pz * 4 + i;
                        let idx1 = (py * seq_stride + head * h_stride + px) as usize;
                        let idx2 = idx1 + grid_x as usize;
                        let (x1, x2) = (inp[idx1], inp[idx2]);
                        out[idx1] = x1 * c - x2 * s;
                        out[idx2] = x1 * s + x2 * c;
                    }
                }
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_mt_rope(dt: DType) -> TestSetup {
        let (n_heads, seq_len, head_dim, theta_base) = (4u32, 8u32, 16u32, 10000.0f32);
        let n = (n_heads * seq_len * head_dim) as usize;
        let inp: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();
        let inp_dt = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected = naive_rope(&inp_dt, n_heads, seq_len, head_dim, theta_base);
        let grid_x = head_dim / 2;
        TestSetup::new(mt_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("h_stride", seq_len * head_dim)
            .constexpr("seq_stride", head_dim)
            .constexpr("grid_x", grid_x)
            .constexpr("base", theta_base.log2())
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(1, 1, 1, [grid_x, seq_len, n_heads / 4])
    }
}

/// New-syntax benchmark for `mt_rope` (vs MLX `metal/rope.metal`). Multi-TG:
/// `tpg = [grid_x, 8, 1]`, grid splits seq_len and the head-quads.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_rope;

    #[bench(name = "mlx/rope", dtypes = [f32, f16, bf16])]
    fn bench_rope(dt: DType) -> BenchSetup {
        let (n_heads, seq_len, head_dim, theta_base) = (32u32, 512u32, 128u32, 10000.0f32);
        let grid_x = head_dim / 2; // 64
        let n = (n_heads * seq_len * head_dim) as usize;
        BenchSetup::new(mt_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("h_stride", seq_len * head_dim)
            .constexpr("seq_stride", head_dim)
            .constexpr("grid_x", grid_x)
            .constexpr("base", theta_base.log2())
            // tpg [64, 8, 1] (=512 lanes); grid covers [grid_x, seq_len, n_heads/4].
            .grid_3d(1, seq_len / 8, n_heads / 4, [grid_x, 8, 1])
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }
}
