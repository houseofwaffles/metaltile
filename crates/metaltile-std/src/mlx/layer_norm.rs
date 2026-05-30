//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Layer normalization benchmark — #[kernel] DSL vs MLX metal/layer_norm.metal

use metaltile::kernel;

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

/// New-syntax correctness for `mt_layer_norm` (Reduction mode, one threadgroup
/// per row, `tpg = n/4`). Per-row oracle on dtype-rounded inputs:
/// `out_i = (x_i - mean) / sqrt(var + eps) * w_i + b_i`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_layer_norm;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&b, dt), dt);
        let mut x = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let row: Vec<f32> =
                (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.03).collect();
            let xr = unpack_f32(&pack_f32(&row, dt), dt);
            let mean: f32 = xr.iter().sum::<f32>() / n as f32;
            let var: f32 = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..n {
                expected.push((xr[i] - mean) * inv * w_dt[i] + b_dt[i]);
            }
            x.extend_from_slice(&row);
        }
        TestSetup::new(mt_layer_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_mt_layer_norm(dt: DType) -> TestSetup { setup(4, 512, dt) }
}

/// New-syntax benchmark for `mt_layer_norm` (vs MLX `metal/layer_norm.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_layer_norm;

    #[bench(name = "mlx/layer_norm", dtypes = [f32, f16, bf16])]
    fn bench_layer_norm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(mt_layer_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::random("b", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
    }
}
