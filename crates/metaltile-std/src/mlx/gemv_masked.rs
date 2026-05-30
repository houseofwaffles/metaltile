//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Masked GEMV benchmark — #[kernel] DSL (no MLX reference)

use metaltile::kernel;

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

/// New-syntax correctness for `mt_gemv_masked` (`out[r] = Σ_j mat[r,j]·vec[j]·mask[j]`).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_gemv_masked;
    use crate::utils::{pack_f32, unpack_f32};

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 1.0])]
    fn test_mt_gemv_masked(dt: DType) -> TestSetup {
        let (m, k) = (16usize, 256usize);
        let mat: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let vec: Vec<f32> = (0..k).map(|j| ((j % 13) as f32 - 6.0) * 0.02).collect();
        let mask: Vec<f32> = (0..k).map(|j| (j % 3 != 0) as u32 as f32).collect();
        let mat_dt = unpack_f32(&pack_f32(&mat, dt), dt);
        let vec_dt = unpack_f32(&pack_f32(&vec, dt), dt);
        let mask_dt = unpack_f32(&pack_f32(&mask, dt), dt);
        let expected: Vec<f32> = (0..m)
            .map(|r| (0..k).map(|j| mat_dt[r * k + j] * vec_dt[j] * mask_dt[j]).sum())
            .collect();
        TestSetup::new(mt_gemv_masked::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("mat", pack_f32(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack_f32(&vec, dt), dt))
            .input(TestBuffer::from_vec("mask", pack_f32(&mask, dt), dt))
            .input(TestBuffer::zeros("out", m, dt))
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }
}

/// New-syntax benchmark for `mt_gemv_masked`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_gemv_masked;

    #[bench(name = "mlx/gemv_masked", dtypes = [f32, f16, bf16])]
    fn bench_gemv_masked(dt: DType) -> BenchSetup {
        let (m, k) = (4096usize, 4096usize);
        BenchSetup::new(mt_gemv_masked::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("mat", m * k, dt))
            .buffer(BenchBuffer::random("vec", k, dt))
            .buffer(BenchBuffer::random("mask", k, dt))
            .buffer(BenchBuffer::zeros("out", m, dt).output())
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
            .bytes_moved((m * k * dt.size_bytes()) as u64)
    }
}
