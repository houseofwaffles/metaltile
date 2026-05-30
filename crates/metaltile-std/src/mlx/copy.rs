//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}

/// New-syntax correctness for `mt_copy` (elementwise, bit-exact).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_copy;
    use crate::utils::pack_f32;

    // Copy is bit-exact within the dtype, so the expected output is just the
    // input packed to `dt` — the GPU reproduces it byte for byte.
    fn setup(n: usize, dt: DType) -> TestSetup {
        let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.1 - 1.0).collect();
        TestSetup::new(mt_copy::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&a, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_copy(dt: DType) -> TestSetup { setup(1024, dt) }
}

/// New-syntax benchmark for `mt_copy` (vs MLX `metal/copy.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_copy;

    // 64M elements (MLX default elementwise size); reads `a`, writes `out`.
    #[bench(name = "mlx/copy", dtypes = [f32, f16, bf16])]
    fn bench_copy(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_copy::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }
}
