//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! binary_two benchmark — #[kernel] DSL fused two-output elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_binary_two<T>(a: Tensor<T>, b: Tensor<T>, mut c: Tensor<T>, mut d: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    let y = load(b[idx]);
    store(c[idx], x + y);
    store(d[idx], x * y);
}

/// New-syntax correctness for `mt_binary_two` (fused two-output elementwise:
/// `c = a + b`, `d = a * b`). Exercises multiple `.expect()` buffers.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_binary_two;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.04 - 0.25).collect();
        let a_dt = unpack_f32(&pack_f32(&a, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&b, dt), dt);
        let c: Vec<f32> = a_dt.iter().zip(&b_dt).map(|(&x, &y)| x + y).collect();
        let d: Vec<f32> = a_dt.iter().zip(&b_dt).map(|(&x, &y)| x * y).collect();
        TestSetup::new(mt_binary_two::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("c", n, dt))
            .input(TestBuffer::zeros("d", n, dt))
            .expect(TestBuffer::from_vec("c", pack_f32(&c, dt), dt))
            .expect(TestBuffer::from_vec("d", pack_f32(&d, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_mt_binary_two(dt: DType) -> TestSetup { setup(512, dt) }
}

/// New-syntax benchmark for `mt_binary_two` — reads a+b, writes c+d.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_binary_two;

    #[bench(name = "mlx/binary_two/add_mul", dtypes = [f32, f16, bf16])]
    fn bench_binary_two(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_binary_two::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a", n, dt))
            .buffer(BenchBuffer::random("b", n, dt))
            .buffer(BenchBuffer::zeros("c", n, dt).output())
            .buffer(BenchBuffer::zeros("d", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((4 * n * dt.size_bytes()) as u64)
    }
}
