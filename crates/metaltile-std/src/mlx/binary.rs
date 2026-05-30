//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Elementwise binary ops — #[kernel] DSL vs MLX metal/binary.metal

use metaltile::kernel;

#[kernel]
pub fn vector_add<T>(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>) {
    let idx = program_id(0);
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[kernel]
pub fn mt_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) * load(b[idx]));
}

#[kernel]
pub fn mt_sub<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) - load(b[idx]));
}

#[kernel]
pub fn mt_div<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) / load(b[idx]));
}

#[kernel]
pub fn mt_max_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], max(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_min_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], min(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_pow<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], pow(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_atan2<T>(y: Tensor<T>, x: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan2(load(y[idx]), load(x[idx])));
}

#[kernel]
pub fn mt_remainder<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], remainder(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_logaddexp<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(exp(load(a[idx])) + exp(load(b[idx]))));
}

/// New-syntax correctness for the elementwise binary ops.
///
/// Each test rounds its inputs to `dt` (so the oracle sees exactly what the GPU
/// loads), computes the reference in f32, and compares per-dtype. Tolerances
/// follow the f32 figures from the legacy test, widened by ~1 ULP for the
/// shorter f16/bf16 mantissas. `remainder` is bench-only (its floored-vs-
/// truncated semantics vs the reference are unresolved — see the legacy test).
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(modulus: usize, scale: f32, offset: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (i % modulus) as f32 * scale + offset).collect()
    }

    /// Build a two-input elementwise `TestSetup`. Inputs are rounded to `dt`
    /// before the oracle runs, so add/sub/mul/min/max stay bit-exact in every
    /// dtype and only the transcendentals need widened tolerances.
    fn bin<F: Fn(f32, f32) -> f32>(
        kernel: Kernel,
        out_name: &str,
        a: &[f32],
        b: &[f32],
        op: F,
        dt: DType,
    ) -> TestSetup {
        let a_dt = unpack_f32(&pack_f32(a, dt), dt);
        let b_dt = unpack_f32(&pack_f32(b, dt), dt);
        let expected: Vec<f32> = a_dt.iter().zip(&b_dt).map(|(&x, &y)| op(x, y)).collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("a", pack_f32(a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(b, dt), dt))
            .input(TestBuffer::zeros(out_name, a.len(), dt))
            .expect(TestBuffer::from_vec(out_name, pack_f32(&expected, dt), dt))
            .grid_1d(a.len(), 256)
    }

    // vector_add names its output `c`; the rest use `out`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_binary_add(dt: DType) -> TestSetup {
        bin(
            vector_add::kernel_ir_for(dt),
            "c",
            &ramp(17, 0.05, -0.4, 512),
            &ramp(13, 0.04, -0.25, 512),
            |x, y| x + y,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_binary_mul(dt: DType) -> TestSetup {
        bin(
            mt_mul::kernel_ir_for(dt),
            "out",
            &ramp(17, 0.05, -0.4, 512),
            &ramp(13, 0.04, -0.25, 512),
            |x, y| x * y,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_binary_sub(dt: DType) -> TestSetup {
        bin(
            mt_sub::kernel_ir_for(dt),
            "out",
            &ramp(19, 0.07, -0.6, 512),
            &ramp(11, 0.05, -0.3, 512),
            |x, y| x - y,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 2e-1])]
    fn test_binary_div(dt: DType) -> TestSetup {
        // b shifted away from zero.
        bin(
            mt_div::kernel_ir_for(dt),
            "out",
            &ramp(17, 0.06, -0.4, 512),
            &ramp(13, 0.08, 0.2, 512),
            |x, y| x / y,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_binary_max(dt: DType) -> TestSetup {
        bin(
            mt_max_elem::kernel_ir_for(dt),
            "out",
            &ramp(17, 0.05, -0.4, 512),
            &ramp(13, 0.06, -0.35, 512),
            f32::max,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-2, 1e-1])]
    fn test_binary_min(dt: DType) -> TestSetup {
        bin(
            mt_min_elem::kernel_ir_for(dt),
            "out",
            &ramp(17, 0.05, -0.4, 512),
            &ramp(13, 0.06, -0.35, 512),
            f32::min,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_binary_pow(dt: DType) -> TestSetup {
        // Base positive to avoid complex results.
        bin(
            mt_pow::kernel_ir_for(dt),
            "out",
            &ramp(9, 0.1, 0.2, 256),
            &ramp(5, 0.4, 0.2, 256),
            f32::powf,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 2e-1])]
    fn test_binary_atan2(dt: DType) -> TestSetup {
        // mt_atan2(y, x, out): first input is y. `bin` packs the first slice as "a"
        // and the second as "b"; the kernel's params are named y/x, so bind by name.
        let y = ramp(17, 0.1, -0.8, 512);
        let x = ramp(11, 0.1, -0.5, 512);
        let y_dt = unpack_f32(&pack_f32(&y, dt), dt);
        let x_dt = unpack_f32(&pack_f32(&x, dt), dt);
        let expected: Vec<f32> = y_dt.iter().zip(&x_dt).map(|(&yy, &xx)| yy.atan2(xx)).collect();
        TestSetup::new(mt_atan2::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("y", pack_f32(&y, dt), dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::zeros("out", 512, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(512, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_binary_logaddexp(dt: DType) -> TestSetup {
        bin(
            mt_logaddexp::kernel_ir_for(dt),
            "out",
            &ramp(11, 0.3, -1.5, 512),
            &ramp(7, 0.4, -1.0, 512),
            |x, y| (x.exp() + y.exp()).ln(),
            dt,
        )
    }
}

/// New-syntax benchmarks for the elementwise binary ops (vs MLX
/// `metal/binary.metal`). Reads `a` + `b`, writes the output.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::*;

    // Buffers bind by name to the kernel's params, so pass the actual param
    // names (vector_add uses a/b/c; atan2 uses y/x/out; the rest a/b/out).
    fn setup(kernel: metaltile::core::ir::Kernel, names: [&str; 3], dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .buffer(BenchBuffer::random(names[0], n, dt))
            .buffer(BenchBuffer::random(names[1], n, dt))
            .buffer(BenchBuffer::zeros(names[2], n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/binary/add", dtypes = [f32, f16, bf16])]
    fn bench_add(dt: DType) -> BenchSetup {
        setup(vector_add::kernel_ir_for(dt), ["a", "b", "c"], dt)
    }

    #[bench(name = "mlx/binary/mul", dtypes = [f32, f16, bf16])]
    fn bench_mul(dt: DType) -> BenchSetup {
        setup(mt_mul::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/sub", dtypes = [f32, f16, bf16])]
    fn bench_sub(dt: DType) -> BenchSetup {
        setup(mt_sub::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/div", dtypes = [f32, f16, bf16])]
    fn bench_div(dt: DType) -> BenchSetup {
        setup(mt_div::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/maximum", dtypes = [f32, f16, bf16])]
    fn bench_max(dt: DType) -> BenchSetup {
        setup(mt_max_elem::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/minimum", dtypes = [f32, f16, bf16])]
    fn bench_min(dt: DType) -> BenchSetup {
        setup(mt_min_elem::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/pow", dtypes = [f32, f16, bf16])]
    fn bench_pow(dt: DType) -> BenchSetup {
        setup(mt_pow::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/atan2", dtypes = [f32, f16, bf16])]
    fn bench_atan2(dt: DType) -> BenchSetup {
        setup(mt_atan2::kernel_ir_for(dt), ["y", "x", "out"], dt)
    }

    #[bench(name = "mlx/binary/remainder", dtypes = [f32, f16, bf16])]
    fn bench_remainder(dt: DType) -> BenchSetup {
        setup(mt_remainder::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }

    #[bench(name = "mlx/binary/logaddexp", dtypes = [f32, f16, bf16])]
    fn bench_logaddexp(dt: DType) -> BenchSetup {
        setup(mt_logaddexp::kernel_ir_for(dt), ["a", "b", "out"], dt)
    }
}
