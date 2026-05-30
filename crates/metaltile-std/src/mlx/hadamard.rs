//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Walsh–Hadamard transform along the last axis (size N = 2^k) —
//! port of MLX's `hadamard_n`.
//!
//! Computes `y = H_N · x` where `H_N` is the order-N Hadamard matrix,
//! then scales by `scale`. Used by the Walsh–Hadamard quantization /
//! rotation path (relevant to AURA's rotation matrix).
//!
//! Expressed as the fast Walsh–Hadamard transform: `log2(N)` in-place
//! butterfly passes over a threadgroup buffer. The MLX kernel uses a
//! radix-decomposed multi-step form for register efficiency; this port
//! keeps the plain butterfly — the codegen handles the rest, and one
//! threadgroup per row covers any `N ≤ 1024`. The non-power-of-2
//! `hadamard_m` factor (M ∈ {12,20,28}) is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [N, 1, 1]`.
//! - `N` a power of two, `32 ≤ N ≤ 1024`; one thread per element.
//!
//! Codegen-only; correctness pinned by
//! `tests/hadamard_gpu_correctness.rs`.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! hadamard_kernel {
    ($name:ident, $n:literal, $log_n:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] scale: f32) {
            let row = program_id::<0>();
            let base = row * $n;
            threadgroup_alloc("buf", $n, "f32");
            threadgroup_store("buf", tid, load(inp[base + tid]).cast::<f32>());
            threadgroup_barrier();

            // log2(N) butterfly passes; stride h doubles each pass.
            for s in range(0u32, $log_n, 1u32) {
                let h = 1u32 << s;
                if (tid & h) == 0u32 {
                    let a = threadgroup_load("buf", tid);
                    let b = threadgroup_load("buf", tid + h);
                    threadgroup_store("buf", tid, a + b);
                    threadgroup_store("buf", tid + h, a - b);
                }
                threadgroup_barrier();
            }

            store(out[base + tid], (threadgroup_load("buf", tid) * scale).cast::<T>());
        }
    };
}

hadamard_kernel!(mt_hadamard_n64, 64u32, 6u32, "n64");
hadamard_kernel!(mt_hadamard_n128, 128u32, 7u32, "n128");
hadamard_kernel!(mt_hadamard_n256, 256u32, 8u32, "n256");
hadamard_kernel!(mt_hadamard_n512, 512u32, 9u32, "n512");
hadamard_kernel!(mt_hadamard_n1024, 1024u32, 10u32, "n1024");

/// New-syntax correctness for the Walsh–Hadamard transforms (Reduction mode,
/// one threadgroup per row, tpg=N). Oracle is the algorithm-independent
/// `scale·Σ_j (-1)^popcount(i&j)·x[j]` on dtype-rounded inputs, scale = 1/√N.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(kernel: Kernel, n: usize, dt: DType) -> TestSetup {
        let rows = 2usize;
        let scale = 1.0f32 / (n as f32).sqrt();
        let x: Vec<f32> = (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let xd = unpack_f32(&pack_f32(&x, dt), dt);
        let mut expected = vec![0.0f32; rows * n];
        for r in 0..rows {
            for i in 0..n {
                let acc: f32 = (0..n)
                    .map(|j| {
                        let sign = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                        sign * xd[r * n + j]
                    })
                    .sum();
                expected[r * n + i] = acc * scale;
            }
        }
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&x, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }

    macro_rules! had_test {
        ($name:ident, $kernel:ident, $n:literal) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
            fn $name(dt: DType) -> TestSetup { setup($kernel::kernel_ir_for(dt), $n, dt) }
        };
    }
    had_test!(test_hadamard_n64, mt_hadamard_n64, 64);
    had_test!(test_hadamard_n128, mt_hadamard_n128, 128);
    had_test!(test_hadamard_n256, mt_hadamard_n256, 256);
    had_test!(test_hadamard_n512, mt_hadamard_n512, 512);
    had_test!(test_hadamard_n1024, mt_hadamard_n1024, 1024);
}

/// New-syntax benchmarks for the Walsh–Hadamard transforms.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::*;

    macro_rules! had_bench {
        ($name:ident, $full:literal, $kernel:ident, $n:literal) => {
            #[bench(name = $full, dtypes = [f32, f16, bf16])]
            fn $name(dt: DType) -> BenchSetup {
                let rows = 8192usize;
                let n = $n;
                BenchSetup::new($kernel::kernel_ir_for(dt))
                    .mode(KernelMode::Reduction)
                    .buffer(BenchBuffer::random("inp", rows * n, dt))
                    .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
                    .constexpr("scale", 1.0f32 / (n as f32).sqrt())
                    .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
                    .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
            }
        };
    }
    had_bench!(bench_hadamard_n64, "mlx/hadamard/n64", mt_hadamard_n64, 64);
    had_bench!(bench_hadamard_n128, "mlx/hadamard/n128", mt_hadamard_n128, 128);
    had_bench!(bench_hadamard_n256, "mlx/hadamard/n256", mt_hadamard_n256, 256);
    had_bench!(bench_hadamard_n512, "mlx/hadamard/n512", mt_hadamard_n512, 512);
    had_bench!(bench_hadamard_n1024, "mlx/hadamard/n1024", mt_hadamard_n1024, 1024);
}
