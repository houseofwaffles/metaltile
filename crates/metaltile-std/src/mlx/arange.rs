//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Arange benchmark — #[kernel] DSL vs MLX metal/arange.metal
//!
//! MLX kernel: arangefloat32 / arangefloat16 / arangebfloat16 (arange.metal)
//!   Params: (start: constant T&, step: constant T&, out: device T*) — slots [0, 1, 2]
//!   Grid: [ceil(N/1024), 1, 1] × [1024, 1, 1]  (TPG=1024)
//!   Algorithm: out[index] = start + index * step  (one thread per element)
//!
//! MetalTile: mt_arange — same one-thread-per-element algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}

/// Correctness tests for `mt_arange` in the new `#[test_kernel]` syntax.
///
/// These run via `tile test` (and the `kernel_tests_harness` cargo bridge)
/// alongside the legacy `tests/arange_gpu_correctness.rs`, so the old and new
/// paths can be A/B-compared on the same kernel IR during migration.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_arange;
    use crate::utils::{pack_f32, scalar_bytes};

    /// Build a `TestSetup`: a zeroed `out`, scalar `start`/`step`, the constexpr
    /// `n`, and an `out` expectation from a CPU oracle computed in `f32`. The
    /// runner packs the oracle to `dt` and diffs against the GPU output.
    fn setup(start: f32, step: f32, n: usize, dt: DType) -> TestSetup {
        let expected: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::zeros("out", n, dt))
            .input(TestBuffer::from_vec("start", scalar_bytes(start, dt), dt))
            .input(TestBuffer::from_vec("step", scalar_bytes(step, dt), dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // Power-of-two step (0.5) at small magnitudes — bit-exact in every dtype
    // (max value 31.5 is representable in f16 and bf16).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_arange_ascending(dt: DType) -> TestSetup { setup(0.0, 0.5, 64, dt) }

    // Negative integer step — small exact integers in every dtype, so the
    // bit-exact f32 tolerance holds across the board.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_arange_descending(dt: DType) -> TestSetup { setup(16.0, -1.0, 16, dt) }

    // Non-power-of-two step (0.1) exercises per-dtype rounding. f32 computes
    // the oracle and the kernel identically, so it stays bit-exact (1e-6); the
    // f16/bf16 tolerances widen to ~one ULP at magnitude ~6 for their shorter
    // mantissas (measured ≈3.9e-3 / 3.1e-2).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-2, 1e-1])]
    fn test_mt_arange_fractional_step(dt: DType) -> TestSetup { setup(0.0, 0.1, 64, dt) }
}

/// Benchmark for `mt_arange` in the new `#[bench]` syntax. Registered
/// alongside the legacy `#[kernel]` above, so it appears in
/// `tile bench` next to the legacy `arange` row for A/B comparison.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_arange;
    use crate::utils::scalar_bytes;

    // 64M elements matches the MLX default elementwise bench size.
    // bytes_moved counts the output only; the two scalar reads are negligible.
    #[bench(name = "mlx/arange", dtypes = [f32, f16, bf16])]
    fn bench_arange(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_arange::kernel_ir_for(dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .buffer(BenchBuffer::from_vec("start", scalar_bytes(0.0, dt), dt))
            .buffer(BenchBuffer::from_vec("step", scalar_bytes(1.0, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
            .bytes_moved((n * dt.size_bytes()) as u64)
    }
}
