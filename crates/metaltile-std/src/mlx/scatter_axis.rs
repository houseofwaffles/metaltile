//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Scatter along an axis — contiguous form of MLX's `scatter_axis`.
//!
//! `out[o, indices[o, a, i], i] = updates[o, a, i]` — each update
//! element is written to a row-`indices`-selected slot of `out`. One
//! thread per update element. `out` is pre-initialized by the caller
//! (typically a copy of the source) and the kernel overwrites the
//! scattered slots.
//!
//! Layout (row-contiguous):
//!   updates: [outer, axis_upd,  inner]  T
//!   indices: [outer, axis_upd,  inner]  u32
//!   out:     [outer, axis_size, inner]  T  (pre-initialized)
//!
//! Assignment (no-reduce) form: distinct `indices` are required for a
//! deterministic result — colliding indices race, matching MLX
//! `scatter_axis` with `reduce = None`. The general strided + reducing
//! kernel is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per update element over `outer*axis_upd*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/scatter_axis_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn mt_scatter_axis<T>(
    updates: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] axis_upd: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_upd * inner);
    let scattered = load(indices[idx]);
    let out_off = (o * axis_size + scattered) * inner + i;
    store(out[out_off], load(updates[idx]));
}

/// New-syntax correctness for `mt_scatter_axis` (exact). Indices are an
/// identity-along-axis pattern so no two updates collide — the scatter is
/// deterministic; unwritten out slots stay zero (matched by the zeroed input).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_scatter_axis;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_scatter_axis(dt: DType) -> TestSetup {
        let (outer, axis_upd, axis_size, inner) = (2usize, 3, 5, 4);
        let upd_len = outer * axis_upd * inner;
        let out_len = outer * axis_size * inner;
        let updates: Vec<f32> = (0..upd_len).map(|i| i as f32 * 0.1 - 1.0).collect();
        let upd_dt = unpack_f32(&pack_f32(&updates, dt), dt);
        // index = the axis coordinate → identity scatter (no collisions).
        let indices: Vec<u32> = (0..upd_len).map(|idx| ((idx / inner) % axis_upd) as u32).collect();
        let mut expected = vec![0.0f32; out_len];
        for idx in 0..upd_len {
            let i = idx % inner;
            let o = idx / (axis_upd * inner);
            let s = indices[idx] as usize;
            expected[(o * axis_size + s) * inner + i] = upd_dt[idx];
        }
        TestSetup::new(mt_scatter_axis::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("updates", pack_f32(&updates, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", out_len, dt))
            .constexpr("axis_upd", axis_upd as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(upd_len, 256)
    }
}

/// New-syntax benchmark for `mt_scatter_axis`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_scatter_axis;

    #[bench(name = "mlx/indexing/scatter_axis", dtypes = [f32, f16, bf16])]
    fn bench_scatter_axis(dt: DType) -> BenchSetup {
        let (outer, axis_upd, axis_size, inner) = (4096usize, 64, 64, 64);
        let upd_len = outer * axis_upd * inner;
        let indices: Vec<u8> = (0..upd_len)
            .flat_map(|idx| (((idx / inner) % axis_upd) as u32).to_le_bytes())
            .collect();
        BenchSetup::new(mt_scatter_axis::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("updates", upd_len, dt))
            .buffer(BenchBuffer::from_vec("indices", indices, DType::U32))
            .buffer(BenchBuffer::zeros("out", outer * axis_size * inner, dt).output())
            .constexpr("axis_upd", axis_upd as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .grid_1d(upd_len, 256)
            .bytes_moved((2 * upd_len * dt.size_bytes()) as u64)
    }
}
