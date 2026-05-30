//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gather along an axis — contiguous form of MLX's `gather_axis`.
//!
//! `out[o, a, i] = src[o, indices[o, a, i], i]` — for each output
//! element, the middle (axis) coordinate is looked up from `indices`
//! while the outer/inner coordinates pass through. One thread per
//! output element.
//!
//! Layout (row-contiguous):
//!   src:     [outer, axis_size, inner]  T
//!   indices: [outer, axis_out,  inner]  u32
//!   out:     [outer, axis_out,  inner]  T
//!
//! The general MLX kernel handles arbitrary strides / non-contiguous
//! src+idx via `elem_to_loc`; this port covers the row-contiguous case
//! (the shape `ensureRowContiguous` produces).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per output element over `outer*axis_out*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/gather_axis_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn mt_gather_axis<T>(
    src: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] axis_out: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    // out / indices share shape [outer, axis_out, inner]; `idx` indexes
    // both directly. Only the outer coord `o` and inner coord `i` are
    // needed to re-address `src` (which has `axis_size`, not `axis_out`).
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_out * inner);
    let gathered = load(indices[idx]);
    let src_off = (o * axis_size + gathered) * inner + i;
    store(out[idx], load(src[src_off]));
}

/// New-syntax correctness for `mt_gather_axis` (exact — copies elements, no
/// arithmetic). Oracle replicates the gather offset math on dtype-rounded src.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_gather_axis;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_gather_axis(dt: DType) -> TestSetup {
        let (outer, axis_out, axis_size, inner) = (2usize, 3, 5, 4);
        let out_len = outer * axis_out * inner;
        let src_len = outer * axis_size * inner;
        let src: Vec<f32> = (0..src_len).map(|i| i as f32 * 0.1 - 1.0).collect();
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let indices: Vec<u32> =
            (0..out_len).map(|idx| ((idx * 7 + 1) % axis_size) as u32).collect();
        let mut expected = vec![0.0f32; out_len];
        for idx in 0..out_len {
            let i = idx % inner;
            let o = idx / (axis_out * inner);
            let g = indices[idx] as usize;
            expected[idx] = src_dt[(o * axis_size + g) * inner + i];
        }
        TestSetup::new(mt_gather_axis::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", out_len, dt))
            .constexpr("axis_out", axis_out as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(out_len, 256)
    }
}

/// New-syntax benchmark for `mt_gather_axis`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_gather_axis;

    #[bench(name = "mlx/indexing/gather_axis", dtypes = [f32, f16, bf16])]
    fn bench_gather_axis(dt: DType) -> BenchSetup {
        let (outer, axis_out, axis_size, inner) = (4096usize, 64, 64, 64);
        let out_len = outer * axis_out * inner;
        let indices: Vec<u8> = (0..out_len)
            .flat_map(|idx| (((idx * 7 + 1) % axis_size) as u32).to_le_bytes())
            .collect();
        BenchSetup::new(mt_gather_axis::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("src", outer * axis_size * inner, dt))
            .buffer(BenchBuffer::from_vec("indices", indices, DType::U32))
            .buffer(BenchBuffer::zeros("out", out_len, dt).output())
            .constexpr("axis_out", axis_out as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .grid_1d(out_len, 256)
            .bytes_moved((2 * out_len * dt.size_bytes()) as u64)
    }
}
