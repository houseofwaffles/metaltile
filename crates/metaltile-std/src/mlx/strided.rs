//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Strided copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel(
    bench(
        op="strided_copy",
        subop="strided_copy",
        class=StridedCopy,
        m=1024,
        n=4096,
        pad=128,
        tol=0.0,
        mlx="copy_g_nd2{tn}{tn}",
        metal_file="copy.metal",
    )
)]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}

// ─── mt_strided_copy_nd ──────────────────────────────────────────────────
//
// General N-D strided copy — the MLX `copy_g` / `copy_g_nd{1,2,3}`
// counterpart. The 2-D `mt_strided_copy` above only handles a
// row-major-padded `[rows, cols]` source; this kernel copies an
// arbitrary-rank logical tensor out of a source buffer whose physical
// layout is described by per-dimension `shape` + `strides` arrays.
//
// The destination is always contiguous row-major: output element `p`
// (a flat index in `[0, n_out)`) maps to the multi-index obtained by
// unravelling `p` against `out_shape` (== logical `shape`), then the
// source byte offset is `Σ_d coord_d · strides[d]`. This is exactly
// MLX's `elem_to_loc` (`mlx/backend/metal/kernels/utils.h`).
//
// Because the source strides are *arbitrary* (not necessarily a
// padded row-major view), this generalises:
//   - padded copies         (the 2-D `mt_strided_copy` case),
//   - transposes            (strides permuted vs shape),
//   - broadcasts            (a stride of 0 on a broadcast axis),
//   - any slice / dilation  (non-unit innermost stride).
//
// Inputs:
//   src     — source data buffer (raw, physically strided)
//   shape   — [rank]   u32  logical extent of each dimension
//   strides — [rank]   u32  element stride of each source dimension
//   out     — [n_out]  contiguous row-major output
//
// Constexpr:
//   rank    — number of dimensions (logical). Compile-time constant so
//             the unravel loop is fully unrolled — no dynamic trip count.
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D** — one thread per output element, no cross-thread
//   cooperation. `program_id::<0>()` is the flat output index.
// - **Grid: `[n_out, 1, 1]`, TPG: `[1, 1, 1]`** (or any
//   `grid·tpg == n_out` split). `n_out == Π shape[d]`.
// - **`rank >= 1`.** `shape` and `strides` must each hold exactly
//   `rank` u32 entries; a short buffer reads out of bounds.
// - The unravel walks dimensions **last → first**: the running
//   remainder is divided by `shape[d]` from `d = rank-1` down to `0`,
//   so `strides` is interpreted in the same major-to-minor order as
//   `shape` (row-major logical indexing).
#[kernel(
    bench(
        op="strided_copy",
        subop="strided_copy_nd",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_strided_copy_nd<T>(
    src: Tensor<T>,
    shape: Tensor<u32>,
    strides: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] rank: u32,
) {
    let p = program_id::<0>();
    // Unravel the flat output index `p` against `shape`, walking
    // dimensions from the innermost (last) to the outermost (first).
    // `rem` carries the not-yet-consumed portion of `p`; at each step
    // `coord = rem % shape[d]` peels off dimension `d`'s index and
    // `rem /= shape[d]` advances to the next-coarser dimension. The
    // source offset accumulates `coord · strides[d]`.
    let mut rem = p;
    let mut src_off = 0u32;
    for _i in range(0u32, rank, 1u32) {
        // d counts down: rank-1, rank-2, ..., 0.
        let d = rank - 1u32 - _i;
        let extent = load(shape[d]);
        let coord = rem - (rem / extent) * extent; // rem % extent
        rem = rem / extent;
        src_off = src_off + coord * load(strides[d]);
    }
    store(out[p], load(src[src_off]));
}

/// New-syntax correctness + benchmarks for the strided-copy kernels. Both are
/// exact gathers (`tol = 0`): the oracle replays the same unravel / submatrix
/// index math the kernel uses. `mt_strided_copy` exercises the `#[strided]`
/// metadata ABI — the runtime (test path) and the in-process bench runner both
/// bind the `src_shape` / `src_strides` companion buffers the author supplies.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_strided_copy, mt_strided_copy_nd};
    use crate::utils::{pack_f32, unpack_f32};

    fn u8u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// N-D strided gather: unravel flat index `p` against `shape`, accumulate
    /// `Σ coord_d · strides[d]` (matching the kernel's last→first walk).
    fn nd_oracle(src: &[f32], shape: &[u32], strides: &[u32]) -> Vec<f32> {
        let rank = shape.len();
        let n_out: usize = shape.iter().map(|&s| s as usize).product();
        let mut out = vec![0.0f32; n_out];
        for (p, slot) in out.iter_mut().enumerate() {
            let mut rem = p as u32;
            let mut off = 0u32;
            for i in 0..rank {
                let d = rank - 1 - i;
                let ext = shape[d];
                off += (rem % ext) * strides[d];
                rem /= ext;
            }
            *slot = src[off as usize];
        }
        out
    }

    /// N-D copy as a transpose: logical `[4, 3]` read from a `[3, 4]` physical
    /// buffer (strides `[1, 4]`) — non-trivial strides exercise the unravel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_strided_copy_nd(dt: DType) -> TestSetup {
        let shape = [4u32, 3u32];
        let strides = [1u32, 4u32];
        let src_len = 12usize; // max off = 3·1 + 2·4 = 11
        let src_f: Vec<f32> = (0..src_len).map(|i| (i as f32 - 6.0) * 0.25).collect();
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let expected = nd_oracle(&src, &shape, &strides);
        let n_out = (shape[0] * shape[1]) as usize;
        TestSetup::new(mt_strided_copy_nd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec("shape", u8u32(&shape), DType::U32))
            .input(TestBuffer::from_vec("strides", u8u32(&strides), DType::U32))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("rank", shape.len() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32, 1, 1, [1, 1, 1])
    }

    /// 2-D padded submatrix copy via the `#[strided]` ABI: copy a
    /// `rows × dest_cols` tile out of a `rows × src_cols` padded source.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_strided_copy(dt: DType) -> TestSetup {
        let (rows, src_cols, dest_cols) = (4usize, 8usize, 4usize);
        // Logical value in the copied region, sentinel in the padding.
        let src_f: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..src_cols).map(move |c| {
                    if c < dest_cols { (r * dest_cols + c) as f32 + 1.0 } else { -999.0 }
                })
            })
            .collect();
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let mut expected = Vec::with_capacity(rows * dest_cols);
        for r in 0..rows {
            expected.extend_from_slice(&src[r * src_cols..r * src_cols + dest_cols]);
        }
        TestSetup::new(mt_strided_copy::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec(
                "src_shape",
                u8u32(&[rows as u32, dest_cols as u32]),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("src_strides", u8u32(&[src_cols as u32, 1]), DType::U32))
            .input(TestBuffer::zeros("out", rows * dest_cols, dt))
            .constexpr("cols", dest_cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, dest_cols as u32, 1, [1, 1, 1])
    }
}

/// New-syntax benchmarks for the strided-copy kernels.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_strided_copy, mt_strided_copy_nd};

    fn u8u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// 2-D padded copy, matching the legacy `m=1024 n=4096 pad=128` shape.
    #[bench(name = "mlx/strided_copy/strided_copy", dtypes = [f32, f16, bf16])]
    fn bench_strided_copy(dt: DType) -> BenchSetup {
        let (rows, dest_cols, pad) = (1024usize, 4096usize, 128usize);
        let src_cols = dest_cols + pad;
        BenchSetup::new(mt_strided_copy::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", rows * src_cols, dt))
            .buffer(BenchBuffer::from_vec(
                "src_shape",
                u8u32(&[rows as u32, dest_cols as u32]),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec("src_strides", u8u32(&[src_cols as u32, 1]), DType::U32))
            .buffer(BenchBuffer::zeros("out", rows * dest_cols, dt).output())
            .constexpr("cols", dest_cols as u32)
            .with_shape_label(format!(
                "{rows}×{dest_cols} pad{pad} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(rows as u32, dest_cols as u32, 1, [1, 1, 1])
            .bytes_moved((2 * rows * dest_cols * dt.size_bytes()) as u64)
    }

    /// N-D copy: a 1024×4096 logical transpose out of a 4096×1024 buffer.
    #[bench(name = "mlx/strided_copy/strided_copy_nd", dtypes = [f32, f16, bf16])]
    fn bench_strided_copy_nd(dt: DType) -> BenchSetup {
        let (d0, d1) = (1024usize, 4096usize);
        let n_out = d0 * d1;
        BenchSetup::new(mt_strided_copy_nd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", n_out, dt))
            .buffer(BenchBuffer::from_vec("shape", u8u32(&[d0 as u32, d1 as u32]), DType::U32))
            .buffer(BenchBuffer::from_vec("strides", u8u32(&[1, d0 as u32]), DType::U32))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("rank", 2u32)
            .with_shape_label(format!(
                "{d0}×{d1} transpose {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_out as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
    }
}
