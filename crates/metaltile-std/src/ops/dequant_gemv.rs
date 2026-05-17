//! MLX-format dequantizing GEMV kernels for int3 / int4 / int5 / int6 /
//! int8 weights. Reduction-mode kernels; one threadgroup per output row.
//! Threads stride across packs (not groups) for max in-row parallelism.
//!
//! Layouts (per dtype, with N = `in_dim`, G = `group_size`):
//!
//!   weight  [out_dim, N * bits / 32]   uint32  (bit-packed)
//!   scales  [out_dim, N / G]           T
//!   biases  [out_dim, N / G]           T
//!   input   [N]                        T
//!   output  [out_dim]                  T
//!
//! Bit-packing layout per variant matches mlx-format conventions exactly
//! (see https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/quantized.metal).

use metaltile::kernel;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const ALL_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

// ─── int4 ────────────────────────────────────────────────────────────

// MLX-format int4 dequantizing GEMV — sub-group cooperative version.
// Each thread handles one pack (8 nibbles) at a time, striding by lsize.
// For Qwen3 4B (in_dim=2560, group_size=64): 320 packs per row vs 40
// groups — 8× more thread work than a group-strided version.
#[kernel]
pub fn dequant_gemv_int4<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();

            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;

            let q0 = (packed >> 0u32) & 15u32;
            let q1 = (packed >> 4u32) & 15u32;
            let q2 = (packed >> 8u32) & 15u32;
            let q3 = (packed >> 12u32) & 15u32;
            let q4 = (packed >> 16u32) & 15u32;
            let q5 = (packed >> 20u32) & 15u32;
            let q6 = (packed >> 24u32) & 15u32;
            let q7 = (packed >> 28u32) & 15u32;

            acc = acc + (q0.cast::<f32>() * scale + bias) * load(input[p_off + 0u32]).cast::<f32>();
            acc = acc + (q1.cast::<f32>() * scale + bias) * load(input[p_off + 1u32]).cast::<f32>();
            acc = acc + (q2.cast::<f32>() * scale + bias) * load(input[p_off + 2u32]).cast::<f32>();
            acc = acc + (q3.cast::<f32>() * scale + bias) * load(input[p_off + 3u32]).cast::<f32>();
            acc = acc + (q4.cast::<f32>() * scale + bias) * load(input[p_off + 4u32]).cast::<f32>();
            acc = acc + (q5.cast::<f32>() * scale + bias) * load(input[p_off + 5u32]).cast::<f32>();
            acc = acc + (q6.cast::<f32>() * scale + bias) * load(input[p_off + 6u32]).cast::<f32>();
            acc = acc + (q7.cast::<f32>() * scale + bias) * load(input[p_off + 7u32]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int4",
        kernel_name: "dequant_gemv_int4",
        kernel_ir: dequant_gemv_int4::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}

// ─── int3 ────────────────────────────────────────────────────────────

// MLX-format int3 dequantizing GEMV. 3-bit values: 8 values in 3 bytes
// (24 bits). uint32 cycle: 4 chunks span 3 uint32 (12 bytes → 32 vals).
#[kernel]
pub fn dequant_gemv_int3<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 3u32 / 32u32;
    let u32_per_group = group_size * 3u32 / 32u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
      let g = g_iter * lsize + tid;
      if g < n_groups {
        let scale = load(scales[row_group_off + g]).cast::<f32>();
        let bias = load(biases[row_group_off + g]).cast::<f32>();
        let g_start = g * group_size;
        let g_u32_off = row_u32_off + g * u32_per_group;
        let cycles = group_size / 32u32;

        for c in range(0u32, cycles, 1u32) {
            let cy = g_u32_off + c * 3u32;
            let u0 = load(weight[cy]);
            let u1 = load(weight[cy + 1u32]);
            let u2 = load(weight[cy + 2u32]);
            let xo = g_start + c * 32u32;

            // Chunk 0 — bytes 0,1,2 of u0
            let v0 = u0 & 7u32;
            let v1 = (u0 >> 3u32) & 7u32;
            let v2 = ((u0 >> 6u32) & 3u32) | (((u0 >> 8u32) & 1u32) << 2u32);
            let v3 = (u0 >> 9u32) & 7u32;
            let v4 = (u0 >> 12u32) & 7u32;
            let v5 = ((u0 >> 15u32) & 1u32) | (((u0 >> 16u32) & 3u32) << 1u32);
            let v6 = (u0 >> 18u32) & 7u32;
            let v7 = (u0 >> 21u32) & 7u32;
            acc = acc + (v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
            acc = acc + (v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
            acc = acc + (v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
            acc = acc + (v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();
            acc = acc + (v4.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
            acc = acc + (v5.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
            acc = acc + (v6.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
            acc = acc + (v7.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

            // Chunk 1 — byte 3 of u0, bytes 0,1 of u1
            let v8 = (u0 >> 24u32) & 7u32;
            let v9 = (u0 >> 27u32) & 7u32;
            let v10 = ((u0 >> 30u32) & 3u32) | ((u1 & 1u32) << 2u32);
            let v11 = (u1 >> 1u32) & 7u32;
            let v12 = (u1 >> 4u32) & 7u32;
            let v13 = ((u1 >> 7u32) & 1u32) | (((u1 >> 8u32) & 3u32) << 1u32);
            let v14 = (u1 >> 10u32) & 7u32;
            let v15 = (u1 >> 13u32) & 7u32;
            acc = acc + (v8.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
            acc = acc + (v9.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
            acc = acc + (v10.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
            acc = acc + (v11.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();
            acc = acc + (v12.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
            acc = acc + (v13.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
            acc = acc + (v14.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
            acc = acc + (v15.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();

            // Chunk 2 — bytes 2,3 of u1, byte 0 of u2
            let v16 = (u1 >> 16u32) & 7u32;
            let v17 = (u1 >> 19u32) & 7u32;
            let v18 = ((u1 >> 22u32) & 3u32) | (((u1 >> 24u32) & 1u32) << 2u32);
            let v19 = (u1 >> 25u32) & 7u32;
            let v20 = (u1 >> 28u32) & 7u32;
            let v21 = ((u1 >> 31u32) & 1u32) | ((u2 & 3u32) << 1u32);
            let v22 = (u2 >> 2u32) & 7u32;
            let v23 = (u2 >> 5u32) & 7u32;
            acc = acc + (v16.cast::<f32>() * scale + bias) * load(input[xo + 16u32]).cast::<f32>();
            acc = acc + (v17.cast::<f32>() * scale + bias) * load(input[xo + 17u32]).cast::<f32>();
            acc = acc + (v18.cast::<f32>() * scale + bias) * load(input[xo + 18u32]).cast::<f32>();
            acc = acc + (v19.cast::<f32>() * scale + bias) * load(input[xo + 19u32]).cast::<f32>();
            acc = acc + (v20.cast::<f32>() * scale + bias) * load(input[xo + 20u32]).cast::<f32>();
            acc = acc + (v21.cast::<f32>() * scale + bias) * load(input[xo + 21u32]).cast::<f32>();
            acc = acc + (v22.cast::<f32>() * scale + bias) * load(input[xo + 22u32]).cast::<f32>();
            acc = acc + (v23.cast::<f32>() * scale + bias) * load(input[xo + 23u32]).cast::<f32>();

            // Chunk 3 — bytes 1,2,3 of u2
            let v24 = (u2 >> 8u32) & 7u32;
            let v25 = (u2 >> 11u32) & 7u32;
            let v26 = ((u2 >> 14u32) & 3u32) | (((u2 >> 16u32) & 1u32) << 2u32);
            let v27 = (u2 >> 17u32) & 7u32;
            let v28 = (u2 >> 20u32) & 7u32;
            let v29 = ((u2 >> 23u32) & 1u32) | (((u2 >> 24u32) & 3u32) << 1u32);
            let v30 = (u2 >> 26u32) & 7u32;
            let v31 = (u2 >> 29u32) & 7u32;
            acc = acc + (v24.cast::<f32>() * scale + bias) * load(input[xo + 24u32]).cast::<f32>();
            acc = acc + (v25.cast::<f32>() * scale + bias) * load(input[xo + 25u32]).cast::<f32>();
            acc = acc + (v26.cast::<f32>() * scale + bias) * load(input[xo + 26u32]).cast::<f32>();
            acc = acc + (v27.cast::<f32>() * scale + bias) * load(input[xo + 27u32]).cast::<f32>();
            acc = acc + (v28.cast::<f32>() * scale + bias) * load(input[xo + 28u32]).cast::<f32>();
            acc = acc + (v29.cast::<f32>() * scale + bias) * load(input[xo + 29u32]).cast::<f32>();
            acc = acc + (v30.cast::<f32>() * scale + bias) * load(input[xo + 30u32]).cast::<f32>();
            acc = acc + (v31.cast::<f32>() * scale + bias) * load(input[xo + 31u32]).cast::<f32>();
        }
      }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int3",
        kernel_name: "dequant_gemv_int3",
        kernel_ir: dequant_gemv_int3::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}

// ─── int5 ────────────────────────────────────────────────────────────

// MLX-format int5 dequantizing GEMV. 5-bit values: 8 values in 5 bytes
// (40 bits). uint32 cycle: 4 chunks span 5 uint32 (20 bytes = 32 vals).
//
//   chunk 0: u0 bytes 0-3 + u1 byte 0
//   chunk 1: u1 bytes 1-3 + u2 bytes 0-1
//   chunk 2: u2 bytes 2-3 + u3 bytes 0-2
//   chunk 3: u3 byte 3   + u4 bytes 0-3
#[kernel]
pub fn dequant_gemv_int5<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 5u32 / 32u32;
    let u32_per_group = group_size * 5u32 / 32u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
      let g = g_iter * lsize + tid;
      if g < n_groups {
        let scale = load(scales[row_group_off + g]).cast::<f32>();
        let bias = load(biases[row_group_off + g]).cast::<f32>();
        let g_start = g * group_size;
        let g_u32_off = row_u32_off + g * u32_per_group;
        let cycles = group_size / 32u32;

        for c in range(0u32, cycles, 1u32) {
            let cy = g_u32_off + c * 5u32;
            let u0 = load(weight[cy]);
            let u1 = load(weight[cy + 1u32]);
            let u2 = load(weight[cy + 2u32]);
            let u3 = load(weight[cy + 3u32]);
            let u4 = load(weight[cy + 4u32]);
            let xo = g_start + c * 32u32;

            // Chunk 0 — u0 bytes 0-3 + u1 byte 0
            let v0 = u0 & 31u32;
            let v1 = ((u0 >> 5u32) & 7u32) | (((u0 >> 8u32) & 3u32) << 3u32);
            let v2 = (u0 >> 10u32) & 31u32;
            let v3 = ((u0 >> 15u32) & 1u32) | (((u0 >> 16u32) & 15u32) << 1u32);
            let v4 = ((u0 >> 20u32) & 15u32) | (((u0 >> 24u32) & 1u32) << 4u32);
            let v5 = (u0 >> 25u32) & 31u32;
            let v6 = ((u0 >> 30u32) & 3u32) | ((u1 & 7u32) << 2u32);
            let v7 = (u1 >> 3u32) & 31u32;
            acc = acc + (v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
            acc = acc + (v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
            acc = acc + (v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
            acc = acc + (v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();
            acc = acc + (v4.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
            acc = acc + (v5.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
            acc = acc + (v6.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
            acc = acc + (v7.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

            // Chunk 1 — u1 bytes 1-3 + u2 bytes 0-1
            let w0 = (u1 >> 8u32) & 31u32;
            let w1 = ((u1 >> 13u32) & 7u32) | (((u1 >> 16u32) & 3u32) << 3u32);
            let w2 = (u1 >> 18u32) & 31u32;
            let w3 = ((u1 >> 23u32) & 1u32) | (((u1 >> 24u32) & 15u32) << 1u32);
            let w4 = ((u1 >> 28u32) & 15u32) | ((u2 & 1u32) << 4u32);
            let w5 = (u2 >> 1u32) & 31u32;
            let w6 = ((u2 >> 6u32) & 3u32) | (((u2 >> 8u32) & 7u32) << 2u32);
            let w7 = (u2 >> 11u32) & 31u32;
            acc = acc + (w0.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
            acc = acc + (w1.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
            acc = acc + (w2.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
            acc = acc + (w3.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();
            acc = acc + (w4.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
            acc = acc + (w5.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
            acc = acc + (w6.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
            acc = acc + (w7.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();

            // Chunk 2 — u2 bytes 2-3 + u3 bytes 0-2
            let x0 = (u2 >> 16u32) & 31u32;
            let x1 = ((u2 >> 21u32) & 7u32) | (((u2 >> 24u32) & 3u32) << 3u32);
            let x2 = (u2 >> 26u32) & 31u32;
            let x3 = ((u2 >> 31u32) & 1u32) | ((u3 & 15u32) << 1u32);
            let x4 = ((u3 >> 4u32) & 15u32) | (((u3 >> 8u32) & 1u32) << 4u32);
            let x5 = (u3 >> 9u32) & 31u32;
            let x6 = ((u3 >> 14u32) & 3u32) | (((u3 >> 16u32) & 7u32) << 2u32);
            let x7 = (u3 >> 19u32) & 31u32;
            acc = acc + (x0.cast::<f32>() * scale + bias) * load(input[xo + 16u32]).cast::<f32>();
            acc = acc + (x1.cast::<f32>() * scale + bias) * load(input[xo + 17u32]).cast::<f32>();
            acc = acc + (x2.cast::<f32>() * scale + bias) * load(input[xo + 18u32]).cast::<f32>();
            acc = acc + (x3.cast::<f32>() * scale + bias) * load(input[xo + 19u32]).cast::<f32>();
            acc = acc + (x4.cast::<f32>() * scale + bias) * load(input[xo + 20u32]).cast::<f32>();
            acc = acc + (x5.cast::<f32>() * scale + bias) * load(input[xo + 21u32]).cast::<f32>();
            acc = acc + (x6.cast::<f32>() * scale + bias) * load(input[xo + 22u32]).cast::<f32>();
            acc = acc + (x7.cast::<f32>() * scale + bias) * load(input[xo + 23u32]).cast::<f32>();

            // Chunk 3 — u3 byte 3 + u4 bytes 0-3
            let y0 = (u3 >> 24u32) & 31u32;
            let y1 = ((u3 >> 29u32) & 7u32) | ((u4 & 3u32) << 3u32);
            let y2 = (u4 >> 2u32) & 31u32;
            let y3 = ((u4 >> 7u32) & 1u32) | (((u4 >> 8u32) & 15u32) << 1u32);
            let y4 = ((u4 >> 12u32) & 15u32) | (((u4 >> 16u32) & 1u32) << 4u32);
            let y5 = (u4 >> 17u32) & 31u32;
            let y6 = ((u4 >> 22u32) & 3u32) | (((u4 >> 24u32) & 7u32) << 2u32);
            let y7 = (u4 >> 27u32) & 31u32;
            acc = acc + (y0.cast::<f32>() * scale + bias) * load(input[xo + 24u32]).cast::<f32>();
            acc = acc + (y1.cast::<f32>() * scale + bias) * load(input[xo + 25u32]).cast::<f32>();
            acc = acc + (y2.cast::<f32>() * scale + bias) * load(input[xo + 26u32]).cast::<f32>();
            acc = acc + (y3.cast::<f32>() * scale + bias) * load(input[xo + 27u32]).cast::<f32>();
            acc = acc + (y4.cast::<f32>() * scale + bias) * load(input[xo + 28u32]).cast::<f32>();
            acc = acc + (y5.cast::<f32>() * scale + bias) * load(input[xo + 29u32]).cast::<f32>();
            acc = acc + (y6.cast::<f32>() * scale + bias) * load(input[xo + 30u32]).cast::<f32>();
            acc = acc + (y7.cast::<f32>() * scale + bias) * load(input[xo + 31u32]).cast::<f32>();
        }
      }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int5",
        kernel_name: "dequant_gemv_int5",
        kernel_ir: dequant_gemv_int5::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}

// ─── int6 ────────────────────────────────────────────────────────────

// MLX-format int6 dequantizing GEMV. 6-bit values: 4 values in 3 bytes;
// `in_dim * 3 / 16` uint32 per row. Packs straddle uint32 boundaries
// with a 4-pack / 3-uint32 cycle.
//
// group_size must be a multiple of 16 (typical 32 / 64 / 128).
#[kernel]
pub fn dequant_gemv_int6<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 3u32 / 16u32;
    let u32_per_group = group_size * 3u32 / 16u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
        let g = g_iter * lsize + tid;
        if g < n_groups {
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let g_start = g * group_size;
            let g_u32_off = row_u32_off + g * u32_per_group;
            let chunks = group_size / 16u32;

            for c in range(0u32, chunks, 1u32) {
                let chunk_off = g_u32_off + c * 3u32;
                let u0 = load(weight[chunk_off]);
                let u1 = load(weight[chunk_off + 1u32]);
                let u2 = load(weight[chunk_off + 2u32]);
                let xo = g_start + c * 16u32;

                // Pack 0 — bytes 0,1,2 of u0
                let p0v0 = u0 & 63u32;
                let p0v1 = ((u0 >> 6u32) & 3u32) | (((u0 >> 8u32) & 15u32) << 2u32);
                let p0v2 = ((u0 >> 12u32) & 15u32) | (((u0 >> 16u32) & 3u32) << 4u32);
                let p0v3 = (u0 >> 18u32) & 63u32;
                acc = acc + (p0v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
                acc = acc + (p0v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
                acc = acc + (p0v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
                acc = acc + (p0v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();

                // Pack 1 — byte 3 of u0, bytes 0,1 of u1
                let p1v0 = (u0 >> 24u32) & 63u32;
                let p1v1 = ((u0 >> 30u32) & 3u32) | ((u1 & 15u32) << 2u32);
                let p1v2 = ((u1 >> 4u32) & 15u32) | (((u1 >> 8u32) & 3u32) << 4u32);
                let p1v3 = (u1 >> 10u32) & 63u32;
                acc = acc + (p1v0.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
                acc = acc + (p1v1.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
                acc = acc + (p1v2.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
                acc = acc + (p1v3.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

                // Pack 2 — bytes 2,3 of u1, byte 0 of u2
                let p2v0 = (u1 >> 16u32) & 63u32;
                let p2v1 = ((u1 >> 22u32) & 3u32) | (((u1 >> 24u32) & 15u32) << 2u32);
                let p2v2 = ((u1 >> 28u32) & 15u32) | ((u2 & 3u32) << 4u32);
                let p2v3 = (u2 >> 2u32) & 63u32;
                acc = acc + (p2v0.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
                acc = acc + (p2v1.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
                acc = acc + (p2v2.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
                acc = acc + (p2v3.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();

                // Pack 3 — bytes 1,2,3 of u2
                let p3v0 = (u2 >> 8u32) & 63u32;
                let p3v1 = ((u2 >> 14u32) & 3u32) | (((u2 >> 16u32) & 15u32) << 2u32);
                let p3v2 = ((u2 >> 20u32) & 15u32) | (((u2 >> 24u32) & 3u32) << 4u32);
                let p3v3 = (u2 >> 26u32) & 63u32;
                acc = acc + (p3v0.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
                acc = acc + (p3v1.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
                acc = acc + (p3v2.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
                acc = acc + (p3v3.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int6",
        kernel_name: "dequant_gemv_int6",
        kernel_ir: dequant_gemv_int6::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}

// ─── int8 ────────────────────────────────────────────────────────────

// MLX-format int8 dequantizing GEMV — sub-group cooperative version.
// One threadgroup per output row; threads stride across packs
// (in_dim/4 packs per row), giving max in-row parallelism.
#[kernel]
pub fn dequant_gemv_int8<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 4u32;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / 4u32;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();

            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 4u32;

            let q0 = (packed >> 0u32) & 255u32;
            let q1 = (packed >> 8u32) & 255u32;
            let q2 = (packed >> 16u32) & 255u32;
            let q3 = (packed >> 24u32) & 255u32;

            acc = acc + (q0.cast::<f32>() * scale + bias) * load(input[p_off + 0u32]).cast::<f32>();
            acc = acc + (q1.cast::<f32>() * scale + bias) * load(input[p_off + 1u32]).cast::<f32>();
            acc = acc + (q2.cast::<f32>() * scale + bias) * load(input[p_off + 2u32]).cast::<f32>();
            acc = acc + (q3.cast::<f32>() * scale + bias) * load(input[p_off + 3u32]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int8",
        kernel_name: "dequant_gemv_int8",
        kernel_ir: dequant_gemv_int8::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
    }
}
