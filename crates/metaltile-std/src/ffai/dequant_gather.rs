//! MLX-format dequantizing gather kernels (quantized embedding tables).
//! For each output element `(token, d)`: look up the packed weight,
//! extract the right value, dequantize via `q * scale + bias`.
//!
//! Layouts (per dtype, with H = `hidden`, G = `group_size`):
//!
//!   weight   [vocab, H * bits / 32]   uint32
//!   scales   [vocab, H / G]           T
//!   biases   [vocab, H / G]           T
//!   indices  [n_tokens]               u32
//!   out      [n_tokens, H]            T
//!
//! One thread per output element. Bit-packing per variant matches
//! mlx-format exactly.
//!
//! Codegen-only — these wrap a quantized gather pattern that mainline
//! MLX has no template for (the MLX gather kernels work on raw tensors,
//! not quantized embeddings). Correctness lives in FFAI integration
//! tests.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const ALL_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

macro_rules! register_dequant_gather {
    ($subop:expr, $name:ident) => {
        inventory::submit! {
            BenchSpec {
                op: "dequant_gather",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: ALL_FLOAT_DTYPES,
                tol: 0.0,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Grid3D),
            }
        }
    };
}

// ─── int4 ────────────────────────────────────────────────────────────

#[kernel]
pub fn dequant_gather_int4<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let packs_per_row = hidden / 8u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let pack_idx = token_id * packs_per_row + d / 8u32;
    let nibble = d & 7u32;
    let packed = load(weight[pack_idx]);
    let q = (packed >> (nibble * 4u32)) & 15u32;
    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}
register_dequant_gather!("int4", dequant_gather_int4);

// ─── int3 ────────────────────────────────────────────────────────────

#[kernel]
pub fn dequant_gather_int3<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 3u32 / 32u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let chunk_idx = d / 8u32;
    let intra = d & 7u32;
    let byte_off = chunk_idx * 3u32;

    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let v0 = b0 & 7u32;
    let v1 = (b0 >> 3u32) & 7u32;
    let v2 = ((b0 >> 6u32) & 3u32) | ((b1 & 1u32) << 2u32);
    let v3 = (b1 >> 1u32) & 7u32;
    let v4 = (b1 >> 4u32) & 7u32;
    let v5 = ((b1 >> 7u32) & 1u32) | ((b2 & 3u32) << 1u32);
    let v6 = (b2 >> 2u32) & 7u32;
    let v7 = (b2 >> 5u32) & 7u32;

    let s01 = select(intra == 0u32, v0, v1);
    let s23 = select(intra == 2u32, v2, v3);
    let s45 = select(intra == 4u32, v4, v5);
    let s67 = select(intra == 6u32, v6, v7);
    let s0123 = select(intra < 2u32, s01, s23);
    let s4567 = select(intra < 6u32, s45, s67);
    let q = select(intra < 4u32, s0123, s4567);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}
register_dequant_gather!("int3", dequant_gather_int3);

// ─── int5 ────────────────────────────────────────────────────────────

#[kernel]
pub fn dequant_gather_int5<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 5u32 / 32u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let chunk_idx = d / 8u32;
    let intra = d & 7u32;
    let byte_off = chunk_idx * 5u32;

    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let s3 = (byte_off + 3u32) & 3u32;
    let s4 = (byte_off + 4u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let in0_3 = (byte_off + 3u32) / 4u32 == u_idx0;
    let in0_4 = (byte_off + 4u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;
    let b3 = (select(in0_3, u0, u1) >> (s3 * 8u32)) & 255u32;
    let b4 = (select(in0_4, u0, u1) >> (s4 * 8u32)) & 255u32;

    let v0 = b0 & 31u32;
    let v1 = ((b0 >> 5u32) & 7u32) | ((b1 & 3u32) << 3u32);
    let v2 = (b1 >> 2u32) & 31u32;
    let v3 = ((b1 >> 7u32) & 1u32) | ((b2 & 15u32) << 1u32);
    let v4 = ((b2 >> 4u32) & 15u32) | ((b3 & 1u32) << 4u32);
    let v5 = (b3 >> 1u32) & 31u32;
    let v6 = ((b3 >> 6u32) & 3u32) | ((b4 & 7u32) << 2u32);
    let v7 = (b4 >> 3u32) & 31u32;

    let s01 = select(intra == 0u32, v0, v1);
    let s23 = select(intra == 2u32, v2, v3);
    let s45 = select(intra == 4u32, v4, v5);
    let s67 = select(intra == 6u32, v6, v7);
    let s0123 = select(intra < 2u32, s01, s23);
    let s4567 = select(intra < 6u32, s45, s67);
    let q = select(intra < 4u32, s0123, s4567);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}
register_dequant_gather!("int5", dequant_gather_int5);

// ─── int6 ────────────────────────────────────────────────────────────

#[kernel]
pub fn dequant_gather_int6<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 3u32 / 16u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let pack_idx = d / 4u32;
    let intra = d & 3u32;
    let byte_off = pack_idx * 3u32;

    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let v0 = b0 & 63u32;
    let v1 = ((b0 >> 6u32) & 3u32) | ((b1 & 15u32) << 2u32);
    let v2 = ((b1 >> 4u32) & 15u32) | ((b2 & 3u32) << 4u32);
    let v3 = (b2 >> 2u32) & 63u32;

    let vsel0 = select(intra == 0u32, v0, v1);
    let vsel1 = select(intra == 2u32, v2, v3);
    let q = select(intra < 2u32, vsel0, vsel1);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}
register_dequant_gather!("int6", dequant_gather_int6);

// ─── int8 ────────────────────────────────────────────────────────────

#[kernel]
pub fn dequant_gather_int8<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let packs_per_row = hidden / 4u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let pack_idx = token_id * packs_per_row + d / 4u32;
    let byte = d & 3u32;
    let packed = load(weight[pack_idx]);
    let q = (packed >> (byte * 8u32)) & 255u32;
    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}
register_dequant_gather!("int8", dequant_gather_int8);
