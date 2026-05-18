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
//! One thread per output element.  All bit widths share one formula:
//! element `d` occupies bits `[d*bits, (d+1)*bits)` in the row's bit stream,
//! spanning at most two adjacent u32 words.
//!
//! ```text
//!   bit_off  = d * bits
//!   word_idx = bit_off / 32
//!   bit_in_w = bit_off & 31
//!   lo_bits  = min(bits, 32 - bit_in_w)        ← bits from word 0
//!   spill    = bits - lo_bits                   ← bits from word 1
//!   lo       = (w0 >> bit_in_w) & ((1 << lo_bits) - 1)
//!   hi       = (w1 & ((1 << spill) - 1)) << lo_bits
//!   q        = lo | hi
//! ```
//!
//! When `spill == 0`, `w1` loads from `word_idx` (same as w0) so the address
//! is always in-bounds; the `(1 << 0) - 1 == 0` mask zeroes `hi` regardless.
//!
//! Each `#[kernel]` below bakes `bits` in as a literal so Metal constant-folds
//! the arithmetic at PSO creation.  The five functions share identical logic —
//! only the `bits` value differs.

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

// ── shared bit-extraction body ────────────────────────────────────────────
//
// All five kernels below expand this identical body.  `bits` is a u32
// literal in each call site so the compiler sees it as a constant.

#[allow(unused_macros)]
macro_rules! dequant_gather_body {
    ($bits:expr) => {
        let idx = program_id::<0>();
        let token = idx / hidden;
        let d = idx - token * hidden;
        let token_id = load(indices[token]);

        let groups_per_row = hidden / group_size;
        let g = d / group_size;
        let u32_per_row = hidden * $bits / 32u32;
        let row_off = token_id * u32_per_row;

        let bit_off = d * $bits;
        let word_idx = bit_off / 32u32;
        let bit_in_w = bit_off & 31u32;

        let bits_in_w0 = 32u32 - bit_in_w;
        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
        let spill = $bits - lo_bits;

        let w0 = load(weight[row_off + word_idx]);
        let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
        let w1 = load(weight[row_off + w1_idx]);

        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
        let q = lo | hi;

        let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
        let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
        let w_real = q.cast::<f32>() * scale + bias;
        store(out[idx], w_real.cast::<T>());
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
    dequant_gather_body!(4u32);
}
register_dequant_gather!("int4", dequant_gather_int4);

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
    dequant_gather_body!(8u32);
}
register_dequant_gather!("int8", dequant_gather_int8);

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
    dequant_gather_body!(3u32);
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
    dequant_gather_body!(5u32);
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
    dequant_gather_body!(6u32);
}
register_dequant_gather!("int6", dequant_gather_int6);
