//! MLX-format dequantizing GEMV kernels for int3 / int4 / int5 / int6 /
//! int8 weights. Reduction-mode kernels; one threadgroup per output row.
//!
//! Layouts (per dtype, with N = `in_dim`, G = `group_size`):
//!
//!   weight  [out_dim, N * bits / 32]   uint32  (bit-packed)
//!   scales  [out_dim, N / G]           T
//!   biases  [out_dim, N / G]           T
//!   input   [N]                        T
//!   output  [out_dim]                  T
//!
//! Two dispatch strategies, chosen per bit-width:
//!
//! **Pack-strided** (int4, int8) — threads stride over u32 packs; each pack
//! yields `32/bits` values.  One u32 load amortises across all values in
//! the pack; no extra bit-extraction arithmetic beyond a simple shift+mask.
//!
//!   n_packs = in_dim / (32/bits)
//!   per pack: 1 u32 load → (32/bits) shift+mask extractions → (32/bits) FMAs
//!
//! **Element-strided** (int3, int5, int6) — threads stride over individual
//! elements using the two-word bit-stream formula from `dequant_gather.rs`.
//! Odd-width packs don't align to u32 boundaries so pack-striding would
//! require complex cycle handling; element-striding is cleaner and achieves
//! the same cache behaviour (adjacent threads share the same u32 words →
//! L1 multicast) while avoiding the idle-thread problem of the old
//! group-strided approach.
//!
//!   n_iters = ceil(in_dim / lsize)
//!   per iter: 1-2 u32 loads + bit-extract (5 ops) + 1 FMA
//!
//! ## Macro structure
//!
//! Each bit-width gets a `#[kernel] pub fn dequant_gemv_int<bits><T>(…)` plus
//! its `BenchSpec` registration.  The body shape is identical across bit-
//! widths within a strategy, so an outer `macro_rules!` (`dequant_gemv_pow2!`
//! / `dequant_gemv_odd!`) emits the whole `#[kernel] fn …` + `inventory::
//! submit!` at module scope; the compiler expands those before the
//! `#[kernel]` proc-macro runs, so the body parser sees concrete tokens.
//! Embedding the body inside an *inner* `macro_rules!` invocation (the
//! previous shape of this file) silently produced empty kernels — the
//! proc-macro doesn't expand inner declarative macros.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const ALL_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

// ── Pack-strided kernel (int4, int8) ──────────────────────────────────────
//
// Each thread strides over u32 packs. One pack load → (32/$bits) extractions.
// `$bits` must divide 32 evenly (i.e. 4 or 8).
macro_rules! dequant_gemv_pow2 {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] group_size: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let row = program_id::<0>();
            let n_packs_per_row = in_dim / vals_per_pack;
            let n_groups = in_dim / group_size;
            let packs_per_group = group_size / vals_per_pack;
            let row_pack_off = row * n_packs_per_row;
            let row_group_off = row * n_groups;

            let mut acc = 0.0f32;
            let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
            for p_iter in range(0u32, p_iters, 1u32) {
                let pack_idx = p_iter * lsize + tid;
                if pack_idx < n_packs_per_row {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales[row_group_off + g]).cast::<f32>();
                    let bias  = load(biases[row_group_off + g]).cast::<f32>();

                    let packed = load(weight[row_pack_off + pack_idx]);
                    let p_off  = pack_idx * vals_per_pack;

                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * $bits)) & mask;
                        acc = acc + (q.cast::<f32>() * scale + bias) * load(input[p_off + i]).cast::<f32>();
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
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: ALL_FLOAT_DTYPES,
                tol: 0.0,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Reduction),
            }
        }
    };
}

// ── Element-strided kernel (int3, int5, int6) ────────────────────────────
//
// Odd-width packs don't divide u32s evenly, so pack-striding requires
// complex multi-u32 cycle handling. Element-striding with the two-word
// bit-stream formula is cleaner and avoids the idle-thread problem of the
// old group-strided approach (where threads past n_groups sat idle).
// Adjacent threads access adjacent elements → same u32 words → L1 multicast.
macro_rules! dequant_gemv_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] group_size: u32,
        ) {
            let row = program_id::<0>();
            let u32_per_row = in_dim * $bits / 32u32;
            let n_groups = in_dim / group_size;
            let row_u32_off = row * u32_per_row;
            let row_group_off = row * n_groups;

            let mut acc = 0.0f32;
            let n_iters = (in_dim + lsize - 1u32) / lsize;
            for _iter in range(0u32, n_iters, 1u32) {
                let d = _iter * lsize + tid;
                if d < in_dim {
                    let g = d / group_size;
                    let scale = load(scales[row_group_off + g]).cast::<f32>();
                    let bias  = load(biases[row_group_off + g]).cast::<f32>();

                    let bit_off    = d * $bits;
                    let word_idx   = bit_off / 32u32;
                    let bit_in_w   = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits    = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill      = $bits - lo_bits;

                    let w0    = load(weight[row_u32_off + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1    = load(weight[row_u32_off + w1idx]);

                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q  = lo | hi;

                    acc = acc + (q.cast::<f32>() * scale + bias) * load(input[d]).cast::<f32>();
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
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: ALL_FLOAT_DTYPES,
                tol: 0.0,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Reduction),
            }
        }
    };
}

dequant_gemv_pow2!(dequant_gemv_int4, 4u32, "int4");
dequant_gemv_pow2!(dequant_gemv_int8, 8u32, "int8");
dequant_gemv_odd!(dequant_gemv_int3, 3u32, "int3");
dequant_gemv_odd!(dequant_gemv_int5, 5u32, "int5");
dequant_gemv_odd!(dequant_gemv_int6, 6u32, "int6");
