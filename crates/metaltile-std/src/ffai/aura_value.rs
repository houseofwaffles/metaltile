//! AURA compressed-domain value aggregation.
//!
//! For each (q_head, dim) output element, computes
//! `Σ_t weight[head, t] · norm[kv_head, t] · codebook[unpack(packed[t, d])]`,
//! skipping tokens whose weight is below `sparse_threshold`.
//!
//! Port of `turbo_value` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `weights   [q_heads, tokens]`                    f32   — softmax(scores).
//! - `packed    [kv_heads, tokens, packed_width]`     u32   — codebook indices.
//! - `norms     [kv_heads, tokens]`                   f32   — per-position norm.
//! - `codebook  [2**bits]`                            f32   — centroids.
//!
//! Output:
//! - `output    [q_heads, dim]`                       f32
//!
//! ## Dispatch
//!
//! Grid3D, one thread per (q_head, dim) output element.
//! `gid.x = d`, `gid.y = head_idx`.  Each thread runs a single
//! sequential loop over tokens and accumulates its dim slot's
//! contribution.  Sparsity check (`w >= sparse_threshold`) skips
//! cheap-to-zero tokens, mirroring the MLX upstream's
//! flash-pass2-style aggregation guard.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const F32_ONLY: &[DType] = &[DType::F32];

macro_rules! aura_value_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weights: Tensor<f32>,
            packed: Tensor<u32>,
            norms: Tensor<f32>,
            codebook: Tensor<f32>,
            mut output: Tensor<f32>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] sparse_threshold: f32,
        ) {
            let d = program_id::<0>();
            let head_idx = program_id::<1>();
            let kv_head = head_idx / repeat_count;
            let mask = (1u32 << $bits) - 1u32;

            // Pre-compute the bit-stream coordinates for this thread's
            // dim slot.  Same for every token — only the base packed
            // pointer changes per t.
            let bit_offset = d * $bits;
            let word_idx = bit_offset / 32u32;
            let shift = bit_offset & 31u32;
            let bits_in_w0 = 32u32 - shift;
            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
            let spill = $bits - lo_bits;

            let mut acc = 0.0f32;
            for t in range(0u32, tokens, 1u32) {
                let w = load(weights[head_idx * tokens + t]);
                if w >= sparse_threshold {
                    let norm_val = load(norms[kv_head * tokens + t]);
                    let packed_row = (kv_head * tokens + t) * packed_width;

                    let w0 = load(packed[packed_row + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[packed_row + w1_idx]);
                    let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let value = (lo | hi) & mask;

                    let centroid = load(codebook[value]);
                    acc = acc + w * norm_val * centroid;
                }
            }

            store(output[head_idx * dim + d], acc);
        }

        inventory::submit! {
            BenchSpec {
                op: "aura",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: F32_ONLY,
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

aura_value_kernel!(aura_value_int2, 2u32, "value_int2");
aura_value_kernel!(aura_value_int3, 3u32, "value_int3");
aura_value_kernel!(aura_value_int4, 4u32, "value_int4");
aura_value_kernel!(aura_value_int6, 6u32, "value_int6");
aura_value_kernel!(aura_value_int8, 8u32, "value_int8");
