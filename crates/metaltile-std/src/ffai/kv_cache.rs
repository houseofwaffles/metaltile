//! KV-cache kernels — raw single-token append + affine-quantized
//! group-quant (int4 / int8) variants used by FFAI's
//! `AffineQuantizedKVCache`.
//!
//! Layouts (all per-layer):
//!
//!   raw    : K, V  [n_kv_heads, max_seq, head_dim]   T
//!   int4/8 : weights [n_kv_heads, max_seq, head_dim / (32/bits)]   u32
//!            scales  [n_kv_heads, max_seq, head_dim / group_size]  T
//!            biases  [n_kv_heads, max_seq, head_dim / group_size]  T
//!
//! Two macros (`quantize_kv_kernel!`, `bulk_dequant_kv_kernel!`) emit the
//! `#[kernel] pub fn …` + `inventory::submit!` blocks at module scope,
//! parameterised by bit-width.  This shape is required: the `#[kernel]`
//! proc-macro doesn't expand inner declarative macros, so embedding the
//! shared body inside an *inner* `macro_rules!` call (the previous file
//! shape) silently produced empty kernels.
//!
//! Codegen-only. End-to-end correctness validated in FFAI integration
//! tests against real model decoding.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// ─── Raw cache append ────────────────────────────────────────────────

// KV cache update — write a one-token K (or V) slice into the
// per-head cache slot at `position`. Source layout: [n_kv_heads, head_dim].
// Dest layout: [n_kv_heads, max_seq, head_dim]. One thread per output
// element (n_kv_heads * head_dim total threads).
#[kernel]
pub fn kv_cache_update<T>(
    src: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] position: u32,
) {
    let idx = program_id::<0>();
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}

inventory::submit! {
    BenchSpec {
        op: "kv_cache",
        subop: "update",
        kernel_name: "kv_cache_update",
        kernel_ir: kv_cache_update::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Grid3D),
    }
}

// ─── Affine quantize (int4 / int8) ────────────────────────────────────
//
// One thread per group.  Scans the group for min/max, derives a safe
// scale + bias, packs `vals_per_pack = 32/bits` quantized values per
// u32 word.  `$bits` is a literal so Metal constant-folds all pack-size
// arithmetic at PSO creation.
macro_rules! quantize_kv_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            src: Tensor<T>,
            mut out_w: Tensor<u32>,
            mut out_s: Tensor<T>,
            mut out_b: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] position: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let max_quant_u = (1u32 << $bits) - 1u32;
            let max_quant_f = max_quant_u.cast::<f32>();

            let g_global = program_id::<0>();
            let groups_per_head = head_dim / group_size;
            let h = g_global / groups_per_head;
            let g_in_h = g_global - h * groups_per_head;
            let d_start = g_in_h * group_size;
            let src_base = h * head_dim;

            let mut mn = load(src[src_base + d_start]).cast::<f32>();
            let mut mx = mn;
            for i in range(1u32, group_size, 1u32) {
                let v = load(src[src_base + d_start + i]).cast::<f32>();
                mn = select(v < mn, v, mn);
                mx = select(v > mx, v, mx);
            }
            let range = mx - mn;
            let safe_scale = select(range == 0.0f32, 1.0f32, range / max_quant_f);
            let inv_scale = 1.0f32 / safe_scale;

            let dst_sb_idx = (h * max_seq + position) * groups_per_head + g_in_h;
            store(out_s[dst_sb_idx], safe_scale.cast::<T>());
            store(out_b[dst_sb_idx], mn.cast::<T>());

            let dst_w_base = (h * max_seq + position) * (head_dim / vals_per_pack) + d_start / vals_per_pack;
            for p in range(0u32, group_size / vals_per_pack, 1u32) {
                let mut packed = 0u32;
                for i in range(0u32, vals_per_pack, 1u32) {
                    let v = load(src[src_base + d_start + p * vals_per_pack + i]).cast::<f32>();
                    let q_f = (v - mn) * inv_scale + 0.5f32;
                    let q_clamped_f = select(q_f > max_quant_f, max_quant_f, select(q_f < 0.0f32, 0.0f32, q_f));
                    let q = q_clamped_f.cast::<u32>();
                    packed = packed | (q << (i * $bits));
                }
                store(out_w[dst_w_base + p], packed);
            }
        }

        inventory::submit! {
            BenchSpec {
                op: "kv_cache",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
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

// ─── Bulk dequant (int4 / int8) ───────────────────────────────────────
//
// One thread per output element.  Looks up scale/bias for its group,
// extracts one quantized value from the packed word, dequantizes.
// Output layout matches raw KVCache: [n_kv_heads, max_seq, head_dim].
macro_rules! bulk_dequant_kv_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            in_w: Tensor<u32>,
            in_s: Tensor<T>,
            in_b: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] n_positions: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;

            let idx = program_id::<0>();
            let total_per_head = n_positions * head_dim;
            let h    = idx / total_per_head;
            let rest = idx - h * total_per_head;
            let pos  = rest / head_dim;
            let d    = rest - pos * head_dim;

            let groups_per_head = head_dim / group_size;
            let g = d / group_size;
            let scale = load(in_s[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();
            let bias  = load(in_b[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();

            let pack_idx = (h * max_seq + pos) * (head_dim / vals_per_pack) + d / vals_per_pack;
            let lane   = d & (vals_per_pack - 1u32);
            let packed = load(in_w[pack_idx]);
            let q      = (packed >> (lane * $bits)) & mask;
            let w_real = q.cast::<f32>() * scale + bias;

            let dst_idx = h * max_seq * head_dim + pos * head_dim + d;
            store(out[dst_idx], w_real.cast::<T>());
        }

        inventory::submit! {
            BenchSpec {
                op: "kv_cache",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
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

quantize_kv_kernel!(quantize_kv_int4, 4u32, "quantize_int4");
quantize_kv_kernel!(quantize_kv_int8, 8u32, "quantize_int8");
bulk_dequant_kv_kernel!(bulk_dequant_kv_int4, 4u32, "bulk_dequant_int4");
bulk_dequant_kv_kernel!(bulk_dequant_kv_int8, 8u32, "bulk_dequant_int8");
