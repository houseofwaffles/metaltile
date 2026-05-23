//! Batched Q/K/V 4-bit quantized GEMV — fuses the three independent
//! Q, K, V projection matvecs of a decode step into one dispatch.
//!
//! The `z` grid axis selects the matrix (`program_id::<2>()`:
//! 0 = Q, 1 = K, 2 = V); the `x` grid axis is the output tile. The
//! result lands in a single contiguous `y` of length
//! `out_q + out_k + out_v`, with Q, K, V concatenated in that order.
//!
//! Two variants:
//!
//! **`ffai_batched_qkv_qgemv`** — one output row per TG (original
//! correctness-first variant). Grid: `[max(out_q,out_k,out_v), 1, 3]`;
//! `program_id::<0>()` = output row, `program_id::<2>()` = matrix.
//!
//! **`ffai_batched_qkv_qgemv_fast`** — 8 output rows per TG, mirroring
//! `mt_qmv`'s geometry. Each TG computes 8 output rows of the matrix
//! selected by `program_id::<2>()`. Grid:
//! `[ceil(max(out_q,out_k,out_v)/8), 1, 3]`, TPG = 64 (2 simdgroups ×
//! 32 lanes). Uses `mt_qmv`'s mask-without-shift trick + algebraic-split
//! accumulator (`s*q_dot + b*xs`) — identical inner loop to
//! `ffai_rms_norm_qgemv_fast` but without the RMSNorm phase.
//! out_q, out_k, out_v must each be multiples of 8; in_dim must be a
//! multiple of 512; group_size must be 64.
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   w_*       [out_*, N/8]   uint32
//!   scales_*  [out_*, N/G]   T
//!   biases_*  [out_*, N/G]   T
//!   x         [N]            T
//!   y         [out_q+out_k+out_v] T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_qkv_qgemv_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Fused Q/K/V int4 quantized GEMV — one output row per TG.
/// `program_id::<2>()` picks the matrix.
#[kernel]
pub fn ffai_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_group_off = row * n_groups;

    if matrix == 0u32 {
        if row < out_q {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_q[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_q[row_group_off + g]).cast::<f32>();
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_q = reduce_sum(acc);
            if tid == 0u32 {
                store(out[row], total_q.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_k[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_k[row_group_off + g]).cast::<f32>();
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_k = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + row], total_k.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_v[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_v[row_group_off + g]).cast::<f32>();
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_v = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + out_k + row], total_v.cast::<T>());
            }
        }
    }
}

/// Perf-tuned fused Q/K/V int4 quantized GEMV — 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups × 32 lanes. Each TG computes
/// 8 output rows of the matrix chosen by `program_id::<2>()`.
/// `simd_id` selects the simdgroup (0 or 1); each simdgroup independently
/// computes 4 output rows (row0..row3). Uses `mt_qmv`'s mask-without-shift
/// trick + algebraic-split accumulator — identical inner loop to
/// `ffai_rms_norm_qgemv_fast` but without the RMSNorm phase.
///
/// Grid: `[ceil(max(out_q,out_k,out_v)/8), 1, 3]`.
/// out_q, out_k, out_v must be multiples of 8; in_dim must be a multiple
/// of 512; group_size must be 64. TGs past a matrix's out_* rows no-op.
#[kernel]
pub fn ffai_batched_qkv_qgemv_fast<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;

    // Each TG covers 8 output rows: SG 0 → rows 0-3, SG 1 → rows 4-7.
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;

    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    // Mask-without-shift constants (inverse nibble position scaling).
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;

    // The DSL has no function calls so each of the three matrix branches
    // is spelled out with its full inner loop.

    if matrix == 0u32 {
        if row0 < out_q {
            for _b in range(0u32, in_dim, 512u32) {
                let xb = _b + lane_x_off;
                let xi0 = xb;
                let xi1 = xb + 1u32;
                let xi2 = xb + 2u32;
                let xi3 = xb + 3u32;
                let xi4 = xb + 4u32;
                let xi5 = xb + 5u32;
                let xi6 = xb + 6u32;
                let xi7 = xb + 7u32;
                let xi8 = xb + 8u32;
                let xi9 = xb + 9u32;
                let xi10 = xb + 10u32;
                let xi11 = xb + 11u32;
                let xi12 = xb + 12u32;
                let xi13 = xb + 13u32;
                let xi14 = xb + 14u32;
                let xi15 = xb + 15u32;
                let x0 = load(x[xi0]).cast::<f32>();
                let x1_raw = load(x[xi1]).cast::<f32>();
                let x2_raw = load(x[xi2]).cast::<f32>();
                let x3_raw = load(x[xi3]).cast::<f32>();
                let x4 = load(x[xi4]).cast::<f32>();
                let x5_raw = load(x[xi5]).cast::<f32>();
                let x6_raw = load(x[xi6]).cast::<f32>();
                let x7_raw = load(x[xi7]).cast::<f32>();
                let x8 = load(x[xi8]).cast::<f32>();
                let x9_raw = load(x[xi9]).cast::<f32>();
                let x10_raw = load(x[xi10]).cast::<f32>();
                let x11_raw = load(x[xi11]).cast::<f32>();
                let x12 = load(x[xi12]).cast::<f32>();
                let x13_raw = load(x[xi13]).cast::<f32>();
                let x14_raw = load(x[xi14]).cast::<f32>();
                let x15_raw = load(x[xi15]).cast::<f32>();
                let xs = x0
                    + x1_raw
                    + x2_raw
                    + x3_raw
                    + x4
                    + x5_raw
                    + x6_raw
                    + x7_raw
                    + x8
                    + x9_raw
                    + x10_raw
                    + x11_raw
                    + x12
                    + x13_raw
                    + x14_raw
                    + x15_raw;
                let x1 = x1_raw * s_16;
                let x2 = x2_raw * s_256;
                let x3 = x3_raw * s_4096;
                let x5 = x5_raw * s_16;
                let x6 = x6_raw * s_256;
                let x7 = x7_raw * s_4096;
                let x9 = x9_raw * s_16;
                let x10 = x10_raw * s_256;
                let x11 = x11_raw * s_4096;
                let x13 = x13_raw * s_16;
                let x14 = x14_raw * s_256;
                let x15 = x15_raw * s_4096;
                let g = xb / group_size;
                let pack_off = _b / 8u32 + lane_pack_off;
                // ── Q Row 0 ──
                let p00 = load(w_q[w_base0 + pack_off]);
                let p01 = load(w_q[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_q[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_q[sb_base0 + g]).cast::<f32>();
                let q00 = (p00 & 15u32).cast::<f32>();
                let q01 = (p00 & 240u32).cast::<f32>();
                let q02 = (p00 & 3840u32).cast::<f32>();
                let q03 = (p00 & 61440u32).cast::<f32>();
                let q04 = (p00_hi & 15u32).cast::<f32>();
                let q05 = (p00_hi & 240u32).cast::<f32>();
                let q06 = (p00_hi & 3840u32).cast::<f32>();
                let q07 = (p00_hi & 61440u32).cast::<f32>();
                let q08 = (p01 & 15u32).cast::<f32>();
                let q09 = (p01 & 240u32).cast::<f32>();
                let q010 = (p01 & 3840u32).cast::<f32>();
                let q011 = (p01 & 61440u32).cast::<f32>();
                let q012 = (p01_hi & 15u32).cast::<f32>();
                let q013 = (p01_hi & 240u32).cast::<f32>();
                let q014 = (p01_hi & 3840u32).cast::<f32>();
                let q015 = (p01_hi & 61440u32).cast::<f32>();
                let qd0 = q00 * x0
                    + q01 * x1
                    + q02 * x2
                    + q03 * x3
                    + q04 * x4
                    + q05 * x5
                    + q06 * x6
                    + q07 * x7
                    + q08 * x8
                    + q09 * x9
                    + q010 * x10
                    + q011 * x11
                    + q012 * x12
                    + q013 * x13
                    + q014 * x14
                    + q015 * x15;
                acc0 = acc0 + s0 * qd0 + bi0 * xs;
                // ── Q Row 1 ──
                let p10 = load(w_q[w_base1 + pack_off]);
                let p11 = load(w_q[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_q[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_q[sb_base1 + g]).cast::<f32>();
                let q10 = (p10 & 15u32).cast::<f32>();
                let q11 = (p10 & 240u32).cast::<f32>();
                let q12 = (p10 & 3840u32).cast::<f32>();
                let q13 = (p10 & 61440u32).cast::<f32>();
                let q14 = (p10_hi & 15u32).cast::<f32>();
                let q15 = (p10_hi & 240u32).cast::<f32>();
                let q16 = (p10_hi & 3840u32).cast::<f32>();
                let q17 = (p10_hi & 61440u32).cast::<f32>();
                let q18 = (p11 & 15u32).cast::<f32>();
                let q19 = (p11 & 240u32).cast::<f32>();
                let q110 = (p11 & 3840u32).cast::<f32>();
                let q111 = (p11 & 61440u32).cast::<f32>();
                let q112 = (p11_hi & 15u32).cast::<f32>();
                let q113 = (p11_hi & 240u32).cast::<f32>();
                let q114 = (p11_hi & 3840u32).cast::<f32>();
                let q115 = (p11_hi & 61440u32).cast::<f32>();
                let qd1 = q10 * x0
                    + q11 * x1
                    + q12 * x2
                    + q13 * x3
                    + q14 * x4
                    + q15 * x5
                    + q16 * x6
                    + q17 * x7
                    + q18 * x8
                    + q19 * x9
                    + q110 * x10
                    + q111 * x11
                    + q112 * x12
                    + q113 * x13
                    + q114 * x14
                    + q115 * x15;
                acc1 = acc1 + s1 * qd1 + bi1 * xs;
                // ── Q Row 2 ──
                let p20 = load(w_q[w_base2 + pack_off]);
                let p21 = load(w_q[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_q[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_q[sb_base2 + g]).cast::<f32>();
                let q20 = (p20 & 15u32).cast::<f32>();
                let q21 = (p20 & 240u32).cast::<f32>();
                let q22 = (p20 & 3840u32).cast::<f32>();
                let q23 = (p20 & 61440u32).cast::<f32>();
                let q24 = (p20_hi & 15u32).cast::<f32>();
                let q25 = (p20_hi & 240u32).cast::<f32>();
                let q26 = (p20_hi & 3840u32).cast::<f32>();
                let q27 = (p20_hi & 61440u32).cast::<f32>();
                let q28 = (p21 & 15u32).cast::<f32>();
                let q29 = (p21 & 240u32).cast::<f32>();
                let q210 = (p21 & 3840u32).cast::<f32>();
                let q211 = (p21 & 61440u32).cast::<f32>();
                let q212 = (p21_hi & 15u32).cast::<f32>();
                let q213 = (p21_hi & 240u32).cast::<f32>();
                let q214 = (p21_hi & 3840u32).cast::<f32>();
                let q215 = (p21_hi & 61440u32).cast::<f32>();
                let qd2 = q20 * x0
                    + q21 * x1
                    + q22 * x2
                    + q23 * x3
                    + q24 * x4
                    + q25 * x5
                    + q26 * x6
                    + q27 * x7
                    + q28 * x8
                    + q29 * x9
                    + q210 * x10
                    + q211 * x11
                    + q212 * x12
                    + q213 * x13
                    + q214 * x14
                    + q215 * x15;
                acc2 = acc2 + s2 * qd2 + bi2 * xs;
                // ── Q Row 3 ──
                let p30 = load(w_q[w_base3 + pack_off]);
                let p31 = load(w_q[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_q[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_q[sb_base3 + g]).cast::<f32>();
                let q30 = (p30 & 15u32).cast::<f32>();
                let q31 = (p30 & 240u32).cast::<f32>();
                let q32 = (p30 & 3840u32).cast::<f32>();
                let q33 = (p30 & 61440u32).cast::<f32>();
                let q34 = (p30_hi & 15u32).cast::<f32>();
                let q35 = (p30_hi & 240u32).cast::<f32>();
                let q36 = (p30_hi & 3840u32).cast::<f32>();
                let q37 = (p30_hi & 61440u32).cast::<f32>();
                let q38 = (p31 & 15u32).cast::<f32>();
                let q39 = (p31 & 240u32).cast::<f32>();
                let q310 = (p31 & 3840u32).cast::<f32>();
                let q311 = (p31 & 61440u32).cast::<f32>();
                let q312 = (p31_hi & 15u32).cast::<f32>();
                let q313 = (p31_hi & 240u32).cast::<f32>();
                let q314 = (p31_hi & 3840u32).cast::<f32>();
                let q315 = (p31_hi & 61440u32).cast::<f32>();
                let qd3 = q30 * x0
                    + q31 * x1
                    + q32 * x2
                    + q33 * x3
                    + q34 * x4
                    + q35 * x5
                    + q36 * x6
                    + q37 * x7
                    + q38 * x8
                    + q39 * x9
                    + q310 * x10
                    + q311 * x11
                    + q312 * x12
                    + q313 * x13
                    + q314 * x14
                    + q315 * x15;
                acc3 = acc3 + s3 * qd3 + bi3 * xs;
            }
            let r0 = simd_sum(acc0);
            let r1 = simd_sum(acc1);
            let r2 = simd_sum(acc2);
            let r3 = simd_sum(acc3);
            if lane == 0u32 {
                store(out[row0], r0.cast::<T>());
                store(out[row1], r1.cast::<T>());
                store(out[row2], r2.cast::<T>());
                store(out[row3], r3.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row0 < out_k {
            for _b in range(0u32, in_dim, 512u32) {
                let xb = _b + lane_x_off;
                let xi0 = xb;
                let xi1 = xb + 1u32;
                let xi2 = xb + 2u32;
                let xi3 = xb + 3u32;
                let xi4 = xb + 4u32;
                let xi5 = xb + 5u32;
                let xi6 = xb + 6u32;
                let xi7 = xb + 7u32;
                let xi8 = xb + 8u32;
                let xi9 = xb + 9u32;
                let xi10 = xb + 10u32;
                let xi11 = xb + 11u32;
                let xi12 = xb + 12u32;
                let xi13 = xb + 13u32;
                let xi14 = xb + 14u32;
                let xi15 = xb + 15u32;
                let x0 = load(x[xi0]).cast::<f32>();
                let x1_raw = load(x[xi1]).cast::<f32>();
                let x2_raw = load(x[xi2]).cast::<f32>();
                let x3_raw = load(x[xi3]).cast::<f32>();
                let x4 = load(x[xi4]).cast::<f32>();
                let x5_raw = load(x[xi5]).cast::<f32>();
                let x6_raw = load(x[xi6]).cast::<f32>();
                let x7_raw = load(x[xi7]).cast::<f32>();
                let x8 = load(x[xi8]).cast::<f32>();
                let x9_raw = load(x[xi9]).cast::<f32>();
                let x10_raw = load(x[xi10]).cast::<f32>();
                let x11_raw = load(x[xi11]).cast::<f32>();
                let x12 = load(x[xi12]).cast::<f32>();
                let x13_raw = load(x[xi13]).cast::<f32>();
                let x14_raw = load(x[xi14]).cast::<f32>();
                let x15_raw = load(x[xi15]).cast::<f32>();
                let xs = x0
                    + x1_raw
                    + x2_raw
                    + x3_raw
                    + x4
                    + x5_raw
                    + x6_raw
                    + x7_raw
                    + x8
                    + x9_raw
                    + x10_raw
                    + x11_raw
                    + x12
                    + x13_raw
                    + x14_raw
                    + x15_raw;
                let x1 = x1_raw * s_16;
                let x2 = x2_raw * s_256;
                let x3 = x3_raw * s_4096;
                let x5 = x5_raw * s_16;
                let x6 = x6_raw * s_256;
                let x7 = x7_raw * s_4096;
                let x9 = x9_raw * s_16;
                let x10 = x10_raw * s_256;
                let x11 = x11_raw * s_4096;
                let x13 = x13_raw * s_16;
                let x14 = x14_raw * s_256;
                let x15 = x15_raw * s_4096;
                let g = xb / group_size;
                let pack_off = _b / 8u32 + lane_pack_off;
                // ── K Row 0 ──
                let p00 = load(w_k[w_base0 + pack_off]);
                let p01 = load(w_k[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_k[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_k[sb_base0 + g]).cast::<f32>();
                let q00 = (p00 & 15u32).cast::<f32>();
                let q01 = (p00 & 240u32).cast::<f32>();
                let q02 = (p00 & 3840u32).cast::<f32>();
                let q03 = (p00 & 61440u32).cast::<f32>();
                let q04 = (p00_hi & 15u32).cast::<f32>();
                let q05 = (p00_hi & 240u32).cast::<f32>();
                let q06 = (p00_hi & 3840u32).cast::<f32>();
                let q07 = (p00_hi & 61440u32).cast::<f32>();
                let q08 = (p01 & 15u32).cast::<f32>();
                let q09 = (p01 & 240u32).cast::<f32>();
                let q010 = (p01 & 3840u32).cast::<f32>();
                let q011 = (p01 & 61440u32).cast::<f32>();
                let q012 = (p01_hi & 15u32).cast::<f32>();
                let q013 = (p01_hi & 240u32).cast::<f32>();
                let q014 = (p01_hi & 3840u32).cast::<f32>();
                let q015 = (p01_hi & 61440u32).cast::<f32>();
                let qd0 = q00 * x0
                    + q01 * x1
                    + q02 * x2
                    + q03 * x3
                    + q04 * x4
                    + q05 * x5
                    + q06 * x6
                    + q07 * x7
                    + q08 * x8
                    + q09 * x9
                    + q010 * x10
                    + q011 * x11
                    + q012 * x12
                    + q013 * x13
                    + q014 * x14
                    + q015 * x15;
                acc0 = acc0 + s0 * qd0 + bi0 * xs;
                // ── K Row 1 ──
                let p10 = load(w_k[w_base1 + pack_off]);
                let p11 = load(w_k[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_k[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_k[sb_base1 + g]).cast::<f32>();
                let q10 = (p10 & 15u32).cast::<f32>();
                let q11 = (p10 & 240u32).cast::<f32>();
                let q12 = (p10 & 3840u32).cast::<f32>();
                let q13 = (p10 & 61440u32).cast::<f32>();
                let q14 = (p10_hi & 15u32).cast::<f32>();
                let q15 = (p10_hi & 240u32).cast::<f32>();
                let q16 = (p10_hi & 3840u32).cast::<f32>();
                let q17 = (p10_hi & 61440u32).cast::<f32>();
                let q18 = (p11 & 15u32).cast::<f32>();
                let q19 = (p11 & 240u32).cast::<f32>();
                let q110 = (p11 & 3840u32).cast::<f32>();
                let q111 = (p11 & 61440u32).cast::<f32>();
                let q112 = (p11_hi & 15u32).cast::<f32>();
                let q113 = (p11_hi & 240u32).cast::<f32>();
                let q114 = (p11_hi & 3840u32).cast::<f32>();
                let q115 = (p11_hi & 61440u32).cast::<f32>();
                let qd1 = q10 * x0
                    + q11 * x1
                    + q12 * x2
                    + q13 * x3
                    + q14 * x4
                    + q15 * x5
                    + q16 * x6
                    + q17 * x7
                    + q18 * x8
                    + q19 * x9
                    + q110 * x10
                    + q111 * x11
                    + q112 * x12
                    + q113 * x13
                    + q114 * x14
                    + q115 * x15;
                acc1 = acc1 + s1 * qd1 + bi1 * xs;
                // ── K Row 2 ──
                let p20 = load(w_k[w_base2 + pack_off]);
                let p21 = load(w_k[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_k[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_k[sb_base2 + g]).cast::<f32>();
                let q20 = (p20 & 15u32).cast::<f32>();
                let q21 = (p20 & 240u32).cast::<f32>();
                let q22 = (p20 & 3840u32).cast::<f32>();
                let q23 = (p20 & 61440u32).cast::<f32>();
                let q24 = (p20_hi & 15u32).cast::<f32>();
                let q25 = (p20_hi & 240u32).cast::<f32>();
                let q26 = (p20_hi & 3840u32).cast::<f32>();
                let q27 = (p20_hi & 61440u32).cast::<f32>();
                let q28 = (p21 & 15u32).cast::<f32>();
                let q29 = (p21 & 240u32).cast::<f32>();
                let q210 = (p21 & 3840u32).cast::<f32>();
                let q211 = (p21 & 61440u32).cast::<f32>();
                let q212 = (p21_hi & 15u32).cast::<f32>();
                let q213 = (p21_hi & 240u32).cast::<f32>();
                let q214 = (p21_hi & 3840u32).cast::<f32>();
                let q215 = (p21_hi & 61440u32).cast::<f32>();
                let qd2 = q20 * x0
                    + q21 * x1
                    + q22 * x2
                    + q23 * x3
                    + q24 * x4
                    + q25 * x5
                    + q26 * x6
                    + q27 * x7
                    + q28 * x8
                    + q29 * x9
                    + q210 * x10
                    + q211 * x11
                    + q212 * x12
                    + q213 * x13
                    + q214 * x14
                    + q215 * x15;
                acc2 = acc2 + s2 * qd2 + bi2 * xs;
                // ── K Row 3 ──
                let p30 = load(w_k[w_base3 + pack_off]);
                let p31 = load(w_k[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_k[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_k[sb_base3 + g]).cast::<f32>();
                let q30 = (p30 & 15u32).cast::<f32>();
                let q31 = (p30 & 240u32).cast::<f32>();
                let q32 = (p30 & 3840u32).cast::<f32>();
                let q33 = (p30 & 61440u32).cast::<f32>();
                let q34 = (p30_hi & 15u32).cast::<f32>();
                let q35 = (p30_hi & 240u32).cast::<f32>();
                let q36 = (p30_hi & 3840u32).cast::<f32>();
                let q37 = (p30_hi & 61440u32).cast::<f32>();
                let q38 = (p31 & 15u32).cast::<f32>();
                let q39 = (p31 & 240u32).cast::<f32>();
                let q310 = (p31 & 3840u32).cast::<f32>();
                let q311 = (p31 & 61440u32).cast::<f32>();
                let q312 = (p31_hi & 15u32).cast::<f32>();
                let q313 = (p31_hi & 240u32).cast::<f32>();
                let q314 = (p31_hi & 3840u32).cast::<f32>();
                let q315 = (p31_hi & 61440u32).cast::<f32>();
                let qd3 = q30 * x0
                    + q31 * x1
                    + q32 * x2
                    + q33 * x3
                    + q34 * x4
                    + q35 * x5
                    + q36 * x6
                    + q37 * x7
                    + q38 * x8
                    + q39 * x9
                    + q310 * x10
                    + q311 * x11
                    + q312 * x12
                    + q313 * x13
                    + q314 * x14
                    + q315 * x15;
                acc3 = acc3 + s3 * qd3 + bi3 * xs;
            }
            let r0 = simd_sum(acc0);
            let r1 = simd_sum(acc1);
            let r2 = simd_sum(acc2);
            let r3 = simd_sum(acc3);
            if lane == 0u32 {
                store(out[out_q + row0], r0.cast::<T>());
                store(out[out_q + row1], r1.cast::<T>());
                store(out[out_q + row2], r2.cast::<T>());
                store(out[out_q + row3], r3.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row0 < out_v {
            for _b in range(0u32, in_dim, 512u32) {
                let xb = _b + lane_x_off;
                let xi0 = xb;
                let xi1 = xb + 1u32;
                let xi2 = xb + 2u32;
                let xi3 = xb + 3u32;
                let xi4 = xb + 4u32;
                let xi5 = xb + 5u32;
                let xi6 = xb + 6u32;
                let xi7 = xb + 7u32;
                let xi8 = xb + 8u32;
                let xi9 = xb + 9u32;
                let xi10 = xb + 10u32;
                let xi11 = xb + 11u32;
                let xi12 = xb + 12u32;
                let xi13 = xb + 13u32;
                let xi14 = xb + 14u32;
                let xi15 = xb + 15u32;
                let x0 = load(x[xi0]).cast::<f32>();
                let x1_raw = load(x[xi1]).cast::<f32>();
                let x2_raw = load(x[xi2]).cast::<f32>();
                let x3_raw = load(x[xi3]).cast::<f32>();
                let x4 = load(x[xi4]).cast::<f32>();
                let x5_raw = load(x[xi5]).cast::<f32>();
                let x6_raw = load(x[xi6]).cast::<f32>();
                let x7_raw = load(x[xi7]).cast::<f32>();
                let x8 = load(x[xi8]).cast::<f32>();
                let x9_raw = load(x[xi9]).cast::<f32>();
                let x10_raw = load(x[xi10]).cast::<f32>();
                let x11_raw = load(x[xi11]).cast::<f32>();
                let x12 = load(x[xi12]).cast::<f32>();
                let x13_raw = load(x[xi13]).cast::<f32>();
                let x14_raw = load(x[xi14]).cast::<f32>();
                let x15_raw = load(x[xi15]).cast::<f32>();
                let xs = x0
                    + x1_raw
                    + x2_raw
                    + x3_raw
                    + x4
                    + x5_raw
                    + x6_raw
                    + x7_raw
                    + x8
                    + x9_raw
                    + x10_raw
                    + x11_raw
                    + x12
                    + x13_raw
                    + x14_raw
                    + x15_raw;
                let x1 = x1_raw * s_16;
                let x2 = x2_raw * s_256;
                let x3 = x3_raw * s_4096;
                let x5 = x5_raw * s_16;
                let x6 = x6_raw * s_256;
                let x7 = x7_raw * s_4096;
                let x9 = x9_raw * s_16;
                let x10 = x10_raw * s_256;
                let x11 = x11_raw * s_4096;
                let x13 = x13_raw * s_16;
                let x14 = x14_raw * s_256;
                let x15 = x15_raw * s_4096;
                let g = xb / group_size;
                let pack_off = _b / 8u32 + lane_pack_off;
                // ── V Row 0 ──
                let p00 = load(w_v[w_base0 + pack_off]);
                let p01 = load(w_v[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_v[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_v[sb_base0 + g]).cast::<f32>();
                let q00 = (p00 & 15u32).cast::<f32>();
                let q01 = (p00 & 240u32).cast::<f32>();
                let q02 = (p00 & 3840u32).cast::<f32>();
                let q03 = (p00 & 61440u32).cast::<f32>();
                let q04 = (p00_hi & 15u32).cast::<f32>();
                let q05 = (p00_hi & 240u32).cast::<f32>();
                let q06 = (p00_hi & 3840u32).cast::<f32>();
                let q07 = (p00_hi & 61440u32).cast::<f32>();
                let q08 = (p01 & 15u32).cast::<f32>();
                let q09 = (p01 & 240u32).cast::<f32>();
                let q010 = (p01 & 3840u32).cast::<f32>();
                let q011 = (p01 & 61440u32).cast::<f32>();
                let q012 = (p01_hi & 15u32).cast::<f32>();
                let q013 = (p01_hi & 240u32).cast::<f32>();
                let q014 = (p01_hi & 3840u32).cast::<f32>();
                let q015 = (p01_hi & 61440u32).cast::<f32>();
                let qd0 = q00 * x0
                    + q01 * x1
                    + q02 * x2
                    + q03 * x3
                    + q04 * x4
                    + q05 * x5
                    + q06 * x6
                    + q07 * x7
                    + q08 * x8
                    + q09 * x9
                    + q010 * x10
                    + q011 * x11
                    + q012 * x12
                    + q013 * x13
                    + q014 * x14
                    + q015 * x15;
                acc0 = acc0 + s0 * qd0 + bi0 * xs;
                // ── V Row 1 ──
                let p10 = load(w_v[w_base1 + pack_off]);
                let p11 = load(w_v[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_v[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_v[sb_base1 + g]).cast::<f32>();
                let q10 = (p10 & 15u32).cast::<f32>();
                let q11 = (p10 & 240u32).cast::<f32>();
                let q12 = (p10 & 3840u32).cast::<f32>();
                let q13 = (p10 & 61440u32).cast::<f32>();
                let q14 = (p10_hi & 15u32).cast::<f32>();
                let q15 = (p10_hi & 240u32).cast::<f32>();
                let q16 = (p10_hi & 3840u32).cast::<f32>();
                let q17 = (p10_hi & 61440u32).cast::<f32>();
                let q18 = (p11 & 15u32).cast::<f32>();
                let q19 = (p11 & 240u32).cast::<f32>();
                let q110 = (p11 & 3840u32).cast::<f32>();
                let q111 = (p11 & 61440u32).cast::<f32>();
                let q112 = (p11_hi & 15u32).cast::<f32>();
                let q113 = (p11_hi & 240u32).cast::<f32>();
                let q114 = (p11_hi & 3840u32).cast::<f32>();
                let q115 = (p11_hi & 61440u32).cast::<f32>();
                let qd1 = q10 * x0
                    + q11 * x1
                    + q12 * x2
                    + q13 * x3
                    + q14 * x4
                    + q15 * x5
                    + q16 * x6
                    + q17 * x7
                    + q18 * x8
                    + q19 * x9
                    + q110 * x10
                    + q111 * x11
                    + q112 * x12
                    + q113 * x13
                    + q114 * x14
                    + q115 * x15;
                acc1 = acc1 + s1 * qd1 + bi1 * xs;
                // ── V Row 2 ──
                let p20 = load(w_v[w_base2 + pack_off]);
                let p21 = load(w_v[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_v[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_v[sb_base2 + g]).cast::<f32>();
                let q20 = (p20 & 15u32).cast::<f32>();
                let q21 = (p20 & 240u32).cast::<f32>();
                let q22 = (p20 & 3840u32).cast::<f32>();
                let q23 = (p20 & 61440u32).cast::<f32>();
                let q24 = (p20_hi & 15u32).cast::<f32>();
                let q25 = (p20_hi & 240u32).cast::<f32>();
                let q26 = (p20_hi & 3840u32).cast::<f32>();
                let q27 = (p20_hi & 61440u32).cast::<f32>();
                let q28 = (p21 & 15u32).cast::<f32>();
                let q29 = (p21 & 240u32).cast::<f32>();
                let q210 = (p21 & 3840u32).cast::<f32>();
                let q211 = (p21 & 61440u32).cast::<f32>();
                let q212 = (p21_hi & 15u32).cast::<f32>();
                let q213 = (p21_hi & 240u32).cast::<f32>();
                let q214 = (p21_hi & 3840u32).cast::<f32>();
                let q215 = (p21_hi & 61440u32).cast::<f32>();
                let qd2 = q20 * x0
                    + q21 * x1
                    + q22 * x2
                    + q23 * x3
                    + q24 * x4
                    + q25 * x5
                    + q26 * x6
                    + q27 * x7
                    + q28 * x8
                    + q29 * x9
                    + q210 * x10
                    + q211 * x11
                    + q212 * x12
                    + q213 * x13
                    + q214 * x14
                    + q215 * x15;
                acc2 = acc2 + s2 * qd2 + bi2 * xs;
                // ── V Row 3 ──
                let p30 = load(w_v[w_base3 + pack_off]);
                let p31 = load(w_v[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_v[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_v[sb_base3 + g]).cast::<f32>();
                let q30 = (p30 & 15u32).cast::<f32>();
                let q31 = (p30 & 240u32).cast::<f32>();
                let q32 = (p30 & 3840u32).cast::<f32>();
                let q33 = (p30 & 61440u32).cast::<f32>();
                let q34 = (p30_hi & 15u32).cast::<f32>();
                let q35 = (p30_hi & 240u32).cast::<f32>();
                let q36 = (p30_hi & 3840u32).cast::<f32>();
                let q37 = (p30_hi & 61440u32).cast::<f32>();
                let q38 = (p31 & 15u32).cast::<f32>();
                let q39 = (p31 & 240u32).cast::<f32>();
                let q310 = (p31 & 3840u32).cast::<f32>();
                let q311 = (p31 & 61440u32).cast::<f32>();
                let q312 = (p31_hi & 15u32).cast::<f32>();
                let q313 = (p31_hi & 240u32).cast::<f32>();
                let q314 = (p31_hi & 3840u32).cast::<f32>();
                let q315 = (p31_hi & 61440u32).cast::<f32>();
                let qd3 = q30 * x0
                    + q31 * x1
                    + q32 * x2
                    + q33 * x3
                    + q34 * x4
                    + q35 * x5
                    + q36 * x6
                    + q37 * x7
                    + q38 * x8
                    + q39 * x9
                    + q310 * x10
                    + q311 * x11
                    + q312 * x12
                    + q313 * x13
                    + q314 * x14
                    + q315 * x15;
                acc3 = acc3 + s3 * qd3 + bi3 * xs;
            }
            let r0 = simd_sum(acc0);
            let r1 = simd_sum(acc1);
            let r2 = simd_sum(acc2);
            let r3 = simd_sum(acc3);
            if lane == 0u32 {
                store(out[out_q + out_k + row0], r0.cast::<T>());
                store(out[out_q + out_k + row1], r1.cast::<T>());
                store(out[out_q + out_k + row2], r2.cast::<T>());
                store(out[out_q + out_k + row3], r3.cast::<T>());
            }
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "batched_qkv_qgemv",
        subop: "batched_qkv_qgemv",
        kernel_name: "ffai_batched_qkv_qgemv",
        kernel_ir: ffai_batched_qkv_qgemv::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

inventory::submit! {
    BenchSpec {
        op: "batched_qkv_qgemv",
        subop: "batched_qkv_qgemv_fast",
        kernel_name: "ffai_batched_qkv_qgemv_fast",
        kernel_ir: ffai_batched_qkv_qgemv_fast::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
