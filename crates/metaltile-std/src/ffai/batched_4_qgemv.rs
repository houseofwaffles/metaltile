//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched 4-output 4-bit quantized GEMV. Fuses FOUR independent
//! projection matvecs that share the same `x` activation into one
//! dispatch. Sibling of `ffai_batched_qkv_qgemv_fast` (3-output) and
//! `ffai_batched_qkv_qmm_fast` (3-output, M>1).
//!
//! Motivation: the Qwen35 GDN layer mixer runs FOUR int4 input
//! projections per decode token off the same `xNorm`: `qkv`, `z`,
//! `b`, `a`. Today that's 4 sequential qmm calls on a shared compute
//! encoder; this collapses them into a single dispatch. At 30 GDN
//! layers per model that's 120 dispatches per decode token saved.
//!
//! Geometry: 8 output rows per TG. `program_id::<2>()` picks the
//! matrix (0=A, 1=B, 2=C, 3=D); `program_id::<0>()` picks the 8-row
//! output tile. TPG = 64 = 2 simdgroups x 32 lanes. Each simdgroup
//! independently computes 4 output rows. Identical inner loop to the
//! 3-output sibling: mask-without-shift int4 dequant + algebraic-split
//! accumulator `acc = s * q_dot + b * xs`.
//!
//! Output split: four separate buffers (`a_out`, `b_out`, `c_out`,
//! `d_out`) so each downstream Tensor consumer reads a contiguous
//! `[out_*]` slice. This mirrors the post-split `ffai_batched_qkv_qmm_fast`
//! layout. Callers can alias all four into one backing allocation if
//! they want; the kernel only sees four base pointers.
//!
//! Grid: `[ceil(max(out_a, out_b, out_c, out_d) / 8), 1, 4]`. TGs past
//! a matrix's `out_*` rows no-op cleanly via the `row0 < out_*` gate.
//!
//! Constraints (same as the 3-output sibling):
//!
//!   * `in_dim % 512 == 0`
//!   * `out_a`, `out_b`, `out_c`, `out_d` each a multiple of 8
//!   * `group_size == 64`
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!
//!   x         [N]           T
//!   w_*       [out_*, N/8]  uint32
//!   scales_*  [out_*, N/G]  T
//!   biases_*  [out_*, N/G]  T
//!   *_out     [out_*]       T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_4_qgemv_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

/// Perf-tuned fused 4-output int4 quantized GEMV. 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups x 32 lanes. Each TG computes
/// 8 output rows of the matrix chosen by `program_id::<2>()`.
/// `simd_id` selects the simdgroup (0 or 1); each simdgroup independently
/// computes 4 output rows (row0..row3). Uses `mt_qmv`'s mask-without-shift
/// trick + algebraic-split accumulator.
///
/// Grid: `[ceil(max(out_a, out_b, out_c, out_d) / 8), 1, 4]`.
/// out_a, out_b, out_c, out_d must be multiples of 8; in_dim must be a
/// multiple of 512; group_size must be 64. TGs past a matrix's `out_*`
/// rows no-op.
#[bench_kernel(
    op="batched_4_qgemv",
    subop="batched_4_qgemv_fast",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_batched_4_qgemv_fast<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<T>,
    biases_a: Tensor<T>,
    w_b: Tensor<u32>,
    scales_b: Tensor<T>,
    biases_b: Tensor<T>,
    w_c: Tensor<u32>,
    scales_c: Tensor<T>,
    biases_c: Tensor<T>,
    w_d: Tensor<u32>,
    scales_d: Tensor<T>,
    biases_d: Tensor<T>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // Each TG covers 8 output rows: SG 0 -> rows 0-3, SG 1 -> rows 4-7.
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
    // The DSL has no function calls so each of the four matrix branches
    // is spelled out with its full inner loop.
    if matrix == 0u32 {
        if row0 < out_a {
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
                // -- A Row 0 --
                let p00 = load(w_a[w_base0 + pack_off]);
                let p01 = load(w_a[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_a[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_a[sb_base0 + g]).cast::<f32>();
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
                // -- A Row 1 --
                let p10 = load(w_a[w_base1 + pack_off]);
                let p11 = load(w_a[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_a[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_a[sb_base1 + g]).cast::<f32>();
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
                // -- A Row 2 --
                let p20 = load(w_a[w_base2 + pack_off]);
                let p21 = load(w_a[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_a[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_a[sb_base2 + g]).cast::<f32>();
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
                // -- A Row 3 --
                let p30 = load(w_a[w_base3 + pack_off]);
                let p31 = load(w_a[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_a[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_a[sb_base3 + g]).cast::<f32>();
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
                store(a_out[row0], r0.cast::<T>());
                store(a_out[row1], r1.cast::<T>());
                store(a_out[row2], r2.cast::<T>());
                store(a_out[row3], r3.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row0 < out_b {
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
                // -- B Row 0 --
                let p00 = load(w_b[w_base0 + pack_off]);
                let p01 = load(w_b[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_b[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_b[sb_base0 + g]).cast::<f32>();
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
                // -- B Row 1 --
                let p10 = load(w_b[w_base1 + pack_off]);
                let p11 = load(w_b[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_b[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_b[sb_base1 + g]).cast::<f32>();
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
                // -- B Row 2 --
                let p20 = load(w_b[w_base2 + pack_off]);
                let p21 = load(w_b[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_b[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_b[sb_base2 + g]).cast::<f32>();
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
                // -- B Row 3 --
                let p30 = load(w_b[w_base3 + pack_off]);
                let p31 = load(w_b[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_b[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_b[sb_base3 + g]).cast::<f32>();
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
                store(b_out[row0], r0.cast::<T>());
                store(b_out[row1], r1.cast::<T>());
                store(b_out[row2], r2.cast::<T>());
                store(b_out[row3], r3.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row0 < out_c {
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
                // -- C Row 0 --
                let p00 = load(w_c[w_base0 + pack_off]);
                let p01 = load(w_c[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_c[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_c[sb_base0 + g]).cast::<f32>();
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
                // -- C Row 1 --
                let p10 = load(w_c[w_base1 + pack_off]);
                let p11 = load(w_c[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_c[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_c[sb_base1 + g]).cast::<f32>();
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
                // -- C Row 2 --
                let p20 = load(w_c[w_base2 + pack_off]);
                let p21 = load(w_c[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_c[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_c[sb_base2 + g]).cast::<f32>();
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
                // -- C Row 3 --
                let p30 = load(w_c[w_base3 + pack_off]);
                let p31 = load(w_c[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_c[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_c[sb_base3 + g]).cast::<f32>();
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
                store(c_out[row0], r0.cast::<T>());
                store(c_out[row1], r1.cast::<T>());
                store(c_out[row2], r2.cast::<T>());
                store(c_out[row3], r3.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row0 < out_d {
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
                // -- D Row 0 --
                let p00 = load(w_d[w_base0 + pack_off]);
                let p01 = load(w_d[w_base0 + pack_off + 1u32]);
                let p00_hi = p00 >> 16u32;
                let p01_hi = p01 >> 16u32;
                let s0 = load(scales_d[sb_base0 + g]).cast::<f32>();
                let bi0 = load(biases_d[sb_base0 + g]).cast::<f32>();
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
                // -- D Row 1 --
                let p10 = load(w_d[w_base1 + pack_off]);
                let p11 = load(w_d[w_base1 + pack_off + 1u32]);
                let p10_hi = p10 >> 16u32;
                let p11_hi = p11 >> 16u32;
                let s1 = load(scales_d[sb_base1 + g]).cast::<f32>();
                let bi1 = load(biases_d[sb_base1 + g]).cast::<f32>();
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
                // -- D Row 2 --
                let p20 = load(w_d[w_base2 + pack_off]);
                let p21 = load(w_d[w_base2 + pack_off + 1u32]);
                let p20_hi = p20 >> 16u32;
                let p21_hi = p21 >> 16u32;
                let s2 = load(scales_d[sb_base2 + g]).cast::<f32>();
                let bi2 = load(biases_d[sb_base2 + g]).cast::<f32>();
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
                // -- D Row 3 --
                let p30 = load(w_d[w_base3 + pack_off]);
                let p31 = load(w_d[w_base3 + pack_off + 1u32]);
                let p30_hi = p30 >> 16u32;
                let p31_hi = p31 >> 16u32;
                let s3 = load(scales_d[sb_base3 + g]).cast::<f32>();
                let bi3 = load(biases_d[sb_base3 + g]).cast::<f32>();
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
                store(d_out[row0], r0.cast::<T>());
                store(d_out[row1], r1.cast::<T>());
                store(d_out[row2], r2.cast::<T>());
                store(d_out[row3], r3.cast::<T>());
            }
        }
    }
}
