//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused RMSNorm + 4-bit quantized GEMV for decode (single-token).
//!
//! Computes `y = qmatmul(rms_norm(x) * norm_weight, W_q)` in one
//! dispatch, eliminating the global-memory round-trip of the normalized
//! activation between a standalone `rms_norm` and a quantized matvec.
//!
//! Two variants:
//!
//! **`ffai_rms_norm_qgemv`** — one output row per TG (the original port).
//! Reduction-mode: one threadgroup per output row. Phase 1 reduces
//! `sum(x²)` across the threadgroup → `inv_rms`; phase 2 is a
//! pack-strided int4 GEMV that feeds on
//! `normed[i] = x[i] * norm_weight[i] * inv_rms` instead of raw `x`, so
//! the normalized activation never leaves registers. Grid: `[out_dim, 1, 1]`,
//! TPG ≥ 32.
//!
//! **`ffai_rms_norm_qgemv_fast`** — 8 output rows per TG, mirroring
//! `mt_qmv`'s geometry. Phase 1 (SSQ → `inv_rms`) is shared across all
//! 8 rows — the TG-wide reduce amortizes the RMSNorm over 8 outputs.
//! Phase 2 uses the `mt_qmv` mask-without-shift trick (X pre-scaled by
//! inverse nibble position, weight mask returns nibble × position-power)
//! plus the algebraic-split accumulator (`s*Σq·normed + b*Σnormed`),
//! exactly as in MLX `rms_norm_qmm`. Grid: `[out_dim/8, 1, 1]`,
//! TPG = 64 (2 simdgroups × 32 lanes).
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   weight  [out_dim, N/8]    uint32  (8 int4 values per u32)
//!   scales  [out_dim, N/G]    T
//!   biases  [out_dim, N/G]    T
//!   x, norm_weight  [N]       T
//!   y               [out_dim] T
//!
//! ## DISPATCH INVARIANTS (fast variant)
//!
//! - **Grid: `[out_dim/8, 1, 1]`** — one TG per 8-row tile.
//! - **TPG = 64** (2 simdgroups × 32 lanes).
//! - `in_dim` a multiple of 512 (block size = 512 K elements per outer
//!   iter; equivalently `in_dim` must be a multiple of 8 and 64 and ≥ 512).
//! - `out_dim` must be a multiple of 8.
//! - `group_size` must be 64 (one group per 512-K block / 4 lanes).
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_qgemv_gpu_correctness.rs`.

use metaltile::kernel;

/// `y[row] = Σ_i (q[row,i]·scale + bias) · (x[i]·norm_weight[i]·inv_rms)`,
/// with `inv_rms = rsqrt(mean(x²) + eps)`, weights int4-packed.
/// One output row per threadgroup (original correctness-first variant).
#[kernel(
    bench(
        op="rms_norm_qgemv",
        subop="rms_norm_qgemv",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    // Phase 1: RMSNorm — per-thread partial sum of squares, then cross-kernel
    // call to mt_rms_inv_scalar for the threadgroup reduce + rsqrt.
    // ssq is a Value arg; eps_buf and in_dim are Tensor args.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: pack-strided int4 GEMV over the normalized activation.
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;
    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for _p in range(0u32, p_iters, 1u32) {
        let pack_idx = _p * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                let normed = xi * nw * inv_rms;
                acc = acc + (q.cast::<f32>() * scale + bias) * normed;
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Perf-tuned fused RMSNorm + int4 GEMV — 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups × 32 lanes. `simd_id` selects the
/// simdgroup (0 or 1); each simdgroup independently computes 4 output rows.
/// Phase 1 (SSQ for RMSNorm) is shared — `mt_rms_inv_scalar` performs a
/// TG-wide reduce, so the same `inv_rms` is broadcast to all 8 rows.
/// Phase 2 reuses `mt_qmv`'s two-pass algebraic split:
///   `acc = scale * q_dot + bias * normed_xs`
/// where `q_dot = Σ q_i * normed_i` and `normed_xs = Σ normed_i`.
/// The mask-without-shift trick (X pre-scaled by 1/16, 1/256, 1/4096 at
/// nibble positions 1, 2, 3 of each half-word; weight mask returns the
/// nibble × its positional power) eliminates per-nibble shifts — identical
/// to `mt_qmv`. Block = 16 X × 32 lanes = 512 K elements per outer iter.
///
/// Grid: `[out_dim/8, 1, 1]`. out_dim must be a multiple of 8;
/// in_dim must be a multiple of 512; group_size must be 64.
#[kernel(
    bench(
        op="rms_norm_qgemv",
        subop="rms_norm_qgemv_fast",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_rms_norm_qgemv_fast<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // Each TG covers 8 output rows: simdgroup 0 → rows 0-3, sg 1 → rows 4-7.
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;
    // Phase 1: TG-wide SSQ for RMSNorm.
    // All 64 threads cooperate — `mt_rms_inv_scalar` performs the full
    // TG reduce + rsqrt + broadcast, identical to the single-row variant.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: 4-row int4 GEMV per simdgroup, mirroring `mt_qmv`.
    // gs_per_row = in_dim / group_size (= in_dim / 64).
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32
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
    // Each lane covers 16 normed-X values per block. Block = 512 K elements.
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants (inverse nibble position scaling).
    // Eliminates 7 shifts per pack × 2 packs × 4 rows = 56 shifts per block.
    // Mirrors `mt_qmv` and MLX `qdot` (quantized.h:235-244).
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        // Load 16 X values and apply RMSNorm + norm_weight in registers.
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
        // Fuse RMSNorm: normed[i] = x[i] * norm_weight[i] * inv_rms.
        // Raw values needed at nibble positions 1/2/3 for mask-without-shift.
        let n0_raw = load(x[xi0]).cast::<f32>() * load(norm_weight[xi0]).cast::<f32>() * inv_rms;
        let n1_raw = load(x[xi1]).cast::<f32>() * load(norm_weight[xi1]).cast::<f32>() * inv_rms;
        let n2_raw = load(x[xi2]).cast::<f32>() * load(norm_weight[xi2]).cast::<f32>() * inv_rms;
        let n3_raw = load(x[xi3]).cast::<f32>() * load(norm_weight[xi3]).cast::<f32>() * inv_rms;
        let n4_raw = load(x[xi4]).cast::<f32>() * load(norm_weight[xi4]).cast::<f32>() * inv_rms;
        let n5_raw = load(x[xi5]).cast::<f32>() * load(norm_weight[xi5]).cast::<f32>() * inv_rms;
        let n6_raw = load(x[xi6]).cast::<f32>() * load(norm_weight[xi6]).cast::<f32>() * inv_rms;
        let n7_raw = load(x[xi7]).cast::<f32>() * load(norm_weight[xi7]).cast::<f32>() * inv_rms;
        let n8_raw = load(x[xi8]).cast::<f32>() * load(norm_weight[xi8]).cast::<f32>() * inv_rms;
        let n9_raw = load(x[xi9]).cast::<f32>() * load(norm_weight[xi9]).cast::<f32>() * inv_rms;
        let n10_raw = load(x[xi10]).cast::<f32>() * load(norm_weight[xi10]).cast::<f32>() * inv_rms;
        let n11_raw = load(x[xi11]).cast::<f32>() * load(norm_weight[xi11]).cast::<f32>() * inv_rms;
        let n12_raw = load(x[xi12]).cast::<f32>() * load(norm_weight[xi12]).cast::<f32>() * inv_rms;
        let n13_raw = load(x[xi13]).cast::<f32>() * load(norm_weight[xi13]).cast::<f32>() * inv_rms;
        let n14_raw = load(x[xi14]).cast::<f32>() * load(norm_weight[xi14]).cast::<f32>() * inv_rms;
        let n15_raw = load(x[xi15]).cast::<f32>() * load(norm_weight[xi15]).cast::<f32>() * inv_rms;
        // Sum of normed activations for the bias term of the algebraic split.
        let ns = n0_raw
            + n1_raw
            + n2_raw
            + n3_raw
            + n4_raw
            + n5_raw
            + n6_raw
            + n7_raw
            + n8_raw
            + n9_raw
            + n10_raw
            + n11_raw
            + n12_raw
            + n13_raw
            + n14_raw
            + n15_raw;
        // Pre-scale normed values at nibble positions 1/2/3 for
        // mask-without-shift. Position 0 stays unscaled (*1).
        let n1 = n1_raw * s_16;
        let n2 = n2_raw * s_256;
        let n3 = n3_raw * s_4096;
        let n5 = n5_raw * s_16;
        let n6 = n6_raw * s_256;
        let n7 = n7_raw * s_4096;
        let n9 = n9_raw * s_16;
        let n10 = n10_raw * s_256;
        let n11 = n11_raw * s_4096;
        let n13 = n13_raw * s_16;
        let n14 = n14_raw * s_256;
        let n15 = n15_raw * s_4096;
        // Group index — one group per 64 K elements, 4 lanes per group.
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        // ── Row 0 ──
        let p00 = load(weight[w_base0 + pack_off]);
        let p01 = load(weight[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
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
        let qd0 = q00 * n0_raw
            + q01 * n1
            + q02 * n2
            + q03 * n3
            + q04 * n4_raw
            + q05 * n5
            + q06 * n6
            + q07 * n7
            + q08 * n8_raw
            + q09 * n9
            + q010 * n10
            + q011 * n11
            + q012 * n12_raw
            + q013 * n13
            + q014 * n14
            + q015 * n15;
        acc0 = acc0 + s0 * qd0 + bi0 * ns;
        // ── Row 1 ──
        let p10 = load(weight[w_base1 + pack_off]);
        let p11 = load(weight[w_base1 + pack_off + 1u32]);
        let p10_hi = p10 >> 16u32;
        let p11_hi = p11 >> 16u32;
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
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
        let qd1 = q10 * n0_raw
            + q11 * n1
            + q12 * n2
            + q13 * n3
            + q14 * n4_raw
            + q15 * n5
            + q16 * n6
            + q17 * n7
            + q18 * n8_raw
            + q19 * n9
            + q110 * n10
            + q111 * n11
            + q112 * n12_raw
            + q113 * n13
            + q114 * n14
            + q115 * n15;
        acc1 = acc1 + s1 * qd1 + bi1 * ns;
        // ── Row 2 ──
        let p20 = load(weight[w_base2 + pack_off]);
        let p21 = load(weight[w_base2 + pack_off + 1u32]);
        let p20_hi = p20 >> 16u32;
        let p21_hi = p21 >> 16u32;
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
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
        let qd2 = q20 * n0_raw
            + q21 * n1
            + q22 * n2
            + q23 * n3
            + q24 * n4_raw
            + q25 * n5
            + q26 * n6
            + q27 * n7
            + q28 * n8_raw
            + q29 * n9
            + q210 * n10
            + q211 * n11
            + q212 * n12_raw
            + q213 * n13
            + q214 * n14
            + q215 * n15;
        acc2 = acc2 + s2 * qd2 + bi2 * ns;
        // ── Row 3 ──
        let p30 = load(weight[w_base3 + pack_off]);
        let p31 = load(weight[w_base3 + pack_off + 1u32]);
        let p30_hi = p30 >> 16u32;
        let p31_hi = p31 >> 16u32;
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
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
        let qd3 = q30 * n0_raw
            + q31 * n1
            + q32 * n2
            + q33 * n3
            + q34 * n4_raw
            + q35 * n5
            + q36 * n6
            + q37 * n7
            + q38 * n8_raw
            + q39 * n9
            + q310 * n10
            + q311 * n11
            + q312 * n12_raw
            + q313 * n13
            + q314 * n14
            + q315 * n15;
        acc3 = acc3 + s3 * qd3 + bi3 * ns;
    }
    // Cross-lane reduce: each row's partial → one value per simdgroup.
    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(output[row0], r0.cast::<T>());
        store(output[row1], r1.cast::<T>());
        store(output[row2], r2.cast::<T>());
        store(output[row3], r3.cast::<T>());
    }
}

// ─── ffai_rms_norm_qgemv_int8_fast ───────────────────────────────────────────
//
// Fused RMSNorm + int8-quantized GEMV — 8-row-per-TG perf variant.
//
// Mirrors `ffai_rms_norm_qgemv_fast` (int4, 8-row-per-TG, 2 SG × 32 lanes)
// but replaces the int4 nibble-unpack with int8 byte-extract:
//   - 4 bytes per u32 (vals_per_pack = 4 vs 8 for int4)
//   - mask = 0xFF, shifts = 0 / 8 / 16 / 24
//   - packs_per_row = in_dim / 4; lane covers 4 consecutive K positions per pack
//
// Phase 1 (TG-wide SSQ → inv_rms via `mt_rms_inv_scalar`) is identical to
// the int4 fast variant — the RMSNorm is independent of the quantization format.
//
// Phase 2 uses the same algebraic-split accumulator (`s*q_dot + b*normed_xs`)
// that the int4 fast variant uses.
//
// ## DISPATCH INVARIANTS
//
// - **Grid: `[out_dim/8, 1, 1]`** — one TG per 8-row tile.
// - **TPG = 64** (2 simdgroups × 32 lanes).
// - `in_dim` must be a multiple of 512.
// - `out_dim` must be a multiple of 8.
// - `group_size` must be 64.

/// Perf-tuned fused RMSNorm + int8 GEMV — 8 output rows per TG.
///
/// int8 variant of `ffai_rms_norm_qgemv_fast`. Byte-extract (4 vals/pack),
/// algebraic-split accumulator. Grid: `[out_dim/8, 1, 1]`.
#[kernel(
    bench(
        op="rms_norm_qgemv",
        subop="rms_norm_qgemv_int8_fast",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn ffai_rms_norm_qgemv_int8_fast<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // Each TG covers 8 output rows: sg 0 → rows 0-3, sg 1 → rows 4-7.
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;
    // Phase 1: TG-wide SSQ for RMSNorm (same as int4 fast variant).
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: 4-row int8 GEMV per simdgroup, algebraic-split accumulator.
    // int8: 4 bytes per u32, packs_per_row = in_dim / 4.
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 4u32;
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
    // Each lane covers 16 K values per block (512 K / 32 lanes).
    // int8: 4 bytes/pack → 4 packs per lane per block.
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 4u32;
    for _b in range(0u32, in_dim, 512u32) {
        // Load 16 X values, fuse RMSNorm.
        let xb = _b + lane_x_off;
        let n0 = load(x[xb]).cast::<f32>() * load(norm_weight[xb]).cast::<f32>() * inv_rms;
        let n1 =
            load(x[xb + 1u32]).cast::<f32>() * load(norm_weight[xb + 1u32]).cast::<f32>() * inv_rms;
        let n2 =
            load(x[xb + 2u32]).cast::<f32>() * load(norm_weight[xb + 2u32]).cast::<f32>() * inv_rms;
        let n3 =
            load(x[xb + 3u32]).cast::<f32>() * load(norm_weight[xb + 3u32]).cast::<f32>() * inv_rms;
        let n4 =
            load(x[xb + 4u32]).cast::<f32>() * load(norm_weight[xb + 4u32]).cast::<f32>() * inv_rms;
        let n5 =
            load(x[xb + 5u32]).cast::<f32>() * load(norm_weight[xb + 5u32]).cast::<f32>() * inv_rms;
        let n6 =
            load(x[xb + 6u32]).cast::<f32>() * load(norm_weight[xb + 6u32]).cast::<f32>() * inv_rms;
        let n7 =
            load(x[xb + 7u32]).cast::<f32>() * load(norm_weight[xb + 7u32]).cast::<f32>() * inv_rms;
        let n8 =
            load(x[xb + 8u32]).cast::<f32>() * load(norm_weight[xb + 8u32]).cast::<f32>() * inv_rms;
        let n9 =
            load(x[xb + 9u32]).cast::<f32>() * load(norm_weight[xb + 9u32]).cast::<f32>() * inv_rms;
        let n10 = load(x[xb + 10u32]).cast::<f32>()
            * load(norm_weight[xb + 10u32]).cast::<f32>()
            * inv_rms;
        let n11 = load(x[xb + 11u32]).cast::<f32>()
            * load(norm_weight[xb + 11u32]).cast::<f32>()
            * inv_rms;
        let n12 = load(x[xb + 12u32]).cast::<f32>()
            * load(norm_weight[xb + 12u32]).cast::<f32>()
            * inv_rms;
        let n13 = load(x[xb + 13u32]).cast::<f32>()
            * load(norm_weight[xb + 13u32]).cast::<f32>()
            * inv_rms;
        let n14 = load(x[xb + 14u32]).cast::<f32>()
            * load(norm_weight[xb + 14u32]).cast::<f32>()
            * inv_rms;
        let n15 = load(x[xb + 15u32]).cast::<f32>()
            * load(norm_weight[xb + 15u32]).cast::<f32>()
            * inv_rms;
        // Bias accumulation sum for algebraic split.
        let ns =
            n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15;
        let g = xb / group_size;
        let pack_off = _b / 4u32 + lane_pack_off;
        // ── Row 0 ──
        let p00 = load(weight[w_base0 + pack_off]);
        let p01 = load(weight[w_base0 + pack_off + 1u32]);
        let p02 = load(weight[w_base0 + pack_off + 2u32]);
        let p03 = load(weight[w_base0 + pack_off + 3u32]);
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        let qd0 = (p00 & 255u32).cast::<f32>() * n0
            + ((p00 >> 8u32) & 255u32).cast::<f32>() * n1
            + ((p00 >> 16u32) & 255u32).cast::<f32>() * n2
            + ((p00 >> 24u32) & 255u32).cast::<f32>() * n3
            + (p01 & 255u32).cast::<f32>() * n4
            + ((p01 >> 8u32) & 255u32).cast::<f32>() * n5
            + ((p01 >> 16u32) & 255u32).cast::<f32>() * n6
            + ((p01 >> 24u32) & 255u32).cast::<f32>() * n7
            + (p02 & 255u32).cast::<f32>() * n8
            + ((p02 >> 8u32) & 255u32).cast::<f32>() * n9
            + ((p02 >> 16u32) & 255u32).cast::<f32>() * n10
            + ((p02 >> 24u32) & 255u32).cast::<f32>() * n11
            + (p03 & 255u32).cast::<f32>() * n12
            + ((p03 >> 8u32) & 255u32).cast::<f32>() * n13
            + ((p03 >> 16u32) & 255u32).cast::<f32>() * n14
            + ((p03 >> 24u32) & 255u32).cast::<f32>() * n15;
        acc0 = acc0 + s0 * qd0 + bi0 * ns;
        // ── Row 1 ──
        let p10 = load(weight[w_base1 + pack_off]);
        let p11 = load(weight[w_base1 + pack_off + 1u32]);
        let p12 = load(weight[w_base1 + pack_off + 2u32]);
        let p13 = load(weight[w_base1 + pack_off + 3u32]);
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let qd1 = (p10 & 255u32).cast::<f32>() * n0
            + ((p10 >> 8u32) & 255u32).cast::<f32>() * n1
            + ((p10 >> 16u32) & 255u32).cast::<f32>() * n2
            + ((p10 >> 24u32) & 255u32).cast::<f32>() * n3
            + (p11 & 255u32).cast::<f32>() * n4
            + ((p11 >> 8u32) & 255u32).cast::<f32>() * n5
            + ((p11 >> 16u32) & 255u32).cast::<f32>() * n6
            + ((p11 >> 24u32) & 255u32).cast::<f32>() * n7
            + (p12 & 255u32).cast::<f32>() * n8
            + ((p12 >> 8u32) & 255u32).cast::<f32>() * n9
            + ((p12 >> 16u32) & 255u32).cast::<f32>() * n10
            + ((p12 >> 24u32) & 255u32).cast::<f32>() * n11
            + (p13 & 255u32).cast::<f32>() * n12
            + ((p13 >> 8u32) & 255u32).cast::<f32>() * n13
            + ((p13 >> 16u32) & 255u32).cast::<f32>() * n14
            + ((p13 >> 24u32) & 255u32).cast::<f32>() * n15;
        acc1 = acc1 + s1 * qd1 + bi1 * ns;
        // ── Row 2 ──
        let p20 = load(weight[w_base2 + pack_off]);
        let p21 = load(weight[w_base2 + pack_off + 1u32]);
        let p22 = load(weight[w_base2 + pack_off + 2u32]);
        let p23 = load(weight[w_base2 + pack_off + 3u32]);
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let qd2 = (p20 & 255u32).cast::<f32>() * n0
            + ((p20 >> 8u32) & 255u32).cast::<f32>() * n1
            + ((p20 >> 16u32) & 255u32).cast::<f32>() * n2
            + ((p20 >> 24u32) & 255u32).cast::<f32>() * n3
            + (p21 & 255u32).cast::<f32>() * n4
            + ((p21 >> 8u32) & 255u32).cast::<f32>() * n5
            + ((p21 >> 16u32) & 255u32).cast::<f32>() * n6
            + ((p21 >> 24u32) & 255u32).cast::<f32>() * n7
            + (p22 & 255u32).cast::<f32>() * n8
            + ((p22 >> 8u32) & 255u32).cast::<f32>() * n9
            + ((p22 >> 16u32) & 255u32).cast::<f32>() * n10
            + ((p22 >> 24u32) & 255u32).cast::<f32>() * n11
            + (p23 & 255u32).cast::<f32>() * n12
            + ((p23 >> 8u32) & 255u32).cast::<f32>() * n13
            + ((p23 >> 16u32) & 255u32).cast::<f32>() * n14
            + ((p23 >> 24u32) & 255u32).cast::<f32>() * n15;
        acc2 = acc2 + s2 * qd2 + bi2 * ns;
        // ── Row 3 ──
        let p30 = load(weight[w_base3 + pack_off]);
        let p31 = load(weight[w_base3 + pack_off + 1u32]);
        let p32 = load(weight[w_base3 + pack_off + 2u32]);
        let p33 = load(weight[w_base3 + pack_off + 3u32]);
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let qd3 = (p30 & 255u32).cast::<f32>() * n0
            + ((p30 >> 8u32) & 255u32).cast::<f32>() * n1
            + ((p30 >> 16u32) & 255u32).cast::<f32>() * n2
            + ((p30 >> 24u32) & 255u32).cast::<f32>() * n3
            + (p31 & 255u32).cast::<f32>() * n4
            + ((p31 >> 8u32) & 255u32).cast::<f32>() * n5
            + ((p31 >> 16u32) & 255u32).cast::<f32>() * n6
            + ((p31 >> 24u32) & 255u32).cast::<f32>() * n7
            + (p32 & 255u32).cast::<f32>() * n8
            + ((p32 >> 8u32) & 255u32).cast::<f32>() * n9
            + ((p32 >> 16u32) & 255u32).cast::<f32>() * n10
            + ((p32 >> 24u32) & 255u32).cast::<f32>() * n11
            + (p33 & 255u32).cast::<f32>() * n12
            + ((p33 >> 8u32) & 255u32).cast::<f32>() * n13
            + ((p33 >> 16u32) & 255u32).cast::<f32>() * n14
            + ((p33 >> 24u32) & 255u32).cast::<f32>() * n15;
        acc3 = acc3 + s3 * qd3 + bi3 * ns;
    }
    // Cross-lane reduce: each row → one value per simdgroup.
    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(output[row0], r0.cast::<T>());
        store(output[row1], r1.cast::<T>());
        store(output[row2], r2.cast::<T>());
        store(output[row3], r3.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{ffai_rms_norm_qgemv, ffai_rms_norm_qgemv_fast, ffai_rms_norm_qgemv_int8_fast};
    use crate::utils::{pack_f32, unpack_f32};

    const EPS: f32 = 1e-5;

    /// Round one f32 through the kernel dtype, matching the GPU load-cast.
    fn round(v: f32, dt: DType) -> f32 { unpack_f32(&pack_f32(&[v], dt), dt)[0] }

    /// Pack u32 weight words to little-endian bytes.
    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

    /// Deterministic xorshift source — matches the legacy test generator so
    /// the dtype-rounded oracle inputs reproduce the same distribution.
    fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
            })
            .collect()
    }

    /// Affine per-group int4 quantize of one weight row, nibble-packed
    /// (8 values per u32). Mirrors the legacy GPU-test quantizer exactly.
    fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let in_dim = row.len();
        let n_groups = in_dim / group_size;
        let mut packed = vec![0u32; in_dim / 8];
        let mut scales = vec![0.0_f32; n_groups];
        let mut biases = vec![0.0_f32; n_groups];
        for g in 0..n_groups {
            let gs = &row[g * group_size..(g + 1) * group_size];
            let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = mx - mn;
            let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
            scales[g] = scale;
            biases[g] = mn;
            for (i, &v) in gs.iter().enumerate() {
                let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
                let d = g * group_size + i;
                packed[d / 8] |= q << ((d % 8) * 4);
            }
        }
        (packed, scales, biases)
    }

    /// Affine per-group int8 quantize of one weight row — 4 bytes per u32,
    /// the byte-strided layout `ffai_rms_norm_qgemv_int8_fast` decodes.
    fn quantize_int8_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let in_dim = row.len();
        let n_groups = in_dim / group_size;
        let mut packed = vec![0u32; in_dim / 4];
        let mut scales = vec![0.0_f32; n_groups];
        let mut biases = vec![0.0_f32; n_groups];
        for g in 0..n_groups {
            let gs = &row[g * group_size..(g + 1) * group_size];
            let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = mx - mn;
            let scale = if range.abs() < 1e-10 { 1.0 } else { range / 255.0 };
            scales[g] = scale;
            biases[g] = mn;
            for (i, &v) in gs.iter().enumerate() {
                let q = ((v - mn) / scale).round().clamp(0.0, 255.0) as u32;
                let d = g * group_size + i;
                packed[d / 4] |= q << ((d % 4) * 8);
            }
        }
        (packed, scales, biases)
    }

    /// Dequant-then-matmul oracle. `bits` is 4 or 8.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        norm_weight: &[f32],
        in_dim: usize,
        group_size: usize,
        out_dim: usize,
        bits: usize,
    ) -> Vec<f32> {
        let ssq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (ssq / in_dim as f32 + EPS).sqrt();
        let vals_per_word = 32 / bits; // 8 (int4) or 4 (int8)
        let mask = (1u32 << bits) - 1;
        let u32_per_row = in_dim / vals_per_word;
        let n_groups = in_dim / group_size;
        (0..out_dim)
            .map(|row| {
                let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
                let rs = &scales[row * n_groups..(row + 1) * n_groups];
                let rb = &biases[row * n_groups..(row + 1) * n_groups];
                let mut acc = 0.0_f32;
                for d in 0..in_dim {
                    let q = (rw[d / vals_per_word] >> ((d % vals_per_word) * bits)) & mask;
                    let g = d / group_size;
                    let w_real = q as f32 * rs[g] + rb[g];
                    acc += w_real * (x[d] * norm_weight[d] * inv_rms);
                }
                acc
            })
            .collect()
    }

    /// Quantize a full `[out_dim, in_dim]` weight matrix row-by-row.
    fn quantize_matrix(
        rows: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        int8: bool,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let mut w = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        for row in 0..out_dim {
            let r = &rows[row * in_dim..(row + 1) * in_dim];
            let (pw, ps, pb) = if int8 {
                quantize_int8_row(r, group_size)
            } else {
                quantize_int4_row(r, group_size)
            };
            w.extend(pw);
            s.extend(ps);
            b.extend(pb);
        }
        (w, s, b)
    }

    /// Assemble the shared buffer set + expected output for any variant.
    fn setup(
        kernel: metaltile::core::ir::Kernel,
        dt: DType,
        in_dim: usize,
        group_size: usize,
        out_dim: usize,
        int8: bool,
    ) -> TestSetup {
        let x: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1).iter().map(|&v| round(v, dt)).collect();
        let norm_weight: Vec<f32> =
            source(in_dim, 0xB2, 0.4, 1.0).iter().map(|&v| round(v, dt)).collect();
        let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);
        let (weight, scales, biases) = quantize_matrix(&w_rows, out_dim, in_dim, group_size, int8);
        let scales_r: Vec<f32> = scales.iter().map(|&v| round(v, dt)).collect();
        let biases_r: Vec<f32> = biases.iter().map(|&v| round(v, dt)).collect();
        let bits = if int8 { 8 } else { 4 };
        let expected = naive(
            &weight,
            &scales_r,
            &biases_r,
            &x,
            &norm_weight,
            in_dim,
            group_size,
            out_dim,
            bits,
        );

        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&norm_weight, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_u32(&weight), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases, dt), dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .input(TestBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
    }

    // ── Scalar variant: grid [out_dim, 1, 1], tpg 128 (one row per TG). ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 2e-1)]
    fn test_rms_norm_qgemv(dt: DType) -> TestSetup {
        let (in_dim, gs, out_dim) = (256usize, 64usize, 8usize);
        setup(ffai_rms_norm_qgemv::kernel_ir_for(dt), dt, in_dim, gs, out_dim, false).grid_3d(
            out_dim as u32,
            1,
            1,
            [128, 1, 1],
        )
    }

    // ── Fast int4 variant: grid [out_dim/8, 1, 1], tpg 64 (8 rows per TG). ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 2e-1)]
    fn test_rms_norm_qgemv_fast(dt: DType) -> TestSetup {
        let (in_dim, gs, out_dim) = (512usize, 64usize, 16usize);
        setup(ffai_rms_norm_qgemv_fast::kernel_ir_for(dt), dt, in_dim, gs, out_dim, false).grid_3d(
            (out_dim / 8) as u32,
            1,
            1,
            [64, 1, 1],
        )
    }

    // ── Fast int8 variant: grid [out_dim/8, 1, 1], tpg 64 (8 rows per TG). ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 2e-1)]
    fn test_rms_norm_qgemv_int8_fast(dt: DType) -> TestSetup {
        let (in_dim, gs, out_dim) = (512usize, 64usize, 16usize);
        setup(ffai_rms_norm_qgemv_int8_fast::kernel_ir_for(dt), dt, in_dim, gs, out_dim, true)
            .grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
    }
}

/// New-syntax benchmarks for the fused RMSNorm + quantized GEMV family —
/// MLX-less reduction kernels (`Ref(GB/s)` blank). Production decode shape:
/// Qwen3-class hidden = 4096, projection out = 4096.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_rms_norm_qgemv, ffai_rms_norm_qgemv_fast, ffai_rms_norm_qgemv_int8_fast};

    const EPS: f32 = 1e-5;

    /// Shared buffer set. `bits` is 4 or 8 (weight word count differs).
    fn buffers(
        s: BenchSetup,
        in_dim: usize,
        gs: usize,
        out_dim: usize,
        dt: DType,
        bits: usize,
    ) -> BenchSetup {
        let n_groups = in_dim / gs;
        let weight_words = out_dim * in_dim / (32 / bits);
        s.buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", in_dim, dt))
            .buffer(BenchBuffer::random("weight", weight_words, DType::U32))
            .buffer(BenchBuffer::random("scales", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases", out_dim * n_groups, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", gs as u32)
            .bytes_moved((weight_words * 4) as u64)
    }

    #[bench(name = "ffai/rms_norm_qgemv", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_qgemv(dt: DType) -> BenchSetup {
        let (in_dim, gs, out_dim) = (4096usize, 64usize, 4096usize);
        let s = BenchSetup::new(ffai_rms_norm_qgemv::kernel_ir_for(dt)).mode(KernelMode::Reduction);
        buffers(s, in_dim, gs, out_dim, dt, 4).grid_3d(out_dim as u32, 1, 1, [128, 1, 1])
    }

    #[bench(name = "ffai/rms_norm_qgemv_fast", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_qgemv_fast(dt: DType) -> BenchSetup {
        let (in_dim, gs, out_dim) = (4096usize, 64usize, 4096usize);
        let s = BenchSetup::new(ffai_rms_norm_qgemv_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction);
        buffers(s, in_dim, gs, out_dim, dt, 4).grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
    }

    #[bench(name = "ffai/rms_norm_qgemv_int8_fast", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_qgemv_int8_fast(dt: DType) -> BenchSetup {
        let (in_dim, gs, out_dim) = (4096usize, 64usize, 4096usize);
        let s = BenchSetup::new(ffai_rms_norm_qgemv_int8_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction);
        buffers(s, in_dim, gs, out_dim, dt, 8).grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
    }
}
