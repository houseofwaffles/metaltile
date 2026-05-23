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
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// `y[row] = Σ_i (q[row,i]·scale + bias) · (x[i]·norm_weight[i]·inv_rms)`,
/// with `inv_rms = rsqrt(mean(x²) + eps)`, weights int4-packed.
/// One output row per threadgroup (original correctness-first variant).
#[kernel]
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
#[kernel]
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

inventory::submit! {
    BenchSpec {
        op: "rms_norm_qgemv",
        subop: "rms_norm_qgemv",
        kernel_name: "ffai_rms_norm_qgemv",
        kernel_ir: ffai_rms_norm_qgemv::kernel_ir_for,
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
        op: "rms_norm_qgemv",
        subop: "rms_norm_qgemv_fast",
        kernel_name: "ffai_rms_norm_qgemv_fast",
        kernel_ir: ffai_rms_norm_qgemv_fast::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
