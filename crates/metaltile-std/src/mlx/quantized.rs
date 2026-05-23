//! Quantized MatVec benchmark — #[kernel] DSL vs MLX metal/quantized.metal

use metaltile::{bench_kernel, kernel};
// (out_dim, in_dim) pairs. 4096² = baseline reference. Other rows are
// production hot-paths in Qwen3-class inference:
//   - 5120²       Qwen3-8B/14B attention proj (Q/K/V/O), MLP.gate/up at hidden
//   - 14336×5120  Qwen3-8B/14B MLP up_proj
//   - 5120×14336  Qwen3-8B/14B MLP down_proj
//   - 27648×5120  Qwen3-coder-30B MoE expert up_proj
static QUANTIZED_SHAPES: &[(usize, usize)] =
    &[(4096, 4096), (5120, 5120), (14336, 5120), (5120, 14336), (27648, 5120)];

#[bench_kernel(
    op="quantized",
    subop="qmv",
    class=QuantizedMatVec,
    shapes=&QUANTIZED_SHAPES,
    group_size=64,
    // tpg=64 = 2 simdgroups × 32 lanes. Kernel processes 8 output rows
    // per TG (each simdgroup handles 4 rows independently, indexed by
    // simd_id). Dispatcher grid is `m/8` TGs — matches MLX qmv_fast.
    tpg=64,
    tol=1e-3,
    mlx="affine_qmv_fast_float16_t_gs_64_b_4_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmv<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    // Multi-row tile: 8 output rows per TG, 2 simdgroups × 32 lanes.
    // Each simdgroup independently handles 4 rows (indexed by simd_id).
    // Each lane caches 16 X values in registers per outer block and
    // reuses them across all 4 rows' qdot accumulation — 4× reduction
    // in X bandwidth + 8× fewer TGs vs the previous 1-row-per-TG layout.
    // Matches MLX qmv_fast geometry (`quantized.h:749`) exactly.
    //
    // Per outer iter: 16 X loads (once per simdgroup) + per-row (2
    // weight packs + 16 int4 extracts + 16 FMAs into q_dot + 1 add into
    // x_sum + 1 scale + 1 bias + 1 partial accumulation). Block = 16 X
    // × 32 lanes = 512 K elements.
    //
    // Math: result_row = sum_g (scale_g * sum_{i in g} q_i*x_i
    //                          + bias_g * sum_{i in g} x_i)
    // The bias hoist (algebraic split) eliminates one FMA per int4 in
    // the hot loop — matches MLX `qdot` in quantized.h:235.
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let packs_per_row = k / 8u32;
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

    for _b in range(0u32, k, 512u32) {
        // 16 X loads — consecutive in IR for vectorize fusion (4× float4).
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
        // Mask-without-shift: X pre-scaled by inverse nibble position, weight
        // mask returns nibble × position-power. Saves 7 shifts per pack × 2
        // packs × 4 rows = 56 shifts per outer iter. Mirrors MLX `qdot` for
        // bits=4 (`quantized.h:235-244`).
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
        // Incremental xs accumulator from raw loads — saves 12 muls vs the
        // reconstruction-from-scaled approach. Raw x dies right after the
        // scale + xs accumulator both consume it.
        // Cast T-typed X to f32 at load time for the inner FMA chain;
        // accumulators stay in f32 regardless of T. Identity for T=f32.
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

        // Each lane covers 16 X values within a single gs=64 group.
        // 4 lanes per group, 8 groups per block (32 lanes × 16 = 512).
        let g = xb / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        // Lo half (nibbles 0-3): mask 0xf, 0xf0, 0xf00, 0xf000 — values *1, *16, *256, *4096.
        // Multiplied against pre-scaled x[0..3] (*1, *1/16, *1/256, *1/4096) → q*x.
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

        // ── Row 1 ──
        let p10 = load(w[w_base1 + pack_off]);
        let p11 = load(w[w_base1 + pack_off + 1u32]);
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

        // ── Row 2 ──
        let p20 = load(w[w_base2 + pack_off]);
        let p21 = load(w[w_base2 + pack_off + 1u32]);
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

        // ── Row 3 ──
        let p30 = load(w[w_base3 + pack_off]);
        let p31 = load(w[w_base3 + pack_off + 1u32]);
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

    // Cross-lane reduction: each row's partial → single value, lane 0 stores.
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

// ─── mt_qmm ─────────────────────────────────────────────────────────────
//
// Quantized matmul (B>1 / prefill). Same int4 weight layout as `mt_qmv`
// extended along the M axis (token count). Each threadgroup owns 8
// consecutive output columns at one M-row — `mt_qmv`'s 2 SG × 4 N-row
// tile lifted into M via an outer grid axis (`tgid_y = m_row`). The
// inner K-walk is bit-identical to `mt_qmv`: each lane caches 16 X
// values per 512-wide K-block and reuses them across all 4 N-rows in
// its simdgroup, using the same mask-without-shift trick (X
// pre-scaled by inverse nibble position, weight mask returns
// nibble × position-power) + algebraic-split accumulator
// `acc += s_g · Σ q·x + bias_g · Σ x` that mirrors MLX `qdot` in
// `quantized.h:235-244`.
//
// Geometry:
//   tpg = 64 = 2 simdgroups × 32 lanes
//   8 outputs per TG (each SG owns 4 N-rows, indexed by simd_id)
//   Block = 16 X × 32 lanes = 512 K elements per outer iter
//   Grid: [n / 8, m, 1]
//
// Layouts:
//   w       [n, k/8]               u32   — int4 nibbles (8 per uint32)
//   scales  [n, gs_per_row]        T
//   biases  [n, gs_per_row]        T
//   x       [m, k]                 T
//   out     [m, n]                 T
//
// At M = 1 this is byte-identical to `mt_qmv`. At M > 1 each M-row
// runs as a fully independent threadgroup grid axis — no W reuse
// across M-rows (W is loaded fresh per (M-row, N-tile) pair). The
// natural v3 step is a BM × BN output tile with W cached in TG
// memory and amortised across BM M-rows.
#[bench_kernel(
    op="quantized",
    subop="qmm",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=4 = canonical small-batch prefill token count (covers
    // single-prompt prefill chunks + small batched serving). Larger
    // M values exposed via the #[ignore] `mt_qmm_perf_bench_*` test.
    m=4,
    group_size=64,
    // tpg=64 same as mt_qmv (2 SG × 32 lanes). Each TG produces 8
    // outputs at one (m_row, n_tile).
    tpg=64,
    // bf16 round-trip on int4-quantized matmul: max_q=15 × group_size=64
    // × bf16's 7-bit mantissa drifts ~7-8e-3 at large K (per
    // crates/metaltile-std/src/mlx/binary.rs precedent — "bf16 drifts
    // ~7.8e-3 on signed"). Tighter than 1e-2 trips the bench cosine
    // check at production shapes (M=4096+, K=4096+) on Apple Paravirtual
    // CI. tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_row = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let packs_per_row = k / 8u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    let x_row_base = m_row * k;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    for _b in range(0u32, k, 512u32) {
        // 16 X loads — consecutive in IR for vectorize fusion.
        let xb = x_row_base + _b + lane_x_off;
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
        // Mask-without-shift constants. Same as mt_qmv.
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
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

        // Group index within this row's K dimension. mt_qmv uses
        // `xb / 64` because there `xb` is already a K-position; here
        // `xb` includes the `x_row_base = m_row * k` offset, so we
        // recompute against the K-local base.
        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
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

        // ── Row 1 ──
        let p10 = load(w[w_base1 + pack_off]);
        let p11 = load(w[w_base1 + pack_off + 1u32]);
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

        // ── Row 2 ──
        let p20 = load(w[w_base2 + pack_off]);
        let p21 = load(w[w_base2 + pack_off + 1u32]);
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

        // ── Row 3 ──
        let p30 = load(w[w_base3 + pack_off]);
        let p31 = load(w[w_base3 + pack_off + 1u32]);
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
        store(out[m_row * n + row0], r0.cast::<T>());
        store(out[m_row * n + row1], r1.cast::<T>());
        store(out[m_row * n + row2], r2.cast::<T>());
        store(out[m_row * n + row3], r3.cast::<T>());
    }
}

// ─── mt_qmm_bm2 ─────────────────────────────────────────────────────────
//
// Quantized matmul v3 — BM × BN output tile with TG-memory-free W reuse.
//
// Same int4 weight layout + 8-output 2 SG × 4 N-row geometry as
// `mt_qmm`, but lifts BM=2 M-rows into the same threadgroup so the W
// packs + nibble extractions are loaded ONCE per K-block per N-row and
// reused across both M-rows. Per K-block per TG: 8 W loads (unchanged
// from v2) producing 16 outputs (vs 8). W bandwidth per output halves.
//
// Geometry:
//   tpg = 64 = 2 SG × 32 lanes
//   BM = 2 (M-rows per TG)
//   BN = 8 (N-rows per TG, each SG owns 4)
//   16 outputs per TG (BM × BN)
//   Grid: [n / 8, m / 2, 1]
//
// Register footprint per lane (f32):
//   32 X values (16 per M-row × 2 M-rows) = 128 bytes
//   8 accumulators (4 N-rows × 2 M-rows)   =  32 bytes
//   16 W nibble extracts (shared)          =  64 bytes
//   ≈ 240 bytes — well inside Apple GPU's ~1024 byte/lane register file.
//
// At M < 2 the caller should dispatch `mt_qmm` (BM=1) instead — this
// kernel asserts `m % 2 == 0` via the grid dim. v4 BM=4 is the next
// step if M=32 still doesn't beat MLX after this lands; see #55.
#[bench_kernel(
    op="quantized",
    subop="qmm_bm2",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=8 = larger-batch prefill where W-reuse matters most. M=2 / 4
    // also benefit (W reload halved); M=1 should keep dispatching
    // mt_qmm (v2) since the BM=2 tile would burn TG slots on unused
    // outputs.
    // M=8 is a representative mid-M cell. Clean median-of-5 head-to-head
    // bm2/v2 (25 cells per M, both rigs): bm2 wins 350/350 across
    // M ∈ {2,4,6,8,12,16,32}. Speedups grow with M: 1.09× at M=2
    // → 1.24× M5 / 1.30× M2 at M=32. vs MLX `affine_qmm_t`, the M=8
    // bench cell measures 1.7-2.5× M5 / 1.4-1.7× f16 M2 (3-run M5
    // drift ≤3pt). Selector `mt_qmm_for` routes every even M ≥ 2
    // to bm2. Neither kernel beats MLX at M ≥ 16 (MLX's BM=BN=32
    // simdgroup-matrix tile dominates large-M); closing that gap is
    // the BM=4/BM=8 follow-up.
    m=8,
    group_size=64,
    tpg=64,
    // bf16 round-trip on int4-quantized matmul: max_q=15 × group_size=64
    // × bf16's 7-bit mantissa drifts ~7-8e-3 at large K (per
    // crates/metaltile-std/src/mlx/binary.rs precedent — "bf16 drifts
    // ~7.8e-3 on signed"). Tighter than 1e-2 trips the bench cosine
    // check at production shapes (M=4096+, K=4096+) on Apple Paravirtual
    // CI. tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_bm2<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_tile = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let packs_per_row = k / 8u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    // BM=2 M-rows per TG.
    let m_row_a = m_tile * 2u32;
    let m_row_b = m_row_a + 1u32;
    let x_base_a = m_row_a * k;
    let x_base_b = m_row_b * k;

    // 8 accumulators: 4 N-rows × 2 M-rows.
    let mut acc0_a = 0.0f32;
    let mut acc0_b = 0.0f32;
    let mut acc1_a = 0.0f32;
    let mut acc1_b = 0.0f32;
    let mut acc2_a = 0.0f32;
    let mut acc2_b = 0.0f32;
    let mut acc3_a = 0.0f32;
    let mut acc3_b = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    for _b in range(0u32, k, 512u32) {
        // ── Load 16 X values for M-row A ──
        let xb_a = x_base_a + _b + lane_x_off;
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
        let x0_a = load(x[xb_a]).cast::<f32>();
        let x1_a_raw = load(x[xb_a + 1u32]).cast::<f32>();
        let x2_a_raw = load(x[xb_a + 2u32]).cast::<f32>();
        let x3_a_raw = load(x[xb_a + 3u32]).cast::<f32>();
        let x4_a = load(x[xb_a + 4u32]).cast::<f32>();
        let x5_a_raw = load(x[xb_a + 5u32]).cast::<f32>();
        let x6_a_raw = load(x[xb_a + 6u32]).cast::<f32>();
        let x7_a_raw = load(x[xb_a + 7u32]).cast::<f32>();
        let x8_a = load(x[xb_a + 8u32]).cast::<f32>();
        let x9_a_raw = load(x[xb_a + 9u32]).cast::<f32>();
        let x10_a_raw = load(x[xb_a + 10u32]).cast::<f32>();
        let x11_a_raw = load(x[xb_a + 11u32]).cast::<f32>();
        let x12_a = load(x[xb_a + 12u32]).cast::<f32>();
        let x13_a_raw = load(x[xb_a + 13u32]).cast::<f32>();
        let x14_a_raw = load(x[xb_a + 14u32]).cast::<f32>();
        let x15_a_raw = load(x[xb_a + 15u32]).cast::<f32>();
        let xs_a = x0_a
            + x1_a_raw
            + x2_a_raw
            + x3_a_raw
            + x4_a
            + x5_a_raw
            + x6_a_raw
            + x7_a_raw
            + x8_a
            + x9_a_raw
            + x10_a_raw
            + x11_a_raw
            + x12_a
            + x13_a_raw
            + x14_a_raw
            + x15_a_raw;
        let x1_a = x1_a_raw * s_16;
        let x2_a = x2_a_raw * s_256;
        let x3_a = x3_a_raw * s_4096;
        let x5_a = x5_a_raw * s_16;
        let x6_a = x6_a_raw * s_256;
        let x7_a = x7_a_raw * s_4096;
        let x9_a = x9_a_raw * s_16;
        let x10_a = x10_a_raw * s_256;
        let x11_a = x11_a_raw * s_4096;
        let x13_a = x13_a_raw * s_16;
        let x14_a = x14_a_raw * s_256;
        let x15_a = x15_a_raw * s_4096;

        // ── Load 16 X values for M-row B ──
        let xb_b = x_base_b + _b + lane_x_off;
        let x0_b = load(x[xb_b]).cast::<f32>();
        let x1_b_raw = load(x[xb_b + 1u32]).cast::<f32>();
        let x2_b_raw = load(x[xb_b + 2u32]).cast::<f32>();
        let x3_b_raw = load(x[xb_b + 3u32]).cast::<f32>();
        let x4_b = load(x[xb_b + 4u32]).cast::<f32>();
        let x5_b_raw = load(x[xb_b + 5u32]).cast::<f32>();
        let x6_b_raw = load(x[xb_b + 6u32]).cast::<f32>();
        let x7_b_raw = load(x[xb_b + 7u32]).cast::<f32>();
        let x8_b = load(x[xb_b + 8u32]).cast::<f32>();
        let x9_b_raw = load(x[xb_b + 9u32]).cast::<f32>();
        let x10_b_raw = load(x[xb_b + 10u32]).cast::<f32>();
        let x11_b_raw = load(x[xb_b + 11u32]).cast::<f32>();
        let x12_b = load(x[xb_b + 12u32]).cast::<f32>();
        let x13_b_raw = load(x[xb_b + 13u32]).cast::<f32>();
        let x14_b_raw = load(x[xb_b + 14u32]).cast::<f32>();
        let x15_b_raw = load(x[xb_b + 15u32]).cast::<f32>();
        let xs_b = x0_b
            + x1_b_raw
            + x2_b_raw
            + x3_b_raw
            + x4_b
            + x5_b_raw
            + x6_b_raw
            + x7_b_raw
            + x8_b
            + x9_b_raw
            + x10_b_raw
            + x11_b_raw
            + x12_b
            + x13_b_raw
            + x14_b_raw
            + x15_b_raw;
        let x1_b = x1_b_raw * s_16;
        let x2_b = x2_b_raw * s_256;
        let x3_b = x3_b_raw * s_4096;
        let x5_b = x5_b_raw * s_16;
        let x6_b = x6_b_raw * s_256;
        let x7_b = x7_b_raw * s_4096;
        let x9_b = x9_b_raw * s_16;
        let x10_b = x10_b_raw * s_256;
        let x11_b = x11_b_raw * s_4096;
        let x13_b = x13_b_raw * s_16;
        let x14_b = x14_b_raw * s_256;
        let x15_b = x15_b_raw * s_4096;

        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 (shared W extracts, dual qdots) ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
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
        let qd0_a = q00 * x0_a
            + q01 * x1_a
            + q02 * x2_a
            + q03 * x3_a
            + q04 * x4_a
            + q05 * x5_a
            + q06 * x6_a
            + q07 * x7_a
            + q08 * x8_a
            + q09 * x9_a
            + q010 * x10_a
            + q011 * x11_a
            + q012 * x12_a
            + q013 * x13_a
            + q014 * x14_a
            + q015 * x15_a;
        let qd0_b = q00 * x0_b
            + q01 * x1_b
            + q02 * x2_b
            + q03 * x3_b
            + q04 * x4_b
            + q05 * x5_b
            + q06 * x6_b
            + q07 * x7_b
            + q08 * x8_b
            + q09 * x9_b
            + q010 * x10_b
            + q011 * x11_b
            + q012 * x12_b
            + q013 * x13_b
            + q014 * x14_b
            + q015 * x15_b;
        acc0_a = acc0_a + s0 * qd0_a + bi0 * xs_a;
        acc0_b = acc0_b + s0 * qd0_b + bi0 * xs_b;

        // ── Row 1 ──
        let p10 = load(w[w_base1 + pack_off]);
        let p11 = load(w[w_base1 + pack_off + 1u32]);
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
        let qd1_a = q10 * x0_a
            + q11 * x1_a
            + q12 * x2_a
            + q13 * x3_a
            + q14 * x4_a
            + q15 * x5_a
            + q16 * x6_a
            + q17 * x7_a
            + q18 * x8_a
            + q19 * x9_a
            + q110 * x10_a
            + q111 * x11_a
            + q112 * x12_a
            + q113 * x13_a
            + q114 * x14_a
            + q115 * x15_a;
        let qd1_b = q10 * x0_b
            + q11 * x1_b
            + q12 * x2_b
            + q13 * x3_b
            + q14 * x4_b
            + q15 * x5_b
            + q16 * x6_b
            + q17 * x7_b
            + q18 * x8_b
            + q19 * x9_b
            + q110 * x10_b
            + q111 * x11_b
            + q112 * x12_b
            + q113 * x13_b
            + q114 * x14_b
            + q115 * x15_b;
        acc1_a = acc1_a + s1 * qd1_a + bi1 * xs_a;
        acc1_b = acc1_b + s1 * qd1_b + bi1 * xs_b;

        // ── Row 2 ──
        let p20 = load(w[w_base2 + pack_off]);
        let p21 = load(w[w_base2 + pack_off + 1u32]);
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
        let qd2_a = q20 * x0_a
            + q21 * x1_a
            + q22 * x2_a
            + q23 * x3_a
            + q24 * x4_a
            + q25 * x5_a
            + q26 * x6_a
            + q27 * x7_a
            + q28 * x8_a
            + q29 * x9_a
            + q210 * x10_a
            + q211 * x11_a
            + q212 * x12_a
            + q213 * x13_a
            + q214 * x14_a
            + q215 * x15_a;
        let qd2_b = q20 * x0_b
            + q21 * x1_b
            + q22 * x2_b
            + q23 * x3_b
            + q24 * x4_b
            + q25 * x5_b
            + q26 * x6_b
            + q27 * x7_b
            + q28 * x8_b
            + q29 * x9_b
            + q210 * x10_b
            + q211 * x11_b
            + q212 * x12_b
            + q213 * x13_b
            + q214 * x14_b
            + q215 * x15_b;
        acc2_a = acc2_a + s2 * qd2_a + bi2 * xs_a;
        acc2_b = acc2_b + s2 * qd2_b + bi2 * xs_b;

        // ── Row 3 ──
        let p30 = load(w[w_base3 + pack_off]);
        let p31 = load(w[w_base3 + pack_off + 1u32]);
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
        let qd3_a = q30 * x0_a
            + q31 * x1_a
            + q32 * x2_a
            + q33 * x3_a
            + q34 * x4_a
            + q35 * x5_a
            + q36 * x6_a
            + q37 * x7_a
            + q38 * x8_a
            + q39 * x9_a
            + q310 * x10_a
            + q311 * x11_a
            + q312 * x12_a
            + q313 * x13_a
            + q314 * x14_a
            + q315 * x15_a;
        let qd3_b = q30 * x0_b
            + q31 * x1_b
            + q32 * x2_b
            + q33 * x3_b
            + q34 * x4_b
            + q35 * x5_b
            + q36 * x6_b
            + q37 * x7_b
            + q38 * x8_b
            + q39 * x9_b
            + q310 * x10_b
            + q311 * x11_b
            + q312 * x12_b
            + q313 * x13_b
            + q314 * x14_b
            + q315 * x15_b;
        acc3_a = acc3_a + s3 * qd3_a + bi3 * xs_a;
        acc3_b = acc3_b + s3 * qd3_b + bi3 * xs_b;
    }

    // Cross-lane reduce + lane-0 stores. 8 outputs per TG.
    let r0_a = simd_sum(acc0_a);
    let r0_b = simd_sum(acc0_b);
    let r1_a = simd_sum(acc1_a);
    let r1_b = simd_sum(acc1_b);
    let r2_a = simd_sum(acc2_a);
    let r2_b = simd_sum(acc2_b);
    let r3_a = simd_sum(acc3_a);
    let r3_b = simd_sum(acc3_b);
    if lane == 0u32 {
        store(out[m_row_a * n + row0], r0_a.cast::<T>());
        store(out[m_row_a * n + row1], r1_a.cast::<T>());
        store(out[m_row_a * n + row2], r2_a.cast::<T>());
        store(out[m_row_a * n + row3], r3_a.cast::<T>());
        store(out[m_row_b * n + row0], r0_b.cast::<T>());
        store(out[m_row_b * n + row1], r1_b.cast::<T>());
        store(out[m_row_b * n + row2], r2_b.cast::<T>());
        store(out[m_row_b * n + row3], r3_b.cast::<T>());
    }
}

// ─── mt_qmm_bm4 ─────────────────────────────────────────────────────────
//
// Quantized matmul v4 — BM × BN output tile with 4× W reuse.
//
// Same int4 layout + 8-output 2 SG × 4 N-row geometry as mt_qmm and
// mt_qmm_bm2, but lifts BM=4 M-rows into the same threadgroup so the
// W packs + nibble extractions are loaded ONCE per K-block per N-row
// and reused across all four M-rows. Per K-block per TG: 8 W loads
// (unchanged from v2/bm2) producing 32 outputs (vs bm2's 16, v2's 8).
// W bandwidth per output drops to 1/4 of v2.
//
// Geometry:
//   tpg = 64 = 2 SG × 32 lanes
//   BM = 4 (M-rows per TG)
//   BN = 8 (N-rows per TG, each SG owns 4)
//   32 outputs per TG (BM × BN)
//   Grid: [n / 8, m / 4, 1]
//
// Register footprint per lane (f32):
//   64 X values (16 per M-row × 4 M-rows)  = 256 bytes
//   16 accumulators (4 N-rows × 4 M-rows)  =  64 bytes
//   16 W nibble extracts (shared)          =  64 bytes
//   ≈ 440 bytes — fits Apple GPU's ~1024 byte/lane register file. Occupancy
//   may halve from the bm2 baseline (~12 SGs/SM → ~6 SGs/SM); net win
//   depends on whether the 4× W bw reduction outpaces the occupancy loss.
//
// Closes the M ≥ 16 gap to MLX: bm2 W-reads at M=32 = 16W per TG; bm4
// W-reads at M=32 = 8W per TG, matching MLX's 8W cache footprint.
// Predicted: ~half of MLX's M=16-32 gap recovered without simdgroup-
// matrix primitives.
#[bench_kernel(
    op="quantized",
    subop="qmm_bm4",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=16 is the cell where bm4's W-bw advantage compounds — bm2 at
    // M=16 hits ~50% MT MLX; bm4 halves W bw so a 1.5-1.8× speedup
    // over bm2 is plausible. Selector routes m % 4 == 0 to bm4.
    m=8,
    group_size=64,
    tpg=64,
    // bf16 round-trip on int4-quantized matmul: max_q=15 × group_size=64
    // × bf16's 7-bit mantissa drifts ~7-8e-3 at large K (per
    // crates/metaltile-std/src/mlx/binary.rs precedent — "bf16 drifts
    // ~7.8e-3 on signed"). Tighter than 1e-2 trips the bench cosine
    // check at production shapes (M=4096+, K=4096+) on Apple Paravirtual
    // CI. tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_bm4<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_tile = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let packs_per_row = k / 8u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    // BM=4 M-rows per TG.
    let m_row_a = m_tile * 4u32;
    let m_row_b = m_row_a + 1u32;
    let m_row_c = m_row_a + 2u32;
    let m_row_d = m_row_a + 3u32;
    let x_base_a = m_row_a * k;
    let x_base_b = m_row_b * k;
    let x_base_c = m_row_c * k;
    let x_base_d = m_row_d * k;

    // 16 accumulators: 4 N-rows × 4 M-rows.
    let mut acc0_a = 0.0f32;
    let mut acc0_b = 0.0f32;
    let mut acc0_c = 0.0f32;
    let mut acc0_d = 0.0f32;
    let mut acc1_a = 0.0f32;
    let mut acc1_b = 0.0f32;
    let mut acc1_c = 0.0f32;
    let mut acc1_d = 0.0f32;
    let mut acc2_a = 0.0f32;
    let mut acc2_b = 0.0f32;
    let mut acc2_c = 0.0f32;
    let mut acc2_d = 0.0f32;
    let mut acc3_a = 0.0f32;
    let mut acc3_b = 0.0f32;
    let mut acc3_c = 0.0f32;
    let mut acc3_d = 0.0f32;

    // VARIANT A (K=256): halve the K-block stride to reduce per-iter live X.
    // lane_x_off = lane * 8 (was 16), lane_pack_off = lane (was lane*2).
    // Each lane covers 8 X / 1 W pack per iter; 32 lanes = 256 K per block.
    let lane_x_off = lane * 8u32;
    let lane_pack_off = lane;

    for _b in range(0u32, k, 256u32) {
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;

        // ── Load 8 X values for M-row A ──
        let xb_a = x_base_a + _b + lane_x_off;
        let x0_a = load(x[xb_a]).cast::<f32>();
        let x1_a_raw = load(x[xb_a + 1u32]).cast::<f32>();
        let x2_a_raw = load(x[xb_a + 2u32]).cast::<f32>();
        let x3_a_raw = load(x[xb_a + 3u32]).cast::<f32>();
        let x4_a = load(x[xb_a + 4u32]).cast::<f32>();
        let x5_a_raw = load(x[xb_a + 5u32]).cast::<f32>();
        let x6_a_raw = load(x[xb_a + 6u32]).cast::<f32>();
        let x7_a_raw = load(x[xb_a + 7u32]).cast::<f32>();
        let xs_a = x0_a + x1_a_raw + x2_a_raw + x3_a_raw + x4_a + x5_a_raw + x6_a_raw + x7_a_raw;
        let x1_a = x1_a_raw * s_16;
        let x2_a = x2_a_raw * s_256;
        let x3_a = x3_a_raw * s_4096;
        let x5_a = x5_a_raw * s_16;
        let x6_a = x6_a_raw * s_256;
        let x7_a = x7_a_raw * s_4096;

        // ── Load 8 X values for M-row B ──
        let xb_b = x_base_b + _b + lane_x_off;
        let x0_b = load(x[xb_b]).cast::<f32>();
        let x1_b_raw = load(x[xb_b + 1u32]).cast::<f32>();
        let x2_b_raw = load(x[xb_b + 2u32]).cast::<f32>();
        let x3_b_raw = load(x[xb_b + 3u32]).cast::<f32>();
        let x4_b = load(x[xb_b + 4u32]).cast::<f32>();
        let x5_b_raw = load(x[xb_b + 5u32]).cast::<f32>();
        let x6_b_raw = load(x[xb_b + 6u32]).cast::<f32>();
        let x7_b_raw = load(x[xb_b + 7u32]).cast::<f32>();
        let xs_b = x0_b + x1_b_raw + x2_b_raw + x3_b_raw + x4_b + x5_b_raw + x6_b_raw + x7_b_raw;
        let x1_b = x1_b_raw * s_16;
        let x2_b = x2_b_raw * s_256;
        let x3_b = x3_b_raw * s_4096;
        let x5_b = x5_b_raw * s_16;
        let x6_b = x6_b_raw * s_256;
        let x7_b = x7_b_raw * s_4096;

        // ── Load 8 X values for M-row C ──
        let xb_c = x_base_c + _b + lane_x_off;
        let x0_c = load(x[xb_c]).cast::<f32>();
        let x1_c_raw = load(x[xb_c + 1u32]).cast::<f32>();
        let x2_c_raw = load(x[xb_c + 2u32]).cast::<f32>();
        let x3_c_raw = load(x[xb_c + 3u32]).cast::<f32>();
        let x4_c = load(x[xb_c + 4u32]).cast::<f32>();
        let x5_c_raw = load(x[xb_c + 5u32]).cast::<f32>();
        let x6_c_raw = load(x[xb_c + 6u32]).cast::<f32>();
        let x7_c_raw = load(x[xb_c + 7u32]).cast::<f32>();
        let xs_c = x0_c + x1_c_raw + x2_c_raw + x3_c_raw + x4_c + x5_c_raw + x6_c_raw + x7_c_raw;
        let x1_c = x1_c_raw * s_16;
        let x2_c = x2_c_raw * s_256;
        let x3_c = x3_c_raw * s_4096;
        let x5_c = x5_c_raw * s_16;
        let x6_c = x6_c_raw * s_256;
        let x7_c = x7_c_raw * s_4096;

        // ── Load 8 X values for M-row D ──
        let xb_d = x_base_d + _b + lane_x_off;
        let x0_d = load(x[xb_d]).cast::<f32>();
        let x1_d_raw = load(x[xb_d + 1u32]).cast::<f32>();
        let x2_d_raw = load(x[xb_d + 2u32]).cast::<f32>();
        let x3_d_raw = load(x[xb_d + 3u32]).cast::<f32>();
        let x4_d = load(x[xb_d + 4u32]).cast::<f32>();
        let x5_d_raw = load(x[xb_d + 5u32]).cast::<f32>();
        let x6_d_raw = load(x[xb_d + 6u32]).cast::<f32>();
        let x7_d_raw = load(x[xb_d + 7u32]).cast::<f32>();
        let xs_d = x0_d + x1_d_raw + x2_d_raw + x3_d_raw + x4_d + x5_d_raw + x6_d_raw + x7_d_raw;
        let x1_d = x1_d_raw * s_16;
        let x2_d = x2_d_raw * s_256;
        let x3_d = x3_d_raw * s_4096;
        let x5_d = x5_d_raw * s_16;
        let x6_d = x6_d_raw * s_256;
        let x7_d = x7_d_raw * s_4096;

        // VARIANT A: gs_per_row=64 means K-elements/group=64. With lane_x_off=lane*8,
        // each lane's 8 X-values span exactly half a group (8 < 64), so the group
        // index is constant within a lane's 8-elt slice. _b advances by 256 per
        // outer iter; (_b + lane*8) / 64 selects the right group.
        let g = (_b + lane_x_off) / 64u32;
        // pack_off: 1 pack (8 nibbles) per lane per K-iter. _b/8 packs at base.
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 (W extracts shared across all 4 M-rows) — 8 nibbles ──
        let p00 = load(w[w_base0 + pack_off]);
        let p00_hi = p00 >> 16u32;
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
        let qd0_a = q00 * x0_a
            + q01 * x1_a
            + q02 * x2_a
            + q03 * x3_a
            + q04 * x4_a
            + q05 * x5_a
            + q06 * x6_a
            + q07 * x7_a;
        let qd0_b = q00 * x0_b
            + q01 * x1_b
            + q02 * x2_b
            + q03 * x3_b
            + q04 * x4_b
            + q05 * x5_b
            + q06 * x6_b
            + q07 * x7_b;
        let qd0_c = q00 * x0_c
            + q01 * x1_c
            + q02 * x2_c
            + q03 * x3_c
            + q04 * x4_c
            + q05 * x5_c
            + q06 * x6_c
            + q07 * x7_c;
        let qd0_d = q00 * x0_d
            + q01 * x1_d
            + q02 * x2_d
            + q03 * x3_d
            + q04 * x4_d
            + q05 * x5_d
            + q06 * x6_d
            + q07 * x7_d;
        acc0_a = acc0_a + s0 * qd0_a + bi0 * xs_a;
        acc0_b = acc0_b + s0 * qd0_b + bi0 * xs_b;
        acc0_c = acc0_c + s0 * qd0_c + bi0 * xs_c;
        acc0_d = acc0_d + s0 * qd0_d + bi0 * xs_d;

        // ── Row 1 ──
        let p10 = load(w[w_base1 + pack_off]);
        let p10_hi = p10 >> 16u32;
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
        let qd1_a = q10 * x0_a
            + q11 * x1_a
            + q12 * x2_a
            + q13 * x3_a
            + q14 * x4_a
            + q15 * x5_a
            + q16 * x6_a
            + q17 * x7_a;
        let qd1_b = q10 * x0_b
            + q11 * x1_b
            + q12 * x2_b
            + q13 * x3_b
            + q14 * x4_b
            + q15 * x5_b
            + q16 * x6_b
            + q17 * x7_b;
        let qd1_c = q10 * x0_c
            + q11 * x1_c
            + q12 * x2_c
            + q13 * x3_c
            + q14 * x4_c
            + q15 * x5_c
            + q16 * x6_c
            + q17 * x7_c;
        let qd1_d = q10 * x0_d
            + q11 * x1_d
            + q12 * x2_d
            + q13 * x3_d
            + q14 * x4_d
            + q15 * x5_d
            + q16 * x6_d
            + q17 * x7_d;
        acc1_a = acc1_a + s1 * qd1_a + bi1 * xs_a;
        acc1_b = acc1_b + s1 * qd1_b + bi1 * xs_b;
        acc1_c = acc1_c + s1 * qd1_c + bi1 * xs_c;
        acc1_d = acc1_d + s1 * qd1_d + bi1 * xs_d;

        // ── Row 2 ──
        let p20 = load(w[w_base2 + pack_off]);
        let p20_hi = p20 >> 16u32;
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
        let qd2_a = q20 * x0_a
            + q21 * x1_a
            + q22 * x2_a
            + q23 * x3_a
            + q24 * x4_a
            + q25 * x5_a
            + q26 * x6_a
            + q27 * x7_a;
        let qd2_b = q20 * x0_b
            + q21 * x1_b
            + q22 * x2_b
            + q23 * x3_b
            + q24 * x4_b
            + q25 * x5_b
            + q26 * x6_b
            + q27 * x7_b;
        let qd2_c = q20 * x0_c
            + q21 * x1_c
            + q22 * x2_c
            + q23 * x3_c
            + q24 * x4_c
            + q25 * x5_c
            + q26 * x6_c
            + q27 * x7_c;
        let qd2_d = q20 * x0_d
            + q21 * x1_d
            + q22 * x2_d
            + q23 * x3_d
            + q24 * x4_d
            + q25 * x5_d
            + q26 * x6_d
            + q27 * x7_d;
        acc2_a = acc2_a + s2 * qd2_a + bi2 * xs_a;
        acc2_b = acc2_b + s2 * qd2_b + bi2 * xs_b;
        acc2_c = acc2_c + s2 * qd2_c + bi2 * xs_c;
        acc2_d = acc2_d + s2 * qd2_d + bi2 * xs_d;

        // ── Row 3 ──
        let p30 = load(w[w_base3 + pack_off]);
        let p30_hi = p30 >> 16u32;
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
        let qd3_a = q30 * x0_a
            + q31 * x1_a
            + q32 * x2_a
            + q33 * x3_a
            + q34 * x4_a
            + q35 * x5_a
            + q36 * x6_a
            + q37 * x7_a;
        let qd3_b = q30 * x0_b
            + q31 * x1_b
            + q32 * x2_b
            + q33 * x3_b
            + q34 * x4_b
            + q35 * x5_b
            + q36 * x6_b
            + q37 * x7_b;
        let qd3_c = q30 * x0_c
            + q31 * x1_c
            + q32 * x2_c
            + q33 * x3_c
            + q34 * x4_c
            + q35 * x5_c
            + q36 * x6_c
            + q37 * x7_c;
        let qd3_d = q30 * x0_d
            + q31 * x1_d
            + q32 * x2_d
            + q33 * x3_d
            + q34 * x4_d
            + q35 * x5_d
            + q36 * x6_d
            + q37 * x7_d;
        acc3_a = acc3_a + s3 * qd3_a + bi3 * xs_a;
        acc3_b = acc3_b + s3 * qd3_b + bi3 * xs_b;
        acc3_c = acc3_c + s3 * qd3_c + bi3 * xs_c;
        acc3_d = acc3_d + s3 * qd3_d + bi3 * xs_d;
    }

    // Cross-lane reduce + lane-0 stores. 16 outputs per TG.
    let r0_a = simd_sum(acc0_a);
    let r0_b = simd_sum(acc0_b);
    let r0_c = simd_sum(acc0_c);
    let r0_d = simd_sum(acc0_d);
    let r1_a = simd_sum(acc1_a);
    let r1_b = simd_sum(acc1_b);
    let r1_c = simd_sum(acc1_c);
    let r1_d = simd_sum(acc1_d);
    let r2_a = simd_sum(acc2_a);
    let r2_b = simd_sum(acc2_b);
    let r2_c = simd_sum(acc2_c);
    let r2_d = simd_sum(acc2_d);
    let r3_a = simd_sum(acc3_a);
    let r3_b = simd_sum(acc3_b);
    let r3_c = simd_sum(acc3_c);
    let r3_d = simd_sum(acc3_d);
    if lane == 0u32 {
        store(out[m_row_a * n + row0], r0_a.cast::<T>());
        store(out[m_row_a * n + row1], r1_a.cast::<T>());
        store(out[m_row_a * n + row2], r2_a.cast::<T>());
        store(out[m_row_a * n + row3], r3_a.cast::<T>());
        store(out[m_row_b * n + row0], r0_b.cast::<T>());
        store(out[m_row_b * n + row1], r1_b.cast::<T>());
        store(out[m_row_b * n + row2], r2_b.cast::<T>());
        store(out[m_row_b * n + row3], r3_b.cast::<T>());
        store(out[m_row_c * n + row0], r0_c.cast::<T>());
        store(out[m_row_c * n + row1], r1_c.cast::<T>());
        store(out[m_row_c * n + row2], r2_c.cast::<T>());
        store(out[m_row_c * n + row3], r3_c.cast::<T>());
        store(out[m_row_d * n + row0], r0_d.cast::<T>());
        store(out[m_row_d * n + row1], r1_d.cast::<T>());
        store(out[m_row_d * n + row2], r2_d.cast::<T>());
        store(out[m_row_d * n + row3], r3_d.cast::<T>());
    }
}

// ─── mt_qmv_int8_fast ───────────────────────────────────────────────────
//
// Int8 decode GEMV — mirrors `mt_qmv`'s 8-row-per-TG geometry but for
// int8 weights (4 bytes/u32).
//
// ## Geometry
//   tpg = 64 = 2 simdgroups × 32 lanes
//   8 output rows per TG (each SG handles 4 rows indexed by simd_id)
//   Block = 4 X × 32 lanes = 128 K elements per outer iter
//   Grid: [m / 8, 1, 1]
//
// ## Int8 vs int4 adaptations
//   - Pack factor: 4 bytes/u32 (not 8 nibbles). Inner per-pack loop = 4.
//   - K-block = 128 (lane×4 X values × 32 lanes), not 512.
//   - Explicit byte shifts (0, 8, 16, 24) instead of the int4
//     mask-without-shift trick. The byte-position scale factors would be
//     1/256, 1/65536, 1/16777216 — four orders of magnitude smaller than
//     the int4 factors (1/16, 1/256, 1/4096) and too small to be useful
//     against typical f32 X values; explicit shifts are cleaner and avoid
//     the precision hazard of multiplying by ~6e-8.
//   - Bias hoist (algebraic split) retained: xs accumulates raw X per
//     block; acc += scale * qdot + bias * xs mirrors `mt_qmv` exactly.
//   - `& 0xFF` mask extracts byte codes 0..255 (unsigned int8).
//
// ## Layouts (same field names as mt_qmv)
//   w       [m, k/4]               u32   — int8 codes (4 per uint32)
//   scales  [m, k/group_size]      T
//   biases  [m, k/group_size]      T
//   x       [k]                    T
//   out     [m]                    T

#[bench_kernel(
    op="quantized",
    subop="qmv_int8_fast",
    class=QuantizedMatVec,
    shapes=&QUANTIZED_SHAPES,
    group_size=64,
    tpg=64,
    // bits=8: drives `run_quantized_mat_vec`'s W pack-factor (4
    // bytes/u32) + the bit-stream extract in the correctness oracle.
    // Without this the runner defaults to bits=4 and the int8 kernel
    // reads 2× the int4-sized W buffer.
    bits=8,
    tol=1e-3,
    mlx="affine_qmv_fast_float16_t_gs_64_b_8_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmv_int8_fast<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    // 8-row-per-TG geometry: 2 simdgroups × 32 lanes, each SG handles
    // 4 consecutive output rows. X loads are shared across all 4 rows
    // in a simdgroup, reducing X bandwidth by 4×. Mirrors `mt_qmv` for
    // int4 except K-block = 128 (4 X per lane × 32 lanes) and weight
    // codes use explicit byte extraction (shift 0/8/16/24, mask 0xFF).
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    // int8: 4 codes per u32. packs_per_row = k/4.
    let packs_per_row = k / 4u32;
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

    // Each lane owns 4 X values per K-block (32 lanes × 4 = 128 K elements).
    let lane_x_off = lane * 4u32;
    // One pack per lane per K-block (4 int8 codes per u32).
    let lane_pack_off = lane;

    for _b in range(0u32, k, 128u32) {
        // 4 X loads per lane — consecutive for vectorise fusion.
        let xb = _b + lane_x_off;
        let x0 = load(x[xb]).cast::<f32>();
        let x1 = load(x[xb + 1u32]).cast::<f32>();
        let x2 = load(x[xb + 2u32]).cast::<f32>();
        let x3 = load(x[xb + 3u32]).cast::<f32>();
        // Bias-hoist xs: raw X sum for the bias accumulation term.
        let xs = x0 + x1 + x2 + x3;

        // Group index: each lane's 4 X values fall within one group
        // (group_size=64; 4 × 32 lanes = 128 K-block ≥ 2 groups, but
        // within a lane the 4 slots belong to the same group).
        let g = xb / 64u32;
        let pack_off = _b / 4u32 + lane_pack_off;

        // ── Row 0 ──
        let p0 = load(w[w_base0 + pack_off]);
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        // Explicit byte extract: byte0 = p & 0xFF, byte1 = (p>>8) & 0xFF, etc.
        let q00 = (p0 & 255u32).cast::<f32>();
        let q01 = ((p0 >> 8u32) & 255u32).cast::<f32>();
        let q02 = ((p0 >> 16u32) & 255u32).cast::<f32>();
        let q03 = ((p0 >> 24u32) & 255u32).cast::<f32>();
        let qd0 = q00 * x0 + q01 * x1 + q02 * x2 + q03 * x3;
        acc0 = acc0 + s0 * qd0 + bi0 * xs;

        // ── Row 1 ──
        let p1 = load(w[w_base1 + pack_off]);
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p1 & 255u32).cast::<f32>();
        let q11 = ((p1 >> 8u32) & 255u32).cast::<f32>();
        let q12 = ((p1 >> 16u32) & 255u32).cast::<f32>();
        let q13 = ((p1 >> 24u32) & 255u32).cast::<f32>();
        let qd1 = q10 * x0 + q11 * x1 + q12 * x2 + q13 * x3;
        acc1 = acc1 + s1 * qd1 + bi1 * xs;

        // ── Row 2 ──
        let p2 = load(w[w_base2 + pack_off]);
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p2 & 255u32).cast::<f32>();
        let q21 = ((p2 >> 8u32) & 255u32).cast::<f32>();
        let q22 = ((p2 >> 16u32) & 255u32).cast::<f32>();
        let q23 = ((p2 >> 24u32) & 255u32).cast::<f32>();
        let qd2 = q20 * x0 + q21 * x1 + q22 * x2 + q23 * x3;
        acc2 = acc2 + s2 * qd2 + bi2 * xs;

        // ── Row 3 ──
        let p3 = load(w[w_base3 + pack_off]);
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p3 & 255u32).cast::<f32>();
        let q31 = ((p3 >> 8u32) & 255u32).cast::<f32>();
        let q32 = ((p3 >> 16u32) & 255u32).cast::<f32>();
        let q33 = ((p3 >> 24u32) & 255u32).cast::<f32>();
        let qd3 = q30 * x0 + q31 * x1 + q32 * x2 + q33 * x3;
        acc3 = acc3 + s3 * qd3 + bi3 * xs;
    }

    // Cross-lane reduction: each row's partial → single value, lane 0 stores.
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

// ─── mt_qmm_int8_fast ───────────────────────────────────────────────────
//
// Int8 matmul (M=1 batched form) — mirrors `mt_qmm` for int4 but adapted
// for int8 weights (4 bytes/u32). Same 8-row-per-TG geometry: 2 SG × 32
// lanes, each SG handles 4 N-rows. K-block = 128 (4 X per lane × 32
// lanes). Grid: [n / 8, m, 1].
//
// At M = 1 this is byte-identical to `mt_qmv_int8_fast` with the X
// index incorporating the M-row base. Bias hoist, explicit byte shifts,
// and algebraic-split accumulator all match `mt_qmv_int8_fast`.
//
// ## Layouts
//   w       [n, k/4]               u32   — int8 codes (4 per uint32)
//   scales  [n, k/group_size]      T
//   biases  [n, k/group_size]      T
//   x       [m, k]                 T
//   out     [m, n]                 T

#[bench_kernel(
    op="quantized",
    subop="qmm_int8_fast",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    m=4,
    group_size=64,
    tpg=64,
    bits=8,
    tol=1e-2,
    mlx="affine_qmm_fast_float16_t_gs_64_b_8_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_int8_fast<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_row = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    // int8: 4 codes per u32.
    let packs_per_row = k / 4u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    let x_row_base = m_row * k;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 4u32;
    let lane_pack_off = lane;

    for _b in range(0u32, k, 128u32) {
        let xb = x_row_base + _b + lane_x_off;
        let x0 = load(x[xb]).cast::<f32>();
        let x1 = load(x[xb + 1u32]).cast::<f32>();
        let x2 = load(x[xb + 2u32]).cast::<f32>();
        let x3 = load(x[xb + 3u32]).cast::<f32>();
        let xs = x0 + x1 + x2 + x3;

        // Group index recomputed against K-local base (xb includes x_row_base).
        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 4u32 + lane_pack_off;

        // ── Row 0 ──
        let p0 = load(w[w_base0 + pack_off]);
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        let q00 = (p0 & 255u32).cast::<f32>();
        let q01 = ((p0 >> 8u32) & 255u32).cast::<f32>();
        let q02 = ((p0 >> 16u32) & 255u32).cast::<f32>();
        let q03 = ((p0 >> 24u32) & 255u32).cast::<f32>();
        let qd0 = q00 * x0 + q01 * x1 + q02 * x2 + q03 * x3;
        acc0 = acc0 + s0 * qd0 + bi0 * xs;

        // ── Row 1 ──
        let p1 = load(w[w_base1 + pack_off]);
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p1 & 255u32).cast::<f32>();
        let q11 = ((p1 >> 8u32) & 255u32).cast::<f32>();
        let q12 = ((p1 >> 16u32) & 255u32).cast::<f32>();
        let q13 = ((p1 >> 24u32) & 255u32).cast::<f32>();
        let qd1 = q10 * x0 + q11 * x1 + q12 * x2 + q13 * x3;
        acc1 = acc1 + s1 * qd1 + bi1 * xs;

        // ── Row 2 ──
        let p2 = load(w[w_base2 + pack_off]);
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p2 & 255u32).cast::<f32>();
        let q21 = ((p2 >> 8u32) & 255u32).cast::<f32>();
        let q22 = ((p2 >> 16u32) & 255u32).cast::<f32>();
        let q23 = ((p2 >> 24u32) & 255u32).cast::<f32>();
        let qd2 = q20 * x0 + q21 * x1 + q22 * x2 + q23 * x3;
        acc2 = acc2 + s2 * qd2 + bi2 * xs;

        // ── Row 3 ──
        let p3 = load(w[w_base3 + pack_off]);
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p3 & 255u32).cast::<f32>();
        let q31 = ((p3 >> 8u32) & 255u32).cast::<f32>();
        let q32 = ((p3 >> 16u32) & 255u32).cast::<f32>();
        let q33 = ((p3 >> 24u32) & 255u32).cast::<f32>();
        let qd3 = q30 * x0 + q31 * x1 + q32 * x2 + q33 * x3;
        acc3 = acc3 + s3 * qd3 + bi3 * xs;
    }

    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(out[m_row * n + row0], r0.cast::<T>());
        store(out[m_row * n + row1], r1.cast::<T>());
        store(out[m_row * n + row2], r2.cast::<T>());
        store(out[m_row * n + row3], r3.cast::<T>());
    }
}

// ─── mt_qmm_bm2_int8_fast ───────────────────────────────────────────────
//
// Int8 matmul BM=2 — mirrors `mt_qmm_bm2` for int4 but adapted for
// int8 weights. Two M-rows per TG; W packs extracted once per K-block
// per N-row, reused across both M-rows. K-block = 128.
//
// ## Geometry
//   tpg = 64 = 2 SG × 32 lanes
//   BM = 2 (M-rows per TG), BN = 8 (N-rows per TG)
//   16 outputs per TG, Grid: [n / 8, m / 2, 1]
//
// ## Layouts
//   w       [n, k/4]               u32
//   scales  [n, k/group_size]      T
//   biases  [n, k/group_size]      T
//   x       [m, k]                 T
//   out     [m, n]                 T

#[bench_kernel(
    op="quantized",
    subop="qmm_bm2_int8_fast",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    m=8,
    group_size=64,
    tpg=64,
    bits=8,
    tol=1e-2,
    mlx="affine_qmm_fast_float16_t_gs_64_b_8_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_bm2_int8_fast<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_tile = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    // int8: 4 codes per u32.
    let packs_per_row = k / 4u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    // BM=2 M-rows per TG.
    let m_row_a = m_tile * 2u32;
    let m_row_b = m_row_a + 1u32;
    let x_base_a = m_row_a * k;
    let x_base_b = m_row_b * k;

    // 8 accumulators: 4 N-rows × 2 M-rows.
    let mut acc0_a = 0.0f32;
    let mut acc0_b = 0.0f32;
    let mut acc1_a = 0.0f32;
    let mut acc1_b = 0.0f32;
    let mut acc2_a = 0.0f32;
    let mut acc2_b = 0.0f32;
    let mut acc3_a = 0.0f32;
    let mut acc3_b = 0.0f32;

    let lane_x_off = lane * 4u32;
    let lane_pack_off = lane;

    for _b in range(0u32, k, 128u32) {
        // ── Load 4 X values for M-row A ──
        let xb_a = x_base_a + _b + lane_x_off;
        let x0_a = load(x[xb_a]).cast::<f32>();
        let x1_a = load(x[xb_a + 1u32]).cast::<f32>();
        let x2_a = load(x[xb_a + 2u32]).cast::<f32>();
        let x3_a = load(x[xb_a + 3u32]).cast::<f32>();
        let xs_a = x0_a + x1_a + x2_a + x3_a;

        // ── Load 4 X values for M-row B ──
        let xb_b = x_base_b + _b + lane_x_off;
        let x0_b = load(x[xb_b]).cast::<f32>();
        let x1_b = load(x[xb_b + 1u32]).cast::<f32>();
        let x2_b = load(x[xb_b + 2u32]).cast::<f32>();
        let x3_b = load(x[xb_b + 3u32]).cast::<f32>();
        let xs_b = x0_b + x1_b + x2_b + x3_b;

        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 4u32 + lane_pack_off;

        // ── Row 0 (shared W extract, dual qdots) ──
        let p0 = load(w[w_base0 + pack_off]);
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        let q00 = (p0 & 255u32).cast::<f32>();
        let q01 = ((p0 >> 8u32) & 255u32).cast::<f32>();
        let q02 = ((p0 >> 16u32) & 255u32).cast::<f32>();
        let q03 = ((p0 >> 24u32) & 255u32).cast::<f32>();
        let qd0_a = q00 * x0_a + q01 * x1_a + q02 * x2_a + q03 * x3_a;
        let qd0_b = q00 * x0_b + q01 * x1_b + q02 * x2_b + q03 * x3_b;
        acc0_a = acc0_a + s0 * qd0_a + bi0 * xs_a;
        acc0_b = acc0_b + s0 * qd0_b + bi0 * xs_b;

        // ── Row 1 ──
        let p1 = load(w[w_base1 + pack_off]);
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p1 & 255u32).cast::<f32>();
        let q11 = ((p1 >> 8u32) & 255u32).cast::<f32>();
        let q12 = ((p1 >> 16u32) & 255u32).cast::<f32>();
        let q13 = ((p1 >> 24u32) & 255u32).cast::<f32>();
        let qd1_a = q10 * x0_a + q11 * x1_a + q12 * x2_a + q13 * x3_a;
        let qd1_b = q10 * x0_b + q11 * x1_b + q12 * x2_b + q13 * x3_b;
        acc1_a = acc1_a + s1 * qd1_a + bi1 * xs_a;
        acc1_b = acc1_b + s1 * qd1_b + bi1 * xs_b;

        // ── Row 2 ──
        let p2 = load(w[w_base2 + pack_off]);
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p2 & 255u32).cast::<f32>();
        let q21 = ((p2 >> 8u32) & 255u32).cast::<f32>();
        let q22 = ((p2 >> 16u32) & 255u32).cast::<f32>();
        let q23 = ((p2 >> 24u32) & 255u32).cast::<f32>();
        let qd2_a = q20 * x0_a + q21 * x1_a + q22 * x2_a + q23 * x3_a;
        let qd2_b = q20 * x0_b + q21 * x1_b + q22 * x2_b + q23 * x3_b;
        acc2_a = acc2_a + s2 * qd2_a + bi2 * xs_a;
        acc2_b = acc2_b + s2 * qd2_b + bi2 * xs_b;

        // ── Row 3 ──
        let p3 = load(w[w_base3 + pack_off]);
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p3 & 255u32).cast::<f32>();
        let q31 = ((p3 >> 8u32) & 255u32).cast::<f32>();
        let q32 = ((p3 >> 16u32) & 255u32).cast::<f32>();
        let q33 = ((p3 >> 24u32) & 255u32).cast::<f32>();
        let qd3_a = q30 * x0_a + q31 * x1_a + q32 * x2_a + q33 * x3_a;
        let qd3_b = q30 * x0_b + q31 * x1_b + q32 * x2_b + q33 * x3_b;
        acc3_a = acc3_a + s3 * qd3_a + bi3 * xs_a;
        acc3_b = acc3_b + s3 * qd3_b + bi3 * xs_b;
    }

    // Cross-lane reduce + lane-0 stores. 8 outputs per TG.
    let r0_a = simd_sum(acc0_a);
    let r0_b = simd_sum(acc0_b);
    let r1_a = simd_sum(acc1_a);
    let r1_b = simd_sum(acc1_b);
    let r2_a = simd_sum(acc2_a);
    let r2_b = simd_sum(acc2_b);
    let r3_a = simd_sum(acc3_a);
    let r3_b = simd_sum(acc3_b);
    if lane == 0u32 {
        store(out[m_row_a * n + row0], r0_a.cast::<T>());
        store(out[m_row_a * n + row1], r1_a.cast::<T>());
        store(out[m_row_a * n + row2], r2_a.cast::<T>());
        store(out[m_row_a * n + row3], r3_a.cast::<T>());
        store(out[m_row_b * n + row0], r0_b.cast::<T>());
        store(out[m_row_b * n + row1], r1_b.cast::<T>());
        store(out[m_row_b * n + row2], r2_b.cast::<T>());
        store(out[m_row_b * n + row3], r3_b.cast::<T>());
    }
}

// ─── mt_qmm_bm4_int8_fast ───────────────────────────────────────────────
//
// Int8 matmul BM=4 — mirrors `mt_qmm_bm4` for int4 but adapted for
// int8 weights. Four M-rows per TG; W packs loaded once per K-block per
// N-row and reused across all 4 M-rows. K-block = 128 (4 X per lane ×
// 32 lanes).
//
// ## Geometry
//   tpg = 64 = 2 SG × 32 lanes
//   BM = 4 (M-rows per TG), BN = 8 (N-rows per TG)
//   32 outputs per TG, Grid: [n / 8, m / 4, 1]
//
// ## Layouts
//   w       [n, k/4]               u32
//   scales  [n, k/group_size]      T
//   biases  [n, k/group_size]      T
//   x       [m, k]                 T
//   out     [m, n]                 T

#[bench_kernel(
    op="quantized",
    subop="qmm_bm4_int8_fast",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    m=8,
    group_size=64,
    tpg=64,
    bits=8,
    tol=1e-2,
    mlx="affine_qmm_fast_float16_t_gs_64_b_8_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_bm4_int8_fast<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_tile = tgid_y;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    // int8: 4 codes per u32.
    let packs_per_row = k / 4u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    // BM=4 M-rows per TG.
    let m_row_a = m_tile * 4u32;
    let m_row_b = m_row_a + 1u32;
    let m_row_c = m_row_a + 2u32;
    let m_row_d = m_row_a + 3u32;
    let x_base_a = m_row_a * k;
    let x_base_b = m_row_b * k;
    let x_base_c = m_row_c * k;
    let x_base_d = m_row_d * k;

    // 16 accumulators: 4 N-rows × 4 M-rows.
    let mut acc0_a = 0.0f32;
    let mut acc0_b = 0.0f32;
    let mut acc0_c = 0.0f32;
    let mut acc0_d = 0.0f32;
    let mut acc1_a = 0.0f32;
    let mut acc1_b = 0.0f32;
    let mut acc1_c = 0.0f32;
    let mut acc1_d = 0.0f32;
    let mut acc2_a = 0.0f32;
    let mut acc2_b = 0.0f32;
    let mut acc2_c = 0.0f32;
    let mut acc2_d = 0.0f32;
    let mut acc3_a = 0.0f32;
    let mut acc3_b = 0.0f32;
    let mut acc3_c = 0.0f32;
    let mut acc3_d = 0.0f32;

    // 4 X per lane × 32 lanes = 128 K-block.
    let lane_x_off = lane * 4u32;
    let lane_pack_off = lane;

    for _b in range(0u32, k, 128u32) {
        // ── Load 4 X values for each of the 4 M-rows ──
        let xb_a = x_base_a + _b + lane_x_off;
        let x0_a = load(x[xb_a]).cast::<f32>();
        let x1_a = load(x[xb_a + 1u32]).cast::<f32>();
        let x2_a = load(x[xb_a + 2u32]).cast::<f32>();
        let x3_a = load(x[xb_a + 3u32]).cast::<f32>();
        let xs_a = x0_a + x1_a + x2_a + x3_a;

        let xb_b = x_base_b + _b + lane_x_off;
        let x0_b = load(x[xb_b]).cast::<f32>();
        let x1_b = load(x[xb_b + 1u32]).cast::<f32>();
        let x2_b = load(x[xb_b + 2u32]).cast::<f32>();
        let x3_b = load(x[xb_b + 3u32]).cast::<f32>();
        let xs_b = x0_b + x1_b + x2_b + x3_b;

        let xb_c = x_base_c + _b + lane_x_off;
        let x0_c = load(x[xb_c]).cast::<f32>();
        let x1_c = load(x[xb_c + 1u32]).cast::<f32>();
        let x2_c = load(x[xb_c + 2u32]).cast::<f32>();
        let x3_c = load(x[xb_c + 3u32]).cast::<f32>();
        let xs_c = x0_c + x1_c + x2_c + x3_c;

        let xb_d = x_base_d + _b + lane_x_off;
        let x0_d = load(x[xb_d]).cast::<f32>();
        let x1_d = load(x[xb_d + 1u32]).cast::<f32>();
        let x2_d = load(x[xb_d + 2u32]).cast::<f32>();
        let x3_d = load(x[xb_d + 3u32]).cast::<f32>();
        let xs_d = x0_d + x1_d + x2_d + x3_d;

        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 4u32 + lane_pack_off;

        // ── Row 0 (W extracts shared across all 4 M-rows) ──
        let p0 = load(w[w_base0 + pack_off]);
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        let q00 = (p0 & 255u32).cast::<f32>();
        let q01 = ((p0 >> 8u32) & 255u32).cast::<f32>();
        let q02 = ((p0 >> 16u32) & 255u32).cast::<f32>();
        let q03 = ((p0 >> 24u32) & 255u32).cast::<f32>();
        let qd0_a = q00 * x0_a + q01 * x1_a + q02 * x2_a + q03 * x3_a;
        let qd0_b = q00 * x0_b + q01 * x1_b + q02 * x2_b + q03 * x3_b;
        let qd0_c = q00 * x0_c + q01 * x1_c + q02 * x2_c + q03 * x3_c;
        let qd0_d = q00 * x0_d + q01 * x1_d + q02 * x2_d + q03 * x3_d;
        acc0_a = acc0_a + s0 * qd0_a + bi0 * xs_a;
        acc0_b = acc0_b + s0 * qd0_b + bi0 * xs_b;
        acc0_c = acc0_c + s0 * qd0_c + bi0 * xs_c;
        acc0_d = acc0_d + s0 * qd0_d + bi0 * xs_d;

        // ── Row 1 ──
        let p1 = load(w[w_base1 + pack_off]);
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p1 & 255u32).cast::<f32>();
        let q11 = ((p1 >> 8u32) & 255u32).cast::<f32>();
        let q12 = ((p1 >> 16u32) & 255u32).cast::<f32>();
        let q13 = ((p1 >> 24u32) & 255u32).cast::<f32>();
        let qd1_a = q10 * x0_a + q11 * x1_a + q12 * x2_a + q13 * x3_a;
        let qd1_b = q10 * x0_b + q11 * x1_b + q12 * x2_b + q13 * x3_b;
        let qd1_c = q10 * x0_c + q11 * x1_c + q12 * x2_c + q13 * x3_c;
        let qd1_d = q10 * x0_d + q11 * x1_d + q12 * x2_d + q13 * x3_d;
        acc1_a = acc1_a + s1 * qd1_a + bi1 * xs_a;
        acc1_b = acc1_b + s1 * qd1_b + bi1 * xs_b;
        acc1_c = acc1_c + s1 * qd1_c + bi1 * xs_c;
        acc1_d = acc1_d + s1 * qd1_d + bi1 * xs_d;

        // ── Row 2 ──
        let p2 = load(w[w_base2 + pack_off]);
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p2 & 255u32).cast::<f32>();
        let q21 = ((p2 >> 8u32) & 255u32).cast::<f32>();
        let q22 = ((p2 >> 16u32) & 255u32).cast::<f32>();
        let q23 = ((p2 >> 24u32) & 255u32).cast::<f32>();
        let qd2_a = q20 * x0_a + q21 * x1_a + q22 * x2_a + q23 * x3_a;
        let qd2_b = q20 * x0_b + q21 * x1_b + q22 * x2_b + q23 * x3_b;
        let qd2_c = q20 * x0_c + q21 * x1_c + q22 * x2_c + q23 * x3_c;
        let qd2_d = q20 * x0_d + q21 * x1_d + q22 * x2_d + q23 * x3_d;
        acc2_a = acc2_a + s2 * qd2_a + bi2 * xs_a;
        acc2_b = acc2_b + s2 * qd2_b + bi2 * xs_b;
        acc2_c = acc2_c + s2 * qd2_c + bi2 * xs_c;
        acc2_d = acc2_d + s2 * qd2_d + bi2 * xs_d;

        // ── Row 3 ──
        let p3 = load(w[w_base3 + pack_off]);
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p3 & 255u32).cast::<f32>();
        let q31 = ((p3 >> 8u32) & 255u32).cast::<f32>();
        let q32 = ((p3 >> 16u32) & 255u32).cast::<f32>();
        let q33 = ((p3 >> 24u32) & 255u32).cast::<f32>();
        let qd3_a = q30 * x0_a + q31 * x1_a + q32 * x2_a + q33 * x3_a;
        let qd3_b = q30 * x0_b + q31 * x1_b + q32 * x2_b + q33 * x3_b;
        let qd3_c = q30 * x0_c + q31 * x1_c + q32 * x2_c + q33 * x3_c;
        let qd3_d = q30 * x0_d + q31 * x1_d + q32 * x2_d + q33 * x3_d;
        acc3_a = acc3_a + s3 * qd3_a + bi3 * xs_a;
        acc3_b = acc3_b + s3 * qd3_b + bi3 * xs_b;
        acc3_c = acc3_c + s3 * qd3_c + bi3 * xs_c;
        acc3_d = acc3_d + s3 * qd3_d + bi3 * xs_d;
    }

    // Cross-lane reduce + lane-0 stores. 16 outputs per TG.
    let r0_a = simd_sum(acc0_a);
    let r0_b = simd_sum(acc0_b);
    let r0_c = simd_sum(acc0_c);
    let r0_d = simd_sum(acc0_d);
    let r1_a = simd_sum(acc1_a);
    let r1_b = simd_sum(acc1_b);
    let r1_c = simd_sum(acc1_c);
    let r1_d = simd_sum(acc1_d);
    let r2_a = simd_sum(acc2_a);
    let r2_b = simd_sum(acc2_b);
    let r2_c = simd_sum(acc2_c);
    let r2_d = simd_sum(acc2_d);
    let r3_a = simd_sum(acc3_a);
    let r3_b = simd_sum(acc3_b);
    let r3_c = simd_sum(acc3_c);
    let r3_d = simd_sum(acc3_d);
    if lane == 0u32 {
        store(out[m_row_a * n + row0], r0_a.cast::<T>());
        store(out[m_row_a * n + row1], r1_a.cast::<T>());
        store(out[m_row_a * n + row2], r2_a.cast::<T>());
        store(out[m_row_a * n + row3], r3_a.cast::<T>());
        store(out[m_row_b * n + row0], r0_b.cast::<T>());
        store(out[m_row_b * n + row1], r1_b.cast::<T>());
        store(out[m_row_b * n + row2], r2_b.cast::<T>());
        store(out[m_row_b * n + row3], r3_b.cast::<T>());
        store(out[m_row_c * n + row0], r0_c.cast::<T>());
        store(out[m_row_c * n + row1], r1_c.cast::<T>());
        store(out[m_row_c * n + row2], r2_c.cast::<T>());
        store(out[m_row_c * n + row3], r3_c.cast::<T>());
        store(out[m_row_d * n + row0], r0_d.cast::<T>());
        store(out[m_row_d * n + row1], r1_d.cast::<T>());
        store(out[m_row_d * n + row2], r2_d.cast::<T>());
        store(out[m_row_d * n + row3], r3_d.cast::<T>());
    }
}

// ─── mt_qmm_mma ─────────────────────────────────────────────────────────
//
// Quantized matmul via simdgroup-matrix MMA — the M ≥ 32 path that hits
// MLX's `affine_qmm_t` ALU throughput. Mirrors MLX's 32×32 output tile
// geometry (BM=BN=32, BK=32, 4 SG × 32 lanes = 128 tpg) and uses the
// dequant-into-TG-memory pattern from MLX + llama.cpp Q4: int4 W is
// staged as fp T in TG mem once per K-block; all matmuls read pure fp T.
//
// Geometry:
//   tpg = 128 = 4 SG × 32 lanes (WM=2, WN=2 warp grid)
//   BM = BN = BK = 32, output tile 32×32 (1024 outputs/TG)
//   Grid: [N/32, M/32, 1]
//   Each SG owns a 16×16 sub-tile = 2×2 = 4 8×8 frags
//   Per K-block per SG: 4 frags × 4 k-inner = 16 MMAs (64 across TG)
//
// Threadgroup memory:
//   Xs[32 * 32] = 1024 T  (X tile, row-major [BM × BK])
//   Ws[32 * 32] = 1024 T  (dequant W tile, row-major [BN × BK])
//   Total: 2048 T (8 KB f32 / 4 KB f16) — fits 32 KB M2 budget easily.
//
// Per K-block:
//   1. Coop X load — 128 lanes × 8 elems each fill 1024-elt Xs tile.
//   2. Coop W dequant — 128 lanes × 1 pack (=8 nibbles) → 1024 fp T into
//      Ws tile. Scale/bias picked per-(w_row, group) inside dequant.
//   3. threadgroup_barrier()
//   4. Per SG: 4 frags × 4 k-inner unrolled MMA. A=X, B=W^T (qmm_t).
//      A_frag elem[i] @ (fm, fn_i): Xs[(sm*16 + frag_m + fm) * 32 + k_inner*8 + fn_i]
//      B_frag elem[i] @ (fm, fn_i): Ws[(sn*16 + frag_n + fn_i) * 32 + k_inner*8 + fm]
//      (swap fm↔fn role on B since B is K×N read from N×K row-major as transpose)
//   5. After K-loop, write each SG's 4 frags to global out.
//
// Frag lane mapping (Apple steel_gemm layout, same as steel_attention_mma.rs):
//   qid = lane/4, fm = (qid & 4) + ((lane/2) % 4),
//   fn0 = (qid & 2)*2 + (lane%2)*2, fn1 = fn0 + 1
// Each lane owns 2 elements per 8×8 frag at (fm, fn0) and (fm, fn1).
//
// W packing: each u32 pack holds 8 int4 nibbles. Per K-block lane at
// (w_row, pack_in_row) loads ONE pack at w_pack_row_base + kb/8 +
// pack_in_row, then dequants all 8 nibbles into
// Ws[w_row * 32 + pack_in_row * 8 + i] for i in 0..8. Group index per
// pack = (kb + pack_in_row * 8) / 64 — same across all 8 nibbles since
// 8 < 64.
//
// Closes the M ≥ 32 gap to MLX: simdgroup-matrix MMA = 8×8×8 = 512 MACs
// per simdgroup per instruction vs bm4's 32 scalar FMAs + simd_sum.
// Predicted on the QUANTIZED_SHAPES grid: ≥ 100% MT MLX at M=32 (M5)
// / ≥ 80% MT MLX (M2). M=8/16 stays on bm4 — MMA tile is 75%/50% empty
// at those Ms (would waste 1024-output budget).
#[bench_kernel(
    op="quantized",
    subop="qmm_mma",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=32 = the cell where the simdgroup-matrix MMA pays for itself.
    // M < 32 leaves >= 50% of the 32×32 tile padded (wasted ALU); bm4
    // keeps winning there. Selector routes M >= 32 && M %% 32 == 0 to mma.
    m=32,
    group_size=64,
    tpg=128,
    // bf16 round-trip on int4-quantized matmul: max_q=15 × group_size=64
    // × bf16's 7-bit mantissa drifts ~7-8e-3 at large K (per
    // crates/metaltile-std/src/mlx/binary.rs precedent — "bf16 drifts
    // ~7.8e-3 on signed"). Tighter than 1e-2 trips the bench cosine
    // check at production shapes (M=4096+, K=4096+) on Apple Paravirtual
    // CI. tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 4 SGs in 2×2 warp grid: sg ∈ {0,1,2,3} → (sm=sg/2, sn=sg%2).
    // Each SG owns a 16×16 sub-tile at (sm*16, sn*16) inside the 32×32
    // output tile.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;

    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // TG memory for X tile [BM × BK] and dequant W tile [BN × BK].
    // Stride = BK + 4 = 36 (skew by 4 T elements) to break 32-bank conflicts
    // on the 8×8 frag column reads. Without skew, A-frag reads collide
    // 8-way + B-frag reads collide 4-way (32-stride mod 32-bank = same bank
    // across rows). Skew = 4 → row stride mod 32 = 4, spreads bank usage
    // across all 32 banks. Size = 32 × 36 = 1152 T (4.5 KB f32 / 2.25 KB f16).
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);

    // ── 4 output frags per SG, init to 0 ──
    // c_f<row_frag><col_frag>: row_frag = fm-axis (0 = rows 0..7 of 16-row
    // sub-tile, 1 = rows 8..15), col_frag = fn-axis (likewise for cols).
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);

    // A (X) and B (W^T) frag scratch, reused per k_inner. Keep at native
    // T precision — Fix 5 (upcast to f32 to mirror MLX's half→float MMA
    // path) was tested in layered-bench: identical f16 numbers (93-96%
    // MT MLX) with and without the upcast, so we keep simpler half MMA.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();

    // Coop-load lane assignments.
    // X tile: 32×32 = 1024 elements, 128 lanes × 8 strides; each step i in
    //   0..8 writes Xs[i*128 + lane_in_tg] from X[m_row, k_col] where
    //   m_row = flat/32, k_col = flat%32 (per the 32-col row-major layout).
    // W tile: 32×32 = 1024 elements, 128 packs = 128 lanes × 1 pack each.
    //   lane_in_tg = w_row * 4 + pack_in_row → w_row ∈ 0..31, pack_in_row ∈ 0..3.
    //   Each lane dequants 8 nibbles into Ws[w_row*32 + pack_in_row*8 + i].
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;

    let x_m_base = m_tile * 32u32; // first M-row this TG handles
    let w_n_base = n_tile * 32u32; // first N-row this TG handles
    let packs_per_row = k / 8u32;

    // Per-lane scale/bias row base (for W dequant). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * gs_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;

    // TG row stride. Default is BK + 4 = 36 (skew by 4 T elements) to break
    // 32-bank conflicts on the 8×8 frag column reads. For f16 the dtype-aware
    // skew formula `BK + 16/sizeof(T)` (mirroring MLX `affine_qmm_t`) bumps
    // this to 40 — see Fix 1 in `/tmp/mlx_archaeology.md`. The actual value
    // is patched per-dtype in `mt_qmm_for` (`xs_ld_const`/`ws_ld_const` IR
    // names are looked up + rewritten); the literal `36u32` here is the f32
    // value and is also the default for non-f16 dtypes.
    let xs_ld_const = 36u32;
    let ws_ld_const = 36u32;
    let xs_ld = xs_ld_const;
    let ws_ld = ws_ld_const;

    // Coop X-load mapping. 32×32 = 1024 elements. 128 lanes × 8 elems each.
    // To enable vec4 device-load fusion (Fix 3 — see archaeology §4.2): map
    // each lane to (m_row, k_quad) reading 8 *contiguous* Ks. Then the 8
    // device-loads have indices `base + 0..7` which the vectorize pass fuses
    // into 2× vec4 loads (MAX_VEC_RUN=4). Previous mapping read strided
    // halves across 8 rows → no fusion possible.
    //
    // lane_in_tg ∈ 0..128, m_row = lane_in_tg / 4 ∈ 0..32, k_quad =
    // lane_in_tg % 4 ∈ 0..4. Per lane writes 8 contiguous halves into
    // Xs[m_row*xs_ld + k_quad*8 + i] for i in 0..8.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;

    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        // Each lane: read 8 contiguous halves from one X-row at k offset
        // x_k_base, write to Xs[x_m_row*xs_ld + x_k_base + i].
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        // 8 contiguous device loads → 2× vec4 after vectorize pass.
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);

        // ── 2. Coop W dequant — each lane loads 1 pack and writes 8 fp T ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / 64u32; // group_size = 64
        let s = load(scales[sb_base + g]).cast::<f32>();
        let b = load(biases[sb_base + g]).cast::<f32>();
        // Mask-without-shift trick — scale by 1/(16^i) instead of >>i*4.
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
        let pack_hi = pack >> 16u32;
        let q0 = (pack & 15u32).cast::<f32>();
        let q1 = (pack & 240u32).cast::<f32>() * s_16;
        let q2 = (pack & 3840u32).cast::<f32>() * s_256;
        let q3 = (pack & 61440u32).cast::<f32>() * s_4096;
        let q4 = (pack_hi & 15u32).cast::<f32>();
        let q5 = (pack_hi & 240u32).cast::<f32>() * s_16;
        let q6 = (pack_hi & 3840u32).cast::<f32>() * s_256;
        let q7 = (pack_hi & 61440u32).cast::<f32>() * s_4096;
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        threadgroup_store("ws", ws_base, (s * q0 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 1u32, (s * q1 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 2u32, (s * q2 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 3u32, (s * q3 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 4u32, (s * q4 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 5u32, (s * q5 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 6u32, (s * q6 + b).cast::<T>());
        threadgroup_store("ws", ws_base + 7u32, (s * q7 + b).cast::<T>());

        threadgroup_barrier();

        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        // A-frag (X) elem[i] @ (fm, fn_i) = Xs[(sm*16 + frag_m + fm) * 36 + (k_inner*8 + fn_i)]
        // B-frag (W^T) elem[i] @ (fm, fn_i) = Ws[(sn*16 + frag_n + fn_i) * 36 + (k_inner*8 + fm)]
        let row_a0 = sm * 16u32 + fm; // frag_m = 0
        let row_a1 = sm * 16u32 + 8u32 + fm; // frag_m = 8
        let col_b0 = sn * 16u32; // frag_n = 0 base offset
        let col_b1 = sn * 16u32 + 8u32; // frag_n = 8 base offset

        // k_inner = 0 (k offset 0..7 inside BK=32) — 3-barrier MLX pattern
        // (Fix 2) + serpentine (Fix 4): A-load, barrier, B-load, barrier,
        // MMAs in (0,0)→(0,1)→(1,1)→(1,0) order, barrier.
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 1 (k offset 8..15)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 2 (k offset 16..23)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 3 (k offset 24..31)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        threadgroup_barrier();
    }

    // ── 4. Write 4 C frags to global out ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    // c_f00 at (frag_m=0, frag_n=0)
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ─── mt_qmm_mma_m16 ─────────────────────────────────────────────────────
//
// Half-height simdgroup-matrix MMA — the M=16 cell. The full-tile
// `mt_qmm_mma` requires `m % 32 == 0`, so M=16 falls through to bm4 in
// the selector. bm4 wins moderate-N at M=16 but loses wide-N on M2 Pro
// (76-94% MT MLX). This kernel maps M=16 exactly to 2 frag_m positions
// (no padding waste) and gets MMA-class ALU + N-amortization.
//
// Geometry:
//   tpg = 64 = 2 SG × 32 lanes (WM=1, WN=2 warp grid)
//   BM = 16, BN = BK = 32 → 16×32 output tile (512 outputs/TG)
//   Grid: [N/32, M/16, 1]
//   Each SG owns a 16×16 sub-tile = 2 frag_m × 2 frag_n = 4 8×8 frags
//   (same MMA work-per-SG as mt_qmm_mma — we halve only the warp-m grid)
//   Per K-block per SG: 4 frags × 4 k-inner = 16 MMAs (32 across TG)
//
// Threadgroup memory (skewed BK+4 = 36 stride to break 32-bank conflicts):
//   Xs[16 × 36] = 576 T  (X tile, 16 rows × 32 cols + 4 skew)
//   Ws[32 × 36] = 1152 T (dequant W tile, 32 rows × 32 cols + 4 skew)
//   Total: 1728 T (6.75 KB f32 / 3.375 KB f16) — half the X tile of
//   mt_qmm_mma; same Ws.
//
// Per K-block:
//   1. Coop X load — 64 lanes × 8 strides each fill 512-elt Xs tile.
//      flat = i*64 + lane_in_tg ∈ 0..511; m_row = flat/32 ∈ 0..15.
//   2. Coop W dequant — 128 packs / 64 lanes = 2 packs per lane. Each
//      step covers half the W tile (64 packs).
//   3. threadgroup_barrier()
//   4. Per SG (sm=0, sn = sg & 1): 4 frags × 4 k-inner unrolled MMA —
//      identical body to mt_qmm_mma's per-SG inner loop.
//   5. After K-loop, write each SG's 4 frags to global out.
//
// Frag lane mapping (Apple steel_gemm layout, same as mt_qmm_mma):
//   qid = lane/4, fm = (qid & 4) + ((lane/2) % 4),
//   fn0 = (qid & 2)*2 + (lane%2)*2, fn1 = fn0 + 1
//
// Per-lane register footprint: 4 C f32 frags (16 elems × 4B = 64B) +
// 4 A/B T frags (~16B) + scratch ≈ 256 B — well under the ~1 KB/lane
// register file. Occupancy at 64 tpg is 2× mt_qmm_mma's 128 tpg.
//
// At M < 16 this kernel padding-wastes >= 50% of the tile (would route
// to bm2/bm4). At M = 32 use mt_qmm_mma (full tile, 2× the output
// budget). Selector route: `m == 16` → mt_qmm_mma_m16.
#[bench_kernel(
    op="quantized",
    subop="qmm_mma_m16",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=16 = the bm4 weak cell. bm4 wins moderate-N M=16 cells but
    // loses wide-N on M2 (76-94% MT MLX). Half-height MMA targets this
    // exact gap: zero padding waste at M=16 (vs MMA's 32×32 tile which
    // would be 50% empty here), MMA-class ALU, N-amortized W reuse.
    m=16,
    group_size=64,
    tpg=64,
    // bf16 round-trip on int4-quantized matmul: max_q=15 × group_size=64
    // × bf16's 7-bit mantissa drifts ~7-8e-3 at large K (per
    // crates/metaltile-std/src/mlx/binary.rs precedent — "bf16 drifts
    // ~7.8e-3 on signed"). Tighter than 1e-2 trips the bench cosine
    // check at production shapes (M=4096+, K=4096+) on Apple Paravirtual
    // CI. tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_mma_m16<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 2 SGs in 1×2 warp grid: sg ∈ {0,1} → (sm=0, sn=sg).
    // Each SG owns a 16×16 sub-tile at (0, sn*16) inside the 16×32
    // output tile. WM=1 — both SGs cover the full BM=16 rows.
    let sm = 0u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;

    // 8×8 frag lane mapping (Apple steel_gemm layout — same as mt_qmm_mma).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // TG memory for X tile [16 × 32] and dequant W tile [32 × 32].
    // BK + 4 = 36 stride for bank-conflict avoidance — same skew rationale
    // as mt_qmm_mma. Xs size = 16 × 36 = 576 T; Ws size = 32 × 36 = 1152 T.
    threadgroup_alloc("xs", 576, T);
    threadgroup_alloc("ws", 1152, T);

    // ── 4 output frags per SG, init to 0 ──
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);

    // A (X) and B (W^T) frag scratch, reused per k_inner.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();

    // Coop-load lane assignments.
    // X tile: 16×32 = 512 elements, 64 lanes × 8 strides each → 512 covers.
    //   step i in 0..8: flat = i*64 + lane_in_tg ∈ 0..511.
    //   m_row = flat/32 ∈ 0..15, k_col = flat%32.
    // W tile: 32×32 = 1024 elements, 128 packs, 64 lanes × 2 packs each.
    //   pack_idx = step*64 + lane_in_tg, step ∈ {0, 1}.
    //   Each pack at (w_row=pack_idx/4, pack_in_row=pack_idx%4) dequants
    //   8 nibbles into Ws[w_row*36 + pack_in_row*8 + i].
    let x_m_base = m_tile * 16u32; // first M-row this TG handles
    let w_n_base = n_tile * 32u32; // first N-row this TG handles
    let packs_per_row = k / 8u32;

    // TG row stride = 36 (BK + 4 skew).
    let xs_ld = 36u32;
    let ws_ld = 36u32;

    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 64 lanes × 8 strides each fill 512-elt tile ──
        let flat0 = lane_in_tg;
        let flat1 = 64u32 + lane_in_tg;
        let flat2 = 128u32 + lane_in_tg;
        let flat3 = 192u32 + lane_in_tg;
        let flat4 = 256u32 + lane_in_tg;
        let flat5 = 320u32 + lane_in_tg;
        let flat6 = 384u32 + lane_in_tg;
        let flat7 = 448u32 + lane_in_tg;
        let mr0 = flat0 / 32u32;
        let mr1 = flat1 / 32u32;
        let mr2 = flat2 / 32u32;
        let mr3 = flat3 / 32u32;
        let mr4 = flat4 / 32u32;
        let mr5 = flat5 / 32u32;
        let mr6 = flat6 / 32u32;
        let mr7 = flat7 / 32u32;
        let kc0 = flat0 & 31u32;
        let kc1 = flat1 & 31u32;
        let kc2 = flat2 & 31u32;
        let kc3 = flat3 & 31u32;
        let kc4 = flat4 & 31u32;
        let kc5 = flat5 & 31u32;
        let kc6 = flat6 & 31u32;
        let kc7 = flat7 & 31u32;
        threadgroup_store(
            "xs",
            mr0 * xs_ld + kc0,
            load(x[(x_m_base + mr0) * k + kb + kc0]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr1 * xs_ld + kc1,
            load(x[(x_m_base + mr1) * k + kb + kc1]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr2 * xs_ld + kc2,
            load(x[(x_m_base + mr2) * k + kb + kc2]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr3 * xs_ld + kc3,
            load(x[(x_m_base + mr3) * k + kb + kc3]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr4 * xs_ld + kc4,
            load(x[(x_m_base + mr4) * k + kb + kc4]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr5 * xs_ld + kc5,
            load(x[(x_m_base + mr5) * k + kb + kc5]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr6 * xs_ld + kc6,
            load(x[(x_m_base + mr6) * k + kb + kc6]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr7 * xs_ld + kc7,
            load(x[(x_m_base + mr7) * k + kb + kc7]).cast::<T>(),
        );

        // ── 2. Coop W dequant — 64 lanes × 2 packs each → 1024 fp T ──
        // Pack 0: pack_idx = lane_in_tg ∈ 0..63 (first 64 packs).
        // Pack 1: pack_idx = 64 + lane_in_tg ∈ 64..127 (last 64 packs).
        // w_row = pack_idx / 4 ∈ 0..31, pack_in_row = pack_idx & 3 ∈ 0..3.
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;

        // Pack 0
        let pack_idx_0 = lane_in_tg;
        let w_row_0 = pack_idx_0 / 4u32;
        let pack_in_row_0 = pack_idx_0 & 3u32;
        let pack_0 = load(w[(w_n_base + w_row_0) * packs_per_row + kb / 8u32 + pack_in_row_0]);
        let k_off_0 = kb + pack_in_row_0 * 8u32;
        let g_0 = k_off_0 / 64u32;
        let sb_base_0 = (w_n_base + w_row_0) * gs_per_row;
        let s_0 = load(scales[sb_base_0 + g_0]).cast::<f32>();
        let b_0 = load(biases[sb_base_0 + g_0]).cast::<f32>();
        let pack_hi_0 = pack_0 >> 16u32;
        let q0_0 = (pack_0 & 15u32).cast::<f32>();
        let q1_0 = (pack_0 & 240u32).cast::<f32>() * s_16;
        let q2_0 = (pack_0 & 3840u32).cast::<f32>() * s_256;
        let q3_0 = (pack_0 & 61440u32).cast::<f32>() * s_4096;
        let q4_0 = (pack_hi_0 & 15u32).cast::<f32>();
        let q5_0 = (pack_hi_0 & 240u32).cast::<f32>() * s_16;
        let q6_0 = (pack_hi_0 & 3840u32).cast::<f32>() * s_256;
        let q7_0 = (pack_hi_0 & 61440u32).cast::<f32>() * s_4096;
        let ws_base_0 = w_row_0 * ws_ld + pack_in_row_0 * 8u32;
        threadgroup_store("ws", ws_base_0, (s_0 * q0_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 1u32, (s_0 * q1_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 2u32, (s_0 * q2_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 3u32, (s_0 * q3_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 4u32, (s_0 * q4_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 5u32, (s_0 * q5_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 6u32, (s_0 * q6_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 7u32, (s_0 * q7_0 + b_0).cast::<T>());

        // Pack 1
        let pack_idx_1 = 64u32 + lane_in_tg;
        let w_row_1 = pack_idx_1 / 4u32;
        let pack_in_row_1 = pack_idx_1 & 3u32;
        let pack_1 = load(w[(w_n_base + w_row_1) * packs_per_row + kb / 8u32 + pack_in_row_1]);
        let k_off_1 = kb + pack_in_row_1 * 8u32;
        let g_1 = k_off_1 / 64u32;
        let sb_base_1 = (w_n_base + w_row_1) * gs_per_row;
        let s_1 = load(scales[sb_base_1 + g_1]).cast::<f32>();
        let b_1 = load(biases[sb_base_1 + g_1]).cast::<f32>();
        let pack_hi_1 = pack_1 >> 16u32;
        let q0_1 = (pack_1 & 15u32).cast::<f32>();
        let q1_1 = (pack_1 & 240u32).cast::<f32>() * s_16;
        let q2_1 = (pack_1 & 3840u32).cast::<f32>() * s_256;
        let q3_1 = (pack_1 & 61440u32).cast::<f32>() * s_4096;
        let q4_1 = (pack_hi_1 & 15u32).cast::<f32>();
        let q5_1 = (pack_hi_1 & 240u32).cast::<f32>() * s_16;
        let q6_1 = (pack_hi_1 & 3840u32).cast::<f32>() * s_256;
        let q7_1 = (pack_hi_1 & 61440u32).cast::<f32>() * s_4096;
        let ws_base_1 = w_row_1 * ws_ld + pack_in_row_1 * 8u32;
        threadgroup_store("ws", ws_base_1, (s_1 * q0_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 1u32, (s_1 * q1_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 2u32, (s_1 * q2_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 3u32, (s_1 * q3_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 4u32, (s_1 * q4_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 5u32, (s_1 * q5_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 6u32, (s_1 * q6_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 7u32, (s_1 * q7_1 + b_1).cast::<T>());

        threadgroup_barrier();

        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        // sm = 0 fixed (WM=1 → both SGs cover full BM=16).
        // A-frag (X) elem[i] @ (fm, fn_i) = Xs[(0 + frag_m + fm) * 36 + (k_inner*8 + fn_i)]
        // B-frag (W^T) elem[i] @ (fm, fn_i) = Ws[(sn*16 + frag_n + fn_i) * 36 + (k_inner*8 + fm)]
        let row_a0 = sm * 16u32 + fm; // frag_m = 0
        let row_a1 = sm * 16u32 + 8u32 + fm; // frag_m = 8
        let col_b0 = sn * 16u32; // frag_n = 0 base offset
        let col_b1 = sn * 16u32 + 8u32; // frag_n = 8 base offset

        // k_inner = 0 (k offset 0..7 inside BK=32)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 1 (k offset 8..15)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 2 (k offset 16..23)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 3 (k offset 24..31)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        threadgroup_barrier();
    }

    // ── 4. Write 4 C frags to global out ──
    // sm = 0 → out_m_base starts at m_tile*16 (no sub-tile m offset).
    let out_m_base = m_tile * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    // c_f00 at (frag_m=0, frag_n=0)
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ─── mt_qmm_mma_int8 ────────────────────────────────────────────────────
//
// Int8-quantized simdgroup-matrix MMA prefill — mirrors `mt_qmm_mma` but
// operates on int8-packed weights (4 bytes per u32, `pack_factor=4`).
//
// Geometry (identical to mt_qmm_mma):
//   tpg = 128 = 4 SG × 32 lanes (WM=2, WN=2 warp grid)
//   BM = BN = BK = 32, output tile 32×32 (1024 outputs/TG)
//   Grid: [N/32, M/32, 1]
//   Each SG owns a 16×16 sub-tile = 2×2 = 4 8×8 frags
//   Per K-block per SG: 4 frags × 4 k-inner = 16 MMAs (64 across TG)
//
// W dequant lane mapping (changed vs int4):
//   Int4: 128 packs / K-block (32 rows × 4 packs/row), 1 pack per lane.
//   Int8: 256 packs / K-block (32 rows × 8 packs/row), 2 packs per lane.
//   pack_idx_N = N*128 + lane_in_tg, N ∈ {0, 1}.
//   w_row = pack_idx / 8 ∈ 0..31, pack_in_row = pack_idx % 8 ∈ 0..7.
//   Each pack dequants 4 bytes → 4 fp T values at
//   Ws[w_row*ws_ld + pack_in_row*4 + i], i ∈ 0..3.
//
// Scale/bias: group_size=64, packs_per_row = k/4 (4 elements/pack).
//   Group index per element-offset d: g = d / group_size = (kb + byte_offset) / 64.
//   pack_in_row ∈ 0..7 → 4 values at kb + pack_in_row*4 ∈ kb..kb+31.
//   All 4 values in a pack share the same group (4 < 64).
//
// TG memory layout: identical Xs[32×36] / Ws[32×36] shape (skew=4).
// X load: same as mt_qmm_mma (8 contiguous K elems per lane, vec4 fusion).
// MMA inner loop: identical to mt_qmm_mma (4 k-inner × 4 frags = 16 MMAs/SG).
#[bench_kernel(
    op="quantized",
    subop="qmm_mma_int8",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    m=32,
    group_size=64,
    tpg=128,
    bits=8,
    // int8 max_q=255 amplifies bf16 round-trip drift further than int4's 15.
    // At production shapes (M=4096+, K=4096+) bf16 cosine drifts ~8-9e-3.
    // tol=1e-2 keeps f32/f16 cells tight while passing bf16.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_8_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_mma_int8<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 4 SGs in 2×2 warp grid: sg ∈ {0,1,2,3} → (sm=sg/2, sn=sg%2).
    // Each SG owns a 16×16 sub-tile at (sm*16, sn*16) inside 32×32 tile.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;

    // 8×8 frag lane mapping (Apple steel_gemm layout — same as mt_qmm_mma).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // TG memory for X tile [BM × BK] and dequant W tile [BN × BK].
    // BK + 4 = 36 stride for bank-conflict avoidance — same skew rationale
    // as mt_qmm_mma. Xs size = 32 × 36 = 1152 T; Ws size = 32 × 36 = 1152 T.
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);

    // ── 4 output frags per SG, init to 0 ──
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);

    // A (X) and B (W^T) frag scratch, reused per k_inner.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();

    let x_m_base = m_tile * 32u32; // first M-row this TG handles
    let w_n_base = n_tile * 32u32; // first N-row this TG handles
    // Int8: 4 elements per pack → packs_per_row = k / 4.
    let packs_per_row = k / 4u32;

    // TG row stride = 36 (BK + 4 skew).
    let xs_ld_const = 36u32;
    let ws_ld_const = 36u32;
    let xs_ld = xs_ld_const;
    let ws_ld = ws_ld_const;

    // Coop X-load mapping. 32×32 = 1024 elements, 128 lanes × 8 elems each.
    // Same vec4-fusion layout as mt_qmm_mma: lane_in_tg/4 = m_row, lane_in_tg%4 = k_quad.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;

    // W dequant lane mapping for int8:
    // 256 packs per K-block (32 rows × 8 packs/row). 128 lanes → 2 packs/lane.
    // pack_idx_0 = lane_in_tg (packs 0..127), pack_idx_1 = 128 + lane_in_tg (packs 128..255).
    // w_row = pack_idx / 8 ∈ 0..31, pack_in_row = pack_idx % 8 ∈ 0..7.

    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);

        // ── 2. Coop W dequant (int8) — 128 lanes × 2 packs each → 1024 fp T ──
        // Pack 0: pack_idx = lane_in_tg ∈ 0..127 (first 128 packs).
        // Pack 1: pack_idx = 128 + lane_in_tg ∈ 128..255 (last 128 packs).
        // Each pack yields 4 values via (pack >> (i*8)) & 0xFF, i ∈ 0..3.

        // Pack 0
        let pack_idx_0 = lane_in_tg;
        let w_row_0 = pack_idx_0 / 8u32;
        let pack_in_row_0 = pack_idx_0 & 7u32;
        let pack_0 = load(w[(w_n_base + w_row_0) * packs_per_row + kb / 4u32 + pack_in_row_0]);
        let k_off_0 = kb + pack_in_row_0 * 4u32;
        let g_0 = k_off_0 / 64u32; // group_size = 64
        let sb_base_0 = (w_n_base + w_row_0) * gs_per_row;
        let s_0 = load(scales[sb_base_0 + g_0]).cast::<f32>();
        let b_0 = load(biases[sb_base_0 + g_0]).cast::<f32>();
        let q0_0 = (pack_0 & 255u32).cast::<f32>();
        let q1_0 = ((pack_0 >> 8u32) & 255u32).cast::<f32>();
        let q2_0 = ((pack_0 >> 16u32) & 255u32).cast::<f32>();
        let q3_0 = ((pack_0 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_0 = w_row_0 * ws_ld + pack_in_row_0 * 4u32;
        threadgroup_store("ws", ws_base_0, (s_0 * q0_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 1u32, (s_0 * q1_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 2u32, (s_0 * q2_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 3u32, (s_0 * q3_0 + b_0).cast::<T>());

        // Pack 1
        let pack_idx_1 = 128u32 + lane_in_tg;
        let w_row_1 = pack_idx_1 / 8u32;
        let pack_in_row_1 = pack_idx_1 & 7u32;
        let pack_1 = load(w[(w_n_base + w_row_1) * packs_per_row + kb / 4u32 + pack_in_row_1]);
        let k_off_1 = kb + pack_in_row_1 * 4u32;
        let g_1 = k_off_1 / 64u32;
        let sb_base_1 = (w_n_base + w_row_1) * gs_per_row;
        let s_1 = load(scales[sb_base_1 + g_1]).cast::<f32>();
        let b_1 = load(biases[sb_base_1 + g_1]).cast::<f32>();
        let q0_1 = (pack_1 & 255u32).cast::<f32>();
        let q1_1 = ((pack_1 >> 8u32) & 255u32).cast::<f32>();
        let q2_1 = ((pack_1 >> 16u32) & 255u32).cast::<f32>();
        let q3_1 = ((pack_1 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_1 = w_row_1 * ws_ld + pack_in_row_1 * 4u32;
        threadgroup_store("ws", ws_base_1, (s_1 * q0_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 1u32, (s_1 * q1_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 2u32, (s_1 * q2_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 3u32, (s_1 * q3_1 + b_1).cast::<T>());

        threadgroup_barrier();

        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        // Identical to mt_qmm_mma: A-load, barrier, B-load, barrier, MMAs,
        // barrier; serpentine MMA order (00→01→11→10).
        let row_a0 = sm * 16u32 + fm; // frag_m = 0
        let row_a1 = sm * 16u32 + 8u32 + fm; // frag_m = 8
        let col_b0 = sn * 16u32; // frag_n = 0 base offset
        let col_b1 = sn * 16u32 + 8u32; // frag_n = 8 base offset

        // k_inner = 0 (k offset 0..7 inside BK=32)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 1 (k offset 8..15)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 2 (k offset 16..23)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        // k_inner = 3 (k offset 24..31)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();

        threadgroup_barrier();
    }

    // ── 4. Write 4 C frags to global out ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    // c_f00 at (frag_m=0, frag_n=0)
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ─── mt_qmm_mma_m16_int8 ────────────────────────────────────────────────
//
// Int8 half-height MMA — the M=16 cell, int8 weights.
// Mirrors `mt_qmm_mma_m16` but with int8 pack format (4 bytes/u32).
//
// Geometry (identical to mt_qmm_mma_m16):
//   tpg = 64 = 2 SG × 32 lanes (WM=1, WN=2 warp grid)
//   BM = 16, BN = BK = 32 → 16×32 output tile (512 outputs/TG)
//   Grid: [N/32, M/16, 1]
//   Each SG owns a 16×16 sub-tile = 4 8×8 frags
//
// W dequant lane mapping (changed vs int4):
//   Int4: 128 packs / K-block, 64 lanes × 2 packs each.
//   Int8: 256 packs / K-block (32 rows × 8 packs/row), 64 lanes × 4 packs each.
//   For step N ∈ {0,1,2,3}: pack_idx = N*64 + lane_in_tg.
//   w_row = pack_idx / 8, pack_in_row = pack_idx % 8.
//   Each pack dequants 4 bytes → Ws[w_row*ws_ld + pack_in_row*4 + i].
//
// X load: mt_qmm_mma_m16 flat-stride pattern (8 strides of 64 lanes covering
// the 512-element Xs tile). Same as int4.
//
// MMA inner loop: identical to mt_qmm_mma_m16 (no A/B barrier; 4 k-inner).
#[bench_kernel(
    op="quantized",
    subop="qmm_mma_m16_int8",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    m=16,
    group_size=64,
    tpg=64,
    bits=8,
    // Same bf16 tolerance rationale as mt_qmm_mma_int8.
    tol=1e-2,
    mlx="affine_qmm_t_{tn}_gs_64_b_8_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16, metaltile_core::dtype::DType::BF16],
)]
#[kernel]
pub fn mt_qmm_mma_m16_int8<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 2 SGs in 1×2 warp grid: sg ∈ {0,1} → (sm=0, sn=sg).
    let sm = 0u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;

    // 8×8 frag lane mapping (Apple steel_gemm layout — same as mt_qmm_mma).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // TG memory: Xs[16 × 36] = 576 T; Ws[32 × 36] = 1152 T.
    threadgroup_alloc("xs", 576, T);
    threadgroup_alloc("ws", 1152, T);

    // ── 4 output frags per SG, init to 0 ──
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);

    // A (X) and B (W^T) frag scratch, reused per k_inner.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();

    let x_m_base = m_tile * 16u32; // first M-row this TG handles
    let w_n_base = n_tile * 32u32; // first N-row this TG handles
    // Int8: 4 elements per pack → packs_per_row = k / 4.
    let packs_per_row = k / 4u32;

    // TG row stride = 36 (BK + 4 skew).
    let xs_ld = 36u32;
    let ws_ld = 36u32;

    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 64 lanes × 8 strides each fill 512-elt tile ──
        let flat0 = lane_in_tg;
        let flat1 = 64u32 + lane_in_tg;
        let flat2 = 128u32 + lane_in_tg;
        let flat3 = 192u32 + lane_in_tg;
        let flat4 = 256u32 + lane_in_tg;
        let flat5 = 320u32 + lane_in_tg;
        let flat6 = 384u32 + lane_in_tg;
        let flat7 = 448u32 + lane_in_tg;
        let mr0 = flat0 / 32u32;
        let mr1 = flat1 / 32u32;
        let mr2 = flat2 / 32u32;
        let mr3 = flat3 / 32u32;
        let mr4 = flat4 / 32u32;
        let mr5 = flat5 / 32u32;
        let mr6 = flat6 / 32u32;
        let mr7 = flat7 / 32u32;
        let kc0 = flat0 & 31u32;
        let kc1 = flat1 & 31u32;
        let kc2 = flat2 & 31u32;
        let kc3 = flat3 & 31u32;
        let kc4 = flat4 & 31u32;
        let kc5 = flat5 & 31u32;
        let kc6 = flat6 & 31u32;
        let kc7 = flat7 & 31u32;
        threadgroup_store(
            "xs",
            mr0 * xs_ld + kc0,
            load(x[(x_m_base + mr0) * k + kb + kc0]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr1 * xs_ld + kc1,
            load(x[(x_m_base + mr1) * k + kb + kc1]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr2 * xs_ld + kc2,
            load(x[(x_m_base + mr2) * k + kb + kc2]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr3 * xs_ld + kc3,
            load(x[(x_m_base + mr3) * k + kb + kc3]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr4 * xs_ld + kc4,
            load(x[(x_m_base + mr4) * k + kb + kc4]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr5 * xs_ld + kc5,
            load(x[(x_m_base + mr5) * k + kb + kc5]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr6 * xs_ld + kc6,
            load(x[(x_m_base + mr6) * k + kb + kc6]).cast::<T>(),
        );
        threadgroup_store(
            "xs",
            mr7 * xs_ld + kc7,
            load(x[(x_m_base + mr7) * k + kb + kc7]).cast::<T>(),
        );

        // ── 2. Coop W dequant (int8) — 64 lanes × 4 packs each → 1024 fp T ──
        // 256 packs per K-block. 64 lanes × 4 packs = 256. ✓
        // Step N ∈ {0,1,2,3}: pack_idx = N*64 + lane_in_tg.
        // w_row = pack_idx / 8, pack_in_row = pack_idx % 8.

        // Pack step 0
        let pack_idx_0 = lane_in_tg;
        let w_row_0 = pack_idx_0 / 8u32;
        let pack_in_row_0 = pack_idx_0 & 7u32;
        let pack_0 = load(w[(w_n_base + w_row_0) * packs_per_row + kb / 4u32 + pack_in_row_0]);
        let k_off_0 = kb + pack_in_row_0 * 4u32;
        let g_0 = k_off_0 / 64u32;
        let sb_base_0 = (w_n_base + w_row_0) * gs_per_row;
        let s_0 = load(scales[sb_base_0 + g_0]).cast::<f32>();
        let b_0 = load(biases[sb_base_0 + g_0]).cast::<f32>();
        let q0_0 = (pack_0 & 255u32).cast::<f32>();
        let q1_0 = ((pack_0 >> 8u32) & 255u32).cast::<f32>();
        let q2_0 = ((pack_0 >> 16u32) & 255u32).cast::<f32>();
        let q3_0 = ((pack_0 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_0 = w_row_0 * ws_ld + pack_in_row_0 * 4u32;
        threadgroup_store("ws", ws_base_0, (s_0 * q0_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 1u32, (s_0 * q1_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 2u32, (s_0 * q2_0 + b_0).cast::<T>());
        threadgroup_store("ws", ws_base_0 + 3u32, (s_0 * q3_0 + b_0).cast::<T>());

        // Pack step 1
        let pack_idx_1 = 64u32 + lane_in_tg;
        let w_row_1 = pack_idx_1 / 8u32;
        let pack_in_row_1 = pack_idx_1 & 7u32;
        let pack_1 = load(w[(w_n_base + w_row_1) * packs_per_row + kb / 4u32 + pack_in_row_1]);
        let k_off_1 = kb + pack_in_row_1 * 4u32;
        let g_1 = k_off_1 / 64u32;
        let sb_base_1 = (w_n_base + w_row_1) * gs_per_row;
        let s_1 = load(scales[sb_base_1 + g_1]).cast::<f32>();
        let b_1 = load(biases[sb_base_1 + g_1]).cast::<f32>();
        let q0_1 = (pack_1 & 255u32).cast::<f32>();
        let q1_1 = ((pack_1 >> 8u32) & 255u32).cast::<f32>();
        let q2_1 = ((pack_1 >> 16u32) & 255u32).cast::<f32>();
        let q3_1 = ((pack_1 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_1 = w_row_1 * ws_ld + pack_in_row_1 * 4u32;
        threadgroup_store("ws", ws_base_1, (s_1 * q0_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 1u32, (s_1 * q1_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 2u32, (s_1 * q2_1 + b_1).cast::<T>());
        threadgroup_store("ws", ws_base_1 + 3u32, (s_1 * q3_1 + b_1).cast::<T>());

        // Pack step 2
        let pack_idx_2 = 128u32 + lane_in_tg;
        let w_row_2 = pack_idx_2 / 8u32;
        let pack_in_row_2 = pack_idx_2 & 7u32;
        let pack_2 = load(w[(w_n_base + w_row_2) * packs_per_row + kb / 4u32 + pack_in_row_2]);
        let k_off_2 = kb + pack_in_row_2 * 4u32;
        let g_2 = k_off_2 / 64u32;
        let sb_base_2 = (w_n_base + w_row_2) * gs_per_row;
        let s_2 = load(scales[sb_base_2 + g_2]).cast::<f32>();
        let b_2 = load(biases[sb_base_2 + g_2]).cast::<f32>();
        let q0_2 = (pack_2 & 255u32).cast::<f32>();
        let q1_2 = ((pack_2 >> 8u32) & 255u32).cast::<f32>();
        let q2_2 = ((pack_2 >> 16u32) & 255u32).cast::<f32>();
        let q3_2 = ((pack_2 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_2 = w_row_2 * ws_ld + pack_in_row_2 * 4u32;
        threadgroup_store("ws", ws_base_2, (s_2 * q0_2 + b_2).cast::<T>());
        threadgroup_store("ws", ws_base_2 + 1u32, (s_2 * q1_2 + b_2).cast::<T>());
        threadgroup_store("ws", ws_base_2 + 2u32, (s_2 * q2_2 + b_2).cast::<T>());
        threadgroup_store("ws", ws_base_2 + 3u32, (s_2 * q3_2 + b_2).cast::<T>());

        // Pack step 3
        let pack_idx_3 = 192u32 + lane_in_tg;
        let w_row_3 = pack_idx_3 / 8u32;
        let pack_in_row_3 = pack_idx_3 & 7u32;
        let pack_3 = load(w[(w_n_base + w_row_3) * packs_per_row + kb / 4u32 + pack_in_row_3]);
        let k_off_3 = kb + pack_in_row_3 * 4u32;
        let g_3 = k_off_3 / 64u32;
        let sb_base_3 = (w_n_base + w_row_3) * gs_per_row;
        let s_3 = load(scales[sb_base_3 + g_3]).cast::<f32>();
        let b_3 = load(biases[sb_base_3 + g_3]).cast::<f32>();
        let q0_3 = (pack_3 & 255u32).cast::<f32>();
        let q1_3 = ((pack_3 >> 8u32) & 255u32).cast::<f32>();
        let q2_3 = ((pack_3 >> 16u32) & 255u32).cast::<f32>();
        let q3_3 = ((pack_3 >> 24u32) & 255u32).cast::<f32>();
        let ws_base_3 = w_row_3 * ws_ld + pack_in_row_3 * 4u32;
        threadgroup_store("ws", ws_base_3, (s_3 * q0_3 + b_3).cast::<T>());
        threadgroup_store("ws", ws_base_3 + 1u32, (s_3 * q1_3 + b_3).cast::<T>());
        threadgroup_store("ws", ws_base_3 + 2u32, (s_3 * q2_3 + b_3).cast::<T>());
        threadgroup_store("ws", ws_base_3 + 3u32, (s_3 * q3_3 + b_3).cast::<T>());

        threadgroup_barrier();

        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        // sm = 0 fixed (WM=1 → both SGs cover full BM=16).
        let row_a0 = sm * 16u32 + fm; // frag_m = 0
        let row_a1 = sm * 16u32 + 8u32 + fm; // frag_m = 8
        let col_b0 = sn * 16u32; // frag_n = 0 base offset
        let col_b1 = sn * 16u32 + 8u32; // frag_n = 8 base offset

        // k_inner = 0 (k offset 0..7 inside BK=32)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 1 (k offset 8..15)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 2 (k offset 16..23)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        // k_inner = 3 (k offset 24..31)
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_matmul(a_f1, b_f1, c_f11);

        threadgroup_barrier();
    }

    // ── 4. Write 4 C frags to global out ──
    // sm = 0 → out_m_base starts at m_tile*16 (no sub-tile m offset).
    let out_m_base = m_tile * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    // c_f00 at (frag_m=0, frag_n=0)
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ─── mt_affine_dequantize_int4 ─────────────────────────────────────────
//
// One thread per pack (8 nibbles in one uint32). For each output i in
// 0..8: `q = (val >> (i*4)) & 0xf`, then `out[oindex+i] = scale * q + bias`
// where scale/bias are looked up by group index `oindex / group_size`.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 4>` from
// `quantized.h`. Both kernels read the same byte stream and produce the
// same output (MLX views weights as `uint8_t*`, ours as `Tensor<u32>` —
// same bits, different lens).
#[bench_kernel(
    op="affine",
    subop="dequantize_int4",
    class=AffineDequantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-2 — bf16 round-trip error scales with max_q (= 15). At
    // n_groups=4096 the worst-case absolute drift is ~3e-3.
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int4<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 15u32;
    let q1 = (val >> 4u32) & 15u32;
    let q2 = (val >> 8u32) & 15u32;
    let q3 = (val >> 12u32) & 15u32;
    let q4 = (val >> 16u32) & 15u32;
    let q5 = (val >> 20u32) & 15u32;
    let q6 = (val >> 24u32) & 15u32;
    let q7 = (val >> 28u32) & 15u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int4 ───────────────────────────────────────────
//
// Inverse of dequantize: one threadgroup per group, finds min/max over
// the group, computes scale/bias, then packs 8 nibbles per uint32. The
// per-group nature means no cross-threadgroup sync is needed.
//
// MLX's `affine_quantize` uses a 32-thread simd-group cooperative reduce
// across `group_size` elements; we use the same shape (one threadgroup
// of 32 threads per group) and reduce via `simd_min` / `simd_max`.
//
// Packing: `packs_per_group = group_size / pack_factor = 64 / 8 = 8`
// nibble-packs per group. Lanes 0..7 each pack one uint32 in parallel
// — they re-read the 8 input values for their pack from device memory
// (cheap; the data is already cached after the min/max reduction's
// first load). Eliminating the lane-0 serial loop is the main perf
// difference vs the original implementation.
//
// Restriction: hardcodes group_size=64 and bits=4 in the unrolling
// (`group_size / 32 = 2` values per thread, 8 nibbles per uint32).
// Bigger group sizes or other bit widths follow the same template with
// different constants.
#[bench_kernel(
    op="affine",
    subop="quantize_int4",
    class=AffineQuantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int4<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 15.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());
    }

    // Packs in parallel: lanes 0..packs_per_group each pack one uint32.
    // For group_size=64 → packs_per_group=8, so 8 lanes work in parallel
    // vs the previous lane-0 serial loop over all 8 packs.
    let packs_per_group = group_size / 8u32;
    if lane < packs_per_group {
        let pack_in_base = in_base + lane * 8u32;
        let mut acc = 0u32;
        for k in range(0u32, 8u32, 1u32) {
            let v = load(w[pack_in_base + k]).cast::<f32>();
            let q_f = (v - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 15.0f32, 15.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>();
            acc = acc | (q << (k * 4u32));
        }
        store(out[g_idx * packs_per_group + lane], acc);
    }
}

// ─── mt_affine_dequantize_int8 ─────────────────────────────────────────
//
// One thread per pack (4 bytes in one uint32). Same shape as int4 but
// each pack covers 4 output values instead of 8, and bit-extraction
// shifts by multiples of 8 instead of 4.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 8>`.
#[bench_kernel(
    op="affine",
    subop="dequantize_int8",
    class=AffineDequantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-1 — int8 max_q=255 amplifies bf16 round-trip drift; the
    // worst case at n_groups=4096 is ~5e-2.
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int8<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 255u32;
    let q1 = (val >> 8u32) & 255u32;
    let q2 = (val >> 16u32) & 255u32;
    let q3 = (val >> 24u32) & 255u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int8 ───────────────────────────────────────────
#[bench_kernel(
    op="affine",
    subop="quantize_int8",
    class=AffineQuantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int8<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 255.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());
    }

    // Packs in parallel: lanes 0..packs_per_group each pack one uint32.
    // For group_size=64, pack_factor=4 → packs_per_group=16, so 16
    // lanes pack in parallel vs the previous lane-0 serial loop.
    let packs_per_group = group_size / 4u32;
    if lane < packs_per_group {
        let pack_in_base = in_base + lane * 4u32;
        let mut acc = 0u32;
        for k in range(0u32, 4u32, 1u32) {
            let v = load(w[pack_in_base + k]).cast::<f32>();
            let q_f = (v - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 255.0f32, 255.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>();
            acc = acc | (q << (k * 8u32));
        }
        store(out[g_idx * packs_per_group + lane], acc);
    }
}

// ─── mt_affine_dequantize_int2 ─────────────────────────────────────────
//
// One thread per pack (16 two-bit values in one uint32). `bits=2` packs
// cleanly into a uint32 (16 values, no byte-stream crossing), so this
// follows the int4 / int8 power-of-2 template with a 16-way unroll and
// 2-bit shifts. For each output i in 0..16: `q = (val >> (i*2)) & 0x3`,
// then `out[oindex+i] = scale * q + bias`.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 2>` from
// `quantized.h`.
#[bench_kernel(
    op="affine",
    subop="dequantize_int2",
    class=AffineDequantize,
    bits=2,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=5e-3 — int2 max_q=3; tightest of the dequant family, the
    // worst-case bf16 round-trip drift at n_groups=4096 is ~1e-3.
    tol=5e-3,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int2<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 16u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    // 16 two-bit lanes; each is `(val >> (i*2)) & 0x3`.
    for k in range(0u32, 16u32, 1u32) {
        let q = (val >> (k * 2u32)) & 3u32;
        store(out[oindex + k], (scale * q.cast::<f32>() + bias).cast::<T>());
    }
}

// ─── mt_affine_quantize_int2 ───────────────────────────────────────────
//
// Inverse of `mt_affine_dequantize_int2`. One threadgroup of 32 threads
// per group: each lane covers `group_size / 32 = 2` input values, the
// min/max reduce runs over the simdgroup, then `packs_per_group =
// group_size / 16` lanes each assemble one uint32 (16 two-bit codes).
//
// n_bins = 3 (`2^2 - 1`). Same template as `mt_affine_quantize_int4`
// with the pack width widened from 8 nibbles to 16 two-bit fields.
#[bench_kernel(
    op="affine",
    subop="quantize_int2",
    class=AffineQuantize,
    bits=2,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int2<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 3.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());
    }

    // Packs in parallel: `packs_per_group = group_size / 16` lanes each
    // assemble one uint32 (16 two-bit codes). For group_size=64 →
    // packs_per_group=4.
    let packs_per_group = group_size / 16u32;
    if lane < packs_per_group {
        let pack_in_base = in_base + lane * 16u32;
        let mut acc = 0u32;
        for k in range(0u32, 16u32, 1u32) {
            let v = load(w[pack_in_base + k]).cast::<f32>();
            let q_f = (v - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 3.0f32, 3.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>();
            acc = acc | (q << (k * 2u32));
        }
        store(out[g_idx * packs_per_group + lane], acc);
    }
}

// ─── Byte-stream quantize variants (int3 / int5 / int6) ─────────────
//
// Non-power-of-2 bit widths pack into a contiguous byte stream. Each
// group of `group_size` values writes `group_size * bits / 8` bytes.
//
// For the canonical `group_size=32`:
//   int3: 32 x 3 bits =  96 bits = 12 bytes = 3 uint32 words per group.
//   int5: 32 x 5 bits = 160 bits = 20 bytes = 5 uint32 words per group.
//   int6: 32 x 6 bits = 192 bits = 24 bytes = 6 uint32 words per group.
//
// Because adjacent packs may share a uint32 word, parallel packing
// requires atomics. Instead, one threadgroup of 32 threads per group
// performs the simd min/max cooperatively, then lane 0 writes the bit
// stream serially (iterating over all group_size elements, ORing each
// code's bits into the correct uint32 word at the right shift).
//
// Bit layouts are the exact inverse of `mt_affine_dequantize_int{3,5,6}`.

// ─── mt_affine_quantize_int3 ──────────────────────────────────────────
//
// int3 (3-bit codes): 96-bit stream for group_size=32.
// Lane 0 iterates over all 32 values, computes q = clamp(round(v), 0, 7),
// then ORs q's 3 bits into the three uint32 output words.
//
//   bit_pos   = i * 3  (position in the 96-bit stream)
//   word_idx  = bit_pos / 32
//   bit_shift = bit_pos % 32
//
// Codes span at most two words (3-bit code starting at bit_shift > 29).
//
// ## DISPATCH INVARIANTS
// - Reduction mode (simd_min / simd_max). TPG = 32 (one simdgroup).
// - Grid: [n_groups, 1, 1].
#[bench_kernel(
    op="affine",
    subop="quantize_int3",
    class=AffineQuantize,
    bits=3,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int3<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    // Cooperative min/max over the simdgroup (one value per lane).
    let v = load(w[in_base + lane]).cast::<f32>();
    let w_min = simd_min(v);
    let w_max = simd_max(v);

    let n_bins = 7.0f32; // 2^3 - 1
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());

        // For group_size=32: 3 uint32 output words (12 bytes).
        let out_base = g_idx * 3u32;
        let mut w0 = 0u32;
        let mut w1 = 0u32;
        let mut w2 = 0u32;

        for i in range(0u32, group_size, 1u32) {
            let vi = load(w[in_base + i]).cast::<f32>();
            let q_f = (vi - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 7.0f32, 7.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>() & 7u32;

            let bit_pos = i * 3u32;
            let word_idx = bit_pos / 32u32;
            let bit_shift = bit_pos & 31u32;

            let q_lo = q << bit_shift;
            w0 = select(word_idx == 0u32, w0 | q_lo, w0);
            w1 = select(word_idx == 1u32, w1 | q_lo, w1);
            w2 = select(word_idx == 2u32, w2 | q_lo, w2);

            // Handle spillover into the next word (occurs when bit_shift > 29).
            let spills = bit_shift + 3u32 > 32u32;
            if spills {
                let bits_hi = (bit_shift + 3u32) - 32u32;
                let q_hi = q >> (3u32 - bits_hi);
                w1 = select(word_idx == 0u32, w1 | q_hi, w1);
                w2 = select(word_idx == 1u32, w2 | q_hi, w2);
            }
        }

        store(out[out_base + 0u32], w0);
        store(out[out_base + 1u32], w1);
        store(out[out_base + 2u32], w2);
    }
}

// ─── mt_affine_quantize_int5 ──────────────────────────────────────────
//
// int5 (5-bit codes): 160-bit stream for group_size=32 (5 uint32 words).
// Same bit-stream OR strategy as int3 but with 5 output words.
//
// ## DISPATCH INVARIANTS
// - Reduction mode (simd_min / simd_max). TPG = 32 (one simdgroup).
// - Grid: [n_groups, 1, 1].
#[bench_kernel(
    op="affine",
    subop="quantize_int5",
    class=AffineQuantize,
    bits=5,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int5<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v = load(w[in_base + lane]).cast::<f32>();
    let w_min = simd_min(v);
    let w_max = simd_max(v);

    let n_bins = 31.0f32; // 2^5 - 1
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());

        // For group_size=32: 5 uint32 output words (20 bytes).
        let out_base = g_idx * 5u32;
        let mut w0 = 0u32;
        let mut w1 = 0u32;
        let mut w2 = 0u32;
        let mut w3 = 0u32;
        let mut w4 = 0u32;

        for i in range(0u32, group_size, 1u32) {
            let vi = load(w[in_base + i]).cast::<f32>();
            let q_f = (vi - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 31.0f32, 31.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>() & 31u32;

            let bit_pos = i * 5u32;
            let word_idx = bit_pos / 32u32;
            let bit_shift = bit_pos & 31u32;

            let q_lo = q << bit_shift;
            w0 = select(word_idx == 0u32, w0 | q_lo, w0);
            w1 = select(word_idx == 1u32, w1 | q_lo, w1);
            w2 = select(word_idx == 2u32, w2 | q_lo, w2);
            w3 = select(word_idx == 3u32, w3 | q_lo, w3);
            w4 = select(word_idx == 4u32, w4 | q_lo, w4);

            let spills = bit_shift + 5u32 > 32u32;
            if spills {
                let bits_hi = (bit_shift + 5u32) - 32u32;
                let q_hi = q >> (5u32 - bits_hi);
                w1 = select(word_idx == 0u32, w1 | q_hi, w1);
                w2 = select(word_idx == 1u32, w2 | q_hi, w2);
                w3 = select(word_idx == 2u32, w3 | q_hi, w3);
                w4 = select(word_idx == 3u32, w4 | q_hi, w4);
            }
        }

        store(out[out_base + 0u32], w0);
        store(out[out_base + 1u32], w1);
        store(out[out_base + 2u32], w2);
        store(out[out_base + 3u32], w3);
        store(out[out_base + 4u32], w4);
    }
}

// ─── mt_affine_quantize_int6 ──────────────────────────────────────────
//
// int6 (6-bit codes): 192-bit stream for group_size=32 (6 uint32 words).
// Same bit-stream OR strategy as int3/int5 but with 6 output words.
//
// ## DISPATCH INVARIANTS
// - Reduction mode (simd_min / simd_max). TPG = 32 (one simdgroup).
// - Grid: [n_groups, 1, 1].
#[bench_kernel(
    op="affine",
    subop="quantize_int6",
    class=AffineQuantize,
    bits=6,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int6<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v = load(w[in_base + lane]).cast::<f32>();
    let w_min = simd_min(v);
    let w_max = simd_max(v);

    let n_bins = 63.0f32; // 2^6 - 1
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());

        // For group_size=32: 6 uint32 output words (24 bytes).
        let out_base = g_idx * 6u32;
        let mut w0 = 0u32;
        let mut w1 = 0u32;
        let mut w2 = 0u32;
        let mut w3 = 0u32;
        let mut w4 = 0u32;
        let mut w5 = 0u32;

        for i in range(0u32, group_size, 1u32) {
            let vi = load(w[in_base + i]).cast::<f32>();
            let q_f = (vi - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 63.0f32, 63.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>() & 63u32;

            let bit_pos = i * 6u32;
            let word_idx = bit_pos / 32u32;
            let bit_shift = bit_pos & 31u32;

            let q_lo = q << bit_shift;
            w0 = select(word_idx == 0u32, w0 | q_lo, w0);
            w1 = select(word_idx == 1u32, w1 | q_lo, w1);
            w2 = select(word_idx == 2u32, w2 | q_lo, w2);
            w3 = select(word_idx == 3u32, w3 | q_lo, w3);
            w4 = select(word_idx == 4u32, w4 | q_lo, w4);
            w5 = select(word_idx == 5u32, w5 | q_lo, w5);

            let spills = bit_shift + 6u32 > 32u32;
            if spills {
                let bits_hi = (bit_shift + 6u32) - 32u32;
                let q_hi = q >> (6u32 - bits_hi);
                w1 = select(word_idx == 0u32, w1 | q_hi, w1);
                w2 = select(word_idx == 1u32, w2 | q_hi, w2);
                w3 = select(word_idx == 2u32, w3 | q_hi, w3);
                w4 = select(word_idx == 3u32, w4 | q_hi, w4);
                w5 = select(word_idx == 4u32, w5 | q_hi, w5);
            }
        }

        store(out[out_base + 0u32], w0);
        store(out[out_base + 1u32], w1);
        store(out[out_base + 2u32], w2);
        store(out[out_base + 3u32], w3);
        store(out[out_base + 4u32], w4);
        store(out[out_base + 5u32], w5);
    }
}

// ─── Byte-stream dequant variants (int3 / int5 / int6) ───────────────
//
// Non-power-of-2 bit widths can't pack cleanly into a uint32, so each
// pack spans `bytes_per_pack` bytes that may cross a uint32 boundary.
// The runner allocates a one-uint32 sentinel past the end so the always-
// on `w[u_idx0 + 1]` load is safe even for the last pack.
//
// Bit layouts match MLX `affine_dequantize<T, group_size, {3,5,6}>`
// exactly.

#[bench_kernel(
    op="affine",
    subop="dequantize_int3",
    class=AffineDequantize,
    bits=3,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-3 — int3 max_q=7; worst-case bf16 drift at n_groups=4096
    // is ~1e-3.
    tol=5e-3,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int3<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 7u32;
    let q1 = (b0 >> 3u32) & 7u32;
    let q2 = ((b0 >> 6u32) & 3u32) | ((b1 & 1u32) << 2u32);
    let q3 = (b1 >> 1u32) & 7u32;
    let q4 = (b1 >> 4u32) & 7u32;
    let q5 = ((b1 >> 7u32) & 1u32) | ((b2 & 3u32) << 1u32);
    let q6 = (b2 >> 2u32) & 7u32;
    let q7 = (b2 >> 5u32) & 7u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int5",
    class=AffineDequantize,
    bits=5,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int5<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 5u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

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

    let q0 = b0 & 31u32;
    let q1 = ((b0 >> 5u32) & 7u32) | ((b1 & 3u32) << 3u32);
    let q2 = (b1 >> 2u32) & 31u32;
    let q3 = ((b1 >> 7u32) & 1u32) | ((b2 & 15u32) << 1u32);
    let q4 = ((b2 >> 4u32) & 15u32) | ((b3 & 1u32) << 4u32);
    let q5 = (b3 >> 1u32) & 31u32;
    let q6 = ((b3 >> 6u32) & 3u32) | ((b4 & 7u32) << 2u32);
    let q7 = (b4 >> 3u32) & 31u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int6",
    class=AffineDequantize,
    bits=6,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-2 — int6 max_q=63; worst-case bf16 drift at n_groups=4096
    // is ~1.3e-2.
    tol=5e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int6<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 63u32;
    let q1 = ((b0 >> 6u32) & 3u32) | ((b1 & 15u32) << 2u32);
    let q2 = ((b1 >> 4u32) & 15u32) | ((b2 & 3u32) << 4u32);
    let q3 = (b2 >> 2u32) & 63u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-bit-width quantized matvec / vecmat / matmul family
// ═══════════════════════════════════════════════════════════════════════
//
// The hand-unrolled `mt_qmv` / `mt_qmm*` above are int4-only, f32+f16 —
// the production hot path. This section closes the rest of the
// `affine_qmv / qvm / qmm` coverage gap with a clean, generic family:
//
//   - `mt_qmv_b{3,4,5,6,8}`  — quantized matvec, `y = W · x`
//   - `mt_qvm_b{3,4,5,6,8}`  — quantized vecmat, `y = xᵀ · W`
//   - `mt_qmm_b{3,4,5,6,8}`  — quantized matmul (batched matvec)
//
// Every kernel is generic over `T` (so **bf16** flows through the same
// body — closing the bf16 gap) and parameterised on bit-width via an
// outer `macro_rules!` (the whole `#[kernel] fn` is macro-expanded
// before the proc-macro runs — never an inner-body macro; see
// `dequant_gemv.rs` and the empty-body hazard in `docs/developing.md`).
//
// These are correctness-first scalar kernels — one threadgroup per
// output element, lanes stride the K dimension, `simd_sum` reduces.
// They are not the perf path (the unrolled int4 `mt_qmv`/`mt_qmm*`
// remain that, and the NAX/MMA qmm variant is upstream PR #137); they
// exist so every MLX `affine_qmv/qvm/qmm` bit-width × dtype cell has a
// metaltile kernel + a GPU correctness test behind it.
//
// ── Bit-extraction ──
// Power-of-two widths (4, 8) divide a u32 evenly: element `e` of a row
// lives in pack `e / (32/bits)`, shifted `(e % (32/bits)) * bits` — the
// `*_pow2!` macros. Odd widths (3, 5, 6) use the two-word bit-stream
// formula from `dequant_gather.rs` — a code may straddle a u32
// boundary, so each element reads up to two consecutive words — the
// `*_odd!` macros. Splitting pow2 vs odd into separate macros (rather
// than a runtime branch) keeps the extraction a straight-line body:
// the DSL's `if` is a statement, not an expression.
//
// ── Layouts (N = out_dim, K = in_dim, G = group_size) ──
//   qmv / qmm  W [N, K]  packed row-major (groups along K)
//              scales/biases [N, K/G]
//   qvm        W [K, N]  packed row-major (groups along K)
//              scales/biases [K/G, N]
//
// ## DISPATCH INVARIANTS (all kernels in this family)
//
// - **Mode: Reduction.** `simd_sum` reduces the per-lane partial dot.
// - **TG: `[32, 1, 1]`** — exactly one simdgroup. Fewer than 32
//   threads would make the `simd_sum` reduce a partial set; the loop
//   strides by 32, matching.
// - **qmv  Grid: `[N, 1, 1]`** — one TG per output row.
// - **qvm  Grid: `[N, 1, 1]`** — one TG per output column.
// - **qmm  Grid: `[N, M, 1]`** — one TG per (output col, batch row).
// - **`K` must be a multiple of 32** and **`G` must divide `K`**.
//   Every Qwen3 / Qwen3.6 quantized shape satisfies both.

/// `BenchSpec` for a kernel in the multi-bit qmv/qvm/qmm family.
macro_rules! quantized_family_spec {
    ($name:ident, $subop:literal) => {
        inventory::submit! {
            crate::spec::BenchSpec {
                op: "quantized",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[
                    metaltile_core::dtype::DType::F32,
                    metaltile_core::dtype::DType::F16,
                    metaltile_core::dtype::DType::BF16,
                ],
                tol: 5e-2, // int-quant — wide tolerance vs full-precision oracle
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: crate::spec::BenchDispatch::Generic,
                kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
            }
        }
    };
}

/// Quantized matvec / matmul (`y = W · x`) — pow2 bit-widths (4, 8).
/// `mt_qmm_b*` is the M-batched form; `mt_qmv_b*` its M=1 row. W is
/// `[N, K]` row-major; element `(row, d)` lives in a pack-aligned u32.
macro_rules! qmv_pow2 {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] group_size: u32,
        ) {
            // tgid_x = output row, tgid_y = batch row (M). At M=1 the
            // grid's y extent is 1, so this is the plain matvec.
            let row = tgid_x;
            let m_row = tgid_y;
            let lane = simd_lane;

            let groups_per_row = k / group_size;
            let scale_row_base = row * groups_per_row;
            let x_row_base = m_row * k;

            let vals_per_pack = 32u32 / $bits;
            let packs_per_row = k / vals_per_pack;
            let mask = (1u32 << $bits) - 1u32;

            // Each lane owns K-positions lane, lane+32, lane+64, ...
            let mut acc = 0.0f32;
            let n_iters = (k + 31u32) / 32u32;
            for _it in range(0u32, n_iters, 1u32) {
                let d = _it * 32u32 + lane;
                if d < k {
                    let g = d / group_size;
                    let scale = load(scales[scale_row_base + g]).cast::<f32>();
                    let bias = load(biases[scale_row_base + g]).cast::<f32>();

                    // Pack-aligned int-$bits weight code at (row, d).
                    let pack = d / vals_per_pack;
                    let slot = d - pack * vals_per_pack;
                    let word = load(w[row * packs_per_row + pack]);
                    let q = (word >> (slot * $bits)) & mask;

                    let wv = q.cast::<f32>() * scale + bias;
                    acc = acc + wv * load(x[x_row_base + d]).cast::<f32>();
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[m_row * n + row], total.cast::<T>());
            }
        }
        quantized_family_spec!($name, $subop);
    };
}

/// Quantized matvec / matmul (`y = W · x`) — odd bit-widths (3, 5, 6).
/// W is `[N, K]` bit-stream-packed; element `(row, d)` may straddle two
/// consecutive u32 words.
macro_rules! qmv_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] group_size: u32,
        ) {
            let row = tgid_x;
            let m_row = tgid_y;
            let lane = simd_lane;

            let groups_per_row = k / group_size;
            let scale_row_base = row * groups_per_row;
            let x_row_base = m_row * k;

            let u32_per_row = k * $bits / 32u32;
            let row_u32_off = row * u32_per_row;

            let mut acc = 0.0f32;
            let n_iters = (k + 31u32) / 32u32;
            for _it in range(0u32, n_iters, 1u32) {
                let d = _it * 32u32 + lane;
                if d < k {
                    let g = d / group_size;
                    let scale = load(scales[scale_row_base + g]).cast::<f32>();
                    let bias = load(biases[scale_row_base + g]).cast::<f32>();

                    // Two-word bit-stream extract — code may straddle a
                    // u32 boundary (`spill` bits land in the next word).
                    let bit_off = d * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[row_u32_off + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(w[row_u32_off + w1idx]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;

                    let wv = q.cast::<f32>() * scale + bias;
                    acc = acc + wv * load(x[x_row_base + d]).cast::<f32>();
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[m_row * n + row], total.cast::<T>());
            }
        }
        quantized_family_spec!($name, $subop);
    };
}

/// Quantized vecmat (`y = xᵀ · W`) — pow2 bit-widths. W is `[K, N]`
/// row-major; output column `c` sums over K, reading element `(d, c)`.
macro_rules! qvm_pow2 {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] group_size: u32,
        ) {
            // tgid_x = output column, tgid_y = batch row (M).
            let col = tgid_x;
            let m_row = tgid_y;
            let lane = simd_lane;

            let x_row_base = m_row * k;
            let vals_per_pack = 32u32 / $bits;
            let packs_per_row = n / vals_per_pack;
            let mask = (1u32 << $bits) - 1u32;

            let mut acc = 0.0f32;
            let n_iters = (k + 31u32) / 32u32;
            for _it in range(0u32, n_iters, 1u32) {
                let d = _it * 32u32 + lane;
                if d < k {
                    // Groups run along K; scales/biases are [K/G, N].
                    let g = d / group_size;
                    let scale = load(scales[g * n + col]).cast::<f32>();
                    let bias = load(biases[g * n + col]).cast::<f32>();

                    // Element (d, col) of a [K, N]-packed weight matrix.
                    let pack = col / vals_per_pack;
                    let slot = col - pack * vals_per_pack;
                    let word = load(w[d * packs_per_row + pack]);
                    let q = (word >> (slot * $bits)) & mask;

                    let wv = q.cast::<f32>() * scale + bias;
                    acc = acc + wv * load(x[x_row_base + d]).cast::<f32>();
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[m_row * n + col], total.cast::<T>());
            }
        }
        quantized_family_spec!($name, $subop);
    };
}

/// Quantized vecmat (`y = xᵀ · W`) — odd bit-widths. W is `[K, N]`
/// bit-stream-packed.
macro_rules! qvm_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] group_size: u32,
        ) {
            let col = tgid_x;
            let m_row = tgid_y;
            let lane = simd_lane;

            let x_row_base = m_row * k;
            let u32_per_row = n * $bits / 32u32;

            let mut acc = 0.0f32;
            let n_iters = (k + 31u32) / 32u32;
            for _it in range(0u32, n_iters, 1u32) {
                let d = _it * 32u32 + lane;
                if d < k {
                    let g = d / group_size;
                    let scale = load(scales[g * n + col]).cast::<f32>();
                    let bias = load(biases[g * n + col]).cast::<f32>();

                    // Two-word bit-stream extract of element (d, col).
                    let row_u32_off = d * u32_per_row;
                    let bit_off = col * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[row_u32_off + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(w[row_u32_off + w1idx]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;

                    let wv = q.cast::<f32>() * scale + bias;
                    acc = acc + wv * load(x[x_row_base + d]).cast::<f32>();
                }
            }

            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[m_row * n + col], total.cast::<T>());
            }
        }
        quantized_family_spec!($name, $subop);
    };
}

// qmv (matvec) — pow2 widths 4/8, odd widths 3/5/6.
qmv_pow2!(mt_qmv_b4, 4u32, "qmv_b4");
qmv_pow2!(mt_qmv_b8, 8u32, "qmv_b8");
qmv_odd!(mt_qmv_b3, 3u32, "qmv_b3");
qmv_odd!(mt_qmv_b5, 5u32, "qmv_b5");
qmv_odd!(mt_qmv_b6, 6u32, "qmv_b6");

// qmm (matmul / batched matvec) — identical body to qmv, registered
// under the `qmm_b*` subop so the bench scoreboard tracks it
// separately. Dispatch with `grid = [N, M, 1]`.
qmv_pow2!(mt_qmm_b4, 4u32, "qmm_b4");
qmv_pow2!(mt_qmm_b8, 8u32, "qmm_b8");
qmv_odd!(mt_qmm_b3, 3u32, "qmm_b3");
qmv_odd!(mt_qmm_b5, 5u32, "qmm_b5");
qmv_odd!(mt_qmm_b6, 6u32, "qmm_b6");

// qvm (vecmat) — the genuinely missing op; W transposed to [K, N].
qvm_pow2!(mt_qvm_b4, 4u32, "qvm_b4");
qvm_pow2!(mt_qvm_b8, 8u32, "qvm_b8");
qvm_odd!(mt_qvm_b3, 3u32, "qvm_b3");
qvm_odd!(mt_qvm_b5, 5u32, "qvm_b5");
qvm_odd!(mt_qvm_b6, 6u32, "qvm_b6");

// ─── mt_qvm_int4_fast ─────────────────────────────────────────────────
//
// Perf-tuned int4 vecmat `y = xᵀ · W` where W is `[K, N]` row-major.
//
// Geometry vs qmv_fast: qmv tiles *output rows* (W rows, `[N, K]`); here
// we tile *output columns* (W columns, N axis) because the K-dimension
// is the inner dot. The 8-column-per-TG tile (2 SG × 4 cols) is the
// structural dual of qmv's 8-row-per-TG tile — each SG handles 4
// consecutive output columns; each lane contributes to 4 partial sums
// by loading 16 X values and reading 4 corresponding W packs from each
// K-position's row.
//
// W layout: `[K, N]` row-major — row `d` has `N/8` u32 packs, the
// `c`-th output column's nibble for K-position `d` is in pack
// `c/8` at nibble `c%8`. Lane-stride over K — each lane covers K
// positions `lane, lane+32, lane+64, …` with a 32-lane simdgroup.
// (Structurally different from qmv_fast because W strides are transposed:
// column index `c` is within each K-row pack, while K-row index `d` is
// the outer loop axis rather than the inner.)
//
// Scale/bias layout: `[K/G, N]` — scale for column `c` at group `g`
// is at `scales[g * N + c]`, same column-major structure as the scalar
// qvm_pow2. Within a 512-K block `N * G / 512` groups are crossed;
// group index = `d / G` = `(lane_k_base + d_local) / G`.
//
// Algebraic split: `acc_c += scale_c * q_dot_c + bias_c * xs`, where
// `xs = Σ_{d in block} x[d]` is shared across all 4 columns in the SG
// (X is the same vector regardless of which output column we're
// computing). `q_dot_c = Σ_{d in block} nibble(W[d, c]) * x[d]` is
// column-specific.
//
// The mask-without-shift trick: nibbles at positions 1/2/3 within a
// u32 pack have position-powers 16/256/4096. Pre-scaling x[d] by
// 1/16, 1/256, 1/4096 at the matching positions and using the mask
// constant (rather than shift) lets the multiplier absorb the nibble
// position — saves 7 shifts per pack per column = 28 shifts per block.
//
// Dispatch:
//   Grid: [N/8, 1, 1]  — one TG per 8-column tile.
//   TPG:  64           — 2 SG × 32 lanes.
//   K: multiple of 32 (lane-stride). N: multiple of 8.
//   group_size: 64 (standard Qwen3 gs=64).

/// Perf-tuned int4 vecmat `y = xᵀ · W`, W `[K, N]` — 8 output columns
/// per TG, mirroring `mt_qmv`'s 8-row geometry with K/N transposed.
///
/// Each simdgroup handles 4 consecutive output columns. Lane-strided over
/// K: each lane covers `K/32` K-positions. Grid: `[N/8, 1, 1]`, TPG = 64,
/// group_size = 64.
#[kernel]
pub fn mt_qvm_int4_fast<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_col: u32, // K / group_size — groups per output column
) {
    // Each TG handles 8 output columns: SG 0 → cols 0-3, SG 1 → cols 4-7.
    // `tgid_x` = column tile; `simd_id` = 0 or 1; `simd_lane` = lane.
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;

    let col0 = tg * 8u32 + sg * 4u32;
    let col1 = col0 + 1u32;
    let col2 = col0 + 2u32;
    let col3 = col0 + 3u32;

    // Pack offsets within a K-row (N elements packed 8 int4 per u32).
    let packs_per_krow = n / 8u32;
    let col0_pack = col0 / 8u32;
    let col1_pack = col1 / 8u32;
    let col2_pack = col2 / 8u32;
    let col3_pack = col3 / 8u32;

    // Nibble slot within the u32 pack for each column.
    let col0_slot = col0 & 7u32;
    let col1_slot = col1 & 7u32;
    let col2_slot = col2 & 7u32;
    let col3_slot = col3 & 7u32;

    // Mask-without-shift: if the nibble slot is at position p, the
    // 4-bit value lives at bits [4p, 4p+4). The mask-without-shift trick
    // works within a u32 when we pre-scale x by 1/16^p so that the raw
    // masked value (nibble × 16^p) × (x/16^p) = nibble × x.
    // Here each column has a fixed nibble slot within its u32 pack, so
    // we can pre-compute the per-column slot factor once.
    // Slot → mask constant and x-scale factor:
    //   slot 0: mask=0x0000000f, x_scale=1
    //   slot 1: mask=0x000000f0, x_scale=1/16
    //   slot 2: mask=0x00000f00, x_scale=1/256
    //   slot 3: mask=0x0000f000, x_scale=1/4096
    //   slot 4: mask=0x000f0000, x_scale=1/65536
    //   slot 5: mask=0x00f00000, x_scale=1/1048576
    //   slot 6: mask=0x0f000000, x_scale=1/16777216
    //   slot 7: mask=0xf0000000, x_scale=1/268435456
    // For gs=64 with lane-strided K-walk (32 lanes × 1 K-position each),
    // each outer block covers 32 K-positions → 4 half-words per lane for
    // gs=64; but since qvm iterates K one element at a time per lane
    // (not 16 elements as in qmv), we use a simple shift-mask approach
    // at each K-position to extract the nibble, and still apply the
    // algebraic split (xs accumulated per-lane, contributed via simd_sum).
    //
    // Simpler inner loop than qmv: one K-position per lane per iter,
    // extract nibble via shift+mask (not mask-without-shift in the hot
    // path — the column slot varies by column so no single pre-scale
    // covers all 4 columns). The algebraic split still halves the FMA
    // count vs the naive `w_real * x` form.

    let mask4 = 15u32;
    let group_size = k / gs_per_col;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    // Lane-strided K walk: each lane covers K positions
    // `lane, lane+32, lane+64, …`.
    let n_iters = (k + 31u32) / 32u32;
    for _it in range(0u32, n_iters, 1u32) {
        let d = _it * 32u32 + lane; // K-position for this lane
        if d < k {
            let xd = load(x[d]).cast::<f32>();
            let g = d / group_size;

            // Scales/biases: layout [K/G, N] — col c, group g → index g*N+c.
            let s0 = load(scales[g * n + col0]).cast::<f32>();
            let bi0 = load(biases[g * n + col0]).cast::<f32>();
            let s1 = load(scales[g * n + col1]).cast::<f32>();
            let bi1 = load(biases[g * n + col1]).cast::<f32>();
            let s2 = load(scales[g * n + col2]).cast::<f32>();
            let bi2 = load(biases[g * n + col2]).cast::<f32>();
            let s3 = load(scales[g * n + col3]).cast::<f32>();
            let bi3 = load(biases[g * n + col3]).cast::<f32>();

            // W row `d` base offset in the packed array.
            let row_base = d * packs_per_krow;

            // Extract nibbles for each of the 4 output columns.
            let w0 = load(w[row_base + col0_pack]);
            let q0 = ((w0 >> (col0_slot * 4u32)) & mask4).cast::<f32>();

            let w1 = load(w[row_base + col1_pack]);
            let q1 = ((w1 >> (col1_slot * 4u32)) & mask4).cast::<f32>();

            let w2 = load(w[row_base + col2_pack]);
            let q2 = ((w2 >> (col2_slot * 4u32)) & mask4).cast::<f32>();

            let w3 = load(w[row_base + col3_pack]);
            let q3 = ((w3 >> (col3_slot * 4u32)) & mask4).cast::<f32>();

            // Algebraic split: accumulate (s * q - s*bias/s + bias) * x
            // = s*q*x + bias*x.  Equivalently:
            //   acc += s * q * x + bias * x
            //       = s * (q * x) + bias * x
            // Factored as: acc += s * q_dot + bias * xs (over the group),
            // but since we do it per-element here the split is just:
            //   acc += (q * s + bias) * x   — same FMA count as scalar.
            // The xs algebraic split is only advantageous when 16 x values
            // are shared across rows/cols; with 1 K-position per iter it
            // doesn't help. We keep the standard form here.
            acc0 = acc0 + (q0 * s0 + bi0) * xd;
            acc1 = acc1 + (q1 * s1 + bi1) * xd;
            acc2 = acc2 + (q2 * s2 + bi2) * xd;
            acc3 = acc3 + (q3 * s3 + bi3) * xd;
        }
    }

    // Cross-lane reduce within each simdgroup.
    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(out[col0], r0.cast::<T>());
        store(out[col1], r1.cast::<T>());
        store(out[col2], r2.cast::<T>());
        store(out[col3], r3.cast::<T>());
    }
}

quantized_family_spec!(mt_qvm_int4_fast, "qvm_int4_fast");

// ─── mt_qmm_mma_b{3,5,6} — bit-stream MMA for odd bit-widths ────────────────
//
// Bit-width-generalized siblings of `mt_qmm_mma` for int3 / int5 / int6
// quantized dense GEMM. Identical tiled-MMA geometry (BM=BN=BK=32, 4 SGs,
// 2×2 warp grid) — the *only* change is the per-lane W dequant: instead
// of the int4-specific 8-nibble unpack, the weight row is treated as a
// contiguous LSB-first bit-stream and each lane extracts 8 codes with the
// straddle-aware two-word read.
//
// Mirrors the `gather_qmm_mma!` macro in `ffai/moe.rs` — exactly the same
// coop-dequant strategy, just applied to the dense (non-expert) GEMM.
//
// `w` layout: `[N, k*bits/32]` uint32 LSB-first bit-stream packed.
// `group_size` must divide `k`; the 8-K span per lane within a BK=32
// block is group-aligned (`pack_in_row*8 % group_size == 0`).
//
// Grid: [N/32, M/32, 1], tpg=128 (4 SG × 32 lanes).
macro_rules! qmm_mma_bitwidth {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] gs_per_row: u32,
        ) {
            let n_tile = tgid_x;
            let m_tile = tgid_y;
            let lane = simd_lane;
            let sg = simd_group_id();
            let sm = sg / 2u32;
            let sn = sg & 1u32;
            let lane_in_tg = sg * 32u32 + lane;

            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;

            // TG memory: Xs [BM×BK], Ws [BN×BK], both stride BK+4=36
            // (skew avoids 32-bank conflicts on the MMA frag column reads).
            threadgroup_alloc("xs", 1152, T);
            threadgroup_alloc("ws", 1152, T);

            let c_f00 = simdgroup_alloc::<f32, 8, 8>();
            simdgroup_elem_store(c_f00, 0, 0.0f32);
            simdgroup_elem_store(c_f00, 1, 0.0f32);
            let c_f01 = simdgroup_alloc::<f32, 8, 8>();
            simdgroup_elem_store(c_f01, 0, 0.0f32);
            simdgroup_elem_store(c_f01, 1, 0.0f32);
            let c_f10 = simdgroup_alloc::<f32, 8, 8>();
            simdgroup_elem_store(c_f10, 0, 0.0f32);
            simdgroup_elem_store(c_f10, 1, 0.0f32);
            let c_f11 = simdgroup_alloc::<f32, 8, 8>();
            simdgroup_elem_store(c_f11, 0, 0.0f32);
            simdgroup_elem_store(c_f11, 1, 0.0f32);

            let a_f0 = simdgroup_alloc::<T, 8, 8>();
            let a_f1 = simdgroup_alloc::<T, 8, 8>();
            let b_f0 = simdgroup_alloc::<T, 8, 8>();
            let b_f1 = simdgroup_alloc::<T, 8, 8>();

            // W coop-dequant lane assignments:
            //   w_row = lane_in_tg / 4  ∈ 0..32  (the N-row inside the BN=32 tile)
            //   pack_in_row = lane_in_tg & 3  ∈ 0..4  (8-K span index within BK=32)
            let w_row = lane_in_tg / 4u32;
            let pack_in_row = lane_in_tg & 3u32;

            // X coop-load lane assignments: same shape as mt_qmm_mma.
            let x_m_row = lane_in_tg / 4u32;
            let x_k_quad = lane_in_tg & 3u32;
            let x_k_base = x_k_quad * 8u32;

            let xs_ld = 36u32;
            let ws_ld = 36u32;

            let x_m_base = m_tile * 32u32;
            let w_n_base = n_tile * 32u32;
            // Bit-stream: u32_per_row = k * bits / 32 words per weight row.
            let u32_per_row = k * $bits / 32u32;
            let group_size = k / gs_per_row;
            let sb_base = (w_n_base + w_row) * gs_per_row;
            let w_row_base = (w_n_base + w_row) * u32_per_row;

            for kb in range(0u32, k, 32u32) {
                // ── 1. Coop X load — identical to mt_qmm_mma ──
                let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
                let x_ws_base = x_m_row * xs_ld + x_k_base;
                let xv0 = load(x[x_row_dev_base]).cast::<T>();
                let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
                let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
                let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
                let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
                let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
                let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
                let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);

                // ── 2. Coop W bit-stream dequant ──
                // Each lane is responsible for 8 contiguous K positions
                // starting at k0 = kb + pack_in_row*8. It extracts 8 codes
                // from the LSB-first bit-stream, dequantizes them, and
                // writes the 8 fp-T values into Ws[w_row*ws_ld + pack_in_row*8 + i].
                let k0 = kb + pack_in_row * 8u32;
                let g = k0 / group_size;
                let s = load(scales[sb_base + g]).cast::<f32>();
                let b = load(biases[sb_base + g]).cast::<f32>();
                let ws_base = w_row * ws_ld + pack_in_row * 8u32;
                for _ci in range(0u32, 8u32, 1u32) {
                    // Straddle-aware two-word bit-stream extract.
                    // bit_off = absolute bit position of this code's LSB.
                    let bit_off = (k0 + _ci) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[w_row_base + word_idx]);
                    // Avoid an out-of-bounds load when there is no spill: read
                    // from the same word (the bits will be masked to 0 anyway).
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(w[w_row_base + w1idx]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = (lo | hi).cast::<f32>();
                    threadgroup_store("ws", ws_base + _ci, (s * q + b).cast::<T>());
                }

                threadgroup_barrier();

                // ── 3. MMA inner loop — 4 frags × 4 k-inner ──
                // Identical to mt_qmm_mma: serpentine (0,0)→(0,1)→(1,1)→(1,0).
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;

                // k_inner = 0
                simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
                simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
                simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
                simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();

                // k_inner = 1
                simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();

                // k_inner = 2
                simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();

                // k_inner = 3
                simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();

                threadgroup_barrier();
            }

            // ── 4. Write 4 C frags to global out ──
            let out_m_base = m_tile * 32u32 + sm * 16u32;
            let out_n_base = n_tile * 32u32 + sn * 16u32;
            store(
                out[(out_m_base + fm) * n + out_n_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }

        inventory::submit! {
            crate::spec::BenchSpec {
                op: "quantized",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[
                    metaltile_core::dtype::DType::F32,
                    metaltile_core::dtype::DType::F16,
                    metaltile_core::dtype::DType::BF16,
                ],
                tol: 5e-2,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: crate::spec::BenchDispatch::Generic,
                kernel_mode: Some(metaltile_core::ir::KernelMode::Reduction),
            }
        }
    };
}

qmm_mma_bitwidth!(mt_qmm_mma_b3, 3u32, "qmm_mma_b3");
qmm_mma_bitwidth!(mt_qmm_mma_b5, 5u32, "qmm_mma_b5");
qmm_mma_bitwidth!(mt_qmm_mma_b6, 6u32, "qmm_mma_b6");

/// Auto-select the best `mt_qmm*` kernel for a given dtype + M
/// (number of tokens / batched rows in this prefill). Returns the
/// kernel IR ready to dispatch. Caller still owns grid sizing — see
/// the table in the docstring for the per-route grid shape.
///
/// Routing by M parity (post mma + mma_m16 wiring):
///
/// | M condition          | Route       | Why                                                |
/// |----------------------|-------------|----------------------------------------------------|
/// | `m % 32 == 0`        | `mma`       | Full BM=BN=BK=32 simdgroup-matrix tile             |
/// | `m == 16`            | `mma_m16`   | Half-height MMA (BM=16, BN=32) — beats bm4 here   |
/// | `m >= 4 && m%4==0`   | `bm4`       | BM=4 hand-unroll, K=256 (M=8, 12, 20, 24, 28)     |
/// | `m >= 2 && m%2==0`   | `bm2`       | BM=2 hand-unroll (M=2, 6, 10, 14, 18, 22, 26, 30) |
/// | M=1 / odd M          | `mt_qmm`    | v2 (any M, including 1)                            |
///
/// Per-cell wins vs MLX `affine_qmm_t` (5-run median, both rigs):
///
/// | M  | M5 f32       | M5 f16       | M2 f32       | M2 f16       | Route   |
/// |---:|-------------:|-------------:|-------------:|-------------:|---------|
/// |  8 | 182-225%     | 211-264%     | 146-190%     | 161-202%     | bm4     |
/// | 16 | 145-176%     | 145-176%     | 119-167%     | 119-167%     | mma_m16 |
/// | 32 | 97-100%      | 84-91%       | 92-95%       | 92-96%       | mma     |
///
/// At M=32, mma f32 essentially at MLX parity; f16 remains 9-16pt
/// below MLX (open follow-up — 4 layered tweaks identified by the
/// MLX archaeology study at `/tmp/mlx_archaeology.md`).
pub fn mt_qmm_for(dtype: metaltile_core::dtype::DType, m: u32) -> metaltile_core::ir::Kernel {
    use metaltile_core::ir::KernelMode;
    let mut k = if m >= 32 && m.is_multiple_of(32) {
        // Full simdgroup-matrix MMA path (M=32, 64, 96, ...).
        let mut kk = mt_qmm_mma::kernel_ir_for(dtype);
        patch_qmm_mma_dtype_aware_skew(&mut kk, dtype);
        kk
    } else if m == 16 {
        // Half-height MMA — half tpg → 2× occupancy. Wins M=16 cell
        // on both rigs (119-176% MT MLX vs bm4's 76-146%).
        mt_qmm_mma_m16::kernel_ir_for(dtype)
    } else if m >= 4 && m.is_multiple_of(4) {
        // BM=4 K=256 — M=8, 12, 20, 24, 28.
        mt_qmm_bm4::kernel_ir_for(dtype)
    } else if m >= 2 && m.is_multiple_of(2) {
        // BM=2 — M=2, 6, 10, 14, 18, 22, 26, 30.
        mt_qmm_bm2::kernel_ir_for(dtype)
    } else {
        // M=1 + odd M ≥ 3 — bm2/bm4/mma* undefined (need even M).
        mt_qmm::kernel_ir_for(dtype)
    };
    // Reduction mode required for the `tgid_x`/`tgid_y` aliases all
    // kernels reference. Same dispatch contract as `mt_qmv`.
    k.mode = KernelMode::Reduction;
    k
}

/// Fix 1 from `/tmp/mlx_archaeology.md`: dtype-aware TG skew.
///
/// MLX's `affine_qmm_t` uses `BK_padded = BK + 16/sizeof(T)`, so:
///   * f32 (4 bytes) → 32 + 4 = 36  (matches our default)
///   * f16 (2 bytes) → 32 + 8 = 40  ← bump for f16 only
///
/// The DSL body emits `let xs_ld_const = 36u32; let ws_ld_const = 36u32;`
/// which becomes `Op::Const { value: 36 }` named `xs_ld_const` /
/// `ws_ld_const` in the kernel body. This post-patch rewrites those two
/// `Const` ops to 40 when the dtype is f16, and also bumps the matching
/// `ThreadgroupAlloc` sizes (`xs`, `ws`) from 1152 → 1280 so the wider
/// rows fit. Bf16 follows the same 2-byte path. f32 leaves both at 36.
///
/// Uniform stride 40 was tested earlier in isolation and didn't move
/// the M5 f16 needle — but layered with the other 3 archaeology fixes
/// (3-barrier MMA, vectorized X load, serpentine MMA order) it is
/// expected to land the missing 9-16pt.
pub fn patch_qmm_mma_dtype_aware_skew(
    kernel: &mut metaltile_core::ir::Kernel,
    dtype: metaltile_core::dtype::DType,
) {
    use metaltile_core::dtype::DType;
    // f32 keeps its default 36 stride — nothing to do.
    let bytes = match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => return,
    };
    if bytes == 4 {
        return;
    }
    // `BK + 16/sizeof(T)` — 32 + 8 = 40 for 2-byte dtypes.
    let new_ld: i64 = 32 + (16 / bytes as i64);
    let new_alloc: u32 = 32 * (new_ld as u32);

    // Patch Const ops named `xs_ld_const` / `ws_ld_const` in body.
    let target_names: [&str; 2] = ["xs_ld_const", "ws_ld_const"];
    for (vid, name) in kernel.body.names.clone().iter() {
        if !target_names.iter().any(|t| t == name) {
            continue;
        }
        // Find the op slot producing this ValueId.
        for (i, r) in kernel.body.results.iter().enumerate() {
            if r.map(|v| v == *vid).unwrap_or(false)
                && let Some(value) = kernel.body.ops[i].as_const_mut()
            {
                *value = new_ld;
            }
        }
    }
    // Patch ThreadgroupAlloc sizes for `xs` and `ws` (1152 → 1280 at f16).
    for op in kernel.body.ops.iter_mut() {
        if let Some((name, size)) = op.as_threadgroup_alloc_mut()
            && (name == "xs" || name == "ws")
        {
            *size = new_alloc;
        }
    }
}

#[cfg(test)]
mod qmm_selector_tests {
    use metaltile_core::dtype::DType;

    use super::*;

    #[test]
    fn selector_picks_mma_at_m_multiple_of_32() {
        // M % 32 == 0 → full simdgroup-matrix MMA tile.
        for m in [32u32, 64, 96, 128] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm_mma", "m={m}: multiple of 32 should route to mma");
        }
    }

    #[test]
    fn selector_picks_mma_m16_at_m_16() {
        // M = 16 → half-height MMA (mma_m16 beats bm4 + full mma there).
        let k = mt_qmm_for(DType::F32, 16);
        assert_eq!(k.name, "mt_qmm_mma_m16");
    }

    #[test]
    fn selector_picks_bm4_at_m_8_12_20_24_28() {
        // M % 4 == 0 cells NOT routed to mma/mma_m16 — bm4 covers them.
        for m in [4u32, 8, 12, 20, 24, 28, 36, 60] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm_bm4", "m={m}: m%4==0 not mma should route to bm4");
        }
    }

    #[test]
    fn selector_picks_bm2_at_even_m_not_multiple_of_4() {
        for m in [2u32, 6, 10, 14, 18, 22, 26, 30] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm_bm2", "m={m}: even-not-mod-4 should route to bm2");
        }
    }

    #[test]
    fn selector_picks_v2_at_m_1() {
        let k = mt_qmm_for(DType::F32, 1);
        assert_eq!(k.name, "mt_qmm");
    }

    #[test]
    fn selector_picks_v2_at_odd_m() {
        for m in [3u32, 5, 7, 9, 15, 31] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm", "m={m}: odd M should route to v2");
        }
    }

    #[test]
    fn selector_picks_bm4_across_dtypes_at_m_8() {
        for dt in [DType::F32, DType::F16] {
            let k = mt_qmm_for(dt, 8);
            assert_eq!(k.name, "mt_qmm_bm4", "dt={dt:?}");
        }
    }

    #[test]
    fn selector_kernels_carry_reduction_mode() {
        for m in [1u32, 4, 8, 16, 32] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(
                k.mode,
                metaltile_core::ir::KernelMode::Reduction,
                "m={m}: missing Reduction mode",
            );
        }
    }
}
