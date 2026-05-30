//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

// ── Pack-strided kernel (int4, int8) ──────────────────────────────────────
//
// Each thread strides over u32 packs. One pack load → (32/$bits) extractions.
// `$bits` must divide 32 evenly (i.e. 2, 4, or 8).
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
                    let bias = load(biases[row_group_off + g]).cast::<f32>();

                    let packed = load(weight[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;

                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * $bits)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias)
                                * load(input[p_off + i]).cast::<f32>();
                    }
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
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
                    let bias = load(biases[row_group_off + g]).cast::<f32>();

                    let bit_off = d * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(weight[row_u32_off + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(weight[row_u32_off + w1idx]);

                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;

                    acc = acc + (q.cast::<f32>() * scale + bias) * load(input[d]).cast::<f32>();
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}

dequant_gemv_pow2!(dequant_gemv_int2, 2u32, "int2");
dequant_gemv_pow2!(dequant_gemv_int4, 4u32, "int4");
dequant_gemv_pow2!(dequant_gemv_int8, 8u32, "int8");
dequant_gemv_odd!(dequant_gemv_int3, 3u32, "int3");
dequant_gemv_odd!(dequant_gemv_int5, 5u32, "int5");
dequant_gemv_odd!(dequant_gemv_int6, 6u32, "int6");

// ── Perf-tuned int4 GEMV — 8 output rows per TG ─────────────────────────
//
// Mirrors `mt_qmv`'s geometry: tpg = 64 (2 simdgroups × 32 lanes);
// each simdgroup computes 4 output rows (indexed by `simd_id`); each
// lane caches 16 X values per 512-wide K-block. Uses the mask-without-
// shift trick + algebraic-split accumulator `s*q_dot + b*xs` from
// `mt_qmv` / MLX `qdot` (quantized.h:235-244).
//
// Kept separate from `dequant_gemv_int4` (the one-row-per-TG scalar)
// for backward compat — FFAI's GPU-router uses the indirect variant of
// the scalar kernel. The fast variant has no indirect consumer today;
// adding one is a one-line edit in `dequant_gemv_wants_indirect`.
//
// Dispatch:
//   Grid: [out_dim/8, 1, 1]  — one TG per 8-row tile.
//   TPG: 64                  — 2 SG × 32 lanes.
//   in_dim: multiple of 512 (block = 16 X × 32 lanes = 512 K elements).
//   out_dim: multiple of 8.
//   group_size: 64.

/// Perf-tuned int4 dequant GEMV — 8 rows per TG, `mt_qmv` geometry.
///
/// `output[row] = Σ_i (q[row,i]·scale_g + bias_g) · input[i]`
/// for 8 consecutive output rows per dispatch. Grid: `[out_dim/8, 1, 1]`,
/// TPG = 64, group_size = 64, in_dim a multiple of 512.
///
/// The existing `dequant_gemv_int4` is kept unchanged for backward compat
/// (FFAI's indirect-dispatch router uses that name). This variant is the
/// perf path for new callers that can guarantee the alignment constraints.
///
/// ## Implementation notes
///
/// The 4-rows-per-simdgroup work is expressed as a `range(0u32, 4u32, 1u32)`
/// loop with a `stack_alloc("accs", 4, f32)` for the per-row accumulators.
/// The DSL unrolls constexpr-bounded `range(...)` loops at codegen, so the
/// emitted MSL is identical to the hand-unrolled form — same 4 weight
/// loads, same 16-nibble mask-without-shift dot per row — just expressed
/// in ~30 lines of loop body instead of 4 × ~40 line copy-pasted blocks.
/// `stack_alloc` accumulators are required because the DSL doesn't lower
/// runtime-indexed `let mut [T; N]` arrays (see the `_m{16,32}` notes in
/// `ffai/moe.rs` for the same constraint).
#[kernel]
pub fn dequant_gemv_int4_fast<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // 8 rows per TG: SG 0 → rows 0-3, SG 1 → rows 4-7. `base_row` is
    // the first of the 4 rows this simdgroup owns.
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // `thread`-private array indexable by a runtime loop variable.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    // Mask-without-shift constants — eliminates 56 shifts per block.
    // Matches `mt_qmv` / MLX `qdot` (quantized.h:235-244): instead of
    // shifting each nibble to position 0, multiply x[1/2/3] by 1/16,
    // 1/256, 1/4096 once and keep the nibble in its native bit slot.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        // 16 X loads per K-block, shared by all 4 rows. Slot 0/4/8/12
        // (the first nibble in each u16 half) is unscaled; the others
        // get pre-scaled by 1/16, 1/256, 1/4096 for mask-without-shift.
        let x0 = load(input[xb]).cast::<f32>();
        let x1_raw = load(input[xb + 1u32]).cast::<f32>();
        let x2_raw = load(input[xb + 2u32]).cast::<f32>();
        let x3_raw = load(input[xb + 3u32]).cast::<f32>();
        let x4 = load(input[xb + 4u32]).cast::<f32>();
        let x5_raw = load(input[xb + 5u32]).cast::<f32>();
        let x6_raw = load(input[xb + 6u32]).cast::<f32>();
        let x7_raw = load(input[xb + 7u32]).cast::<f32>();
        let x8 = load(input[xb + 8u32]).cast::<f32>();
        let x9_raw = load(input[xb + 9u32]).cast::<f32>();
        let x10_raw = load(input[xb + 10u32]).cast::<f32>();
        let x11_raw = load(input[xb + 11u32]).cast::<f32>();
        let x12 = load(input[xb + 12u32]).cast::<f32>();
        let x13_raw = load(input[xb + 13u32]).cast::<f32>();
        let x14_raw = load(input[xb + 14u32]).cast::<f32>();
        let x15_raw = load(input[xb + 15u32]).cast::<f32>();
        // Algebraic-split: acc = scale * q_dot + bias * xs, where
        // xs = Σ input[i] over the 16-element block.
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
        // Pre-scale at nibble positions 1/2/3 (within each u16 half).
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
        // 4 rows × identical work, looped — DSL unrolls at codegen.
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let w_base = row * packs_per_row;
            let sb_base = row * gs_per_row;
            let p_lo = load(weight[w_base + pack_off]);
            let p_hi_word = load(weight[w_base + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi_word >> 16u32;
            let s = load(scales[sb_base + g]).cast::<f32>();
            let bi = load(biases[sb_base + g]).cast::<f32>();
            // 16-nibble dot, mask-without-shift form. Each u32 carries
            // 8 nibbles split as 4 in the low 16 bits + 4 in the high
            // 16 bits; the four masks `15 / 240 / 3840 / 61440` peel off
            // the nibble at slot 0/1/2/3 of each half.
            let qd = (p_lo & 15u32).cast::<f32>() * x0
                + (p_lo & 240u32).cast::<f32>() * x1
                + (p_lo & 3840u32).cast::<f32>() * x2
                + (p_lo & 61440u32).cast::<f32>() * x3
                + (p_lo_hi & 15u32).cast::<f32>() * x4
                + (p_lo_hi & 240u32).cast::<f32>() * x5
                + (p_lo_hi & 3840u32).cast::<f32>() * x6
                + (p_lo_hi & 61440u32).cast::<f32>() * x7
                + (p_hi_word & 15u32).cast::<f32>() * x8
                + (p_hi_word & 240u32).cast::<f32>() * x9
                + (p_hi_word & 3840u32).cast::<f32>() * x10
                + (p_hi_word & 61440u32).cast::<f32>() * x11
                + (p_hi_hi & 15u32).cast::<f32>() * x12
                + (p_hi_hi & 240u32).cast::<f32>() * x13
                + (p_hi_hi & 3840u32).cast::<f32>() * x14
                + (p_hi_hi & 61440u32).cast::<f32>() * x15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * xs);
        }
    }
    // Cross-lane reduce: one simd_sum per row → one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_row + _r], r.cast::<T>());
        }
    }
}

/// Per-kernel opt-in for the indirect Swift-wrapper variant. FFAI's
/// GPU-router dispatches the int4 dequant-GEMV indirectly so the GPU
/// can drive the per-MoE-layer grid shape from a buffer; the other
/// dequant-GEMV bit-widths have no indirect consumer today.
///
/// Lives here (next to the kernel definitions) rather than in
/// `metaltile-codegen` so that adding a new kernel that wants the
/// indirect variant is a one-line edit in the same file as the
/// kernel, not a special-case match buried in the codegen pass.
/// The `tile emit` driver consumes this on the way to setting
/// `Kernel::wants_indirect_variant` before codegen runs.
pub fn dequant_gemv_wants_indirect(kernel_name: &str) -> bool {
    matches!(kernel_name, "dequant_gemv_int4_f16" | "dequant_gemv_int4_bf16")
}

/// New-syntax correctness tests for the `dequant_gemv_int{2,3,4,5,6,8}`
/// family + the perf-tuned `dequant_gemv_int4_fast`. All are Reduction-mode
/// (one threadgroup per output row, `reduce_sum` across the threadgroup).
///
/// Oracle: synthesize bit-stream-packed int-`bits` weights `[out_dim, in_dim]`
/// (the same `lo | hi` two-word layout the kernel decodes — works for both the
/// pack-strided pow2 widths and the odd widths), per-group scale/bias, then
/// replay the dequant-then-dot `output[row] = Σ_d (q·scale_g + bias_g)·input[d]`
/// in f32. Inputs are dtype-rounded so the GPU sees exactly what the oracle does.
///
/// Grid (scalar variants): `grid_3d(out_dim, 1, 1, [tpg, 1, 1])` — one TG per
/// output row, tpg = 64 (≥32, multiple of 32). The `_fast` variant does 8 rows
/// per TG so `grid_3d(out_dim/8, 1, 1, [64, 1, 1])`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    /// Bytes for a u32 slice (packed weights bind as a `DType::U32` buffer).
    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// One threadgroup-row's worth of lanes. ≥ 32 and a multiple of 32 per the
    /// Reduction dispatch contract; 64 lanes give a healthy `reduce_sum` tree.
    const TPG: u32 = 64;

    /// Synthesize bit-stream-packed int-`bits` weights for an `[out_dim, in_dim]`
    /// matrix. Codes are written into the row's u32 bit-stream at bit offset
    /// `d * bits`, spilling into the next word when they straddle a u32 boundary
    /// — the exact layout the kernel's `lo | hi` decode (and the legacy test's
    /// `quantize_row`) expects. Works for every supported bit width.
    fn synth_bitstream_w(out_dim: usize, in_dim: usize, bits: u32) -> Vec<u32> {
        let mask = (1u32 << bits) - 1;
        let u32_per_row = in_dim * bits as usize / 32;
        let mut packed = vec![0u32; out_dim * u32_per_row];
        for row in 0..out_dim {
            let row_base = row * u32_per_row;
            for d in 0..in_dim {
                // Deterministic, in-range code; varies per (row, d).
                let code =
                    ((row * in_dim + d) as u32).wrapping_mul(2_654_435_761).wrapping_add(d as u32)
                        & mask;
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                if bits_in_w0 >= bits {
                    packed[row_base + word] |= code << in_w;
                } else {
                    packed[row_base + word] |= code << in_w;
                    packed[row_base + word + 1] |= code >> bits_in_w0;
                }
            }
        }
        packed
    }

    /// Dequant-then-dot reference (mirrors the legacy `naive_dequant_gemv`).
    /// `weight` packs `[out_dim, in_dim]` int-`bits` codes, `scales`/`biases`
    /// are `[out_dim, in_dim/group_size]`, `input` is `[in_dim]`, out `[out_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn dequant_gemv_oracle(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        input: &[f32],
        in_dim: usize,
        group_size: usize,
        bits: u32,
        out_dim: usize,
    ) -> Vec<f32> {
        let u32_per_row = in_dim * bits as usize / 32;
        let n_groups = in_dim / group_size;
        let mask: u64 = (1u64 << bits) - 1;
        let mut out = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut acc = 0.0f32;
            let row_w = &weight[row * u32_per_row..(row + 1) * u32_per_row];
            for (d, &x_d) in input.iter().enumerate().take(in_dim) {
                let g = d / group_size;
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                let q = if bits_in_w0 >= bits {
                    ((row_w[word] as u64) >> in_w) & mask
                } else {
                    let lo_bits = bits_in_w0;
                    let spill = bits - lo_bits;
                    let lo = ((row_w[word] as u64) >> in_w) & ((1u64 << lo_bits) - 1);
                    let hi = ((row_w[word + 1] as u64) & ((1u64 << spill) - 1)) << lo_bits;
                    lo | hi
                };
                acc += ((q as f32) * scales[row * n_groups + g] + biases[row * n_groups + g]) * x_d;
            }
            out[row] = acc;
        }
        out
    }

    /// Shared setup for the scalar (one-row-per-TG) variants. `grid_rows` is the
    /// number of x-groups dispatched (out_dim) and `tpg` the lanes per row.
    #[allow(clippy::too_many_arguments)]
    fn gemv_setup(
        kernel: Kernel,
        bits: u32,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        grid_rows: u32,
        tpg: u32,
        dt: DType,
    ) -> TestSetup {
        let n_groups = in_dim / group_size;
        let w = synth_bitstream_w(out_dim, in_dim, bits);
        let scales_f: Vec<f32> =
            (0..out_dim * n_groups).map(|i| 0.004 + (i % 7) as f32 * 0.0008).collect();
        let biases_f: Vec<f32> =
            (0..out_dim * n_groups).map(|i| ((i % 5) as f32 - 2.0) * 0.0009).collect();
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = dequant_gemv_oracle(&w, &s, &b, &x, in_dim, group_size, bits, out_dim);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d(grid_rows, 1, 1, [tpg, 1, 1])
    }

    // ── Pack-strided (int2 / int4 / int8) ──────────────────────────────────
    // in_dim a multiple of 32/bits; group_size 64; one TG per output row.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int2(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int2::kernel_ir_for(dt), 2, 4, 256, 64, 4, TPG, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int4::kernel_ir_for(dt), 4, 4, 256, 64, 4, TPG, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int8(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int8::kernel_ir_for(dt), 8, 4, 256, 64, 4, TPG, dt)
    }

    // ── Element-strided odd widths (int3 / int5 / int6) ─────────────────────
    // in_dim*bits must be a multiple of 32 (u32-aligned packed row):
    //   int3: 64*3 = 192 = 6 u32; int5: 64*5 = 320 = 10 u32; int6: 64*6 = 384.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int3(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int3::kernel_ir_for(dt), 3, 4, 64, 32, 4, TPG, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int5(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int5::kernel_ir_for(dt), 5, 4, 64, 32, 4, TPG, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int6(dt: DType) -> TestSetup {
        gemv_setup(dequant_gemv_int6::kernel_ir_for(dt), 6, 4, 64, 32, 4, TPG, dt)
    }

    // ── Perf-tuned int4_fast: 8 rows per TG ─────────────────────────────────
    // in_dim a multiple of 512, out_dim a multiple of 8, group_size 64.
    // Grid: [out_dim/8, 1, 1], TPG = 64.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4_fast(dt: DType) -> TestSetup {
        let (out_dim, in_dim, group_size) = (8usize, 512usize, 64usize);
        gemv_setup(
            dequant_gemv_int4_fast::kernel_ir_for(dt),
            4,
            out_dim,
            in_dim,
            group_size,
            (out_dim / 8) as u32,
            64,
            dt,
        )
    }
}

/// New-syntax benchmarks for the dequant GEMV family. Production-ish shapes
/// (out_dim/in_dim 4096, group_size 64). bytes_moved counts the packed-weight
/// stream (dominant) + scales/biases + input + output.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn gb(
        kernel: Kernel,
        bits: u32,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        grid_rows: u32,
        tpg: u32,
        dt: DType,
    ) -> BenchSetup {
        let n_groups = in_dim / group_size;
        let u32_per_row = in_dim * bits as usize / 32;
        let sz = dt.size_bytes();
        let bytes =
            out_dim * u32_per_row * 4 + 2 * out_dim * n_groups * sz + in_dim * sz + out_dim * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", out_dim * u32_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d(grid_rows, 1, 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/dequant_gemv/int2", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int2(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int2::kernel_ir_for(dt), 2, 4096, 4096, 64, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gemv/int3", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int3(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int3::kernel_ir_for(dt), 3, 4096, 4096, 64, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gemv/int4", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int4(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int4::kernel_ir_for(dt), 4, 4096, 4096, 64, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gemv/int5", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int5(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int5::kernel_ir_for(dt), 5, 4096, 4096, 64, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gemv/int6", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int6(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int6::kernel_ir_for(dt), 6, 4096, 4096, 64, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gemv/int8", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int8(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int8::kernel_ir_for(dt), 8, 4096, 4096, 64, 4096, 64, dt)
    }

    // 8-rows-per-TG fast int4: grid [out_dim/8, 1, 1], TPG 64.
    #[bench(name = "ffai/dequant_gemv/int4_fast", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int4_fast(dt: DType) -> BenchSetup {
        gb(dequant_gemv_int4_fast::kernel_ir_for(dt), 4, 4096, 4096, 64, 4096 / 8, 64, dt)
    }
}
