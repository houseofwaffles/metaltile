//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused gated-RMSNorm + 4-bit quantized GEMV for the Qwen3.5 / Qwen3.6
//! Gated-DeltaNet (GDN) decode tail.
//!
//! Collapses the two back-to-back dispatches that close every GDN layer
//! into ONE kernel launch:
//!
//!   1. `mt_gated_rmsnorm`: per-row RMSNorm with SiLU gate.
//!      `inner[r, d] = w[d] * y[r, d] * rsqrt(mean(y[r]^2) + eps) * silu(z[r, d])`
//!      `y` is fp32 (GDN recurrence accumulates in fp32), `z` / `w` /
//!      `inner` are model dtype `T`.
//!
//!   2. `ffai_dequant_gemv_int4` (the GDN out projection):
//!      `out[o] = sum_i (q[o, i] * scale + bias) * inner_flat[i]`
//!      where `inner_flat[r * Dv + d] = inner[r, d]` and `i in [0, Hv*Dv)`.
//!
//! Fusing them eliminates one encoder begin/end pair per GDN layer plus
//! the global-memory round trip on `inner` (Hv * Dv * sizeof(T) per
//! layer, ~4 KiB at Qwen3.6-A3B). Pattern mirrors `rms_norm_qgemv_fast`
//! (8-row-per-TG fused norm + int4 GEMV for finalNorm+lmHead) and the
//! `moe_down_swiglu_accum` TG-staged-intermediate fusion.
//!
//! ## Geometry
//!
//! - **Grid: `[out_dim / 8, 1, 1]`** - one TG per 8-row tile.
//! - **TPG = 64** (2 simdgroups x 32 lanes).
//!
//! Phase 1 (gated-RMSNorm) stages the post-gated activation into a
//! threadgroup-memory buffer `tg_inner[Hv * Dv]` at fp32. The kernel
//! processes the `Hv` rows two at a time (one row per simdgroup). For
//! each row pair `(r0, r1) = (2*it + 0, 2*it + 1)`:
//!   * Each lane computes a per-lane partial sum of squares across its
//!     `Dv/32` elements of `y[r]`.
//!   * `simd_sum` folds the partial across the simdgroup - gives the
//!     full row SSQ in every lane.
//!   * `inv_rms[r] = rsqrt(ssq / Dv + eps)` is computed locally per lane.
//!   * Each lane writes its `Dv/32` gated-and-normed elements to
//!     `tg_inner` (`silu` of the `z` gate is inlined in fp32).
//!
//! After all rows are filled, a single `threadgroup_barrier` flips the
//! data into Phase-2 visibility.
//!
//! Phase 2 (int4 GEMV) reuses the `rms_norm_qgemv_fast` 8-row-per-TG
//! pattern verbatim: 2 simdgroups, each computing 4 output rows via the
//! mask-without-shift trick (X pre-scaled by inverse nibble position,
//! algebraic-split accumulator `acc = scale * q_dot + bias * normed_xs`).
//! The only delta is that the X stripe is loaded from `tg_inner` (fp32,
//! no further casts) instead of fused on the fly from device `x`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - `in_dim = Hv * Dv` must be a multiple of 512 (kernel reads 16 X
//!   per lane x 32 lanes = 512 per Phase-2 block).
//! - `out_dim` must be a multiple of 8 (8-row-per-TG tiling).
//! - `group_size` must be 64 (one quant group per 4 lanes in Phase 2).
//! - `dv` must be a multiple of 32 (one Phase-1 simdgroup per row).
//! - `hv` must be even (rows are assigned in pairs across the 2
//!   simdgroups).
//! - **TG memory budget: `Hv * Dv * 4` bytes** of fp32 in `tg_inner`.
//!   Apple9 cap is 32 KiB, so `Hv * Dv <= 8192`. At Qwen3.6-A3B
//!   (`Hv=16`, `Dv=128`) this is 8 KiB. Bumping the literal in
//!   `threadgroup_alloc` is required for larger geometries.
//!
//! For Qwen3.6-A3B: `Hv=16`, `Dv=128`, `in_dim=2048`, `out_dim=hidden=2048`.
//! All four invariants hold.
//!
//! ## Correctness invariant
//!
//! At identical inputs (within the f32 reorder envelope of
//! `simd_sum`-based reductions), this kernel produces the same output
//! as the unfused chain:
//!
//! ```text
//!   inner = mt_gated_rmsnorm(y, z, w, eps)        // [Hv, Dv]
//!   out   = ffai_dequant_gemv_int4(inner, Wq, S, B)  // [out_dim]
//! ```
//!
//! Pinned by `tests/gated_rms_norm_qgemv_int4_gpu_correctness.rs`.

use metaltile::kernel;

/// Fused gated-RMSNorm + int4 GEMV - 8 output rows per TG.
///
/// Phase 1 stages `inner[r, d] = w[d] * y[r, d] * rsqrt(mean(y[r]^2) +
/// eps) * silu(z[r, d])` into `tg_inner` (fp32). Phase 2 runs the
/// int4 GEMV reading the staged activation. Grid: `[out_dim/8, 1, 1]`,
/// TPG = 64. See module doc for invariants.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_gated_rms_norm_qgemv_int4_fast<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    q_weight: Tensor<u32>,
    q_scales: Tensor<T>,
    q_biases: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    // ── Threadgroup scratch ────────────────────────────────────────────
    // 8192 = 8 KiB at fp32. Covers Qwen3.6-A3B (Hv*Dv = 2048) with 4x
    // headroom for future heads/widths. Apple9 hard cap is 32 KiB, so a
    // 16384-element bump is still safe should a model need it.
    threadgroup_alloc("tg_inner", 8192, "f32");

    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;

    // ── Phase 1: gated RMSNorm into `tg_inner` ─────────────────────────
    //
    // Each simdgroup owns one row at a time: `sg=0` does even rows,
    // `sg=1` does odd rows. Row index r = it * 2 + sg, it in [0, hv/2).
    // Per row, the 32 lanes of the simdgroup cover Dv elements with a
    // per-lane stride of `dv / 32` - exactly one simd_sum per row gives
    // the full SSQ.
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        // SSQ across this lane's stripe of the row, in fp32.
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        // Write the gated-and-normed stripe to `tg_inner`. The qmm in
        // Phase 2 reads from here in fp32, so cast-up once at the
        // gate/weight loads and store fp32.
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            // silu(z) = z / (1 + exp(-z)), inline fp32 - same form as
            // `ffai_gated_rmsnorm` / `moe_down_swiglu_accum`.
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    // RAW barrier: Phase 2 reads `tg_inner` filled by all lanes above.
    threadgroup_barrier();

    // ── Phase 2: 8-row int4 GEMV against `tg_inner` ────────────────────
    //
    // Mirrors `ffai_rms_norm_qgemv_fast` Phase 2 verbatim, except the
    // 16-element X stripe per lane is loaded from `tg_inner` (fp32) in
    // place of the on-the-fly `x[xi] * norm_weight[xi] * inv_rms` fuse.
    //
    // The 4-rows-per-simdgroup work is expressed as a `range(0u32, 4u32,
    // 1u32)` loop with a `stack_alloc("accs", 4, f32)` for the per-row
    // accumulators — same shape as `dequant_gemv_int4_fast`. Replaces the
    // four hand-unrolled `acc0/acc1/acc2/acc3` blocks (~140 lines each)
    // with one ~30-line loop body that the DSL lowers identically.
    let in_dim = hv * dv;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // `thread`-private array indexable by a runtime loop variable; the DSL
    // doesn't lower runtime-indexed `let mut [T; N]` arrays (see the
    // `_m{16,32}` notes in `ffai/moe.rs` for the same constraint).
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants - identical to `rms_norm_qgemv_fast`.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        // Pull this lane's 16-element X stripe from staged `tg_inner`.
        let n0_raw = threadgroup_load("tg_inner", xb);
        let n1_raw = threadgroup_load("tg_inner", xb + 1u32);
        let n2_raw = threadgroup_load("tg_inner", xb + 2u32);
        let n3_raw = threadgroup_load("tg_inner", xb + 3u32);
        let n4_raw = threadgroup_load("tg_inner", xb + 4u32);
        let n5_raw = threadgroup_load("tg_inner", xb + 5u32);
        let n6_raw = threadgroup_load("tg_inner", xb + 6u32);
        let n7_raw = threadgroup_load("tg_inner", xb + 7u32);
        let n8_raw = threadgroup_load("tg_inner", xb + 8u32);
        let n9_raw = threadgroup_load("tg_inner", xb + 9u32);
        let n10_raw = threadgroup_load("tg_inner", xb + 10u32);
        let n11_raw = threadgroup_load("tg_inner", xb + 11u32);
        let n12_raw = threadgroup_load("tg_inner", xb + 12u32);
        let n13_raw = threadgroup_load("tg_inner", xb + 13u32);
        let n14_raw = threadgroup_load("tg_inner", xb + 14u32);
        let n15_raw = threadgroup_load("tg_inner", xb + 15u32);
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
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        // 4 rows × identical work, looped — mirrors `dequant_gemv_int4_fast`
        // and the `stack_alloc + range` pattern from TheTom/metaltile#1.
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let w_base = row * packs_per_row;
            let sb_base = row * gs_per_row;
            let p_lo = load(q_weight[w_base + pack_off]);
            let p_hi_word = load(q_weight[w_base + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi_word >> 16u32;
            let s = load(q_scales[sb_base + g]).cast::<f32>();
            let bi = load(q_biases[sb_base + g]).cast::<f32>();
            // 16-nibble dot, mask-without-shift form. Each u32 carries
            // 8 nibbles split as 4 in the low 16 bits + 4 in the high
            // 16 bits; the four masks `15 / 240 / 3840 / 61440` peel off
            // the nibble at slot 0/1/2/3 of each half.
            let qd = (p_lo & 15u32).cast::<f32>() * n0_raw
                + (p_lo & 240u32).cast::<f32>() * n1
                + (p_lo & 3840u32).cast::<f32>() * n2
                + (p_lo & 61440u32).cast::<f32>() * n3
                + (p_lo_hi & 15u32).cast::<f32>() * n4_raw
                + (p_lo_hi & 240u32).cast::<f32>() * n5
                + (p_lo_hi & 3840u32).cast::<f32>() * n6
                + (p_lo_hi & 61440u32).cast::<f32>() * n7
                + (p_hi_word & 15u32).cast::<f32>() * n8_raw
                + (p_hi_word & 240u32).cast::<f32>() * n9
                + (p_hi_word & 3840u32).cast::<f32>() * n10
                + (p_hi_word & 61440u32).cast::<f32>() * n11
                + (p_hi_hi & 15u32).cast::<f32>() * n12_raw
                + (p_hi_hi & 240u32).cast::<f32>() * n13
                + (p_hi_hi & 3840u32).cast::<f32>() * n14
                + (p_hi_hi & 61440u32).cast::<f32>() * n15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * ns);
        }
    }
    // Cross-lane reduce: one simd_sum per row -> one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(out[base_row + _r], r.cast::<T>());
        }
    }
}

mod oracle {
    use crate::{bench_types::DType, utils::pack_f32};

    /// Per-row affine int4 quantize, 8 nibbles per u32 — same packing the
    /// kernel decodes. Returns (packed_weight, scales, biases) for one row.
    pub fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
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

    /// Deterministic xorshift source, matching the legacy test's generator.
    pub fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
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

    pub fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// CPU oracle: per-row gated RMSNorm (`silu(z)` gate) → int4 GEMV.
    /// Mirrors `tests/gated_rms_norm_qgemv_int4_gpu_correctness.rs::naive`.
    #[allow(clippy::too_many_arguments)]
    pub fn naive(
        y: &[f32],
        z: &[f32],
        norm_weight: &[f32],
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        hv: usize,
        dv: usize,
        out_dim: usize,
        group_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let in_dim = hv * dv;
        let mut inner = vec![0.0_f32; in_dim];
        for r in 0..hv {
            let base = r * dv;
            let row = &y[base..base + dv];
            let ssq: f32 = row.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / dv as f32 + eps).sqrt();
            for d in 0..dv {
                let g = z[base + d] / (1.0 + (-z[base + d]).exp());
                inner[base + d] = y[base + d] * inv_rms * norm_weight[d] * g;
            }
        }
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / group_size;
        (0..out_dim)
            .map(|row| {
                let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
                let rs = &scales[row * n_groups..(row + 1) * n_groups];
                let rb = &biases[row * n_groups..(row + 1) * n_groups];
                let mut acc = 0.0_f32;
                for d in 0..in_dim {
                    let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                    let g = d / group_size;
                    let w_real = q as f32 * rs[g] + rb[g];
                    acc += w_real * inner[d];
                }
                acc
            })
            .collect()
    }

    /// Round f32 vals through `dt` (so the CPU oracle sees the GPU's load
    /// precision) and re-pack to f32 for the oracle. Used for z/scales/etc.
    pub fn round(v: &[f32], dt: DType) -> Vec<f32> {
        crate::utils::unpack_f32(&pack_f32(v, dt), dt)
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{
        ffai_gated_rms_norm_qgemv_int4_fast,
        oracle::{naive, quantize_int4_row, round, source, u32_bytes},
    };
    use crate::utils::pack_f32;

    /// Build the test for one dtype. Fast-variant constraints: `in_dim =
    /// hv*dv` a multiple of 512, `out_dim` a multiple of 8, `group_size = 64`,
    /// `dv` a multiple of 32, `hv` even.
    fn setup(hv: usize, dv: usize, out_dim: usize, group_size: usize, dt: DType) -> TestSetup {
        let in_dim = hv * dv;
        let eps = 1e-5_f32;
        // `y` stays fp32 (GDN recurrence output crosses the boundary in fp32).
        let y: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1);
        let z: Vec<f32> = round(&source(in_dim, 0xD4, 1.5, 0.0), dt);
        let norm_weight: Vec<f32> = round(&source(dv, 0xB2, 0.4, 1.0), dt);
        let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);

        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for row in 0..out_dim {
            let (w, s, b) =
                quantize_int4_row(&w_rows[row * in_dim..(row + 1) * in_dim], group_size);
            weight.extend(w);
            scales.extend(s);
            biases.extend(b);
        }
        let scales_r = round(&scales, dt);
        let biases_r = round(&biases, dt);

        let expected = naive(
            &y,
            &z,
            &norm_weight,
            &weight,
            &scales_r,
            &biases_r,
            hv,
            dv,
            out_dim,
            group_size,
            eps,
        );

        TestSetup::new(ffai_gated_rms_norm_qgemv_int4_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("y", pack_f32(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&norm_weight, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("q_weight", u32_bytes(&weight), DType::U32))
            .input(TestBuffer::from_vec("q_scales", pack_f32(&scales, dt), dt))
            .input(TestBuffer::from_vec("q_biases", pack_f32(&biases, dt), dt))
            .input(TestBuffer::zeros("out", out_dim, dt))
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
    }

    // Small shape: hv=4, dv=128, in_dim=512, out_dim=512, group_size=64.
    // int4 quant + dependent reductions → relative-tolerance bands per dtype.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 3e-2, 6e-2])]
    fn test_gated_rms_norm_qgemv_int4_fast(dt: DType) -> TestSetup { setup(4, 128, 512, 64, dt) }
}

/// New-syntax benchmark for `ffai_gated_rms_norm_qgemv_int4_fast` at the
/// Qwen3.6-A3B production shape (hv=16, dv=128, in_dim=2048, out_dim=2048).
/// MLX-less reduction kernel (`class=GenericEmpty`), so `Ref(GB/s)` blank.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gated_rms_norm_qgemv_int4_fast;

    #[bench(name = "ffai/gated_rms_norm_qgemv_int4_fast", dtypes = [f32, f16, bf16])]
    fn bench_gated_rms_norm_qgemv_int4_fast(dt: DType) -> BenchSetup {
        let (hv, dv, out_dim, group_size) = (16usize, 128usize, 2048usize, 64usize);
        let in_dim = hv * dv;
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / group_size;
        BenchSetup::new(ffai_gated_rms_norm_qgemv_int4_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("y", in_dim, DType::F32))
            .buffer(BenchBuffer::random("z", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", dv, dt))
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::random("q_weight", out_dim * u32_per_row, DType::U32))
            .buffer(BenchBuffer::random("q_scales", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("q_biases", out_dim * n_groups, dt))
            .buffer(BenchBuffer::zeros("out", out_dim, dt).output())
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
            // Weight matrix dominates traffic: out_dim*in_dim int4 = bytes/2.
            .bytes_moved((out_dim * in_dim / 2) as u64)
    }
}
