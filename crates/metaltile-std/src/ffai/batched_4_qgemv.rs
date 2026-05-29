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
//! matrix (0=A, 1=B, 2=C, 3=D); `tgid_x` picks the 8-row output
//! tile. TPG = 64 = 2 simdgroups × 32 lanes. Each simdgroup computes
//! 4 output rows via a `range(0, 4)` loop with `stack_alloc`
//! accumulators — identical to `dequant_gemv_int4_fast`. The DSL
//! unrolls constexpr-bounded loops at codegen, so the emitted MSL is
//! identical to the hand-unrolled form.
//!
//! The x-preload (16 activations per lane per 512-element K-block) is
//! hoisted before the per-matrix dispatch and shared across all four
//! branches: one x load regardless of which matrix is being projected.
//!
//! Output split: four separate buffers (`a_out`, `b_out`, `c_out`,
//! `d_out`). Callers can alias all four into one backing allocation if
//! they want; the kernel only sees four base pointers.
//!
//! Grid: `[ceil(max(out_a, out_b, out_c, out_d) / 8), 1, 4]`. TGs past
//! a matrix's `out_*` rows no-op via the `base_row < out_limit` guard.
//!
//! Constraints:
//!   * `in_dim % 512 == 0`
//!   * `out_a`, `out_b`, `out_c`, `out_d` each a multiple of 8
//!   * `group_size == 64`
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   x         [N]           T
//!   w_*       [out_*, N/8]  uint32
//!   scales_*  [out_*, N/G]  T
//!   biases_*  [out_*, N/G]  T
//!   *_out     [out_*]       T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_4_qgemv_gpu_correctness.rs`.

use metaltile::kernel;

/// Perf-tuned fused 4-output int4 quantized GEMV. 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups × 32 lanes. `simd_id` picks the
/// simdgroup; each simdgroup computes 4 rows via `range(0,4)` +
/// `stack_alloc` (DSL unrolls at codegen → same MSL as hand-unrolled).
/// Uses `dequant_gemv_int4_fast`'s mask-without-shift + algebraic-split.
///
/// Grid: `[ceil(max(out_a,out_b,out_c,out_d)/8), 1, 4]`. All out_*
/// must be multiples of 8; in_dim a multiple of 512; group_size = 64.
#[kernel(
    bench(
        op="batched_4_qgemv",
        subop="batched_4_qgemv_fast",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
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
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants — eliminates 56 shifts per block.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    // Route the row guard to the output size for this matrix slice.
    let out_limit = select(
        matrix == 0u32,
        out_a,
        select(matrix == 1u32, out_b, select(matrix == 2u32, out_c, out_d)),
    );
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    if base_row < out_limit {
        for _b in range(0u32, in_dim, 512u32) {
            // 16 x loads per K-block, shared across all four matrix branches.
            let xb = _b + lane_x_off;
            let x0 = load(x[xb]).cast::<f32>();
            let x1r = load(x[xb + 1u32]).cast::<f32>();
            let x2r = load(x[xb + 2u32]).cast::<f32>();
            let x3r = load(x[xb + 3u32]).cast::<f32>();
            let x4 = load(x[xb + 4u32]).cast::<f32>();
            let x5r = load(x[xb + 5u32]).cast::<f32>();
            let x6r = load(x[xb + 6u32]).cast::<f32>();
            let x7r = load(x[xb + 7u32]).cast::<f32>();
            let x8 = load(x[xb + 8u32]).cast::<f32>();
            let x9r = load(x[xb + 9u32]).cast::<f32>();
            let x10r = load(x[xb + 10u32]).cast::<f32>();
            let x11r = load(x[xb + 11u32]).cast::<f32>();
            let x12 = load(x[xb + 12u32]).cast::<f32>();
            let x13r = load(x[xb + 13u32]).cast::<f32>();
            let x14r = load(x[xb + 14u32]).cast::<f32>();
            let x15r = load(x[xb + 15u32]).cast::<f32>();
            // xs = Σ x[i] over the 16-element block (bias term).
            let xs = x0
                + x1r
                + x2r
                + x3r
                + x4
                + x5r
                + x6r
                + x7r
                + x8
                + x9r
                + x10r
                + x11r
                + x12
                + x13r
                + x14r
                + x15r;
            // Pre-scale nibble positions 1/2/3 for mask-without-shift.
            let x1 = x1r * s_16;
            let x2 = x2r * s_256;
            let x3 = x3r * s_4096;
            let x5 = x5r * s_16;
            let x6 = x6r * s_256;
            let x7 = x7r * s_4096;
            let x9 = x9r * s_16;
            let x10 = x10r * s_256;
            let x11 = x11r * s_4096;
            let x13 = x13r * s_16;
            let x14 = x14r * s_256;
            let x15 = x15r * s_4096;
            let g = xb / group_size;
            let pack_off = _b / 8u32 + lane_pack_off;
            // Per-matrix dispatch. Only tensor names differ across branches;
            // the inner loop body is identical and expressed once per branch.
            if matrix == 0u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_a[wb + pack_off]);
                    let p_hi = load(w_a[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_a[sb + g]).cast::<f32>();
                    let bi = load(biases_a[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 1u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_b[wb + pack_off]);
                    let p_hi = load(w_b[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_b[sb + g]).cast::<f32>();
                    let bi = load(biases_b[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 2u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_c[wb + pack_off]);
                    let p_hi = load(w_c[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_c[sb + g]).cast::<f32>();
                    let bi = load(biases_c[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 3u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_d[wb + pack_off]);
                    let p_hi = load(w_d[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_d[sb + g]).cast::<f32>();
                    let bi = load(biases_d[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
        }
        // Cross-lane reduce + store. out_* are multiples of 8 so all four
        // rows are valid whenever base_row < out_limit.
        //
        // The simd_sums are hoisted into per-row locals BEFORE the lane==0
        // gate so the per-matrix branches can reference them by name. The
        // DSL unroller mangles SSA bindings when a `simd_sum` result is
        // referenced inside a nested `if lane == 0 { if matrix == N { ... } }`
        // block emitted from a `range(0,4)` body (the cast op keeps a stale
        // pre-unroll SSA handle), so the 4 reductions are written by hand
        // here rather than looped. The accumulation loop above keeps the
        // stack_alloc + range dedup intact.
        let r0 = simd_sum(stack_load("accs", 0u32));
        let r1 = simd_sum(stack_load("accs", 1u32));
        let r2 = simd_sum(stack_load("accs", 2u32));
        let r3 = simd_sum(stack_load("accs", 3u32));
        if lane == 0u32 {
            if matrix == 0u32 {
                store(a_out[base_row + 0u32], r0.cast::<T>());
                store(a_out[base_row + 1u32], r1.cast::<T>());
                store(a_out[base_row + 2u32], r2.cast::<T>());
                store(a_out[base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 1u32 {
                store(b_out[base_row + 0u32], r0.cast::<T>());
                store(b_out[base_row + 1u32], r1.cast::<T>());
                store(b_out[base_row + 2u32], r2.cast::<T>());
                store(b_out[base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 2u32 {
                store(c_out[base_row + 0u32], r0.cast::<T>());
                store(c_out[base_row + 1u32], r1.cast::<T>());
                store(c_out[base_row + 2u32], r2.cast::<T>());
                store(c_out[base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 3u32 {
                store(d_out[base_row + 0u32], r0.cast::<T>());
                store(d_out[base_row + 1u32], r1.cast::<T>());
                store(d_out[base_row + 2u32], r2.cast::<T>());
                store(d_out[base_row + 3u32], r3.cast::<T>());
            }
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_batched_4_qgemv_fast;
    use crate::utils::{pack_f32, unpack_f32};

    fn round(v: f32, dt: DType) -> f32 { unpack_f32(&pack_f32(&[v], dt), dt)[0] }
    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

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

    fn quantize_matrix(
        rows: &[f32],
        out_dim: usize,
        in_dim: usize,
        gs: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let (mut w, mut s, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..out_dim {
            let (pw, ps, pb) = quantize_int4_row(&rows[row * in_dim..(row + 1) * in_dim], gs);
            w.extend(pw);
            s.extend(ps);
            b.extend(pb);
        }
        (w, s, b)
    }

    fn naive_gemv(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        in_dim: usize,
        gs: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / gs;
        (0..out_dim)
            .map(|row| {
                let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
                let rs = &scales[row * n_groups..(row + 1) * n_groups];
                let rb = &biases[row * n_groups..(row + 1) * n_groups];
                let mut acc = 0.0_f32;
                for d in 0..in_dim {
                    let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                    let g = d / gs;
                    acc += (q as f32 * rs[g] + rb[g]) * x[d];
                }
                acc
            })
            .collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 2e-1)]
    fn test_batched_4_qgemv_fast(dt: DType) -> TestSetup {
        // Small GDN-style shape: in_dim multiple of 512; out_* multiples of 8.
        let (in_dim, gs) = (512usize, 64usize);
        let (out_a, out_b, out_c, out_d) = (16usize, 8usize, 8usize, 8usize);
        let x: Vec<f32> = source(in_dim, 0x11, 2.0, 0.05).iter().map(|&v| round(v, dt)).collect();
        let wa = source(out_a * in_dim, 0x22, 3.0, 0.0);
        let wb = source(out_b * in_dim, 0x33, 3.0, 0.0);
        let wc = source(out_c * in_dim, 0x44, 3.0, 0.0);
        let wd = source(out_d * in_dim, 0x55, 3.0, 0.0);
        let (wa_p, sa, bias_a) = quantize_matrix(&wa, out_a, in_dim, gs);
        let (wb_p, sb, bb) = quantize_matrix(&wb, out_b, in_dim, gs);
        let (wc_p, sc, bc) = quantize_matrix(&wc, out_c, in_dim, gs);
        let (wd_p, sd, bd) = quantize_matrix(&wd, out_d, in_dim, gs);
        let r = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&v| round(v, dt)).collect() };

        let ea = naive_gemv(&wa_p, &r(&sa), &r(&bias_a), &x, in_dim, gs, out_a);
        let eb = naive_gemv(&wb_p, &r(&sb), &r(&bb), &x, in_dim, gs, out_b);
        let ec = naive_gemv(&wc_p, &r(&sc), &r(&bc), &x, in_dim, gs, out_c);
        let ed = naive_gemv(&wd_p, &r(&sd), &r(&bd), &x, in_dim, gs, out_d);

        let n_tgs = out_a.max(out_b).max(out_c).max(out_d).div_ceil(8);
        TestSetup::new(ffai_batched_4_qgemv_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w_a", pack_u32(&wa_p), DType::U32))
            .input(TestBuffer::from_vec("scales_a", pack_f32(&sa, dt), dt))
            .input(TestBuffer::from_vec("biases_a", pack_f32(&bias_a, dt), dt))
            .input(TestBuffer::from_vec("w_b", pack_u32(&wb_p), DType::U32))
            .input(TestBuffer::from_vec("scales_b", pack_f32(&sb, dt), dt))
            .input(TestBuffer::from_vec("biases_b", pack_f32(&bb, dt), dt))
            .input(TestBuffer::from_vec("w_c", pack_u32(&wc_p), DType::U32))
            .input(TestBuffer::from_vec("scales_c", pack_f32(&sc, dt), dt))
            .input(TestBuffer::from_vec("biases_c", pack_f32(&bc, dt), dt))
            .input(TestBuffer::from_vec("w_d", pack_u32(&wd_p), DType::U32))
            .input(TestBuffer::from_vec("scales_d", pack_f32(&sd, dt), dt))
            .input(TestBuffer::from_vec("biases_d", pack_f32(&bd, dt), dt))
            .input(TestBuffer::zeros("a_out", out_a, dt))
            .input(TestBuffer::zeros("b_out", out_b, dt))
            .input(TestBuffer::zeros("c_out", out_c, dt))
            .input(TestBuffer::zeros("d_out", out_d, dt))
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", gs as u32)
            .expect(TestBuffer::from_vec("a_out", pack_f32(&ea, dt), dt))
            .expect(TestBuffer::from_vec("b_out", pack_f32(&eb, dt), dt))
            .expect(TestBuffer::from_vec("c_out", pack_f32(&ec, dt), dt))
            .expect(TestBuffer::from_vec("d_out", pack_f32(&ed, dt), dt))
            .grid_3d(n_tgs as u32, 1, 4, [64, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_batched_4_qgemv_fast` — MLX-less reduction
/// kernel. GDN input-projection shape (Qwen35): hidden 2048 → conv/value/gate.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_batched_4_qgemv_fast;

    #[bench(name = "ffai/batched_4_qgemv_fast", dtypes = [f32, f16, bf16])]
    fn bench_batched_4_qgemv_fast(dt: DType) -> BenchSetup {
        let (in_dim, gs) = (2048usize, 64usize);
        let (out_a, out_b, out_c, out_d) = (768usize, 1024usize, 64usize, 64usize);
        let ng = in_dim / gs;
        let words = |o: usize| o * in_dim / 8;
        let total = words(out_a) + words(out_b) + words(out_c) + words(out_d);
        let n_tgs = out_a.max(out_b).max(out_c).max(out_d).div_ceil(8);
        BenchSetup::new(ffai_batched_4_qgemv_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("w_a", words(out_a), DType::U32))
            .buffer(BenchBuffer::random("scales_a", out_a * ng, dt))
            .buffer(BenchBuffer::random("biases_a", out_a * ng, dt))
            .buffer(BenchBuffer::random("w_b", words(out_b), DType::U32))
            .buffer(BenchBuffer::random("scales_b", out_b * ng, dt))
            .buffer(BenchBuffer::random("biases_b", out_b * ng, dt))
            .buffer(BenchBuffer::random("w_c", words(out_c), DType::U32))
            .buffer(BenchBuffer::random("scales_c", out_c * ng, dt))
            .buffer(BenchBuffer::random("biases_c", out_c * ng, dt))
            .buffer(BenchBuffer::random("w_d", words(out_d), DType::U32))
            .buffer(BenchBuffer::random("scales_d", out_d * ng, dt))
            .buffer(BenchBuffer::random("biases_d", out_d * ng, dt))
            .buffer(BenchBuffer::zeros("a_out", out_a, dt).output())
            .buffer(BenchBuffer::zeros("b_out", out_b, dt).output())
            .buffer(BenchBuffer::zeros("c_out", out_c, dt).output())
            .buffer(BenchBuffer::zeros("d_out", out_d, dt).output())
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", gs as u32)
            .bytes_moved((total * 4) as u64)
            .grid_3d(n_tgs as u32, 1, 4, [64, 1, 1])
    }
}
