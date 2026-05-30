//! Batched Q/K/V 4-bit quantized QMM (M>1) — fuses the three independent
//! Q, K, V projection matmuls of a *prefill* step into one dispatch.
//!
//! This is the M>1 sibling of `ffai_batched_qkv_qgemv_fast`. At
//! `program_id::<1>() = m` we load row `m` of the batched input
//! `x: [M, in_dim]` and produce row `m` of THREE separate output tensors:
//!   q_buf: [M, out_q] T
//!   k_buf: [M, out_k] T
//!   v_buf: [M, out_v] T
//!
//! Three separate buffers keep each projection contiguous in memory so
//! the downstream attention pipeline reads `[M, dim]` per projection.
//! Callers can alias all three into one backing allocation if they want;
//! the kernel only sees three base pointers.
//!
//! Dispatch geometry mirrors `ffai_batched_qkv_qgemv_fast`:
//!   * `program_id::<2>()` selects matrix (0 = Q, 1 = K, 2 = V).
//!   * `program_id::<1>()` selects batched row m (0..M).
//!   * `tgid_x` selects an 8-row output tile. TPG = 64 = 2 SG × 32 lanes.
//!
//! Grid: `[ceil(max(out_q,out_k,out_v)/8), M, 3]`, TPG = `[64,1,1]`.
//!
//! The inner loop is the same `stack_alloc` + `range(0,4)` pattern as
//! `dequant_gemv_int4_fast` — DSL unrolls at codegen. The x-preload is
//! hoisted before the per-matrix dispatch and shared across all branches.
//!
//! Constraints (same as the fast GEMV variant):
//!   * `in_dim % 512 == 0`
//!   * `out_q`, `out_k`, `out_v` each a multiple of 8
//!   * `group_size == 64`
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   x         [M, N]          T
//!   w_*       [out_*, N/8]    uint32
//!   scales_*  [out_*, N/G]    T
//!   biases_*  [out_*, N/G]    T
//!   *_buf     [M, out_*]      T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_qkv_qmm_gpu_correctness.rs`.

use metaltile::kernel;

/// Perf-tuned fused Q/K/V int4 QMM (M>1) — 8 output rows per TG.
///
/// Grid: `[ceil(max(out_q,out_k,out_v)/8), M, 3]`. See module docs for
/// the full geometry contract. TGs past a matrix's `out_*` rows no-op.
#[kernel]
pub fn ffai_batched_qkv_qmm_fast<T>(
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
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let m = program_id::<1>();
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Row-m offsets into x and per-projection output buffers.
    let x_row_off = m * in_dim;
    let q_row_off = m * out_q;
    let k_row_off = m * out_k;
    let v_row_off = m * out_v;
    // Mask-without-shift constants — eliminates 56 shifts per block.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    // Route the row guard to the output size for this matrix slice.
    let out_limit = select(matrix == 0u32, out_q, select(matrix == 1u32, out_k, out_v));
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    if base_row < out_limit {
        for _b in range(0u32, in_dim, 512u32) {
            // 16 x loads per K-block, shared across all three matrix branches.
            // xb includes the batch-row offset; group index uses the
            // in-row column offset only (scales/biases are per weight row).
            let xb = x_row_off + _b + lane_x_off;
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
            // Group index uses the in-row column offset (not the batched
            // global offset) since scales/biases are per weight row × group.
            let g = (_b + lane_x_off) / group_size;
            let pack_off = _b / 8u32 + lane_pack_off;
            // Per-matrix dispatch. Only tensor names differ across branches.
            if matrix == 0u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_q[wb + pack_off]);
                    let p_hi = load(w_q[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_q[sb + g]).cast::<f32>();
                    let bi = load(biases_q[sb + g]).cast::<f32>();
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
                    let p_lo = load(w_k[wb + pack_off]);
                    let p_hi = load(w_k[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_k[sb + g]).cast::<f32>();
                    let bi = load(biases_k[sb + g]).cast::<f32>();
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
                    let p_lo = load(w_v[wb + pack_off]);
                    let p_hi = load(w_v[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_v[sb + g]).cast::<f32>();
                    let bi = load(biases_v[sb + g]).cast::<f32>();
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
                store(q_buf[q_row_off + base_row + 0u32], r0.cast::<T>());
                store(q_buf[q_row_off + base_row + 1u32], r1.cast::<T>());
                store(q_buf[q_row_off + base_row + 2u32], r2.cast::<T>());
                store(q_buf[q_row_off + base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 1u32 {
                store(k_buf[k_row_off + base_row + 0u32], r0.cast::<T>());
                store(k_buf[k_row_off + base_row + 1u32], r1.cast::<T>());
                store(k_buf[k_row_off + base_row + 2u32], r2.cast::<T>());
                store(k_buf[k_row_off + base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 2u32 {
                store(v_buf[v_row_off + base_row + 0u32], r0.cast::<T>());
                store(v_buf[v_row_off + base_row + 1u32], r1.cast::<T>());
                store(v_buf[v_row_off + base_row + 2u32], r2.cast::<T>());
                store(v_buf[v_row_off + base_row + 3u32], r3.cast::<T>());
            }
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_batched_qkv_qmm_fast;
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

    /// Dequant-then-matmul QMM for one projection: `[M, out_dim]` row-major.
    #[allow(clippy::too_many_arguments)]
    fn naive_qmm(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        m: usize,
        in_dim: usize,
        gs: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / gs;
        let mut out = vec![0.0_f32; m * out_dim];
        for mi in 0..m {
            let xr = &x[mi * in_dim..(mi + 1) * in_dim];
            for row in 0..out_dim {
                let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
                let rs = &scales[row * n_groups..(row + 1) * n_groups];
                let rb = &biases[row * n_groups..(row + 1) * n_groups];
                let mut acc = 0.0_f32;
                for d in 0..in_dim {
                    let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                    let g = d / gs;
                    acc += (q as f32 * rs[g] + rb[g]) * xr[d];
                }
                out[mi * out_dim + row] = acc;
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 2e-1)]
    fn test_batched_qkv_qmm_fast(dt: DType) -> TestSetup {
        let (m, in_dim, gs) = (2usize, 512usize, 64usize);
        let (out_q, out_k, out_v) = (16usize, 8usize, 8usize);
        let x: Vec<f32> =
            source(m * in_dim, 0x11, 2.0, 0.05).iter().map(|&v| round(v, dt)).collect();
        let wq = source(out_q * in_dim, 0x22, 3.0, 0.0);
        let wk = source(out_k * in_dim, 0x33, 3.0, 0.0);
        let wv = source(out_v * in_dim, 0x44, 3.0, 0.0);
        let (wq_p, sq, bq) = quantize_matrix(&wq, out_q, in_dim, gs);
        let (wk_p, sk, bk) = quantize_matrix(&wk, out_k, in_dim, gs);
        let (wv_p, sv, bv) = quantize_matrix(&wv, out_v, in_dim, gs);
        let r = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&v| round(v, dt)).collect() };

        let eq = naive_qmm(&wq_p, &r(&sq), &r(&bq), &x, m, in_dim, gs, out_q);
        let ek = naive_qmm(&wk_p, &r(&sk), &r(&bk), &x, m, in_dim, gs, out_k);
        let ev = naive_qmm(&wv_p, &r(&sv), &r(&bv), &x, m, in_dim, gs, out_v);

        let n_tgs = out_q.max(out_k).max(out_v).div_ceil(8);
        TestSetup::new(ffai_batched_qkv_qmm_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w_q", pack_u32(&wq_p), DType::U32))
            .input(TestBuffer::from_vec("scales_q", pack_f32(&sq, dt), dt))
            .input(TestBuffer::from_vec("biases_q", pack_f32(&bq, dt), dt))
            .input(TestBuffer::from_vec("w_k", pack_u32(&wk_p), DType::U32))
            .input(TestBuffer::from_vec("scales_k", pack_f32(&sk, dt), dt))
            .input(TestBuffer::from_vec("biases_k", pack_f32(&bk, dt), dt))
            .input(TestBuffer::from_vec("w_v", pack_u32(&wv_p), DType::U32))
            .input(TestBuffer::from_vec("scales_v", pack_f32(&sv, dt), dt))
            .input(TestBuffer::from_vec("biases_v", pack_f32(&bv, dt), dt))
            .input(TestBuffer::zeros("q_buf", m * out_q, dt))
            .input(TestBuffer::zeros("k_buf", m * out_k, dt))
            .input(TestBuffer::zeros("v_buf", m * out_v, dt))
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", gs as u32)
            .expect(TestBuffer::from_vec("q_buf", pack_f32(&eq, dt), dt))
            .expect(TestBuffer::from_vec("k_buf", pack_f32(&ek, dt), dt))
            .expect(TestBuffer::from_vec("v_buf", pack_f32(&ev, dt), dt))
            .grid_3d(n_tgs as u32, m as u32, 3, [64, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_batched_qkv_qmm_fast` — MLX-less reduction
/// kernel. Small-batch prefill (M=4) GQA shape: hidden 4096, out_q 4096, k/v 1024.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_batched_qkv_qmm_fast;

    #[bench(name = "ffai/batched_qkv_qmm_fast", dtypes = [f32, f16, bf16])]
    fn bench_batched_qkv_qmm_fast(dt: DType) -> BenchSetup {
        let (m, in_dim, gs) = (4usize, 4096usize, 64usize);
        let (out_q, out_k, out_v) = (4096usize, 1024usize, 1024usize);
        let ng = in_dim / gs;
        let words = |o: usize| o * in_dim / 8;
        let total = words(out_q) + words(out_k) + words(out_v);
        let n_tgs = out_q.max(out_k).max(out_v).div_ceil(8);
        BenchSetup::new(ffai_batched_qkv_qmm_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::random("w_q", words(out_q), DType::U32))
            .buffer(BenchBuffer::random("scales_q", out_q * ng, dt))
            .buffer(BenchBuffer::random("biases_q", out_q * ng, dt))
            .buffer(BenchBuffer::random("w_k", words(out_k), DType::U32))
            .buffer(BenchBuffer::random("scales_k", out_k * ng, dt))
            .buffer(BenchBuffer::random("biases_k", out_k * ng, dt))
            .buffer(BenchBuffer::random("w_v", words(out_v), DType::U32))
            .buffer(BenchBuffer::random("scales_v", out_v * ng, dt))
            .buffer(BenchBuffer::random("biases_v", out_v * ng, dt))
            .buffer(BenchBuffer::zeros("q_buf", m * out_q, dt).output())
            .buffer(BenchBuffer::zeros("k_buf", m * out_k, dt).output())
            .buffer(BenchBuffer::zeros("v_buf", m * out_v, dt).output())
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", gs as u32)
            .bytes_moved((total * 4) as u64)
            .grid_3d(n_tgs as u32, m as u32, 3, [64, 1, 1])
    }
}
