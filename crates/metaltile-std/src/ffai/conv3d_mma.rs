//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MMA-tiled implicit-GEMM 3D convolution.
//!
//! Perf follow-up to `ffai/conv3d.rs` — the 3D counterpart of `conv2d_mma`.
//! Same approach: treat the conv as a GEMM where the A matrix is implicit
//! im2col over `(kd, kh, kw, ic)` gather indices, and use simdgroup-matrix
//! MMA to tile the output `[n_voxels, out_ch]` efficiently.
//!
//! ## Implicit im2col as a matmul
//!
//!   out[BN_voxels, BM_oc] = A[BN_voxels, BK] × B[BK, BM_oc]
//!
//! where:
//!   - `BK  = in_ch * kd * kh * kw` (filter taps per output voxel)
//!   - `BN  = batch * out_d * out_h * out_w` (number of output voxels)
//!   - `BM  = out_ch`
//!
//! The A matrix is never materialised — each lane decodes its
//! `(kd, kh, kw, ic)` → `(d_in, h_in, w_in, ic)` gather index on-the-fly.
//!
//! ## Tile geometry (identical to `conv2d_mma`)
//!
//!   tpg = 128 = 4 SG × 32 lanes  (2×2 warp grid)
//!   BM = BN = 32, BK = 32        (output tile 32×32)
//!   Grid: [out_ch/32, (batch*out_d*out_h*out_w)/32, 1]
//!
//! ## Constraints (first cut)
//!
//! - stride = 1, dilation = 1, padding = 0.
//! - `out_ch` and `batch * out_d * out_h * out_w` both divisible by 32.
//! - NCDHW input, OIDHW weight.
//!
//! ## A-tile implicit-im2col indexing (5D)
//!
//! For flat voxel `pv ∈ [0, batch*out_d*out_h*out_w)` and tap `kt ∈ [0, total_k)`:
//!
//!   total_k = in_ch * kd * kh * kw
//!   out_dhw = out_d * out_h * out_w;  out_hw = out_h * out_w
//!   n    = pv / out_dhw
//!   od   = (pv % out_dhw) / out_hw
//!   oh   = (pv % out_hw) / out_w
//!   ow   = pv % out_w
//!   kdhw = kd * kh * kw;  khw = kh * kw
//!   ic   = kt / kdhw
//!   kz   = (kt % kdhw) / khw
//!   ky   = (kt % khw) / kw
//!   kx   = kt % kw
//!   id   = od + kz   (stride=1, pad=0)
//!   ih   = oh + ky
//!   iw   = ow + kx
//!   A[pv, kt] = input[n*in_ch*in_d*in_h*in_w + ic*in_d*in_h*in_w + id*in_h*in_w + ih*in_w + iw]
//!
//! Codegen-only. Correctness validated by `conv3d_mma_gpu_correctness`.

use metaltile::kernel;

/// MMA-tiled 3D convolution (stride=1, dilation=1, pad=0).
///
/// Grid `[out_ch/32, (batch*out_d*out_h*out_w)/32, 1]`, tpg = 128.
///
/// Correctness pinned by `conv3d_mma_gpu_correctness`.
#[kernel(
    bench(
        op="conv3d",
        subop="mma",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
) {
    // BM (oc-axis) tile = tgid_x * 32, BN (voxel-axis) tile = tgid_y * 32.
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // ── 8×8 frag lane mapping (Apple steel_gemm layout) ──────────────────
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // ── TG memory: A and B tiles, skewed stride = 36 ─────────────────────
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T);
    threadgroup_alloc("bs", 1152, T);
    // ── Accumulator frags ─────────────────────────────────────────────────
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
    // ── Precompute K-space extents ────────────────────────────────────────
    let khw = kh * kw;
    let kdhw = kd * khw; // taps per input channel
    let total_k = in_ch * kdhw; // total tap dimension
    // ── Voxel-axis im2col decode for this TG's A rows ────────────────────
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    // Coop A-load lane assignment: lane_in_tg = pv_row * 4 + k_quad.
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    // Base device offset for this voxel's batch + channel-0 position.
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    // Coop B-load (weight).
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let w_oc_base = global_oc * total_k;
    // ── K-block loop ──────────────────────────────────────────────────────
    // K-tail handling: `total_k = in_ch * kd * kh * kw` rarely lands on
    // a multiple of 32 (e.g. in_ch=4, k=3³ → 108). The A/B coop loads
    // mask out-of-bound K-taps with `select(kt < total_k, load(...), 0)`
    // and clamp the gather index to 0 on OOB so we never read past the
    // input/weight buffer. Zero contributions on both sides leave the
    // partial-K MMA accumulator correct.
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit 5D im2col gather) ───────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            // Decompose kt_safe into (ic, kz, ky, kx).
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            // Gather indices (stride=1, pad=0).
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (weight, dense OIDHW) ─────────────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let w_idx = w_oc_base + kt_safe;
            let raw = load(weight[w_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop (4 k-inner × 4 frags = 16 MMAs / SG) ──────
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write 4 C frags to global out ─────────────────────────────────
    // out layout: [batch * out_d * out_h * out_w, out_ch].
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::conv3d_mma;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 3D conv oracle, voxel-major output `[n_voxels, out_ch]`.
    /// stride=1, dilation=1, pad=0, no bias. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv3d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let out_dhw = out_d * out_hw;
        let n_voxels = batch * out_dhw;
        let in_plane = in_h * in_w;
        let in_vol = in_d * in_plane;
        let mut out = vec![0.0f32; n_voxels * out_ch];
        for n in 0..batch {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let voxel = n * out_dhw + od * out_hw + oh * out_w + ow;
                        for oc in 0..out_ch {
                            let mut acc = 0.0f32;
                            for ic in 0..in_ch {
                                for kz in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let id = od + kz;
                                            let ih = oh + ky;
                                            let iw = ow + kx;
                                            let in_idx = n * in_ch * in_vol
                                                + ic * in_vol
                                                + id * in_plane
                                                + ih * in_w
                                                + iw;
                                            let w_idx = oc * in_ch * kd * kh * kw
                                                + ic * kd * kh * kw
                                                + kz * kh * kw
                                                + ky * kw
                                                + kx;
                                            acc += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            out[voxel * out_ch + oc] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_voxels % 32, 0, "n_voxels must be a multiple of 32 for the MMA tile");
        let n_out = n_voxels * out_ch;
        let input_f = ramp(batch * in_ch * in_d * in_h * in_w, 13, 2.0);
        let weight_f = ramp(out_ch * in_ch * kd * kh * kw, 11, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let expected =
            naive_conv3d_mma(&input, &weight, batch, in_ch, in_d, in_h, in_w, out_ch, kd, kh, kw);
        TestSetup::new(conv3d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_ch / 32) as u32, (n_voxels / 32) as u32, 1, [128, 1, 1])
    }

    // 1×1×1 conv: in 8×8×8 → n_voxels=512 (16 tiles), out_ch=32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_mma_1x1x1(dt: DType) -> TestSetup { mma_setup(1, 2, 8, 8, 8, 32, 1, 1, 1, dt) }

    // Multi-batch 1×1×1: batch=4, 4×4×4 → n_voxels=256 (8 tiles), out_ch=32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_mma_multi_batch(dt: DType) -> TestSetup {
        mma_setup(4, 4, 4, 4, 4, 32, 1, 1, 1, dt)
    }
}

/// New-syntax bench for `conv3d_mma` (volumetric 1×1×1 projection).
/// Reduction mode, `grid_3d(out_ch/32, n_voxels/32, 1, [128,1,1])`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::conv3d_mma;

    #[bench(name = "ffai/conv3d/mma", dtypes = [f32, f16, bf16])]
    fn bench_conv3d_mma(dt: DType) -> BenchSetup {
        // 1×1×1 conv on a 16×16×16 volume → n_voxels=4096, out_ch=256.
        let (batch, in_ch, in_d, in_h, in_w, out_ch) =
            (1usize, 64usize, 16usize, 16usize, 16usize, 256usize);
        let (kd, kh, kw) = (1usize, 1usize, 1usize);
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        let n_out = n_voxels * out_ch;
        BenchSetup::new(conv3d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_d * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kd * kh * kw, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .grid_3d((out_ch / 32) as u32, (n_voxels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
