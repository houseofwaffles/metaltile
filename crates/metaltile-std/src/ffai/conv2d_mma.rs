//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MMA-tiled implicit-GEMM 2D convolution.
//!
//! Perf follow-up to `ffai/conv2d.rs` — the direct-conv kernels there are
//! one-thread-per-output (O(N²) in the pixel×channel sense). For vision
//! encoder shapes where `out_ch` is large (ViT-L hidden=1024, ViT-H 1280)
//! and `batch × out_h × out_w` is large enough to fill a 32×32 tile, the
//! implicit-im2col + simdgroup-matrix MMA path delivers ~5–10× more ALU
//! utilisation.
//!
//! ## Implicit im2col as a matmul
//!
//! Treat the convolution as a GEMM:
//!
//!   out[BN_pixels, BM_oc] = A[BN_pixels, BK] × B[BK, BM_oc]
//!
//! where:
//!   - `BK  = in_ch * kh * kw` (number of filter taps per output position)
//!   - `BN  = batch * out_h * out_w` (number of output positions = "pixels")
//!   - `BM  = out_ch` (output channels)
//!
//! The A matrix (input) is **never materialised** — each lane computes its
//! `(kh, kw, ic)` → `(h_in, w_in, ic)` gather index on-the-fly from the
//! flat (pixel, k_tap) position, doing a scatter-load from the NCHW image.
//! The B matrix (weight, OIHW) is dense and loaded cooperatively.
//!
//! ## Tile geometry (mirrors `mt_qmm_mma`)
//!
//!   tpg = 128 = 4 SG × 32 lanes  (2×2 warp grid: sm = sg/2, sn = sg%2)
//!   BM = BN = 32, BK = 32        (output tile 32×32 — 1024 outputs/TG)
//!   Grid: [out_ch/32, n_pixels/32, batch]
//!   Each SG owns a 16×16 sub-tile: 4 8×8 frags (c_f00..c_f11)
//!
//! TG memory:
//!   as[32 × 36] = 1152 T  (A tile: input gathers, row-major [BN × BK])
//!   bs[32 × 36] = 1152 T  (B tile: weight, row-major [BM × BK])
//!   Skew by 4 (stride = BK+4 = 36) to break 32-bank conflicts.
//!
//! ## Constraints (first cut)
//!
//! - stride = 1, dilation = 1, padding = 0 (vision patch-conv style)
//! - `out_ch` divisible by 32, `n_pixels` (`batch*out_h*out_w`) divisible
//!   by 32.  Padding extensions are a follow-up (same as `mt_qmm_mma` →
//!   `mt_qmm_mma_m16`).
//! - NCHW input, OIHW weight — the standard PyTorch layout.
//!
//! ## A-tile implicit-im2col indexing
//!
//! For a flat pixel position `px ∈ [0, batch×out_h×out_w)` and a flat
//! tap index `kt ∈ [0, in_ch×kh×kw)`:
//!
//!   n    = px / (out_h * out_w)
//!   oh   = (px % (out_h * out_w)) / out_w
//!   ow   = px % out_w
//!   ic   = kt / (kh * kw)
//!   ky   = (kt % (kh * kw)) / kw
//!   kx   = kt % kw
//!   ih   = oh + ky   (stride=1, pad=0)
//!   iw   = ow + kx
//!   A[px, kt] = input[n * in_ch * in_h * in_w + ic * in_h * in_w + ih * in_w + iw]
//!
//! ## B-tile (weight) indexing
//!
//! Weight is OIHW `[out_ch, in_ch, kh, kw]`. For output channel `oc` and
//! tap `kt = ic * kh * kw + ky * kw + kx`:
//!
//!   B[oc, kt] = weight[oc * in_ch * kh * kw + kt]
//!
//! The cooperative weight load mirrors the X-load in `mt_qmm_mma`: the B
//! tile is `[BM_oc × BK_taps]` row-major in TG memory; each of the 128
//! lanes loads 8 contiguous K-elements for one oc-row, then the MMA reads
//! rows of B as K-vectors (no transpose needed vs `mt_qmm_mma`'s W^T).
//!
//! Codegen-only. Correctness validated by `conv2d_mma_gpu_correctness`.

use metaltile::kernel;

/// MMA-tiled 2D convolution (stride=1, dilation=1, pad=0).
///
/// Grid `[out_ch/32, (batch*out_h*out_w)/32, 1]`, tpg = 128.
/// Each TG computes a 32×32 tile of `out[pixels, out_channels]`.
///
/// Correctness pinned by `conv2d_mma_gpu_correctness`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
) {
    // BM (oc-axis) tile = tgid_x * 32, BN (pixel-axis) tile = tgid_y * 32.
    let oc_tile = tgid_x;
    let px_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    // 4 SGs in a 2×2 warp grid: sm = row half, sn = col half.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // ── 8×8 frag lane mapping (Apple steel_gemm layout) ──────────────────
    // Same mapping as `mt_qmm_mma`.
    // Each lane owns 2 elements per 8×8 frag at (fm, fn0) and (fm, fn1).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // ── TG memory: A (input gathers) and B (weight) tiles ─────────────
    // Row stride = BK + 4 = 36 (skew by 4 T to break 32-bank conflicts).
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T); // [32 × 36] A tile
    threadgroup_alloc("bs", 1152, T); // [32 × 36] B tile
    // ── Accumulator frags, init to 0 ─────────────────────────────────────
    // c_f<row_half><col_half>: 4 8×8 frags per SG covering the 16×16 sub-tile.
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
    // A and B frag scratch, reused per k-inner.
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    // ── Precompute K-space (BK = in_ch * kh * kw) ──────────────────────
    let kk = kh * kw;
    let total_k = in_ch * kk; // total tap dimension
    // ── Pixel-axis indices for this TG ──────────────────────────────────
    // Pixel = flat index into batch * out_h * out_w.
    let out_hw = out_h * out_w;
    // Lane mapping for coop A-load: 128 lanes × 8 contiguous K per lane.
    // lane_in_tg = px_row * 4 + k_quad → px_row ∈ 0..32, k_quad ∈ 0..4.
    let a_px_row = lane_in_tg / 4u32; // which of the 32 pixel rows
    let a_k_quad = lane_in_tg & 3u32; // which K-quad (8-elem chunk) in [0,4)
    let a_k_base = a_k_quad * 8u32;
    // Global pixel index for this lane's A row.
    let global_px = px_tile * 32u32 + a_px_row;
    // Decode pixel → (n, oh, ow) for im2col gather.
    let n_px = global_px / out_hw;
    let rem_px = global_px - n_px * out_hw;
    let oh_px = rem_px / out_w;
    let ow_px = rem_px - oh_px * out_w;
    // Base offset into input for this pixel's batch/spatial position.
    let in_n_stride = in_ch * in_h * in_w;
    let px_in_base = n_px * in_n_stride;
    // Lane mapping for coop B-load (weight): same as X-load in mt_qmm_mma.
    // lane_in_tg = oc_row * 4 + k_quad → oc_row ∈ 0..32, k_quad ∈ 0..4.
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    // Weight row base: [oc * total_k] (OIHW flattened).
    let w_oc_base = global_oc * total_k;
    // ── K-block loop (step BK=32 through total_k = in_ch * kh * kw) ────
    // The K-loop steps by 32, but `total_k` is rarely a multiple of 32
    // (e.g. ViT-patch14: in_ch=3, kh=kw=14 → total_k=588). The A/B coop
    // loads use `select(kt < total_k, load(...), 0)` to zero-fill the
    // K-tail; the index itself is clamped to 0 when OOB so the gather
    // never reads past the input/weight buffer. Both A and B contributors
    // are zero, so the partial-K MMA accumulator stays correct.
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit im2col gather) ─────────────────────
        // Each lane loads 8 contiguous K-elements for its pixel row.
        // K-tap index = kb + a_k_base + i, decomposed to (ic, ky, kx):
        //   ic = kt / kk;  ky = (kt % kk) / kw;  kx = kt % kw
        // ih = oh_px + ky (stride=1, pad=0), iw = ow_px + kx.
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            // Clamp the K-tap index to 0 on OOB so the gather stays
            // in-range (the loaded value is masked to 0 by `select` below).
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kk;
            let rem_kt = kt_safe - ic * kk;
            let ky = rem_kt / kw;
            let kx = rem_kt - ky * kw;
            let ih = oh_px + ky;
            let iw = ow_px + kx;
            let in_idx = px_in_base + ic * in_h * in_w + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_px_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (weight, dense OIHW) ─────────────────────────
        // Each lane loads 8 contiguous K-elements for its oc row.
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
        // ─ 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ───
        // A-frag (pixels) at (fm, fn_i): as[(sm*16 + frag_m + fm)*36 + k_inner*8 + fn_i]
        // B-frag (weight) at (fm, fn_i): bs[(sn*16 + frag_n + fn_i)*36 + k_inner*8 + fm]
        // (B is loaded row-major [BM_oc × BK], read as transpose for the
        // [BK × BM_oc] B-matrix — same swap as mt_qmm_mma's W^T read.)
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
    // out layout: [batch * out_h * out_w, out_ch] row-major (pixel-major).
    // out_px_base = pixel-axis base; out_oc_base = oc-axis base.
    let out_px_base = px_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    // Stride along the oc axis (number of columns) = out_ch.
    // c_f00 at (frag_m=0, frag_n=0)
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::conv2d_mma;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle, pixel-major output `[n_pixels, out_ch]`.
    /// stride=1, dilation=1, pad=0, no bias. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv2d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let n_pixels = batch * out_hw;
        let mut out = vec![0.0f32; n_pixels * out_ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let pixel = n * out_hw + oh * out_w + ow;
                    for oc in 0..out_ch {
                        let mut acc = 0.0f32;
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ih = oh + ky;
                                    let iw = ow + kx;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    let w_idx = ((oc * in_ch + ic) * kh + ky) * kw + kx;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        out[pixel * out_ch + oc] = acc;
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
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_pixels % 32, 0, "n_pixels must be a multiple of 32 for the MMA tile");
        let n_out = n_pixels * out_ch;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 2.0);
        let weight_f = ramp(out_ch * in_ch * kh * kw, 11, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let expected = naive_conv2d_mma(&input, &weight, batch, in_ch, in_h, in_w, out_ch, kh, kw);
        TestSetup::new(conv2d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
    }

    // 3×3 conv: in 10×10 → out 8×8, n_pixels=64, out_ch=32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_mma_3x3(dt: DType) -> TestSetup { mma_setup(1, 4, 10, 10, 32, 3, 3, dt) }

    // Multi-tile 1×1: batch=4, 8×8 image → n_pixels=256 (8 tiles), out_ch=32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_mma_multi_tile(dt: DType) -> TestSetup { mma_setup(4, 4, 8, 8, 32, 1, 1, dt) }
}

/// New-syntax bench for `conv2d_mma` (ViT-L patch14 stem, hidden 1024).
/// Reduction mode, `grid_3d(out_ch/32, n_pixels/32, 1, [128,1,1])`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::conv2d_mma;

    #[bench(name = "ffai/conv2d/mma", dtypes = [f32, f16, bf16])]
    fn bench_conv2d_mma(dt: DType) -> BenchSetup {
        // 14×14 stride-1 conv on a 32×32 feature map → out 19×19=361 px
        // → not %32. Pick a 1×1 conv on a 32×32 map: n_pixels=1024, out_ch=1024.
        let (batch, in_ch, in_h, in_w, out_ch, kh, kw) =
            (1usize, 256usize, 32usize, 32usize, 1024usize, 1usize, 1usize);
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        let n_out = n_pixels * out_ch;
        BenchSetup::new(conv2d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kh * kw, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
