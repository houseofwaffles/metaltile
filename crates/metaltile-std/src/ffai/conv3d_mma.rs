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
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const ALL_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

/// MMA-tiled 3D convolution (stride=1, dilation=1, pad=0).
///
/// Grid `[out_ch/32, (batch*out_d*out_h*out_w)/32, 1]`, tpg = 128.
///
/// Correctness pinned by `conv3d_mma_gpu_correctness`.
#[kernel]
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
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit 5D im2col gather) ───────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            // Decompose kt into (ic, kz, ky, kx).
            let ic = kt / kdhw;
            let rem_kt = kt - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            // Gather indices (stride=1, pad=0).
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let val = load(input[in_idx]).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }

        // ─ 2. Coop B load (weight, dense OIDHW) ─────────────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let w_idx = w_oc_base + kt;
            let val = load(weight[w_idx]).cast::<T>();
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

inventory::submit! {
    BenchSpec {
        op: "conv3d",
        subop: "mma",
        kernel_name: "conv3d_mma",
        kernel_ir: conv3d_mma::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
