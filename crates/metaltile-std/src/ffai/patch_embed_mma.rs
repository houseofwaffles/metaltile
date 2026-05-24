//! MMA-tiled patch embedding.
//!
//! Perf follow-up to `ffai/patch_embed.rs`. The original kernel is one
//! thread per output element `(patch, hidden)` — a good baseline but
//! memory-bandwidth-limited for large hidden dimensions (ViT-L hidden=1024,
//! ViT-H hidden=1280) where each output row (hidden dim) requires reading
//! the entire `patch_dim = in_ch * patch_h * patch_w` weight row.
//!
//! The patch embedding is algebraically identical to the conv2d with
//! `stride = patch`, no overlap, no padding — just a different output
//! layout convention: `[num_patches, hidden]` instead of NCHW. The MMA
//! kernel exploits this:
//!
//!   out[num_patches, hidden] = A[num_patches, patch_dim] × B[patch_dim, hidden]
//!
//! where:
//!   - A is the implicit im2col unfolding — each lane computes its patch
//!     pixel via `(patch_idx, ic, py, px)` → `(ic, py0+py, px0+px)` gather.
//!   - B is the flat weight matrix `[hidden, patch_dim]` (weight row =
//!     the hidden axis). We read it as `B[patch_dim, hidden]^T` — i.e.,
//!     the weight's row is the *hidden* dimension, our "oc" axis.
//!
//! ## Tile geometry (same as `conv2d_mma`)
//!
//!   tpg = 128 = 4 SG × 32 lanes  (2×2 warp grid)
//!   BM = BN = 32, BK = 32        (output tile 32×32)
//!   Grid: [hidden/32, num_patches/32, 1]
//!   TG memory: as[32×36] + bs[32×36] (skew-36 stride)
//!
//! ## A-tile implicit-patch-unfold indexing
//!
//! For flat patch index `pat ∈ [0, num_patches)` and tap `kt ∈ [0, patch_dim)`:
//!
//!   patch_dim = in_ch * patch_h * patch_w
//!   patches_w = in_w / patch_w
//!   py0 = (pat / patches_w) * patch_h
//!   px0 = (pat % patches_w) * patch_w
//!   ic  = kt / (patch_h * patch_w)
//!   py  = (kt % (patch_h * patch_w)) / patch_w
//!   px  = kt % patch_w
//!   A[pat, kt] = image[ic * in_h * in_w + (py0 + py) * in_w + (px0 + px)]
//!
//! ## B-tile (weight) indexing
//!
//! Weight is `[hidden, patch_dim]` (one row per hidden unit). For hidden
//! unit `h` and tap `kt`:
//!
//!   B[h, kt] = weight[h * patch_dim + kt]
//!
//! This is the B matrix in the matmul: `[hidden, patch_dim]` =
//! `[BM_oc, BK]`. In TG memory we lay it out `[BM × BK]` row-major and
//! read it transposed in the MMA step (same as B in `conv2d_mma`).
//!
//! ## Constraints (first cut)
//!
//! - `hidden` and `num_patches` both divisible by 32.
//! - `patch_dim` divisible by 32 (typical: `3*14*14=588` is not, so a
//!   follow-up adds remainder-K handling — the same K-tail that
//!   `mt_qmm_mma_m16` defers).
//! - Single image (no batch — matches `patch_embed.rs` layout).
//!
//! Codegen-only. Correctness validated by `patch_embed_mma_gpu_correctness`.

use metaltile::{bench_kernel, kernel};

/// MMA-tiled patch embedding.
///
/// Grid `[hidden/32, num_patches/32, 1]`, tpg = 128.
///
/// Correctness pinned by `patch_embed_mma_gpu_correctness`.
#[bench_kernel(
    op="patch_embed",
    subop="mma",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
) {
    // BM (hidden-axis) tile = tgid_x * 32, BN (patch-axis) tile = tgid_y * 32.
    let h_tile = tgid_x;
    let pat_tile = tgid_y;

    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;

    // ── 8×8 frag lane mapping ────────────────────────────────────────────
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // ── TG memory: skew-36 stride ─────────────────────────────────────────
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T); // [32 × 36] A (patch unfold)
    threadgroup_alloc("bs", 1152, T); // [32 × 36] B (weight)

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

    // ── Precompute patch-space extents ────────────────────────────────────
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw; // total tap dimension
    let patches_w = in_w / patch_w; // patches along width axis
    let input_plane = in_h * in_w;

    // ── Coop A-load lane assignment (patch unfold) ────────────────────────
    // lane_in_tg = pat_row * 4 + k_quad; pat_row ∈ 0..32, k_quad ∈ 0..4.
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;

    let global_pat = pat_tile * 32u32 + a_pat_row;
    // Decode patch → (py0, px0): top-left pixel of this patch.
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;

    // ── Coop B-load lane assignment (weight) ──────────────────────────────
    let b_h_row = lane_in_tg / 4u32; // which hidden-unit row
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let w_h_base = global_h * patch_dim;

    // ── K-block loop (step 32 through patch_dim) ─────────────────────────
    // K-tail handling: `patch_dim = in_ch * patch_h * patch_w` isn't
    // always a multiple of 32 (e.g. ViT-patch14, in_ch=3 → 588). Both
    // coop loads mask OOB K-taps with `select(kt < patch_dim, load(...), 0)`
    // and clamp the gather index to 0, so the partial K-tile leaves the
    // MMA accumulator correct without reading past the buffers.
    for kb in range(0u32, patch_dim, 32u32) {
        // ─ 1. Coop A load (implicit patch unfold gather) ─────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            // Decompose kt_safe into (ic, py, px).
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }

        // ─ 2. Coop B load (weight `[hidden, patch_dim]` row-major) ───────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let w_idx = w_h_base + kt_safe;
            let raw = load(weight[w_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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

    // ── 4. Add bias and write frags to global out ─────────────────────────
    // out layout: [num_patches, hidden].
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;

    // Load bias for each of the 4 output oc positions this lane writes.
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();

    // c_f00 at (frag_m=0, frag_n=0)
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    // c_f01 at (frag_m=0, frag_n=8)
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    // c_f10 at (frag_m=8, frag_n=0)
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    // c_f11 at (frag_m=8, frag_n=8)
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}
