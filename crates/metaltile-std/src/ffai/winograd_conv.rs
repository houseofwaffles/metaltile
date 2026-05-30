//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Winograd fast convolution — F(2×2, 3×3).
//!
//! The 3×3 stride-1 convolution is the workhorse of every CNN vision
//! backbone (ResNet stems, the conv layers inside ConvNeXt / EfficientViT
//! hybrids, the depthwise-separable blocks in MobileNet-class encoders).
//! A direct conv spends `2·2·3·3 = 36` multiplies per 2×2 output tile;
//! the Winograd F(2×2, 3×3) minimal-filtering algorithm computes the same
//! tile with only the `4·4 = 16` element-wise products of the transformed
//! domain — a 2.25× cut in multiplies. The remaining work is three small
//! fixed transforms (input, filter, output) built from adds and shifts.
//!
//! This closes the Winograd row of `docs/KERNEL_AUDIT.md`: the direct
//! `naive_unfold` / depthwise paths are already covered by `conv2d.rs`
//! and `conv3d.rs`; Winograd is the 3×3-stride-1 perf specialization.
//!
//! ## The algorithm
//!
//! For one 2×2 output tile, with `d` the 4×4 input tile and `g` the 3×3
//! filter:
//!
//!   V = Bᵀ · d · B          (input transform,  4×4)
//!   U = G  · g · Gᵀ         (filter transform, 4×4)
//!   M = U ⊙ V               (element-wise product, summed over in_ch)
//!   Y = Aᵀ · M · A          (output transform,  2×2)
//!
//! with the F(2×2, 3×3) transform matrices
//!
//!   Bᵀ = ⎡ 1  0 -1  0 ⎤   G = ⎡ 1     0    0   ⎤   Aᵀ = ⎡ 1  1  1  0 ⎤
//!        ⎢ 0  1  1  0 ⎥       ⎢ ½     ½    ½   ⎥        ⎣ 0  1 -1 -1 ⎦
//!        ⎢ 0 -1  1  0 ⎥       ⎢ ½    -½    ½   ⎥
//!        ⎣ 0  1  0 -1 ⎦       ⎣ 0     0    1   ⎦
//!
//! `M` is accumulated over the input channels before the output
//! transform — the transform is linear, so summing in the transformed
//! domain and transforming once is exact and cheaper than transforming
//! per channel.
//!
//! ## Layouts
//!
//! NCHW input, OIHW weight — the PyTorch / safetensors default:
//!
//!   input    [batch, in_ch,  in_h,  in_w]    T
//!   weight   [out_ch, in_ch, 3,     3]       T
//!   bias     [out_ch]                        T
//!   out      [batch, out_ch, out_h, out_w]   T
//!
//! One thread computes one 2×2 output tile of one `(batch, out_ch)`
//! plane. The filter transform is recomputed per tile per channel — a
//! correct-first implementation; hoisting it into a separate
//! filter-transform kernel (the cuDNN split: transform → batched GEMM →
//! untransform) is the perf follow-up noted in `KERNEL_AUDIT.md`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Grid3D**, one thread per output tile over
//!   `batch · out_ch · tiles_h · tiles_w`.
//! - **`out_h` and `out_w` must be even.** F(2×2, 3×3) emits a 2×2 tile;
//!   every tile is fully in-bounds only when the output dims are even, so
//!   `tiles_h = out_h / 2`, `tiles_w = out_w / 2` partition the output
//!   exactly. Odd-sized outputs are out of scope — `conv2d_generic`
//!   handles those as a direct conv.
//! - **Kernel size is fixed at 3×3, stride at 1.** Padding is a runtime
//!   constexpr; padded taps outside the real image contribute zero.
//!
//! Codegen-only. Correctness validated by `winograd_conv_gpu_correctness`.

use metaltile::kernel;

/// Winograd F(2×2, 3×3) convolution. See the module docs for the
/// algorithm and the dispatch invariants.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn winograd_conv2d_3x3<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    // tiles_h = out_h / 2, tiles_w = out_w / 2 — passed pre-divided so
    // the flat-index decode never divides by a non-constant.
    #[constexpr] tiles_h: u32,
    #[constexpr] tiles_w: u32,
) {
    // Flat tile index → (n, oc, th, tw). One thread per 2×2 output tile.
    let idx = program_id::<0>();
    let tw = idx % tiles_w;
    let r1 = idx / tiles_w;
    let th = r1 % tiles_h;
    let r2 = r1 / tiles_h;
    let oc = r2 % out_ch;
    let n = r2 / out_ch;
    // Input tile origin in the *padded* frame. Output rows 2·th and
    // 2·th+1 have receptive fields starting at padded rows 2·th and
    // 2·th+1, so the 4×4 tile spans padded rows [2·th, 2·th+3]. A real
    // pixel at padded row `pr` sits at unpadded row `pr - pad_h`, valid
    // iff `pad_h <= pr < pad_h + in_h` (same trick as `conv2d.rs`).
    let pr0 = th * 2u32;
    let pc0 = tw * 2u32;
    // Per-row / per-column validity + unpadded coordinate, computed once
    // and reused across all 16 tile loads.
    let pr_0 = pr0;
    let pr_1 = pr0 + 1u32;
    let pr_2 = pr0 + 2u32;
    let pr_3 = pr0 + 3u32;
    let row_ok_0 = (pr_0 >= pad_h) & (pr_0 < pad_h + in_h);
    let row_ok_1 = (pr_1 >= pad_h) & (pr_1 < pad_h + in_h);
    let row_ok_2 = (pr_2 >= pad_h) & (pr_2 < pad_h + in_h);
    let row_ok_3 = (pr_3 >= pad_h) & (pr_3 < pad_h + in_h);
    let ih_0 = select(row_ok_0, pr_0 - pad_h, 0u32);
    let ih_1 = select(row_ok_1, pr_1 - pad_h, 0u32);
    let ih_2 = select(row_ok_2, pr_2 - pad_h, 0u32);
    let ih_3 = select(row_ok_3, pr_3 - pad_h, 0u32);
    let pc_0 = pc0;
    let pc_1 = pc0 + 1u32;
    let pc_2 = pc0 + 2u32;
    let pc_3 = pc0 + 3u32;
    let col_ok_0 = (pc_0 >= pad_w) & (pc_0 < pad_w + in_w);
    let col_ok_1 = (pc_1 >= pad_w) & (pc_1 < pad_w + in_w);
    let col_ok_2 = (pc_2 >= pad_w) & (pc_2 < pad_w + in_w);
    let col_ok_3 = (pc_3 >= pad_w) & (pc_3 < pad_w + in_w);
    let iw_0 = select(col_ok_0, pc_0 - pad_w, 0u32);
    let iw_1 = select(col_ok_1, pc_1 - pad_w, 0u32);
    let iw_2 = select(col_ok_2, pc_2 - pad_w, 0u32);
    let iw_3 = select(col_ok_3, pc_3 - pad_w, 0u32);
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let n_base = n * in_n_stride;
    // Weight: [out_ch, in_ch, 3, 3] — 9 contiguous taps per (oc, ic).
    let w_oc_base = oc * in_ch * 9u32;
    // M = Σ_ic (U ⊙ V) — the 4×4 transformed-domain accumulator.
    let mut m00 = 0.0f32;
    let mut m01 = 0.0f32;
    let mut m02 = 0.0f32;
    let mut m03 = 0.0f32;
    let mut m10 = 0.0f32;
    let mut m11 = 0.0f32;
    let mut m12 = 0.0f32;
    let mut m13 = 0.0f32;
    let mut m20 = 0.0f32;
    let mut m21 = 0.0f32;
    let mut m22 = 0.0f32;
    let mut m23 = 0.0f32;
    let mut m30 = 0.0f32;
    let mut m31 = 0.0f32;
    let mut m32 = 0.0f32;
    let mut m33 = 0.0f32;
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n_base + ic * input_plane;
        let row0 = in_ic_base + ih_0 * in_w;
        let row1 = in_ic_base + ih_1 * in_w;
        let row2 = in_ic_base + ih_2 * in_w;
        let row3 = in_ic_base + ih_3 * in_w;
        // Load the 4×4 input tile; padded taps contribute zero.
        let d00 = select(row_ok_0 & col_ok_0, load(input[row0 + iw_0]).cast::<f32>(), 0.0f32);
        let d01 = select(row_ok_0 & col_ok_1, load(input[row0 + iw_1]).cast::<f32>(), 0.0f32);
        let d02 = select(row_ok_0 & col_ok_2, load(input[row0 + iw_2]).cast::<f32>(), 0.0f32);
        let d03 = select(row_ok_0 & col_ok_3, load(input[row0 + iw_3]).cast::<f32>(), 0.0f32);
        let d10 = select(row_ok_1 & col_ok_0, load(input[row1 + iw_0]).cast::<f32>(), 0.0f32);
        let d11 = select(row_ok_1 & col_ok_1, load(input[row1 + iw_1]).cast::<f32>(), 0.0f32);
        let d12 = select(row_ok_1 & col_ok_2, load(input[row1 + iw_2]).cast::<f32>(), 0.0f32);
        let d13 = select(row_ok_1 & col_ok_3, load(input[row1 + iw_3]).cast::<f32>(), 0.0f32);
        let d20 = select(row_ok_2 & col_ok_0, load(input[row2 + iw_0]).cast::<f32>(), 0.0f32);
        let d21 = select(row_ok_2 & col_ok_1, load(input[row2 + iw_1]).cast::<f32>(), 0.0f32);
        let d22 = select(row_ok_2 & col_ok_2, load(input[row2 + iw_2]).cast::<f32>(), 0.0f32);
        let d23 = select(row_ok_2 & col_ok_3, load(input[row2 + iw_3]).cast::<f32>(), 0.0f32);
        let d30 = select(row_ok_3 & col_ok_0, load(input[row3 + iw_0]).cast::<f32>(), 0.0f32);
        let d31 = select(row_ok_3 & col_ok_1, load(input[row3 + iw_1]).cast::<f32>(), 0.0f32);
        let d32 = select(row_ok_3 & col_ok_2, load(input[row3 + iw_2]).cast::<f32>(), 0.0f32);
        let d33 = select(row_ok_3 & col_ok_3, load(input[row3 + iw_3]).cast::<f32>(), 0.0f32);
        // Input transform V = Bᵀ·d·B. First t = Bᵀ·d (rows mix), then
        // V = t·B (columns mix). Bᵀ rows: [1,0,-1,0] [0,1,1,0]
        // [0,-1,1,0] [0,1,0,-1].
        let t00 = d00 - d20;
        let t01 = d01 - d21;
        let t02 = d02 - d22;
        let t03 = d03 - d23;
        let t10 = d10 + d20;
        let t11 = d11 + d21;
        let t12 = d12 + d22;
        let t13 = d13 + d23;
        let t20 = d20 - d10;
        let t21 = d21 - d11;
        let t22 = d22 - d12;
        let t23 = d23 - d13;
        let t30 = d10 - d30;
        let t31 = d11 - d31;
        let t32 = d12 - d32;
        let t33 = d13 - d33;
        // V[r][·] = [tr0-tr2, tr1+tr2, tr2-tr1, tr1-tr3].
        let v00 = t00 - t02;
        let v01 = t01 + t02;
        let v02 = t02 - t01;
        let v03 = t01 - t03;
        let v10 = t10 - t12;
        let v11 = t11 + t12;
        let v12 = t12 - t11;
        let v13 = t11 - t13;
        let v20 = t20 - t22;
        let v21 = t21 + t22;
        let v22 = t22 - t21;
        let v23 = t21 - t23;
        let v30 = t30 - t32;
        let v31 = t31 + t32;
        let v32 = t32 - t31;
        let v33 = t31 - t33;
        // Load the 3×3 filter for this (oc, ic).
        let w_base = w_oc_base + ic * 9u32;
        let g00 = load(weight[w_base + 0u32]).cast::<f32>();
        let g01 = load(weight[w_base + 1u32]).cast::<f32>();
        let g02 = load(weight[w_base + 2u32]).cast::<f32>();
        let g10 = load(weight[w_base + 3u32]).cast::<f32>();
        let g11 = load(weight[w_base + 4u32]).cast::<f32>();
        let g12 = load(weight[w_base + 5u32]).cast::<f32>();
        let g20 = load(weight[w_base + 6u32]).cast::<f32>();
        let g21 = load(weight[w_base + 7u32]).cast::<f32>();
        let g22 = load(weight[w_base + 8u32]).cast::<f32>();
        // Filter transform U = G·g·Gᵀ. First s = G·g (rows mix), then
        // U = s·Gᵀ (columns mix). G rows: [1,0,0] [½,½,½] [½,-½,½]
        // [0,0,1].
        let s00 = g00;
        let s01 = g01;
        let s02 = g02;
        let s10 = 0.5f32 * (g00 + g10 + g20);
        let s11 = 0.5f32 * (g01 + g11 + g21);
        let s12 = 0.5f32 * (g02 + g12 + g22);
        let s20 = 0.5f32 * (g00 - g10 + g20);
        let s21 = 0.5f32 * (g01 - g11 + g21);
        let s22 = 0.5f32 * (g02 - g12 + g22);
        let s30 = g20;
        let s31 = g21;
        let s32 = g22;
        // U[i][·] = [si0, ½(si0+si1+si2), ½(si0-si1+si2), si2].
        let u00 = s00;
        let u01 = 0.5f32 * (s00 + s01 + s02);
        let u02 = 0.5f32 * (s00 - s01 + s02);
        let u03 = s02;
        let u10 = s10;
        let u11 = 0.5f32 * (s10 + s11 + s12);
        let u12 = 0.5f32 * (s10 - s11 + s12);
        let u13 = s12;
        let u20 = s20;
        let u21 = 0.5f32 * (s20 + s21 + s22);
        let u22 = 0.5f32 * (s20 - s21 + s22);
        let u23 = s22;
        let u30 = s30;
        let u31 = 0.5f32 * (s30 + s31 + s32);
        let u32 = 0.5f32 * (s30 - s31 + s32);
        let u33 = s32;
        // Element-wise product, accumulated across input channels.
        m00 = m00 + u00 * v00;
        m01 = m01 + u01 * v01;
        m02 = m02 + u02 * v02;
        m03 = m03 + u03 * v03;
        m10 = m10 + u10 * v10;
        m11 = m11 + u11 * v11;
        m12 = m12 + u12 * v12;
        m13 = m13 + u13 * v13;
        m20 = m20 + u20 * v20;
        m21 = m21 + u21 * v21;
        m22 = m22 + u22 * v22;
        m23 = m23 + u23 * v23;
        m30 = m30 + u30 * v30;
        m31 = m31 + u31 * v31;
        m32 = m32 + u32 * v32;
        m33 = m33 + u33 * v33;
    }
    // Output transform Y = Aᵀ·M·A. First p = Aᵀ·M (rows mix), then
    // Y = p·A (columns mix). Aᵀ rows: [1,1,1,0] [0,1,-1,-1].
    let p00 = m00 + m10 + m20;
    let p01 = m01 + m11 + m21;
    let p02 = m02 + m12 + m22;
    let p03 = m03 + m13 + m23;
    let p10 = m10 - m20 - m30;
    let p11 = m11 - m21 - m31;
    let p12 = m12 - m22 - m32;
    let p13 = m13 - m23 - m33;
    // Y[i][·] = [pi0+pi1+pi2, pi1-pi2-pi3].
    let bias_v = load(bias[oc]).cast::<f32>();
    let y00 = p00 + p01 + p02 + bias_v;
    let y01 = p01 - p02 - p03 + bias_v;
    let y10 = p10 + p11 + p12 + bias_v;
    let y11 = p11 - p12 - p13 + bias_v;
    // Scatter the 2×2 tile. out_h / out_w are even (dispatch invariant),
    // so every (oh, ow) is in-bounds.
    let out_plane = out_h * out_w;
    let out_oc_base = (n * out_ch + oc) * out_plane;
    let oh0 = th * 2u32;
    let ow0 = tw * 2u32;
    let out_row0 = out_oc_base + oh0 * out_w;
    let out_row1 = out_oc_base + (oh0 + 1u32) * out_w;
    store(out[out_row0 + ow0], y00.cast::<T>());
    store(out[out_row0 + ow0 + 1u32], y01.cast::<T>());
    store(out[out_row1 + ow0], y10.cast::<T>());
    store(out[out_row1 + ow0 + 1u32], y11.cast::<T>());
}

// ─────────────────────────────────────────────────────────────────────────
// cuDNN-style split: hoist the filter transform.
//
// `winograd_conv2d_3x3` recomputes `U = G·g·Gᵀ` for every output tile —
// the filter transform of an `(oc, ic)` pair is redundantly done
// `tiles_h·tiles_w` times. The split pre-transforms every filter once
// (`winograd_filter_transform_3x3`) into a `[out_ch, in_ch, 4, 4]` buffer,
// and `winograd_conv2d_3x3_split` loads those 16 values instead of the 9
// raw taps + the transform. Two dispatches, but the O(tiles) redundant
// transform work is gone.
// ─────────────────────────────────────────────────────────────────────────

/// Pre-transform every 3×3 filter into its 4×4 Winograd form
/// `U = G·g·Gᵀ`. One thread per `(oc, ic)` pair; `u` is `[out_ch, in_ch,
/// 4, 4]` row-major. Dispatch: Grid3D, `program_id<0>` over
/// `out_ch · in_ch`.
#[kernel]
pub fn winograd_filter_transform_3x3<T>(
    weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] out_ch: u32,
) {
    let idx = program_id::<0>();
    let total = out_ch * in_ch;
    if idx < total {
        let w_base = idx * 9u32;
        let g00 = load(weight[w_base + 0u32]).cast::<f32>();
        let g01 = load(weight[w_base + 1u32]).cast::<f32>();
        let g02 = load(weight[w_base + 2u32]).cast::<f32>();
        let g10 = load(weight[w_base + 3u32]).cast::<f32>();
        let g11 = load(weight[w_base + 4u32]).cast::<f32>();
        let g12 = load(weight[w_base + 5u32]).cast::<f32>();
        let g20 = load(weight[w_base + 6u32]).cast::<f32>();
        let g21 = load(weight[w_base + 7u32]).cast::<f32>();
        let g22 = load(weight[w_base + 8u32]).cast::<f32>();
        // s = G·g (rows mix). G rows: [1,0,0] [½,½,½] [½,-½,½] [0,0,1].
        let s00 = g00;
        let s01 = g01;
        let s02 = g02;
        let s10 = 0.5f32 * (g00 + g10 + g20);
        let s11 = 0.5f32 * (g01 + g11 + g21);
        let s12 = 0.5f32 * (g02 + g12 + g22);
        let s20 = 0.5f32 * (g00 - g10 + g20);
        let s21 = 0.5f32 * (g01 - g11 + g21);
        let s22 = 0.5f32 * (g02 - g12 + g22);
        let s30 = g20;
        let s31 = g21;
        let s32 = g22;
        // U = s·Gᵀ (columns mix).
        let u_base = idx * 16u32;
        store(out[u_base + 0u32], s00.cast::<T>());
        store(out[u_base + 1u32], (0.5f32 * (s00 + s01 + s02)).cast::<T>());
        store(out[u_base + 2u32], (0.5f32 * (s00 - s01 + s02)).cast::<T>());
        store(out[u_base + 3u32], s02.cast::<T>());
        store(out[u_base + 4u32], s10.cast::<T>());
        store(out[u_base + 5u32], (0.5f32 * (s10 + s11 + s12)).cast::<T>());
        store(out[u_base + 6u32], (0.5f32 * (s10 - s11 + s12)).cast::<T>());
        store(out[u_base + 7u32], s12.cast::<T>());
        store(out[u_base + 8u32], s20.cast::<T>());
        store(out[u_base + 9u32], (0.5f32 * (s20 + s21 + s22)).cast::<T>());
        store(out[u_base + 10u32], (0.5f32 * (s20 - s21 + s22)).cast::<T>());
        store(out[u_base + 11u32], s22.cast::<T>());
        store(out[u_base + 12u32], s30.cast::<T>());
        store(out[u_base + 13u32], (0.5f32 * (s30 + s31 + s32)).cast::<T>());
        store(out[u_base + 14u32], (0.5f32 * (s30 - s31 + s32)).cast::<T>());
        store(out[u_base + 15u32], s32.cast::<T>());
    }
}

/// Winograd F(2×2, 3×3) convolution consuming a *pre-transformed* filter
/// buffer `u` (`[out_ch, in_ch, 4, 4]`, produced by
/// `winograd_filter_transform_3x3`). Identical to `winograd_conv2d_3x3`
/// except the per-`(oc, ic)` filter transform is replaced by 16 loads of
/// the precomputed `U`. Pair them: filter-transform once, then this.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn winograd_conv2d_3x3_split<T>(
    input: Tensor<T>,
    u: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] tiles_h: u32,
    #[constexpr] tiles_w: u32,
) {
    let idx = program_id::<0>();
    let tw = idx % tiles_w;
    let r1 = idx / tiles_w;
    let th = r1 % tiles_h;
    let r2 = r1 / tiles_h;
    let oc = r2 % out_ch;
    let n = r2 / out_ch;
    let pr0 = th * 2u32;
    let pc0 = tw * 2u32;
    let pr_0 = pr0;
    let pr_1 = pr0 + 1u32;
    let pr_2 = pr0 + 2u32;
    let pr_3 = pr0 + 3u32;
    let row_ok_0 = (pr_0 >= pad_h) & (pr_0 < pad_h + in_h);
    let row_ok_1 = (pr_1 >= pad_h) & (pr_1 < pad_h + in_h);
    let row_ok_2 = (pr_2 >= pad_h) & (pr_2 < pad_h + in_h);
    let row_ok_3 = (pr_3 >= pad_h) & (pr_3 < pad_h + in_h);
    let ih_0 = select(row_ok_0, pr_0 - pad_h, 0u32);
    let ih_1 = select(row_ok_1, pr_1 - pad_h, 0u32);
    let ih_2 = select(row_ok_2, pr_2 - pad_h, 0u32);
    let ih_3 = select(row_ok_3, pr_3 - pad_h, 0u32);
    let pc_0 = pc0;
    let pc_1 = pc0 + 1u32;
    let pc_2 = pc0 + 2u32;
    let pc_3 = pc0 + 3u32;
    let col_ok_0 = (pc_0 >= pad_w) & (pc_0 < pad_w + in_w);
    let col_ok_1 = (pc_1 >= pad_w) & (pc_1 < pad_w + in_w);
    let col_ok_2 = (pc_2 >= pad_w) & (pc_2 < pad_w + in_w);
    let col_ok_3 = (pc_3 >= pad_w) & (pc_3 < pad_w + in_w);
    let iw_0 = select(col_ok_0, pc_0 - pad_w, 0u32);
    let iw_1 = select(col_ok_1, pc_1 - pad_w, 0u32);
    let iw_2 = select(col_ok_2, pc_2 - pad_w, 0u32);
    let iw_3 = select(col_ok_3, pc_3 - pad_w, 0u32);
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let n_base = n * in_n_stride;
    let u_oc_base = oc * in_ch * 16u32;
    let mut m00 = 0.0f32;
    let mut m01 = 0.0f32;
    let mut m02 = 0.0f32;
    let mut m03 = 0.0f32;
    let mut m10 = 0.0f32;
    let mut m11 = 0.0f32;
    let mut m12 = 0.0f32;
    let mut m13 = 0.0f32;
    let mut m20 = 0.0f32;
    let mut m21 = 0.0f32;
    let mut m22 = 0.0f32;
    let mut m23 = 0.0f32;
    let mut m30 = 0.0f32;
    let mut m31 = 0.0f32;
    let mut m32 = 0.0f32;
    let mut m33 = 0.0f32;
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n_base + ic * input_plane;
        let row0 = in_ic_base + ih_0 * in_w;
        let row1 = in_ic_base + ih_1 * in_w;
        let row2 = in_ic_base + ih_2 * in_w;
        let row3 = in_ic_base + ih_3 * in_w;
        let d00 = select(row_ok_0 & col_ok_0, load(input[row0 + iw_0]).cast::<f32>(), 0.0f32);
        let d01 = select(row_ok_0 & col_ok_1, load(input[row0 + iw_1]).cast::<f32>(), 0.0f32);
        let d02 = select(row_ok_0 & col_ok_2, load(input[row0 + iw_2]).cast::<f32>(), 0.0f32);
        let d03 = select(row_ok_0 & col_ok_3, load(input[row0 + iw_3]).cast::<f32>(), 0.0f32);
        let d10 = select(row_ok_1 & col_ok_0, load(input[row1 + iw_0]).cast::<f32>(), 0.0f32);
        let d11 = select(row_ok_1 & col_ok_1, load(input[row1 + iw_1]).cast::<f32>(), 0.0f32);
        let d12 = select(row_ok_1 & col_ok_2, load(input[row1 + iw_2]).cast::<f32>(), 0.0f32);
        let d13 = select(row_ok_1 & col_ok_3, load(input[row1 + iw_3]).cast::<f32>(), 0.0f32);
        let d20 = select(row_ok_2 & col_ok_0, load(input[row2 + iw_0]).cast::<f32>(), 0.0f32);
        let d21 = select(row_ok_2 & col_ok_1, load(input[row2 + iw_1]).cast::<f32>(), 0.0f32);
        let d22 = select(row_ok_2 & col_ok_2, load(input[row2 + iw_2]).cast::<f32>(), 0.0f32);
        let d23 = select(row_ok_2 & col_ok_3, load(input[row2 + iw_3]).cast::<f32>(), 0.0f32);
        let d30 = select(row_ok_3 & col_ok_0, load(input[row3 + iw_0]).cast::<f32>(), 0.0f32);
        let d31 = select(row_ok_3 & col_ok_1, load(input[row3 + iw_1]).cast::<f32>(), 0.0f32);
        let d32 = select(row_ok_3 & col_ok_2, load(input[row3 + iw_2]).cast::<f32>(), 0.0f32);
        let d33 = select(row_ok_3 & col_ok_3, load(input[row3 + iw_3]).cast::<f32>(), 0.0f32);
        let t00 = d00 - d20;
        let t01 = d01 - d21;
        let t02 = d02 - d22;
        let t03 = d03 - d23;
        let t10 = d10 + d20;
        let t11 = d11 + d21;
        let t12 = d12 + d22;
        let t13 = d13 + d23;
        let t20 = d20 - d10;
        let t21 = d21 - d11;
        let t22 = d22 - d12;
        let t23 = d23 - d13;
        let t30 = d10 - d30;
        let t31 = d11 - d31;
        let t32 = d12 - d32;
        let t33 = d13 - d33;
        let v00 = t00 - t02;
        let v01 = t01 + t02;
        let v02 = t02 - t01;
        let v03 = t01 - t03;
        let v10 = t10 - t12;
        let v11 = t11 + t12;
        let v12 = t12 - t11;
        let v13 = t11 - t13;
        let v20 = t20 - t22;
        let v21 = t21 + t22;
        let v22 = t22 - t21;
        let v23 = t21 - t23;
        let v30 = t30 - t32;
        let v31 = t31 + t32;
        let v32 = t32 - t31;
        let v33 = t31 - t33;
        // Load the pre-transformed 4×4 filter U for this (oc, ic).
        let u_base = u_oc_base + ic * 16u32;
        let u00 = load(u[u_base + 0u32]).cast::<f32>();
        let u01 = load(u[u_base + 1u32]).cast::<f32>();
        let u02 = load(u[u_base + 2u32]).cast::<f32>();
        let u03 = load(u[u_base + 3u32]).cast::<f32>();
        let u10 = load(u[u_base + 4u32]).cast::<f32>();
        let u11 = load(u[u_base + 5u32]).cast::<f32>();
        let u12 = load(u[u_base + 6u32]).cast::<f32>();
        let u13 = load(u[u_base + 7u32]).cast::<f32>();
        let u20 = load(u[u_base + 8u32]).cast::<f32>();
        let u21 = load(u[u_base + 9u32]).cast::<f32>();
        let u22 = load(u[u_base + 10u32]).cast::<f32>();
        let u23 = load(u[u_base + 11u32]).cast::<f32>();
        let u30 = load(u[u_base + 12u32]).cast::<f32>();
        let u31 = load(u[u_base + 13u32]).cast::<f32>();
        let u32_ = load(u[u_base + 14u32]).cast::<f32>();
        let u33 = load(u[u_base + 15u32]).cast::<f32>();
        m00 = m00 + u00 * v00;
        m01 = m01 + u01 * v01;
        m02 = m02 + u02 * v02;
        m03 = m03 + u03 * v03;
        m10 = m10 + u10 * v10;
        m11 = m11 + u11 * v11;
        m12 = m12 + u12 * v12;
        m13 = m13 + u13 * v13;
        m20 = m20 + u20 * v20;
        m21 = m21 + u21 * v21;
        m22 = m22 + u22 * v22;
        m23 = m23 + u23 * v23;
        m30 = m30 + u30 * v30;
        m31 = m31 + u31 * v31;
        m32 = m32 + u32_ * v32;
        m33 = m33 + u33 * v33;
    }
    let p00 = m00 + m10 + m20;
    let p01 = m01 + m11 + m21;
    let p02 = m02 + m12 + m22;
    let p03 = m03 + m13 + m23;
    let p10 = m10 - m20 - m30;
    let p11 = m11 - m21 - m31;
    let p12 = m12 - m22 - m32;
    let p13 = m13 - m23 - m33;
    let bias_v = load(bias[oc]).cast::<f32>();
    let y00 = p00 + p01 + p02 + bias_v;
    let y01 = p01 - p02 - p03 + bias_v;
    let y10 = p10 + p11 + p12 + bias_v;
    let y11 = p11 - p12 - p13 + bias_v;
    let out_plane = out_h * out_w;
    let out_oc_base = (n * out_ch + oc) * out_plane;
    let oh0 = th * 2u32;
    let ow0 = tw * 2u32;
    let out_row0 = out_oc_base + oh0 * out_w;
    let out_row1 = out_oc_base + (oh0 + 1u32) * out_w;
    store(out[out_row0 + ow0], y00.cast::<T>());
    store(out[out_row0 + ow0 + 1u32], y01.cast::<T>());
    store(out[out_row1 + ow0], y10.cast::<T>());
    store(out[out_row1 + ow0 + 1u32], y11.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{winograd_conv2d_3x3, winograd_conv2d_3x3_split, winograd_filter_transform_3x3};
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 3×3 stride-1 conv oracle (NCHW input, OIHW weight). Padding
    /// taps contribute zero. f32.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn naive_conv3x3(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        pad: usize,
    ) -> Vec<f32> {
        let out_h = in_h + 2 * pad - 2;
        let out_w = in_w + 2 * pad - 2;
        let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for ic in 0..in_ch {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let ph = oh + ky;
                                    let pw = ow + kx;
                                    if ph < pad || ph >= pad + in_h || pw < pad || pw >= pad + in_w
                                    {
                                        continue;
                                    }
                                    let ih = ph - pad;
                                    let iw = pw - pad;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    let w_idx = ((oc * in_ch + ic) * 3 + ky) * 3 + kx;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        let o_idx = ((n * out_ch + oc) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        out
    }

    /// Filter transform oracle `U = G·g·Gᵀ` for one 3×3 filter `g` →
    /// 4×4 `U`, row-major. Mirrors `winograd_filter_transform_3x3`.
    fn filter_transform_one(g: &[f32]) -> [f32; 16] {
        let (g00, g01, g02) = (g[0], g[1], g[2]);
        let (g10, g11, g12) = (g[3], g[4], g[5]);
        let (g20, g21, g22) = (g[6], g[7], g[8]);
        // s = G·g (rows mix). G rows: [1,0,0] [½,½,½] [½,-½,½] [0,0,1].
        let s00 = g00;
        let s01 = g01;
        let s02 = g02;
        let s10 = 0.5 * (g00 + g10 + g20);
        let s11 = 0.5 * (g01 + g11 + g21);
        let s12 = 0.5 * (g02 + g12 + g22);
        let s20 = 0.5 * (g00 - g10 + g20);
        let s21 = 0.5 * (g01 - g11 + g21);
        let s22 = 0.5 * (g02 - g12 + g22);
        let s30 = g20;
        let s31 = g21;
        let s32 = g22;
        // U = s·Gᵀ (columns mix).
        [
            s00,
            0.5 * (s00 + s01 + s02),
            0.5 * (s00 - s01 + s02),
            s02,
            s10,
            0.5 * (s10 + s11 + s12),
            0.5 * (s10 - s11 + s12),
            s12,
            s20,
            0.5 * (s20 + s21 + s22),
            0.5 * (s20 - s21 + s22),
            s22,
            s30,
            0.5 * (s30 + s31 + s32),
            0.5 * (s30 - s31 + s32),
            s32,
        ]
    }

    // ── winograd_conv2d_3x3 (single kernel) ──────────────────────────────

    fn conv_setup(
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = in_h + 2 * pad - 2;
        let out_w = in_w + 2 * pad - 2;
        assert!(
            out_h.is_multiple_of(2) && out_w.is_multiple_of(2),
            "Winograd needs even output dims"
        );
        let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
        let n_out = batch * out_ch * out_h * out_w;
        let n_tiles = batch * out_ch * tiles_h * tiles_w;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 4.0);
        let weight_f = ramp(out_ch * in_ch * 9, 11, 2.0);
        let bias_f = ramp(out_ch, 5, 1.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv3x3(&input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, pad);
        TestSetup::new(winograd_conv2d_3x3::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("pad_h", pad as u32)
            .constexpr("pad_w", pad as u32)
            .constexpr("tiles_h", tiles_h as u32)
            .constexpr("tiles_w", tiles_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_tiles, 64)
    }

    // Unpadded single-channel — the minimal transform check (in 8×8 → out 6×6).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_winograd_unpadded(dt: DType) -> TestSetup { conv_setup(1, 1, 8, 8, 1, 0, dt) }

    // Padded multi-channel — clamp on every tile edge + per-channel accum.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_winograd_padded(dt: DType) -> TestSetup { conv_setup(2, 4, 8, 10, 5, 1, dt) }

    // ── winograd_filter_transform_3x3 ────────────────────────────────────

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_winograd_filter_transform(dt: DType) -> TestSetup {
        let (out_ch, in_ch) = (5usize, 4usize);
        let n_filt = out_ch * in_ch;
        let weight_f = ramp(n_filt * 9, 11, 2.0);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let mut expected = vec![0.0f32; n_filt * 16];
        for f in 0..n_filt {
            let u = filter_transform_one(&weight[f * 9..f * 9 + 9]);
            expected[f * 16..f * 16 + 16].copy_from_slice(&u);
        }
        TestSetup::new(winograd_filter_transform_3x3::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::zeros("out", n_filt * 16, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("out_ch", out_ch as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_filt, 64)
    }

    // ── winograd_conv2d_3x3_split (consumes pre-transformed U) ────────────

    fn split_setup(
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = in_h + 2 * pad - 2;
        let out_w = in_w + 2 * pad - 2;
        assert!(
            out_h.is_multiple_of(2) && out_w.is_multiple_of(2),
            "Winograd needs even output dims"
        );
        let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
        let n_out = batch * out_ch * out_h * out_w;
        let n_tiles = batch * out_ch * tiles_h * tiles_w;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 4.0);
        let weight_f = ramp(out_ch * in_ch * 9, 11, 2.0);
        let bias_f = ramp(out_ch, 5, 1.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        // Pre-transform every filter on the CPU into the [out_ch, in_ch, 4, 4]
        // U buffer the split kernel consumes — packed/rounded through `dt`
        // so the kernel sees the same precision a real two-pass run would.
        let n_filt = out_ch * in_ch;
        let mut u_f = vec![0.0f32; n_filt * 16];
        for f in 0..n_filt {
            let u = filter_transform_one(&weight[f * 9..f * 9 + 9]);
            u_f[f * 16..f * 16 + 16].copy_from_slice(&u);
        }
        let expected = naive_conv3x3(&input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, pad);
        TestSetup::new(winograd_conv2d_3x3_split::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("u", pack_f32(&u_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("pad_h", pad as u32)
            .constexpr("pad_w", pad as u32)
            .constexpr("tiles_h", tiles_h as u32)
            .constexpr("tiles_w", tiles_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_tiles, 64)
    }

    // bf16 tol 8e-2: F(2×2,3×3) Winograd accumulates more rounding than direct
    // conv, and the split path stages the filter transform through bf16 too.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 8e-2])]
    fn test_winograd_split(dt: DType) -> TestSetup { split_setup(2, 4, 8, 10, 5, 1, dt) }
}

/// New-syntax benches for the Winograd family. Grid3D. The conv kernels
/// run one thread per 2×2 tile; the filter transform one per (oc, ic).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{winograd_conv2d_3x3, winograd_conv2d_3x3_split, winograd_filter_transform_3x3};

    #[bench(name = "ffai/conv2d/winograd_3x3", dtypes = [f32, f16, bf16])]
    fn bench_winograd_3x3(dt: DType) -> BenchSetup {
        // ResNet-ish 3×3 stride-1 pad-1: 64ch 56×56.
        let (batch, in_ch, in_h, in_w, out_ch, pad) =
            (1usize, 64usize, 56usize, 56usize, 64usize, 1usize);
        let out_h = in_h + 2 * pad - 2;
        let out_w = in_w + 2 * pad - 2;
        let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
        let n_out = batch * out_ch * out_h * out_w;
        let n_tiles = batch * out_ch * tiles_h * tiles_w;
        BenchSetup::new(winograd_conv2d_3x3::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * 9, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("pad_h", pad as u32)
            .constexpr("pad_w", pad as u32)
            .constexpr("tiles_h", tiles_h as u32)
            .constexpr("tiles_w", tiles_w as u32)
            .grid_1d(n_tiles, 64)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/conv2d/winograd_filter_transform_3x3", dtypes = [f32, f16, bf16])]
    fn bench_winograd_filter_transform(dt: DType) -> BenchSetup {
        let (out_ch, in_ch) = (256usize, 256usize);
        let n_filt = out_ch * in_ch;
        BenchSetup::new(winograd_filter_transform_3x3::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("weight", n_filt * 9, dt))
            .buffer(BenchBuffer::zeros("out", n_filt * 16, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("out_ch", out_ch as u32)
            .grid_1d(n_filt, 64)
            .bytes_moved((n_filt * 16 * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/conv2d/winograd_3x3_split", dtypes = [f32, f16, bf16])]
    fn bench_winograd_3x3_split(dt: DType) -> BenchSetup {
        let (batch, in_ch, in_h, in_w, out_ch, pad) =
            (1usize, 64usize, 56usize, 56usize, 64usize, 1usize);
        let out_h = in_h + 2 * pad - 2;
        let out_w = in_w + 2 * pad - 2;
        let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
        let n_out = batch * out_ch * out_h * out_w;
        let n_tiles = batch * out_ch * tiles_h * tiles_w;
        BenchSetup::new(winograd_conv2d_3x3_split::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("u", out_ch * in_ch * 16, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("pad_h", pad as u32)
            .constexpr("pad_w", pad as u32)
            .constexpr("tiles_h", tiles_h as u32)
            .constexpr("tiles_w", tiles_w as u32)
            .grid_1d(n_tiles, 64)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
