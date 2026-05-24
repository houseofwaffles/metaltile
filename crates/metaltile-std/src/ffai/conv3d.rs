//! 3D convolution — the volumetric counterpart of `ffai::conv2d`.
//!
//! Closes the `steel_conv 3D` row of the kernel audit: MLX's
//! `steel_conv_3d` is an implicit-GEMM convolution over 5D NCDHW
//! tensors indexed through `MLXConvParams<3>`. This is the same direct
//! convolution `conv2d.rs` uses for 2D — one thread per output element,
//! the im2col unfold implicit — extended with a depth axis threaded
//! through the receptive-field walk.
//!
//! 3D convolution shows up in volumetric vision (medical imaging,
//! video) and in some audio/video VLM front-ends that convolve a
//! `(time, height, width)` patch grid. The structure is identical to
//! 2D: each output voxel gathers `in_ch · kd · kh · kw` input voxels
//! and dots them with the corresponding filter, accumulating in fp32.
//!
//! Layouts (NCDHW input, OIDHW weight — the PyTorch / safetensors
//! default a `Conv3d` checkpoint ships):
//!
//!   input    [batch, in_ch,  in_d,  in_h,  in_w]    T
//!   weight   [out_ch, in_ch, kd,    kh,    kw]      T
//!   bias     [out_ch]                               T
//!   out      [batch, out_ch, out_d, out_h, out_w]   T
//!
//!   out_d = (in_d + 2*pad_d - ((kd-1)*dilation_d + 1)) / stride_d + 1
//!   out_h = (in_h + 2*pad_h - ((kh-1)*dilation_h + 1)) / stride_h + 1
//!   out_w = (in_w + 2*pad_w - ((kw-1)*dilation_w + 1)) / stride_w + 1
//!
//! One thread per output element `(n, oc, od, oh, ow)`. The thread
//! walks the `in_ch × kd × kh × kw` receptive field, accumulating in
//! fp32, and masks out-of-range (padding) reads to contribute zero.
//! Generic over T — fp16 / bf16 / f32 all flow through the same
//! `#[kernel] fn`.
//!
//! As in `conv2d.rs`, receptive-field anchors are computed in the
//! *padded* input frame so every index stays a non-negative u32 — a
//! real voxel at padded coordinate `p` sits at unpadded `p - pad`,
//! valid iff `pad <= p < pad + extent`. No i32 arithmetic.
//!
//! Two kernels:
//!
//! - `conv3d_generic` — strided / padded dense 3D conv (unit dilation,
//!   one channel group). The direct-conv equivalent of MLX's
//!   `steel_conv_3d`.
//! - `conv3d_grouped` — adds dilation (atrous) and grouped channels
//!   (`groups == in_ch` is depthwise) on top, mirroring `conv2d_grouped`.
//!   The 3D counterpart of `steel_conv_general` for `NDIM == 3`.
//!
//! Codegen-only. Correctness validated by `conv3d_gpu_correctness`.

use metaltile::{bench_kernel, kernel};

/// Dense 3D convolution — strides and padding, unit dilation, one
/// channel group.
///
/// One thread per output voxel `(n, oc, od, oh, ow)`; the thread walks
/// the `in_ch × kd × kh × kw` receptive field with fp32 accumulation.
/// Padding voxels (a depth/row/col outside the real input) contribute
/// zero — the load is clamped to index 0 and masked, never reading OOB.
///
/// Layouts: NCDHW input, OIDHW weight, NCDHW output (see the module
/// doc). Generic over `T`.
///
/// ## DISPATCH INVARIANTS
///
/// - **Grid3D**, one thread per output element over
///   `batch · out_ch · out_d · out_h · out_w`.
/// - The caller computes `out_d / out_h / out_w` from the standard
///   convolution output-size formula and passes them as constexprs so
///   no division happens on the hot path.
///
/// Codegen-only; correctness pinned by `conv3d_gpu_correctness`.
#[bench_kernel(
    op="conv3d",
    subop="generic",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv3d_generic<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
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
    #[constexpr] stride_d: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_d: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
) {
    // Flat output index → (n, oc, od, oh, ow). One thread per output.
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let od = t2 % out_d;
    let t3 = t2 / out_d;
    let oc = t3 % out_ch;
    let n = t3 / out_ch;

    // Receptive-field anchors in the *padded* input frame — a real
    // voxel at padded coordinate `p` sits at unpadded `p - pad`, valid
    // iff `pad <= p < pad + extent`. Working in the padded frame keeps
    // every index a non-negative u32.
    let pd0 = od * stride_d;
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let input_vol = in_d * input_plane;
    let in_n_stride = in_ch * input_vol;
    let w_kd_stride = kh * kw;
    let w_in_stride = kd * w_kd_stride;
    let w_oc_stride = in_ch * w_in_stride;

    let mut acc = load(bias[oc]).cast::<f32>();

    // Walk the in_ch × kd × kh × kw receptive field. Padding voxels
    // (depth/row/col outside the real input) contribute zero — the load
    // is clamped to index 0 and masked out, so it never reads OOB.
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_vol;
        let w_ic_base = oc * w_oc_stride + ic * w_in_stride;
        for kz in range(0u32, kd, 1u32) {
            let pd = pd0 + kz;
            let dep_ok = (pd >= pad_d) & (pd < pad_d + in_d);
            let id = select(dep_ok, pd - pad_d, 0u32);
            for ky in range(0u32, kh, 1u32) {
                let ph = ph0 + ky;
                let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                let ih = select(row_ok, ph - pad_h, 0u32);
                for kx in range(0u32, kw, 1u32) {
                    let pw = pw0 + kx;
                    let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                    let valid = dep_ok & row_ok & col_ok;
                    let iw = select(col_ok, pw - pad_w, 0u32);

                    let in_idx = in_ic_base + id * input_plane + ih * in_w + iw;
                    let pix = load(input[in_idx]).cast::<f32>();
                    let pix_m = select(valid, pix, 0.0f32);

                    let w_idx = w_ic_base + kz * w_kd_stride + ky * kw + kx;
                    let wt = load(weight[w_idx]).cast::<f32>();
                    acc = acc + pix_m * wt;
                }
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// Fully general 3D convolution — strides, dilation, padding, and
/// grouped channels.
///
/// `conv3d_generic` above covers strided / padded 3D convs at unit
/// dilation and a single channel group. This kernel adds the two
/// remaining degrees of freedom — the 3D counterpart of
/// `conv2d_grouped`:
///
/// - **Dilation.** Filter tap `(kz, ky, kx)` samples the input at
///   `pd0 + kz·dilation_d` (etc.) rather than `pd0 + kz` — the atrous
///   convolution. The effective receptive field along depth is
///   `(kd-1)·dilation_d + 1` voxels.
/// - **Groups.** Output channel `oc` belongs to group
///   `g = oc / ocpg` and convolves only the input-channel slice
///   `[g·icpg, (g+1)·icpg)` where `icpg = in_ch / groups`.
///   `groups == in_ch` is depthwise conv; `groups == 1` is the dense
///   case `conv3d_generic` already covers.
///
/// Layouts (NCDHW input, OIDHW weight). The weight's I dimension is
/// `in_ch / groups`, not `in_ch`, exactly as PyTorch stores a grouped
/// `Conv3d`:
///
///   input    [batch, in_ch,         in_d,  in_h,  in_w]   T
///   weight   [out_ch, in_ch/groups, kd,    kh,    kw]      T
///   bias     [out_ch]                                     T
///   out      [batch, out_ch,        out_d, out_h, out_w]  T
///
/// One thread per output voxel, fp32 accumulation, padding taps masked
/// to contribute zero — the same direct-convolution structure as
/// `conv3d_generic`. Generic over `T`.
///
/// ## DISPATCH INVARIANTS
///
/// - **Grid3D**, one thread per output element over
///   `batch · out_ch · out_d · out_h · out_w`.
/// - **`out_ch` divisible by `groups`** and **`in_ch` divisible by
///   `groups`** — the caller passes the pre-divided per-group channel
///   counts `icpg`/`ocpg` so no division happens on the hot path and
///   `groups` itself is not needed inside the body.
///
/// Codegen-only; correctness pinned by `conv3d_gpu_correctness`.
#[bench_kernel(
    op="conv3d",
    subop="grouped",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv3d_grouped<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
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
    #[constexpr] stride_d: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_d: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] dilation_d: u32,
    #[constexpr] dilation_h: u32,
    #[constexpr] dilation_w: u32,
    // Per-group channel counts: icpg = in_ch / groups, ocpg = out_ch /
    // groups. Passed pre-divided so the kernel never divides on the hot
    // path and `groups` itself is not needed inside the body.
    #[constexpr] icpg: u32,
    #[constexpr] ocpg: u32,
) {
    // Flat output index → (n, oc, od, oh, ow). One thread per output.
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let od = t2 % out_d;
    let t3 = t2 / out_d;
    let oc = t3 % out_ch;
    let n = t3 / out_ch;

    // Group of this output channel, and the first input channel it
    // convolves. The weight's I dimension is `icpg`, so the weight
    // channel index runs `0..icpg` and the real input channel is
    // `ic_base + wic`.
    let group = oc / ocpg;
    let ic_base = group * icpg;

    // Receptive-field anchors in the *padded* input frame (see the
    // `conv3d_generic` comment).
    let pd0 = od * stride_d;
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let input_vol = in_d * input_plane;
    let in_n_stride = in_ch * input_vol;
    let w_kd_stride = kh * kw;
    let w_in_stride = kd * w_kd_stride;
    let w_oc_stride = icpg * w_in_stride;

    let mut acc = load(bias[oc]).cast::<f32>();

    // Walk the icpg × kd × kh × kw receptive field. Dilation scales the
    // tap offsets; padding taps contribute zero (clamped load, masked).
    for wic in range(0u32, icpg, 1u32) {
        let real_ic = ic_base + wic;
        let in_ic_base = n * in_n_stride + real_ic * input_vol;
        let w_ic_base = oc * w_oc_stride + wic * w_in_stride;
        for kz in range(0u32, kd, 1u32) {
            let pd = pd0 + kz * dilation_d;
            let dep_ok = (pd >= pad_d) & (pd < pad_d + in_d);
            let id = select(dep_ok, pd - pad_d, 0u32);
            for ky in range(0u32, kh, 1u32) {
                let ph = ph0 + ky * dilation_h;
                let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                let ih = select(row_ok, ph - pad_h, 0u32);
                for kx in range(0u32, kw, 1u32) {
                    let pw = pw0 + kx * dilation_w;
                    let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                    let valid = dep_ok & row_ok & col_ok;
                    let iw = select(col_ok, pw - pad_w, 0u32);

                    let in_idx = in_ic_base + id * input_plane + ih * in_w + iw;
                    let pix = load(input[in_idx]).cast::<f32>();
                    let pix_m = select(valid, pix, 0.0f32);

                    let w_idx = w_ic_base + kz * w_kd_stride + ky * kw + kx;
                    let wt = load(weight[w_idx]).cast::<f32>();
                    acc = acc + pix_m * wt;
                }
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}
