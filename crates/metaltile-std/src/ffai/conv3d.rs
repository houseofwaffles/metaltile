//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile::kernel;

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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{conv3d_generic, conv3d_grouped};
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 3D conv oracle (NCDHW input, OIDHW weight, I = in_ch/groups).
    /// Padding taps contribute zero; dilation scales tap offsets. f32.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn naive_conv3d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        pad_d: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_d: usize,
        dilation_h: usize,
        dilation_w: usize,
        icpg: usize,
        ocpg: usize,
    ) -> Vec<f32> {
        let out_d = (in_d + 2 * pad_d - ((kd - 1) * dilation_d + 1)) / stride_d + 1;
        let out_h = (in_h + 2 * pad_h - ((kh - 1) * dilation_h + 1)) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - ((kw - 1) * dilation_w + 1)) / stride_w + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_d * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                let group = oc / ocpg;
                let ic_base = group * icpg;
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut acc = bias[oc];
                            for wic in 0..icpg {
                                let real_ic = ic_base + wic;
                                for kz in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let pd = od * stride_d + kz * dilation_d;
                                            let ph = oh * stride_h + ky * dilation_h;
                                            let pw = ow * stride_w + kx * dilation_w;
                                            if pd < pad_d
                                                || pd >= pad_d + in_d
                                                || ph < pad_h
                                                || ph >= pad_h + in_h
                                                || pw < pad_w
                                                || pw >= pad_w + in_w
                                            {
                                                continue;
                                            }
                                            let id = pd - pad_d;
                                            let ih = ph - pad_h;
                                            let iw = pw - pad_w;
                                            let in_idx =
                                                (((n * in_ch + real_ic) * in_d + id) * in_h + ih)
                                                    * in_w
                                                    + iw;
                                            let w_idx =
                                                (((oc * icpg + wic) * kd + kz) * kh + ky) * kw + kx;
                                            acc += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            let o_idx =
                                (((n * out_ch + oc) * out_d + od) * out_h + oh) * out_w + ow;
                            out[o_idx] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    /// Setup for the dense `conv3d_generic` path (groups=1, dilation=1).
    #[allow(clippy::too_many_arguments)]
    fn generic_setup(
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        pad_d: usize,
        pad_h: usize,
        pad_w: usize,
        dt: DType,
    ) -> TestSetup {
        let out_d = (in_d + 2 * pad_d - kd) / stride_d + 1;
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_d * out_h * out_w;
        let input_f = ramp(batch * in_ch * in_d * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * kd * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv3d(
            &input, &weight, &bias, batch, in_ch, in_d, in_h, in_w, out_ch, kd, kh, kw, stride_d,
            stride_h, stride_w, pad_d, pad_h, pad_w, 1, 1, 1, in_ch, out_ch,
        );
        TestSetup::new(conv3d_generic::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
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
            .constexpr("stride_d", stride_d as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_d", pad_d as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    /// Setup for the grouped / dilated `conv3d_grouped` path.
    #[allow(clippy::too_many_arguments)]
    fn grouped_setup(
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        pad_d: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_d: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        dt: DType,
    ) -> TestSetup {
        let (icpg, ocpg) = (in_ch / groups, out_ch / groups);
        let out_d = (in_d + 2 * pad_d - ((kd - 1) * dilation_d + 1)) / stride_d + 1;
        let out_h = (in_h + 2 * pad_h - ((kh - 1) * dilation_h + 1)) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - ((kw - 1) * dilation_w + 1)) / stride_w + 1;
        let n_out = batch * out_ch * out_d * out_h * out_w;
        let input_f = ramp(batch * in_ch * in_d * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * icpg * kd * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv3d(
            &input, &weight, &bias, batch, in_ch, in_d, in_h, in_w, out_ch, kd, kh, kw, stride_d,
            stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w, icpg,
            ocpg,
        );
        TestSetup::new(conv3d_grouped::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
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
            .constexpr("stride_d", stride_d as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_d", pad_d as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .constexpr("dilation_d", dilation_d as u32)
            .constexpr("dilation_h", dilation_h as u32)
            .constexpr("dilation_w", dilation_w as u32)
            .constexpr("icpg", icpg as u32)
            .constexpr("ocpg", ocpg as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Padded 3×3×3 stride-1 dense conv — padding clamp on every axis.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_generic(dt: DType) -> TestSetup {
        generic_setup(2, 3, 7, 9, 8, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, dt)
    }

    // Strided anisotropic conv, no padding (video-VLM patch stem shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_generic_strided(dt: DType) -> TestSetup {
        generic_setup(1, 4, 12, 16, 14, 6, 2, 3, 3, 2, 2, 2, 0, 0, 0, dt)
    }

    // Depthwise 3D conv: groups == in_ch == out_ch.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_grouped_depthwise(dt: DType) -> TestSetup {
        grouped_setup(2, 6, 8, 10, 10, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, dt)
    }

    // groups=2 + dilation=2 + stride=2 — every degree of freedom at once.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv3d_grouped_full(dt: DType) -> TestSetup {
        grouped_setup(1, 6, 16, 18, 18, 8, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, dt)
    }
}

/// New-syntax benches for the conv3d family. Grid3D, `grid_1d(n_out, 256)`.
/// bytes_moved counts the output stream.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{conv3d_generic, conv3d_grouped};

    #[bench(name = "ffai/conv3d/generic", dtypes = [f32, f16, bf16])]
    fn bench_conv3d_generic(dt: DType) -> BenchSetup {
        let (batch, in_ch, in_d, in_h, in_w, out_ch) =
            (1usize, 16usize, 16usize, 32usize, 32usize, 32usize);
        let (kd, kh, kw) = (3usize, 3usize, 3usize);
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_out = batch * out_ch * out_d * out_h * out_w;
        BenchSetup::new(conv3d_generic::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_d * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kd * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
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
            .constexpr("stride_d", 1u32)
            .constexpr("stride_h", 1u32)
            .constexpr("stride_w", 1u32)
            .constexpr("pad_d", 0u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/conv3d/grouped", dtypes = [f32, f16, bf16])]
    fn bench_conv3d_grouped(dt: DType) -> BenchSetup {
        // Depthwise 3×3×3 stride-1, groups == in_ch == out_ch.
        let (batch, ch, in_d, in_h, in_w) = (1usize, 32usize, 16usize, 32usize, 32usize);
        let (kd, kh, kw) = (3usize, 3usize, 3usize);
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_out = batch * ch * out_d * out_h * out_w;
        BenchSetup::new(conv3d_grouped::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_d * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", ch * kd * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_d", 1u32)
            .constexpr("stride_h", 1u32)
            .constexpr("stride_w", 1u32)
            .constexpr("pad_d", 0u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .constexpr("dilation_d", 1u32)
            .constexpr("dilation_h", 1u32)
            .constexpr("dilation_w", 1u32)
            .constexpr("icpg", 1u32)
            .constexpr("ocpg", 1u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
