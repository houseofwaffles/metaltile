//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! 2D convolution for vision-transformer patch embedding.
//!
//! Every VLM vision encoder (Qwen2.5-VL / Qwen3.5-VL / Gemma 3-VL /
//! Gemma 4-VL) starts by convolving the raw image with a `patch×patch`
//! kernel at `stride = patch` to project pixel patches into the model's
//! hidden dimension. There is no overlap between patches, so this is a
//! tiled GEMM in disguise — the im2col unfold is implicit: each output
//! element gathers exactly `in_ch * kh * kw` input pixels and dots them
//! with the corresponding filter row.
//!
//! Layouts (NCHW input, OIHW weight — the PyTorch / safetensors default
//! every VLM checkpoint ships):
//!
//!   input    [batch, in_ch,  in_h,  in_w]    T
//!   weight   [out_ch, in_ch, kh,    kw]      T
//!   bias     [out_ch]                        T
//!   out      [batch, out_ch, out_h, out_w]   T
//!
//!   out_h = (in_h + 2*pad_h - kh) / stride_h + 1
//!   out_w = (in_w + 2*pad_w - kw) / stride_w + 1
//!
//! One thread per output element `(n, oc, oh, ow)`. The thread walks the
//! `in_ch × kh × kw` receptive field, accumulating in fp32, and clamps
//! out-of-range (padding) reads to contribute zero. Generic over T —
//! fp16 / bf16 / f32 all flow through the same `#[kernel] fn`.
//!
//! Two macro variants bake in the common patch configs so the inner
//! `kh / kw / stride` loop bounds are compile-time constants the codegen
//! can unroll: `conv2d_patch14` (14×14 stride 14 — Qwen-VL / SigLIP) and
//! `conv2d_patch16` (16×16 stride 16 — CLIP / Gemma-VL). `conv2d_generic`
//! keeps the kernel size and stride as runtime constexprs for any other
//! configuration.
//!
//! `conv2d_grouped` is the fully general 2D convolution — it adds
//! dilation (atrous convs) and grouped channels (depthwise / grouped
//! convs) on top of the strided/padded direct-conv structure. It is the
//! direct-conv counterpart of MLX's implicit-GEMM `steel_conv_general`.
//!
//! ## Macro structure
//!
//! `conv2d_kernel!` emits the whole `#[kernel(bench(...))] pub fn …` at
//! module scope. The compiler expands the outer macro before the `#[kernel]`
//! proc-macro runs, so the body parser sees concrete `$kh / $kw / $stride`
//! tokens — never an inner `macro_rules!` inside a kernel body (which
//! silently empties the kernel; see `dequant_gather.rs`).
//!
//! Codegen-only. Correctness validated by `conv2d_gpu_correctness`.

use metaltile::kernel;

/// Emit a conv2d kernel. `$kh / $kw / $stride` are either literals (the
/// fixed-patch variants) or the `kh / kw / stride_h / stride_w`
/// constexpr idents (the generic variant). Padding is always a runtime
/// constexpr — vision patch convs are typically unpadded but Gemma-VL's
/// pan-and-scan tiles can carry a small pad.
macro_rules! conv2d_kernel {
    ($name:ident, $subop:literal, $kh:expr, $kw:expr, $sh:expr, $sw:expr) => {
        #[kernel(
            bench(op="conv2d", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)
        )]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<T>,
            bias: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] batch: u32,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] stride_h: u32,
            #[constexpr] stride_w: u32,
            #[constexpr] pad_h: u32,
            #[constexpr] pad_w: u32,
        ) {
            // Flat output index → (n, oc, oh, ow). One thread per output.
            let idx = program_id::<0>();
            let ow = idx % out_w;
            let t1 = idx / out_w;
            let oh = t1 % out_h;
            let t2 = t1 / out_h;
            let oc = t2 % out_ch;
            let n = t2 / out_ch;

            // Receptive-field anchors expressed as indices into the
            // *padded* input — `oh*stride` lands at column `pad_h` of the
            // padded grid. A real input pixel at row `ph` therefore sits
            // at unpadded row `ph - pad_h`, valid iff
            // `pad_h <= ph < pad_h + in_h`. Working in this padded frame
            // keeps every index a non-negative u32 — no i32 arithmetic.
            let kh_v = $kh;
            let kw_v = $kw;
            let sh_v = $sh;
            let sw_v = $sw;
            let ph0 = oh * sh_v;
            let pw0 = ow * sw_v;

            let input_plane = in_h * in_w;
            let in_n_stride = in_ch * input_plane;
            let w_in_stride = kh_v * kw_v;
            let w_oc_stride = in_ch * w_in_stride;

            let mut acc = load(bias[oc]).cast::<f32>();

            // Walk the in_ch × kh × kw receptive field. Padding pixels
            // (row/col outside the real input) contribute zero — the load
            // is clamped to index 0 and masked out, so it never reads OOB.
            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * input_plane;
                let w_ic_base = oc * w_oc_stride + ic * w_in_stride;
                for ky in range(0u32, kh_v, 1u32) {
                    let ph = ph0 + ky;
                    let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                    let ih = select(row_ok, ph - pad_h, 0u32);
                    for kx in range(0u32, kw_v, 1u32) {
                        let pw = pw0 + kx;
                        let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                        let valid = row_ok & col_ok;
                        let iw = select(col_ok, pw - pad_w, 0u32);

                        let in_idx = in_ic_base + ih * in_w + iw;
                        let pix = load(input[in_idx]).cast::<f32>();
                        let pix_m = select(valid, pix, 0.0f32);

                        let w_idx = w_ic_base + ky * kw_v + kx;
                        let wt = load(weight[w_idx]).cast::<f32>();
                        acc = acc + pix_m * wt;
                    }
                }
            }

            store(out[idx], acc.cast::<T>());
        }
    };
}

// Fixed-patch variants: kernel size and stride are compile-time
// constants so the receptive-field loops unroll. 14×14/14 is the
// Qwen-VL / SigLIP patch; 16×16/16 is CLIP / Gemma-VL.
conv2d_kernel!(conv2d_patch14, "patch14", 14u32, 14u32, 14u32, 14u32);
conv2d_kernel!(conv2d_patch16, "patch16", 16u32, 16u32, 16u32, 16u32);

// Generic variant: kernel size and stride stay runtime constexprs for
// any other (kh, kw, stride) configuration.
conv2d_kernel!(conv2d_generic, "generic", kh, kw, stride_h, stride_w);

/// Fully general 2D convolution — strides, dilation, padding, and
/// grouped channels.
///
/// `conv2d_generic` above covers strided / padded convs but a unit
/// dilation and a single channel group. This kernel adds the two
/// remaining degrees of freedom of MLX's `steel_conv_general`:
///
/// - **Dilation.** Filter tap `(ky, kx)` samples the input at
///   `ph0 + ky·dilation_h` rather than `ph0 + ky` — the *atrous* /
///   dilated convolution used by segmentation backbones and some audio
///   front-ends. The effective receptive field is
///   `(kh-1)·dilation_h + 1` rows.
/// - **Groups.** Output channel `oc` belongs to group
///   `g = oc / (out_ch / groups)` and convolves only the input-channel
///   slice `[g·icpg, (g+1)·icpg)` where `icpg = in_ch / groups`.
///   `groups == in_ch` is depthwise conv; `groups == 1` is the dense
///   case `conv2d_generic` already covers.
///
/// Layouts (NCHW input, OIHW weight — the standard PyTorch /
/// safetensors layout). Note the weight's I dimension is `in_ch /
/// groups`, not `in_ch`, exactly as PyTorch stores a grouped conv:
///
///   input    [batch, in_ch,           in_h,  in_w]    T
///   weight   [out_ch, in_ch/groups,   kh,    kw]      T
///   bias     [out_ch]                                T
///   out      [batch, out_ch,          out_h, out_w]   T
///
///   out_h = (in_h + 2·pad_h - ((kh-1)·dilation_h + 1)) / stride_h + 1
///   out_w = (in_w + 2·pad_w - ((kw-1)·dilation_w + 1)) / stride_w + 1
///
/// One thread per output element, fp32 accumulation, padding taps
/// clamped to contribute zero — the same direct-convolution structure
/// as `conv2d_generic`. Generic over `T`.
///
/// ## DISPATCH INVARIANTS
///
/// - **Grid3D**, one thread per output element over
///   `batch · out_ch · out_h · out_w`.
/// - **`out_ch` divisible by `groups`** and **`in_ch` divisible by
///   `groups`** — the caller computes `icpg`/`ocpg` and must pass an
///   `out_ch / out_w / out_h` consistent with the dilation formula
///   above. The kernel takes the per-group channel counts as
///   constexprs so no division happens on the hot path.
///
/// Codegen-only; correctness pinned by `conv2d_gpu_correctness`.
#[kernel(
    bench(
        op="conv2d",
        subop="grouped",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_grouped<T>(
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
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] dilation_h: u32,
    #[constexpr] dilation_w: u32,
    // Per-group channel counts: icpg = in_ch / groups, ocpg = out_ch /
    // groups. Passed pre-divided so the kernel never divides on the hot
    // path and `groups` itself is not needed inside the body.
    #[constexpr] icpg: u32,
    #[constexpr] ocpg: u32,
) {
    // Flat output index → (n, oc, oh, ow). One thread per output.
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;
    // Group of this output channel, and the first input channel it
    // convolves. The weight's I dimension is `icpg`, so the weight
    // channel index runs `0..icpg` and the *real* input channel is
    // `ic_base + wic`.
    let group = oc / ocpg;
    let ic_base = group * icpg;
    // Receptive-field anchors in the *padded* input frame (see the
    // `conv2d_kernel!` comment) — a real pixel at padded row `ph` sits
    // at unpadded row `ph - pad_h`, valid iff `pad_h <= ph < pad_h+in_h`.
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let w_in_stride = kh * kw;
    let w_oc_stride = icpg * w_in_stride;
    let mut acc = load(bias[oc]).cast::<f32>();
    // Walk the icpg × kh × kw receptive field. Dilation scales the tap
    // offsets; padding taps contribute zero (clamped load, masked out).
    for wic in range(0u32, icpg, 1u32) {
        let real_ic = ic_base + wic;
        let in_ic_base = n * in_n_stride + real_ic * input_plane;
        let w_ic_base = oc * w_oc_stride + wic * w_in_stride;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky * dilation_h;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx * dilation_w;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);
                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);
                let w_idx = w_ic_base + ky * kw + kx;
                let wt = load(weight[w_idx]).cast::<f32>();
                acc = acc + pix_m * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::{conv2d_generic, conv2d_grouped, conv2d_patch14, conv2d_patch16};
    use crate::utils::{pack_f32, unpack_f32};

    /// Deterministic ramp identical in spirit to the GPU-correctness
    /// `ramp` helper: a bounded zig-zag so f16/bf16 stay in range.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle (NCHW input, OIHW weight). Padding taps
    /// contribute zero. `icpg`/`ocpg` carry grouped/depthwise layout;
    /// dilation scales the tap offsets. All f32.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn naive_conv2d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        icpg: usize,
        ocpg: usize,
    ) -> Vec<f32> {
        let out_h = (in_h + 2 * pad_h - ((kh - 1) * dilation_h + 1)) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - ((kw - 1) * dilation_w + 1)) / stride_w + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                let group = oc / ocpg;
                let ic_base = group * icpg;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for wic in 0..icpg {
                            let real_ic = ic_base + wic;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ph = oh * stride_h + ky * dilation_h;
                                    let pw = ow * stride_w + kx * dilation_w;
                                    if ph < pad_h
                                        || ph >= pad_h + in_h
                                        || pw < pad_w
                                        || pw >= pad_w + in_w
                                    {
                                        continue;
                                    }
                                    let ih = ph - pad_h;
                                    let iw = pw - pad_w;
                                    let in_idx = ((n * in_ch + real_ic) * in_h + ih) * in_w + iw;
                                    let w_idx = ((oc * icpg + wic) * kh + ky) * kw + kx;
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

    /// Shared setup for the patch / generic variants (groups=1,
    /// dilation=1). The patch variants bake kh/kw/stride but still take
    /// them as constexprs in the signature, so bind every constexpr.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv2d(
            &input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w,
            pad_h, pad_w, 1, 1, in_ch, out_ch,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // 14×14 stride-14 patch conv (SigLIP / Qwen-VL stem). Small 28×42
    // image → 2×3 patch grid.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_patch14(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_patch14::kernel_ir_for(dt), 1, 3, 28, 42, 8, 14, 14, 14, 14, 0, 0, dt)
    }

    // 16×16 stride-16 patch conv (CLIP / Gemma-VL stem).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_patch16(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_patch16::kernel_ir_for(dt), 1, 3, 32, 48, 6, 16, 16, 16, 16, 0, 0, dt)
    }

    // Overlapping 3×3 stride-1 conv with 1-px padding — exercises the
    // runtime kh/kw/stride/pad constexprs and the in-kernel padding clamp.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_generic(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_generic::kernel_ir_for(dt), 2, 4, 9, 11, 5, 3, 3, 1, 1, 1, 1, dt)
    }

    /// Setup for the grouped / dilated variant. `out_ch`/`in_ch` must be
    /// divisible by `groups`; the kernel takes pre-divided `icpg`/`ocpg`.
    #[allow(clippy::too_many_arguments)]
    fn grouped_setup(
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        dt: DType,
    ) -> TestSetup {
        let (icpg, ocpg) = (in_ch / groups, out_ch / groups);
        let eff_kh = (kh - 1) * dilation_h + 1;
        let eff_kw = (kw - 1) * dilation_w + 1;
        let out_h = (in_h + 2 * pad_h - eff_kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - eff_kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * icpg * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv2d(
            &input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, icpg, ocpg,
        );
        TestSetup::new(conv2d_grouped::kernel_ir_for(dt))
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
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .constexpr("dilation_h", dilation_h as u32)
            .constexpr("dilation_w", dilation_w as u32)
            .constexpr("icpg", icpg as u32)
            .constexpr("ocpg", ocpg as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Depthwise conv: groups == in_ch == out_ch, 3×3 stride-1 pad-1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_grouped_depthwise(dt: DType) -> TestSetup {
        grouped_setup(2, 8, 12, 14, 8, 3, 3, 1, 1, 1, 1, 1, 1, 8, dt)
    }

    // groups=2 + dilation=2 + stride=2 — every degree of freedom at once.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_grouped_full(dt: DType) -> TestSetup {
        grouped_setup(1, 6, 20, 22, 8, 3, 3, 2, 2, 2, 2, 2, 2, 2, dt)
    }
}

/// New-syntax benches for the conv2d family. Grid3D, `grid_1d(n_out, 256)`.
/// bytes_moved counts the output stream (the input/weight reuse makes a
/// precise count shape-dependent; the output stream is the stable proxy).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::{conv2d_generic, conv2d_grouped, conv2d_patch14, conv2d_patch16};

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bench(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = (in_h - kh) / stride_h + 1;
        let out_w = (in_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/conv2d/patch14", dtypes = [f32, f16, bf16])]
    fn bench_conv2d_patch14(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_patch14::kernel_ir_for(dt), 1, 3, 224, 224, 1024, 14, 14, 14, 14, dt)
    }

    #[bench(name = "ffai/conv2d/patch16", dtypes = [f32, f16, bf16])]
    fn bench_conv2d_patch16(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_patch16::kernel_ir_for(dt), 1, 3, 224, 224, 768, 16, 16, 16, 16, dt)
    }

    #[bench(name = "ffai/conv2d/generic", dtypes = [f32, f16, bf16])]
    fn bench_conv2d_generic(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_generic::kernel_ir_for(dt), 1, 32, 56, 56, 64, 3, 3, 1, 1, dt)
    }

    #[bench(name = "ffai/conv2d/grouped", dtypes = [f32, f16, bf16])]
    fn bench_conv2d_grouped(dt: DType) -> BenchSetup {
        // Depthwise 3×3 stride-1, groups == in_ch == out_ch.
        let (batch, ch, in_h, in_w, kh, kw) = (1usize, 64usize, 56usize, 56usize, 3usize, 3usize);
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_out = batch * ch * out_h * out_w;
        BenchSetup::new(conv2d_grouped::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", ch * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", 1u32)
            .constexpr("stride_w", 1u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .constexpr("dilation_h", 1u32)
            .constexpr("dilation_w", 1u32)
            .constexpr("icpg", 1u32)
            .constexpr("ocpg", 1u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
