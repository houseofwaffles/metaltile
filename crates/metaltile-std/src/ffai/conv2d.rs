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
//! `conv2d_kernel!` emits the whole `#[kernel] pub fn …` plus its
//! `inventory::submit!` at module scope. The compiler expands the outer
//! macro before the `#[kernel]` proc-macro runs, so the body parser sees
//! concrete `$kh / $kw / $stride` tokens — never an inner `macro_rules!`
//! inside a kernel body (which silently empties the kernel; see
//! `dequant_gather.rs`).
//!
//! Codegen-only. Correctness validated by `conv2d_gpu_correctness`.

use metaltile::{bench_kernel, kernel};

/// Emit a conv2d kernel. `$kh / $kw / $stride` are either literals (the
/// fixed-patch variants) or the `kh / kw / stride_h / stride_w`
/// constexpr idents (the generic variant). Padding is always a runtime
/// constexpr — vision patch convs are typically unpadded but Gemma-VL's
/// pan-and-scan tiles can carry a small pad.
macro_rules! conv2d_kernel {
    ($name:ident, $subop:literal, $kh:expr, $kw:expr, $sh:expr, $sw:expr) => {
        #[bench_kernel(op="conv2d", subop=$subop, class=GenericEmpty, tol=1e-3, kernel_mode=Grid3D,)]
        #[kernel]
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
#[bench_kernel(
    op="conv2d",
    subop="grouped",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
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
