//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Depthwise 2D convolution — `groups == channels`.
//!
//! Each output channel convolves **only its own** input channel with a
//! single `k × k` filter (no cross-channel mixing). This is the core
//! primitive of MobileNet/FastViT-style conv stems and the patch-embed
//! depthwise conv used by the PaliGemma SigLIP and FastVLM (MobileCLIP)
//! vision towers — the `Phase D` D.1 / D.3 gap. It is distinct from the
//! dense multi-channel [`audio_conv1d`](super::audio_conv1d) (which sums
//! a full `in_ch` receptive field) and from the depthwise *1D* streaming
//! `conv1d_causal_step` in `ssm.rs`.
//!
//! Layouts (NCHW — the PyTorch `nn.Conv2d(C, C, k, …, groups=C)`
//! convention, with the depthwise weight squeezed to `[C, k, k]` since
//! `in_ch_per_group == 1`):
//!
//!   input    [batch, ch, in_h,  in_w]    T
//!   weight   [ch, k, k]                   T
//!   bias     [ch]                         T
//!   out      [batch, ch, out_h, out_w]    T
//!
//!   out_h = (in_h + 2*pad - dilation*(k-1) - 1) / stride + 1   (same for out_w)
//!
//! One thread per output element `(n, c, oh, ow)`. The thread walks the
//! `k × k` receptive field of its own channel, accumulating in fp32.
//! Padding/dilation taps that fall outside the real input contribute
//! zero — the load is clamped to index 0 and masked. All indices stay in
//! the *padded* frame so every value is a non-negative u32 (no i32
//! arithmetic). Generic over T.
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(n_out, 256)` (NOT `grid_3d(n_out, …)`, which would launch
//! `n_out × tpg` threads and stride past the output buffer). `out_h` /
//! `out_w` must match the formula above for the given `(k, stride, pad,
//! dilation)`, and `bias` must have `ch` elements.

use metaltile::kernel;

#[kernel]
pub fn depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
) {
    // Flat output index → (n, c, oh, ow). One thread per output element.
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    // Receptive-field anchor in the *padded* input frame: tap (ky, kx)
    // of output (oh, ow) lands at padded index
    // (oh*stride + ky*dilation, ow*stride + kx*dilation), which maps to a
    // real input index iff it lies in [pad, pad+in_*).
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let w_c_base = c * k * k;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let wt = load(weight[w_c_base + ky * k + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::depthwise_conv2d;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    fn out_dim(in_d: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> usize {
        (in_d + 2 * pad - dilation * (k - 1) - 1) / stride + 1
    }

    /// Direct depthwise 2D conv oracle (NCHW input, `[C, k, k]` weight).
    /// Padding/dilation taps zero. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_depthwise_conv2d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let mut out = vec![0.0f32; batch * ch * out_h * out_w];
        for n in 0..batch {
            for c in 0..ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[c];
                        for ky in 0..k {
                            let ph = oh * stride + ky * dilation;
                            if ph < pad || ph >= pad + in_h {
                                continue;
                            }
                            let ih = ph - pad;
                            for kx in 0..k {
                                let pw = ow * stride + kx * dilation;
                                if pw < pad || pw >= pad + in_w {
                                    continue;
                                }
                                let iw = pw - pad;
                                let in_idx = ((n * ch + c) * in_h + ih) * in_w + iw;
                                let w_idx = (c * k + ky) * k + kx;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                        out[((n * ch + c) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        kernel: Kernel,
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let n_out = batch * ch * out_h * out_w;
        let input_f = ramp(batch * ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(ch * k * k, 11, 4.0);
        let bias_f = ramp(ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_depthwise_conv2d(
            &input, &weight, &bias, batch, ch, in_h, in_w, k, stride, pad, dilation,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // FastViT/MobileCLIP-style depthwise 3×3 stride-1 pad-1 (same-size).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_3x3_s1(dt: DType) -> TestSetup {
        setup(depthwise_conv2d::kernel_ir_for(dt), 1, 8, 16, 16, 3, 1, 1, 1, dt)
    }

    // Strided downsample: depthwise 3×3 stride-2 pad-1 (halves H/W).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_3x3_s2(dt: DType) -> TestSetup {
        setup(depthwise_conv2d::kernel_ir_for(dt), 2, 6, 24, 24, 3, 2, 1, 1, dt)
    }

    // Patch-embed-style depthwise: k=stride (non-overlapping), no pad.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_patch(dt: DType) -> TestSetup {
        setup(depthwise_conv2d::kernel_ir_for(dt), 1, 4, 28, 28, 14, 14, 0, 1, dt)
    }

    // Dilated depthwise 3×3 (dilation 2) — atrous receptive field.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_dilated(dt: DType) -> TestSetup {
        setup(depthwise_conv2d::kernel_ir_for(dt), 1, 5, 20, 20, 3, 1, 2, 2, dt)
    }
}

/// New-syntax bench for `depthwise_conv2d` (FastViT stem shape: 64 channels,
/// 112×112 feature map, depthwise 3×3 stride-2). Grid3D, `grid_1d(n_out, 256)`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::depthwise_conv2d;

    #[bench(name = "ffai/depthwise_conv2d/depthwise_conv2d", dtypes = [f32, f16, bf16])]
    fn bench_depthwise_conv2d(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, k, stride, pad, dilation) =
            (1usize, 64usize, 112usize, 112usize, 3usize, 2usize, 1usize, 1usize);
        let out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * ch * out_h * out_w;
        BenchSetup::new(depthwise_conv2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", ch * k * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
