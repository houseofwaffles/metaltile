//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Wide-stride multi-channel 1D convolution — the STT audio patch
//! embedding.
//!
//! After the log-Mel front-end (`mel_spectrogram`), a speech encoder
//! (Whisper, Qwen-Omni audio, Parakeet) downsamples the Mel sequence
//! with one or two strided 1D convolutions before the transformer
//! stack. Whisper's stem is `Conv1d(n_mels→d_model, k=3, s=1)` then
//! `Conv1d(d_model→d_model, k=3, s=2)`; the strided second conv halves
//! the time axis. This is a *dense, multi-channel, strided* conv —
//! distinct from the depthwise single-channel `conv1d_causal_step` in
//! `ssm.rs`, which streams one SSM-state column with `groups == channels`.
//!
//! Layouts (NCL — the PyTorch `nn.Conv1d` convention):
//!
//!   input    [batch, in_ch,  in_len]    T
//!   weight   [out_ch, in_ch, k]         T
//!   bias     [out_ch]                   T
//!   out      [batch, out_ch, out_len]   T
//!
//!   out_len = (in_len + 2*pad - k) / stride + 1
//!
//! One thread per output element `(n, oc, op)`. The thread walks the
//! `in_ch × k` receptive field, accumulating in fp32. Padding taps
//! (position outside the real input) contribute zero — the load is
//! clamped to index 0 and masked. Indices stay in the *padded* frame so
//! every value is a non-negative u32 (no i32 arithmetic). Generic over T.
//!
//! Codegen-only. Correctness validated by `audio_conv1d_gpu_correctness`.

use metaltile::kernel;

#[kernel(
    bench(
        op="audio_conv1d",
        subop="audio_conv1d",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
) {
    // Flat output index → (n, oc, op). One thread per output element.
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    // Receptive-field anchor in the *padded* input frame: tap `kx` of
    // output position `op` lands at padded index `op*stride + kx`, which
    // maps to real input index `p - pad`, valid iff `pad <= p < pad+in_len`.
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let w_oc_stride = in_ch * k;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let w_ic_base = oc * w_oc_stride + ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let wt = load(weight[w_ic_base + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::audio_conv1d;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 1D conv oracle (NCL input, OIK weight). Padding taps zero. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv1d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            let in_idx = (n * in_ch + ic) * in_len + ix;
                            let w_idx = (oc * in_ch + ic) * k + kx;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected =
            naive_conv1d(&input, &weight, &bias, batch, in_ch, in_len, out_ch, k, stride, pad);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Whisper stem conv #1: k=3, stride 1, pad 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_audio_conv1d_stride1(dt: DType) -> TestSetup {
        conv1d_setup(audio_conv1d::kernel_ir_for(dt), 1, 8, 50, 16, 3, 1, 1, dt)
    }

    // Whisper stem conv #2: k=3, stride 2 (halves time), pad 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_audio_conv1d_stride2(dt: DType) -> TestSetup {
        conv1d_setup(audio_conv1d::kernel_ir_for(dt), 2, 12, 64, 12, 3, 2, 1, dt)
    }

    // Wide-stride patch-embed-style conv: k=10, stride 5, no padding.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_audio_conv1d_wide(dt: DType) -> TestSetup {
        conv1d_setup(audio_conv1d::kernel_ir_for(dt), 1, 4, 100, 8, 10, 5, 0, dt)
    }
}

/// New-syntax bench for `audio_conv1d` (Whisper stride-2 stem shape).
/// Grid3D, `grid_1d(n_out, 256)`; bytes_moved counts the output stream.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::audio_conv1d;

    #[bench(name = "ffai/audio_conv1d/audio_conv1d", dtypes = [f32, f16, bf16])]
    fn bench_audio_conv1d(dt: DType) -> BenchSetup {
        // Whisper-large stem conv #2: d_model=1280, k=3, stride 2, pad 1.
        let (batch, ch, in_len, k, stride, pad) =
            (1usize, 1280usize, 1500usize, 3usize, 2usize, 1usize);
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * ch * out_len;
        BenchSetup::new(audio_conv1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_len, dt))
            .buffer(BenchBuffer::random("weight", ch * ch * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
