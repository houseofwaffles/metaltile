//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FishSpeech vocoder 1D convolutions: **dilated** (ResBlock) and
//! **transposed / upsampling** (HiFi-GAN-style generator).
//!
//! The FishSpeech (and most HiFi-GAN-derived) vocoder generator is two
//! interleaved conv families:
//!
//!   * **Dilated `Conv1d`** inside each MRF ResBlock — same shape as the
//!     dense [`audio_conv1d`](super::audio_conv1d) but with a `dilation`
//!     stride between kernel taps (`dilations = 1, 3, 5, …` widen the
//!     receptive field without extra parameters). `same` padding keeps
//!     the time axis length.
//!   * **`ConvTranspose1d`** for each upsampling stage — fractionally-
//!     strided convolution that *expands* the time axis by `stride`.
//!
//! Both are NCL (PyTorch convention), generic over T, one thread per
//! output element, fp32 accumulation. Grid3D — dispatch with
//! `grid_1d(n_out, 256)`.
//!
//!   dilated:    out_len = (in_len + 2*pad - dilation*(k-1) - 1)/stride + 1
//!   transpose:  out_len = (in_len - 1)*stride - 2*pad + dilation*(k-1)
//!                         + output_padding + 1   (caller passes out_len)

use metaltile::kernel;

// ─── Dilated dense Conv1d (ResBlock) ───────────────────────────────
//
//   input  [batch, in_ch,  in_len]   T
//   weight [out_ch, in_ch, k]        T   (OIK)
//   bias   [out_ch]                  T
//   out    [batch, out_ch, out_len]  T
//
// Tap `kx` of output `op` reads padded input index
// `op*stride + kx*dilation`, valid iff it lies in `[pad, pad+in_len)`.

#[kernel]
pub fn conv1d_dilated<T>(
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
    #[constexpr] dilation: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let w_oc_stride = in_ch * k;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let w_ic_base = oc * w_oc_stride + ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx * dilation;
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

// ─── Transposed (upsampling) Conv1d ────────────────────────────────
//
//   input  [batch, in_ch,  in_len]   T
//   weight [in_ch, out_ch, k]        T   (IOK — ConvTranspose1d layout)
//   bias   [out_ch]                  T
//   out    [batch, out_ch, out_len]  T
//
// Gather (adjoint) form: output position `op` collects every input
// position `ip` and tap `kx` for which `op + pad == ip*stride +
// kx*dilation`. So `ip = (op + pad - kx*dilation) / stride`, valid iff
// the numerator is non-negative, divisible by `stride`, and `ip <
// in_len`. One thread per output element; no scatter / no atomics.

#[kernel]
pub fn conv1d_transpose<T>(
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
    #[constexpr] dilation: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    // Padded output coordinate. `ip*stride + kx*dilation == op + pad`.
    let opp = op + pad;
    let in_n_stride = in_ch * in_len;
    let w_in_stride = out_ch * k;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        // weight is [in_ch, out_ch, k]: stride over in_ch then out_ch.
        let w_ic_base = ic * w_in_stride + oc * k;
        for kx in range(0u32, k, 1u32) {
            let tap = kx * dilation;
            // ip = (opp - tap) / stride, only when opp >= tap, the
            // remainder is zero, and ip is in range.
            let has = opp >= tap;
            let num = select(has, opp - tap, 0u32);
            let on_grid = (num % stride) == 0u32;
            let ip = num / stride;
            let valid = has & on_grid & (ip < in_len);
            let ix = select(valid, ip, 0u32);
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

    use super::{conv1d_dilated, conv1d_transpose};
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    // ── Dilated dense conv1d oracle (NCL input, OIK weight) ──
    #[allow(clippy::too_many_arguments)]
    fn naive_dilated(
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
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx * dilation;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            acc += input[(n * in_ch + ic) * in_len + ix]
                                * weight[(oc * in_ch + ic) * k + kx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    // ── Transposed conv1d oracle (NCL input, IOK weight) ──
    #[allow(clippy::too_many_arguments)]
    fn naive_transpose(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        out_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    let opp = op + pad;
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let tap = kx * dilation;
                            if opp < tap {
                                continue;
                            }
                            let num = opp - tap;
                            if !num.is_multiple_of(stride) {
                                continue;
                            }
                            let ip = num / stride;
                            if ip >= in_len {
                                continue;
                            }
                            acc += input[(n * in_ch + ic) * in_len + ip]
                                * weight[(ic * out_ch + oc) * k + kx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn dilated_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_dilated(
            &input, &weight, &bias, batch, in_ch, in_len, out_ch, k, stride, pad, dilation,
        );
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
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[allow(clippy::too_many_arguments)]
    fn transpose_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        output_padding: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + output_padding + 1 - 2 * pad;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(in_ch * out_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_transpose(
            &input, &weight, &bias, batch, in_ch, in_len, out_ch, out_len, k, stride, pad, dilation,
        );
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
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // MRF ResBlock dilated convs: k=3, same padding (pad = dilation),
    // dilations 1 / 3 / 5.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_dilated_d1(dt: DType) -> TestSetup {
        dilated_setup(conv1d_dilated::kernel_ir_for(dt), 1, 16, 50, 16, 3, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_dilated_d3(dt: DType) -> TestSetup {
        dilated_setup(conv1d_dilated::kernel_ir_for(dt), 1, 12, 60, 12, 3, 1, 3, 3, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_dilated_d5(dt: DType) -> TestSetup {
        dilated_setup(conv1d_dilated::kernel_ir_for(dt), 2, 8, 48, 8, 3, 1, 5, 5, dt)
    }

    // HiFi-GAN upsampling: ConvTranspose1d(k=2*stride, stride, pad=stride/2).
    // stride 8, k=16, pad=4 → ~8× time upsample.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_transpose_up8(dt: DType) -> TestSetup {
        transpose_setup(conv1d_transpose::kernel_ir_for(dt), 1, 8, 16, 6, 16, 8, 4, 1, 0, dt)
    }

    // Smaller 2× upsample: stride 2, k=4, pad 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_transpose_up2(dt: DType) -> TestSetup {
        transpose_setup(conv1d_transpose::kernel_ir_for(dt), 2, 6, 24, 4, 4, 2, 1, 1, 0, dt)
    }
}

/// New-syntax benches: a dilated ResBlock conv and a HiFi-GAN upsample.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{conv1d_dilated, conv1d_transpose};

    #[bench(name = "ffai/fishspeech_conv1d/conv1d_dilated", dtypes = [f32, f16, bf16])]
    fn bench_conv1d_dilated(dt: DType) -> BenchSetup {
        // ResBlock dilated conv: 512 ch, len 1024, k=3, dilation 3, same pad.
        let (batch, ch, in_len, k, stride, pad, dilation) =
            (1usize, 512usize, 1024usize, 3usize, 1usize, 3usize, 3usize);
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * ch * out_len;
        BenchSetup::new(conv1d_dilated::kernel_ir_for(dt))
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
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/fishspeech_conv1d/conv1d_transpose", dtypes = [f32, f16, bf16])]
    fn bench_conv1d_transpose(dt: DType) -> BenchSetup {
        // HiFi-GAN upsample stage: 256→128 ch, len 256, stride 8, k=16, pad 4.
        let (batch, in_ch, in_len, out_ch, k, stride, pad, dilation) =
            (1usize, 256usize, 256usize, 128usize, 16usize, 8usize, 4usize, 1usize);
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * pad;
        let n_out = batch * out_ch * out_len;
        BenchSetup::new(conv1d_transpose::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_len, dt))
            .buffer(BenchBuffer::random("weight", in_ch * out_ch * k, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
