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

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="audio_conv1d",
    subop="audio_conv1d",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
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
