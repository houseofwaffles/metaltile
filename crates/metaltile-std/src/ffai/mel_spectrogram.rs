//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Log-Mel spectrogram — the STT / audio-in front-end.
//!
//! Whisper, Qwen-Omni audio-in, Parakeet and every other speech model
//! start by turning a raw waveform into a log-Mel spectrogram: window
//! the signal into overlapping frames, take the short-time Fourier
//! transform (STFT), square to a power spectrum, project the power
//! spectrum through a Mel filterbank, and take the log. This kernel
//! fuses the STFT, the filterbank projection and the log into one
//! dispatch.
//!
//! One thread per output element `(frame, mel_bin)`. The thread:
//!   1. for each FFT frequency bin `k ∈ [0, n_freq)` computes the real
//!      and imaginary DFT coefficients of the windowed frame directly
//!      (a length-`n_fft` dot product against cos/sin) — power = re²+im²;
//!   2. accumulates `mel_weight[mel_bin, k] * power[k]` over all `k`;
//!   3. writes `log(acc + log_eps)`.
//!
//! A direct DFT (not an FFT) is O(n_fft · n_freq) per thread. For STT
//! front-ends `n_fft` is 400–512 and `n_freq` ≈ 201–257, so the inner
//! work is a few×10⁴ multiply-adds — comfortably GPU-bound, one dispatch
//! covering every `(frame, mel_bin)` in parallel. A radix-FFT path is a
//! perf follow-up (it needs complex-type codegen — see the `fft` row in
//! `KERNEL_AUDIT.md`); the direct DFT is exact and unblocks the model
//! family now.
//!
//! Layouts:
//!
//!   audio       [n_samples]                  T   (mono waveform)
//!   window      [n_fft]                      T   (e.g. periodic Hann)
//!   mel_weight  [n_mels, n_freq]             T   (Mel filterbank)
//!   out         [n_frames, n_mels]           T   (log-Mel)
//!
//!   n_freq   = n_fft / 2 + 1
//!   frame f covers audio samples [f * hop_length, f * hop_length + n_fft)
//!
//! The caller pre-pads `audio` so every frame is in-bounds (Whisper pads
//! by `n_fft/2` reflect on each side); this kernel does no bounds check
//! on the frame walk — `n_samples >= (n_frames-1)*hop + n_fft` is a
//! caller precondition. Generic over T; accumulation is fp32.
//!
//! Codegen-only. Correctness validated by `mel_spectrogram_gpu_correctness`.

use metaltile::kernel;

#[kernel(
    bench(
        op="mel_spectrogram",
        subop="mel_spectrogram",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mel_spectrogram<T>(
    audio: Tensor<T>,
    window: Tensor<T>,
    mel_weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] n_mels: u32,
    #[constexpr] hop_length: u32,
    #[constexpr] log_eps: f32,
) {
    // Flat output index → (frame, mel_bin). One thread per output.
    let idx = program_id::<0>();
    let mel_bin = idx % n_mels;
    let frame = idx / n_mels;
    let frame_start = frame * hop_length;
    let n_fft_f = n_fft.cast::<f32>();
    // -2π / n_fft — the DFT twiddle-angle step.
    let neg_two_pi_over_n = -6.283185307179586f32 / n_fft_f;
    let mel_row = mel_bin * n_freq;
    let mut mel_acc = 0.0f32;
    // For each frequency bin: direct DFT of the windowed frame, square
    // to power, weight by the Mel filterbank coefficient, accumulate.
    for k in range(0u32, n_freq, 1u32) {
        let k_f = k.cast::<f32>();
        let angle_step = neg_two_pi_over_n * k_f;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for t in range(0u32, n_fft, 1u32) {
            let sample = load(audio[frame_start + t]).cast::<f32>();
            let win = load(window[t]).cast::<f32>();
            let xw = sample * win;
            let angle = angle_step * t.cast::<f32>();
            re = re + xw * cos(angle);
            im = im + xw * sin(angle);
        }
        let power = re * re + im * im;
        let w = load(mel_weight[mel_row + k]).cast::<f32>();
        mel_acc = mel_acc + w * power;
    }
    let log_mel = log(mel_acc + log_eps);
    store(out[idx], log_mel.cast::<T>());
}

// ─────────────────────────────────────────────────────────────────────────
// FFT-routed STFT path.
//
// `mel_spectrogram` does a direct DFT *inside every (frame, mel_bin)
// thread* — so the full O(n_freq·n_fft) power spectrum is recomputed
// `n_mels` times per frame. The FFT route splits it into three stages:
//
//   1. `mel_stft_window`  — extract + window each frame into FFT input
//                           planes (real = windowed sample, imag = 0).
//   2. `mt_fft_n{n_fft}`  — one radix-2 FFT per frame (O(n_fft·log n_fft)).
//   3. `mel_filterbank`   — power = re²+im², Mel-weight, log.
//
// The spectrum is now computed once per (frame, k) and the transform is
// O(N log N) instead of O(N²). `n_fft` must be a power of two (the
// `mt_fft_n*` set). The single-kernel `mel_spectrogram` is kept for
// non-pow2 `n_fft` and single-dispatch callers.
// ─────────────────────────────────────────────────────────────────────────

/// STFT stage 1 — extract and window each frame into the real/imag input
/// planes the `mt_fft_n*` kernels expect. `out_re[frame*n_fft + t] =
/// audio[frame*hop + t] · window[t]`, `out_im` zeroed. One thread per
/// `(frame, t)`; dispatch flat over `n_frames * n_fft`.
#[kernel(
    bench(
        op="mel_spectrogram",
        subop="stft_window",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mel_stft_window<T>(
    audio: Tensor<T>,
    window: Tensor<T>,
    mut out_re: Tensor<T>,
    mut out_im: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] hop_length: u32,
) {
    let idx = program_id::<0>();
    let t = idx % n_fft;
    let frame = idx / n_fft;
    let sample = load(audio[frame * hop_length + t]).cast::<f32>();
    let win = load(window[t]).cast::<f32>();
    store(out_re[idx], (sample * win).cast::<T>());
    store(out_im[idx], 0.0f32.cast::<T>());
}

/// STFT stage 3 — Mel filterbank over an FFT'd frame buffer. `out[frame,
/// mel] = log(Σ_{k<n_freq} mel_weight[mel,k]·(re²+im²) + log_eps)`, where
/// `re`/`im` are `fft_re`/`fft_im` from `mt_fft_n{n_fft}`. One thread per
/// `(frame, mel)`; dispatch flat over `n_frames * n_mels`. Output is
/// bit-identical in form to `mel_spectrogram` — only the spectrum source
/// (FFT vs in-thread DFT) differs.
#[kernel(
    bench(
        op="mel_spectrogram",
        subop="filterbank",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mel_filterbank<T>(
    fft_re: Tensor<T>,
    fft_im: Tensor<T>,
    mel_weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] n_mels: u32,
    #[constexpr] log_eps: f32,
) {
    let idx = program_id::<0>();
    let mel_bin = idx % n_mels;
    let frame = idx / n_mels;
    let frame_base = frame * n_fft;
    let mel_row = mel_bin * n_freq;
    let mut mel_acc = 0.0f32;
    for k in range(0u32, n_freq, 1u32) {
        let re = load(fft_re[frame_base + k]).cast::<f32>();
        let im = load(fft_im[frame_base + k]).cast::<f32>();
        let power = re * re + im * im;
        let w = load(mel_weight[mel_row + k]).cast::<f32>();
        mel_acc = mel_acc + w * power;
    }
    let log_mel = log(mel_acc + log_eps);
    store(out[idx], log_mel.cast::<T>());
}
