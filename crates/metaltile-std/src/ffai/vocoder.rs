//! Inverse-STFT overlap-add — the TTS vocoder waveform-synthesis tail.
//!
//! iSTFTNet-style vocoders (Kokoro, the StyleTTS2 decoder family) end by
//! turning a predicted complex spectrogram back into a time-domain
//! waveform: inverse-DFT each frame, apply the synthesis window,
//! overlap-add the windowed frames, and divide by the overlapped
//! window-energy so the redundancy of the overlap is normalised out.
//! This kernel is that whole iSTFT tail.
//!
//! The spectrogram arrives as separate real / imaginary planes (the
//! `n_freq = n_fft/2 + 1` non-redundant bins of a real-input STFT).
//!
//! Layouts:
//!
//!   spec_re   [n_frames, n_freq]   T   real part of each STFT bin
//!   spec_im   [n_frames, n_freq]   T   imaginary part
//!   window    [n_fft]              T   synthesis window
//!   out       [out_len]            T   reconstructed waveform
//!
//!   n_freq  = n_fft / 2 + 1
//!   out_len = (n_frames - 1) * hop_length + n_fft
//!   frame f writes into output samples [f*hop_length, f*hop_length + n_fft)
//!
//! ## Why one thread per output sample
//!
//! The natural form of overlap-add is one thread per frame scattering
//! into the output — but that needs atomic accumulation, since adjacent
//! frames write the same samples. This kernel inverts the loop: **one
//! thread per output sample `t`**, gathering every frame that overlaps
//! `t`. No atomics, no inter-thread sync, each output written exactly
//! once.
//!
//! A sample `t` is covered by frame `f` iff `f*hop <= t < f*hop + n_fft`,
//! i.e. `f ∈ [ceil((t-n_fft+1)/hop), floor(t/hop)]`. For each covering
//! frame the thread computes the inverse DFT at in-frame offset
//! `tau = t - f*hop` directly:
//!
//!   x[tau] = (1/n_fft) * Σ_{k=0}^{n_fft-1} Re( X[k] · e^{+i 2π k tau / n_fft} )
//!
//! Using the Hermitian symmetry of a real-signal spectrum
//! (`X[n_fft-k] = conj(X[k])`), only the `n_freq` stored bins are read:
//! bin 0 and (for even `n_fft`) bin `n_fft/2` count once, every other
//! bin counts twice. The windowed inverse-DFT value is accumulated into
//! `num`, and the squared synthesis-window value into `den`; the output
//! is `num / den` — the standard COLA (constant-overlap-add)
//! normalisation. `den` is guarded against zero at the signal edges.
//!
//! A direct inverse DFT is O(n_fft) per covering frame; iSTFTNet vocoder
//! tails run small `n_fft` (Kokoro: 20, hop 5) so this is cheap. A
//! radix-FFT inverse is a perf follow-up (needs complex-type codegen).
//!
//! Codegen-only. Correctness validated by `vocoder_gpu_correctness`.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="vocoder",
    subop="istft",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn vocoder_istft<T>(
    spec_re: Tensor<T>,
    spec_im: Tensor<T>,
    window: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_frames: u32,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] hop_length: u32,
) {
    // One thread per output sample.
    let t = program_id::<0>();

    let n_fft_f = n_fft.cast::<f32>();
    let inv_n = 1.0f32 / n_fft_f;
    // +2π / n_fft — inverse-DFT twiddle-angle step (note the + sign;
    // the forward transform uses −).
    let two_pi_over_n = 6.283185307179586f32 / n_fft_f;
    let nyquist = n_fft / 2u32; // last bin with no conjugate partner (even n_fft)

    // Frame range covering sample t: f*hop <= t < f*hop + n_fft.
    //   lo = ceil((t - n_fft + 1) / hop), clamped at 0
    //   hi = min(t / hop, n_frames - 1)
    let f_hi_raw = t / hop_length;
    let f_hi = select(f_hi_raw < n_frames, f_hi_raw, n_frames - 1u32);
    // (t - n_fft + 1) can be negative — guard before the unsigned divide.
    let has_lo = t + 1u32 > n_fft;
    let lo_num = select(has_lo, t + 1u32 - n_fft, 0u32);
    // ceil-divide of a non-negative numerator by hop.
    let f_lo = select(has_lo, (lo_num + hop_length - 1u32) / hop_length, 0u32);

    let mut num = 0.0f32;
    let mut den = 0.0f32;

    // Gather every covering frame. `f_lo > f_hi` can only happen when
    // the range is empty, which the covering-set algebra rules out for
    // a valid out_len — but the loop bound is inclusive of f_hi, so it
    // simply runs zero times if the range degenerates.
    for f in range(f_lo, f_hi + 1u32, 1u32) {
        let tau = t - f * hop_length; // in-frame offset, 0 <= tau < n_fft
        let tau_f = tau.cast::<f32>();
        let angle_step = two_pi_over_n * tau_f;
        let row = f * n_freq;

        // Inverse DFT at offset tau using Hermitian symmetry: bins 1..nyquist
        // each stand in for their conjugate partner, so they count twice;
        // bin 0 and the Nyquist bin count once.
        let mut sample = 0.0f32;
        for k in range(0u32, n_freq, 1u32) {
            let re = load(spec_re[row + k]).cast::<f32>();
            let im = load(spec_im[row + k]).cast::<f32>();
            let angle = angle_step * k.cast::<f32>();
            // Re( X[k] · e^{+i angle} ) = re*cos(angle) - im*sin(angle).
            let contrib = re * cos(angle) - im * sin(angle);
            // Mirror weight: 1 for DC and Nyquist, 2 otherwise.
            let is_unpaired = (k == 0u32) | (k == nyquist);
            let weight_k = select(is_unpaired, 1.0f32, 2.0f32);
            sample = sample + weight_k * contrib;
        }
        sample = sample * inv_n;

        // Apply the synthesis window and overlap-add. The COLA
        // normaliser is the sum of squared window taps that land on t.
        let win = load(window[tau]).cast::<f32>();
        num = num + sample * win;
        den = den + win * win;
    }

    // COLA normalisation; guard the signal edges where den underflows.
    let safe_den = select(den > 1e-8f32, den, 1.0f32);
    let out_val = select(den > 1e-8f32, num / safe_den, 0.0f32);
    store(out[t], out_val.cast::<T>());
}
