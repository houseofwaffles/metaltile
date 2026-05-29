//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Radix-2 Cooley–Tukey FFT along the last axis (size N = 2^k) — a
//! port of the radix path of MLX's `metal/fft.metal`.
//!
//! Computes the discrete Fourier transform of each length-`N` row:
//!
//!   X[k] = Σ_{n=0}^{N-1} x[n] · e^{∓ i 2π k n / N}
//!
//! The `−` sign is the forward transform; the `+` sign with a `1/N`
//! scale is the inverse. A single `inv` constexpr (`0` forward, `1`
//! inverse) selects between them, so one kernel covers `fft` and
//! `ifft`.
//!
//! ## Complex numbers without complex-type codegen
//!
//! The MLX kernel uses a `float2` complex type and `complex_mul`. The
//! metaltile DSL has no complex type — but it does not need one: a
//! complex array is just **two parallel real `f32` buffers**, one for
//! the real part and one for the imaginary part. This is the same
//! representation `mel_spectrogram` and `vocoder` already use for their
//! direct-DFT inner loops. The butterfly's complex multiply expands to
//! the textbook four-real-multiply form
//!
//!   (a+bi)(c+di) = (ac − bd) + (ad + bc) i
//!
//! so the whole transform is real arithmetic over two `threadgroup`
//! `f32` buffers. No codegen change is required — the existing
//! `threadgroup_alloc` / `_load` / `_store`, the bit ops (`<<`, `>>`,
//! `&`, `^`), `cos` / `sin` and `select` are sufficient.
//!
//! ## Algorithm — iterative radix-2 with bit-reversal
//!
//! 1. **Bit-reversal load.** Thread `tid` reads input element
//!    `bitrev(tid)` into `buf[tid]`. `bitrev` reverses the low
//!    `log2(N)` bits; it is computed with a `log2(N)`-iteration DSL
//!    loop (one shift / mask / or per bit).
//! 2. **`log2(N)` butterfly stages.** Stage `s` has half-block size
//!    `h = 2^s`. A thread whose index has bit `s` clear is the "top"
//!    of a butterfly: it combines `buf[tid]` and `buf[tid + h]` with
//!    the twiddle `w = e^{∓ i π (tid mod h) / h}`. A `threadgroup_barrier`
//!    separates stages.
//! 3. **Inverse scale.** For `inv = 1` the result is divided by `N`.
//!
//! This is a genuine O(N log N) transform — not a direct O(N²) DFT —
//! so it is a meaningful counterpart to the MLX radix kernel. The
//! prime-length (Rader) and arbitrary-length (Bluestein) paths from
//! `fft.metal` remain a follow-up; this covers the power-of-two radix
//! path that the STFT / iSTFT front-ends and the MLX `fft` op use most.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [N, 1, 1]`.
//! - `N` a power of two, `32 ≤ N ≤ 1024`; one thread per element.
//! - Input / output are split real / imaginary planes, each
//!   `[rows, N]`. A real-input transform passes an all-zero `in_im`.
//!
//! Codegen-only; correctness pinned by `tests/fft_gpu_correctness.rs`.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! fft_kernel {
    ($name:ident, $n:literal, $log_n:literal, $inv_n:literal, $subop:literal) => {
        /// Radix-2 FFT of one length-`N` row. `in_re` / `in_im` are the
        /// real / imaginary input planes, `out_re` / `out_im` the
        /// outputs; `inv` is `0` for the forward transform, `1` for the
        /// inverse (conjugated twiddles + `1/N` scale).
        #[kernel(
            bench(
                op="fft",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Reduction,
            )
        )]
        pub fn $name<T>(
            in_re: Tensor<T>,
            in_im: Tensor<T>,
            mut out_re: Tensor<T>,
            mut out_im: Tensor<T>,
            #[constexpr] inv: u32,
        ) {
            let row = program_id::<0>();
            let base = row * $n;

            threadgroup_alloc("re", $n, "f32");
            threadgroup_alloc("im", $n, "f32");

            // ---- bit-reversal permutation -------------------------------
            // Reverse the low log2(N) bits of `tid`. One shift/mask/or
            // per bit; `src` accumulates the reversed index.
            let mut src = 0u32;
            let mut rem = tid;
            for _b in range(0u32, $log_n, 1u32) {
                src = (src << 1u32) | (rem & 1u32);
                rem = rem >> 1u32;
            }
            // Load input element `src` into this thread's slot.
            threadgroup_store("re", tid, load(in_re[base + src]).cast::<f32>());
            threadgroup_store("im", tid, load(in_im[base + src]).cast::<f32>());
            threadgroup_barrier();

            // ---- log2(N) butterfly stages -------------------------------
            // Stage s: half-block h = 2^s. The twiddle-angle sign is
            // negative for the forward transform, positive for inverse.
            let pi = 3.141592653589793f32;
            let angle_sign = select(inv == 0u32, -1.0f32, 1.0f32);

            for s in range(0u32, $log_n, 1u32) {
                let h = 1u32 << s;
                // Top-of-butterfly threads: bit `s` of `tid` is clear.
                if (tid & h) == 0u32 {
                    // Twiddle exponent k = tid mod h, span = 2h.
                    let k = tid & (h - 1u32);
                    let h_f = h.cast::<f32>();
                    let angle = angle_sign * pi * k.cast::<f32>() / h_f;
                    let wr = cos(angle);
                    let wi = sin(angle);

                    let ar = threadgroup_load("re", tid);
                    let ai = threadgroup_load("im", tid);
                    let br = threadgroup_load("re", tid + h);
                    let bi = threadgroup_load("im", tid + h);

                    // t = w · b  (complex multiply, four-real-mul form).
                    let tr = wr * br - wi * bi;
                    let ti = wr * bi + wi * br;

                    // Butterfly: out[tid] = a + t, out[tid+h] = a − t.
                    threadgroup_store("re", tid, ar + tr);
                    threadgroup_store("im", tid, ai + ti);
                    threadgroup_store("re", tid + h, ar - tr);
                    threadgroup_store("im", tid + h, ai - ti);
                }
                threadgroup_barrier();
            }

            // ---- write back, inverse scale ------------------------------
            // Forward: scale 1. Inverse: 1/N (the `$inv_n` literal).
            let scale = select(inv == 0u32, 1.0f32, $inv_n);
            let res_re = threadgroup_load("re", tid) * scale;
            let res_im = threadgroup_load("im", tid) * scale;
            store(out_re[base + tid], res_re.cast::<T>());
            store(out_im[base + tid], res_im.cast::<T>());
        }
    };
}

fft_kernel!(mt_fft_n32, 32u32, 5u32, 0.031_25f32, "n32");
fft_kernel!(mt_fft_n64, 64u32, 6u32, 0.015_625f32, "n64");
fft_kernel!(mt_fft_n128, 128u32, 7u32, 0.007_812_5f32, "n128");
fft_kernel!(mt_fft_n256, 256u32, 8u32, 0.003_906_25f32, "n256");
fft_kernel!(mt_fft_n512, 512u32, 9u32, 0.001_953_125f32, "n512");
fft_kernel!(mt_fft_n1024, 1024u32, 10u32, 0.000_976_562_5f32, "n1024");

// ── Bluestein chirp-Z transform — arbitrary-length DFT ───────────────────
//
// Implements DFT of an arbitrary-length N signal in O(N log N) time using
// Bluestein's algorithm. The three-step approach:
//
//   1. Pre-multiply (bluestein_preprocess): per-sample pointwise multiply
//      input `x[n]` by chirp `w[n] = exp(±iπn²/N)`, zero-pad to the next
//      power-of-two M ≥ 2N.
//
//   2. Convolution via radix-2 FFT: the caller dispatches the existing
//      `mt_fft_n*` kernel on the padded M-length sequence (and on the
//      pre-computed chirp sequence `a[n]` for the convolution filter). The
//      element-wise product in frequency domain followed by an IFFT computes
//      the circular convolution.
//
//   3. Post-multiply (bluestein_postprocess): multiply each DFT output bin
//      `k` by `exp(∓iπk²/N)` and scale by `1/N` for the inverse transform.
//
// This module provides step 1 and step 3 as GPU kernels. Step 2 (the FFT
// convolution) uses the existing `mt_fft_n1024` (M=1024 covers both
// N=400: 2×400=800 ≤ 1024, and N=480: 2×480=960 ≤ 1024).
//
// Additionally, `bluestein_chirp_filter` pre-computes the frequency-domain
// chirp filter `A[k] = FFT(a)[k]` where `a[n] = exp(-iπn²/N)` for
// `n ∈ [0,N)`, `a[M-n] = a[n]` for `n ∈ [1,N)`, and zeros elsewhere.
// This filter only depends on N and M so it can be pre-computed once per
// model load.
//
// ## Bluestein identity
//
// The key identity: `nk = (n² + k² - (k-n)²) / 2`. Substituting into the
// forward DFT sum `X[k] = Σ_n x[n] exp(-i2πnk/N)`:
//
//   X[k] = exp(-iπk²/N) · Σ_n [x[n] · exp(-iπn²/N)] · exp(+iπ(k-n)²/N)
//
// So pre-multiply by `exp(-iπn²/N)`, convolve against kernel
// `a[m] = exp(+iπm²/N)` (positive-sign chirp), post-multiply by
// `exp(-iπk²/N)`. The chirp pre-multiply turns the DFT into a linear
// convolution in m=(k-n), which the FFT evaluates in O(N log N).
//
// ## Usage pattern (caller side)
//
//   bluestein_preprocess<T>(x_re, x_im, chirp_re, chirp_im, N, M, inv=0|1)
//     → padded `[rows, M]` pre-multiplied sequences
//   mt_fft_n1024<T>(filter_re, filter_im, F_re, F_im, inv=0)   // once per N
//   mt_fft_n1024<T>(padded_re, padded_im, Y_re, Y_im, inv=0)   // per batch
//   elementwise_cmul(Y_re, Y_im, F_re, F_im)                   // in-place
//   mt_fft_n1024<T>(Y_re, Y_im, conv_re, conv_im, inv=1)        // IFFT
//   bluestein_postprocess<T>(conv_re, conv_im, out_re, out_im, N, M, inv)
//     → final DFT output `[rows, N]`

/// Bluestein step 1: pre-multiply + zero-pad.
///
/// For each row and each sample `n ∈ [0, N)`:
///   `pre[n] = x[n] · chirp[n]`   (complex multiply)
///   chirp[n] = `exp(±iπn²/N)` — `+` for forward (inv=0), `−` for inverse.
///   `pre[m] = 0` for `m ∈ [N, M)` (zero-pad to radix-2 length M).
///
/// Input: `in_re / in_im [rows, N]`; output: `out_re / out_im [rows, M]`.
/// Constexprs: `n_len` (original length), `m_len` (padded power-of-two
/// length), `rows` (batch dimension), `inv` (0 forward, 1 inverse).
///
/// Grid3D: one thread per element of `[rows, M]`. `rows` is a constexpr
/// so the kernel can bounds-check against `rows * m_len` — the caller
/// dispatches `ceil(rows*M/tpg)` workgroups and the trailing threads
/// must skip OOB writes (Apple Silicon silently lands OOB writes into
/// adjacent buffer slots and corrupts the output).
#[kernel(
    bench(
        op="fft",
        subop="bluestein_preprocess",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_fft_bluestein_preprocess<T>(
    in_re: Tensor<T>,
    in_im: Tensor<T>,
    mut out_re: Tensor<T>,
    mut out_im: Tensor<T>,
    #[constexpr] n_len: u32,
    #[constexpr] m_len: u32,
    #[constexpr] rows: u32,
    #[constexpr] inv: u32,
) {
    // One thread per output element (row, col) in [rows, M].
    let idx = program_id::<0>();
    // Bounds guard for trailing threads when rows*M isn't a multiple of
    // tpg — wrap the whole body so OOB threads write nothing. (DSL has
    // no `return`; an outer `if` is the only escape hatch.)
    if idx < rows * m_len {
        let col = idx % m_len;
        let row = idx / m_len;
        let pi = 3.141592653589793f32;
        // Standard Bluestein for the forward DFT uses exp(-iπn²/N) on
        // the input side and exp(-iπk²/N) on the output side (paired
        // with a positive-sign convolution kernel built by
        // `mt_fft_bluestein_chirp_filter`). Inverse flips both signs.
        //   forward (inv=0): angle_sign = -1   (chirp = exp(-iπn²/N))
        //   inverse (inv=1): angle_sign = +1   (chirp = exp(+iπn²/N))
        let angle_sign = select(inv == 0u32, -1.0f32, 1.0f32);
        // Zero-pad region: col >= n_len writes zero.
        if col >= n_len {
            store(out_re[row * m_len + col], 0.0f32.cast::<T>());
            store(out_im[row * m_len + col], 0.0f32.cast::<T>());
        } else {
            // Chirp angle: angle_sign * π * n² / N.
            let n_f = col.cast::<f32>();
            let n_len_f = n_len.cast::<f32>();
            let angle = angle_sign * pi * n_f * n_f / n_len_f;
            let wr = cos(angle);
            let wi = sin(angle);
            // Load input (complex).
            let xr = load(in_re[row * n_len + col]).cast::<f32>();
            let xi = load(in_im[row * n_len + col]).cast::<f32>();
            // Complex multiply: (xr + xi·i)(wr + wi·i).
            let pr = xr * wr - xi * wi;
            let pi_v = xr * wi + xi * wr;
            store(out_re[row * m_len + col], pr.cast::<T>());
            store(out_im[row * m_len + col], pi_v.cast::<T>());
        }
    }
}

/// Bluestein step 2: build the chirp convolution filter in the time domain.
///
/// Computes `a[m]` for the circular convolution `filter_re / filter_im`
/// `[1, M]` (single row, M = padded power-of-two length):
///
///   a[0]     = 1 + 0i              (n=0 chirp is always 1)
///   a[n]     = exp(+i π n² / N)    for n ∈ [1, N)   (positive-sign chirp)
///   a[M-n]   = a[n]                for n ∈ [1, N)   (time-reversal — `a` is symmetric since |n|²=|-n|²)
///   a[m]     = 0                   for m ∈ [N, M-N+1)
///
/// The positive sign matches the Bluestein identity for the forward DFT
/// when pre/postprocess use `exp(-iπn²/N)` (see module docs). Using the
/// negative sign here would make the convolution sum collapse to noise.
///
/// This is a single-row kernel (grid = [M, 1, 1]).
///
/// The caller FFTs this filter once and stores it for reuse across all
/// rows/frames.
#[kernel(
    bench(
        op="fft",
        subop="bluestein_chirp_filter",
        class=GenericEmpty,
        tol=1e-3,
        dtypes=&[DType::F32],
        kernel_mode=Grid3D,
    )
)]
pub fn mt_fft_bluestein_chirp_filter(
    mut filter_re: Tensor<f32>,
    mut filter_im: Tensor<f32>,
    #[constexpr] n_len: u32,
    #[constexpr] m_len: u32,
) {
    let m = program_id::<0>(); // column index in [0, M)
    let pi = 3.141592653589793f32;
    // n = min(m, M-m) — the "wrapped" tap index.
    let m_minus = m_len - m;
    let n_tap = select(m < n_len, m, select(m_minus < n_len, m_minus, n_len));
    let in_range = (m < n_len) | ((m_minus < n_len) & (m > 0u32));
    if in_range {
        let n_f = n_tap.cast::<f32>();
        let n_len_f = n_len.cast::<f32>();
        // Positive-sign chirp: exp(+iπn²/N). Matches the Bluestein
        // identity's `exp(+iπ(k-n)²/N)` convolution kernel.
        let angle = pi * n_f * n_f / n_len_f;
        let wr = cos(angle);
        let wi = sin(angle);
        store(filter_re[m], wr);
        store(filter_im[m], wi);
    } else {
        store(filter_re[m], 0.0f32);
        store(filter_im[m], 0.0f32);
    }
}

/// Bluestein convolution step: element-wise complex multiply of two `[rows, M]`
/// frequency-domain arrays. Performed between the two FFT calls:
///
///   Y[k] = Y_input[k] · F_filter[k]
///
/// Both inputs are `[rows, M]`; `filter_re / filter_im` are `[1, M]` and
/// broadcast across rows. Generic over `T` for the per-row input;
/// the filter is always f32 (pre-computed once as f32).
#[kernel(
    bench(
        op="fft",
        subop="bluestein_cmul",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_fft_bluestein_cmul<T>(
    y_re: Tensor<T>,
    y_im: Tensor<T>,
    filter_re: Tensor<f32>,
    filter_im: Tensor<f32>,
    mut out_re: Tensor<T>,
    mut out_im: Tensor<T>,
    #[constexpr] m_len: u32,
    #[constexpr] rows: u32,
) {
    let idx = program_id::<0>();
    // Bounds guard — see preprocess kernel for rationale.
    if idx < rows * m_len {
        let col = idx % m_len;
        let yr = load(y_re[idx]).cast::<f32>();
        let yi = load(y_im[idx]).cast::<f32>();
        // Filter broadcasts across rows (index by col only).
        let fr = load(filter_re[col]);
        let fi = load(filter_im[col]);
        // Complex multiply.
        let pr = yr * fr - yi * fi;
        let pi_v = yr * fi + yi * fr;
        store(out_re[idx], pr.cast::<T>());
        store(out_im[idx], pi_v.cast::<T>());
    }
}

/// Bluestein step 3: post-multiply and extract N outputs from M-length IFFT.
///
/// For each row and each output bin `k ∈ [0, N)`:
///   `X[k] = conv[k] · exp(angle_sign · iπk²/N) [· scale]`
///
/// where:
///   - `conv_re / conv_im [rows, M]` is the IFFT output of the circular
///     convolution (already inverse-scaled by 1/M from the FFT kernel).
///   - `angle_sign = -1` for forward (inv=0), `+1` for inverse (inv=1).
///   - `scale = 1/N` for the inverse transform, `1` for forward.
///   - Output is `[rows, N]` (truncated to the first N bins).
///
/// Grid3D: one thread per element of `[rows, N]`.
#[kernel(
    bench(
        op="fft",
        subop="bluestein_postprocess",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_fft_bluestein_postprocess<T>(
    conv_re: Tensor<T>,
    conv_im: Tensor<T>,
    mut out_re: Tensor<T>,
    mut out_im: Tensor<T>,
    #[constexpr] n_len: u32,
    #[constexpr] m_len: u32,
    #[constexpr] rows: u32,
    #[constexpr] inv: u32,
) {
    let idx = program_id::<0>();
    // Bounds guard — see preprocess kernel for rationale.
    if idx < rows * n_len {
        let k = idx % n_len;
        let row = idx / n_len;
        let pi = 3.141592653589793f32;
        // Post-multiply chirp: same sign convention as pre-multiply.
        let angle_sign = select(inv == 0u32, -1.0f32, 1.0f32);
        let k_f = k.cast::<f32>();
        let n_len_f = n_len.cast::<f32>();
        let angle = angle_sign * pi * k_f * k_f / n_len_f;
        let wr = cos(angle);
        let wi = sin(angle);
        // Load from the IFFT output at position (row, k). The circular
        // convolution result is in [rows, M]; we only need the first N values.
        let cr = load(conv_re[row * m_len + k]).cast::<f32>();
        let ci = load(conv_im[row * m_len + k]).cast::<f32>();
        // Complex multiply.
        let pr = cr * wr - ci * wi;
        let pi_v = cr * wi + ci * wr;
        // Inverse scale: 1/N for the inverse DFT, 1 for forward.
        let scale = select(inv == 0u32, 1.0f32, 1.0f32 / n_len_f);
        store(out_re[idx], (pr * scale).cast::<T>());
        store(out_im[idx], (pi_v * scale).cast::<T>());
    }
}
