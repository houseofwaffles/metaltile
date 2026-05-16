//! fft benchmarks — metal/fft.metal  (MLX, Apache-2.0)
//!
//! FFT kernels:
//!   fft             — radix-2/4/8 Cooley-Tukey FFT
//!   rader_fft       — Rader's algorithm for prime-length FFT
//!   bluestein_fft   — Bluestein's algorithm for arbitrary-length FFT
//!   four_step_fft   — four-step FFT for large N
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   The MLX FFT kernels use bit-reversal permutation, butterfly
//!   operations with complex arithmetic, sin/cos twiddle-factor
//!   tables, and multi-pass shared-memory staging. These patterns
//!   require indirect indexing, complex number types, and
//!   stage-dependent threadgroup synchronisation that are not
//!   expressible in the current DSL primitives.
//!
//!   A direct O(N²) DFT would be trivial to write but is not a
//!   meaningful comparison against the O(N log N) MLX reference.

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/metal/fft.metal"));

pub fn bench_fft(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
