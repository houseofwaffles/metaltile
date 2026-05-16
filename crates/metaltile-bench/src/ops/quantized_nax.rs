//! quantized_nax benchmarks — metal/quantized_nax.metal  (MLX, Apache-2.0)
//!
//! NAX (non-Apple-Silicon) variants of quantized kernels.
//!
//! Kernels:
//!   affine_qmm_t_nax, affine_qmm_n_nax
//!   affine_gather_qmm_t_nax, affine_gather_qmm_n_nax, affine_gather_qmm_rhs_nax
//!
//! TODO: implement benchmarks (skip on Apple Silicon — NAX targets older hardware)

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/metal/quantized_nax.metal"));

pub fn bench_quantized_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
