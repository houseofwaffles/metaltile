//! fp_quantized_nax benchmarks — metal/fp_quantized_nax.metal  (MLX, Apache-2.0)
//!
//! NAX (non-Apple-Silicon) variants of fp_quantized kernels.
//!
//! Kernels:
//!   fp_qmm_t_nax, fp_qmm_n_nax
//!   fp_gather_qmm_t_nax, fp_gather_qmm_n_nax, fp_gather_qmm_rhs_nax
//!
//! TODO: implement benchmarks (skip on Apple Silicon — NAX targets older hardware)

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../metal/fp_quantized_nax.metal");

pub fn bench_fp_quantized_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
