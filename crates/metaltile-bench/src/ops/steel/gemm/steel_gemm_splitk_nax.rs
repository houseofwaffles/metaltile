//! steel_gemm_splitk_nax benchmarks — metal/steel/gemm/steel_gemm_splitk_nax.metal  (MLX, Apache-2.0)
//!
//! NAX variant of steel_gemm_splitk.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../../../metal/steel/gemm/steel_gemm_splitk_nax.metal");

pub fn bench_matmul_splitk_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
