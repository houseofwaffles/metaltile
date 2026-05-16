//! steel_gemm_gather_nax benchmarks — metal/steel/gemm/steel_gemm_gather_nax.metal  (MLX, Apache-2.0)
//!
//! NAX variant of steel_gemm_gather.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_gather_nax.metal"));

pub fn bench_matmul_gather_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
