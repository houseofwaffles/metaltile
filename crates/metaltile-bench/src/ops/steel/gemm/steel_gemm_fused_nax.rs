//! steel_gemm_fused_nax benchmarks — metal/steel/gemm/steel_gemm_fused_nax.metal  (MLX, Apache-2.0)
//!
//! NAX (Non-Apple-Silicon) variant of steel_gemm_fused for older hardware.
//! Same kernel names as steel_gemm_fused but instantiated for non-SIMD-matrix hardware.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_fused_nax.metal"));

pub fn bench_matmul_fp16_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
