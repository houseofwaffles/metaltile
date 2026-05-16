//! steel_gemm_splitk benchmarks  (MLX, Apache-2.0)
//!
//! Split-K GEMM: tiles the K dimension across threadgroups with
//! an accumulation pass to combine partial sums.
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires a two-kernel pipeline: a compute kernel writes partial
//!   accumulated sums to a scratch buffer, then an accumulation kernel
//!   combines them into the final output. The DSL has no support for
//!   multi-kernel workflows, cross-threadgroup scratch buffers, or
//!   atomic accumulation. With k_partitions=1 the compute kernel
//!   alone is equivalent to plain matmul, but the GEMMSplitKParams
//!   struct layout differs from GEMMParams.

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_splitk.metal"));

pub fn bench_matmul_splitk(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
