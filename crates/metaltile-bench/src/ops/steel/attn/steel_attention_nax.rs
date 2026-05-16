//! steel_attention_nax benchmarks — metal/steel/attn/steel_attention_nax.metal  (MLX, Apache-2.0)
//!
//! NAX variant of steel_attention for older hardware without SIMD matrix ops.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/attn/steel_attention_nax.metal"));

pub fn bench_steel_attention_nax(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
