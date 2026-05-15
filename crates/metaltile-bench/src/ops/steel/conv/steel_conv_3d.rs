//! steel_conv_3d benchmarks — metal/steel/conv/steel_conv_3d.metal  (MLX, Apache-2.0)
//!
//! Tiled 3D convolution using SIMD matrix multiply instructions.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../../../metal/steel/conv/steel_conv_3d.metal");

pub fn bench_conv3d(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
