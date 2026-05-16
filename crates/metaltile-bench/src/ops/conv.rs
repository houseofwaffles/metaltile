//! conv benchmarks — metal/conv.metal  (MLX, Apache-2.0)
//!
//! Naive (unfold-based) convolution fallback kernels:
//!   naive_unfold_Nd            — forward N-D unfold
//!   naive_unfold_transpose_Nd  — transposed (gradient) unfold
//!   depthwise_conv_2d          — depthwise 2D conv
//!   depthwise_conv_1d          — depthwise 1D conv
//!   winograd_conv_2d           — Winograd-transformed 2D conv
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   The MLX conv kernels use im2col/unfold + tiled GEMM or Winograd
//!   transforms. These require runtime-shape-dependent shared-memory
//!   blocking, multiple levels of tiling, and indirect indexing that
//!   are not expressible in the current DSL primitives.
//!
//!   A direct convolution (each thread computes one output pixel via
//!   nested loops over filter dimensions) is possible but would be
//!   orders of magnitude slower than the MLX reference and not
//!   a meaningful comparison.

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/metal/conv.metal"));

pub fn bench_conv(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
