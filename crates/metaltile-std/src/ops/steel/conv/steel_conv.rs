//! steel_conv benchmarks — metal/steel/conv/steel_conv.metal  (MLX, Apache-2.0)
//!
//! Tiled 2D convolution via implicit-GEMM (im2col + matmul fusion).
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   The MLX kernel fuses im2col memory layout transformation with
//!   tiled matmul using SIMD matrix instructions. This requires
//!   indirect strided memory access and runtime-shape-dependent
//!   tiling that are not expressible in the current DSL primitives.
