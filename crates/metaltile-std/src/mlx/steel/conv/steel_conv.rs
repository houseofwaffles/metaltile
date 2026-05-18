//! Steel implicit-GEMM 2D conv — metal/steel/conv/kernels/steel_conv.metal
//!
//! 2D convolution via implicit GEMM (im2col × filter matrix):
//!   implicit_gemm_conv_2d_{dtype}_bm{M}_bn{N}_bk{K}_wm{wm}_wn{wn}_channel_{c}_filter_{f}
//!   Block shapes: 32×8, 64×8, 32×32, 32×64, 64×32, 64×64 (all ×16 K)
//!   Channel variants: l (general), 1/2/3/4 (small fixed channel count)
//!   Filter variants: s (small/separable), l (large)
//!   Dtypes: float32, float16, bfloat16
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Implicit GEMM convolution unfolds the input patch neighbourhood
//!   into a virtual matrix via the `MLXConvParams` descriptor, then
//!   runs tiled GEMM over the unfolded layout. This requires simdgroup
//!   matrix ops (same blocker as steel_gemm_fused) plus the im2col
//!   index arithmetic driven by `ImplicitGemmConv2DParams`. The DSL
//!   has neither simdgroup matmul nor im2col primitives.
