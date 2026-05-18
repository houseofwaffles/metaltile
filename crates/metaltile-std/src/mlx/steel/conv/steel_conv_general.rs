//! Steel general implicit-GEMM 2D conv — metal/steel/conv/kernels/steel_conv_general.metal
//!
//! 2D convolution supporting arbitrary strides, dilation, padding, and groups:
//!   implicit_gemm_conv_2d_general_{dtype}_bm{M}_bn{N}_bk{K}_wm{wm}_wn{wn}
//!   Block shapes: 32×8, 64×8, 32×32, 32×64, 64×32, 64×64 (all ×16 K)
//!   Dtypes: float32, float16, bfloat16
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Extends steel_conv with `Conv2DGeneralJumpParams` and
//!   `Conv2DGeneralBaseInfo` for non-unit strides/dilation and group
//!   convolution. Same simdgroup matrix + im2col blockers as steel_conv.
