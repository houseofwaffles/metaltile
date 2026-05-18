//! Steel implicit-GEMM 3D conv — metal/steel/conv/kernels/steel_conv_3d.metal
//!
//! 3D convolution (D×H×W input) via implicit GEMM:
//!   implicit_gemm_conv_3d_{dtype}_bm{M}_bn{N}_bk{K}_wm{wm}_wn{wn}_filter_{f}
//!   Block shapes: 32×8, 64×8, 32×32, 32×64, 64×32, 64×64 (all ×16 K)
//!   Filter variants: s (small), l (large)
//!   Dtypes: float32, float16, bfloat16
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Same blockers as steel_conv with the additional 3D volume indexing
//!   over `MLXConvParams<3>`. No DSL support for 3D im2col or simdgroup
//!   matrix ops.
