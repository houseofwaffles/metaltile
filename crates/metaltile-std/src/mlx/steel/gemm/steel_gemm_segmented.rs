//! Steel segmented GEMM — metal/steel/gemm/kernels/steel_gemm_segmented.metal
//!
//! Batched GEMM where each batch segment can have a different K extent:
//!   steel_segmented_mm_{nn|nt|tn|tt}_{dtype}
//!   Block shapes: 64×64×16, 64×32×32, 32×64×16, 32×32×16
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires simdgroup matrix ops (same as steel_gemm_fused) plus
//!   per-segment K offsets stored in a segment descriptor buffer. The
//!   DSL has no notion of ragged/variable-K batched matmul.
