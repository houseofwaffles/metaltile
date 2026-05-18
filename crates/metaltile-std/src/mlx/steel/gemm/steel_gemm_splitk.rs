//! Steel split-K GEMM — metal/steel/gemm/kernels/steel_gemm_splitk.metal
//!
//! GEMM that partitions the K dimension across threadgroups and reduces partial
//! sums in a second pass:
//!   steel_gemm_splitk_{nn|nt|tn|tt}_{dtype}_{MN_aligned}_{K_aligned}
//!   steel_gemm_splitk_accum_{dtype}_{accum_dtype}        — reduction pass
//!   steel_gemm_splitk_accum_{dtype}_{accum_dtype}_axbpy  — α·X + β·Y reduction
//!   Block shapes: 16×16, 16×32, 32×16, 32×32 (all ×16 K)
//!   Input dtypes: float16, bfloat16, float32, complex64
//!   Accumulator: float32 (for f16/bf16 inputs), complex64
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires simdgroup matrix ops (same as steel_gemm_fused) plus a
//!   two-kernel dispatch pattern: the split-K kernel writes partial sums
//!   to a temporary float32 buffer, then a separate accumulation kernel
//!   reduces them. The DSL has no split-K scheduling primitive or
//!   inter-kernel temporary buffer handoff.
