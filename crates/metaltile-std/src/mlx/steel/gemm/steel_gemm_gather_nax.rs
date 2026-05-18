//! Steel NAX gather GEMM — metal/steel/gemm/kernels/steel_gemm_gather_nax.metal
//!
//! NAX-optimized gather-on-RHS GEMM:
//!   steel_gather_mm_rhs_nax_{nn|nt}_{dtype}
//!   Block shapes: 16×128×128, 32×128×128, 64×128×128
//!   Dtypes: float16, bfloat16
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Same blockers as steel_gemm_gather plus NAX feature gate.
