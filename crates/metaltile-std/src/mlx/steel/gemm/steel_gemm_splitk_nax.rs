//! Steel NAX split-K GEMM — metal/steel/gemm/kernels/steel_gemm_splitk_nax.metal
//!
//! NAX-optimized split-K GEMM with large blocks:
//!   steel_gemm_splitk_nax_{nn|nt|tn|tt}_{dtype}
//!   Block shapes: 64×64×256, 128×128×512
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Same blockers as steel_gemm_splitk plus NAX feature gate.
