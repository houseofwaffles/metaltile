//! Steel NAX tiled GEMM — metal/steel/gemm/kernels/steel_gemm_fused_nax.metal
//!
//! NAX-optimized GEMM with larger block sizes for Apple Neural Engine co-scheduling:
//!   steel_gemm_fused_nax_{nn|nt|tn|tt}_{dtype}
//!   Block shapes: 64×64×256, 64×128×64, 64×128×256, 128×128×64/256/512
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Same simdgroup matrix blocker as steel_gemm_fused. Additionally
//!   feature-gated behind the `nax` feature flag — NAX kernels use
//!   Apple-internal memory layout extensions not available on all devices.
