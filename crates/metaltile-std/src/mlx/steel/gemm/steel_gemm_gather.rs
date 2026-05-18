//! Steel gather GEMM — metal/steel/gemm/kernels/steel_gemm_gather.metal
//!
//! Tiled GEMM with random-access (gather) row indexing into one operand:
//!   steel_gather_mm_{nn|nt|tn|tt}_{dtype}       — full gather matmul
//!   steel_gather_mm_rhs_{nn|nt}_{dtype}          — gather on RHS only
//!   Block shapes: 64×64×16, 64×32×32, 32×64×16, 32×32×16, 16×64×16
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires simdgroup matrix ops (same as steel_gemm_fused) plus
//!   indirect indexing — the gather index buffer maps output rows to
//!   non-contiguous input rows. The DSL has no gather/scatter load
//!   primitive compatible with tiled matmul staging.
