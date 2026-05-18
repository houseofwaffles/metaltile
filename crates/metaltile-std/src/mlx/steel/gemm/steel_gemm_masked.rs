//! Steel block-masked GEMM — metal/steel/gemm/kernels/steel_gemm_masked.metal
//!
//! Tiled GEMM that skips output blocks and/or operand blocks based on masks:
//!   steel_gemm_block_outmask_{outmask}_opmask_{opmask}_{nn|nt|tn|tt}_{dtype}
//!   Output mask types: bool, float16/bfloat16/float32, nomask
//!   Op mask types: same
//!   Block shapes: 32×32×16, 64×64×16
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires simdgroup matrix ops (same as steel_gemm_fused) plus
//!   block-level predication — each threadgroup checks an output-block
//!   mask and early-exits before loading tiles. The DSL has no mechanism
//!   for block-level conditional dispatch or mask-indexed tile skipping.
