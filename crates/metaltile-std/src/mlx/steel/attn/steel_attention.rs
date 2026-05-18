//! Steel Flash Attention — metal/steel/attn/kernels/steel_attention.metal
//!
//! Tiled multi-head attention (SDPA) using simdgroup matrix ops:
//!   steel_attention_{dtype}_bq{Q}_bk{K}_bd{D}_wm{wm}_wn{wn}_mask{mtype}
//!   Block shapes (bq×bk×bd): 32×16×128, 32×32×80, 32×32×64
//!   Mask types: same dtype as input, bool
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Uses simdgroup matrix ops for the Q×K^T and P×V products within each
//!   attention tile, plus online-softmax across the K tile loop. The DSL
//!   `Op::FlashAttention` lowers to an error placeholder; no simdgroup
//!   matrix or multi-level attention tiling is implemented. Compare
//!   `scaled_dot_product_attention.rs` which implements a scalar SDPA
//!   sufficient for decode (small sequence) but not prefill workloads.
