//! Steel NAX Flash Attention — metal/steel/attn/kernels/steel_attention_nax.metal
//!
//! NAX-optimized Flash Attention with larger query tiles:
//!   steel_attention_{dtype}_bq{Q}_bk{K}_bd{D}_wm{wm}_wn{wn}_mask{mtype}
//!   Block shapes (bq×bk×bd): 64×32×128, 64×32×64, 64×64×128, 64×64×64
//!   Mask types: same dtype as input, bool
//!   Dtypes: float16, bfloat16, float32
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Same blockers as steel_attention plus NAX feature gate.
