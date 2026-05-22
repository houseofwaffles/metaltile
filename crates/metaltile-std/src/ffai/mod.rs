//! FFAI / model-specific kernels.
//!
//! Kernels here are ports from FFAI / mlx-swift-lm / ekryski's `mlx` fork
//! that don't have a matching template in mainline MLX at the pinned
//! commit (see `metaltile-std/build.rs` — `MLX_COMMIT`). They register
//! a `BenchSpec` so `tile build` / `tile inspect` can find them, but
//! the spec uses `shapes: &[]` and `dispatch: BenchDispatch::Generic`,
//! so `tile bench` skips them (no MLX side-by-side, no GPU shapes).
//!
//! Correctness for these kernels is validated end-to-end in FFAI's
//! integration tests against real models. Once a kernel has been
//! verified there, the shape spec / bench dispatch can be added back
//! here so `tile bench` can track it for regressions — and if its MLX
//! counterpart lands in mainline at a future pin, the file moves to
//! `mlx/`.

pub mod arg_reduce;
pub mod aura_dequant_rotated;
pub mod aura_encode;
pub mod aura_flash_p1;
pub mod aura_flash_pass2;
pub mod aura_flash_sdpa;
pub mod aura_score;
pub mod aura_value;
pub mod batched_qkv_qgemv;
pub mod dequant_gather;
pub mod dequant_gemv;
pub mod flash_quantized_sdpa;
pub mod gated_delta;
pub mod gated_delta_prep;
pub mod gated_delta_replay;
pub mod gated_delta_wy;
pub mod gather;
pub mod kv_cache;
pub mod logits_min_p;
pub mod logits_processors;
pub mod logits_top_p;
pub mod logits_topk;
pub mod moe;
pub mod moe_mpp;
pub mod moe_mpp_bm64;
pub mod moe_mpp_bm8;
pub mod rms_norm_qgemv;
pub mod rms_norm_residual;
pub mod rms_norm_rope;
pub mod rope_llama;
pub mod sampling;
pub mod sdpa_decode;
pub mod sdpa_decode_2pass;
pub mod sdpa_decode_batched;
pub mod sdpa_decode_batched_prefill;
pub mod sdpa_decode_d256;
pub mod sdpa_decode_d64;
pub mod ssm;
pub mod ssm_replay;
