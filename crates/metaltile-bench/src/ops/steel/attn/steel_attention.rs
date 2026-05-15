//! steel_attention benchmarks — metal/steel/attn/steel_attention.metal  (MLX, Apache-2.0)
//!
//! FlashAttention-style tiled SDPA for prefill (Q-sequence > 1).
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires online softmax with per-tile rescaling, double-buffered
//!   shared-memory tiles for Q/K/V, and causal/boolean mask logic
//!   integrated into the inner attention loop. The DSL has no support
//!   for multi-tile attention, online softmax state tracking, or
//!   tiled shared-memory staging beyond basic threadgroup_alloc.
//!
//!   The existing sdpa_vector kernel (scaled_dot_product_attention.rs)
//!   implements single-query decode attention as hand-written MSL —
//!   a decode-mode DSL kernel for Steel attention may be feasible
//!   once online-softmax primitives are added to the IR.

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../../../metal/steel/attn/steel_attention.metal");

pub fn bench_steel_attention(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }
