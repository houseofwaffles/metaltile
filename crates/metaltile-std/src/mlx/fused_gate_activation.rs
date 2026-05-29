//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused gate-activation — `activation(gate) * up`.
//!
//! The general form of the SwiGLU MLP activation. Given two
//! equally-sized inputs `gate` and `up` (the two halves of an MLP's
//! `w_gate · x` and `w_up · x` projections), produce
//!
//! ```text
//!   out[i] = activation(gate[i]) * up[i]
//! ```
//!
//! The `silu` activation is the standard modern-transformer case and
//! already ships as the dedicated [`mt_swiglu`](super::swiglu) kernel.
//! This module covers the other two activation variants MLX's
//! `fused_gate_activation.metal` carries:
//!
//! - **`gelu_approx`** — the tanh approximation of GELU
//!   (`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`). Used by GELU-MLP
//!   transformers that ship a fused gate/up projection.
//! - **`clipped_swiglu`** — the GPT-OSS clipped variant: both halves
//!   are clamped to `[-7, 7]`, the gate side uses `sigmoid(1.702·g)`,
//!   and the up side carries a `+1` bias before the multiply.
//!
//! All three are one-thread-per-output Grid3D kernels — no
//! cross-thread cooperation, so the reduction-mode dispatch hazards do
//! not apply. The `single_row` / `looped` dispatch split MLX uses is a
//! threadgroup-tiling perf detail; a flat element-parallel grid is the
//! tractable port and is what FFAI's elementwise activation path wants.
//!
//! MLX reference: `mlx/backend/metal/kernels/fused_gate_activation.metal`
//! — `apply_gate<activation_type>` with `activation_type ∈ {0,1,2}`.
//! `activation_type == 0` (silu) is `mt_swiglu`; this file ports `1`
//! and `2`.
//!
//! Codegen-only; correctness pinned by
//! `tests/fused_gate_activation_gpu_correctness.rs`.

use metaltile::kernel;

// Numeric constants are inlined as literals inside the kernel bodies
// below — the `#[kernel]` proc-macro parses the body as a token stream
// and does not substitute Rust `const` items. They are named here for
// documentation only:
//   GELU_SQRT_2_OVER_PI    = 0.7978845608  — √(2/π), the gelu-tanh
//                                            inner scale.
//   GELU_CUBIC_COEFF       = 0.044715      — gelu-tanh cubic coeff.
//   CLIPPED_SWIGLU_CLIP    = 7.0           — GPT-OSS clamp bound.
//   CLIPPED_SWIGLU_GATE_SCALE = 1.702      — GPT-OSS gate sigmoid scale.

/// `out[i] = gelu_approx(gate[i]) * up[i]`.
///
/// GELU via the tanh approximation MLX uses — matches
/// `gelu_approx_act` in `fused_gate_activation.metal`. Computed in f32
/// regardless of `T` so the cubic + tanh keep their precision; the
/// result is cast back to `T` at the store.
#[kernel(
    bench(
        op="fused_gate_activation",
        subop="gelu_approx",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_fused_gate_gelu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // gelu_approx(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    let x3 = g * g * g;
    let inner = 0.7978845608f32 * (g + 0.044715f32 * x3);
    let act = 0.5f32 * g * (1.0f32 + tanh(inner));
    store(out[idx], (act * u).cast::<T>());
}

/// `out[i] = clipped_swiglu(gate[i], up[i])` — the GPT-OSS variant.
///
/// Both halves are clamped to `[-7, 7]`; the gate uses
/// `sigmoid(1.702·g)`; the up side has a `+1` bias before the
/// multiply: `g·sigmoid(1.702·g)·(u + 1)`. Matches `clipped_swiglu`
/// in `fused_gate_activation.metal`. The clamp is composed from two
/// `select`s (the DSL has no `clamp` builtin).
#[kernel(
    bench(
        op="fused_gate_activation",
        subop="clipped_swiglu",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mt_fused_gate_clipped_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g_raw = load(gate[idx]).cast::<f32>();
    let u_raw = load(up[idx]).cast::<f32>();
    // clamp(x, -7, 7) = min(max(x, -7), 7), composed from select since
    // the DSL exposes no `clamp` builtin (see ffai/rope_yarn.rs).
    let g_hi = select(g_raw > 7.0f32, 7.0f32, g_raw);
    let g = select(g_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, g_hi);
    let u_hi = select(u_raw > 7.0f32, 7.0f32, u_raw);
    let u = select(u_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, u_hi);
    // gate side: g · sigmoid(1.702·g)
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - 1.702f32 * g));
    // up side carries a +1 bias before the multiply.
    let act = g * sig * (u + 1.0f32);
    store(out[idx], act.cast::<T>());
}
