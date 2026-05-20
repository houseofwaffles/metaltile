//! SwiGLU activation — `silu(gate) * up`.
//!
//! Fused element-wise activation used in every modern transformer MLP
//! (Llama 4, Qwen3 dense + MoE, Gemma, Mistral families): given two
//! equally-sized inputs `gate` and `up` (the two halves of an MLP's
//! `w_gate · x` and `w_up · x` outputs), produce
//!
//! ```text
//!   out[i] = silu(gate[i]) * up[i]
//!         = (gate[i] * sigmoid(gate[i])) * up[i]
//! ```
//!
//! Existing baseline: two separate kernel launches — one applies
//! `silu(gate)` elementwise (`mt_silu` in `unary.rs`), the second
//! multiplies by `up` (`mt_binary` mul). Each load+store cycles the
//! intermediate `silu(gate)` value through device memory.
//!
//! Fusion saves one full-tensor RMW: the intermediate value stays in
//! registers, halving global memory traffic on the activation path.
//! At Qwen3-MoE expert intermediate=768 × prefill 512 tokens =
//! ~400KB per layer per expert; across 48 layers × 8 active experts
//! the saved bandwidth adds up.
//!
//! MLX reference: `mx.fast.swiglu` lives in
//! `mlx/mlx/backend/metal/kernels/fast.metal` as a single launch with
//! `silu(g) * u` in the body. We mirror that pattern.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="swiglu",
    subop="swiglu",
    class=Binary,
    input_a=Signed,
    input_b=Signed,
    tol=1e-3,
)]
#[kernel]
pub fn mt_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). Inlined here
    // because the DSL's `silu()` builtin emits `mt_silu(...)` which
    // requires the `template<typename T> inline T mt_silu` preamble
    // that codegen only emits for kernels routed through certain
    // feature-detection paths. Inlining the math sidesteps the
    // preamble dependency and lets MLX's compiler do the constant
    // folding / fast-math sigmoid pattern matching on its own.
    let s = g / (1.0f32 + exp(0.0f32 - g));
    store(out[idx], (s * u).cast::<T>());
}
