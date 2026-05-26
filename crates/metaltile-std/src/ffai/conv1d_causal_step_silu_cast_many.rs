//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched depthwise causal conv1d step + SiLU + cast-to-f32 — sweeps T
//! tokens for one channel in ONE dispatch with the K-1 conv state held
//! in thread-local registers across the sweep.
//!
//! Identical math to the per-token pair `conv1d_causal_step` (in
//! `ssm.rs`) + `mt_silu_cast_to_f32` (in `mlx/unary.rs`) looped T times,
//! but collapses T × 2 dispatches into one and keeps the K-1-element
//! rolling state in registers for the duration of the channel's T-sweep
//! instead of round-tripping it through device memory once per token.
//!
//! Bandwidth motivation (Qwen3.6-A3B GDN prefill, T=512, 30 layers,
//! conv_dim=2048, conv_kernel=4):
//!   - per-token kernel reads 3 state rows and writes 3 state rows back
//!     to device memory every t-step: 6 * 2048 * 2 B = 24 KB per token
//!   - 24 KB × T × 30 layers ≈ 360 MB conv-state traffic per prefill
//!   - this kernel reads state once at start of channel sweep, writes
//!     once at end → 24 KB per channel sweep, ~12 MB per prefill (30×
//!     less)
//! On top of the bandwidth save the per-channel kernel collapses
//! `T * (conv_step + silu_cast) = 2T` dispatches per layer into one —
//! same dispatch-saving pattern as `rope_llama_many` and
//! `kv_cache_update_many`.
//!
//! Layout:
//!
//!   src       [T, conv_dim]                T
//!   w         [conv_kernel, conv_dim]      T  (depthwise weights)
//!   b         [conv_dim]                   T  (bias)
//!   state_in  [conv_kernel - 1, conv_dim]  T  (K-1 prior input rows)
//!   out_f32   [T, conv_dim]                f32  (silu(conv) cast f32)
//!   state_out [conv_kernel - 1, conv_dim]  T  (last K-1 rows from src)
//!
//! Grid: `[conv_dim]` threads, one per channel. Each thread owns one
//! channel's complete T-sweep, sequential. Conv state lives in 3 scalar
//! registers (`s0, s1, s2`) — the rolling window of the K-1 most recent
//! inputs. After computing the conv output for time r, the state is
//! shifted: `s0 = s1; s1 = s2; s2 = src[r]`. Final state written back
//! to `state_out` after the sweep terminates.
//!
//! **conv_kernel is fixed at 4 in this kernel.** That covers every
//! production user of `conv1d_causal_step` in the FFAI tree today —
//! Qwen3.5 GDN, Mamba 2, NemotronH, FalconH1, GraniteMoeHybrid, Jamba
//! all set `conv_kernel = 4` (see e.g. `Qwen35.swift:237`,
//! `Mamba2.swift:105`, `NemotronH.swift:182`). A K!=4 model would need
//! a fresh kernel; declaring 3 explicit state scalars is the cleanest
//! way to keep the state purely in registers without DSL-level
//! cooperation between threads. `conv_kernel` is still a constexpr in
//! the signature for documentation + a runtime-side assert hook.
//!
//! Weight convention (inherited verbatim from `conv1d_causal_step`):
//!   y[d] = bias[d]
//!        + w[K-1, d] * x[r, d]                  (current input)
//!        + w[k,   d] * state[k, d]  for k=0..K-2 (K-1 prior inputs)
//! After the y compute the per-channel state shifts: state[k] gets
//! state[k+1] for k<K-2, then state[K-2] = x[r, d]. With K=4 the
//! K-1=3 state slots `(s0, s1, s2)` map to state rows `(0, 1, 2)`.
//!
//! SiLU + cast: `out = (acc * sigmoid(acc)).cast::<f32>()` in fp32.
//! Identical to `mt_silu_cast_to_f32` (`x * sigmoid(x)` with
//! `sigmoid(x) = 1 / (1 + exp(-x))`).
//!
//! `state_in` and `state_out` are decoupled tensors in the kernel
//! signature so the kernel doesn't have to alias-handle a single mutable
//! buffer mid-sweep. Callers are free to pass the same Metal buffer for
//! both (the kernel reads the K-1 state rows once at sweep start before
//! the first store to `state_out`, so the read-then-write aliasing is
//! safe). The runtime decides the residency / barrier story.
//!
//! Codegen-only. Correctness validated against the
//! `conv1d_causal_step` + `mt_silu_cast_to_f32` looped pair in
//! `tests/conv1d_causal_step_silu_cast_many_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="ssm",
    subop="conv1d_causal_step_silu_cast_many",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn ffai_conv1d_causal_step_silu_cast_many<T>(
    src: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    state_in: Tensor<T>,
    mut out_f32: Tensor<f32>,
    mut state_out: Tensor<T>,
    #[constexpr] t_len: u32,
    #[constexpr] conv_dim: u32,
    #[constexpr] conv_kernel: u32,
) {
    let d = program_id::<0>();
    // K=4 hardcoded — see file docstring. `conv_kernel` constexpr is
    // kept in the signature to mirror `conv1d_causal_step`. A non-K=4
    // model wouldn't hit this kernel anyway (the host wrapper gates on
    // kernel_size == 4); the constexpr is unused by the body but keeps
    // the bench spec deterministic.
    let _unused = conv_kernel;
    // Bias and the per-channel weights are constant across the T-sweep
    // → load once, keep in registers. The 4 weights `w0..w3` map to
    // state slots: `w0` weights `s0` (oldest prior input), `w1`→`s1`,
    // `w2`→`s2`, `w3`→`x_r` (current input). This is the K-1==3 form
    // of the generic `conv1d_causal_step` weight convention.
    let b_d = load(b[d]).cast::<f32>();
    let w0 = load(w[0u32 * conv_dim + d]).cast::<f32>();
    let w1 = load(w[1u32 * conv_dim + d]).cast::<f32>();
    let w2 = load(w[2u32 * conv_dim + d]).cast::<f32>();
    let w3 = load(w[3u32 * conv_dim + d]).cast::<f32>();
    // Load the K-1=3 state rows once at sweep start.
    let mut s0 = load(state_in[0u32 * conv_dim + d]).cast::<f32>();
    let mut s1 = load(state_in[1u32 * conv_dim + d]).cast::<f32>();
    let mut s2 = load(state_in[2u32 * conv_dim + d]).cast::<f32>();
    for r in range(0u32, t_len, 1u32) {
        let x_r = load(src[r * conv_dim + d]).cast::<f32>();
        // y = b + w0*s0 + w1*s1 + w2*s2 + w3*x_r — same op order as
        // `conv1d_causal_step` (bias + last-tap first, then state taps
        // in ascending index). Matters for f32 reassociation parity
        // with the per-row oracle.
        let acc = b_d + w3 * x_r + w0 * s0 + w1 * s1 + w2 * s2;
        // SiLU + cast — identical form to `mt_silu_cast_to_f32`:
        // sigmoid via `1 / (1 + exp(-x))`, all in fp32.
        let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - acc));
        let y = acc * sig;
        store(out_f32[r * conv_dim + d], y);
        // Shift state for next step: drop the oldest, slide the rest
        // forward, append x_r at the tail (s2). Three scalar moves,
        // no cross-thread coordination required because the state
        // lives entirely in this thread's registers.
        s0 = s1;
        s1 = s2;
        s2 = x_r;
    }
    // Final state — write the K-1 most recent inputs (last K-1 rows of
    // `src` for this channel) back to device memory. Matches what the
    // single-step kernel leaves in `state` after T iterations.
    store(state_out[0u32 * conv_dim + d], s0.cast::<T>());
    store(state_out[1u32 * conv_dim + d], s1.cast::<T>());
    store(state_out[2u32 * conv_dim + d], s2.cast::<T>());
}
