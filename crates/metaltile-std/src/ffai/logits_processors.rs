//! Logits-processor kernels for the sampling pipeline.
//!
//! Decode-form samplers (other than the bare-softmax fused
//! `softmax_categorical_sample`) compose a small set of in-place
//! transforms on the logits vector before the final categorical
//! draw. Pipeline shape: temperature → repetition penalty → top-k →
//! top-p (nucleus) → categorical sample. This file ships kernels 1
//! and 2 (temperature, repetition penalty); top-k / top-p require a
//! sort or quickselect pass and live in a follow-up.
//!
//! Semantic contracts:
//!
//!   - **temperature**: `logits[i] /= temperature` (no-op at 1.0;
//!     small T sharpens toward argmax). Caller clamps to a positive
//!     floor before dispatch.
//!   - **repetition penalty**: for each token id in `token_ids`,
//!     `v > 0 → v /= penalty`, `v ≤ 0 → v *= penalty`. Matches the
//!     HuggingFace `transformers.LogitsProcessorList` and vLLM
//!     conventions. `penalty == 1.0` is a no-op.
//!
//! Top-k and top-p require a sort or quickselect pass — they live
//! in a follow-up kernel since the sort dispatch geometry doesn't
//! fit the simple one-thread-per-element shape these two use.
//!
//! Generic over T; all values are upcast to f32 internally so f16/bf16
//! logits accumulate cleanly across the scale and don't drift on the
//! repeated-token gather. Output dtype matches input dtype.

use metaltile::{bench_kernel, kernel};

// ── Temperature scaling ───────────────────────────────────────────────────
//
// Pure elementwise `logits[i] /= temperature`. Generic-T, one thread per
// vocab position. At `temperature == 1.0` this is a copy; at very small
// temperature it sharpens the distribution toward greedy argmax (the
// downstream `softmax_categorical_sample` handles the softmax itself).
//
// Caller contract: `temperature > 0`. A zero or negative temperature
// produces inf / sign-flipped logits — callers should clamp before
// dispatch (`max(temperature, 1e-5)` is the standard guard).
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D.** One thread per vocab position.
// - **Grid: `[ceil(n / TPG), 1, 1]`, TG: `[TPG, 1, 1]`** (TPG = 256 is the
//   tested geometry; any value works since the kernel is pure elementwise
//   and uses no `threadgroup_*` / `simd_*` cooperation).
// - **`n = grid.x * tg.x`** — the caller is responsible for `n` covering
//   the full logits length. Threads with `program_id::<0>() >= n` would
//   read/write out of bounds; the runtime should size the dispatch so the
//   total thread count exactly matches the logits length.
#[bench_kernel(
    op="logits_processors",
    subop="temperature",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn logits_temperature<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] temperature: f32) {
    let i = program_id::<0>();
    let inv_t = 1.0f32 / temperature;
    let v = load(inp[i]).cast::<f32>();
    store(out[i], (v * inv_t).cast::<T>());
}

// ── Repetition penalty ────────────────────────────────────────────────────
//
// In-place mutate the logits at every position appearing in `token_ids`,
// scaling toward 0 to discourage repeats. Convention matches HuggingFace
// `transformers.LogitsProcessorList`:
//
//   for tok in token_ids:
//       if logits[tok] > 0: logits[tok] /= penalty
//       else:               logits[tok] *= penalty
//
// `penalty == 1.0` is a no-op; `penalty > 1.0` discourages repeats;
// `penalty < 1.0` encourages repeats (rare).
//
// Dispatch: one thread per `token_ids` entry. The kernel reads
// `logits[token_ids[i]]`, updates, and writes back. With duplicate
// token ids the operation is **idempotent in expectation but
// non-deterministic in order** — multiple threads racing on the same
// vocab slot pick a write order. Callers MUST dedupe `token_ids` before
// dispatch (or accept the last-writer-wins semantics, which matches
// what a sequential CPU pass produces *only* on a deduped input).
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D.** One thread per `token_ids` entry.
// - **Grid / TG: `grid.x * tg.x == token_ids.len()`** — caller must size
//   the dispatch to exactly the token-id count. TPG = 256 (or smaller for
//   small contexts) is the tested geometry.
// - **No `threadgroup_*` / `simd_*` cooperation** — every thread is
//   independent. The only invariant is the dedupe contract above.
#[bench_kernel(
    op="logits_processors",
    subop="repetition_penalty",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn logits_repetition_penalty<T>(
    mut logits: Tensor<T>,
    token_ids: Tensor<u32>,
    #[constexpr] penalty: f32,
) {
    let i = program_id::<0>();
    let tok = load(token_ids[i]);
    let v = load(logits[tok]).cast::<f32>();
    let scaled = select(v > 0.0f32, v / penalty, v * penalty);
    store(logits[tok], scaled.cast::<T>());
}
