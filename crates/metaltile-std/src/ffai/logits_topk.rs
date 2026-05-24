//! Top-K filter — masking variant.
//!
//! The full top-K filter pipeline is:
//!
//!   1. Find the K-th largest logit value: `threshold = argpartition(logits, -K)`
//!   2. For every logit, if `logit >= threshold` keep it; else set to `-inf`
//!
//! Step 1 is a selection / partial-sort. On GPU at typical serving K (50, 100)
//! and Qwen-scale vocab (152K) the per-call cost is dominated by Metal command-
//! buffer overhead, not arithmetic — a CPU argpartition + threshold-pass is
//! roughly the same wall-clock as a GPU select kernel and one less dispatch.
//! This file ships the GPU mask kernel and leaves threshold computation to
//! the caller. A future PR can add a GPU-side selection kernel when serving
//! batch sizes make a single fused dispatch pull ahead.
//!
//! Caller contract:
//!   - Compute `threshold` = the K-th largest value (descending) on the host.
//!   - Pass it as the constexpr `threshold` parameter.
//!   - Logits below `threshold` are replaced with `-INFINITY` (this is the
//!     standard sentinel — downstream softmax sees `exp(-inf) = 0` and the
//!     filtered tokens contribute zero probability).
//!
//! Generic over T. Grid3D one-thread-per-vocab-position.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Grid3D.** One thread per vocab position.
//! - **Grid: `[ceil(n / TPG), 1, 1]`, TG: `[TPG, 1, 1]`** (TPG = 256 is the
//!   tested geometry; the kernel is pure elementwise so any TPG works).
//! - **`n = grid.x * tg.x`** — caller sizes the dispatch so the total
//!   thread count exactly matches the vocab length. Threads past `n`
//!   would read/write out of bounds; the runtime should not overshoot.
//! - **No `threadgroup_*` / `simd_*` cooperation** — every thread is
//!   independent. The only invariant is the threshold semantic above.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="logits_processors",
    subop="topk_mask",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn logits_topk_mask<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] threshold: f32) {
    let i = program_id::<0>();
    let v = load(inp[i]).cast::<f32>();
    // `select(cond, lhs, rhs)` returns lhs when cond is true.
    // Keep value when v >= threshold; otherwise sentinel to -inf.
    let neg_inf = neg_infinity();
    let masked = select(v >= threshold, v, neg_inf);
    store(out[i], masked.cast::<T>());
}
