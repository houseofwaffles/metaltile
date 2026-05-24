//! Min-p (minimum-probability) logits filter for the sampling pipeline.
//!
//! Min-p sampling keeps every token whose probability is at least
//! `min_p` times the probability of the most-likely token, and masks
//! the rest:
//!
//!   keep token i  ⇔  P(i) ≥ min_p · P_max
//!
//! Working in logit space avoids a full softmax. For any shift `C`,
//! `P(i) / P_max = exp(logit_i − logit_max)`, so the keep test is
//! simply `exp(logit_i − logit_max) ≥ min_p`. The kernel finds the
//! row max with one threadgroup reduction, then masks every logit
//! below the cutoff to `-INFINITY` in a second pass. Downstream
//! `softmax_categorical_sample` sees `exp(-inf) = 0`, so masked tokens
//! contribute zero probability.
//!
//! This is the reduction-mode sibling of `logits_topk_mask`: top-K
//! needs a host-computed K-th-largest threshold, but min-p's cutoff is
//! defined purely by the row max, so the whole filter fits in one
//! self-contained GPU kernel — no host round-trip, no sort.
//!
//! Reduction-mode, generic over T; the max and the ratio are computed
//! in f32 so f16/bf16 logits don't drift. One threadgroup per row;
//! `n` is the vocab length, looped so any `n` works at any
//! (multiple-of-32) threadgroup size.
//!
//! Caller contract: `0 < min_p < 1`. As `min_p → 0` nothing is masked;
//! as `min_p → 1` only the argmax (and exact ties) survive. A typical
//! serving value is 0.05–0.1.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="logits_processors",
    subop="min_p_mask",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn logits_min_p_mask<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] min_p: f32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;

    // Pass 1: threadgroup-wide max of the row's logits.
    let mut lm = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        lm = max(lm, load(inp[_i]).cast::<f32>());
    }
    let row_max = reduce_max(lm);

    // Pass 2: keep a logit iff exp(logit - row_max) >= min_p, else -inf.
    let neg_inf = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        store(out[_i], select(exp(v - row_max) >= min_p, v, neg_inf).cast::<T>());
    }
}
