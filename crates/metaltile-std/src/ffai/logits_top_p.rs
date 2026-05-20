//! Top-p (nucleus) logits filter for the sampling pipeline.
//!
//! Top-p sampling keeps the smallest set of most-likely tokens whose
//! cumulative probability reaches `top_p`, and masks the rest. The
//! reference definition sorts the probabilities descending and walks
//! the prefix until the running sum clears `top_p`. Equivalently — and
//! without a sort — there is a probability cutoff `c` such that the
//! kept set is exactly `{ i : P(i) ≥ c }`, and that set's mass is the
//! smallest that reaches `top_p`. This kernel finds `c` directly.
//!
//! Working in logit space avoids a full softmax: for any shift, the
//! unnormalised weight of token `i` is `w_i = exp(logit_i − logit_max)`
//! and `Z = Σ w_i`. The keep test `P(i) ≥ c` becomes `w_i ≥ c·Z`, so
//! the cutoff search runs entirely on `w ∈ (0, 1]`.
//!
//! `w` is not sorted, so `c` is found by **bisection**: the kept mass
//! `S(t) = Σ_{w_i ≥ t} w_i` is monotonically non-increasing in `t`, so
//! a binary search on `t ∈ [0, 1]` converges on the threshold where
//! `S(t)` just reaches `top_p·Z`. 24 halvings pin `t` to a `2⁻²⁴`
//! (≈ 6e-8) interval — far finer than the gap between adjacent token
//! weights near any realistic cutoff. A final pass masks every logit
//! whose weight is below the converged floor to `-INFINITY`, so the
//! downstream `softmax_categorical_sample` sees `exp(-inf) = 0`.
//!
//! This is the iterative-search sibling of `logits_min_p_mask`: min-p's
//! cutoff is a closed form of the row max (one reduction), but top-p's
//! cutoff depends on the whole mass profile, so it costs one reduction
//! per bisection step. The whole filter is still a single self-contained
//! GPU kernel — one threadgroup per row, no host round-trip, no sort.
//!
//! Reduction-mode, generic over T; the max, the partition function and
//! every kept-mass sum are computed in f32 so f16/bf16 logits don't
//! drift. One threadgroup per row; `n` is the vocab length, looped so
//! any `n` works at any (multiple-of-32) threadgroup size.
//!
//! Caller contract: `0 < top_p < 1`. As `top_p → 0` only the argmax
//! survives; as `top_p → 1` nothing is masked. A typical serving value
//! is 0.9–0.95.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn logits_top_p_mask<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] top_p: f32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;

    // Pass 1: one streaming pass for both the row max and the partition
    // function Z = Σ exp(logit − row_max). Each lane keeps a running
    // (max, sum) pair in online-softmax form; the pair is then merged
    // across the threadgroup. This mirrors `mt_softmax`'s looped path.
    let mut lm = neg_infinity();
    let mut ls = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        let nm = max(lm, v);
        ls = ls * exp(lm - nm) + exp(v - nm);
        lm = nm;
    }
    let row_max = reduce_max(lm);
    // Rescale every lane's partial sum to the common max before reducing.
    let z = reduce_sum(ls * exp(lm - row_max));

    // Bisection: find the largest weight threshold `t` whose kept mass
    // S(t) = Σ_{w_i ≥ t} w_i still reaches `target`. `lo` is the highest
    // threshold known to keep enough mass, `hi` the lowest known to keep
    // too little; the kept set shrinks as the threshold rises.
    // 24 halvings of `t ∈ [0, 1]` pin the cutoff to a ≈ 6e-8 interval;
    // each step costs one threadgroup reduction over the row.
    let target = top_p * z;
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _k in range(0u32, 24u32, 1u32) {
        let mid = (lo + hi) * 0.5f32;
        let mut partial = 0.0f32;
        for _i in range(rs + tid, re, lsize) {
            let w = exp(load(inp[_i]).cast::<f32>() - row_max);
            partial = partial + select(w >= mid, w, 0.0f32);
        }
        let kept_mass = reduce_sum(partial);
        // S is non-increasing in the threshold: if `mid` still keeps
        // enough mass we can raise the floor, otherwise we must lower it.
        let enough = kept_mass >= target;
        lo = select(enough, mid, lo);
        hi = select(enough, hi, mid);
    }

    // Pass 2: keep a logit iff its weight clears the converged floor
    // `lo`, else -inf. `lo` starts at 0, so a token whose weight equals
    // the floor is kept — the argmax (weight 1) therefore always survives.
    let neg_inf = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        store(out[_i], select(exp(v - row_max) >= lo, v, neg_inf).cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "logits_processors",
        subop: "top_p_mask",
        kernel_name: "logits_top_p_mask",
        kernel_ir: logits_top_p_mask::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
