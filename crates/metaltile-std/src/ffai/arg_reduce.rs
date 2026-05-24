//! Generic `argmax<T>` with u32 index output — FFAI's decode-form
//! greedy-sampler workhorse.
//!
//! Adapted from `mt_argmax_f32` (in `mlx/arg_reduce.rs`) but generic
//! over input dtype and emitting a `u32` index rather than a float-cast
//! version. Decode-form samplers (greedy token pick) need an integer
//! token id; the f32-output upstream variant doesn't fit that contract.
//!
//! Tie-breaking: strict `>` on values, smallest index on ties — matches
//! NumPy / PyTorch / MLX `argmax` semantics.
//!
//! Codegen-only — there's no MLX argmax template with the same
//! u32-output signature. Correctness validated in FFAI integration
//! tests against reference decoder output.

use metaltile::{bench_kernel, kernel};

// Tree-reduction strides: 128 → 64 → 32 → 16 → 8 → 4 → 2.
// Each iteration: threads with `lid < stride` merge the upper half into
// the lower half (take higher value; on ties take smaller index — NumPy
// argmax semantics).  Final stride-1 merge writes the result directly
// to `out[0]` and is kept inline below.
//
// Originally hand-unrolled via a `macro_rules! argmax_step!` invoked
// 7×; the proc-macro does not expand inner declarative macros, so the
// expansion silently produced no IR.  A DSL `for` loop over the seven
// stages yields identical MSL and survives the proc-macro intact.

#[bench_kernel(
    op="arg_reduce",
    subop="argmax_u32",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_argmax<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
    let lid = tid;
    let mut best_val = neg_infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>();
            let better = v > best_val;
            if better {
                best_val = v;
                best_idx = pos;
            }
        }
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();

    // 7-stage power-of-two halving reduction over the 256-thread group.
    for _stage in range(0u32, 7u32, 1u32) {
        let stride = 128u32 >> _stage;
        if lid < stride {
            let ov = threadgroup_load("tg_vals", lid + stride);
            let oi = threadgroup_load("tg_idxs", lid + stride);
            let tv = threadgroup_load("tg_vals", lid);
            let ti = threadgroup_load("tg_idxs", lid);
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }

    // Final stride-1 merge writes result directly to output.
    if lid == 0u32 {
        let ov = threadgroup_load("tg_vals", 1u32);
        let oi = threadgroup_load("tg_idxs", 1u32);
        let tv = threadgroup_load("tg_vals", 0u32);
        let ti = threadgroup_load("tg_idxs", 0u32);
        let bet = (ov > tv) | ((ov == tv) & (oi < ti));
        let final_idx = select(bet, oi, ti);
        store(out[0], final_idx);
    }
}
