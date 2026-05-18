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

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// ── threadgroup argmax tree-reduction helper ─────────────────────────────
//
// Each step of the binary tree merges two halves of the threadgroup
// arrays `tg_vals` / `tg_idxs`.  The combine rule: take the higher
// value; on ties take the smaller index (matches NumPy argmax semantics).
//
// Used for strides 128 → 2 (7 invocations).  The final stride-1 step
// writes directly to `out[0]` and is kept inline below.

#[allow(unused_macros)]
macro_rules! argmax_step {
    ($lid:expr, $stride:expr) => {
        if $lid < $stride {
            let ov = threadgroup_load("tg_vals", $lid + $stride);
            let oi = threadgroup_load("tg_idxs", $lid + $stride);
            let tv = threadgroup_load("tg_vals", $lid);
            let ti = threadgroup_load("tg_idxs", $lid);
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", $lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", $lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    };
}

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

    argmax_step!(lid, 128u32);
    argmax_step!(lid, 64u32);
    argmax_step!(lid, 32u32);
    argmax_step!(lid, 16u32);
    argmax_step!(lid, 8u32);
    argmax_step!(lid, 4u32);
    argmax_step!(lid, 2u32);

    // Final step writes result directly to output.
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

inventory::submit! {
    BenchSpec {
        op: "arg_reduce",
        // Distinct subop so it sorts alongside `mt_argmax_f32` (subop
        // "argmax" in mlx/arg_reduce.rs) but doesn't collide with it
        // in the bench table.
        subop: "argmax_u32",
        kernel_name: "ffai_argmax",
        kernel_ir: ffai_argmax::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
