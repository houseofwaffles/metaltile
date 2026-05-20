//! GPU correctness coverage for `logits_top_p_mask`.
//!
//! Top-p (nucleus) sampling keeps the smallest set of most-likely
//! tokens whose cumulative probability reaches `top_p` and masks the
//! rest. The kernel finds the probability cutoff without a sort: it
//! bisects a weight threshold `t ∈ [0, 1]` until the kept mass
//! `Σ_{w_i ≥ t} w_i` converges on `top_p·Z`, then masks every logit
//! below the converged floor to `-INFINITY`.
//!
//! This file pins the kernel two ways:
//!
//! - **Against a CPU oracle** that replays the identical bisection, so
//!   the GPU kernel is verified to implement the intended algorithm.
//!   Inputs are well-separated ramps, so the converged cutoff lands in
//!   a wide gap between token-weight levels — robust to the ULP-level
//!   drift between GPU and CPU reduction orders.
//! - **Against the top-p invariants** the algorithm must satisfy
//!   regardless of arithmetic: the kept set is a downward-closed
//!   nucleus (every kept logit ranks above every masked one) and its
//!   mass reaches `top_p·Z`. Plus the limits — `top_p → 0` keeps only
//!   the argmax, `top_p → 1` keeps every token.
//!
//! f32 / f16 / bf16: the max, the partition function and every
//! kept-mass sum run in f32 regardless of `T`; a kept logit is stored
//! bit-identical to the input, so the compare is exact.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::ffai::logits_top_p::logits_top_p_mask;

/// Threadgroup size for every dispatch here. 256 is a multiple of 32
/// (no sub-simdgroup `reduce_*` hazard) and the kernel's strided
/// `range(rs + tid, re, lsize)` loops cover any vocab at this tpg.
const TPG: usize = 256;

/// Bisection halvings — must match the kernel's loop bound so the
/// oracle converges on the same cutoff.
const BISECT_ITERS: usize = 24;

/// CPU oracle: replays the kernel's sort-free bisection. `logits` is
/// already rounded through the dtype, so a kept logit is returned
/// verbatim — exactly what the kernel stores.
fn cpu_top_p_mask(logits: &[f32], n: usize, rows: usize, top_p: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let base = r * n;
        let row = &logits[base..base + n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let z: f32 = w.iter().sum();
        let target = top_p * z;

        // Binary-search the weight floor: `lo` keeps enough mass,
        // `hi` keeps too little. Kept mass is non-increasing in the
        // threshold, so raise `lo` while the floor still clears target.
        let mut lo = 0.0f32;
        let mut hi = 1.0f32;
        for _ in 0..BISECT_ITERS {
            let mid = (lo + hi) * 0.5;
            let kept: f32 = w.iter().filter(|&&x| x >= mid).sum();
            if kept >= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        for (i, &wi) in w.iter().enumerate() {
            out[base + i] = if wi >= lo { row[i] } else { f32::NEG_INFINITY };
        }
    }
    out
}

/// Dispatch `logits_top_p_mask` over `rows × n` logits at `dtype`.
/// `n` and `top_p` are `#[constexpr]` kernel args; like every other
/// constexpr in the suite they're handed in through the buffer map.
fn run_top_p_mask(logits: &[f32], n: usize, rows: usize, dtype: Dt, top_p: f32) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dtype));
    buffers.insert("out".into(), vec![0u8; rows * n * dtype.bytes()]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("top_p".into(), top_p.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = logits_top_p_mask::kernel_ir_for(dtype.to_dtype());
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [TPG, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dtype)
}

/// Run the kernel and the oracle on the same logits and assert they
/// agree element-for-element. Masked positions must be `-inf` on both
/// sides; kept positions must match bit-exactly.
fn check_oracle(logits: &[f32], n: usize, rows: usize, dtype: Dt, top_p: f32) {
    let _g = gpu_lock();

    let rounded: Vec<f32> = logits.iter().map(|&v| dtype.round(v)).collect();
    let expected = cpu_top_p_mask(&rounded, n, rows, top_p);
    let actual = run_top_p_mask(&rounded, n, rows, dtype, top_p);

    assert_eq!(actual.len(), expected.len(), "output element count");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if *e == f32::NEG_INFINITY {
            assert_eq!(
                *a,
                f32::NEG_INFINITY,
                "idx {i}: oracle masked this token but kernel kept {a} \
                 (n={n} rows={rows} dtype={:?} top_p={top_p})",
                dtype.to_dtype(),
            );
        } else {
            assert_eq!(
                a,
                e,
                "idx {i}: kept-token mismatch (n={n} rows={rows} \
                 dtype={:?} top_p={top_p})",
                dtype.to_dtype(),
            );
        }
    }
}

/// Wide-spread logit ramp: `(i % 53)` stepped by 0.2. Adjacent
/// distinct logits map to token weights a factor `exp(0.2) ≈ 1.22`
/// apart, so the converged cutoff lands in a gap far wider than any
/// GPU-vs-CPU reduction-order ULP — the oracle compare stays stable.
fn ramp(n: usize, rows: usize) -> Vec<f32> {
    (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect()
}

// ── oracle-replay coverage ───────────────────────────────────────────

#[test]
fn top_p_mid_range_matches_oracle_f32() {
    // top_p = 0.9 over a wide ramp: a genuine nucleus that is neither
    // the whole row nor a single token.
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check_oracle(&logits, n, rows, Dt::F32, 0.9);

    // The partition must be non-trivial, else the test proves nothing.
    let out = run_top_p_mask(&logits, n, rows, Dt::F32, 0.9);
    assert!(out.iter().any(|v| v.is_finite()), "expected some kept tokens");
    assert!(out.iter().any(|v| !v.is_finite()), "expected some masked tokens");
}

#[test]
fn top_p_mid_range_matches_oracle_f16() {
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check_oracle(&logits, n, rows, Dt::F16, 0.9);
}

#[test]
fn top_p_mid_range_matches_oracle_bf16() {
    // bf16's 7-bit mantissa quantises the ramp coarsely, but the
    // bisection runs in f32 and a kept logit is stored verbatim, so
    // the compare is still exact once the input is pre-rounded.
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check_oracle(&logits, n, rows, Dt::Bf16, 0.9);
}

#[test]
fn top_p_qwen3_vocab_stress_f32() {
    // Qwen3 vocab = 152 064. One threadgroup per row at tpg=256: each
    // bisection step strides the full row, so this exercises the
    // looped reduction 24× over plus the mask pass at production scale.
    let (n, rows) = (152_064, 2);
    let logits = ramp(n, rows);
    check_oracle(&logits, n, rows, Dt::F32, 0.9);
}

// ── top-p invariants ─────────────────────────────────────────────────

#[test]
fn top_p_kept_set_is_a_downward_closed_nucleus() {
    // The defining property of top-p: the kept set is the highest-
    // probability slice. Because token weight is monotone in the
    // logit, every kept logit must rank at or above every masked one,
    // and the kept mass must reach top_p·Z.
    let _g = gpu_lock();
    let (n, rows, top_p) = (512, 3, 0.85);
    let logits = ramp(n, rows);
    let out = run_top_p_mask(&logits, n, rows, Dt::F32, top_p);

    for r in 0..rows {
        let base = r * n;
        let row = &logits[base..base + n];
        let mask = &out[base..base + n];

        let min_kept =
            (0..n).filter(|&i| mask[i].is_finite()).map(|i| row[i]).fold(f32::INFINITY, f32::min);
        let max_masked = (0..n)
            .filter(|&i| !mask[i].is_finite())
            .map(|i| row[i])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_kept >= max_masked,
            "row {r}: kept set is not downward-closed (min kept logit {min_kept} \
             < max masked logit {max_masked})",
        );

        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|&v| (v - m).exp()).sum();
        let kept_mass: f32 =
            (0..n).filter(|&i| mask[i].is_finite()).map(|i| (row[i] - m).exp()).sum();
        assert!(
            kept_mass >= top_p * z - 1e-3 * z,
            "row {r}: kept mass {kept_mass} below top_p·Z {}",
            top_p * z,
        );
    }
}

#[test]
fn top_p_near_zero_keeps_only_argmax() {
    // Strictly increasing logits ⇒ a unique argmax at the last index.
    // top_p = 0.01: the argmax alone already clears the tiny target,
    // so the nucleus collapses to exactly that one token.
    let _g = gpu_lock();
    let n = 64;
    let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let out = run_top_p_mask(&logits, n, 1, Dt::F32, 0.01);

    let kept: Vec<usize> =
        out.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
    assert_eq!(kept, vec![n - 1], "top_p→0 must keep only the argmax");
}

#[test]
fn top_p_near_one_keeps_every_token() {
    // Near-uniform logits ⇒ every token holds ≈ 1/n of the mass, so
    // dropping even the smallest token falls short of top_p = 0.995
    // (1 − 1/64 ≈ 0.984 < 0.995). The nucleus is the whole row.
    let _g = gpu_lock();
    let n = 64;
    let logits: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.001).collect();
    let out = run_top_p_mask(&logits, n, 1, Dt::F32, 0.995);

    assert!(out.iter().all(|v| v.is_finite()), "top_p→1 must keep every token");
}
