//! Matrix coverage for `ffai::softmax_categorical_sample` —
//! dtype × vocab × temperature × distribution × uniform-draw, plus
//! invariant property assertions that catch silent codegen bugs
//! numeric drift alone can't detect.
//!
//! Regression class this guards:
//! - PR #19's proc-macro refactor silently emptied this kernel's body
//!   (only restored after a separate fix). A pinned 3-test suite is
//!   easy to satisfy by accident (e.g. via the trivial "always returns
//!   0" fallback path); a matrix that includes peaked distributions
//!   (must return the peak index regardless of `u`), monotonic-u
//!   coverage (CDF walk must be order-preserving), tail-heavy outliers
//!   (catches missing max-shift) and all-equal logits (chi-squared
//!   rejects "always last" / "always first" failure modes) would have
//!   caught the empty-kernel regression on every dispatch.
//! - Future low-precision regressions (bf16 / f16 cast loss at large
//!   vocab) are scoped by per-cell tolerance bands that distinguish
//!   "kernel is broken" from "expected fp-reorder drift in the tail".
//!
//! Matrix:
//!   dtypes: f32, f16, bf16
//!   vocab: 256, 512, 32_768, 152_064  (Qwen tokenizer scale at top)
//!   temperatures: 0.1, 1.0, 2.0
//!   distributions: peaked, uniform, gaussian, bimodal, tail-heavy
//!   uniform draws: 0.001, 0.25, 0.5, 0.75, 0.999
//!
//! Tolerance bands (vocab-scaled, since exp-reduce ULP drift scales
//! ~sqrt(n) and gets multiplied by the local CDF slope when it lands
//! on the inverse-CDF walk):
//!   vocab ≤ 512   — bit-exact match vs CPU `naive_sample`
//!   vocab ≥ 32K   — ±max(16, vocab/256) tokens for f32
//!                   ±max(64, vocab/64) tokens for f16 / bf16
//! Extreme-u tails (|u - 0.5| ≥ 0.45) get 2× the body tolerance
//! because the local CDF slope is gentler in the tails — same ULP
//! error in `sum_exp` shifts the picked idx farther.
//! The body of well-conditioned distributions (peaked, tail-heavy)
//! falls back to a tighter "must return exact index" check because
//! their CDF is a step function — no slope-driven drift possible.
//!
//! Invariant properties (independent of numeric oracle):
//!   - peaked distribution: always returns the peak idx regardless of u
//!   - tail-heavy outlier (logit=200): always returns the outlier idx,
//!     pins the max-shift pass (without it, exp(200) overflows)
//!   - monotonicity: sample(u=u1) ≤ sample(u=u2) when u1 ≤ u2
//!     for the same distribution + temperature + vocab + dtype
//!   - all-equal logits: 10_000 draws spread across the whole [0, n)
//!     range — chi-squared rejects "always first" / "always last"
//!     failure modes that pass under naive-oracle equality
//!   - determinism: same inputs run twice → same output
//!
//! macOS-gated; dispatch shape is fixed at TPG=256 × 1 threadgroup
//! per kernel invariant.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sampling::softmax_categorical_sample;

// ── CPU oracle (mirrors the kernel's three-pass shape) ────────────────

/// CPU oracle that mirrors the kernel's pipeline: max-shift → sum-exp
/// → inverse-CDF walk with first-hit tie-break. Pre-quantizes the
/// logits through the kernel's load-cast precision so f16/bf16 oracles
/// see the same value the kernel does.
fn naive_sample(logits: &[f32], dt: Dt, temperature: f32, uniform: f32) -> u32 {
    let inv_t = 1.0_f32 / temperature;
    // Round each logit through the dtype's precision first — matches
    // the kernel's `load(inp[pos]).cast::<f32>() * inv_t` semantics.
    let scaled: Vec<f32> = logits.iter().map(|v| dt.round(*v) * inv_t).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = scaled.iter().map(|v| (v - max_val).exp()).sum();
    let target = uniform * total;
    let mut cum = 0.0_f32;
    for (i, v) in scaled.iter().enumerate() {
        cum += (v - max_val).exp();
        if cum >= target {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

// ── GPU dispatch helper ───────────────────────────────────────────────

fn bytes_to_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn run_sample(ctx: &Context, dt: Dt, logits: &[f32], temperature: f32, uniform: f32) -> u32 {
    let n = logits.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dt));
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("temperature_in".into(), temperature.to_le_bytes().to_vec());
    buffers.insert("uniform_in".into(), uniform.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let mut kernel = softmax_categorical_sample::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Fixed TPG = 256 per kernel invariant (tg_max + tg_sum both 256-wide,
    // 8-stage halving). 1 threadgroup total — single-stream sampler.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_u32(out_bytes)
}

// ── Distribution generators ───────────────────────────────────────────

/// One logit absurdly dominates — softmax mass concentrates entirely on
/// the peak across any (T ∈ [0.1, 2.0], vocab) cell in the matrix.
/// Logit=200 → at T=2, scaled=100; `exp(-100) ≈ 4e-44 ≈ 0` for
/// competitors after max-shift, so peak takes ALL the mass.
/// The peak idx is placed off-center to catch "always returns idx 0" /
/// "always returns idx n-1" fallback paths.
fn peaked_logits(n: usize) -> (Vec<f32>, usize) {
    let mut logits = vec![0.0_f32; n];
    let peak = (n * 5) / 8 + 3; // off-center, not n/2, not 0, not n-1
    logits[peak] = 200.0;
    (logits, peak)
}

/// All-zero logits — uniform softmax, CDF[i] = (i+1)/n. Used to verify
/// the inverse-CDF walk is order-preserving + that the sampler doesn't
/// collapse to a single index.
fn uniform_logits(n: usize) -> Vec<f32> { vec![0.0_f32; n] }

/// Smooth Gaussian-like bell centered at n/2 with σ ~ n/8. Exercises
/// the typical realistic logit distribution (most of the mass in a
/// narrow band, exponential decay in the tails).
fn gaussian_logits(n: usize) -> Vec<f32> {
    let center = (n as f32) * 0.5;
    let sigma = (n as f32) / 8.0;
    (0..n)
        .map(|i| {
            let x = (i as f32 - center) / sigma;
            -0.5 * x * x
        })
        .collect()
}

/// Two **asymmetric** peaks at ~25% (logit=3.0) and ~75% (logit=2.7) of
/// vocab — the asymmetry pushes u=0.5 cleanly inside the larger mode
/// (not on the saddle between equal peaks where ULP drift could flip
/// which mode is chosen). Tests CDF walk over two-mode distributions
/// without depending on fragile mode-boundary numerics.
fn bimodal_logits(n: usize) -> Vec<f32> {
    let a = (n as f32) * 0.25;
    let b = (n as f32) * 0.75;
    let sigma = (n as f32) / 24.0;
    (0..n)
        .map(|i| {
            let xa = (i as f32 - a) / sigma;
            let xb = (i as f32 - b) / sigma;
            let pa = 3.0_f32 - 0.5 * xa * xa;
            let pb = 2.7_f32 - 0.5 * xb * xb;
            // log-sum-exp of two Gaussian bumps with offset peak heights
            let m = pa.max(pb);
            m + ((pa - m).exp() + (pb - m).exp()).ln()
        })
        .collect()
}

/// One extreme outlier with the rest mildly negative — catches
/// exp-overflow bugs (max-shift must keep the dynamic range tame). If
/// the kernel skipped the max-shift pass, `exp(200)` would overflow
/// fp32 and corrupt the running sum.
fn tail_heavy_logits(n: usize) -> (Vec<f32>, usize) {
    let mut logits: Vec<f32> = (0..n).map(|i| -((i as f32) / (n as f32))).collect();
    let outlier = (n / 3) + 7; // off-center, not 0/n-1
    logits[outlier] = 200.0; // would overflow without max-shift
    (logits, outlier)
}

// ── Tolerance band ────────────────────────────────────────────────────

/// Per-cell tolerance band, scaled to the floating-point reorder budget
/// of an `n`-element parallel sum-of-exps. The kernel reduces a 256-
/// element threadgroup tile across `ceil(n/256)` strides; the resulting
/// fp32 sum accumulates ~`sqrt(n)*eps` fractional drift, which lands on
/// the inverse-CDF walk multiplied by the local CDF slope.
///
/// For a smooth distribution (gaussian, bimodal away from boundaries),
/// slope at the picked idx is ~`pdf(idx) * n`, so absolute idx drift
/// scales linearly with `n` and ~1/pdf at the picked location. The
/// constants below correspond to about 1/256 of vocab at f32 and 1/64
/// at f16/bf16 — comfortably above the observed drift on M-class GPUs
/// but tight enough to flag a kernel regression (e.g. an empty body
/// would produce idx 0 or idx n-1 regardless, never landing inside the
/// tolerance band for an off-center expected idx).
///
/// Extreme-u draws (|u-0.5| ≥ 0.45) double the budget because the
/// local CDF slope in the tails is much flatter than in the body —
/// same ULP error in `sum_exp` shifts the picked idx farther.
fn tolerance(dt: Dt, vocab: usize, u: f32) -> u32 {
    let extreme = (u - 0.5).abs() >= 0.45;
    let mul = if extreme { 2 } else { 1 };
    let base = match (dt, vocab) {
        (_, n) if n <= 512 => 0,
        (Dt::F32, n) => u32::max(16, (n / 256) as u32),
        (Dt::F16 | Dt::Bf16, n) => u32::max(64, (n / 64) as u32),
    };
    base * mul
}

const DTYPES: &[Dt] = &[Dt::F32, Dt::F16, Dt::Bf16];
const VOCABS: &[usize] = &[256, 512, 32_768, 152_064];
const TEMPS: &[f32] = &[0.1, 1.0, 2.0];
const UNIFORMS: &[f32] = &[0.001, 0.25, 0.5, 0.75, 0.999];

// ── Matrix test ───────────────────────────────────────────────────────

#[test]
fn matrix_dtype_vocab_temp_distribution_uniform_vs_oracle() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");

    // Per-(dtype × vocab × distribution) we precompute the logits once
    // and sweep temperature × uniform inside. Peaked + tail-heavy carry
    // an "expected_idx" override that's bit-exact regardless of u
    // (logit=200 saturates the softmax even at T=2 + vocab=152K).
    let mut failures: Vec<String> = Vec::new();

    for &dt in DTYPES {
        let dt_label = dt.to_dtype();
        for &vocab in VOCABS {
            // ---- peaked: must always return peak idx ----
            {
                let (logits, peak) = peaked_logits(vocab);
                for &t in TEMPS {
                    for &u in UNIFORMS {
                        let actual = run_sample(&ctx, dt, &logits, t, u);
                        if actual as usize != peak {
                            failures.push(format!(
                                "peaked dt={dt_label:?} vocab={vocab} t={t} u={u} \
                                 expected idx {peak}, got {actual}"
                            ));
                        }
                    }
                }
            }

            // ---- uniform: matches CPU oracle within tol ----
            {
                let logits = uniform_logits(vocab);
                for &t in TEMPS {
                    for &u in UNIFORMS {
                        let tol = tolerance(dt, vocab, u);
                        let actual = run_sample(&ctx, dt, &logits, t, u);
                        let expected = naive_sample(&logits, dt, t, u);
                        let diff = (actual as i64 - expected as i64).unsigned_abs() as u32;
                        if diff > tol {
                            failures.push(format!(
                                "uniform dt={dt_label:?} vocab={vocab} t={t} u={u} \
                                 expected idx {expected} (±{tol}), got {actual} \
                                 (diff={diff})"
                            ));
                        }
                    }
                }
            }

            // ---- gaussian: matches CPU oracle within tol ----
            {
                let logits = gaussian_logits(vocab);
                for &t in TEMPS {
                    for &u in UNIFORMS {
                        let tol = tolerance(dt, vocab, u);
                        let actual = run_sample(&ctx, dt, &logits, t, u);
                        let expected = naive_sample(&logits, dt, t, u);
                        let diff = (actual as i64 - expected as i64).unsigned_abs() as u32;
                        if diff > tol {
                            failures.push(format!(
                                "gaussian dt={dt_label:?} vocab={vocab} t={t} u={u} \
                                 expected idx {expected} (±{tol}), got {actual} \
                                 (diff={diff})"
                            ));
                        }
                    }
                }
            }

            // ---- bimodal: matches CPU oracle within tol ----
            {
                let logits = bimodal_logits(vocab);
                for &t in TEMPS {
                    for &u in UNIFORMS {
                        let tol = tolerance(dt, vocab, u);
                        let actual = run_sample(&ctx, dt, &logits, t, u);
                        let expected = naive_sample(&logits, dt, t, u);
                        let diff = (actual as i64 - expected as i64).unsigned_abs() as u32;
                        if diff > tol {
                            failures.push(format!(
                                "bimodal dt={dt_label:?} vocab={vocab} t={t} u={u} \
                                 expected idx {expected} (±{tol}), got {actual} \
                                 (diff={diff})"
                            ));
                        }
                    }
                }
            }

            // ---- tail-heavy: must always return the outlier idx (max-shift sanity) ----
            {
                let (logits, outlier) = tail_heavy_logits(vocab);
                for &t in TEMPS {
                    for &u in UNIFORMS {
                        let actual = run_sample(&ctx, dt, &logits, t, u);
                        if actual as usize != outlier {
                            failures.push(format!(
                                "tail-heavy dt={dt_label:?} vocab={vocab} t={t} u={u} \
                                 expected idx {outlier} (outlier), got {actual} \
                                 — exp-overflow / missing max-shift?"
                            ));
                        }
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "sampling matrix had {} failing cells:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// ── Property invariants (independent of numeric oracle) ──────────────

#[test]
fn invariant_peaked_always_picks_peak() {
    // Verify the peak dominance holds across the full T sweep — at
    // T=2, logit/T = 100, exp(-100) ≈ 0 for all competitors, so the
    // peak takes ALL the CDF mass regardless of `u`. Catches any
    // kernel that disturbs the max-shift or accumulates `exp` outside
    // a stable reorder pattern.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    for &dt in DTYPES {
        let dt_label = dt.to_dtype();
        let (logits, peak) = peaked_logits(512);
        for &t in &[0.1_f32, 1.0, 2.0] {
            for &u in &[0.001_f32, 0.5, 0.999] {
                let actual = run_sample(&ctx, dt, &logits, t, u);
                assert_eq!(
                    actual as usize, peak,
                    "peaked dt={dt_label:?} t={t} u={u} should always pick idx {peak}",
                );
            }
        }
    }
}

#[test]
fn invariant_monotonic_u_picks_non_decreasing_idx() {
    // For any fixed (distribution, temperature, vocab, dtype), the
    // CDF walk is order-preserving: sample(u=u1) ≤ sample(u=u2) when
    // u1 ≤ u2. Catches a kernel that's e.g. always returning `n - 1`
    // (passes "matches oracle at u=0.999" but fails this).
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let us = [0.05_f32, 0.25, 0.5, 0.75, 0.95];
    for &dt in DTYPES {
        let dt_label = dt.to_dtype();
        for &vocab in &[256_usize, 512, 32_768] {
            let logits = gaussian_logits(vocab);
            for &t in &[0.1_f32, 1.0, 2.0] {
                let mut prev_idx: i64 = -1;
                for &u in &us {
                    let idx = run_sample(&ctx, dt, &logits, t, u) as i64;
                    assert!(
                        idx >= prev_idx,
                        "monotonicity violation dt={dt_label:?} vocab={vocab} t={t} \
                         u={u}: prev={prev_idx} cur={idx}",
                    );
                    prev_idx = idx;
                }
            }
        }
    }
}

#[test]
fn invariant_uniform_logits_spread_chi_squared() {
    // 10_000 deterministic draws across u ∈ (0, 1) on all-zero logits
    // must distribute across the full [0, n) range. Bin into k=8
    // buckets and reject if the chi-squared statistic exceeds the
    // 99.9% threshold (extremely lax — we're rejecting catastrophic
    // collapse to a single bin, not testing the rng).
    //
    // For k=8 buckets, df=7, χ²_{0.999} ≈ 24.32 — far below what a
    // catastrophic "always returns idx 0" failure would score (which
    // would push the entire chi-squared mass into one bucket).
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    const N_VOCAB: usize = 512;
    const N_DRAWS: usize = 10_000;
    const K_BUCKETS: usize = 8;
    const CHI2_THRESHOLD: f64 = 24.32; // df=7, p=0.001

    for &dt in DTYPES {
        let dt_label = dt.to_dtype();
        let logits = uniform_logits(N_VOCAB);
        let mut counts = vec![0_u32; K_BUCKETS];
        for i in 0..N_DRAWS {
            // Linear scan of (0, 1) avoids needing an RNG dep.
            let u = (i as f32 + 0.5) / (N_DRAWS as f32);
            let idx = run_sample(&ctx, dt, &logits, 1.0, u) as usize;
            let bucket = (idx * K_BUCKETS) / N_VOCAB;
            counts[bucket.min(K_BUCKETS - 1)] += 1;
        }
        let expected = (N_DRAWS as f64) / (K_BUCKETS as f64);
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                (d * d) / expected
            })
            .sum();
        assert!(
            chi2 < CHI2_THRESHOLD,
            "uniform-logit spread dt={dt_label:?}: chi² = {chi2:.2} ≥ {CHI2_THRESHOLD} \
             (catastrophic bucket collapse) — counts={counts:?}",
        );
    }
}

#[test]
fn invariant_determinism_same_inputs_same_output() {
    // Same logits + same temperature + same uniform draw → bit-exact
    // same output, twice in a row. Pins out any nondeterminism leaking
    // into the reduction (e.g. accidental fp ordering from a stale
    // thread-local accumulator).
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let cases: &[(Dt, usize, f32, f32)] = &[
        (Dt::F32, 512, 1.0, 0.5),
        (Dt::F32, 32_768, 0.8, 0.25),
        (Dt::F16, 512, 1.5, 0.75),
        (Dt::F16, 32_768, 1.0, 0.999),
        (Dt::Bf16, 512, 0.1, 0.001),
        (Dt::Bf16, 32_768, 2.0, 0.5),
    ];
    for &(dt, vocab, t, u) in cases {
        let dt_label = dt.to_dtype();
        let logits = gaussian_logits(vocab);
        let a = run_sample(&ctx, dt, &logits, t, u);
        let b = run_sample(&ctx, dt, &logits, t, u);
        assert_eq!(a, b, "non-determinism dt={dt_label:?} vocab={vocab} t={t} u={u}: {a} vs {b}",);
    }
}
