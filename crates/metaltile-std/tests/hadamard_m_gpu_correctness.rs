//! GPU correctness for `mlx::hadamard_m::kernel_ir_for(M, DType)`.
//!
//! `mt_hadamard_m` applies the Hadamard transform H_M to a batch of
//! M-element vectors (M ∈ {12, 20, 28}), then multiplies by `scale`.
//! Built via `Op::InlineMsl` because the `#[kernel]` DSL cannot index a
//! per-thread constant array with a dynamic thread id.
//!
//! ## DISPATCH INVARIANTS (from the kernel module)
//!
//! - **Reduction mode**, `grid = [n_rows, 1, 1]`, `tpg = [M, 1, 1]`.
//! - One threadgroup per M-element vector; `tpg = M` (12, 20, or 28).
//! - Constexpr `scale: f32` passed as 4 LE bytes under key `"scale"`.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::hadamard_m;

// ── Sign-bit tables ────────────────────────────────────────────────────────
// Mirrors the constants in `hadamard_m.rs`. Kept here separately so the
// test oracle is independent of the kernel implementation — if the kernel
// constants diverge, the test catches it.

const H12_SIGNS: [u32; 12] = [4093, 1364, 3127, 1681, 223, 2629, 883, 2329, 3523, 1129, 1807, 421];

const H20_SIGNS: [u32; 20] = [
    445473, 859202, 702596, 389384, 747024, 641086, 234589, 469147, 938263, 828943, 984492, 953176,
    889521, 762211, 508614, 34194, 68357, 135722, 270452, 540873,
];

const H28_SIGNS: [u32; 28] = [
    53043585, 106070914, 210061060, 153783816, 41229328, 80377888, 160739520, 79265980, 156451192,
    44483185, 88966243, 177932359, 87445519, 172810270, 125848794, 251697461, 237056618, 207758549,
    149162411, 31986518, 63972909, 3206502, 4315853, 8631579, 17246902, 34477548, 68954969,
    135812787,
];

/// CPU oracle: apply H_M (via sign-bit table) to each M-element row of `data`,
/// then multiply by `scale`. Promotes to f32 for accumulation.
fn oracle_hadamard_m(data: &[f32], m: usize, signs: &[u32], scale: f32) -> Vec<f32> {
    assert_eq!(data.len() % m, 0, "data.len() must be a multiple of M");
    let n_rows = data.len() / m;
    let mut out = vec![0.0f32; data.len()];
    for row in 0..n_rows {
        let base = row * m;
        for t in 0..m {
            let mut acc = 0.0f32;
            for j in 0..m {
                let sign = if (signs[t] >> j) & 1 == 1 { 1.0f32 } else { -1.0 };
                acc += sign * data[base + j];
            }
            out[base + t] = acc * scale;
        }
    }
    out
}

/// Dispatch `mt_hadamard_m<T>` for one batch of `n_rows` M-element vectors.
///
/// Returns the f32-decoded output (promoted from the dispatch dtype).
fn run_hadamard_m(data: &[f32], dt: Dt, m: u32, scale: f32) -> Vec<f32> {
    let n = data.len();
    assert_eq!(n % m as usize, 0, "n must be a multiple of M");
    let n_rows = n / m as usize;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(data, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
    // scale: f32 constexpr — 4 LE bytes.
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = hadamard_m::kernel_ir_for(m, dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction: one threadgroup per row, tpg = M threads.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_rows, 1, 1], [m as usize, 1, 1])
        .expect("hadamard_m dispatch");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(n);
    out
}

// ── H_12 tests ────────────────────────────────────────────────────────────

#[test]
fn hadamard_m12_matches_oracle_f32() {
    let _g = gpu_lock();
    let m = 12usize;
    let n_rows = 16;
    let scale = 1.0f32 / (m as f32).sqrt(); // normalised Hadamard
    let data: Vec<f32> = (0..n_rows * m).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();
    let expected = oracle_hadamard_m(&data, m, &H12_SIGNS, scale);
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, scale);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "hadamard_m12 f32: max |diff| = {diff:.2e} > 1e-4");
}

#[test]
fn hadamard_m12_matches_oracle_f16() {
    let _g = gpu_lock();
    let m = 12usize;
    let n_rows = 8;
    let scale = 1.0f32;
    // Round through f16 so oracle uses same load precision.
    let data: Vec<f32> =
        (0..n_rows * m).map(|i| Dt::F16.round(((i % 17) as f32 - 8.0) * 0.25)).collect();
    let expected = oracle_hadamard_m(&data, m, &H12_SIGNS, scale);
    let actual = run_hadamard_m(&data, Dt::F16, m as u32, scale);
    let diff = max_abs_diff(&actual, &expected);
    // f16 trig/accumulation: larger tolerance.
    assert!(diff < 5e-2, "hadamard_m12 f16: max |diff| = {diff:.2e} > 5e-2");
}

#[test]
fn hadamard_m12_matches_oracle_bf16() {
    let _g = gpu_lock();
    let m = 12usize;
    let n_rows = 8;
    let scale = 1.0f32;
    // Round through bf16 so oracle uses same load precision.
    let data: Vec<f32> =
        (0..n_rows * m).map(|i| Dt::Bf16.round(((i % 17) as f32 - 8.0) * 0.25)).collect();
    let expected = oracle_hadamard_m(&data, m, &H12_SIGNS, scale);
    let actual = run_hadamard_m(&data, Dt::Bf16, m as u32, scale);
    let diff = max_abs_diff(&actual, &expected);
    // bf16 7-bit mantissa: looser tolerance than f16.
    assert!(diff < 1e-1, "hadamard_m12 bf16: max |diff| = {diff:.2e} > 1e-1");
}

#[test]
fn hadamard_m12_identity_vector_f32() {
    let _g = gpu_lock();
    // H_M applied twice with scale = 1/M gives the identity:
    //   H · (H · x / M) = x  →  (H · x) scaled by 1/M, then H · again gives x.
    // Here we verify the one-way: H_12 · e_0 (standard basis) = row 0 of H_12.
    let m = 12usize;
    let scale = 1.0f32;
    let mut data = vec![0.0f32; m];
    data[0] = 1.0; // e_0
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, scale);
    // H_12 · e_0 = column 0 of H_12^T = row 0 of H_12 (since it's symmetric up to sign)
    // Actually H_12 · e_0 picks out column 0 of H_12, which is the first *column*.
    // The sign of column 0 = H_12[t][0] = (signs[t] >> 0) & 1 → +1 : -1.
    let expected: Vec<f32> =
        (0..m).map(|t| if H12_SIGNS[t] & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "hadamard_m12 e0 f32: max |diff| = {diff:.2e}");
}

#[test]
fn hadamard_m12_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let m = 12u32;
    let data: Vec<f32> = (1..=m as usize).map(|i| i as f32).collect();
    let actual = run_hadamard_m(&data, Dt::F32, m, 1.0);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "hadamard_m12: all-zero output for non-zero input (empty kernel body?)",
    );
}

// ── H_20 tests ────────────────────────────────────────────────────────────

#[test]
fn hadamard_m20_matches_oracle_f32() {
    let _g = gpu_lock();
    let m = 20usize;
    let n_rows = 10;
    let scale = 1.0f32 / (m as f32).sqrt();
    let data: Vec<f32> = (0..n_rows * m).map(|i| ((i % 19) as f32 - 9.0) * 0.3).collect();
    let expected = oracle_hadamard_m(&data, m, &H20_SIGNS, scale);
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, scale);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "hadamard_m20 f32: max |diff| = {diff:.2e} > 1e-4");
}

#[test]
fn hadamard_m20_identity_vector_f32() {
    let _g = gpu_lock();
    let m = 20usize;
    let mut data = vec![0.0f32; m];
    data[0] = 1.0;
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, 1.0);
    // H_20 · e_0 = column 0 = sign of H_20[t][0] for each row t.
    let expected: Vec<f32> =
        (0..m).map(|t| if H20_SIGNS[t] & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "hadamard_m20 e0 f32: max |diff| = {diff:.2e}");
}

#[test]
fn hadamard_m20_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let m = 20u32;
    let data: Vec<f32> = (1..=m as usize).map(|i| i as f32 * 0.5).collect();
    let actual = run_hadamard_m(&data, Dt::F32, m, 1.0);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "hadamard_m20: all-zero output for non-zero input (empty kernel body?)",
    );
}

// ── H_28 tests ────────────────────────────────────────────────────────────

#[test]
fn hadamard_m28_matches_oracle_f32() {
    let _g = gpu_lock();
    let m = 28usize;
    let n_rows = 8;
    let scale = 1.0f32 / (m as f32).sqrt();
    let data: Vec<f32> = (0..n_rows * m).map(|i| ((i % 23) as f32 - 11.0) * 0.2).collect();
    let expected = oracle_hadamard_m(&data, m, &H28_SIGNS, scale);
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, scale);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "hadamard_m28 f32: max |diff| = {diff:.2e} > 1e-4");
}

#[test]
fn hadamard_m28_identity_vector_f32() {
    let _g = gpu_lock();
    let m = 28usize;
    let mut data = vec![0.0f32; m];
    data[0] = 1.0;
    let actual = run_hadamard_m(&data, Dt::F32, m as u32, 1.0);
    // H_28 · e_0 = column 0 = H_28[t][0] for each row t.
    let expected: Vec<f32> =
        (0..m).map(|t| if H28_SIGNS[t] & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "hadamard_m28 e0 f32: max |diff| = {diff:.2e}");
}

#[test]
fn hadamard_m28_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let m = 28u32;
    let data: Vec<f32> = (1..=m as usize).map(|i| i as f32 * 0.1).collect();
    let actual = run_hadamard_m(&data, Dt::F32, m, 1.0);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "hadamard_m28: all-zero output for non-zero input (empty kernel body?)",
    );
}

// ── Perf benchmarks (ignored — run with --ignored --nocapture) ─────────────

#[test]
#[ignore = "perf bench — run with --ignored --nocapture"]
fn hadamard_m12_perf_bench_f32() {
    use std::time::Instant;
    let _g = gpu_lock();
    let m = 12u32;
    let n_rows = 1 << 17; // ~128k rows
    let data: Vec<f32> = (0..n_rows * m as usize).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let ctx = Context::new().expect("Context::new");
    let kernel = {
        let mut k = hadamard_m::kernel_ir_for(m, metaltile_core::dtype::DType::F32);
        k.mode = KernelMode::Reduction;
        k
    };
    let scale = 1.0f32 / (m as f32).sqrt();
    let make_bufs = || {
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("inp".into(), pack_bytes(&data, Dt::F32));
        b.insert("out".into(), pack_bytes(&vec![0.0f32; n_rows * m as usize], Dt::F32));
        b.insert("scale".into(), scale.to_le_bytes().to_vec());
        b
    };
    for _ in 0..3 {
        ctx.dispatch_with_grid(&kernel, &make_bufs(), &BTreeMap::new(), [n_rows, 1, 1], [
            m as usize, 1, 1,
        ])
        .expect("warmup");
    }
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        ctx.dispatch_with_grid(&kernel, &make_bufs(), &BTreeMap::new(), [n_rows, 1, 1], [
            m as usize, 1, 1,
        ])
        .expect("bench");
    }
    let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
    let n_elems = n_rows * m as usize;
    let gb_s = n_elems as f64 * 4.0 * 2.0 / elapsed_us / 1e3;
    println!("hadamard_m12 f32 n_rows={n_rows}: {elapsed_us:.1} µs  |  {gb_s:.1} GB/s");
}
