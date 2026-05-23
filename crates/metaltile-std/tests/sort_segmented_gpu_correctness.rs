//! GPU correctness for `mlx::sort::mt_sort_segmented` — per-row (segmented)
//! bitonic sort over a `[batch, n]` matrix.
//!
//! Oracle: Rust `sort_unstable_by(f32::total_cmp)` applied independently
//! to each row of the input. The GPU output must match the CPU oracle
//! exactly (exact-permutation requirement: the sort is a rearrangement of
//! the input values, so bit-level equality is expected for f32 inputs that
//! are exactly representable after the dtype round-trip).
//!
//! Covers:
//!   - Various n values: 64, 256, 512, 1024 (full block).
//!   - Multiple batch rows to verify per-row independence.
//!   - Reverse-sorted input (worst case for many sort algorithms).
//!   - f32, f16, bf16.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::sort::mt_sort_segmented;

/// Dispatch `mt_sort_segmented` over a `[batch, n]` matrix.
///
/// TPG = 256 (each thread handles 4 elements → 1024 capacity/row).
/// Grid = `[batch, 1, 1]` — one threadgroup per row.
fn run_sort_segmented(inp: &[f32], batch: usize, n: usize, dt: Dt) -> Vec<f32> {
    assert_eq!(inp.len(), batch * n);
    assert!(n <= 1024, "mt_sort_segmented only supports n ≤ 1024");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; inp.len()], dt));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_sort_segmented::kernel_ir_for(dt.to_dtype());
    // Reduction mode: one threadgroup per row; tgid_x = row index.
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [batch, 1, 1], [256, 1, 1])
        .expect("mt_sort_segmented dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(batch * n);
    out
}

/// CPU oracle: sort each row independently in ascending order.
fn cpu_sort_segmented(inp: &[f32], batch: usize, n: usize) -> Vec<f32> {
    let mut out = inp.to_vec();
    for r in 0..batch {
        out[r * n..(r + 1) * n].sort_unstable_by(f32::total_cmp);
    }
    out
}

#[test]
fn sort_segmented_n1024_single_row_reverse_f32() {
    let _g = gpu_lock();
    let n = 1024;
    let batch = 1;
    let inp: Vec<f32> = (0..n).rev().map(|i| i as f32 * 0.1).collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::F32);
    assert!(actual.iter().any(|&v| v != 0.0), "output all zeros — empty kernel body?");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-6,
            "sort_segmented n1024 mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

#[test]
fn sort_segmented_n512_two_rows_f32() {
    let _g = gpu_lock();
    let n = 512;
    let batch = 2;
    // Two rows with different patterns to verify per-row independence.
    let row0: Vec<f32> = (0..n).rev().map(|i| i as f32).collect();
    let row1: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 1000) as f32 * 0.01).collect();
    let inp: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::F32);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-6,
            "sort_segmented n512 2rows mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

#[test]
fn sort_segmented_n256_four_rows_f32() {
    let _g = gpu_lock();
    let n = 256;
    let batch = 4;
    let inp: Vec<f32> = (0..batch * n).map(|i| ((i * 37 + 11) % 500) as f32 * 0.01 - 2.5).collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::F32);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-6,
            "sort_segmented n256 4rows mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

#[test]
fn sort_segmented_n64_f32() {
    let _g = gpu_lock();
    let n = 64;
    let batch = 8;
    let inp: Vec<f32> = (0..batch * n).rev().map(|i| i as f32 * 0.25).collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::F32);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-6,
            "sort_segmented n64 mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

#[test]
fn sort_segmented_output_is_non_decreasing_per_row_f32() {
    let _g = gpu_lock();
    let n = 512;
    let batch = 3;
    let inp: Vec<f32> = (0..batch * n).map(|i| ((i * 97 + 31) % 200) as f32 - 100.0).collect();
    let actual = run_sort_segmented(&inp, batch, n, Dt::F32);
    for r in 0..batch {
        let row = &actual[r * n..(r + 1) * n];
        for w in row.windows(2) {
            assert!(w[0] <= w[1], "sort_segmented row {r} not non-decreasing at {:?}", w);
        }
    }
}

#[test]
fn sort_segmented_n1024_f16() {
    let _g = gpu_lock();
    let n = 1024;
    let batch = 2;
    // Values exactly representable in f16 to avoid rounding confusion.
    let inp: Vec<f32> =
        (0..batch * n).map(|i| Dt::F16.round(((batch * n - 1 - i) as f32) * 0.25)).collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::F16);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        // f16 has 10-bit mantissa: 1e-2 tolerance for values ~0-256.
        assert!(
            (e - a).abs() < 1e-2,
            "sort_segmented f16 mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

#[test]
fn sort_segmented_n512_bf16() {
    let _g = gpu_lock();
    let n = 512;
    let batch = 2;
    // bf16 has 7-bit mantissa: use values with small representational error.
    let inp: Vec<f32> =
        (0..batch * n).map(|i| Dt::Bf16.round(((batch * n - 1 - i) as f32) * 0.5)).collect();
    let expected = cpu_sort_segmented(&inp, batch, n);
    let actual = run_sort_segmented(&inp, batch, n, Dt::Bf16);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 5e-2,
            "sort_segmented bf16 mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}
