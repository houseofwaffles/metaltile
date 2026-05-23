//! GPU correctness for the multi-op scan variants: prod, max, min
//! (inclusive and exclusive) from `mlx::scan`.
//!
//! Oracle: sequential CPU prefix scan for each operation and variant.
//! The kernels use a two-level (per-TG-thread + cross-thread-sequential)
//! prefix-scan via a `tgs` buffer — this test verifies the result matches
//! the trivial sequential implementation for both chunk-aligned and ragged
//! row lengths.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::scan::{
    mt_scan_max,
    mt_scan_max_exclusive,
    mt_scan_min,
    mt_scan_min_exclusive,
    mt_scan_prod,
    mt_scan_prod_exclusive,
};

const TPG: usize = 256;

// ── CPU oracles ──────────────────────────────────────────────────────────

fn cpu_scan_prod(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = 1.0f32;
        for c in 0..n {
            acc *= inp[r * n + c];
            out[r * n + c] = acc;
        }
    }
    out
}

fn cpu_scan_prod_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = 1.0f32;
        for c in 0..n {
            out[r * n + c] = acc;
            acc *= inp[r * n + c];
        }
    }
    out
}

fn cpu_scan_max(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = f32::NEG_INFINITY;
        for c in 0..n {
            if inp[r * n + c] > acc {
                acc = inp[r * n + c];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

fn cpu_scan_max_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = f32::NEG_INFINITY;
        for c in 0..n {
            out[r * n + c] = acc;
            if inp[r * n + c] > acc {
                acc = inp[r * n + c];
            }
        }
    }
    out
}

fn cpu_scan_min(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = f32::INFINITY;
        for c in 0..n {
            if inp[r * n + c] < acc {
                acc = inp[r * n + c];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

fn cpu_scan_min_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = f32::INFINITY;
        for c in 0..n {
            out[r * n + c] = acc;
            if inp[r * n + c] < acc {
                acc = inp[r * n + c];
            }
        }
    }
    out
}

// ── GPU dispatcher ────────────────────────────────────────────────────────

fn dispatch_f32<KernelFn>(inp: &[f32], rows: usize, n: usize, kernel_ir_for: KernelFn) -> Vec<f32>
where KernelFn: Fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("inp".into(), pack_bytes(inp, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0f32; rows * n], Dt::F32));
    b.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir_for(metaltile_core::dtype::DType::F32);
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, rows, 1], [TPG, 1, 1])
        .expect("dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(rows * n);
    out
}

// ── Product scan tests ────────────────────────────────────────────────────

#[test]
fn scan_prod_matches_cpu_aligned_n() {
    let _g = gpu_lock();
    let (rows, n) = (3, 1024);
    // Values bounded in [0.9, 1.1] so the product doesn't over/underflow.
    let inp: Vec<f32> = (0..rows * n).map(|i| 0.9 + (i % 5) as f32 * 0.05).collect();
    let expected = cpu_scan_prod(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_prod::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        // Floating-point prefix products accumulate error; 1% relative tol.
        let tol = (e.abs() * 0.01).max(1e-3);
        assert!(
            (e - a).abs() <= tol,
            "scan_prod aligned elem {i}: got {a:.6}, want {e:.6} (tol {tol:.6})"
        );
    }
}

#[test]
fn scan_prod_matches_cpu_ragged_n() {
    let _g = gpu_lock();
    let (rows, n) = (2, 1500);
    let inp: Vec<f32> = (0..rows * n).map(|i| 0.95 + (i % 3) as f32 * 0.025).collect();
    let expected = cpu_scan_prod(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_prod::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        let tol = (e.abs() * 0.01).max(1e-3);
        assert!((e - a).abs() <= tol, "scan_prod ragged elem {i}: got {a:.6}, want {e:.6}");
    }
}

#[test]
fn scan_prod_exclusive_first_element_is_one() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| 0.9 + (i % 5) as f32 * 0.05).collect();
    let actual = dispatch_f32(&inp, rows, n, mt_scan_prod_exclusive::kernel_ir_for);
    for r in 0..rows {
        assert!(
            (actual[r * n] - 1.0).abs() < 1e-6,
            "scan_prod_exclusive row {r} element 0 must be 1.0, got {}",
            actual[r * n]
        );
    }
}

#[test]
fn scan_prod_exclusive_matches_cpu() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| 0.9 + (i % 7) as f32 * 0.03).collect();
    let expected = cpu_scan_prod_exclusive(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_prod_exclusive::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        let tol = (e.abs() * 0.01).max(1e-3);
        assert!((e - a).abs() <= tol, "scan_prod_exclusive elem {i}: got {a:.6}, want {e:.6}");
    }
}

// ── Max scan tests ────────────────────────────────────────────────────────

#[test]
fn scan_max_matches_cpu_aligned_n() {
    let _g = gpu_lock();
    let (rows, n) = (3, 1024);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 37 + 11) % 200) as f32 - 100.0).collect();
    let expected = cpu_scan_max(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_max::kernel_ir_for);
    assert!(actual.iter().any(|&v| v.is_finite()), "output all non-finite — kernel body empty?");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-4, "scan_max aligned elem {i}: got {a:.4}, want {e:.4}");
    }
}

#[test]
fn scan_max_matches_cpu_ragged_n() {
    let _g = gpu_lock();
    let (rows, n) = (2, 3000);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 17 + 3) % 100) as f32 - 50.0).collect();
    let expected = cpu_scan_max(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_max::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-4, "scan_max ragged elem {i}: got {a:.4}, want {e:.4}");
    }
}

#[test]
fn scan_max_exclusive_first_element_is_neg_infinity() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| i as f32).collect();
    let actual = dispatch_f32(&inp, rows, n, mt_scan_max_exclusive::kernel_ir_for);
    for r in 0..rows {
        assert!(
            actual[r * n] == f32::NEG_INFINITY,
            "scan_max_exclusive row {r} element 0 must be -inf, got {}",
            actual[r * n]
        );
    }
}

#[test]
fn scan_max_exclusive_matches_cpu() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 53 + 7) % 200) as f32 - 100.0).collect();
    let expected = cpu_scan_max_exclusive(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_max_exclusive::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        // -inf expected → allow -inf actual; otherwise float equality.
        if e.is_infinite() && e.is_sign_negative() {
            assert!(
                a.is_infinite() && a.is_sign_negative(),
                "scan_max_exclusive elem {i}: got {a}, want -inf"
            );
        } else {
            assert!((e - a).abs() < 1e-4, "scan_max_exclusive elem {i}: got {a:.4}, want {e:.4}");
        }
    }
}

// ── Min scan tests ────────────────────────────────────────────────────────

#[test]
fn scan_min_matches_cpu_aligned_n() {
    let _g = gpu_lock();
    let (rows, n) = (3, 1024);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 37 + 11) % 200) as f32 - 100.0).collect();
    let expected = cpu_scan_min(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_min::kernel_ir_for);
    assert!(actual.iter().any(|&v| v.is_finite()), "output all non-finite — kernel body empty?");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-4, "scan_min aligned elem {i}: got {a:.4}, want {e:.4}");
    }
}

#[test]
fn scan_min_matches_cpu_ragged_n() {
    let _g = gpu_lock();
    let (rows, n) = (2, 2500);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 23 + 5) % 100) as f32 - 50.0).collect();
    let expected = cpu_scan_min(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_min::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-4, "scan_min ragged elem {i}: got {a:.4}, want {e:.4}");
    }
}

#[test]
fn scan_min_exclusive_first_element_is_infinity() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| i as f32).collect();
    let actual = dispatch_f32(&inp, rows, n, mt_scan_min_exclusive::kernel_ir_for);
    for r in 0..rows {
        assert!(
            actual[r * n] == f32::INFINITY,
            "scan_min_exclusive row {r} element 0 must be +inf, got {}",
            actual[r * n]
        );
    }
}

#[test]
fn scan_min_exclusive_matches_cpu() {
    let _g = gpu_lock();
    let (rows, n) = (2, 512);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 53 + 7) % 200) as f32 - 100.0).collect();
    let expected = cpu_scan_min_exclusive(&inp, rows, n);
    let actual = dispatch_f32(&inp, rows, n, mt_scan_min_exclusive::kernel_ir_for);
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        if e.is_infinite() && e.is_sign_positive() {
            assert!(
                a.is_infinite() && a.is_sign_positive(),
                "scan_min_exclusive elem {i}: got {a}, want +inf"
            );
        } else {
            assert!((e - a).abs() < 1e-4, "scan_min_exclusive elem {i}: got {a:.4}, want {e:.4}");
        }
    }
}
