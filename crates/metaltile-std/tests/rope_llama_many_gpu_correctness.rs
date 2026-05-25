//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end GPU correctness for `ffai::rope_llama_many`.
//!
//! `rope_llama_many` collapses the per-row T-loop of `ffai_rope_llama`
//! into ONE dispatch by lifting `position` from a constexpr to a
//! `Tensor<u32>` of length T and adding a row grid axis. The rotation
//! math, banding logic, and pair indexing inside a single row are
//! intentionally identical to `rope_llama`, so the cleanest oracle is
//! `rope_llama` itself looped row-by-row — that pins that the batched
//! kernel is bit-identical to the per-row primitive (modulo float
//! rounding from the same sequence of ops).
//!
//! For each `(T, n_heads, head_dim)` shape we:
//!   1. Pick random Q rows + random per-row positions.
//!   2. Run `ffai_rope_llama_many` in one dispatch.
//!   3. Run `ffai_rope_llama` once per row as the reference oracle.
//!   4. Assert the outputs match within the dtype's rotation tolerance.
//!
//! Coverage rationale: integration tests in FFAI's Swift side catch
//! garbage decode but not "per-row position lookup is silently off-by-1"
//! kinds of bug — this test pins the indexing.
//!
//! Dtype coverage: f32 / f16 / bf16.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rope_llama::ffai_rope_llama;
use metaltile_std::ffai::rope_llama_many::ffai_rope_llama_many;

/// Dispatch the batched `ffai_rope_llama_many` over T rows and read back
/// the rotated tensor as f32. `row_stride` = `n_heads * head_dim` for the
/// dense layout used in these tests.
#[allow(clippy::too_many_arguments)]
fn run_rope_llama_many(
    qk: &[f32],
    positions: &[u32],
    dt: Dt,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let row_stride = n_heads * head_dim;
    let elem_count = qk.len();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qk".into(), pack_bytes(qk, dt));
    buffers.insert("positions".into(), pack_u32_bytes(positions));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; elem_count], dt));
    buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
    buffers.insert("half_dim".into(), half_dim.to_le_bytes().to_vec());
    buffers.insert("row_stride".into(), row_stride.to_le_bytes().to_vec());
    buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());
    buffers.insert("scale_factor".into(), scale_factor.to_le_bytes().to_vec());
    buffers.insert("low_freq_factor".into(), low_freq_factor.to_le_bytes().to_vec());
    buffers.insert("high_freq_factor".into(), high_freq_factor.to_le_bytes().to_vec());
    buffers.insert("original_max_position".into(), original_max_position.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_rope_llama_many::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id<0>=row, <1>=head, <2>=i. One thread per
    // (row, head, i). For test shapes (max 32 * 16 * 32 = 16384) we go
    // multi-TG so just spread along one axis — pick `row` for the grid
    // axis since rows are independent of one another.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tokens as usize, 1, 1], [
            1,
            n_heads as usize,
            half_dim as usize,
        ])
        .expect("rope_llama_many dispatch");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
}

/// Per-row oracle: invoke `ffai_rope_llama` once per row with that row's
/// scalar position constexpr. Same kernel math, dispatched T times. This
/// is the "before" state of the prefill T-loop the batched kernel is
/// replacing.
#[allow(clippy::too_many_arguments)]
fn run_rope_llama_per_row(
    qk: &[f32],
    positions: &[u32],
    dt: Dt,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let row_elems = (n_heads * head_dim) as usize;
    let mut out_all = vec![0.0_f32; qk.len()];

    let ctx = Context::new().expect("Context::new on macOS");

    for r in 0..n_tokens as usize {
        let row_slice = &qk[r * row_elems..(r + 1) * row_elems];
        let position = positions[r];

        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("qk".into(), pack_bytes(row_slice, dt));
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; row_elems], dt));
        buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
        buffers.insert("half_dim".into(), half_dim.to_le_bytes().to_vec());
        buffers.insert("position".into(), position.to_le_bytes().to_vec());
        buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());
        buffers.insert("scale_factor".into(), scale_factor.to_le_bytes().to_vec());
        buffers.insert("low_freq_factor".into(), low_freq_factor.to_le_bytes().to_vec());
        buffers.insert("high_freq_factor".into(), high_freq_factor.to_le_bytes().to_vec());
        buffers
            .insert("original_max_position".into(), original_max_position.to_le_bytes().to_vec());

        let mut kernel = ffai_rope_llama::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Grid3D;

        // Per-row primitive's grid: [n_heads, half_dim, 1] — see the
        // existing `rope_llama_gpu_correctness` test for the same shape.
        assert!(
            n_heads as usize * half_dim as usize <= 1024,
            "single-TG dispatch: keep n_heads * half_dim <= 1024",
        );
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [
                n_heads as usize,
                half_dim as usize,
                1,
            ])
            .expect("rope_llama per-row dispatch");

        let row_out = unpack_bytes(result.outputs.get("out").expect("out buffer"), dt);
        out_all[r * row_elems..(r + 1) * row_elems].copy_from_slice(&row_out);
    }

    out_all
}

/// `no_scaling_params` returns Llama-3 params that disable banding.
fn no_scaling_params() -> (f32, f32, f32, f32) {
    (
        1.0,    // scale_factor — no compression
        1.0,    // low_freq_factor
        1.0,    // high_freq_factor
        1.0e10, // original_max_position — huge so wavelen never crosses
    )
}

/// Deterministic pseudo-random qk values. Same recipe as the existing
/// rope_llama / rope_2d tests so dtype tolerances carry over.
fn make_qk(n_tokens: u32, n_heads: u32, head_dim: u32, seed: u32) -> Vec<f32> {
    let n = (n_tokens * n_heads * head_dim) as usize;
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed) as f32;
            ((x * 0.0001).sin() + (x * 0.013).cos()) * 0.5
        })
        .collect()
}

/// Deterministic per-row positions. Mix small + large so banding kicks
/// in on the Llama-3 scaling case.
fn make_positions(n_tokens: u32, seed: u32) -> Vec<u32> {
    (0..n_tokens)
        .map(|r| {
            let v = r.wrapping_mul(2_654_435_761).wrapping_add(seed);
            // 0..32k range — covers both pre- and post-original_max for Llama-3.
            v % 32_000
        })
        .collect()
}

/// `(T, n_heads, head_dim)` cases requested in the issue. head_dim = 64
/// (half_dim = 32) gives `n_heads * half_dim` up to 16 * 32 = 512 which
/// still fits a single TG for the per-row oracle.
const CASES: &[(u32, u32, u32)] = &[
    (2, 4, 64),
    (8, 8, 64),
    (32, 16, 64),
    // An odd-shape sanity check — exercises a non-power-of-two row count
    // through the row grid axis.
    (5, 4, 64),
];

/// Standard RoPE (no banding) — exercises the common-case branch.
fn check_dtype_standard(dt: Dt, abs_tol: f32, rel_tol: f32) {
    let theta_base = 10000.0_f32;
    let (scale, low, high, max_pos) = no_scaling_params();

    for &(n_tokens, n_heads, head_dim) in CASES {
        let qk = make_qk(n_tokens, n_heads, head_dim, 0x1234);
        let positions = make_positions(n_tokens, 0x5678);

        let many = run_rope_llama_many(
            &qk,
            &positions,
            dt,
            n_tokens,
            n_heads,
            head_dim,
            theta_base,
            scale,
            low,
            high,
            max_pos,
        );
        let oracle = run_rope_llama_per_row(
            &qk,
            &positions,
            dt,
            n_tokens,
            n_heads,
            head_dim,
            theta_base,
            scale,
            low,
            high,
            max_pos,
        );

        assert_eq!(many.len(), oracle.len(), "length mismatch");
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (idx, (a, e)) in many.iter().zip(oracle.iter()).enumerate() {
            let d = (a - e).abs();
            let rel = d / e.abs().max(1e-3);
            if d > max_abs {
                max_abs = d;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                d <= abs_tol || rel <= rel_tol,
                "{dt:?} standard: shape=(T={n_tokens}, H={n_heads}, D={head_dim}) idx={idx}: \
                 many={a} oracle={e} abs={d:.3e} rel={rel:.3e}",
            );
        }
        eprintln!(
            "{dt:?} standard shape=(T={n_tokens}, H={n_heads}, D={head_dim}): \
             max_abs={max_abs:.2e} max_rel={max_rel:.2e}",
        );
    }
}

#[test]
fn rope_llama_many_matches_per_row_oracle_f32() {
    let _g = gpu_lock();
    // f32 path is the same float arithmetic on both sides — every op
    // sequence and exp2/log2 form is identical between the per-row and
    // batched kernels, so the result should be bit-equal (abs tol 0).
    // Keep a 5e-5 cushion to cover anything that goes through a CPU-side
    // pack/unpack on the boundary.
    check_dtype_standard(Dt::F32, 5e-5, 5e-5);
}

#[test]
fn rope_llama_many_matches_per_row_oracle_f16() {
    let _g = gpu_lock();
    // f16 loads/stores match between kernels — both cast to f32 for the
    // rotation. Any drift is dtype-round on output only.
    check_dtype_standard(Dt::F16, 2e-3, 5e-3);
}

#[test]
fn rope_llama_many_matches_per_row_oracle_bf16() {
    let _g = gpu_lock();
    // bf16 has 7-bit mantissa — wider tolerance than f16, same as the
    // existing rope_llama bf16 test.
    check_dtype_standard(Dt::Bf16, 1e-2, 2e-2);
}

#[test]
fn rope_llama_many_llama3_banding_matches_per_row_oracle_f32() {
    let _g = gpu_lock();
    // Llama-3.1 banding active — exercises both the low/medium/high
    // branches of the `select` chain. Large positions push beyond
    // `original_max_position=8192` so banding actually fires.
    let theta_base = 500000.0_f32;
    let (scale, low, high, max_pos) = (8.0_f32, 1.0_f32, 4.0_f32, 8192.0_f32);

    for &(n_tokens, n_heads, head_dim) in CASES {
        let qk = make_qk(n_tokens, n_heads, head_dim, 0xABCD);
        let positions = make_positions(n_tokens, 0xBEEF);

        let many = run_rope_llama_many(
            &qk,
            &positions,
            Dt::F32,
            n_tokens,
            n_heads,
            head_dim,
            theta_base,
            scale,
            low,
            high,
            max_pos,
        );
        let oracle = run_rope_llama_per_row(
            &qk,
            &positions,
            Dt::F32,
            n_tokens,
            n_heads,
            head_dim,
            theta_base,
            scale,
            low,
            high,
            max_pos,
        );

        let mut max_diff = 0.0_f32;
        for (a, e) in many.iter().zip(oracle.iter()) {
            max_diff = max_diff.max((a - e).abs());
        }
        // f32 same-arith path on both kernels — large-argument sin/cos
        // can still drop ULPs, but stay <2e-3 (matches the rope_llama
        // banding test budget).
        assert!(
            max_diff < 2e-3,
            "Llama-3 banding f32 shape=(T={n_tokens}, H={n_heads}, D={head_dim}): \
             max_diff={max_diff:.3e}",
        );
    }
}
