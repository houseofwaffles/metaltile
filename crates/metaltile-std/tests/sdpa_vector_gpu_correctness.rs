//! End-to-end GPU correctness tests for `mt_sdpa_vector` — the decode-step
//! attention kernel introduced in PR #43 (faithful port of MLX's
//! `sdpa_vector<T, D, V=D>` template, see
//! `crates/metaltile-std/src/mlx/sdpa_vector.rs`).
//!
//! Pre-this-file, `mt_sdpa_vector` had no direct CPU-oracle correctness
//! test: validation flowed exclusively through the `tile bench`
//! head-to-head against MLX's compiled kernel. That suffices as a
//! same-shape regression guard but does not catch a bug if both kernels
//! drift the same way, and it is gated on the MLX library being
//! available at bench time. This file pins the algorithm — proc-macro
//! → IR → MSL → PSO → dispatch → readback — against a straight
//! triple-loop softmax(Q·Kᵀ·scale)·V reference, with no MLX dependency.
//!
//! The kernel signature is:
//!   q   : Tensor<T>   [n_q_heads, head_dim]
//!   k   : Tensor<T>   [n_kv_heads, n_kv, head_dim]
//!   v   : Tensor<T>   [n_kv_heads, n_kv, head_dim]
//!   out : Tensor<T>   [n_q_heads, head_dim]
//!   constexprs: head_dim, n_kv, gqa_factor, scale
//! Geometry: tpg = 1024 (BN × BD = 32 × 32 simdgroups × lanes),
//! grid = [n_q_heads, 1, 1]. `head_dim` is hardcoded to 128 in the
//! kernel; all tests use that value.
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::sdpa_vector::mt_sdpa_vector;

/// Triple-loop CPU SDPA reference: `O = softmax(Q · Kᵀ · scale) · V`
/// per Q head, GQA via `kv_head = q_head / gqa`. Pure f32 throughout —
/// dtype-specific quantisation is modeled by round-tripping the inputs
/// through the target dtype before this oracle runs (see f16 tests).
fn cpu_sdpa_decode_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    n_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(n_q_heads.is_multiple_of(n_kv_heads));
    let gqa = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let kv_h = h / gqa;
        // scores[j] = scale · ⟨q[h], k[kv_h, j]⟩
        let mut scores = vec![0.0_f32; n_kv];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k[kv_h * n_kv * head_dim + j * head_dim + d];
            }
            *score = dot * scale;
        }
        // softmax(scores) — online with max-subtraction for numeric
        // stability (matches the kernel's `run_max` rescale path).
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut e: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for ej in e.iter_mut() {
            *ej *= inv;
        }
        // o[h, d] = Σⱼ p[j] · v[kv_h, j, d]
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (j, ej) in e.iter().enumerate() {
                acc += *ej * v[kv_h * n_kv * head_dim + j * head_dim + d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

/// Per-dtype kernel dispatch. Packs inputs through `dt`, dispatches
/// `mt_sdpa_vector::kernel_ir_for(dt)` with `KernelMode::Reduction`
/// (matches `run_spec.rs::run_sdpa_vector`), unpacks `out` back to
/// f32 for comparison.
#[allow(clippy::too_many_arguments)]
fn run_sdpa_vector(
    ctx: &Context,
    dt: Dt,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    gqa_factor: usize,
    scale: f32,
) -> Vec<f32> {
    let mut kernel = mt_sdpa_vector::kernel_ir_for(dt.to_dtype());
    // `kernel_ir_for` returns the default Elementwise mode; the proc-
    // macro `bench_kernel(class=SdpaVector,...)` annotation drives a
    // reduction-shaped codegen in the bench path. Mirror that here so
    // the threadgroup_alloc / simd_sum / cross-sg-reduction ops emit
    // the right MSL. Sibling sdpa_decode test does the same.
    kernel.mode = KernelMode::Reduction;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * dt.bytes()]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dt)
}

/// Round-trip a slice through `dt` so the CPU oracle sees what the
/// kernel sees after the load-cast quantisation. No-op for f32.
fn round_through(vals: &[f32], dt: Dt) -> Vec<f32> { vals.iter().map(|&v| dt.round(v)).collect() }

fn assert_close_abs(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

fn assert_close_rel(actual: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        // Relative tol with a small floor: at output magnitudes ~0.1
        // pure relative explodes from accumulated softmax-times-V noise.
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    assert!(
        max_rel < rel_tol,
        "{label}: max rel diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

// ── Tests ────────────────────────────────────────────────────────────

/// Small dense shape, gqa=1 (one Q head per KV head). Pins the algo
/// against the CPU oracle in pure fp32: any deviation is from kernel
/// arithmetic (simd_sum reordering, cross-sg reduction), not load-cast
/// quantisation.
#[test]
fn mt_sdpa_vector_matches_cpu_reference_f32() {
    let _g = gpu_lock();
    let head_dim = 128usize;
    let n_kv = 256usize;
    let n_q_heads = 4usize;
    let gqa_factor = 1usize;
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);

    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let actual =
        run_sdpa_vector(&ctx, Dt::F32, &q, &k, &v, n_q_heads, head_dim, n_kv, gqa_factor, scale);

    // 1e-3 absolute envelope: at this shape (n_kv=256, head_dim=128)
    // the softmax-weighted V accumulation involves ~256 fp32 multiply-
    // adds per output element. ULP drift from simd_sum reordering of
    // the score partial + exp ulp noise stays well inside 1e-3.
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector f32 vs CPU");
}

/// Same small shape, f16 storage. Round-trips Q/K/V through f16
/// before the oracle so what the kernel loads (after the f16→f32 cast
/// at the head of the loop) matches what the reference accumulates.
#[test]
fn mt_sdpa_vector_matches_cpu_reference_f16() {
    let _g = gpu_lock();
    let head_dim = 128usize;
    let n_kv = 256usize;
    let n_q_heads = 4usize;
    let gqa_factor = 1usize;
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);

    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let actual =
        run_sdpa_vector(&ctx, Dt::F16, &q, &k, &v, n_q_heads, head_dim, n_kv, gqa_factor, scale);

    // f16 has 10 bits of mantissa → ULP ≈ |v| · 2^-10. With a final
    // f32 → f16 narrowing at the store + 256 KV positions accumulated,
    // 5e-3 relative is the same envelope qmm_f16 uses. Per-element
    // |diff| is small but relative to small outputs goes up.
    assert_close_rel(&actual, &expected, 5e-3, "mt_sdpa_vector f16 vs CPU");
}

/// GQA gather: 8 Q heads, gqa=4 → 2 KV heads. Heads 0..4 read kv_head 0,
/// heads 4..8 read kv_head 1. Verifies the `q_head / gqa_factor` index
/// math in the kernel matches the CPU oracle's `h / gqa` derivation.
/// Any off-by-one in the kv_head computation would show up as a whole-
/// head output mismatch here (catastrophic, far past tolerance).
#[test]
fn mt_sdpa_vector_gqa_factor_4_f32() {
    let _g = gpu_lock();
    let head_dim = 128usize;
    let n_kv = 256usize;
    let n_q_heads = 8usize;
    let gqa_factor = 4usize;
    let n_kv_heads = n_q_heads / gqa_factor;
    assert_eq!(n_kv_heads, 2);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);

    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let actual =
        run_sdpa_vector(&ctx, Dt::F32, &q, &k, &v, n_q_heads, head_dim, n_kv, gqa_factor, scale);

    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector f32 GQA=4 vs CPU");
}

/// Production decode cell: Qwen3-7B / 14B class — 32 Q heads, gqa=4
/// (8 KV heads), n_kv=4096, head_dim=128, f16. This exercises the
/// long-n_kv strided walk where each simdgroup iterates n_kv / 32 = 128
/// positions, with the online-softmax rescale firing many times. f16
/// at n_kv=4096 is where the kernel actually lives in production —
/// any quality bug in the rescale path would surface here.
#[test]
fn mt_sdpa_vector_qwen3_decode_shape_f16() {
    let _g = gpu_lock();
    let head_dim = 128usize;
    let n_kv = 4096usize;
    let n_q_heads = 32usize;
    let gqa_factor = 4usize;
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);

    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let actual =
        run_sdpa_vector(&ctx, Dt::F16, &q, &k, &v, n_q_heads, head_dim, n_kv, gqa_factor, scale);

    // 1e-2 relative: at n_kv=4096 the softmax sums O(thousands) of
    // exp() partials and each f16 narrowing accumulates ~10-bit
    // mantissa error. The bench's tol=1e-3 is fp32-only — f16 at
    // this length needs the wider envelope. Same scale used by
    // sdpa_decode_2pass f16 tests.
    assert_close_rel(&actual, &expected, 1e-2, "mt_sdpa_vector f16 Qwen3 shape vs CPU");
}

// ── New head_dim variants ─────────────────────────────────────────────────

use metaltile_std::mlx::sdpa_vector::{
    mt_sdpa_vector_d64,
    mt_sdpa_vector_d96,
    mt_sdpa_vector_d192,
    mt_sdpa_vector_d256,
};

/// Generic dispatch helper for the new head_dim kernels.
#[allow(clippy::too_many_arguments)]
fn run_sdpa_vector_generic(
    ctx: &Context,
    dt: Dt,
    kernel_ir: metaltile_core::ir::Kernel,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    gqa_factor: usize,
    scale: f32,
) -> Vec<f32> {
    use metaltile_core::ir::KernelMode;
    let mut kernel = kernel_ir;
    kernel.mode = KernelMode::Reduction;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * dt.bytes()]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dt)
}

// ── d=64 ──────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_vector_d64_matches_cpu_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (64usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d64::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d64 f32 vs CPU");
}

#[test]
fn mt_sdpa_vector_d64_matches_cpu_f16() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (64usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F16,
        mt_sdpa_vector_d64::kernel_ir_for(metaltile_core::dtype::DType::F16),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_rel(&actual, &expected, 5e-3, "mt_sdpa_vector_d64 f16 vs CPU");
}

#[test]
fn mt_sdpa_vector_d64_gqa_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (64usize, 256usize, 8usize, 4usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 19, 9.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d64::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d64 f32 GQA=4 vs CPU");
}

// ── d=96 ──────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_vector_d96_matches_cpu_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (96usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d96::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d96 f32 vs CPU");
}

#[test]
fn mt_sdpa_vector_d96_matches_cpu_f16() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (96usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F16,
        mt_sdpa_vector_d96::kernel_ir_for(metaltile_core::dtype::DType::F16),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_rel(&actual, &expected, 5e-3, "mt_sdpa_vector_d96 f16 vs CPU");
}

// ── d=192 ─────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_vector_d192_matches_cpu_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (192usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d192::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d192 f32 vs CPU");
}

#[test]
fn mt_sdpa_vector_d192_matches_cpu_f16() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (192usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F16,
        mt_sdpa_vector_d192::kernel_ir_for(metaltile_core::dtype::DType::F16),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_rel(&actual, &expected, 5e-3, "mt_sdpa_vector_d192 f16 vs CPU");
}

// ── d=256 ─────────────────────────────────────────────────────────────────

#[test]
fn mt_sdpa_vector_d256_matches_cpu_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (256usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d256::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d256 f32 vs CPU");
}

#[test]
fn mt_sdpa_vector_d256_matches_cpu_f16() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (256usize, 256usize, 4usize, 1usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_f32 = ramp(n_q_heads * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let q = round_through(&q_f32, Dt::F16);
    let k = round_through(&k_f32, Dt::F16);
    let v = round_through(&v_f32, Dt::F16);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F16,
        mt_sdpa_vector_d256::kernel_ir_for(metaltile_core::dtype::DType::F16),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_rel(&actual, &expected, 5e-3, "mt_sdpa_vector_d256 f16 vs CPU");
}

#[test]
fn mt_sdpa_vector_d256_gqa_f32() {
    let _g = gpu_lock();
    let (head_dim, n_kv, n_q_heads, gqa_factor) = (256usize, 256usize, 8usize, 4usize);
    let n_kv_heads = n_q_heads / gqa_factor;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 19, 9.0);
    let k = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * n_kv * head_dim, 11, 5.0);
    let expected = cpu_sdpa_decode_reference(&q, &k, &v, n_q_heads, n_kv_heads, n_kv, head_dim);
    let ctx = Context::new().expect("Context::new");
    let actual = run_sdpa_vector_generic(
        &ctx,
        Dt::F32,
        mt_sdpa_vector_d256::kernel_ir_for(metaltile_core::dtype::DType::F32),
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        gqa_factor,
        scale,
    );
    assert_close_abs(&actual, &expected, 1e-3, "mt_sdpa_vector_d256 f32 GQA=4 vs CPU");
}
