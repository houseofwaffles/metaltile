//! Long-context + batched coverage for `mt_sdpa_prefill_mma`.
//!
//! The kernel is bench-wired at B=1, T=512 in
//! `mlx/steel/attn/steel_attention_mma.rs`. The dispatch geometry
//! (BQ=32 walked over T via `tgid_x`, BK=16 walked over K inside the
//! body, batch via `tgid_z` folded into the slab offsets via the
//! `n_q_heads` + `n_kv_heads` constexprs) supports any (B, T) where
//! T is a multiple of BQ. This file pins:
//!
//! * single-batch correctness at T ∈ {2048, 4096} (production
//!   prefill lengths),
//! * kernel-side B>1 correctness at (B=2, T=1024) and (B=4, T=512),
//! * legacy `[B*n_heads, T, D]` head-flatten callers still work at
//!   `tgid_z = 0` (B=2 test) — useful for any consumer that shipped
//!   against the pre-B>1 kernel API.
//!
//! All shapes compare against a CPU naive SDPA reference. Tolerance
//! is `2e-2` (same as the bench's `tol=2e-2` for sdpa_prefill_mma).
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::attn::steel_attention_mma::mt_sdpa_prefill_mma;

/// Naive SDPA prefill (single batch, full causal). Q/K/V are
/// `[n_heads_or_kv * T * D]` row-major; output is `[n_heads * T * D]`.
/// GQA via `kv_head = q_head / gqa_factor`.
#[allow(clippy::too_many_arguments)]
fn naive_sdpa_prefill_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(n_heads.is_multiple_of(n_kv_heads));
    let gqa = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * t * head_dim];
    for qh in 0..n_heads {
        let kvh = qh / gqa;
        let q_head_off = qh * t * head_dim;
        let kv_head_off = kvh * t * head_dim;
        for qi in 0..t {
            let q_off = q_head_off + qi * head_dim;
            // Causal: attend only to k positions [0..=qi].
            let mut scores = vec![0.0f32; qi + 1];
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_off = kv_head_off + ki * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (ki, s) in scores.iter().enumerate() {
                    acc += *s * inv * v[kv_head_off + ki * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_prefill(
    ctx: &Context,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    head_dim: usize,
    scale: f32,
    dt: Dt,
) -> Vec<f32> {
    let dt_bytes = dt.bytes();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), q_bytes.to_vec());
    buffers.insert("k".into(), k_bytes.to_vec());
    buffers.insert("v".into(), v_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; batch * n_heads * t * head_dim * dt_bytes]);
    buffers.insert("q_len".into(), (t as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (t as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), ((n_heads / n_kv_heads) as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let mut kernel = mt_sdpa_prefill_mma::kernel_ir_for(dt.to_dtype());
    // SimdGroup2D is the bench's dispatch mode for this kernel (see
    // `run_sdpa_prefill` in `crates/metaltile-std/src/run_spec.rs`).
    // It's required because the body reads `tgid_x`/`tgid_y`/`tgid_z`
    // directly — only SimdGroup2D maps `uint3 tid
    // [[threadgroup_position_in_grid]]` so the three axes resolve.
    kernel.mode = metaltile_core::ir::KernelMode::SimdGroup2D;
    // Grid: (q_len / BQ=32, n_heads, batch). 128 threads = 4 SGs.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [t / 32, n_heads, batch], [
            128, 1, 1,
        ])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dt)
}

#[test]
fn mt_sdpa_prefill_mma_matches_cpu_reference_t2048_f32() {
    let _g = gpu_lock();

    // T=2048 hits 64 q tiles per head; small enough that the CPU
    // naive reference (O(n_heads × T² × D)) runs in ~1s on
    // 4 heads × 4M ops × 128 D ≈ 2G fp ops.
    let n_heads = 4usize;
    let n_kv_heads = 1usize;
    let t = 2048usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_heads * t * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * t * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * t * head_dim, 11, 5.0);
    let expected = naive_sdpa_prefill_f32(&q, &k, &v, n_heads, n_kv_heads, t, head_dim, scale);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::F32);
    let k_b = pack_bytes(&k, Dt::F32);
    let v_b = pack_bytes(&v, Dt::F32);
    let actual = run_sdpa_prefill(
        &ctx,
        &q_b,
        &k_b,
        &v_b,
        1,
        n_heads,
        n_kv_heads,
        t,
        head_dim,
        scale,
        Dt::F32,
    );

    assert_eq!(actual.len(), expected.len());

    // 2e-2 tolerance matches the bench's `tol=2e-2` for sdpa_prefill_mma.
    // Sources of drift: simd_shuffle_xor row-reduction reorders sums,
    // simdgroup matmul uses fp32 accumulators against the MLX naive's
    // sequential summation order, and scale*log2(e) is applied post-
    // Q·K^T (kernel) vs pre-baked (CPU) — small numerical-pipeline
    // reordering that compounds with T.
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
        max_diff < 2e-2,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_sdpa_prefill_mma_b2_via_head_flatten_t1024_f32() {
    let _g = gpu_lock();

    // Backwards-compat pin: callers that flatten Q/K/V to the
    // `[B * n_heads, T, D]` layout and dispatch with `tgid_y`
    // ranging over `[0, B * n_q_heads)` and `tgid_z = 0` (effective
    // batch=1) should still produce the same per-batch outputs as
    // the true B>1 dispatch below. Useful for early adopters that
    // already shipped against the pre-B>1 kernel.
    let batch = 2usize;
    let n_heads = 4usize;
    let n_kv_heads = 1usize;
    let t = 1024usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Per-batch buffers: lay out flat `[B * n_heads, T, D]` row-major.
    // Use distinct ramps per batch so a missed batch index would show
    // up as a max-diff blowup (vs both batches sharing the same data
    // and the kernel being "right" for either).
    let q: Vec<f32> = (0..batch * n_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_heads;
            let rest = i - bh * t * head_dim;
            (rest % 17) as f32 * 0.05 - 0.4 + (b as f32) * 0.13
        })
        .collect();
    let k: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 13) as f32 * 0.05 - 0.3 + (b as f32) * 0.11
        })
        .collect();
    let v: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 11) as f32 * 0.05 - 0.25 + (b as f32) * 0.17
        })
        .collect();

    // CPU reference: same naive prefill, applied independently per
    // batch. Concatenate per-batch outputs into the flat
    // `[B * n_heads, T, D]` layout.
    let mut expected = vec![0.0f32; batch * n_heads * t * head_dim];
    for b in 0..batch {
        let q_b = &q[b * n_heads * t * head_dim..(b + 1) * n_heads * t * head_dim];
        let k_b = &k[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let v_b = &v[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let out_b = naive_sdpa_prefill_f32(q_b, k_b, v_b, n_heads, n_kv_heads, t, head_dim, scale);
        let dst = &mut expected[b * n_heads * t * head_dim..(b + 1) * n_heads * t * head_dim];
        dst.copy_from_slice(&out_b);
    }

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::F32);
    let k_b = pack_bytes(&k, Dt::F32);
    let v_b = pack_bytes(&v, Dt::F32);

    // Dispatch with the flattened-head grid: `tgid_y` walks
    // `B * n_q_heads` rows; the kernel's gqa div still resolves
    // correctly because B*n_kv_heads has the same flatten.
    let flat_n_heads = batch * n_heads;
    let flat_n_kv_heads = batch * n_kv_heads;
    let actual = run_sdpa_prefill(
        &ctx,
        &q_b,
        &k_b,
        &v_b,
        1, // batch=1 via head-flatten layout; real batches are folded into flat_n_heads
        flat_n_heads,
        flat_n_kv_heads,
        t,
        head_dim,
        scale,
        Dt::F32,
    );

    assert_eq!(actual.len(), expected.len());

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
        max_diff < 2e-2,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_sdpa_prefill_mma_kernel_side_b2_t1024_f32() {
    let _g = gpu_lock();

    // True B>1 dispatch: kernel reads `batch = tgid_z` and folds it
    // into Q/K/V/O slab offsets via the `n_q_heads` + `n_kv_heads`
    // constexprs. Q/K/V are laid out as `[B, n_*_heads, T, D]`
    // (per-batch slab) and the grid spans all 3 axes.
    let batch = 2usize;
    let n_q_heads = 4usize;
    let n_kv_heads = 1usize;
    let t = 1024usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Per-batch buffers: distinct ramps so a missed batch offset
    // would surface as a magnitude mismatch.
    let q: Vec<f32> = (0..batch * n_q_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_q_heads;
            let rest = i - bh * t * head_dim;
            (rest % 17) as f32 * 0.05 - 0.4 + (b as f32) * 0.13
        })
        .collect();
    let k: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 13) as f32 * 0.05 - 0.3 + (b as f32) * 0.11
        })
        .collect();
    let v: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 11) as f32 * 0.05 - 0.25 + (b as f32) * 0.17
        })
        .collect();

    // CPU reference: naive prefill independently per batch.
    let mut expected = vec![0.0f32; batch * n_q_heads * t * head_dim];
    for b in 0..batch {
        let q_b = &q[b * n_q_heads * t * head_dim..(b + 1) * n_q_heads * t * head_dim];
        let k_b = &k[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let v_b = &v[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let out_b =
            naive_sdpa_prefill_f32(q_b, k_b, v_b, n_q_heads, n_kv_heads, t, head_dim, scale);
        let dst = &mut expected[b * n_q_heads * t * head_dim..(b + 1) * n_q_heads * t * head_dim];
        dst.copy_from_slice(&out_b);
    }

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::F32);
    let k_b = pack_bytes(&k, Dt::F32);
    let v_b = pack_bytes(&v, Dt::F32);
    let actual = run_sdpa_prefill(
        &ctx,
        &q_b,
        &k_b,
        &v_b,
        batch,
        n_q_heads,
        n_kv_heads,
        t,
        head_dim,
        scale,
        Dt::F32,
    );

    assert_eq!(actual.len(), expected.len());

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
        max_diff < 2e-2,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_sdpa_prefill_mma_kernel_side_b4_t512_f32() {
    let _g = gpu_lock();

    // Tighter-batch (B=4) coverage at T=512: catches per-batch
    // stride bugs that B=2 might cover up if the bug only shows
    // up at batch ≥ 3 (e.g., truncated shift constants).
    let batch = 4usize;
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let t = 512usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..batch * n_q_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_q_heads;
            let rest = i - bh * t * head_dim;
            (rest % 19) as f32 * 0.04 - 0.35 + (b as f32) * 0.09
        })
        .collect();
    let k: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 13) as f32 * 0.04 - 0.25 + (b as f32) * 0.07
        })
        .collect();
    let v: Vec<f32> = (0..batch * n_kv_heads * t * head_dim)
        .map(|i| {
            let bh = i / (t * head_dim);
            let b = bh / n_kv_heads;
            let rest = i - bh * t * head_dim;
            (rest % 11) as f32 * 0.04 - 0.2 + (b as f32) * 0.13
        })
        .collect();

    let mut expected = vec![0.0f32; batch * n_q_heads * t * head_dim];
    for b in 0..batch {
        let q_b = &q[b * n_q_heads * t * head_dim..(b + 1) * n_q_heads * t * head_dim];
        let k_b = &k[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let v_b = &v[b * n_kv_heads * t * head_dim..(b + 1) * n_kv_heads * t * head_dim];
        let out_b =
            naive_sdpa_prefill_f32(q_b, k_b, v_b, n_q_heads, n_kv_heads, t, head_dim, scale);
        let dst = &mut expected[b * n_q_heads * t * head_dim..(b + 1) * n_q_heads * t * head_dim];
        dst.copy_from_slice(&out_b);
    }

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::F32);
    let k_b = pack_bytes(&k, Dt::F32);
    let v_b = pack_bytes(&v, Dt::F32);
    let actual = run_sdpa_prefill(
        &ctx,
        &q_b,
        &k_b,
        &v_b,
        batch,
        n_q_heads,
        n_kv_heads,
        t,
        head_dim,
        scale,
        Dt::F32,
    );

    assert_eq!(actual.len(), expected.len());

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
        max_diff < 2e-2,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_sdpa_prefill_mma_matches_cpu_reference_t4096_f32() {
    let _g = gpu_lock();

    // T=4096 hits 128 q tiles per head. n_heads=2 caps CPU ref cost.
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let t = 4096usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_heads * t * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * t * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * t * head_dim, 11, 5.0);
    let expected = naive_sdpa_prefill_f32(&q, &k, &v, n_heads, n_kv_heads, t, head_dim, scale);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::F32);
    let k_b = pack_bytes(&k, Dt::F32);
    let v_b = pack_bytes(&v, Dt::F32);
    let actual = run_sdpa_prefill(
        &ctx,
        &q_b,
        &k_b,
        &v_b,
        1,
        n_heads,
        n_kv_heads,
        t,
        head_dim,
        scale,
        Dt::F32,
    );

    assert_eq!(actual.len(), expected.len());

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
        max_diff < 2e-2,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}
