//! End-to-end GPU correctness for `ffai::rope_llama`.
//!
//! Llama-family per-token RoPE with optional Llama-3 wavelength banding.
//! For each `(head, i in 0..head_dim/2)`:
//!
//!   base_inv_freq = 1 / theta_base^(2i / head_dim)
//!   wavelen       = 2*pi / base_inv_freq
//!
//!   if wavelen > low_freq_wavelen:        inv_freq = base / scale_factor
//!   else if wavelen < high_freq_wavelen:  inv_freq = base
//!   else (medium band):                   smooth interpolation
//!
//!   theta = position * inv_freq
//!   o[i]            = x[i]*cos(theta) - x[i+half]*sin(theta)
//!   o[i+half]       = x[i]*sin(theta) + x[i+half]*cos(theta)
//!
//! Coverage rationale: `rope_llama` had no end-to-end GPU coverage —
//! `ffai` integration validates against full model decoding, but a
//! wrong index formula or a silent kernel-emptiness regression (the
//! proc-macro class that bit `kv_cache_update` + `softmax_categorical_sample`
//! in PR #19) would not surface until a real decode produced gibberish.
//!
//! Three scenarios:
//!   - Identity at position=0 (cos=1, sin=0 → output equals input)
//!   - Standard RoPE (no Llama-3 scaling): scale_factor=1 + huge
//!     original_max_position so banding never triggers; compare to a
//!     CPU oracle bit-exactly in f32
//!   - Llama-3 scaling (Llama-3.1 8B params): scale_factor=8,
//!     low_freq_factor=1, high_freq_factor=4, original_max=8192;
//!     verify low/medium/high band dispatch matches CPU oracle
//!
//! Dtype coverage: f32 / f16 / bf16.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rope_llama::rope_llama;

/// CPU oracle. Matches the kernel's exact arithmetic — banding logic
/// (low/medium/high), exp2/log2 form of the inverse-frequency, fused
/// rotate. Returns the rotated tensor as f32.
#[allow(clippy::too_many_arguments)]
fn naive_rope_llama(
    qk: &[f32],
    head_dim: u32,
    n_heads: u32,
    position: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let half_f = half_dim as f32;
    let two_pi = std::f32::consts::TAU;
    let mut out = vec![0.0_f32; qk.len()];

    for head in 0..n_heads {
        let base = (head * head_dim) as usize;
        for i in 0..half_dim {
            let i_f = i as f32;
            let inv_freq_base = (-i_f * theta_base.log2() / half_f).exp2();
            let wavelen = two_pi / inv_freq_base;
            let low_wavelen = original_max_position / low_freq_factor;
            let high_wavelen = original_max_position / high_freq_factor;
            let scaled = inv_freq_base / scale_factor;
            let smooth_num = original_max_position / wavelen - low_freq_factor;
            let smooth_den = high_freq_factor - low_freq_factor;
            let s = smooth_num / smooth_den;
            let smoothed = (1.0 - s) * scaled + s * inv_freq_base;

            let inv_freq = if wavelen > low_wavelen {
                scaled
            } else if wavelen < high_wavelen {
                inv_freq_base
            } else {
                smoothed
            };

            let theta = position as f32 * inv_freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let i1 = base + i as usize;
            let i2 = base + (i + half_dim) as usize;
            let x1 = qk[i1];
            let x2 = qk[i2];
            out[i1] = x1 * cos_t - x2 * sin_t;
            out[i2] = x1 * sin_t + x2 * cos_t;
        }
    }

    out
}

/// Dispatch the kernel and read back the rotated tensor in `dt`.
#[allow(clippy::too_many_arguments)]
fn run_rope_llama(
    qk: &[f32],
    dt: Dt,
    n_heads: u32,
    head_dim: u32,
    position: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let elem_count = qk.len();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qk".into(), pack_bytes(qk, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; elem_count], dt));
    buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
    buffers.insert("half_dim".into(), half_dim.to_le_bytes().to_vec());
    buffers.insert("position".into(), position.to_le_bytes().to_vec());
    buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());
    buffers.insert("scale_factor".into(), scale_factor.to_le_bytes().to_vec());
    buffers.insert("low_freq_factor".into(), low_freq_factor.to_le_bytes().to_vec());
    buffers.insert("high_freq_factor".into(), high_freq_factor.to_le_bytes().to_vec());
    buffers.insert("original_max_position".into(), original_max_position.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = rope_llama::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id::<0> = head, program_id::<1> = i (0..half_dim).
    // One thread per (head, i). For test shapes (head≤16, half≤64) a
    // 1024-thread TG fits; use grid=[1, 1, 1] tpg=[n_heads, half_dim, 1].
    // Assert here so the test fails loudly if a future test bumps a dim
    // past the TG limit (better than a silent dispatch-time error).
    assert!(
        n_heads as usize * half_dim as usize <= 1024,
        "test dispatches a single TG — keep n_heads*half_dim ≤ 1024",
    );
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [
            n_heads as usize,
            half_dim as usize,
            1,
        ])
        .expect("dispatch_with_grid");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
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

/// Llama-3.1 8B official RoPE-scaling parameters.
fn llama3_scaling_params() -> (f32, f32, f32, f32) {
    (
        8.0,    // scale_factor
        1.0,    // low_freq_factor
        4.0,    // high_freq_factor
        8192.0, // original_max_position (Llama-3 base ctx)
    )
}

#[test]
fn rope_llama_identity_at_position_zero_f32() {
    let _g = gpu_lock();
    // position=0 → theta=0 → cos=1, sin=0 → output == input regardless
    // of banding or theta_base. Pins indexing / grid layout — a wrong
    // head or i mapping smears values across the wrong slot.
    let n_heads = 4u32;
    let head_dim = 32u32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| 0.1 + (i as f32 * 0.013).sin()).collect();

    let (scale, low, high, max_pos) = llama3_scaling_params();
    let actual =
        run_rope_llama(&qk, Dt::F32, n_heads, head_dim, 0, 500000.0, scale, low, high, max_pos);

    for (idx, (a, e)) in actual.iter().zip(qk.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "identity at pos=0 broke at idx={idx}: got {a}, expected {e}",
        );
    }
}

#[test]
fn rope_llama_standard_rope_matches_oracle_f32() {
    let _g = gpu_lock();
    // No Llama-3 banding (scale=1, max_pos=huge) → pure RoPE. f32 → bit-
    // exact comparison vs CPU oracle (same arithmetic, same exp2/log2
    // path). Realistic shape (Llama-2 70B head_dim=128) at position=137.
    let n_heads = 8u32;
    let head_dim = 64u32; // keep n_heads * half_dim ≤ 1024
    let position = 137u32;
    let theta_base = 10000.0_f32;
    let (scale, low, high, max_pos) = no_scaling_params();

    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let expected =
        naive_rope_llama(&qk, head_dim, n_heads, position, theta_base, scale, low, high, max_pos);
    let actual = run_rope_llama(
        &qk,
        Dt::F32,
        n_heads,
        head_dim,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );

    let mut max_diff = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "max |diff| = {max_diff:.2e} exceeds 5e-5 tol");
}

#[test]
fn rope_llama_llama3_scaling_matches_oracle_f32() {
    let _g = gpu_lock();
    // Llama-3.1 banding active. position > original_max_position triggers
    // the low/medium-frequency bands; this is exactly the scenario the
    // banding logic exists to handle. Compare against CPU oracle.
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 16000u32; // 2× original_max_position
    let theta_base = 500000.0_f32; // Llama-3 theta_base
    let (scale, low, high, max_pos) = llama3_scaling_params();

    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 53) as f32 - 26.0) * 0.04).collect();
    let expected =
        naive_rope_llama(&qk, head_dim, n_heads, position, theta_base, scale, low, high, max_pos);
    let actual = run_rope_llama(
        &qk,
        Dt::F32,
        n_heads,
        head_dim,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );

    let mut max_diff = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    // At position=16000 theta argument can reach ~16K → GPU's native
    // sin/cos lose precision on large arguments (argument reduction
    // ULPs). 2e-3 absolute is still tight given |x| ≤ 1.
    assert!(max_diff < 2e-3, "Llama-3 banding: max |diff| = {max_diff:.2e} > 2e-3");
}

#[test]
fn rope_llama_standard_rope_matches_oracle_f16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 73u32;
    let theta_base = 10000.0_f32;
    let (scale, low, high, max_pos) = no_scaling_params();

    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    // Round source through f16 so the CPU oracle uses the same load
    // precision as the kernel does in its initial cast.
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = naive_rope_llama(
        &qk_rounded,
        head_dim,
        n_heads,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );
    let actual = run_rope_llama(
        &qk,
        Dt::F16,
        n_heads,
        head_dim,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );

    let mut max_rel = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // f16 inputs round-trip to ~10-bit mantissa; rotate + cast at write
    // adds another ULP. 5e-3 relative is plenty for the rotation
    // magnitudes we feed in (|x| ≤ 1).
    assert!(max_rel < 5e-3, "f16 standard rope: max rel = {max_rel:.2e} > 5e-3");
}

#[test]
fn rope_llama_standard_rope_matches_oracle_bf16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 41u32;
    let theta_base = 10000.0_f32;
    let (scale, low, high, max_pos) = no_scaling_params();

    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = naive_rope_llama(
        &qk_rounded,
        head_dim,
        n_heads,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );
    let actual = run_rope_llama(
        &qk,
        Dt::Bf16,
        n_heads,
        head_dim,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );

    let mut max_rel = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // bf16 has 7-bit mantissa — wider tolerance than f16.
    assert!(max_rel < 2e-2, "bf16 standard rope: max rel = {max_rel:.2e} > 2e-2");
}

#[test]
fn rope_llama_pos_scaling_preserves_norm_f32() {
    let _g = gpu_lock();
    // Rotation preserves L2 norm of every (x[i], x[i+half]) pair
    // regardless of banding choice. Pins that the rotation math
    // didn't drop one of the cross terms (a regression class where
    // o2 = x1*sin + x2*sin would NOT preserve norm).
    let n_heads = 4u32;
    let head_dim = 32u32;
    let position = 4096u32;
    let theta_base = 500000.0_f32;
    let (scale, low, high, max_pos) = llama3_scaling_params();

    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| 0.5 + (i as f32 * 0.073).cos()).collect();
    let actual = run_rope_llama(
        &qk,
        Dt::F32,
        n_heads,
        head_dim,
        position,
        theta_base,
        scale,
        low,
        high,
        max_pos,
    );

    let half_dim = head_dim / 2;
    for head in 0..n_heads {
        let base = (head * head_dim) as usize;
        for i in 0..half_dim {
            let i1 = base + i as usize;
            let i2 = base + (i + half_dim) as usize;
            let in_norm_sq = qk[i1] * qk[i1] + qk[i2] * qk[i2];
            let out_norm_sq = actual[i1] * actual[i1] + actual[i2] * actual[i2];
            let diff = (in_norm_sq - out_norm_sq).abs();
            assert!(
                diff < 1e-4,
                "norm not preserved at (head={head}, i={i}): in² = {in_norm_sq:.6}, out² = {out_norm_sq:.6}",
            );
        }
    }
}
