//! GPU correctness test for `mlx::rms_norm::mt_gated_mixer_norm`.
//!
//! Fused `out = rms_norm(y, w) · silu(z)` per `[Hv, Dv]` row. Replaces
//! the host-loop phase 2 of `Qwen35GDNMixer.forward` (drops one
//! commit + waitUntilCompleted + 3 buffer toFloatArray copies per GDN
//! layer per token). Inputs: `y` fp32, `z` / `w` in T (typically bf16
//! on Qwen3.6); output in T.
//!
//! Three correctness cells across the three shipped Qwen3 hybrid head
//! dims (Dv ∈ {128, 256}) and dtypes ({f32, f16, bf16}):
//!   - bf16: max |Δ| < 5e-3 against the f32 oracle round-tripped
//!     through bf16 loads (Metal `bfloat → float` widening is exact +
//!     the same RNE truncation `half::bf16::from_f32` does, so the
//!     load is bit-equal; the diff is just the bf16 storage error).
//!   - f16:  max |Δ| < 5e-4.
//!   - f32:  max |Δ| < 1e-5.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_gated_mixer_norm;

/// CPU oracle. Mirrors the GPU pipeline: load y as fp32, load z/w
/// quantised through dt (`Dt::round`), compute rms over Dv lane, fuse
/// silu(z) · norm(y, w), round result back through dt for the store.
fn oracle(y: &[f32], z: &[f32], w: &[f32], hv: usize, dv: usize, eps: f32, dt: Dt) -> Vec<f32> {
    let mut out = vec![0.0f32; hv * dv];
    for h in 0..hv {
        let base = h * dv;
        let mut ssq = 0.0f32;
        for i in 0..dv {
            let v = y[base + i];
            ssq += v * v;
        }
        let inv = 1.0 / (ssq / (dv as f32) + eps).sqrt();
        for i in 0..dv {
            let normed = y[base + i] * inv * dt.round(w[i]);
            let zq = dt.round(z[base + i]);
            let silu = zq / (1.0 + (-zq).exp());
            out[base + i] = dt.round(normed * silu);
        }
    }
    out
}

fn run_gated_mixer_norm(
    y: &[f32],
    z: &[f32],
    w: &[f32],
    hv: usize,
    dv: usize,
    eps: f32,
    dt: Dt,
) -> Vec<f32> {
    let tpg = dv / 4; // N = TPG * 4 invariant (same as mt_rms_norm).
    assert!(dv.is_multiple_of(128), "dv must be multiple of 128");
    assert!(tpg <= 1024, "dv / 4 must fit in 1024");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("y".into(), pack_bytes(y, Dt::F32));
    buffers.insert("z".into(), pack_bytes(z, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("out".into(), vec![0u8; hv * dv * dt.bytes()]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (dv as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_gated_mixer_norm::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [hv, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` in dispatch result");
    unpack_bytes(out_bytes, dt)
}

fn run_cell(dt: Dt, hv: usize, dv: usize, tol: f32) {
    let _g = gpu_lock();
    let eps = 1e-5f32;
    // y stays fp32 (matches GDN kernel output). z/w are quantised
    // through dt via pack_bytes, so the cpu side must match.
    let y = ramp(hv * dv, 17, 8.0).iter().map(|v| 0.1 * v).collect::<Vec<_>>();
    let z = ramp(hv * dv, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect::<Vec<_>>();
    let w = ramp(dv, 7, 3.0).iter().map(|v| 1.0 + 0.05 * v).collect::<Vec<_>>();

    let expected = oracle(&y, &z, &w, hv, dv, eps, dt);
    let actual = run_gated_mixer_norm(&y, &z, &w, hv, dv, eps, dt);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "gated_mixer_norm dt={:?} Hv={} Dv={}: max |Δ| = {:.2e} (expected < {:.0e})",
        dt.to_dtype(),
        hv,
        dv,
        diff,
        tol
    );
}

// Qwen3.6-A3B: Hv=32 value heads, Dv=128 value-head dim.

#[test]
fn gated_mixer_norm_f32_qwen36() { run_cell(Dt::F32, 32, 128, 1e-5); }

#[test]
fn gated_mixer_norm_f16_qwen36() { run_cell(Dt::F16, 32, 128, 5e-4); }

#[test]
fn gated_mixer_norm_bf16_qwen36() { run_cell(Dt::Bf16, 32, 128, 5e-3); }

// Wider Dv=256 cell exercises 2× larger reduction across the same TPG
// budget — pins the rms_norm reduction tree behaviour at Dv=256.

#[test]
fn gated_mixer_norm_bf16_dv256() { run_cell(Dt::Bf16, 8, 256, 5e-3); }
