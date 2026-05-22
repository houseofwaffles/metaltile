//! GPU correctness test for `mlx::unary::mt_sigmoid_scalar_fma`.
//!
//! Fused scalar-sigmoid fan-out + FMA:
//!   `out[i] = base[i] + sigmoid(gate[0]) * value[i]`
//! across `i in 0..hidden`, broadcasting the scalar `gate` across the
//! `[hidden]` vectors. Replaces FFAI's Qwen3.5/3.6 shared-expert host
//! detour (`gateLogit.toFloatArray()` + host sigmoid + `Tensor.filled`
//! broadcast + mul + add) plus the `commit + waitUntilCompleted` that
//! the host scalar read required.
//!
//! Three correctness cells across the three shipped Qwen3 hybrid model
//! dtypes ({f32, f16, bf16}) at Qwen3.6-A3B's `hidden=2048`:
//!   - f32:  max |Δ| < 1e-5
//!   - f16:  max |Δ| < 5e-4
//!   - bf16: max |Δ| < 5e-3
//!
//! Plus a saturating-gate cell (large |gate| where sigmoid is near
//! 0 or 1) to make sure the fp32-internal accumulation path stays
//! correct when the model dtype would round the sigmoid scalar to
//! 0 or 1.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::unary::mt_sigmoid_scalar_fma;

/// CPU oracle. Mirrors the GPU pipeline:
///   - load `gate[0]` as fp32 (already rounded through dt via pack)
///   - sigmoid(gate) = 1 / (1 + exp(-gate))
///   - for each `i`, out[i] = base[i] + sigmoid(gate) * value[i],
///     final store rounded through dt.
fn oracle(gate: f32, value: &[f32], base: &[f32], dt: Dt) -> Vec<f32> {
    let g_quant = dt.round(gate);
    let s = 1.0 / (1.0 + (-g_quant).exp());
    let n = value.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let v = dt.round(value[i]);
        let b = dt.round(base[i]);
        out[i] = dt.round(b + s * v);
    }
    out
}

fn run_sigmoid_scalar_fma(gate: f32, value: &[f32], base: &[f32], dt: Dt) -> Vec<f32> {
    let n = value.len();
    assert_eq!(base.len(), n, "value and base lengths must match");

    let tgw = n.min(256);
    assert!(n.is_multiple_of(tgw), "test fixture: n must be multiple of tg ({tgw})");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // Metal requires a 4-byte minimum buffer allocation; bf16/f16 single
    // scalars are 2 bytes so pad with a second (unused) element. The
    // kernel reads `gate[0]` only, the trailing element is dead.
    buffers.insert("gate".into(), pack_bytes(&[gate, 0.0], dt));
    buffers.insert("value".into(), pack_bytes(value, dt));
    buffers.insert("base".into(), pack_bytes(base, dt));
    buffers.insert("out".into(), vec![0u8; n * dt.bytes()]);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_sigmoid_scalar_fma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Elementwise;

    let groups = n / tgw;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tgw, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` in dispatch result");
    unpack_bytes(out_bytes, dt)
}

fn run_cell(dt: Dt, gate: f32, n: usize, tol: f32, label: &str) {
    let _g = gpu_lock();
    let value = ramp(n, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect::<Vec<_>>();
    let base = ramp(n, 17, 8.0).iter().map(|v| 0.1 * v).collect::<Vec<_>>();

    let expected = oracle(gate, &value, &base, dt);
    let actual = run_sigmoid_scalar_fma(gate, &value, &base, dt);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "[{}] dt={:?} n={} gate={}: max |Δ| = {:.2e} (expected < {:.0e})",
        label,
        dt.to_dtype(),
        n,
        gate,
        diff,
        tol
    );
}

// Qwen3.6-A3B hidden=2048. Mid-range gate logits (sigmoid ≈ 0.5).

#[test]
fn sigmoid_scalar_fma_f32_qwen36() { run_cell(Dt::F32, 0.5, 2048, 1e-5, "qwen36"); }

#[test]
fn sigmoid_scalar_fma_f16_qwen36() { run_cell(Dt::F16, 0.5, 2048, 5e-4, "qwen36"); }

#[test]
fn sigmoid_scalar_fma_bf16_qwen36() { run_cell(Dt::Bf16, 0.5, 2048, 5e-3, "qwen36"); }

// Saturating-gate cells: gate=+8 and gate=-8 push sigmoid to ~1.0 / ~0.0.
// Tests that the internal fp32 accumulation stays correct when the dtype
// can't represent the sigmoid scalar precisely (bf16 7-bit mantissa rounds
// near-1.0 to exactly 1.0 — the oracle's `dt.round` handles that the same
// way the kernel's load-side `.cast` does, so the diff stays zero).

#[test]
fn sigmoid_scalar_fma_bf16_saturating_high() { run_cell(Dt::Bf16, 8.0, 2048, 5e-3, "sat_high"); }

#[test]
fn sigmoid_scalar_fma_bf16_saturating_low() { run_cell(Dt::Bf16, -8.0, 2048, 5e-3, "sat_low"); }
