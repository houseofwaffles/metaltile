//! End-to-end correctness for `mt_swiglu` — fused `silu(gate) * up`.
//!
//! Verifies the fused single-launch kernel against a CPU oracle that
//! computes `silu(gate) * up` element-wise. Tested across f32 / f16 /
//! bf16 dtypes and at Qwen3-MoE-realistic intermediate sizes.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::swiglu::mt_swiglu;

fn cpu_swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect()
}

fn run_swiglu(
    ctx: &Context,
    dtype: DType,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    n_elems: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("gate".into(), gate_bytes.to_vec());
    buffers.insert("up".into(), up_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; n_elems * elem_bytes]);

    let kernel = mt_swiglu::kernel_ir_for(dtype);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_elems, 1, 1], [256, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("out").clone()
}

#[test]
fn mt_swiglu_matches_cpu_reference_f32() {
    let n = 1024usize;
    // Mix of positive + negative + near-zero gate values to exercise
    // silu's full activation curve.
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017) % 6.0 - 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.029) % 4.0 - 2.0).collect();
    let expected = cpu_swiglu_reference(&gate, &up);

    let gate_bytes: Vec<u8> = gate.iter().flat_map(|v| v.to_le_bytes()).collect();
    let up_bytes: Vec<u8> = up.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let out_bytes = run_swiglu(&ctx, DType::F32, &gate_bytes, &up_bytes, n, 4);
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    // silu uses fast exp; tolerance covers ~3 ULP drift in f32.
    assert!(
        max_diff < 1e-5,
        "swiglu f32: max |diff| = {max_diff:.2e} at [{max_at}] (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_swiglu_matches_cpu_reference_f16() {
    let n = 2048usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013) % 8.0 - 4.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.021) % 3.0 - 1.5).collect();
    // Round through f16 so the oracle sees the same input precision.
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let gate: Vec<f32> = gate_f32.iter().map(|&v| round_f16(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_swiglu_reference(&gate, &up);

    let gate_bytes: Vec<u8> =
        gate_f32.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let up_bytes: Vec<u8> =
        up_f32.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_swiglu(&ctx, DType::F16, &gate_bytes, &up_bytes, n, 2);
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let mut max_rel = 0.0f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    // f16 silu + mul stacks ~2 ULPs of rounding drift.
    assert!(
        max_rel < 5e-3,
        "swiglu f16: max rel = {max_rel:.2e} at [{max_at}] (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_swiglu_qwen3_moe_intermediate_shape_f16() {
    // Qwen3-MoE per-expert intermediate size = 768. Each MLP applies
    // SwiGLU to [B*T, intermediate] gate/up tensors. Use n = 768 * 8
    // (8 tokens at one expert's slot).
    let n = 768 * 8;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0091) % 5.0 - 2.5).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0073) % 4.0 - 2.0).collect();
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let gate: Vec<f32> = gate_f32.iter().map(|&v| round_f16(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_swiglu_reference(&gate, &up);

    let gate_bytes: Vec<u8> =
        gate_f32.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let up_bytes: Vec<u8> =
        up_f32.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_swiglu(&ctx, DType::F16, &gate_bytes, &up_bytes, n, 2);
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let mut max_rel = 0.0f32;
    for (e, a) in expected.iter().zip(actual.iter()) {
        let rel = (e - a).abs() / e.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(max_rel < 5e-3, "swiglu Qwen3-MoE shape f16: max rel {max_rel:.2e}");
}

#[test]
fn mt_swiglu_matches_cpu_reference_bf16() {
    let n = 1024usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019) % 6.0 - 3.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.023) % 4.0 - 2.0).collect();
    let to_bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let from_bf16 = |b: u16| -> f32 { f32::from_bits((b as u32) << 16) };
    let gate: Vec<f32> = gate_f32.iter().map(|&v| from_bf16(to_bf16(v))).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| from_bf16(to_bf16(v))).collect();
    let expected = cpu_swiglu_reference(&gate, &up);

    let gate_bytes: Vec<u8> = gate_f32.iter().flat_map(|v| to_bf16(*v).to_le_bytes()).collect();
    let up_bytes: Vec<u8> = up_f32.iter().flat_map(|v| to_bf16(*v).to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_swiglu(&ctx, DType::BF16, &gate_bytes, &up_bytes, n, 2);
    let actual: Vec<f32> =
        out_bytes.chunks_exact(2).map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]]))).collect();
    let mut max_rel = 0.0f32;
    for (e, a) in expected.iter().zip(actual.iter()) {
        let rel = (e - a).abs() / e.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // bf16 has 7-bit mantissa — wider tolerance than f16.
    assert!(max_rel < 2e-2, "swiglu bf16: max rel {max_rel:.2e}");
}
