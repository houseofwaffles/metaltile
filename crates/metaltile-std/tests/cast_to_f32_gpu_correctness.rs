//! GPU correctness test for `mlx::unary::mt_cast_to_f32` across dtypes.
//!
//! Per-element T → fp32 cast kernel. Used by FFAI's fused GDN prep
//! step to bridge bf16 model activations into the fp32 recurrence
//! pipeline on the GPU without a host round-trip.
//!
//! Three correctness cells:
//!  - `mt_cast_to_f32_bf16`: bit-correct against the CPU bf16-to-f32
//!    cast (truncate trailing 16 bits into a `f32` bit pattern).
//!  - `mt_cast_to_f32_f16`:  cosine ≥ 1 - 1e-6 against `half::f16` round
//!    trip (the kernel goes through `metal::float(half)` which is the
//!    canonical Metal fp16→fp32 widening — exact on every value).
//!  - `mt_cast_to_f32_f32`:  identity (output bit-equal to input).
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::pack_bytes;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::unary::mt_cast_to_f32;

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Dispatch the cast kernel for `dt` over `n` elements (input bytes
/// already packed in the source dtype). Returns the output as fp32.
fn run_cast(input_bytes: &[u8], n: usize, dt: DType) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), input_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; n * 4]);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let kernel = mt_cast_to_f32::kernel_ir_for(dt);

    // `dispatch_with_grid` counts THREADGROUPS, not threads — `n / tg`
    // groups × `tg` threads per group covers all elements exactly. The
    // kernel is Elementwise (`program_id(0)` per thread); callers must
    // ensure `n % tg == 0` (the test fixtures pick `n` as a multiple of
    // 256 to satisfy this).
    let tg = n.min(256);
    assert!(n.is_multiple_of(tg), "test fixture: n ({n}) must be multiple of tg ({tg})");
    let groups = n / tg;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tg, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn mt_cast_to_f32_bf16_bit_correct() {
    // Deterministic spread across positive / negative / small / large /
    // subnormal-ish values. bf16 has 7-bit mantissa, so most values pass
    // through with bounded relative error; the CPU oracle goes through
    // the *same* truncation so the GPU↔CPU diff is identically zero.
    let n = 1024usize;
    let values_f32: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32) * 0.137 - (n as f32) * 0.068;
            x * (1.0 + ((i % 5) as f32) * 0.01)
        })
        .collect();

    // CPU oracle: round through bf16 the same way `pack_bytes` does
    // (`half::bf16::from_f32` is the canonical round-to-nearest-even
    // formula). Metal's `bfloat → float` widening is exact mantissa
    // zero-extension, so the GPU↔CPU diff is identically zero.
    let expected: Vec<f32> = values_f32.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();

    let input_bytes = pack_bytes(&values_f32, common::Dt::Bf16);
    let actual = run_cast(&input_bytes, n, DType::BF16);
    assert_eq!(actual.len(), expected.len());
    let mut max_err = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..n {
        let e = (actual[i] - expected[i]).abs();
        if e > max_err {
            max_err = e;
            max_at = i;
        }
    }
    assert!(
        max_err == 0.0,
        "bf16 cast must be bit-equal to CPU oracle (max |Δ| = {max_err:.2e} at idx {max_at}, \
         expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_cast_to_f32_f16_round_trip() {
    let n = 1024usize;
    let values_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.041 - (n as f32) * 0.020).collect();
    // CPU oracle: f32 → f16 (via half::f16::from_f32 RNE) → f32.
    // Metal's `float(half)` widening is exact; the GPU↔CPU diff is
    // exactly zero on every element.
    let expected: Vec<f32> = values_f32.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let input_bytes = pack_bytes(&values_f32, common::Dt::F16);
    let actual = run_cast(&input_bytes, n, DType::F16);
    let mut max_err = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..n {
        let e = (actual[i] - expected[i]).abs();
        if e > max_err {
            max_err = e;
            max_at = i;
        }
    }
    assert!(
        max_err == 0.0,
        "f16 cast must be bit-equal to CPU oracle (max |Δ| = {max_err:.2e} at idx {max_at})",
    );
}

#[test]
fn mt_cast_to_f32_f32_identity() {
    let n = 256usize;
    let values_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let input_bytes = pack_bytes(&values_f32, common::Dt::F32);
    let actual = run_cast(&input_bytes, n, DType::F32);
    for i in 0..n {
        assert_eq!(
            actual[i].to_bits(),
            values_f32[i].to_bits(),
            "f32 cast must be bit-identity at idx {i}"
        );
    }
}
