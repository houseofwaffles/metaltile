//! GPU correctness oracle for the MPP `matmul2d` smoke kernel.
//!
//! Dispatches `mt_mpp_matmul_smoke` (single threadgroup × single simdgroup,
//! 16×32 fp16 @ 32×16 fp16 → 16×16 fp32) and validates against a naïve
//! triple-loop CPU oracle.
//!
//! Requires macOS 26+ / Metal 4 — the kernel includes
//! `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls
//! `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`. The
//! smoke kernel emits a pre-Metal-4 fallback branch that copies a single
//! scalar instead so the metallib still links on older toolchains; on
//! such toolchains this test will fail the correctness check, which is
//! the intended signal.
//!
//! Run: `cargo test --release -p metaltile-std --test mpp_matmul_smoke -- --nocapture`

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_runtime::Context;
use metaltile_std::probe::mpp_matmul_smoke;

const M: usize = 16;
const N: usize = 16;
const K: usize = 32;

fn pack_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::f16::from_le_bytes(half::f16::from_f32(*v).to_le_bytes()).to_le_bytes())
        .collect()
}

fn unpack_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Round each value through fp16 — the MPP kernel reads `half` inputs.
fn round_f16(vals: &[f32]) -> Vec<f32> {
    vals.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect()
}

fn cpu_matmul_ref(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = acc;
        }
    }
    c
}

#[test]
fn mpp_matmul_smoke_matches_cpu_reference() {
    let _lock = gpu_lock();

    let ctx = Context::new().expect("Context::new");

    // MPP `tensor_ops::matmul2d` requires Apple10 (gen-17) + macOS 26.2+.
    // On older silicon or virtualised CI runners the kernel falls through
    // to the pre-Metal-4 stub branch which copies one scalar — the output
    // is mostly zero and the assertion would fail. Skip rather than fail
    // so coverage runners stay green; M5 Max + dev hardware still cover it.
    let family = ctx.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: mpp_matmul_smoke needs Apple10+ GPU (got chip_family={family:?})");
        return;
    }

    // Deterministic small-magnitude inputs in [-1, 1] so the fp16 round-trip
    // doesn't lose precision and so the dot-product accumulators stay well
    // inside fp32 finite range.
    let mut a_f32 = Vec::with_capacity(M * K);
    for i in 0..(M * K) {
        a_f32.push(((i as f32 * 0.137).sin()) * 0.5);
    }
    let mut b_f32 = Vec::with_capacity(K * N);
    for i in 0..(K * N) {
        b_f32.push(((i as f32 * 0.211).cos()) * 0.5);
    }

    // Round through fp16 BEFORE running the CPU reference so the oracle
    // sees the same load-quantised values the kernel does.
    let a_q = round_f16(&a_f32);
    let b_q = round_f16(&b_f32);
    let expected = cpu_matmul_ref(&a_q, &b_q);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("A".into(), pack_f16_bytes(&a_q));
    buffers.insert("B".into(), pack_f16_bytes(&b_q));
    buffers.insert("C".into(), vec![0u8; M * N * 4]);

    let kernel = mpp_matmul_smoke::kernel_ir();

    // 1 threadgroup × (32 threads = one simdgroup).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [32, 1, 1])
        .expect("dispatch mpp_matmul_smoke");

    let got = unpack_f32_bytes(result.outputs.get("C").expect("C buffer"));
    assert_eq!(got.len(), M * N);

    // fp16 inputs + fp32 accumulator, K=32 — accumulation error is bounded
    // by ~K * ULP(fp16) * |a|*|b|. With |a|,|b| ≤ 0.5 → tolerance 5e-3 is
    // comfortable; we use 1e-2 to also tolerate any minor NAX rounding.
    let mut max_err: f32 = 0.0;
    let mut max_idx = 0;
    for i in 0..(M * N) {
        let e = (got[i] - expected[i]).abs();
        if e > max_err {
            max_err = e;
            max_idx = i;
        }
    }
    println!(
        "max |Δ| = {:.4e} at idx {} (got={:.4}, expected={:.4})",
        max_err, max_idx, got[max_idx], expected[max_idx]
    );
    assert!(
        max_err < 1e-2,
        "MPP matmul smoke diverged from CPU reference: max err {:.4e} >= 1e-2.\n\
         If the toolchain is pre-Metal-4 the kernel falls into a stub branch \
         and this test is expected to fail — check `xcrun --sdk macosx metal --version` \
         and `sw_vers -productVersion` (need macOS 26+).",
        max_err
    );
}
