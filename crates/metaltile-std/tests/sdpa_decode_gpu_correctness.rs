//! End-to-end correctness test for the rewritten `ffai::sdpa_decode`
//! kernel: dispatch on the real Metal pipeline and compare against a
//! straight-translation CPU reference.
//!
//! Validates the algorithm as a whole — proc-macro → IR → MSL → PSO
//! → dispatch → readback. Smoke tests in `sdpa_decode.rs` only check
//! that the emitted MSL contains the right primitives; this test
//! catches end-to-end mismatches the smoke tests can't see (wrong
//! threadgroup layout, off-by-one in indexing, miscomputed rescale,
//! lost simdgroup contributions, etc.).
//!
//! Test shape is intentionally small (n_q_heads=2, n_kv_heads=1,
//! head_dim=128, n_kv=4, kv_stride=4) so the CPU reference runs
//! instantly + the comparison is easy to eyeball. The kernel is
//! hardcoded to head_dim=128, so this test pins that path.
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{
    Dt,
    SdpaShape,
    gpu_lock,
    naive_sdpa_f32,
    naive_sdpa_swa_f32,
    pack_bytes,
    ramp,
    unpack_bytes,
};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode::sdpa_decode;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

#[allow(clippy::too_many_arguments)]
fn run_sdpa_decode_f32(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    sink_end: usize,
    window_start: usize,
    scale: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(q));
    buffers.insert("k".into(), f32_slice_to_bytes(k));
    buffers.insert("v".into(), f32_slice_to_bytes(v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("sink_end".into(), (sink_end as u32).to_le_bytes().to_vec());
    buffers.insert("window_start".into(), (window_start as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output element count");
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

#[test]
fn sdpa_decode_matches_naive_cpu_reference_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 4usize;
    let kv_stride = 4usize; // equal to n_kv for this test (no slack capacity)
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = naive_sdpa_f32(&q, &k, &v, &shape);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    // `kernel_ir_for` returns the kernel with its default
    // `KernelMode::Elementwise`; sdpa_decode needs Reduction-mode
    // codegen (mirrors what `tile bench` does for its SDPA path).
    let mut kernel = sdpa_decode::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // Dense path: sink_end = 0, window_start = 0 → the sink loop body
    // never emits, the window loop walks the full [0, n_kv) range
    // exactly as the pre-SWA kernel did. This test pins the dense
    // behavior is bit-identical to the original.
    let actual = run_sdpa_decode_f32(
        &ctx,
        &kernel,
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        0,
        0,
        scale,
    );

    // Tolerance: 1e-4 covers fp32 accumulation noise + `exp` ulp drift.
    // The kernel and the CPU reference both run in fp32 throughout, so
    // worst-case divergence is from `simd_sum` reordering of the score
    // partial — bounded by a few ulp at the magnitudes we use here.
    assert_close(&actual, &expected, 1e-4, "sdpa_decode dense vs CPU naive");
}

// Sliding-window + sink-token correctness: pins the kernel against a
// CPU reference applying the same mask. Shape stays small (n_kv=16) so
// both the sink and window passes contribute multiple positions while
// keeping the CPU naive eyeball-able.
#[test]
fn sdpa_decode_swa_matches_naive_cpu_reference_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 16usize;
    let kv_stride = 16usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    // Sinks = first 2, sliding window = last 8. Masked range
    // [2, 8) — non-trivial gap that exercises the bound split.
    let sink_end = 2usize;
    let window_start = 8usize;

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = naive_sdpa_swa_f32(&q, &k, &v, &shape, sink_end, window_start);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let actual = run_sdpa_decode_f32(
        &ctx,
        &kernel,
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        sink_end,
        window_start,
        scale,
    );

    assert_close(&actual, &expected, 1e-4, "sdpa_decode SWA vs CPU naive");
}

// Edge: sliding window with zero sinks — the deployment shape used by
// the industry SWA-only configs (window=4096, no attention sinks).
// `sink_end = 0` collapses the sink loop; only the window pass runs.
#[test]
fn sdpa_decode_swa_no_sinks_matches_cpu_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 16usize;
    let kv_stride = 16usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let sink_end = 0usize;
    let window_start = 8usize;

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = naive_sdpa_swa_f32(&q, &k, &v, &shape, sink_end, window_start);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let actual = run_sdpa_decode_f32(
        &ctx,
        &kernel,
        &q,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        sink_end,
        window_start,
        scale,
    );

    assert_close(&actual, &expected, 1e-4, "sdpa_decode SWA (no sinks) vs CPU naive");
}

// ── Perf bench ───────────────────────────────────────────────────────────
//
// Ignored by default (`cargo test` skips). Run manually to refresh the
// numbers cited in the PR body / commit history:
//
//   cargo test --release -p metaltile-std --test sdpa_decode_gpu_correctness \
//     -- --ignored --nocapture
//
// Reports median GB/s over 100 measured iterations (20 warmup) per shape,
// computed from `DispatchResult.elapsed_us` (GPU time, not wall time).
// Bandwidth model: bytes/iter = sizeof(q) + sizeof(k) + sizeof(v) + sizeof(out).

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_perf_bench_f32() {
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    // (n_q_heads, n_kv_heads, n_kv) — Qwen3-class GQA shapes.
    let shapes = [
        (32, 8, 128usize), // short context
        (32, 8, 1024),     // medium
        (32, 8, 4096),     // common decode
        (32, 8, 16384),    // long context
    ];

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    println!();
    println!("sdpa_decode f32 perf — Apple M5 Max (median of 100 iters)");
    println!("  {:>4} {:>4} {:>6}  {:>10}  {:>9}", "nQH", "nKVH", "n_kv", "GPU µs", "GB/s");
    for (n_q_heads, n_kv_heads, n_kv) in shapes {
        let kv_stride = n_kv;
        let heads_per_group = n_q_heads / n_kv_heads;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("q".into(), f32_slice_to_bytes(&q));
        buffers.insert("k".into(), f32_slice_to_bytes(&k));
        buffers.insert("v".into(), f32_slice_to_bytes(&v));
        buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * 4]);
        buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
        buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
        buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
        buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
        buffers.insert("sink_end".into(), 0u32.to_le_bytes().to_vec());
        buffers.insert("window_start".into(), 0u32.to_le_bytes().to_vec());
        buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

        // 20 warmup + 100 measure.
        let mut samples = Vec::with_capacity(100);
        for i in 0..120 {
            let r = ctx
                .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [
                    1024, 1, 1,
                ])
                .expect("dispatch_with_grid should succeed");
            if i >= 20 {
                samples.push(r.elapsed_us);
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = samples[samples.len() / 2];

        let bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim) * 4;
        let gbps = (bytes as f64) / (median_us * 1e-6) / 1e9;

        println!(
            "  {:>4} {:>4} {:>6}  {:>10.2}  {:>9.1}",
            n_q_heads, n_kv_heads, n_kv, median_us, gbps,
        );
    }
}
