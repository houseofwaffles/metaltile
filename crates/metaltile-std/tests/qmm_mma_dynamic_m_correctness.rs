//! GPU correctness oracle for the dynamic-M qmm path
//! (`metaltile_std::mlx::quantized_mma_dynamic_m`). This is the
//! bandwidth-bound prefill unlock for FFAI's `Qwen35Model.forwardMany`:
//! a single dispatch handles any `T` token count (rounded up to a
//! multiple of 32 via X-row zero padding) so int4 weights are read
//! once per layer-projection per chunk instead of `T` times.
//!
//! The driver pads `T → m_padded = ceil(T/32) * 32`, dispatches
//! `mt_qmm_mma` with grid `[N/32, m_padded/32, 1]`, then slices
//! the first `T` rows of the output. Padded rows of X are zero so
//! the masked tail contributes nothing to the valid outputs (and is
//! discarded by the caller anyway).
//!
//! Coverage matrix:
//!   - f16 small T=1 (decode shape)
//!   - f16 T=8 (small batch)
//!   - f16 T=64 (chunk-friendly, Qwen3.6 qkv shape)
//!   - f16 T=1000 (long prefill)
//!   - bf16 T=4096 (production prefill — N=2048, K=2048)
//!   - f32 T=32 (reference)
//!   - ragged T=37 (not a multiple of 32 — exercises padding)
//!
//! Run:
//!   cargo test --release -p metaltile-std --test qmm_mma_dynamic_m_correctness -- --nocapture

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized_mma_dynamic_m as dyn_m;

// ── Triple-loop CPU oracle — bit-identical algorithm to ──────────────────
//    `cpu_qmm_reference` in `qmm_gpu_correctness.rs`. Replicated for
//    test-file isolation per integration-test convention in this crate.

#[allow(clippy::too_many_arguments)]
fn cpu_qmm_reference(
    w: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gs_per_row {
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let mut q_dot = 0.0f32;
                let mut x_sum = 0.0f32;
                for p in 0..8usize {
                    let packed = w[n_col * k / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let q = ((packed >> (bit * 4)) & 0xF) as f32;
                        let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                        q_dot += q * xv;
                        x_sum += xv;
                    }
                }
                acc += s * q_dot + bias * x_sum;
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

// ── Host-side dispatcher that exercises the dynamic-M path. ───────────────
//
// Pads X (zero-fill) to m_padded = ceil(T/32)*32, dispatches
// `mt_qmm_mma` with grid `[N/32, m_padded/32, 1]`, then slices the
// first `T * N` element-bytes of the output.

#[allow(clippy::too_many_arguments)]
fn run_dynamic_m(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    t: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(n.is_multiple_of(32), "n must be multiple of 32 (BN tile)");
    assert!(k.is_multiple_of(32), "k must be multiple of 32 (BK step)");

    let m_padded = dyn_m::pad_t_to_bm(t);
    let padded_x = dyn_m::pad_x_rows_bytes(x_bytes, t, k, out_bytes_per_elem);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), padded_x);
    buffers.insert("out".into(), vec![0u8; m_padded * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let kernel = dyn_m::kernel_ir_for(dtype);
    let grid = dyn_m::dispatch_grid(t, n);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, [128, 1, 1])
        .expect("dispatch dynamic-M qmm");
    let out_padded = result.outputs.get("out").expect("`out` buffer").clone();
    // Slice the first T rows — discard trailing m_padded - T padding rows.
    out_padded[..(t * n * out_bytes_per_elem)].to_vec()
}

// ── Dtype byte helpers. ─────────────────────────────────────────────────

fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}
fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect()
}
fn round_f16(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }
fn round_bf16(v: f32) -> f32 { half::bf16::from_f32(v).to_f32() }

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

// ── Deterministic q4 weights — same per-pack pattern as the other ────────
//    qmm correctness tests (`qmm_mpp_correctness.rs`, `qmm_gpu_correctness.rs`).

fn build_quant_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    (w, scales, biases, x)
}

// ── Smaller-magnitude inputs for large-K production-shape tests where ────
//    K=2048 accumulation in low-mantissa dtypes (bf16) blows up otherwise.

fn build_quant_inputs_small_mag(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<u32> =
        (0..n * k / 8).map(|i| ((i as u32) % 17).wrapping_mul(0x12345678u32)).collect();
    // Smaller scales/biases to keep CPU oracle accumulation manageable.
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.005 + ((i % 7) as f32) * 0.0007).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| ((i % 5) as f32) * 0.00005).collect();
    // Bounded x in [0.05, 0.116] — keeps Σ q·x within bf16 range at K=2048.
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 + ((i % 23) as f32) * 0.003).collect();
    (w, scales, biases, x)
}

// ═════════════════════════════════════════════════════════════════════════
// Case 1: f16 small T=1 (decode shape).
//
// Padded m=32, N=128, K=128 = 2 groups. Exercises the path that any
// FFAI decode step would hit — single token in, full BM tile padded.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f16_t1_decode_shape() {
    let t = 1usize;
    let n = 128usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=1 row count");
    let cos = cosine(&expected, &actual);
    println!("[f16 T=1 decode] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=1)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 2: f16 T=8 small batch.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f16_t8_small_batch() {
    let t = 8usize;
    let n = 128usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[f16 T=8 small-batch] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=8)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 3: f16 T=64 chunk-friendly (Qwen3.6 qkv shape — N=512, K=2048).
//
// Exact multiple of BM=32, so no padding rows. Production-realistic
// shape: the q_proj of Qwen3.6-A3B at hidden=2048 lands one of N=2048
// (full q) or N=512 (kv). We pick N=512 to keep CPU oracle reasonable.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f16_t64_qkv_shape() {
    let t = 64usize;
    let n = 512usize;
    let k = 2048usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs_small_mag(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[f16 T=64 qkv N={n} K={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=64)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 4: f16 T=1000 long prefill.
//
// T=1000 → m_padded = 1024 (24 padding rows). Same N/K as Case 3.
// Validates that batched prefill at realistic chunk sizes preserves
// numerics across many tiles.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f16_t1000_long_prefill() {
    let t = 1000usize;
    let n = 512usize;
    let k = 2048usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs_small_mag(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=1000 row slice length");
    let cos = cosine(&expected, &actual);
    println!("[f16 T=1000 long-prefill N={n} K={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=1000)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 5: bf16 T=4096 production prefill (N=2048, K=2048).
//
// **This is the validation that matters.** Qwen3.6-A3B T=4K prefill
// cell at the o_proj / mlp_down shape (hidden=2048). bf16 is the
// production dtype. cos ≥ 0.999 confirms the dynamic-M path is sound
// at production scale + dtype.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_bf16_t4096_production() {
    let t = 4096usize;
    let n = 2048usize;
    let k = 2048usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs_small_mag(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::BF16,
        &w,
        &f32_to_bf16_bytes(&scales),
        &f32_to_bf16_bytes(&biases),
        &f32_to_bf16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let cos = cosine(&expected, &actual);
    println!("[bf16 T=4096 prod N={n} K={k}] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (bf16 T=4096 production)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 6: f32 T=32 reference.
//
// Exactly one BM tile, no padding, fp32 (no dtype rounding error).
// Sanity check that the simple case matches the standalone
// `mt_qmm_mma` path.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f32_t32_reference() {
    let t = 32usize;
    let n = 64usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_quant_inputs(t, n, k, gs_per_row);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&biases),
        &f32_to_f32_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len(), "T=32 element count");
    let cos = cosine(&expected, &actual);
    let mut max_diff = 0.0f32;
    for (e, a) in expected.iter().zip(actual.iter()) {
        max_diff = max_diff.max((e - a).abs());
    }
    println!("[f32 T=32 ref] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 T=32)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 7: ragged T=37 (not a multiple of 32).
//
// Padded m=64 (27 padding rows). Validates the slice step — the
// returned `T * N` slice must contain only the valid rows, never
// the trailing zeros / undefined-grid output. This is the canonical
// case the per-token loop would hit in `forwardMany` at any non-32
// chunk boundary.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dynamic_m_f16_t37_ragged() {
    let t = 37usize;
    let n = 128usize;
    let k = 128usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_dynamic_m(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=37 sliced row count");
    let cos = cosine(&expected, &actual);
    println!("[f16 T=37 ragged → m_padded=64] cos={cos:.6}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=37 ragged)");
}
