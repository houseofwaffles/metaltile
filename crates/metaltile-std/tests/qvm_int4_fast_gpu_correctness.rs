//! GPU correctness for `mlx::quantized::mt_qvm_int4_fast` — perf-tuned
//! int4 vecmat `y = xᵀ · W`, W `[K, N]` row-major.
//!
//! CPU oracle: `y[c] = Σ_k (q[k,c]·scale_g + bias_g) · x[k]` where
//! scale/bias for K-position `k` at column `c` live at
//! `scales[g * N + c]`, `g = k / group_size` — the `[K/G, N]` layout
//! matching `mt_qvm_b4`.
//!
//! The fast variant (`mt_qvm_int4_fast`) uses 8-column-per-TG geometry:
//! 2 simdgroups × 4 output columns each, lane-strided over K. Grid:
//! `[N/8, 1, 1]`, TPG = 64. Requires N a multiple of 8, K a multiple
//! of 32, group_size = 64.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::mt_qvm_int4_fast;

/// Affine per-group int4 quantize of one row of W, groups along N-dimension.
/// W is `[K, N]`: each K-row is packed with 8 nibbles per u32 along N.
/// Returns (packed_row[N/8 u32s], scales[N/G], biases[N/G]).
fn quantize_krow(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let n = row.len();
    let n_groups = n / group_size;
    let mut packed = vec![0u32; n / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let gs = &row[g * group_size..(g + 1) * group_size];
        let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in gs.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
            let d = g * group_size + i;
            packed[d / 8] |= q << ((d % 8) * 4);
        }
    }
    (packed, scales, biases)
}

/// CPU oracle: `y[c] = Σ_k (q[k,c]·scale_g + bias_g) · x[k]`.
/// W `[K, N]` row-major; scales/biases `[K/G, N]` — scale for column `c`
/// at group `g` is at `scales[g * n + c]`.
fn oracle_vecmat(
    w_packed: &[u32], // [K, N/8] u32 row-major
    scales: &[f32],   // [K/G, N]
    biases: &[f32],   // [K/G, N]
    x: &[f32],        // [K]
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<f32> {
    let packs_per_krow = n / 8;
    let mut y = vec![0.0_f32; n];
    #[allow(clippy::needless_range_loop)]
    for kk in 0..k {
        let g = kk / group_size;
        let xk = x[kk];
        let row_base = kk * packs_per_krow;
        for c in 0..n {
            let pack = c / 8;
            let slot = c % 8;
            let q = ((w_packed[row_base + pack] >> (slot * 4)) & 0xf) as f32;
            let scale = scales[g * n + c];
            let bias = biases[g * n + c];
            y[c] += (q * scale + bias) * xk;
        }
    }
    y
}

fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
        })
        .collect()
}

/// Dispatch `mt_qvm_int4_fast` and return the output vector.
/// `k` = inner dimension (K), `n` = output columns (N).
/// Grid: `[N/8, 1, 1]`, TPG = 64.
/// Kernel constexpr args: `k`, `n`, `gs_per_col = k / group_size`.
#[allow(clippy::too_many_arguments)]
fn run(
    w: &[u32],      // [K, N/8] flattened
    scales: &[f32], // [K/G, N]
    biases: &[f32], // [K/G, N]
    x: &[f32],      // [K]
    dt: Dt,
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<f32> {
    let gs_per_col = k / group_size;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(w));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_col".into(), (gs_per_col as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_qvm_int4_fast::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [N/8, 1, 1]; TPG = 64 (2 SG × 32 lanes).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, 1, 1], [64, 1, 1])
        .expect("mt_qvm_int4_fast dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

fn run_case(dt: Dt, k: usize, n: usize, group_size: usize, tol: f32) {
    assert_eq!(k % 32, 0, "k must be a multiple of 32");
    assert_eq!(n % 8, 0, "n must be a multiple of 8");
    assert_eq!(group_size, 64, "fast variant requires group_size == 64");

    let _g = gpu_lock();

    let x_raw = source(k, 0x1A2B, 2.0, 0.05);
    let x: Vec<f32> = x_raw.iter().map(|&v| dt.round(v)).collect();

    // Build W [K, N] — one row per K-position.
    // Flatten into [K * N/8] u32 (packed nibbles along N).
    let w_rows_raw = source(k * n, 0x3C4D, 3.0, 0.0);
    let packs_per_krow = n / 8;
    let n_groups_k = k / group_size;
    let mut w_packed = vec![0u32; k * packs_per_krow];
    // scales/biases layout: [K/G, N] — for each K-group, N scales/biases.
    let mut scales = vec![0.0_f32; n_groups_k * n];
    let mut biases = vec![0.0_f32; n_groups_k * n];

    for kk in 0..k {
        let krow = &w_rows_raw[kk * n..(kk + 1) * n];
        let (pw, ps, pb) = quantize_krow(krow, group_size);
        w_packed[kk * packs_per_krow..(kk + 1) * packs_per_krow].copy_from_slice(&pw);
        // Map scales/biases from per-k-row to [K/G, N].
        let g = kk / group_size;
        let g_start = kk % group_size;
        // Only store on the first K-position of each group (the others would
        // overwrite with the same group's scale/bias from a different row,
        // but each K-row has its own per-N-group quant params here). Since
        // each K-row is independently quantized, the scale for column c at
        // K-position kk should be at scales[g * n + c]. For simplicity we
        // build the scale/bias from the per-K-row quantization and place it
        // at scales[(kk) * n + c]. However, the kernel expects [K/G, N]
        // with G rows sharing one scale — so we need to quantize in groups.
        // Re-quantize at the group granularity: for each K-group, combine all
        // K-rows in the group into a single (scale, bias) per N-column group.
        // For correctness in the test we use per-K-row quant and verify
        // group_size=1 would work, but since group_size=64 the kernel reads
        // scales[g * n + c]. We build the test to match: each K-group (64
        // K-rows) gets one scale/bias per N-group. To keep the test simple,
        // we set group_size=64 but within a K-group we use a single scale/bias
        // derived from the FIRST K-row in the group's range.
        if g_start == 0 {
            // Store per-N-column scales for this K-group.
            let n_groups_n = n / group_size;
            for ng in 0..n_groups_n {
                scales[g * n + ng * group_size] = ps[ng];
                biases[g * n + ng * group_size] = pb[ng];
                // Broadcast to all N positions in this N-group.
                for nc in 1..group_size {
                    scales[g * n + ng * group_size + nc] = ps[ng];
                    biases[g * n + ng * group_size + nc] = pb[ng];
                }
            }
        }
    }

    // The oracle must use the same [K/G, N] scale layout as the kernel.
    let scales_r: Vec<f32> = scales.iter().map(|&v| dt.round(v)).collect();
    let biases_r: Vec<f32> = biases.iter().map(|&v| dt.round(v)).collect();

    let expected = oracle_vecmat(&w_packed, &scales_r, &biases_r, &x, k, n, group_size);
    let actual = run(&w_packed, &scales, &biases, &x, dt, k, n, group_size);

    assert_eq!(actual.len(), n, "output length mismatch");
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");

    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel <= tol,
        "mt_qvm_int4_fast dt={:?} k={k} n={n}: max rel = {max_rel:.3e} > {tol:.3e}",
        dt as u32,
    );
}

#[test]
fn qvm_int4_fast_f32_k64_n64() {
    // Minimal shape: K=64 (2 groups of gs=64), N=64 (8 columns per TG).
    run_case(Dt::F32, 64, 64, 64, 5e-3);
}

#[test]
fn qvm_int4_fast_f16_k64_n64() { run_case(Dt::F16, 64, 64, 64, 2e-2); }

#[test]
fn qvm_int4_fast_bf16_k64_n64() { run_case(Dt::Bf16, 64, 64, 64, 5e-2); }

#[test]
fn qvm_int4_fast_f32_k128_n128() {
    // Larger: K=128 (2 K-groups), N=128 (16 col tiles).
    run_case(Dt::F32, 128, 128, 64, 5e-3);
}
