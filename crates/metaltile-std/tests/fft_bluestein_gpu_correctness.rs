//! GPU correctness for the Bluestein chirp-Z FFT — arbitrary-length DFT.
//!
//! Tests lengths 400 (Whisper standard) and 480 (Whisper large-v3 variant),
//! neither of which is a power of two.
//!
//! ## Pipeline per row
//!
//!   1. `mt_fft_bluestein_chirp_filter` — build time-domain chirp filter
//!      `[M]` (once per N). M = 1024 (next pow2 ≥ 2N for N=400,480).
//!   2. `mt_fft_n1024` (forward) — FFT of the chirp filter → frequency-
//!      domain filter `F[M]`.
//!   3. `mt_fft_bluestein_preprocess` — pre-multiply input `[rows, N]` by
//!      the forward chirp, zero-pad to `[rows, M]`.
//!   4. `mt_fft_n1024` (forward) — FFT of the padded signal → `Y[rows, M]`.
//!   5. `mt_fft_bluestein_cmul` — element-wise `Y *= F` (broadcast filter).
//!   6. `mt_fft_n1024` (inverse) — IFFT → circular convolution `[rows, M]`.
//!   7. `mt_fft_bluestein_postprocess` — post-multiply + extract `[rows, N]`.
//!
//! Compared against a naive O(N²) DFT reference. Tolerance: relative error
//! ≤ 1e-3 (f32). f16 / bf16 variants use looser tolerances.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::fft::{
    mt_fft_bluestein_chirp_filter,
    mt_fft_bluestein_cmul,
    mt_fft_bluestein_postprocess,
    mt_fft_bluestein_preprocess,
    mt_fft_n1024,
};

/// Bluestein M = 1024 (next power of two ≥ 2·max(N)). Both N=400 and N=480
/// satisfy 2N ≤ 1024.
const M: usize = 1024;

/// Naive O(N²) DFT reference.
fn naive_dft(re: &[f32], im: &[f32], n: usize, inv: bool) -> (Vec<f32>, Vec<f32>) {
    let rows = re.len() / n;
    let mut out_re = vec![0.0_f32; rows * n];
    let mut out_im = vec![0.0_f32; rows * n];
    let sign = if inv { 1.0_f32 } else { -1.0_f32 };
    let scale = if inv { 1.0_f32 / n as f32 } else { 1.0_f32 };
    for r in 0..rows {
        let base = r * n;
        for k in 0..n {
            let mut acc_re = 0.0_f32;
            let mut acc_im = 0.0_f32;
            for t in 0..n {
                let angle = sign * std::f32::consts::TAU * (k as f32) * (t as f32) / n as f32;
                let (s, c) = angle.sin_cos();
                let xr = re[base + t];
                let xi = im[base + t];
                acc_re += xr * c - xi * s;
                acc_im += xr * s + xi * c;
            }
            out_re[base + k] = acc_re * scale;
            out_im[base + k] = acc_im * scale;
        }
    }
    (out_re, out_im)
}

/// Dispatch the f32-only `mt_fft_bluestein_chirp_filter` kernel.
fn dispatch_chirp_filter(ctx: &Context, n: usize, m: usize) -> (Vec<f32>, Vec<f32>) {
    let mut bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bufs.insert("filter_re".into(), vec![0u8; m * 4]);
    bufs.insert("filter_im".into(), vec![0u8; m * 4]);
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    bufs.insert("n_len".into(), u(n));
    bufs.insert("m_len".into(), u(m));

    // chirp_filter is not generic — kernel_ir_for() takes no DType arg.
    let mut k = mt_fft_bluestein_chirp_filter::kernel_ir_for();
    k.mode = KernelMode::Grid3D;
    let tpg = 256usize;
    let grid = m.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&k, &bufs, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("chirp_filter dispatch");
    let re: Vec<f32> = bytemuck::cast_slice(result.outputs.get("filter_re").unwrap()).to_vec();
    let im: Vec<f32> = bytemuck::cast_slice(result.outputs.get("filter_im").unwrap()).to_vec();
    (re, im)
}

/// Run the full Bluestein pipeline for `rows` rows of length-`n` input.
/// Returns (out_re, out_im) of shape [rows * n].
fn run_bluestein(
    in_re: &[f32],
    in_im: &[f32],
    dt: Dt,
    n: usize,
    rows: usize,
    inv: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(in_re.len(), rows * n);
    assert_eq!(in_im.len(), rows * n);

    let ctx = Context::new().expect("Context::new on macOS");

    // ── Step 1: build chirp filter in the time domain ─────────────────────
    let (chirp_filter_re, chirp_filter_im) = dispatch_chirp_filter(&ctx, n, M);

    // ── Step 2: FFT the chirp filter (f32, forward) ───────────────────────
    // Filter is [1, M]; run as 1 row.
    let inv_u32_bytes = (0u32).to_le_bytes().to_vec();
    let zero_row: Vec<u8> = vec![0u8; M * 4];

    let mut fft_filter_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    fft_filter_bufs.insert("in_re".into(), bytemuck::cast_slice(&chirp_filter_re).to_vec());
    fft_filter_bufs.insert("in_im".into(), bytemuck::cast_slice(&chirp_filter_im).to_vec());
    fft_filter_bufs.insert("out_re".into(), zero_row.clone());
    fft_filter_bufs.insert("out_im".into(), zero_row.clone());
    fft_filter_bufs.insert("inv".into(), inv_u32_bytes.clone());

    let mut k_fft = mt_fft_n1024::kernel_ir_for(DType::F32);
    k_fft.mode = KernelMode::Reduction;
    let fft_filter_result = ctx
        .dispatch_with_grid(&k_fft, &fft_filter_bufs, &BTreeMap::new(), [1, 1, 1], [M, 1, 1])
        .expect("chirp filter fft");
    let f_re: Vec<f32> =
        bytemuck::cast_slice(fft_filter_result.outputs.get("out_re").unwrap()).to_vec();
    let f_im: Vec<f32> =
        bytemuck::cast_slice(fft_filter_result.outputs.get("out_im").unwrap()).to_vec();

    // ── Step 3: pre-multiply + zero-pad (bluestein_preprocess) ───────────
    let inv_const = if inv { 1u32 } else { 0u32 };
    let inv_bytes = inv_const.to_le_bytes().to_vec();

    let pre_n_out = rows * M;
    let mut pre_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pre_bufs.insert("in_re".into(), pack_bytes(in_re, dt));
    pre_bufs.insert("in_im".into(), pack_bytes(in_im, dt));
    pre_bufs.insert("out_re".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    pre_bufs.insert("out_im".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    pre_bufs.insert("n_len".into(), u(n));
    pre_bufs.insert("m_len".into(), u(M));
    pre_bufs.insert("inv".into(), inv_bytes.clone());

    let mut k_pre = mt_fft_bluestein_preprocess::kernel_ir_for(dt.to_dtype());
    k_pre.mode = KernelMode::Grid3D;
    let tpg = 256usize;
    let pre_grid = pre_n_out.div_ceil(tpg);
    let pre_result = ctx
        .dispatch_with_grid(&k_pre, &pre_bufs, &BTreeMap::new(), [pre_grid, 1, 1], [tpg, 1, 1])
        .expect("preprocess dispatch");
    let pre_re_bytes = pre_result.outputs.get("out_re").unwrap().clone();
    let pre_im_bytes = pre_result.outputs.get("out_im").unwrap().clone();

    // ── Step 4: FFT the padded rows ────────────────────────────────────────
    let mut fft_in_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    fft_in_bufs.insert("in_re".into(), pre_re_bytes.clone());
    fft_in_bufs.insert("in_im".into(), pre_im_bytes.clone());
    // Output at same dtype; zero-init.
    fft_in_bufs.insert("out_re".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    fft_in_bufs.insert("out_im".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    fft_in_bufs.insert("inv".into(), inv_u32_bytes.clone());

    let mut k_fft_in = mt_fft_n1024::kernel_ir_for(dt.to_dtype());
    k_fft_in.mode = KernelMode::Reduction;
    let fft_in_result = ctx
        .dispatch_with_grid(&k_fft_in, &fft_in_bufs, &BTreeMap::new(), [rows, 1, 1], [M, 1, 1])
        .expect("fft_in dispatch");
    let y_re_bytes = fft_in_result.outputs.get("out_re").unwrap().clone();
    let y_im_bytes = fft_in_result.outputs.get("out_im").unwrap().clone();

    // ── Step 5: element-wise complex multiply Y *= F ──────────────────────
    let mut cmul_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    cmul_bufs.insert("y_re".into(), y_re_bytes);
    cmul_bufs.insert("y_im".into(), y_im_bytes);
    // Filter is f32; broadcast from [1, M] across rows.
    cmul_bufs.insert("filter_re".into(), bytemuck::cast_slice(&f_re).to_vec());
    cmul_bufs.insert("filter_im".into(), bytemuck::cast_slice(&f_im).to_vec());
    cmul_bufs.insert("out_re".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    cmul_bufs.insert("out_im".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    cmul_bufs.insert("m_len".into(), u(M));

    let mut k_cmul = mt_fft_bluestein_cmul::kernel_ir_for(dt.to_dtype());
    k_cmul.mode = KernelMode::Grid3D;
    let cmul_grid = pre_n_out.div_ceil(tpg);
    let cmul_result = ctx
        .dispatch_with_grid(&k_cmul, &cmul_bufs, &BTreeMap::new(), [cmul_grid, 1, 1], [tpg, 1, 1])
        .expect("cmul dispatch");
    let cy_re_bytes = cmul_result.outputs.get("out_re").unwrap().clone();
    let cy_im_bytes = cmul_result.outputs.get("out_im").unwrap().clone();

    // ── Step 6: IFFT of the convolution (inv=1) ───────────────────────────
    let mut ifft_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    ifft_bufs.insert("in_re".into(), cy_re_bytes);
    ifft_bufs.insert("in_im".into(), cy_im_bytes);
    ifft_bufs.insert("out_re".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    ifft_bufs.insert("out_im".into(), pack_bytes(&vec![0.0f32; pre_n_out], dt));
    ifft_bufs.insert("inv".into(), (1u32).to_le_bytes().to_vec()); // inverse

    let mut k_ifft = mt_fft_n1024::kernel_ir_for(dt.to_dtype());
    k_ifft.mode = KernelMode::Reduction;
    let ifft_result = ctx
        .dispatch_with_grid(&k_ifft, &ifft_bufs, &BTreeMap::new(), [rows, 1, 1], [M, 1, 1])
        .expect("ifft dispatch");
    let conv_re_bytes = ifft_result.outputs.get("out_re").unwrap().clone();
    let conv_im_bytes = ifft_result.outputs.get("out_im").unwrap().clone();

    // ── Step 7: post-multiply and extract N outputs ───────────────────────
    let n_out_total = rows * n;
    let mut post_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    post_bufs.insert("conv_re".into(), conv_re_bytes);
    post_bufs.insert("conv_im".into(), conv_im_bytes);
    post_bufs.insert("out_re".into(), pack_bytes(&vec![0.0f32; n_out_total], dt));
    post_bufs.insert("out_im".into(), pack_bytes(&vec![0.0f32; n_out_total], dt));
    post_bufs.insert("n_len".into(), u(n));
    post_bufs.insert("m_len".into(), u(M));
    post_bufs.insert("inv".into(), inv_bytes.clone());

    let mut k_post = mt_fft_bluestein_postprocess::kernel_ir_for(dt.to_dtype());
    k_post.mode = KernelMode::Grid3D;
    let post_grid = n_out_total.div_ceil(tpg);
    let post_result = ctx
        .dispatch_with_grid(&k_post, &post_bufs, &BTreeMap::new(), [post_grid, 1, 1], [tpg, 1, 1])
        .expect("postprocess dispatch");
    let out_re = unpack_bytes(post_result.outputs.get("out_re").unwrap(), dt);
    let out_im = unpack_bytes(post_result.outputs.get("out_im").unwrap(), dt);
    (out_re[..n_out_total].to_vec(), out_im[..n_out_total].to_vec())
}

/// Relative error = |a - b| / max(|a|, |b|, 1e-8) for each element.
fn max_rel_err(a_re: &[f32], a_im: &[f32], b_re: &[f32], b_im: &[f32]) -> f32 {
    a_re.iter()
        .zip(a_im)
        .zip(b_re.iter().zip(b_im))
        .map(|((ar, ai), (br, bi))| {
            let ea = (ar - br).abs();
            let eb = (ai - bi).abs();
            let norm = ar.abs().max(br.abs()).max(ai.abs()).max(bi.abs()).max(1e-8);
            ea.max(eb) / norm
        })
        .fold(0.0_f32, f32::max)
}

fn ramp_signal(rows: usize, n: usize, offset: f32, stride: usize) -> Vec<f32> {
    (0..rows * n).map(|i| ((i % stride) as f32 - offset) * 0.05).collect()
}

// ── N = 400 (Whisper standard) ────────────────────────────────────────────────

#[test]
fn bluestein_n400_forward_f32_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (400, 2);
    let re = ramp_signal(rows, n, 12.0, 25);
    let im = vec![0.0_f32; rows * n]; // real-input signal
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::F32, n, rows, false);
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 1e-3, "bluestein n=400 f32 forward: max_rel_err = {err:.2e}");
}

#[test]
fn bluestein_n400_complex_input_f32_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (400, 2);
    let re = ramp_signal(rows, n, 10.0, 21);
    let im: Vec<f32> = (0..rows * n).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::F32, n, rows, false);
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 1e-3, "bluestein n=400 complex f32 forward: max_rel_err = {err:.2e}");
}

// ── N = 480 (Whisper large-v3 variant) ───────────────────────────────────────

#[test]
fn bluestein_n480_forward_f32_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (480, 2);
    let re = ramp_signal(rows, n, 12.0, 27);
    let im = vec![0.0_f32; rows * n];
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::F32, n, rows, false);
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 1e-3, "bluestein n=480 f32 forward: max_rel_err = {err:.2e}");
}

#[test]
fn bluestein_n480_complex_input_f32_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (480, 2);
    let re = ramp_signal(rows, n, 11.0, 23);
    let im: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.025).collect();
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::F32, n, rows, false);
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 1e-3, "bluestein n=480 complex f32 forward: max_rel_err = {err:.2e}");
}

// ── f16 / bf16 (looser tolerance) ────────────────────────────────────────────

#[test]
fn bluestein_n400_forward_f16_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (400, 1);
    // Keep magnitudes small so f16 butterfly accumulation stays in range.
    let re: Vec<f32> = (0..rows * n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let im = vec![0.0_f32; rows * n];
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::F16, n, rows, false);
    // f16 Bluestein: ~6-stage pipeline; allow 5% relative error.
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 5e-2, "bluestein n=400 f16: max_rel_err = {err:.2e}");
}

#[test]
fn bluestein_n480_forward_bf16_vs_naive_dft() {
    let _g = gpu_lock();
    let (n, rows) = (480, 1);
    let re: Vec<f32> = (0..rows * n).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
    let im = vec![0.0_f32; rows * n];
    let (exp_re, exp_im) = naive_dft(&re, &im, n, false);
    let (act_re, act_im) = run_bluestein(&re, &im, Dt::Bf16, n, rows, false);
    // bf16 has 7-bit mantissa; 10% relative tolerance for the 6-stage pipeline.
    let err = max_rel_err(&act_re, &act_im, &exp_re, &exp_im);
    assert!(err < 1e-1, "bluestein n=480 bf16: max_rel_err = {err:.2e}");
}
