//! GPU correctness for `ffai::conv2d_mma` — MMA-tiled implicit-GEMM 2D conv.
//!
//! Validates the simdgroup-matrix tiled output against the same six-loop
//! direct-conv CPU reference used by `conv2d_gpu_correctness.rs`. Covers
//! f32 / f16 / bf16. Tile constraints: out_ch and n_pixels both divisible
//! by 32; stride=1, dilation=1, pad=0.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::conv2d_mma::conv2d_mma;

/// Shape for stride=1, dilation=1, pad=0 — MMA kernel constraints.
#[derive(Clone, Copy)]
struct MmaConvShape {
    batch: usize,
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
}

impl MmaConvShape {
    fn out_h(&self) -> usize { self.in_h - self.kh + 1 }
    fn out_w(&self) -> usize { self.in_w - self.kw + 1 }
    fn n_pixels(&self) -> usize { self.batch * self.out_h() * self.out_w() }
}

/// CPU reference for stride=1, dilation=1, pad=0.
/// Output layout: [batch * out_h * out_w, out_ch] (pixel-major).
#[allow(clippy::needless_range_loop)]
fn naive_conv2d_mma(input: &[f32], weight: &[f32], s: &MmaConvShape) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    let out_hw = out_h * out_w;
    let n_pixels = s.batch * out_hw;
    let mut out = vec![0.0f32; n_pixels * s.out_ch];
    for n in 0..s.batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let pixel = n * out_hw + oh * out_w + ow;
                for oc in 0..s.out_ch {
                    let mut acc = 0.0f32;
                    for ic in 0..s.in_ch {
                        for ky in 0..s.kh {
                            for kx in 0..s.kw {
                                let ih = oh + ky;
                                let iw = ow + kx;
                                let in_idx = ((n * s.in_ch + ic) * s.in_h + ih) * s.in_w + iw;
                                let w_idx = ((oc * s.in_ch + ic) * s.kh + ky) * s.kw + kx;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    out[pixel * s.out_ch + oc] = acc;
                }
            }
        }
    }
    out
}

/// Dispatch conv2d_mma and read back output.
fn run_conv2d_mma(input: &[f32], weight: &[f32], dt: Dt, s: &MmaConvShape) -> Vec<f32> {
    let n_pixels = s.n_pixels();
    let n_out = n_pixels * s.out_ch;
    assert_eq!(s.out_ch % 32, 0, "out_ch must be divisible by 32 for MMA tile");
    assert_eq!(n_pixels % 32, 0, "n_pixels must be divisible by 32 for MMA tile");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_h".into(), u(s.out_h()));
    buffers.insert("out_w".into(), u(s.out_w()));
    buffers.insert("kh".into(), u(s.kh));
    buffers.insert("kw".into(), u(s.kw));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv2d_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [out_ch/32, n_pixels/32, 1], tpg = 128 (4 SG × 32 lanes).
    let grid_x = s.out_ch / 32;
    let grid_y = n_pixels / 32;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, grid_y, 1], [128, 1, 1])
        .expect("conv2d_mma dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// Cosine similarity (for vector comparison — more robust than max_abs for
/// accumulated fp errors on large K dimensions).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 1.0 } else { dot / (na * nb) }
}

// ── f32 tests ────────────────────────────────────────────────────────────────

#[test]
fn conv2d_mma_f32_3x3_kernel() {
    let _g = gpu_lock();
    // 3×3 kernel, stride=1, pad=0, out_h = in_h-2 = 6, out_w = in_w-2 = 6.
    // n_pixels = 1 × 6 × 6 = 36 (divisible by 32? — no, 36%32=4). Use larger.
    // Choose: in_h=10, in_w=10 → out_h=8, out_w=8, n_pixels=64. out_ch=32.
    let s = MmaConvShape { batch: 1, in_ch: 4, in_h: 10, in_w: 10, out_ch: 32, kh: 3, kw: 3 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 41, 20.0);
    let expected = naive_conv2d_mma(&input, &weight, &s);
    let actual = run_conv2d_mma(&input, &weight, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv2d_mma f32 3x3: cosine = {cs:.6}");
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-2, "conv2d_mma f32 3x3: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_mma_f32_1x1_kernel() {
    let _g = gpu_lock();
    // 1×1 conv: out_h = in_h, out_w = in_w. n_pixels=64, out_ch=32.
    let s = MmaConvShape { batch: 1, in_ch: 8, in_h: 8, in_w: 8, out_ch: 32, kh: 1, kw: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 17, 8.0);
    let expected = naive_conv2d_mma(&input, &weight, &s);
    let actual = run_conv2d_mma(&input, &weight, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv2d_mma f32 1x1: cosine = {cs:.6}");
}

#[test]
fn conv2d_mma_f32_multi_tile() {
    let _g = gpu_lock();
    // 64 out_ch (2 tiles), 64 n_pixels (2 tiles) would need out_h*out_w%32==0.
    // Use batch=4, in_h=8, in_w=8, kh=1, kw=1: n_pixels=4*8*8=256, out_ch=32.
    let s = MmaConvShape { batch: 4, in_ch: 4, in_h: 8, in_w: 8, out_ch: 32, kh: 1, kw: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 13, 6.0);
    let expected = naive_conv2d_mma(&input, &weight, &s);
    let actual = run_conv2d_mma(&input, &weight, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv2d_mma f32 multi-tile: cosine = {cs:.6}");
}

// ── f16 tests ────────────────────────────────────────────────────────────────

#[test]
fn conv2d_mma_f16_3x3_kernel() {
    let _g = gpu_lock();
    let s = MmaConvShape { batch: 1, in_ch: 4, in_h: 10, in_w: 10, out_ch: 32, kh: 3, kw: 3 };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    // Use small values to stay in f16 range.
    let input: Vec<f32> =
        (0..s.batch * s.in_ch * s.in_h * s.in_w).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    let weight: Vec<f32> =
        (0..s.out_ch * s.in_ch * s.kh * s.kw).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let expected = naive_conv2d_mma(&round(&input), &round(&weight), &s);
    let actual = run_conv2d_mma(&input, &weight, Dt::F16, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv2d_mma f16 3x3: cosine = {cs:.6}");
}

// ── bf16 tests ───────────────────────────────────────────────────────────────

#[test]
fn conv2d_mma_bf16_1x1_kernel() {
    let _g = gpu_lock();
    let s = MmaConvShape { batch: 1, in_ch: 8, in_h: 8, in_w: 8, out_ch: 32, kh: 1, kw: 1 };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let input: Vec<f32> =
        (0..s.batch * s.in_ch * s.in_h * s.in_w).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();
    let weight: Vec<f32> =
        (0..s.out_ch * s.in_ch * s.kh * s.kw).map(|i| ((i % 5) as f32 - 2.0) * 0.04).collect();
    let expected = naive_conv2d_mma(&round(&input), &round(&weight), &s);
    let actual = run_conv2d_mma(&input, &weight, Dt::Bf16, &s);
    let cs = cosine_sim(&expected, &actual);
    // bf16 has 7-bit mantissa, slightly looser tolerance.
    assert!(cs >= 0.997, "conv2d_mma bf16 1x1: cosine = {cs:.6}");
}
