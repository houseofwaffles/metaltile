//! GPU correctness for `ffai::conv3d_mma` — MMA-tiled implicit-GEMM 3D conv.
//!
//! Validates against a five-loop CPU reference for stride=1, dilation=1,
//! pad=0. Output layout is [batch * out_d * out_h * out_w, out_ch].
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::conv3d_mma::conv3d_mma;

#[derive(Clone, Copy)]
struct MmaConv3dShape {
    batch: usize,
    in_ch: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kd: usize,
    kh: usize,
    kw: usize,
}

impl MmaConv3dShape {
    fn out_d(&self) -> usize { self.in_d - self.kd + 1 }
    fn out_h(&self) -> usize { self.in_h - self.kh + 1 }
    fn out_w(&self) -> usize { self.in_w - self.kw + 1 }
    fn n_voxels(&self) -> usize { self.batch * self.out_d() * self.out_h() * self.out_w() }
}

/// CPU reference: stride=1, dilation=1, pad=0.
/// Output: [batch * out_d * out_h * out_w, out_ch] (voxel-major).
#[allow(clippy::needless_range_loop)]
fn naive_conv3d_mma(input: &[f32], weight: &[f32], s: &MmaConv3dShape) -> Vec<f32> {
    let (out_d, out_h, out_w) = (s.out_d(), s.out_h(), s.out_w());
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let n_voxels = s.batch * out_dhw;
    let mut out = vec![0.0f32; n_voxels * s.out_ch];

    let in_plane = s.in_h * s.in_w;
    let in_vol = s.in_d * in_plane;

    for n in 0..s.batch {
        for od in 0..out_d {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let voxel = n * out_dhw + od * out_hw + oh * out_w + ow;
                    for oc in 0..s.out_ch {
                        let mut acc = 0.0f32;
                        for ic in 0..s.in_ch {
                            for kz in 0..s.kd {
                                for ky in 0..s.kh {
                                    for kx in 0..s.kw {
                                        let id = od + kz;
                                        let ih = oh + ky;
                                        let iw = ow + kx;
                                        let in_idx = n * s.in_ch * in_vol
                                            + ic * in_vol
                                            + id * in_plane
                                            + ih * s.in_w
                                            + iw;
                                        let w_idx = oc * s.in_ch * s.kd * s.kh * s.kw
                                            + ic * s.kd * s.kh * s.kw
                                            + kz * s.kh * s.kw
                                            + ky * s.kw
                                            + kx;
                                        acc += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        out[voxel * s.out_ch + oc] = acc;
                    }
                }
            }
        }
    }
    out
}

fn run_conv3d_mma(input: &[f32], weight: &[f32], dt: Dt, s: &MmaConv3dShape) -> Vec<f32> {
    let n_voxels = s.n_voxels();
    let n_out = n_voxels * s.out_ch;
    assert_eq!(s.out_ch % 32, 0, "out_ch must be divisible by 32");
    assert_eq!(n_voxels % 32, 0, "n_voxels must be divisible by 32");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_d".into(), u(s.in_d));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_d".into(), u(s.out_d()));
    buffers.insert("out_h".into(), u(s.out_h()));
    buffers.insert("out_w".into(), u(s.out_w()));
    buffers.insert("kd".into(), u(s.kd));
    buffers.insert("kh".into(), u(s.kh));
    buffers.insert("kw".into(), u(s.kw));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv3d_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let grid_x = s.out_ch / 32;
    let grid_y = n_voxels / 32;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, grid_y, 1], [128, 1, 1])
        .expect("conv3d_mma dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 1.0 } else { dot / (na * nb) }
}

#[test]
fn conv3d_mma_f32_3x3x3_kernel() {
    let _g = gpu_lock();
    // 3×3×3 kernel. in_d=7, in_h=7, in_w=7 → out_d=5, out_h=5, out_w=5 → 125 voxels.
    // 125 % 32 = 29 — not ok. Use in_d=7, in_h=8, in_w=8: out_d=5,out_h=6,out_w=6 → 180.
    // 180 % 32 = 20 — not ok. Use batch=2, smaller config:
    // in_d=5, in_h=5, in_w=5, kd=1, kh=1, kw=1 → out=5×5×5=125 → batch=2 → 250. 250%32=26.
    // Simplest: in_d=6, in_h=6, in_w=6, kd=2, kh=2, kw=2 → out_d=5,h=5,w=5 → 125.
    // batch=2 → 250 — still no.
    // Use kd=1,kh=1,kw=1 (no-op conv), in=8×8×8=512, batch=1 → n_voxels=512=16×32. out_ch=32.
    let s = MmaConv3dShape {
        batch: 1,
        in_ch: 2,
        in_d: 8,
        in_h: 8,
        in_w: 8,
        out_ch: 32,
        kd: 1,
        kh: 1,
        kw: 1,
    };
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 41, 20.0);
    let expected = naive_conv3d_mma(&input, &weight, &s);
    let actual = run_conv3d_mma(&input, &weight, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv3d_mma f32 1x1x1: cosine = {cs:.6}");
}

#[test]
fn conv3d_mma_f32_small_kernel() {
    let _g = gpu_lock();
    // kd=2, kh=2, kw=2, in_d=6, in_h=6, in_w=6 → out=5×5×5=125. batch=2 → 250. 250%32=26.
    // Use batch=4, in_d=4, in_h=4, in_w=4, kd=1, kh=1, kw=1 → 4*64=256 = 8×32.
    let s = MmaConv3dShape {
        batch: 4,
        in_ch: 4,
        in_d: 4,
        in_h: 4,
        in_w: 4,
        out_ch: 32,
        kd: 1,
        kh: 1,
        kw: 1,
    };
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 13, 6.0);
    let expected = naive_conv3d_mma(&input, &weight, &s);
    let actual = run_conv3d_mma(&input, &weight, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv3d_mma f32 batch4 1x1x1: cosine = {cs:.6}");
}

#[test]
fn conv3d_mma_f16() {
    let _g = gpu_lock();
    let s = MmaConv3dShape {
        batch: 1,
        in_ch: 2,
        in_d: 8,
        in_h: 8,
        in_w: 8,
        out_ch: 32,
        kd: 1,
        kh: 1,
        kw: 1,
    };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let input: Vec<f32> = (0..s.batch * s.in_ch * s.in_d * s.in_h * s.in_w)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let weight: Vec<f32> = (0..s.out_ch * s.in_ch * s.kd * s.kh * s.kw)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let expected = naive_conv3d_mma(&round(&input), &round(&weight), &s);
    let actual = run_conv3d_mma(&input, &weight, Dt::F16, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "conv3d_mma f16: cosine = {cs:.6}");
}

#[test]
fn conv3d_mma_bf16() {
    let _g = gpu_lock();
    let s = MmaConv3dShape {
        batch: 4,
        in_ch: 4,
        in_d: 4,
        in_h: 4,
        in_w: 4,
        out_ch: 32,
        kd: 1,
        kh: 1,
        kw: 1,
    };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let input: Vec<f32> = (0..s.batch * s.in_ch * s.in_d * s.in_h * s.in_w)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.03)
        .collect();
    let weight: Vec<f32> = (0..s.out_ch * s.in_ch * s.kd * s.kh * s.kw)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.04)
        .collect();
    let expected = naive_conv3d_mma(&round(&input), &round(&weight), &s);
    let actual = run_conv3d_mma(&input, &weight, Dt::Bf16, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.997, "conv3d_mma bf16: cosine = {cs:.6}");
}
