//! GPU correctness for `ffai::patch_embed_mma` — MMA-tiled patch embedding.
//!
//! Validates against the same CPU reference as `patch_embed_gpu_correctness.rs`
//! (explicit unfold + GEMM). Tile constraints: `hidden` and `num_patches`
//! both divisible by 32; `patch_dim` divisible by 32.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::patch_embed_mma::patch_embed_mma;

#[derive(Clone, Copy)]
struct MmaPatchShape {
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    patch_h: usize,
    patch_w: usize,
    hidden: usize,
}

impl MmaPatchShape {
    fn patches_w(&self) -> usize { self.in_w / self.patch_w }
    fn num_patches(&self) -> usize { (self.in_h / self.patch_h) * (self.in_w / self.patch_w) }
    fn patch_dim(&self) -> usize { self.in_ch * self.patch_h * self.patch_w }
}

/// CPU reference: explicit unfold + projection + bias.
/// Output: [num_patches, hidden].
fn naive_patch_embed_mma(
    image: &[f32],
    weight: &[f32],
    bias: &[f32],
    s: &MmaPatchShape,
) -> Vec<f32> {
    let patch_dim = s.patch_dim();
    let input_plane = s.in_h * s.in_w;
    let patches_h = s.in_h / s.patch_h;
    let patches_w = s.patches_w();
    let num_patches = patches_h * patches_w;

    let mut out = vec![0.0f32; num_patches * s.hidden];
    for ph in 0..patches_h {
        for pw in 0..patches_w {
            let pat = ph * patches_w + pw;
            let py0 = ph * s.patch_h;
            let px0 = pw * s.patch_w;
            for h in 0..s.hidden {
                let mut acc = bias[h];
                for ic in 0..s.in_ch {
                    for py in 0..s.patch_h {
                        for px in 0..s.patch_w {
                            let img_idx = ic * input_plane + (py0 + py) * s.in_w + (px0 + px);
                            let w_idx =
                                h * patch_dim + ic * s.patch_h * s.patch_w + py * s.patch_w + px;
                            acc += image[img_idx] * weight[w_idx];
                        }
                    }
                }
                out[pat * s.hidden + h] = acc;
            }
        }
    }
    out
}

fn run_patch_embed_mma(
    image: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &MmaPatchShape,
) -> Vec<f32> {
    let num_patches = s.num_patches();
    let n_out = num_patches * s.hidden;
    assert_eq!(s.hidden % 32, 0, "hidden must be divisible by 32 for MMA tile");
    assert_eq!(num_patches % 32, 0, "num_patches must be divisible by 32 for MMA tile");
    assert_eq!(s.patch_dim() % 32, 0, "patch_dim must be divisible by 32 for MMA K-loop");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("image".into(), pack_bytes(image, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("patch_h".into(), u(s.patch_h));
    buffers.insert("patch_w".into(), u(s.patch_w));
    buffers.insert("hidden".into(), u(s.hidden));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = patch_embed_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Grid: [hidden/32, num_patches/32, 1], tpg = 128.
    let grid_x = s.hidden / 32;
    let grid_y = num_patches / 32;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, grid_y, 1], [128, 1, 1])
        .expect("patch_embed_mma dispatch");
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
fn patch_embed_mma_f32_8x8_patch() {
    let _g = gpu_lock();
    // 8×8 patch, 4-channel, hidden=64, 8×8 image → 1 patch. 1 % 32 = 1. Not ok.
    // Need num_patches % 32 == 0. Use 8×8=64 patches: in_h=64, in_w=64, patch=8×8 → 64.
    // patch_dim = 4 * 8 * 8 = 256. 256 % 32 = 0. hidden=32. num_patches=64. All ok.
    let s = MmaPatchShape { in_ch: 4, in_h: 64, in_w: 64, patch_h: 8, patch_w: 8, hidden: 32 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.hidden * s.patch_dim(), 41, 20.0);
    let bias = ramp(s.hidden, 11, 5.0);
    let expected = naive_patch_embed_mma(&image, &weight, &bias, &s);
    let actual = run_patch_embed_mma(&image, &weight, &bias, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "patch_embed_mma f32 8×8: cosine = {cs:.6}");
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-2, "patch_embed_mma f32 8×8: max |diff| = {diff:.2e}");
}

#[test]
fn patch_embed_mma_f32_4x4_patch() {
    let _g = gpu_lock();
    // 4×4 patch, 8-channel, patch_dim=128, hidden=32, in_h=32, in_w=32 → 64 patches.
    let s = MmaPatchShape { in_ch: 8, in_h: 32, in_w: 32, patch_h: 4, patch_w: 4, hidden: 32 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.hidden * s.patch_dim(), 17, 8.0);
    let bias = ramp(s.hidden, 5, 2.0);
    let expected = naive_patch_embed_mma(&image, &weight, &bias, &s);
    let actual = run_patch_embed_mma(&image, &weight, &bias, Dt::F32, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "patch_embed_mma f32 4×4: cosine = {cs:.6}");
}

#[test]
fn patch_embed_mma_f16_8x8_patch() {
    let _g = gpu_lock();
    let s = MmaPatchShape { in_ch: 4, in_h: 64, in_w: 64, patch_h: 8, patch_w: 8, hidden: 32 };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let image: Vec<f32> =
        (0..s.in_ch * s.in_h * s.in_w).map(|i| ((i % 11) as f32 - 5.0) * 0.005).collect();
    let weight: Vec<f32> =
        (0..s.hidden * s.patch_dim()).map(|i| ((i % 7) as f32 - 3.0) * 0.005).collect();
    let bias: Vec<f32> = (0..s.hidden).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
    let expected = naive_patch_embed_mma(&round(&image), &round(&weight), &round(&bias), &s);
    let actual = run_patch_embed_mma(&image, &weight, &bias, Dt::F16, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.999, "patch_embed_mma f16 8×8: cosine = {cs:.6}");
}

#[test]
fn patch_embed_mma_bf16_4x4_patch() {
    let _g = gpu_lock();
    let s = MmaPatchShape { in_ch: 8, in_h: 32, in_w: 32, patch_h: 4, patch_w: 4, hidden: 32 };
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let image: Vec<f32> =
        (0..s.in_ch * s.in_h * s.in_w).map(|i| ((i % 9) as f32 - 4.0) * 0.01).collect();
    let weight: Vec<f32> =
        (0..s.hidden * s.patch_dim()).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
    let bias: Vec<f32> = (0..s.hidden).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
    let expected = naive_patch_embed_mma(&round(&image), &round(&weight), &round(&bias), &s);
    let actual = run_patch_embed_mma(&image, &weight, &bias, Dt::Bf16, &s);
    let cs = cosine_sim(&expected, &actual);
    assert!(cs >= 0.997, "patch_embed_mma bf16 4×4: cosine = {cs:.6}");
}
