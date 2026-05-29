//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused image-unfold + linear-projection patch embedding for vision
//! transformers.
//!
//! A ViT patch embedding takes an image, cuts it into non-overlapping
//! `patch_h × patch_w` tiles, flattens each tile into a
//! `in_ch · patch_h · patch_w` vector, and linearly projects every
//! vector into the model's hidden dimension. Done as two ops it
//! materialises the unfolded `[num_patches, patch_dim]` tensor in global
//! memory — pure bandwidth waste, since each unfolded value is read
//! exactly once by the projection GEMM. This kernel fuses the unfold and
//! the projection: each thread gathers its patch's pixels straight from
//! the image and dots them with one weight row, no intermediate buffer.
//!
//! It differs from `conv2d` in layout, not arithmetic — `conv2d` keeps
//! the NCHW image convention and writes NCHW output; `patch_embed` takes
//! the same NCHW image but treats the weight as a flat linear matrix
//! `[hidden, patch_dim]` and writes transformer-token output
//! `[num_patches, hidden]`, which is what a ViT block consumes directly.
//!
//! Layouts (NCHW image, flat linear weight):
//!
//!   image    [in_ch, in_h, in_w]                   T   (single image)
//!   weight   [hidden, in_ch * patch_h * patch_w]   T
//!   bias     [hidden]                               T
//!   out      [num_patches, hidden]                  T
//!
//!   patches_h  = in_h / patch_h
//!   patches_w  = in_w / patch_w
//!   num_patches = patches_h * patches_w
//!   patch_dim  = in_ch * patch_h * patch_w
//!
//! One thread per output element `(patch, h)` where `patch` indexes the
//! flattened patch grid (row-major over the `patches_h × patches_w`
//! grid) and `h` indexes the hidden dimension. The thread walks the
//! patch's `in_ch × patch_h × patch_w` pixels, dotting each with the
//! matching weight column, accumulating in fp32. Generic over T.
//!
//! Patch order matches PyTorch `unfold` / `nn.Conv2d` flattening:
//! the weight column for `(ic, py, px)` is at
//! `ic*patch_h*patch_w + py*patch_w + px`.
//!
//! Codegen-only. Correctness validated by `patch_embed_gpu_correctness`.

use metaltile::kernel;

#[kernel(
    bench(
        op="patch_embed",
        subop="patch_embed",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
) {
    // Flat output index → (patch, h). One thread per output element.
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    // Top-left pixel of this patch in the image.
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let w_row_base = h * patch_dim;
    let mut acc = load(bias[h]).cast::<f32>();
    // Walk the patch's in_ch × patch_h × patch_w pixels, dotting each
    // with the corresponding weight column. The patch grid divides the
    // image exactly (caller precondition), so every read is in-bounds —
    // no padding / clamp logic needed.
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic_base = ic * input_plane;
        let w_ic_base = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic_base + (py0 + py) * in_w;
            let w_row = w_row_base + w_ic_base + py * patch_w;
            for px in range(0u32, patch_w, 1u32) {
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let wt = load(weight[w_row + px]).cast::<f32>();
                acc = acc + pix * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::patch_embed;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Explicit unfold + projection oracle (NCHW single image, flat
    /// `[hidden, patch_dim]` weight, `[num_patches, hidden]` output). f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_patch_embed(
        image: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let patches_h = in_h / patch_h;
        let patches_w = in_w / patch_w;
        let num_patches = patches_h * patches_w;
        let patch_dim = in_ch * patch_h * patch_w;
        let input_plane = in_h * in_w;
        let mut out = vec![0.0f32; num_patches * hidden];
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let patch = ph * patches_w + pw;
                for h in 0..hidden {
                    let mut acc = bias[h];
                    for ic in 0..in_ch {
                        for py in 0..patch_h {
                            for px in 0..patch_w {
                                let img_y = ph * patch_h + py;
                                let img_x = pw * patch_w + px;
                                let img_idx = ic * input_plane + img_y * in_w + img_x;
                                let col = ic * patch_h * patch_w + py * patch_w + px;
                                acc += image[img_idx] * weight[h * patch_dim + col];
                            }
                        }
                    }
                    out[patch * hidden + h] = acc;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn patch_embed_setup(
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = num_patches * hidden;
        let image_f = ramp(in_ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(hidden * patch_dim, 11, 4.0);
        let bias_f = ramp(hidden, 5, 2.0);
        let image = unpack_f32(&pack_f32(&image_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected =
            naive_patch_embed(&image, &weight, &bias, in_ch, in_h, in_w, patch_h, patch_w, hidden);
        TestSetup::new(patch_embed::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("image", pack_f32(&image_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // SigLIP / Qwen-VL: 14×14 patch, 3 channels, 28×42 image → 2×3 grid.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_patch_embed_patch14(dt: DType) -> TestSetup {
        patch_embed_setup(3, 28, 42, 14, 14, 32, dt)
    }

    // CLIP / Gemma-VL: 16×16 patch.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_patch_embed_patch16(dt: DType) -> TestSetup {
        patch_embed_setup(3, 32, 48, 16, 16, 24, dt)
    }

    // Small 8×8 patch — a tighter K reduction for the lower-noise path.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_patch_embed_patch8(dt: DType) -> TestSetup {
        patch_embed_setup(2, 16, 16, 8, 8, 12, dt)
    }
}

/// New-syntax bench for `patch_embed` (ViT-L SigLIP stem shape).
/// Grid3D, `grid_1d(num_patches * hidden, 256)`; bytes_moved = output stream.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::patch_embed;

    #[bench(name = "ffai/patch_embed/patch_embed", dtypes = [f32, f16, bf16])]
    fn bench_patch_embed(dt: DType) -> BenchSetup {
        // SigLIP-L: 14×14 patch, 3 channels, 224×224 → 16×16=256 patches,
        // hidden 1024.
        let (in_ch, in_h, in_w, patch_h, patch_w, hidden) =
            (3usize, 224usize, 224usize, 14usize, 14usize, 1024usize);
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = num_patches * hidden;
        BenchSetup::new(patch_embed::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("image", in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", hidden * patch_dim, dt))
            .buffer(BenchBuffer::random("bias", hidden, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}
