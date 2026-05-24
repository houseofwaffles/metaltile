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

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="patch_embed",
    subop="patch_embed",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Grid3D,
)]
#[kernel]
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
