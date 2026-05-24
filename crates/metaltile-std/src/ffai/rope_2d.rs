//! 2D positional RoPE for vision transformers.
//!
//! A vision transformer lays its tokens out on a 2-D `(row, col)` grid
//! (the patch grid from `patch_embed` / `conv2d`), so a single scalar
//! position can't encode where a token is. 2D RoPE splits each head's
//! `head_dim` into two equal halves: the first half is rotated by the
//! token's **row** index, the second half by its **column** index. Each
//! half then runs the ordinary rotate-half RoPE (Qwen2-VL / Qwen3-VL
//! `VisionRotaryEmbedding`, the "M-RoPE" spatial component).
//!
//! Within each half, the rotation pairs element `i` with element
//! `i + quarter_dim` (rotate-half over that half), exactly as
//! `rope_llama` pairs `i` with `i + half_dim` over the whole head:
//!
//!   half 0 (rows):   dims [0, half)            position = row
//!   half 1 (cols):   dims [half, head_dim)     position = col
//!
//!   for j in 0..quarter_dim:                 (quarter_dim = head_dim/4)
//!     inv_freq = theta_base^(-2j / half_dim)
//!     theta    = position * inv_freq
//!     x1, x2   = pair (j, j + quarter_dim) within the half
//!     o1 = x1*cos - x2*sin
//!     o2 = x1*sin + x2*cos
//!
//! This is distinct from `rope_llama` (1-D scalar position, optional
//! Llama-3 banding) — `rope_2d` has no banding and consumes a per-token
//! `(row, col)` pair instead of one global position.
//!
//! Layout:
//!
//!   qk        [n_tokens, n_heads, head_dim]   T
//!   positions [n_tokens, 2]                   u32  — (row, col) per token
//!   out       [n_tokens, n_heads, head_dim]   T
//!
//! Grid3D: one thread per `(token, head, j)` where `j ∈ [0, quarter_dim)`.
//! Each thread emits four output values — the two rotated pairs, one in
//! the row half and one in the column half. `head_dim` must be a
//! multiple of 4 (caller precondition — checked in the test / wrapper).
//!
//! Codegen-only. Correctness validated by `rope_2d_gpu_correctness`.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rope",
    subop="rope_2d",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn ffai_rope_2d<T>(
    qk: Tensor<T>,
    positions: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] n_heads: u32,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] quarter_dim: u32,
    #[constexpr] theta_base: f32,
) {
    let token = program_id::<0>();
    let head = program_id::<1>();
    let j = program_id::<2>();

    // Inverse frequency for this paired index. The exponent denominator
    // is `half_dim` because each spatial half is treated as its own
    // RoPE block of width `half_dim` (= 2 * quarter_dim).
    let j_f = j.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    let inv_freq = exp2(-2.0f32 * j_f * log2(theta_base) / half_f);

    // Per-token (row, col) grid position.
    let row = load(positions[token * 2u32]).cast::<f32>();
    let col = load(positions[token * 2u32 + 1u32]).cast::<f32>();

    let theta_row = row * inv_freq;
    let cos_r = cos(theta_row);
    let sin_r = sin(theta_row);

    let theta_col = col * inv_freq;
    let cos_c = cos(theta_col);
    let sin_c = sin(theta_col);

    let head_base = token * n_heads * head_dim + head * head_dim;

    // Row half: dims [0, half_dim), pair (j, j + quarter_dim).
    let r1 = head_base + j;
    let r2 = head_base + j + quarter_dim;
    let xr1 = load(qk[r1]).cast::<f32>();
    let xr2 = load(qk[r2]).cast::<f32>();
    store(out[r1], (xr1 * cos_r - xr2 * sin_r).cast::<T>());
    store(out[r2], (xr1 * sin_r + xr2 * cos_r).cast::<T>());

    // Column half: dims [half_dim, head_dim), pair (j, j + quarter_dim)
    // measured from the start of the second half.
    let c1 = head_base + half_dim + j;
    let c2 = head_base + half_dim + j + quarter_dim;
    let xc1 = load(qk[c1]).cast::<f32>();
    let xc2 = load(qk[c2]).cast::<f32>();
    store(out[c1], (xc1 * cos_c - xc2 * sin_c).cast::<T>());
    store(out[c2], (xc1 * sin_c + xc2 * cos_c).cast::<T>());
}
