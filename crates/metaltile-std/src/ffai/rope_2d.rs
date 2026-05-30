//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile::kernel;

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

/// New-syntax correctness + bench for `ffai_rope_2d` (vision 2D positional
/// RoPE). Grid3D, grid `[n_tokens, n_heads, quarter_dim]`, tpg `[1,1,1]`.
/// Oracle rotates the row-half by the token's row position and the col-half by
/// its col position.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rope_2d;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_2d(dt: DType) -> TestSetup {
        let (n_tokens, n_heads, head_dim) = (4usize, 2usize, 64usize);
        let half = head_dim / 2;
        let quarter = head_dim / 4;
        let theta_base = 10_000.0f32;
        let qk_f: Vec<f32> =
            (0..n_tokens * n_heads * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        // Per-token (row, col) grid positions.
        let positions: Vec<u32> = (0..n_tokens).flat_map(|t| [t as u32, (t * 2) as u32]).collect();
        let qk = unpack_f32(&pack_f32(&qk_f, dt), dt);
        let mut exp = vec![0.0f32; n_tokens * n_heads * head_dim];
        for tok in 0..n_tokens {
            let (row, col) = (positions[tok * 2] as f32, positions[tok * 2 + 1] as f32);
            for h in 0..n_heads {
                for j in 0..quarter {
                    let invf = (-2.0 * j as f32 * theta_base.log2() / half as f32).exp2();
                    let (cr, sr) = ((row * invf).cos(), (row * invf).sin());
                    let (cc, sc) = ((col * invf).cos(), (col * invf).sin());
                    let hb = tok * n_heads * head_dim + h * head_dim;
                    let (xr1, xr2) = (qk[hb + j], qk[hb + j + quarter]);
                    exp[hb + j] = xr1 * cr - xr2 * sr;
                    exp[hb + j + quarter] = xr1 * sr + xr2 * cr;
                    let (xc1, xc2) = (qk[hb + half + j], qk[hb + half + j + quarter]);
                    exp[hb + half + j] = xc1 * cc - xc2 * sc;
                    exp[hb + half + j + quarter] = xc1 * sc + xc2 * cc;
                }
            }
        }
        TestSetup::new(ffai_rope_2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk_f, dt), dt))
            .input(TestBuffer::from_vec("positions", u32_bytes(&positions), DType::U32))
            .input(TestBuffer::zeros("out", n_tokens * n_heads * head_dim, dt))
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("quarter_dim", quarter as u32)
            .constexpr("theta_base", theta_base)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n_tokens as u32, n_heads as u32, quarter as u32, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_rope_2d`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rope_2d;

    #[bench(name = "ffai/rope/rope_2d", dtypes = [f32, f16, bf16])]
    fn bench_rope_2d(dt: DType) -> BenchSetup {
        let (n_tokens, n_heads, head_dim) = (1024usize, 16usize, 64usize);
        let half = head_dim / 2;
        let quarter = head_dim / 4;
        BenchSetup::new(ffai_rope_2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n_tokens * n_heads * head_dim, dt))
            .buffer(BenchBuffer::random("positions", n_tokens * 2, DType::U32))
            .buffer(BenchBuffer::zeros("out", n_tokens * n_heads * head_dim, dt).output())
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("quarter_dim", quarter as u32)
            .constexpr("theta_base", 10_000.0f32)
            .with_shape_label(format!(
                "tok{n_tokens} h{n_heads} d{head_dim} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_tokens as u32, n_heads as u32, quarter as u32, [1, 1, 1])
            .bytes_moved((2 * n_tokens * n_heads * head_dim * dt.size_bytes()) as u64)
    }
}
