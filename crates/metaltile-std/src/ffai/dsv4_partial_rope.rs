//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 partial RoPE — rotates only the tail `n_rot` dims of each
//! head, leaving the leading `n_nope = head_dim - n_rot` dims
//! untouched.
//!
//! DSv4 head_dim=512 splits as nope=448 + rope=64. Q and K are
//! computed full-rank, then this kernel rotates only the last 64
//! dims of each head using the split-pair convention
//! `(dim_i, dim_i + n_rot/2)` for `dim_i ∈ [n_nope, n_nope + n_rot/2)`.
//!
//! ## Forward / inverse
//!
//! Forward rotation:  `(x, y) → (x·cos θ − y·sin θ, x·sin θ + y·cos θ)`
//! Inverse rotation:  `(x, y) → (x·cos θ + y·sin θ, −x·sin θ + y·cos θ)`
//!
//! The inverse is needed on the attention output before the grouped
//! O-LoRA — since K and V share the same tensor in MQA mode and only
//! K is logically rotated, the V-side contribution to the output
//! carries an extra rotation that has to be undone. Toggle via the
//! `inverse_flag` constexpr (`0` = forward, `1` = inverse).
//!
//! ## Dispatch
//!
//! Grid3D `[n_heads, half_rot, 1]` (one thread per rotation pair).
//! Operates in-place when `out == qk` — only the rope tail dims are
//! written. Caller is responsible for copying / passing through the
//! nope dims.

use metaltile::kernel;

#[kernel(
    bench(
        op="rope",
        subop="dsv4_partial_rope",
        class=GenericEmpty,
        tol=1e-4,
        kernel_mode=Grid3D,
    )
)]
pub fn ffai_dsv4_partial_rope<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_nope: u32,
    #[constexpr] half_rot: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] inverse_flag: u32,
) {
    let head = program_id::<0>();
    let pair_idx = program_id::<1>();
    let pair_f = pair_idx.cast::<f32>();
    let n_rot_f = (2u32 * half_rot).cast::<f32>();
    let inv_freq = exp2(0.0f32 - 2.0f32 * pair_f * log2(theta_base) / n_rot_f);
    let pos_f = position.cast::<f32>();
    let theta_raw = pos_f * inv_freq;
    // Inverse rotation: flip the sign of theta (cos stays, sin flips).
    let theta_signed = select(inverse_flag == 0u32, theta_raw, 0.0f32 - theta_raw);
    let cos_t = cos(theta_signed);
    let sin_t = sin(theta_signed);
    let head_base = head * head_dim;
    let dim_lo = head_base + n_nope + pair_idx;
    let dim_hi = head_base + n_nope + pair_idx + half_rot;
    let x_lo = load(qk[dim_lo]).cast::<f32>();
    let x_hi = load(qk[dim_hi]).cast::<f32>();
    let o_lo = x_lo * cos_t - x_hi * sin_t;
    let o_hi = x_lo * sin_t + x_hi * cos_t;
    store(out[dim_lo], o_lo.cast::<T>());
    store(out[dim_hi], o_hi.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_partial_rope;
    use crate::utils::{pack_f32, unpack_f32};

    #[allow(clippy::too_many_arguments)]
    fn cpu_reference(
        qk: &[f32],
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        position: u32,
        theta_base: f32,
        inverse: bool,
    ) -> Vec<f32> {
        let mut out = qk.to_vec();
        for head in 0..n_heads {
            for p in 0..half_rot {
                let inv_freq =
                    (-(p as f32) * 2.0 * theta_base.ln() / (2.0 * half_rot as f32)).exp();
                let theta_raw = position as f32 * inv_freq;
                let theta = if inverse { -theta_raw } else { theta_raw };
                let c = theta.cos();
                let s = theta.sin();
                let lo = head * head_dim + n_nope + p;
                let hi = head * head_dim + n_nope + p + half_rot;
                let x_lo = qk[lo];
                let x_hi = qk[hi];
                out[lo] = x_lo * c - x_hi * s;
                out[hi] = x_lo * s + x_hi * c;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        position: u32,
        theta_base: f32,
        inverse: bool,
        dt: DType,
    ) -> TestSetup {
        let qk: Vec<f32> =
            (0..n_heads * head_dim).map(|i| (i as f32 * 0.011 - 0.4).sin() * 1.2).collect();
        let qk_dt = unpack_f32(&pack_f32(&qk, dt), dt);
        let expected = cpu_reference(
            &qk_dt, n_heads, head_dim, n_nope, half_rot, position, theta_base, inverse,
        );
        let inv_flag: u32 = if inverse { 1 } else { 0 };
        // `out` buffer is initialized to a copy of `qk` so the
        // untouched nope dims pass through — kernel only writes the
        // rope pairs (the in-place pattern callers rely on).
        TestSetup::new(ffai_dsv4_partial_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk, dt), dt))
            .input(TestBuffer::from_vec("out", pack_f32(&qk, dt), dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("position", position)
            .constexpr("theta_base", theta_base)
            .constexpr("inverse_flag", inv_flag)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_heads as u32, half_rot as u32, 1, [1, 1, 1])
    }

    /// DSv4 production shape, forward, pos=64.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_dsv4_forward(dt: DType) -> TestSetup {
        setup(64, 512, 448, 32, 64, 10_000.0, false, dt)
    }

    /// DSv4 production shape, inverse, pos=64.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_dsv4_inverse(dt: DType) -> TestSetup {
        setup(64, 512, 448, 32, 64, 10_000.0, true, dt)
    }

    /// HCA compressed-stream theta (160K base).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_hca(dt: DType) -> TestSetup {
        setup(64, 512, 448, 32, 16, 160_000.0, false, dt)
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_partial_rope;

    #[bench(name = "ffai/dsv4_partial_rope", dtypes = [f32, f16, bf16])]
    fn bench_rope(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, n_nope, half_rot) = (64usize, 512usize, 448usize, 32usize);
        BenchSetup::new(ffai_dsv4_partial_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n_heads * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("position", 64u32)
            .constexpr("theta_base", 10_000.0_f32)
            .constexpr("inverse_flag", 0u32)
            .grid_3d(n_heads as u32, half_rot as u32, 1, [1, 1, 1])
            .bytes_moved((4 * n_heads * half_rot * dt.size_bytes()) as u64)
    }
}
