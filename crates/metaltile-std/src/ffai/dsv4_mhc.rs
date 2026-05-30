//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DeepSeek V4 Manifold-Constrained Hyper-Connections (mHC).
//!
//! mHC replaces the standard residual add with a `n_ch`-channel hidden
//! state that is mixed via Sinkhorn-Knopp-normalized `(n_ch × 1)` and
//! `(1 × n_ch)` combiners at each sublayer boundary. DSv4 uses
//! `n_ch = 4`.
//!
//! Two kernels per sublayer:
//!
//! 1. **`ffai_dsv4_mhc_pre`** — produces the sublayer's hidden-state
//!    input as a weighted sum over the 4 channels:
//!    `x[d] = sum_c pre[c] * H[c, d]`.
//! 2. **`ffai_dsv4_mhc_post`** — accumulates the sublayer's output
//!    back into the 4-channel state via an outer-product update:
//!    `H[c, d] += post[c] * y[d]`.
//!
//! The Sinkhorn-Knopp normalisation of `pre` and `post` happens **at
//! load time on host** (it's a property of the weights, not the
//! activations), so the kernels are simple weighted-sum / outer-add
//! patterns.
//!
//! ## Apple GPU shape
//!
//! Both kernels are 1D over `hidden_dim`. `n_ch=4` is constexpr so the
//! 4-element inner loop unrolls. No threadgroup memory needed.

use metaltile::kernel;

/// mHC pre-mix — collapse 4-channel state to single-channel sublayer
/// input via a per-channel weight.
///
/// `H` layout: `[n_ch, hidden_dim]` row-major (channel-major). For
/// `n_ch=4` and DSv4 `hidden_dim=4096`, the entire state is a 64 KB
/// fp32 / 32 KB fp16 slab.
#[kernel]
pub fn ffai_dsv4_mhc_pre<T>(
    state: Tensor<T>,
    pre: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] hidden_dim: u32,
    #[constexpr] n_ch: u32,
) {
    let d = tid;
    if d < hidden_dim {
        let mut acc = 0.0f32;
        for _c in range(0u32, n_ch, 1u32) {
            let w = load(pre[_c]);
            let h = load(state[_c * hidden_dim + d]).cast::<f32>();
            acc = acc + w * h;
        }
        store(out[d], acc);
    }
}

/// mHC post-mix — broadcast sublayer output back into the 4-channel
/// state via an outer-product accumulate. Each output element is
/// written `n_ch` times (one per channel slot).
///
/// `state` is read-modify-write (in-place residual update).
#[kernel]
pub fn ffai_dsv4_mhc_post<T>(
    sublayer_out: Tensor<T>,
    post: Tensor<f32>,
    mut state: Tensor<T>,
    #[constexpr] hidden_dim: u32,
    #[constexpr] n_ch: u32,
) {
    let d = tid;
    if d < hidden_dim {
        let y = load(sublayer_out[d]).cast::<f32>();
        for _c in range(0u32, n_ch, 1u32) {
            let w = load(post[_c]);
            let h = load(state[_c * hidden_dim + d]).cast::<f32>();
            store(state[_c * hidden_dim + d], h + w * y);
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{ffai_dsv4_mhc_post, ffai_dsv4_mhc_pre};
    use crate::utils::{pack_f32, unpack_f32};

    // ─── mhc_pre ─────────────────────────────────────────────────────

    fn cpu_pre(state: &[f32], pre: &[f32], n_ch: usize, hidden_dim: usize) -> Vec<f32> {
        let mut out = vec![0f32; hidden_dim];
        for (d, slot) in out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for c in 0..n_ch {
                acc += pre[c] * state[c * hidden_dim + d];
            }
            *slot = acc;
        }
        out
    }

    fn setup_pre(n_ch: usize, hidden_dim: usize, dt: DType) -> TestSetup {
        let state: Vec<f32> =
            (0..n_ch * hidden_dim).map(|i| (i as f32 * 0.011 - 0.7).sin() * 1.4).collect();
        let pre: Vec<f32> = (0..n_ch).map(|c| (c as f32 - 1.5) * 0.3 + 0.6).collect();
        let state_dt = unpack_f32(&pack_f32(&state, dt), dt);
        let expected = cpu_pre(&state_dt, &pre, n_ch, hidden_dim);
        TestSetup::new(ffai_dsv4_mhc_pre::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("state", pack_f32(&state, dt), dt))
            .input(TestBuffer::from_vec("pre", pack_f32(&pre, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", hidden_dim, dt))
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_ch", n_ch as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(hidden_dim, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_pre_dsv4(dt: DType) -> TestSetup { setup_pre(4, 4096, dt) }

    // ─── mhc_post ────────────────────────────────────────────────────

    fn cpu_post(
        state: &[f32],
        post: &[f32],
        y: &[f32],
        n_ch: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut out = state.to_vec();
        for c in 0..n_ch {
            for d in 0..hidden_dim {
                out[c * hidden_dim + d] += post[c] * y[d];
            }
        }
        out
    }

    fn setup_post(n_ch: usize, hidden_dim: usize, dt: DType) -> TestSetup {
        let state_init: Vec<f32> =
            (0..n_ch * hidden_dim).map(|i| (i as f32 * 0.0083 - 0.1).cos() * 0.9).collect();
        let post: Vec<f32> = (0..n_ch).map(|c| (c as f32 + 0.5) * 0.25).collect();
        let y: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017 - 1.2).sin() * 0.7).collect();
        let state_dt = unpack_f32(&pack_f32(&state_init, dt), dt);
        let y_dt = unpack_f32(&pack_f32(&y, dt), dt);
        let expected = cpu_post(&state_dt, &post, &y_dt, n_ch, hidden_dim);
        TestSetup::new(ffai_dsv4_mhc_post::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("sublayer_out", pack_f32(&y, dt), dt))
            .input(TestBuffer::from_vec("post", pack_f32(&post, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("state", pack_f32(&state_init, dt), dt))
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_ch", n_ch as u32)
            .expect(TestBuffer::from_vec("state", pack_f32(&expected, dt), dt))
            .grid_1d(hidden_dim, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_post_dsv4(dt: DType) -> TestSetup { setup_post(4, 4096, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_dsv4_mhc_post, ffai_dsv4_mhc_pre};

    #[bench(name = "ffai/dsv4_mhc_pre", dtypes = [f32, f16, bf16])]
    fn bench_pre(dt: DType) -> BenchSetup {
        let (n_ch, hidden_dim) = (4usize, 4096usize);
        BenchSetup::new(ffai_dsv4_mhc_pre::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("state", n_ch * hidden_dim, dt))
            .buffer(BenchBuffer::random("pre", n_ch, DType::F32))
            .buffer(BenchBuffer::zeros("out", hidden_dim, dt).output())
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_ch", n_ch as u32)
            .grid_1d(hidden_dim, 256)
            .bytes_moved(
                ((n_ch * hidden_dim + n_ch) * dt.size_bytes() + hidden_dim * dt.size_bytes())
                    as u64,
            )
    }

    #[bench(name = "ffai/dsv4_mhc_post", dtypes = [f32, f16, bf16])]
    fn bench_post(dt: DType) -> BenchSetup {
        let (n_ch, hidden_dim) = (4usize, 4096usize);
        BenchSetup::new(ffai_dsv4_mhc_post::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("sublayer_out", hidden_dim, dt))
            .buffer(BenchBuffer::random("post", n_ch, DType::F32))
            .buffer(BenchBuffer::random("state", n_ch * hidden_dim, dt).output())
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_ch", n_ch as u32)
            .grid_1d(hidden_dim, 256)
            .bytes_moved(((hidden_dim + n_ch + 2 * n_ch * hidden_dim) * dt.size_bytes()) as u64)
    }
}
