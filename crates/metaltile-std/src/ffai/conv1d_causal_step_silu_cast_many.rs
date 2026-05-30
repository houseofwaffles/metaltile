//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched depthwise causal conv1d step + SiLU + cast-to-f32 — sweeps T
//! tokens for one channel in ONE dispatch with the K-1 conv state held
//! in thread-local registers across the sweep.
//!
//! Identical math to the per-token pair `conv1d_causal_step` (in
//! `ssm.rs`) + `mt_silu_cast_to_f32` (in `mlx/unary.rs`) looped T times,
//! but collapses T × 2 dispatches into one and keeps the K-1-element
//! rolling state in registers for the duration of the channel's T-sweep
//! instead of round-tripping it through device memory once per token.
//!
//! Bandwidth motivation (Qwen3.6-A3B GDN prefill, T=512, 30 layers,
//! conv_dim=2048, conv_kernel=4):
//!   - per-token kernel reads 3 state rows and writes 3 state rows back
//!     to device memory every t-step: 6 * 2048 * 2 B = 24 KB per token
//!   - 24 KB × T × 30 layers ≈ 360 MB conv-state traffic per prefill
//!   - this kernel reads state once at start of channel sweep, writes
//!     once at end → 24 KB per channel sweep, ~12 MB per prefill (30×
//!     less)
//!
//! On top of the bandwidth save the per-channel kernel collapses
//! `T * (conv_step + silu_cast) = 2T` dispatches per layer into one —
//! same dispatch-saving pattern as `rope_llama_many` and
//! `kv_cache_update_many`.
//!
//! Layout:
//!
//!   src       [T, conv_dim]                T
//!   w         [conv_kernel, conv_dim]      T  (depthwise weights)
//!   b         [conv_dim]                   T  (bias)
//!   state_in  [conv_kernel - 1, conv_dim]  T  (K-1 prior input rows)
//!   out_f32   [T, conv_dim]                f32  (silu(conv) cast f32)
//!   state_out [conv_kernel - 1, conv_dim]  T  (last K-1 rows from src)
//!
//! Grid: `[conv_dim]` threads, one per channel. Each thread owns one
//! channel's complete T-sweep, sequential. Conv state lives in 3 scalar
//! registers (`s0, s1, s2`) — the rolling window of the K-1 most recent
//! inputs. After computing the conv output for time r, the state is
//! shifted: `s0 = s1; s1 = s2; s2 = src[r]`. Final state written back
//! to `state_out` after the sweep terminates.
//!
//! **conv_kernel is fixed at 4 in this kernel.** That covers every
//! production user of `conv1d_causal_step` in the FFAI tree today —
//! Qwen3.5 GDN, Mamba 2, NemotronH, FalconH1, GraniteMoeHybrid, Jamba
//! all set `conv_kernel = 4` (see e.g. `Qwen35.swift:237`,
//! `Mamba2.swift:105`, `NemotronH.swift:182`). A K!=4 model would need
//! a fresh kernel; declaring 3 explicit state scalars is the cleanest
//! way to keep the state purely in registers without DSL-level
//! cooperation between threads. `conv_kernel` is still a constexpr in
//! the signature for documentation + a runtime-side assert hook.
//!
//! Weight convention (inherited verbatim from `conv1d_causal_step`):
//!   y[d] = bias[d]
//!        + w[K-1, d] * x[r, d]                  (current input)
//!        + w[k,   d] * state[k, d]  for k=0..K-2 (K-1 prior inputs)
//! After the y compute the per-channel state shifts: state[k] gets
//! state[k+1] for k<K-2, then state[K-2] = x[r, d]. With K=4 the
//! K-1=3 state slots `(s0, s1, s2)` map to state rows `(0, 1, 2)`.
//!
//! SiLU + cast: `out = (acc * sigmoid(acc)).cast::<f32>()` in fp32.
//! Identical to `mt_silu_cast_to_f32` (`x * sigmoid(x)` with
//! `sigmoid(x) = 1 / (1 + exp(-x))`).
//!
//! `state_in` and `state_out` are decoupled tensors in the kernel
//! signature so the kernel doesn't have to alias-handle a single mutable
//! buffer mid-sweep. Callers are free to pass the same Metal buffer for
//! both (the kernel reads the K-1 state rows once at sweep start before
//! the first store to `state_out`, so the read-then-write aliasing is
//! safe). The runtime decides the residency / barrier story.
//!
//! Codegen-only. Correctness validated against the
//! `conv1d_causal_step` + `mt_silu_cast_to_f32` looped pair in
//! `tests/conv1d_causal_step_silu_cast_many_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn ffai_conv1d_causal_step_silu_cast_many<T>(
    src: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    state_in: Tensor<T>,
    mut out_f32: Tensor<f32>,
    mut state_out: Tensor<T>,
    #[constexpr] t_len: u32,
    #[constexpr] conv_dim: u32,
    #[constexpr] conv_kernel: u32,
) {
    let d = program_id::<0>();
    // K=4 hardcoded — see file docstring. `conv_kernel` is kept in the
    // signature for documentation + the runtime-side `kernel_size == 4`
    // assert hook; the kernel body never reads it.
    // Bias and the per-channel weights are constant across the T-sweep
    // → load once, keep in registers. The 4 weights `w0..w3` map to
    // state slots: `w0` weights `s0` (oldest prior input), `w1`→`s1`,
    // `w2`→`s2`, `w3`→`x_r` (current input). This is the K-1==3 form
    // of the generic `conv1d_causal_step` weight convention.
    let b_d = load(b[d]).cast::<f32>();
    let w0 = load(w[0u32 * conv_dim + d]).cast::<f32>();
    let w1 = load(w[1u32 * conv_dim + d]).cast::<f32>();
    let w2 = load(w[2u32 * conv_dim + d]).cast::<f32>();
    let w3 = load(w[3u32 * conv_dim + d]).cast::<f32>();
    // Load the K-1=3 state rows once at sweep start.
    let mut s0 = load(state_in[0u32 * conv_dim + d]).cast::<f32>();
    let mut s1 = load(state_in[1u32 * conv_dim + d]).cast::<f32>();
    let mut s2 = load(state_in[2u32 * conv_dim + d]).cast::<f32>();
    for r in range(0u32, t_len, 1u32) {
        let x_r = load(src[r * conv_dim + d]).cast::<f32>();
        // y = b + w0*s0 + w1*s1 + w2*s2 + w3*x_r — same op order as
        // `conv1d_causal_step` (bias + last-tap first, then state taps
        // in ascending index). Matters for f32 reassociation parity
        // with the per-row oracle.
        let acc = b_d + w3 * x_r + w0 * s0 + w1 * s1 + w2 * s2;
        // SiLU + cast — identical form to `mt_silu_cast_to_f32`:
        // sigmoid via `1 / (1 + exp(-x))`, all in fp32.
        let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - acc));
        let y = acc * sig;
        store(out_f32[r * conv_dim + d], y);
        // Shift state for next step: drop the oldest, slide the rest
        // forward, append x_r at the tail (s2). Three scalar moves,
        // no cross-thread coordination required because the state
        // lives entirely in this thread's registers.
        s0 = s1;
        s1 = s2;
        s2 = x_r;
    }
    // Final state — write the K-1 most recent inputs (last K-1 rows of
    // `src` for this channel) back to device memory. Matches what the
    // single-step kernel leaves in `state` after T iterations.
    store(state_out[0u32 * conv_dim + d], s0.cast::<T>());
    store(state_out[1u32 * conv_dim + d], s1.cast::<T>());
    store(state_out[2u32 * conv_dim + d], s2.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_conv1d_causal_step_silu_cast_many;
    use crate::utils::{pack_f32, unpack_f32};

    const CONV_KERNEL: u32 = 4;

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Per-channel CPU reference. Returns `(out, state_out)`, both
    /// dtype-rounded to match the GPU load/store quantisation.
    fn cpu_reference(
        src: &[f32],
        w: &[f32],
        b: &[f32],
        state_in: &[f32],
        t_len: usize,
        conv_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0f32; t_len * conv_dim];
        let mut state_out = vec![0.0f32; 3 * conv_dim];
        for d in 0..conv_dim {
            let b_d = b[d];
            let w0 = w[d];
            let w1 = w[conv_dim + d];
            let w2 = w[2 * conv_dim + d];
            let w3 = w[3 * conv_dim + d];
            let mut s0 = state_in[d];
            let mut s1 = state_in[conv_dim + d];
            let mut s2 = state_in[2 * conv_dim + d];
            for r in 0..t_len {
                let x_r = src[r * conv_dim + d];
                let acc = b_d + w3 * x_r + w0 * s0 + w1 * s1 + w2 * s2;
                let sig = 1.0f32 / (1.0f32 + (-acc).exp());
                out[r * conv_dim + d] = acc * sig;
                s0 = s1;
                s1 = s2;
                s2 = x_r;
            }
            state_out[d] = s0;
            state_out[conv_dim + d] = s1;
            state_out[2 * conv_dim + d] = s2;
        }
        (out, state_out)
    }

    fn setup(t_len: usize, conv_dim: usize, dt: DType) -> TestSetup {
        let state_rows = (CONV_KERNEL - 1) as usize;
        // Bounded inputs so SiLU's exp(-acc) stays well-conditioned.
        let src_f = ramp(t_len * conv_dim, 13, 2.0);
        let w_f = ramp(CONV_KERNEL as usize * conv_dim, 7, 0.6);
        let b_f = ramp(conv_dim, 5, 0.2);
        let state_f = ramp(state_rows * conv_dim, 11, 2.0);
        // Round every input through `dt` so the f32-internal oracle sees
        // the same load precision the kernel does.
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let w = unpack_f32(&pack_f32(&w_f, dt), dt);
        let b = unpack_f32(&pack_f32(&b_f, dt), dt);
        let state_in = unpack_f32(&pack_f32(&state_f, dt), dt);
        let (out_exp, state_exp) = cpu_reference(&src, &w, &b, &state_in, t_len, conv_dim);
        // `out_f32` is f32 regardless of `dt`; `state_out` is dtype-T and
        // gets rounded on store, so round the expected state through `dt`.
        let state_exp_t = unpack_f32(&pack_f32(&state_exp, dt), dt);
        TestSetup::new(ffai_conv1d_causal_step_silu_cast_many::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w_f, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b_f, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_f, dt), dt))
            .input(TestBuffer::zeros("out_f32", t_len * conv_dim, DType::F32))
            .input(TestBuffer::zeros("state_out", state_rows * conv_dim, dt))
            .constexpr("t_len", t_len as u32)
            .constexpr("conv_dim", conv_dim as u32)
            .constexpr("conv_kernel", CONV_KERNEL)
            .expect(TestBuffer::from_vec("out_f32", pack_f32(&out_exp, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp_t, dt), dt))
            .grid_3d(conv_dim as u32, 1, 1, [1, 1, 1])
    }

    // Short T-sweep, single-group conv_dim.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_causal_many_small(dt: DType) -> TestSetup { setup(2, 64, dt) }

    // Medium T-sweep, multi-group conv_dim.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv1d_causal_many_medium(dt: DType) -> TestSetup { setup(8, 256, dt) }
}

/// New-syntax bench for `ffai_conv1d_causal_step_silu_cast_many`
/// (Qwen3.6-A3B GDN prefill shape). Grid3D, `grid_3d(conv_dim, 1, 1,
/// [1, 1, 1])`. bytes_moved counts the f32 output stream + the T×conv_dim
/// src read.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::ffai_conv1d_causal_step_silu_cast_many;

    const CONV_KERNEL: u32 = 4;

    fn bench_shape(t_len: usize, conv_dim: usize, dt: DType) -> BenchSetup {
        let state_rows = (CONV_KERNEL - 1) as usize;
        let kernel: Kernel = ffai_conv1d_causal_step_silu_cast_many::kernel_ir_for(dt);
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", t_len * conv_dim, dt))
            .buffer(BenchBuffer::random("w", CONV_KERNEL as usize * conv_dim, dt))
            .buffer(BenchBuffer::random("b", conv_dim, dt))
            .buffer(BenchBuffer::random("state_in", state_rows * conv_dim, dt))
            .buffer(BenchBuffer::zeros("out_f32", t_len * conv_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("state_out", state_rows * conv_dim, dt).output())
            .constexpr("t_len", t_len as u32)
            .constexpr("conv_dim", conv_dim as u32)
            .constexpr("conv_kernel", CONV_KERNEL)
            .grid_3d(conv_dim as u32, 1, 1, [1, 1, 1])
            .bytes_moved(((t_len * conv_dim) * (DType::F32.size_bytes() + dt.size_bytes())) as u64)
    }

    #[bench(name = "ffai/ssm/conv1d_causal_step_silu_cast_many", dtypes = [f32, f16, bf16])]
    fn bench_conv1d_causal_many(dt: DType) -> BenchSetup {
        // Qwen3.6-A3B GDN prefill: T=512, conv_dim=2048.
        bench_shape(512, 2048, dt)
    }
}
