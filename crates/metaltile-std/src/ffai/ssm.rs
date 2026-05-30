//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mamba 2 (SSD-form) building blocks: the selective-scan single-token
//! decode step and the depthwise causal-conv streaming step. Plus
//! `ssm_step_a2d` — the Mamba 1 (Jamba) variant carrying a 2-D
//! per-(channel, state) `A_log` instead of the scalar-per-head `A`.
//!
//! `mt_ssm_step` is a faithful port of MLX's `ssm_step<T, Dh, Ds, H, G>`
//! from ekryski's `mlx` fork (`alpha` branch) — semantically MLX-aligned
//! but mainline MLX (pinned by `metaltile-std/build.rs`) doesn't ship
//! the `ssm.metal` source yet, so there's no side-by-side comparison
//! today. When the pin moves to a commit that ships `ssm.metal`, this
//! file (or just `mt_ssm_step` alone) graduates to `mlx/ssm.rs` and
//! picks up an MLX bench comparison via the standard `mlx=` /
//! `metal_file=` annotations.
//!
//! All three kernels run their `h`/state accumulators in fp32 — the
//! `exp(A*dt)*h + dt*B*x` recurrence in bf16 drifts in a few dozen
//! decode steps. Activation tensors stay in whatever dtype the model
//! runs at (typically bf16).
//!
//! Codegen-only. Correctness validated end-to-end in FFAI integration
//! tests against real Mamba/Nemotron decoding.

use metaltile::kernel;

// Mamba 2 / Mamba 1D depthwise causal-conv step — streaming-decode form.
//
//   y[d] = bias[d]
//        + w[K-1][d] * x[d]
//        + Σ_{k=0..K-2} w[k][d] * state[k][d]
//
// `state` holds the K-1 most recent inputs. After computing y the kernel
// shifts state in-place: state[k][d] = state[k+1][d], state[K-2][d] = x[d].
// Each channel d is owned by exactly one thread, so the read-then-write
// shift is safe within the thread without barriers.
//
// Grid: n_channels threads (one per channel). For Mamba 2 with conv_dim
// ~1500 channels and K=4 this is a tiny dispatch. Activation (Mamba 2
// follows the conv with SiLU) is the caller's concern — kept separate.
#[kernel]
pub fn conv1d_causal_step<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    mut state: Tensor<T>,
    mut y: Tensor<T>,
    #[constexpr] n_channels: u32,
    #[constexpr] kernel_size: u32,
) {
    let d = program_id::<0>();
    let x_d = load(x[d]).cast::<f32>();
    let b_d = load(b[d]).cast::<f32>();
    // Convolution: w[K-1] pairs with current input x[d]; w[0]..w[K-2]
    // pair with state[0]..state[K-2].
    let w_last = load(w[(kernel_size - 1u32) * n_channels + d]).cast::<f32>();
    let mut acc = b_d + w_last * x_d;
    // `kernel_size` is contractually >= 2 (a causal conv with state).
    // Guard the unsigned subtraction anyway: a stray `kernel_size == 0`
    // would make `kernel_size - 1` underflow to ~4e9 — a GPU-pinning
    // loop. `select` clamps the trip count to 0 instead.
    let conv_taps = select(kernel_size > 1u32, kernel_size - 1u32, 0u32);
    for k in range(0u32, conv_taps, 1u32) {
        let s_kd = load(state[k * n_channels + d]).cast::<f32>();
        let w_kd = load(w[k * n_channels + d]).cast::<f32>();
        acc = acc + w_kd * s_kd;
    }
    store(y[d], acc.cast::<T>());
    // Shift state up by one (drop state[0], append x[d] at the tail).
    // Sequential within the thread → safe even though state[k] is read
    // after being written: we read state[k+1] each iteration, never
    // state[k].
    // Same underflow guard: `kernel_size - 2` would wrap to ~4e9 for
    // any `kernel_size < 2`.
    let shift_taps = select(kernel_size > 2u32, kernel_size - 2u32, 0u32);
    for k in range(0u32, shift_taps, 1u32) {
        let next = load(state[(k + 1u32) * n_channels + d]);
        store(state[k * n_channels + d], next);
    }
    store(state[(kernel_size - 2u32) * n_channels + d], load(x[d]));
}

// Mamba 2 selective-scan single-token decode step. One thread per
// (head, d) — no cross-thread sync needed because each (head, d)
// column of h is owned by exactly one thread.
//
// This is the decode form. Chunked prefill uses a parallel-scan
// variant — separate kernel, not in this drop.
#[kernel]
pub fn ssm_step<T>(
    x: Tensor<T>,
    a: Tensor<T>,
    b: Tensor<T>,
    c: Tensor<T>,
    dt: Tensor<T>,
    mut h: Tensor<f32>,
    mut y: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] state_dim: u32,
) {
    let idx = program_id::<0>();
    let h_id = idx / head_dim;
    let d = idx - h_id * head_dim;
    let dt_val = load(dt[h_id]).cast::<f32>();
    let a_val = load(a[h_id]).cast::<f32>();
    let decay = exp(a_val * dt_val);
    let x_d = load(x[h_id * head_dim + d]).cast::<f32>();
    let mut y_d = 0.0f32;
    let h_base = h_id * state_dim * head_dim;
    for n in range(0u32, state_dim, 1u32) {
        let h_idx = h_base + n * head_dim + d;
        let h_old = load(h[h_idx]);
        let b_n = load(b[n]).cast::<f32>();
        let new_h = decay * h_old + dt_val * b_n * x_d;
        store(h[h_idx], new_h);
        let c_n = load(c[n]).cast::<f32>();
        y_d = y_d + c_n * new_h;
    }
    store(y[h_id * head_dim + d], y_d.cast::<T>());
}

// Mamba 1 (Jamba) selective-scan single-token decode step — the
// 2D-`A_log` variant of `ssm_step` above.
//
// The scalar `ssm_step` bakes in a per-channel scalar `A` (`a[h_id]`),
// so the decay `exp(A·dt)` is constant across the state dimension.
// Jamba's Mamba 1 mixer instead carries a *2-D* `A_log` of shape
// `[n_heads*head_dim, state_dim]` — one decay coefficient per
// `(channel, state)` pair — so `decay` varies with `n` inside the
// state loop. Mainline Mamba 2 families (Mamba2, FalconH1, NemotronH,
// GraniteMoeHybrid) use the scalar-`A` kernel and are unaffected;
// this variant exists purely to move Jamba's selective scan onto the
// GPU (it otherwise runs host-side).
//
// `A_log` is the raw log-parameter; the kernel applies the canonical
// Mamba `A = -exp(A_log)` reparam (matching `mt_ssm_step`). Per state
// element `(h, d, n)`:
//
//   A      = -exp(A_log[(h*head_dim + d), n])
//   decay  = exp(A · dt[h])
//   h'     = decay · h_old + dt[h] · B[n] · x[h, d]
//   y[h,d] = Σ_n C[n] · h'[h, d, n]
//
// One thread per `(head, d)` — same Grid3D geometry as `ssm_step`; no
// cross-thread sync because each `(head, d)` column of `h` is owned by
// exactly one thread. The state `h` runs in fp32 (the recurrence
// drifts in bf16 within a few dozen decode steps).
#[kernel]
pub fn ssm_step_a2d<T>(
    x: Tensor<T>,
    a_log: Tensor<T>,
    b: Tensor<T>,
    c: Tensor<T>,
    dt: Tensor<T>,
    mut h: Tensor<f32>,
    mut y: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] state_dim: u32,
) {
    let idx = program_id::<0>();
    let h_id = idx / head_dim;
    let d = idx - h_id * head_dim;
    let dt_val = load(dt[h_id]).cast::<f32>();
    let x_d = load(x[h_id * head_dim + d]).cast::<f32>();
    // `A_log` row for this channel: channel = h_id*head_dim + d, the
    // same flat index `idx` already computed.
    let a_log_base = idx * state_dim;
    let mut y_d = 0.0f32;
    let h_base = h_id * state_dim * head_dim;
    for n in range(0u32, state_dim, 1u32) {
        // Per-(channel, state) decay — the 2-D `A_log` difference.
        let a_val = 0.0f32 - exp(load(a_log[a_log_base + n]).cast::<f32>());
        let decay = exp(a_val * dt_val);
        let h_idx = h_base + n * head_dim + d;
        let h_old = load(h[h_idx]);
        let b_n = load(b[n]).cast::<f32>();
        let new_h = decay * h_old + dt_val * b_n * x_d;
        store(h[h_idx], new_h);
        let c_n = load(c[n]).cast::<f32>();
        y_d = y_d + c_n * new_h;
    }
    store(y[h_id * head_dim + d], y_d.cast::<T>());
}

// Faithful port of MLX's `ssm_step<T, Dh, Ds, H, G>` (alpha branch). One
// threadgroup per `(d_idx, n)` output element, where `n ∈ [0, n_heads*batch)`
// and `d_idx ∈ [0, dh)`. Each threadgroup runs 32 threads (one simd-group)
// and reduces across the state dimension via `simd_sum`.
//
// Required: `ds % 32 == 0` (one thread handles `ds/32` state elements).
//
// `heads_per_group` is MLX's `G`: number of Q heads sharing one (B, C)
// slot. Total distinct (B, C) groups = n_heads / heads_per_group.
#[kernel]
pub fn mt_ssm_step<T>(
    x: Tensor<T>,             // [n_heads*batch, dh]
    a_log: Tensor<T>,         // [n_heads]
    b_mat: Tensor<T>,         // [batch, n_heads/heads_per_group, ds]
    c_mat: Tensor<T>,         // [batch, n_heads/heads_per_group, ds]
    d_skip: Tensor<T>,        // [n_heads]
    dt: Tensor<T>,            // [n_heads*batch]
    state_in: Tensor<T>,      // [n_heads*batch, dh, ds]
    mut state_out: Tensor<T>, // [n_heads*batch, dh, ds]
    mut out: Tensor<T>,       // [n_heads*batch, dh]
    #[constexpr] dh: u32,
    #[constexpr] ds: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] heads_per_group: u32,
) {
    let d_idx = tgid_x;
    let n = tgid_y;
    let ds_idx = tid;
    // h_idx = n % n_heads (which head within the batch).
    // g_idx = n / heads_per_group (which (B, C) group this head reads from).
    let h_idx = n - (n / n_heads) * n_heads;
    let g_idx = n / heads_per_group;
    let dt_val = load(dt[n]).cast::<f32>();
    let a_val = 0.0f32 - exp(load(a_log[h_idx]).cast::<f32>());
    let da = exp(a_val * dt_val);
    let x_val = load(x[n * dh + d_idx]).cast::<f32>();
    let n_per_t = ds / 32u32;
    let bc_base = g_idx * ds;
    let state_base = n * dh * ds + d_idx * ds;
    let mut acc = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * ds_idx + i;
        let idx = state_base + s_idx;
        let db_by_x = x_val * dt_val * load(b_mat[bc_base + s_idx]).cast::<f32>();
        let new_state = da * load(state_in[idx]).cast::<f32>() + db_by_x;
        store(state_out[idx], new_state.cast::<T>());
        acc = acc + new_state * load(c_mat[bc_base + s_idx]).cast::<f32>();
    }
    let total = simd_sum(acc);
    if ds_idx == 0u32 {
        let d_val = load(d_skip[h_idx]).cast::<f32>();
        store(out[n * dh + d_idx], (total + x_val * d_val).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{conv1d_causal_step, ssm_step};
    use crate::utils::pack_f32;

    // ── conv1d_causal_step ──────────────────────────────────────────────

    /// CPU oracle: `y[d] = b[d] + w[K-1][d]·x[d] + Σ_{k<K-1} w[k][d]·state[k][d]`,
    /// then shift state up and append `x`. Returns `(y, shifted_state)`.
    fn conv1d_oracle(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state_in: &[f32],
        n_channels: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_channels];
        let mut state = state_in.to_vec();
        let k_last = kernel_size - 1;
        for d in 0..n_channels {
            let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
            for k in 0..k_last {
                acc += w[k * n_channels + d] * state_in[k * n_channels + d];
            }
            y[d] = acc;
        }
        for d in 0..n_channels {
            for k in 0..kernel_size.saturating_sub(2) {
                state[k * n_channels + d] = state_in[(k + 1) * n_channels + d];
            }
            if kernel_size >= 2 {
                state[(kernel_size - 2) * n_channels + d] = x[d];
            }
        }
        (y, state)
    }

    fn conv1d_setup(n_channels: usize, kernel_size: usize, dt: DType) -> TestSetup {
        let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_in: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();

        let (y_exp, state_exp) = conv1d_oracle(&x, &w, &b, &state_in, n_channels, kernel_size);

        TestSetup::new(conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::from_vec("state", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("y", n_channels, dt))
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state", pack_f32(&state_exp, dt), dt))
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
    }

    // Mamba 2 short-conv: kernel_size=4. One thread per channel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_conv1d_causal_step(dt: DType) -> TestSetup { conv1d_setup(128, 4, dt) }

    // ── ssm_step ────────────────────────────────────────────────────────

    /// CPU oracle for the scalar-A selective-scan decode step. `h` is f32;
    /// returns `(y, h_new)`.
    #[allow(clippy::too_many_arguments)]
    fn ssm_step_oracle(
        x: &[f32],
        a: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt_in: &[f32],
        h_state: &[f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        let mut h = h_state.to_vec();
        for hh in 0..n_heads {
            let decay = (a[hh] * dt_in[hh]).exp();
            let h_base = hh * state_dim * head_dim;
            for d in 0..head_dim {
                let x_d = x[hh * head_dim + d];
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let h_idx = h_base + n * head_dim + d;
                    let new_h = decay * h_state[h_idx] + dt_in[hh] * b_vec[n] * x_d;
                    h[h_idx] = new_h;
                    y_d += c_vec[n] * new_h;
                }
                y[hh * head_dim + d] = y_d;
            }
        }
        (y, h)
    }

    fn ssm_step_setup(n_heads: usize, head_dim: usize, state_dim: usize, dt: DType) -> TestSetup {
        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let a: Vec<f32> = (0..n_heads).map(|i| -0.5 - (i as f32) * 0.1).collect();
        let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let h_state: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();

        let (y_exp, h_exp) =
            ssm_step_oracle(&x, &a, &b_vec, &c_vec, &dt_in, &h_state, n_heads, head_dim, state_dim);

        // `h` is always f32 in the kernel signature; `y` carries the tested dt.
        TestSetup::new(ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b_vec, dt), dt))
            .input(TestBuffer::from_vec("c", pack_f32(&c_vec, dt), dt))
            .input(TestBuffer::from_vec("dt", pack_f32(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("h", pack_f32(&h_state, DType::F32), DType::F32))
            .input(TestBuffer::zeros("y", n_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("h", pack_f32(&h_exp, DType::F32), DType::F32))
            .grid_3d((n_heads * head_dim) as u32, 1, 1, [1, 1, 1])
    }

    // One thread per (head, d). `y` tolerance loosens for f16/bf16; `h` is
    // f32 so it must track tightly across all dtype runs.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_ssm_step(dt: DType) -> TestSetup { ssm_step_setup(4, 16, 8, dt) }
}

/// New-syntax benchmarks for all four `ffai::ssm` kernels. `conv1d_causal_step`
/// and `ssm_step` are also correctness-tested above; `ssm_step_a2d` (2-D
/// per-(channel,state) A_log) and `mt_ssm_step` (MLX-aligned reduction form)
/// are bench-only — both carry recurrent state with no clean one-step oracle
/// inside this harness. All MLX-less (`class=GenericEmpty`), `Ref(GB/s)` blank.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{conv1d_causal_step, mt_ssm_step, ssm_step, ssm_step_a2d};

    // Mamba 2 short-conv at a realistic channel count, K=4. One thread/channel.
    #[bench(name = "ffai/conv1d_causal_step", dtypes = [f32, f16, bf16])]
    fn bench_conv1d_causal_step(dt: DType) -> BenchSetup {
        let (n_channels, kernel_size) = (1536usize, 4usize);
        BenchSetup::new(conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", n_channels, dt))
            .buffer(BenchBuffer::random("w", kernel_size * n_channels, dt))
            .buffer(BenchBuffer::random("b", n_channels, dt))
            .buffer(BenchBuffer::random("state", (kernel_size - 1) * n_channels, dt).output())
            .buffer(BenchBuffer::zeros("y", n_channels, dt).output())
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
            .bytes_moved((kernel_size * n_channels * dt.size_bytes()) as u64)
    }

    // Scalar-A selective-scan decode. One thread per (head, d).
    #[bench(name = "ffai/ssm_step", dtypes = [f32, f16, bf16])]
    fn bench_ssm_step(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, state_dim) = (32usize, 64usize, 16usize);
        BenchSetup::new(ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", n_heads * head_dim, dt))
            .buffer(BenchBuffer::random("a", n_heads, dt))
            .buffer(BenchBuffer::random("b", state_dim, dt))
            .buffer(BenchBuffer::random("c", state_dim, dt))
            .buffer(BenchBuffer::random("dt", n_heads, dt))
            .buffer(BenchBuffer::random("h", n_heads * state_dim * head_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("y", n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .grid_3d((n_heads * head_dim) as u32, 1, 1, [1, 1, 1])
            .bytes_moved((n_heads * state_dim * head_dim * 2 * 4) as u64)
    }

    // Mamba 1 (Jamba) 2-D A_log variant. `a_log` is [n_heads*head_dim, state_dim].
    #[bench(name = "ffai/ssm_step_a2d", dtypes = [f32, f16, bf16])]
    fn bench_ssm_step_a2d(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, state_dim) = (32usize, 64usize, 16usize);
        let channels = n_heads * head_dim;
        BenchSetup::new(ssm_step_a2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", channels, dt))
            .buffer(BenchBuffer::random("a_log", channels * state_dim, dt))
            .buffer(BenchBuffer::random("b", state_dim, dt))
            .buffer(BenchBuffer::random("c", state_dim, dt))
            .buffer(BenchBuffer::random("dt", n_heads, dt))
            .buffer(BenchBuffer::random("h", n_heads * state_dim * head_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("y", channels, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .grid_3d(channels as u32, 1, 1, [1, 1, 1])
            .bytes_moved((channels * state_dim * 4) as u64)
    }

    // MLX-aligned reduction form: one simdgroup per (d_idx, n) reduces the
    // state axis via simd_sum. Grid `[dh, n_heads*batch, 1]`, TG `[32,1,1]`.
    #[bench(name = "ffai/mt_ssm_step", dtypes = [f32, f16, bf16])]
    fn bench_mt_ssm_step(dt: DType) -> BenchSetup {
        let (n_heads, heads_per_group, batch, dh, ds) = (8usize, 2usize, 2usize, 64usize, 32usize);
        let n_total = n_heads * batch;
        let groups = n_total / heads_per_group;
        BenchSetup::new(mt_ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", n_total * dh, dt))
            .buffer(BenchBuffer::random("a_log", n_heads, dt))
            .buffer(BenchBuffer::random("b_mat", groups * ds, dt))
            .buffer(BenchBuffer::random("c_mat", groups * ds, dt))
            .buffer(BenchBuffer::random("d_skip", n_heads, dt))
            .buffer(BenchBuffer::random("dt", n_total, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dh * ds, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dh * ds, dt).output())
            .buffer(BenchBuffer::zeros("out", n_total * dh, dt).output())
            .constexpr("dh", dh as u32)
            .constexpr("ds", ds as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .grid_3d(dh as u32, n_total as u32, 1, [32, 1, 1])
            .bytes_moved((n_total * dh * ds * 2 * dt.size_bytes()) as u64)
    }
}
