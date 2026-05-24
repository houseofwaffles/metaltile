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

use metaltile::{bench_kernel, kernel};

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
#[bench_kernel(
    op="ssm",
    subop="conv1d_causal_step",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
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
#[bench_kernel(
    op="ssm",
    subop="step",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
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
#[bench_kernel(
    op="ssm",
    subop="step_a2d",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
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
#[bench_kernel(
    op="ssm",
    subop="mt_step",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
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
