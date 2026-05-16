//! Mamba 2 (SSD-form) building blocks: the selective-scan single-token
//! decode step and the depthwise causal-conv streaming step.
//!
//! The selective-scan kernel runs in fp32 state because `h` accumulates
//! `exp(A*dt)*h + dt*B*x` over many decode steps and bf16's 7-bit
//! mantissa drifts in a few dozen steps. The activation tensors stay in
//! whatever dtype the model runs at (typically bf16).

use metaltile::kernel;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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
    for k in range(0u32, kernel_size - 1u32, 1u32) {
        let s_kd = load(state[k * n_channels + d]).cast::<f32>();
        let w_kd = load(w[k * n_channels + d]).cast::<f32>();
        acc = acc + w_kd * s_kd;
    }
    store(y[d], acc.cast::<T>());

    // Shift state up by one (drop state[0], append x[d] at the tail).
    // Sequential within the thread → safe even though state[k] is read
    // after being written: we read state[k+1] each iteration, never
    // state[k].
    for k in range(0u32, kernel_size - 2u32, 1u32) {
        let next = load(state[(k + 1u32) * n_channels + d]);
        store(state[k * n_channels + d], next);
    }
    store(state[(kernel_size - 2u32) * n_channels + d], load(x[d]));
}

inventory::submit! {
    BenchSpec {
        op: "ssm",
        subop: "conv1d_causal_step",
        kernel_name: "conv1d_causal_step",
        kernel_ir: conv1d_causal_step::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: None,
    }
}

// Mamba 2 selective-scan single-token decode step. Per the SSD form
// (restricted to single-token decode):
//
//   h[head, n, d]_new = exp(A[head] * dt[head]) * h[head, n, d]_old
//                       + dt[head] * B[n] * x[head, d]
//   y[head, d]         = Σ_n  C[n] * h[head, n, d]_new
//
// One thread per (head, d) — total n_heads * head_dim threads. Each
// thread walks the state_dim axis once. No cross-thread sync needed
// because each (head, d) column of h is owned by exactly one thread.
//
// This is the decode form. Chunked prefill uses a parallel-scan
// variant — separate kernel, not needed for the Phase 5e drop.
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

inventory::submit! {
    BenchSpec {
        op: "ssm",
        subop: "step",
        kernel_name: "ssm_step",
        kernel_ir: ssm_step::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: None,
    }
}
