//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused gate-activation — `activation(gate) * up`.
//!
//! The general form of the SwiGLU MLP activation. Given two
//! equally-sized inputs `gate` and `up` (the two halves of an MLP's
//! `w_gate · x` and `w_up · x` projections), produce
//!
//! ```text
//!   out[i] = activation(gate[i]) * up[i]
//! ```
//!
//! The `silu` activation is the standard modern-transformer case and
//! already ships as the dedicated [`mt_swiglu`](super::swiglu) kernel.
//! This module covers the other two activation variants MLX's
//! `fused_gate_activation.metal` carries:
//!
//! - **`gelu_approx`** — the tanh approximation of GELU
//!   (`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`). Used by GELU-MLP
//!   transformers that ship a fused gate/up projection.
//! - **`clipped_swiglu`** — the GPT-OSS clipped variant: both halves
//!   are clamped to `[-7, 7]`, the gate side uses `sigmoid(1.702·g)`,
//!   and the up side carries a `+1` bias before the multiply.
//!
//! All three are one-thread-per-output Grid3D kernels — no
//! cross-thread cooperation, so the reduction-mode dispatch hazards do
//! not apply. The `single_row` / `looped` dispatch split MLX uses is a
//! threadgroup-tiling perf detail; a flat element-parallel grid is the
//! tractable port and is what FFAI's elementwise activation path wants.
//!
//! MLX reference: `mlx/backend/metal/kernels/fused_gate_activation.metal`
//! — `apply_gate<activation_type>` with `activation_type ∈ {0,1,2}`.
//! `activation_type == 0` (silu) is `mt_swiglu`; this file ports `1`
//! and `2`.
//!
//! Codegen-only; correctness pinned by
//! `tests/fused_gate_activation_gpu_correctness.rs`.

use metaltile::kernel;

// Numeric constants are inlined as literals inside the kernel bodies
// below — the `#[kernel]` proc-macro parses the body as a token stream
// and does not substitute Rust `const` items. They are named here for
// documentation only:
//   GELU_SQRT_2_OVER_PI    = 0.7978845608  — √(2/π), the gelu-tanh
//                                            inner scale.
//   GELU_CUBIC_COEFF       = 0.044715      — gelu-tanh cubic coeff.
//   CLIPPED_SWIGLU_CLIP    = 7.0           — GPT-OSS clamp bound.
//   CLIPPED_SWIGLU_GATE_SCALE = 1.702      — GPT-OSS gate sigmoid scale.

/// `out[i] = gelu_approx(gate[i]) * up[i]`.
///
/// GELU via the tanh approximation MLX uses — matches
/// `gelu_approx_act` in `fused_gate_activation.metal`. Computed in f32
/// regardless of `T` so the cubic + tanh keep their precision; the
/// result is cast back to `T` at the store.
#[kernel]
pub fn mt_fused_gate_gelu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // gelu_approx(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    let x3 = g * g * g;
    let inner = 0.7978845608f32 * (g + 0.044715f32 * x3);
    let act = 0.5f32 * g * (1.0f32 + tanh(inner));
    store(out[idx], (act * u).cast::<T>());
}

/// `out[i] = clipped_swiglu(gate[i], up[i])` — the GPT-OSS variant.
///
/// Both halves are clamped to `[-7, 7]`; the gate uses
/// `sigmoid(1.702·g)`; the up side has a `+1` bias before the
/// multiply: `g·sigmoid(1.702·g)·(u + 1)`. Matches `clipped_swiglu`
/// in `fused_gate_activation.metal`. The clamp is composed from two
/// `select`s (the DSL has no `clamp` builtin).
#[kernel]
pub fn mt_fused_gate_clipped_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g_raw = load(gate[idx]).cast::<f32>();
    let u_raw = load(up[idx]).cast::<f32>();
    // clamp(x, -7, 7) = min(max(x, -7), 7), composed from select since
    // the DSL exposes no `clamp` builtin (see ffai/rope_yarn.rs).
    let g_hi = select(g_raw > 7.0f32, 7.0f32, g_raw);
    let g = select(g_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, g_hi);
    let u_hi = select(u_raw > 7.0f32, 7.0f32, u_raw);
    let u = select(u_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, u_hi);
    // gate side: g · sigmoid(1.702·g)
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - 1.702f32 * g));
    // up side carries a +1 bias before the multiply.
    let act = g * sig * (u + 1.0f32);
    store(out[idx], act.cast::<T>());
}

/// New-syntax correctness for the fused gate-activation kernels (Grid3D, f32
/// internal). Oracles mirror the kernels exactly on dtype-rounded inputs.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_fused_gate_clipped_swiglu, mt_fused_gate_gelu};
    use crate::utils::{pack_f32, unpack_f32};

    fn inputs(n: usize, dt: DType) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Range spans beyond +/-7 so clipped_swiglu's clamp is exercised.
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 - 8.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
        let g = unpack_f32(&pack_f32(&gate, dt), dt);
        let u = unpack_f32(&pack_f32(&up, dt), dt);
        (gate, up, g, u)
    }

    fn build(
        kernel: metaltile::core::ir::Kernel,
        gate: &[f32],
        up: &[f32],
        expected: &[f32],
        dt: DType,
    ) -> TestSetup {
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("gate", pack_f32(gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(up, dt), dt))
            .input(TestBuffer::zeros("out", gate.len(), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(expected, dt), dt))
            .grid_1d(gate.len(), 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_fused_gate_gelu(dt: DType) -> TestSetup {
        let (gate, up, g, u) = inputs(512, dt);
        const C: f32 = 0.797_884_6; // sqrt(2/pi)
        let expected: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| 0.5 * g * (1.0 + (C * (g + 0.044715 * g * g * g)).tanh()) * u)
            .collect();
        build(mt_fused_gate_gelu::kernel_ir_for(dt), &gate, &up, &expected, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_fused_gate_clipped_swiglu(dt: DType) -> TestSetup {
        let (gate, up, g, u) = inputs(512, dt);
        let expected: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| {
                let g = g.clamp(-7.0, 7.0);
                let u = u.clamp(-7.0, 7.0);
                g * (1.0 / (1.0 + (-1.702 * g).exp())) * (u + 1.0)
            })
            .collect();
        build(mt_fused_gate_clipped_swiglu::kernel_ir_for(dt), &gate, &up, &expected, dt)
    }
}

/// New-syntax benchmarks for the fused gate-activation kernels.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_fused_gate_clipped_swiglu, mt_fused_gate_gelu};

    fn fb(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/fused_gate/gelu_approx", dtypes = [f32, f16, bf16])]
    fn bench_gelu(dt: DType) -> BenchSetup { fb(mt_fused_gate_gelu::kernel_ir_for(dt), dt) }

    #[bench(name = "mlx/fused_gate/clipped_swiglu", dtypes = [f32, f16, bf16])]
    fn bench_clipped(dt: DType) -> BenchSetup {
        fb(mt_fused_gate_clipped_swiglu::kernel_ir_for(dt), dt)
    }
}
