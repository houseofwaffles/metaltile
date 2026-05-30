//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! SwiGLU activation — `silu(gate) * up`.
//!
//! Fused element-wise activation used in every modern transformer MLP
//! (Llama 4, Qwen3 dense + MoE, Gemma, Mistral families): given two
//! equally-sized inputs `gate` and `up` (the two halves of an MLP's
//! `w_gate · x` and `w_up · x` outputs), produce
//!
//! ```text
//!   out[i] = silu(gate[i]) * up[i]
//!         = (gate[i] * sigmoid(gate[i])) * up[i]
//! ```
//!
//! Existing baseline: two separate kernel launches — one applies
//! `silu(gate)` elementwise (`mt_silu` in `unary.rs`), the second
//! multiplies by `up` (`mt_binary` mul). Each load+store cycles the
//! intermediate `silu(gate)` value through device memory.
//!
//! Fusion saves one full-tensor RMW: the intermediate value stays in
//! registers, halving global memory traffic on the activation path.
//! At Qwen3-MoE expert intermediate=768 × prefill 512 tokens =
//! ~400KB per layer per expert; across 48 layers × 8 active experts
//! the saved bandwidth adds up.
//!
//! MLX reference: `mx.fast.swiglu` lives in
//! `mlx/mlx/backend/metal/kernels/fast.metal` as a single launch with
//! `silu(g) * u` in the body. We mirror that pattern.
//!
//! ## Cross-kernel calling
//!
//! `mt_swiglu` calls `mt_silu` via the DSL cross-kernel call syntax
//! (just the kernel name). `KernelInlinePass` splices the silu body
//! inline before MSL emission — no extra memory round-trip, same code
//! quality as a manual inline, with a clear compositional structure
//! that future fusion passes can reason about.
//!
//! Type-efficiency: `g` and `u` are loaded and cast to f32 before the
//! call. `KernelInlinePass` replaces `mt_silu`'s input-param load with
//! the actual f32 arg, so all arithmetic stays in f32 regardless of T.
//! No T→f32→T precision loss in the silu path.

use metaltile::kernel;

#[kernel]
pub fn mt_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // Cross-kernel call: KernelInlinePass splices mt_silu's scalar body
    // here. mt_silu's input-param load is replaced by g (already f32),
    // so silu runs in f32. Future fusion passes can identify the
    // (silu, mul) → swiglu composition pattern from this call site.
    let s = mt_silu(g);
    store(out[idx], (s * u).cast::<T>());
}

/// New-syntax correctness for `mt_swiglu` (`silu(gate) * up`, computed in f32).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_swiglu;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.2 - 1.0).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let expected: Vec<f32> =
            g_dt.iter().zip(&u_dt).map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
        TestSetup::new(mt_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(&up, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_swiglu(dt: DType) -> TestSetup { setup(1024, dt) }
}

/// New-syntax benchmark for `mt_swiglu` (vs MLX `mx.fast.swiglu`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_swiglu;

    #[bench(name = "mlx/swiglu", dtypes = [f32, f16, bf16])]
    fn bench_swiglu(dt: DType) -> BenchSetup {
        // `idx = tid` (global) — keep within a comfortable single-grid size.
        let n = 1024 * 1024usize;
        BenchSetup::new(mt_swiglu::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }
}
