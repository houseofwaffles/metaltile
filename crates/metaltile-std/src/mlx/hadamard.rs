//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Walsh–Hadamard transform along the last axis (size N = 2^k) —
//! port of MLX's `hadamard_n`.
//!
//! Computes `y = H_N · x` where `H_N` is the order-N Hadamard matrix,
//! then scales by `scale`. Used by the Walsh–Hadamard quantization /
//! rotation path (relevant to AURA's rotation matrix).
//!
//! Expressed as the fast Walsh–Hadamard transform: `log2(N)` in-place
//! butterfly passes over a threadgroup buffer. The MLX kernel uses a
//! radix-decomposed multi-step form for register efficiency; this port
//! keeps the plain butterfly — the codegen handles the rest, and one
//! threadgroup per row covers any `N ≤ 1024`. The non-power-of-2
//! `hadamard_m` factor (M ∈ {12,20,28}) is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [N, 1, 1]`.
//! - `N` a power of two, `32 ≤ N ≤ 1024`; one thread per element.
//!
//! Codegen-only; correctness pinned by
//! `tests/hadamard_gpu_correctness.rs`.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! hadamard_kernel {
    ($name:ident, $n:literal, $log_n:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="hadamard",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Reduction,
            )
        )]
        pub fn $name<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] scale: f32) {
            let row = program_id::<0>();
            let base = row * $n;
            threadgroup_alloc("buf", $n, "f32");
            threadgroup_store("buf", tid, load(inp[base + tid]).cast::<f32>());
            threadgroup_barrier();

            // log2(N) butterfly passes; stride h doubles each pass.
            for s in range(0u32, $log_n, 1u32) {
                let h = 1u32 << s;
                if (tid & h) == 0u32 {
                    let a = threadgroup_load("buf", tid);
                    let b = threadgroup_load("buf", tid + h);
                    threadgroup_store("buf", tid, a + b);
                    threadgroup_store("buf", tid + h, a - b);
                }
                threadgroup_barrier();
            }

            store(out[base + tid], (threadgroup_load("buf", tid) * scale).cast::<T>());
        }
    };
}

hadamard_kernel!(mt_hadamard_n64, 64u32, 6u32, "n64");
hadamard_kernel!(mt_hadamard_n128, 128u32, 7u32, "n128");
hadamard_kernel!(mt_hadamard_n256, 256u32, 8u32, "n256");
hadamard_kernel!(mt_hadamard_n512, 512u32, 9u32, "n512");
hadamard_kernel!(mt_hadamard_n1024, 1024u32, 10u32, "n1024");
