//! Arange benchmark — #[kernel] DSL vs MLX metal/arange.metal
//!
//! MLX kernel: arangefloat32 / arangefloat16 / arangebfloat16 (arange.metal)
//!   Params: (start: constant T&, step: constant T&, out: device T*) — slots [0, 1, 2]
//!   Grid: [ceil(N/1024), 1, 1] × [1024, 1, 1]  (TPG=1024)
//!   Algorithm: out[index] = start + index * step  (one thread per element)
//!
//! MetalTile: mt_arange — same one-thread-per-element algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="arange",
    subop="arange",
    class=Arange,
    start=0.0,
    step=1.0,
    tol=1.0,
    mlx="arange{tn}",
    metal_file="arange.metal",
)]
#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}
