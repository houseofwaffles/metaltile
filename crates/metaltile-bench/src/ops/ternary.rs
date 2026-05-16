//! Ternary select benchmark — #[kernel] DSL vs MLX metal/ternary.metal
//!
//! MLX kernel: v_Selectfloat32 / v_Selectfloat16 / v_Selectbfloat16 (ternary.metal)
//!   Params: (cond: device T*, a: device T*, b: device T*, dst: device T*,
//!            size: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Grid: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: dst[i] = cond[i] != 0 ? a[i] : b[i]  (one thread per element)
//!
//! MetalTile: mt_select — same algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="select",
    subop="select",
    class=Select,
    tol=1e-4,
    mlx="v_Select{tn}",
    metal_file="ternary.metal",
)]
#[kernel]
pub fn mt_select<T>(cond: Tensor<T>, on_true: Tensor<T>, on_false: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let c = load(cond[idx]);
    let t = load(on_true[idx]);
    let f = load(on_false[idx]);
    store(out[idx], select(c, t, f));
}
