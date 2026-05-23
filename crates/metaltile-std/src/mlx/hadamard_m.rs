//! Non-power-of-2 Hadamard transform — `hadamard_m` factor M ∈ {12, 20, 28}.
//!
//! This is the second stage in MLX's `hadamard_mn_contiguous` pipeline, which
//! computes `y = H_{M·N} · x` by factoring it as `(H_M ⊗ I_N) · (I_M ⊗ H_N)`.
//! The metaltile-std version ships a **standalone** kernel for the pure M-element
//! Hadamard of any batch of M-vectors.
//!
//! Expressed in the `#[kernel]` DSL — no `Op::InlineMsl`. The three M values
//! get their own monomorphized DSL function (`mt_hadamard_m12`, `_m20`, `_m28`)
//! because each has its own compile-time sign table. The Rust-level wrapper
//! `kernel_ir_for(m, dt)` dispatches to the right inner function.
//!
//! ## Algorithm
//!
//! One threadgroup processes one vector of M elements:
//! 1. All M threads load their element into threadgroup memory and barrier.
//! 2. Each thread `t` accumulates `out[t] = Σ_j H_M[t][j] · buf[j]`.
//! 3. The ±1 entries of each row are encoded as a compile-time bitmask
//!    constant: bit j set = H[t][j] = +1, bit j clear = H[t][j] = −1.
//!    These are seeded into a per-thread `stack_alloc` signs table.
//! 4. Result is scaled by `scale` and stored.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [n_rows, 1, 1]`, `tg = [M, 1, 1]`.
//! - One threadgroup per M-element vector; `tpg = M` (12, 20, or 28).
//! - `M < 32` is safe because the kernel uses a plain threadgroup-barrier
//!   accumulate (no `simd_*` intrinsics); `simd_lane` doubles as the
//!   thread-in-threadgroup index since one partial simdgroup covers the TG.
//! - `n_rows * M` must equal the total element count of the input tensor.
//!
//! Correctness pinned by `tests/hadamard_m_gpu_correctness.rs`.
//!
//! ## Sign-bit encoding
//!
//! From Sloane's table (<http://neilsloane.com/hadamard/>), mirroring
//! `mlx/backend/common/hadamard.h`. Each entry `signs[t]` is a 32-bit
//! integer where bit j = 1 means H_M[t][j] = +1 (otherwise −1).
//! Verified for orthogonality: H · H^T = M · I.

use metaltile::kernel;
use metaltile_core::{dtype::DType, ir::Kernel};

use crate::bench_types::DType as BenchDType;

// ── H_M sign-bit encodings ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h`. Each entry `signs[t]` is a
// 32-bit integer where bit j = 1 means H_M[t][j] = +1.  These are only used
// to verify H · H^T = M · I in tests; the kernel inlines the same constants
// as `stack_store` arguments (the DSL has no compile-time loop over Rust
// arrays, so each M gets its own monomorphized `#[kernel]` fn).
#[cfg(test)]
const H12_SIGNS: [u32; 12] = [4093, 1364, 3127, 1681, 223, 2629, 883, 2329, 3523, 1129, 1807, 421];

#[cfg(test)]
const H20_SIGNS: [u32; 20] = [
    445473, 859202, 702596, 389384, 747024, 641086, 234589, 469147, 938263, 828943, 984492, 953176,
    889521, 762211, 508614, 34194, 68357, 135722, 270452, 540873,
];

#[cfg(test)]
const H28_SIGNS: [u32; 28] = [
    53043585, 106070914, 210061060, 153783816, 41229328, 80377888, 160739520, 79265980, 156451192,
    44483185, 88966243, 177932359, 87445519, 172810270, 125848794, 251697461, 237056618, 207758549,
    149162411, 31986518, 63972909, 3206502, 4315853, 8631579, 17246902, 34477548, 68954969,
    135812787,
];

// One #[kernel] per M — each has its own compile-time sign table seed.
// The 12/20/28 unrolled `stack_store("signs", j, sign[j])` calls are the
// reason we can't share a single DSL function: there's no compile-time
// loop over Rust constants in the DSL.

/// M=12 specialization. Sign table = `H12_SIGNS`.
#[kernel]
pub fn mt_hadamard_m12<T>(inp: Tensor<T>, mut out: Tensor<T>, #[constexpr] scale: f32) {
    threadgroup_alloc("buf", 12u32, f32);
    stack_alloc("signs", 12u32, "u32");
    stack_store("signs", 0u32, 4093u32);
    stack_store("signs", 1u32, 1364u32);
    stack_store("signs", 2u32, 3127u32);
    stack_store("signs", 3u32, 1681u32);
    stack_store("signs", 4u32, 223u32);
    stack_store("signs", 5u32, 2629u32);
    stack_store("signs", 6u32, 883u32);
    stack_store("signs", 7u32, 2329u32);
    stack_store("signs", 8u32, 3523u32);
    stack_store("signs", 9u32, 1129u32);
    stack_store("signs", 10u32, 1807u32);
    stack_store("signs", 11u32, 421u32);

    let t = simd_lane;
    let row = tgid_x;
    let base = row * 12u32;
    let tg = base + t;

    let inp_f = load(inp[tg]).cast::<f32>();
    threadgroup_store("buf", t, inp_f);
    threadgroup_barrier();

    let signs_t = stack_load("signs", t);
    let mut acc = 0.0f32;
    for j in range(0u32, 12u32, 1u32) {
        let bit = ((signs_t >> j) & 1u32).cast::<f32>();
        let sign = 2.0f32 * bit - 1.0f32; // ∈ {−1, +1}
        let buf_j = threadgroup_load("buf", j);
        acc = acc + sign * buf_j;
    }

    let scaled = acc * scale;
    store(out[tg], scaled.cast::<T>());
}

/// M=20 specialization. Sign table = `H20_SIGNS`.
#[kernel]
pub fn mt_hadamard_m20<T>(inp: Tensor<T>, mut out: Tensor<T>, #[constexpr] scale: f32) {
    threadgroup_alloc("buf", 20u32, f32);
    stack_alloc("signs", 20u32, "u32");
    stack_store("signs", 0u32, 445473u32);
    stack_store("signs", 1u32, 859202u32);
    stack_store("signs", 2u32, 702596u32);
    stack_store("signs", 3u32, 389384u32);
    stack_store("signs", 4u32, 747024u32);
    stack_store("signs", 5u32, 641086u32);
    stack_store("signs", 6u32, 234589u32);
    stack_store("signs", 7u32, 469147u32);
    stack_store("signs", 8u32, 938263u32);
    stack_store("signs", 9u32, 828943u32);
    stack_store("signs", 10u32, 984492u32);
    stack_store("signs", 11u32, 953176u32);
    stack_store("signs", 12u32, 889521u32);
    stack_store("signs", 13u32, 762211u32);
    stack_store("signs", 14u32, 508614u32);
    stack_store("signs", 15u32, 34194u32);
    stack_store("signs", 16u32, 68357u32);
    stack_store("signs", 17u32, 135722u32);
    stack_store("signs", 18u32, 270452u32);
    stack_store("signs", 19u32, 540873u32);

    let t = simd_lane;
    let row = tgid_x;
    let base = row * 20u32;
    let tg = base + t;

    let inp_f = load(inp[tg]).cast::<f32>();
    threadgroup_store("buf", t, inp_f);
    threadgroup_barrier();

    let signs_t = stack_load("signs", t);
    let mut acc = 0.0f32;
    for j in range(0u32, 20u32, 1u32) {
        let bit = ((signs_t >> j) & 1u32).cast::<f32>();
        let sign = 2.0f32 * bit - 1.0f32;
        let buf_j = threadgroup_load("buf", j);
        acc = acc + sign * buf_j;
    }

    let scaled = acc * scale;
    store(out[tg], scaled.cast::<T>());
}

/// M=28 specialization. Sign table = `H28_SIGNS`.
#[kernel]
pub fn mt_hadamard_m28<T>(inp: Tensor<T>, mut out: Tensor<T>, #[constexpr] scale: f32) {
    threadgroup_alloc("buf", 28u32, f32);
    stack_alloc("signs", 28u32, "u32");
    stack_store("signs", 0u32, 53043585u32);
    stack_store("signs", 1u32, 106070914u32);
    stack_store("signs", 2u32, 210061060u32);
    stack_store("signs", 3u32, 153783816u32);
    stack_store("signs", 4u32, 41229328u32);
    stack_store("signs", 5u32, 80377888u32);
    stack_store("signs", 6u32, 160739520u32);
    stack_store("signs", 7u32, 79265980u32);
    stack_store("signs", 8u32, 156451192u32);
    stack_store("signs", 9u32, 44483185u32);
    stack_store("signs", 10u32, 88966243u32);
    stack_store("signs", 11u32, 177932359u32);
    stack_store("signs", 12u32, 87445519u32);
    stack_store("signs", 13u32, 172810270u32);
    stack_store("signs", 14u32, 125848794u32);
    stack_store("signs", 15u32, 251697461u32);
    stack_store("signs", 16u32, 237056618u32);
    stack_store("signs", 17u32, 207758549u32);
    stack_store("signs", 18u32, 149162411u32);
    stack_store("signs", 19u32, 31986518u32);
    stack_store("signs", 20u32, 63972909u32);
    stack_store("signs", 21u32, 3206502u32);
    stack_store("signs", 22u32, 4315853u32);
    stack_store("signs", 23u32, 8631579u32);
    stack_store("signs", 24u32, 17246902u32);
    stack_store("signs", 25u32, 34477548u32);
    stack_store("signs", 26u32, 68954969u32);
    stack_store("signs", 27u32, 135812787u32);

    let t = simd_lane;
    let row = tgid_x;
    let base = row * 28u32;
    let tg = base + t;

    let inp_f = load(inp[tg]).cast::<f32>();
    threadgroup_store("buf", t, inp_f);
    threadgroup_barrier();

    let signs_t = stack_load("signs", t);
    let mut acc = 0.0f32;
    for j in range(0u32, 28u32, 1u32) {
        let bit = ((signs_t >> j) & 1u32).cast::<f32>();
        let sign = 2.0f32 * bit - 1.0f32;
        let buf_j = threadgroup_load("buf", j);
        acc = acc + sign * buf_j;
    }

    let scaled = acc * scale;
    store(out[tg], scaled.cast::<T>());
}

/// Build the kernel IR for `mt_hadamard_m{M}` with M ∈ {12, 20, 28}.
/// Dispatches to the appropriate monomorphized DSL function.
pub fn kernel_ir_for(m: u32, dt: DType) -> Kernel {
    assert!(matches!(m, 12 | 20 | 28), "mt_hadamard_m only supports M ∈ {{12, 20, 28}}, got {m}");
    assert!(
        matches!(dt, DType::F32 | DType::F16 | DType::BF16),
        "mt_hadamard_m only supports F32/F16/BF16, got {dt:?}"
    );
    match m {
        12 => mt_hadamard_m12::kernel_ir_for(dt),
        20 => mt_hadamard_m20::kernel_ir_for(dt),
        28 => mt_hadamard_m28::kernel_ir_for(dt),
        _ => unreachable!(),
    }
}

// Keep `BenchDType` referenced so the `use` survives even when no
// inventory submit needs it (the inventory is registered per-M below).
const _: &[BenchDType] = &[BenchDType::F32, BenchDType::F16, BenchDType::BF16];

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops mirror the H_m matrix math
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::Op;

    use super::*;

    #[test]
    fn kernel_ir_constructs_for_all_m_and_dtypes() {
        for m in [12u32, 20, 28] {
            for dt in [DType::F32, DType::F16, DType::BF16] {
                let k = kernel_ir_for(m, dt);
                assert_eq!(k.name, format!("mt_hadamard_m{m}"));
                assert_eq!(k.params.len(), 2);
                assert_eq!(k.params[0].name, "inp");
                assert!(!k.params[0].is_output);
                assert_eq!(k.params[1].name, "out");
                assert!(k.params[1].is_output);
                assert_eq!(k.constexprs.len(), 1);
                assert_eq!(k.constexprs[0].name.name(), "scale");
                let all_ops =
                    || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
                assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
                assert!(all_ops().any(|op| matches!(op, Op::StackAlloc { .. })));
                assert!(all_ops().any(|op| matches!(op, Op::StackLoad { .. })));
            }
        }
    }

    #[test]
    #[should_panic(expected = "only supports M")]
    fn kernel_ir_rejects_invalid_m() { let _ = kernel_ir_for(16, DType::F32); }

    /// Codegen sanity — the generated MSL builds and carries the sign table.
    #[test]
    fn codegen_emits_kernel_decl() {
        for m in [12u32, 20, 28] {
            for dt in [DType::F32, DType::F16, DType::BF16] {
                let mut k = kernel_ir_for(m, dt);
                let suffix = match dt {
                    DType::F32 => "f32",
                    DType::F16 => "f16",
                    DType::BF16 => "bf16",
                    _ => unreachable!(),
                };
                k.name = format!("mt_hadamard_m{m}_{suffix}");
                let msl = MslGenerator::default().generate(&k).expect("codegen");
                assert!(msl.contains(&format!("kernel void mt_hadamard_m{m}_{suffix}")));
                assert!(!msl.contains("InlineMsl"));
            }
        }
    }

    /// Verify H_12 is orthogonal: H · H^T = 12 · I.
    #[test]
    fn h12_is_orthogonal() {
        let m = 12usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H12_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H12_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H12[{i}]·H12[{j}] = {dot}, expected {expected}");
            }
        }
    }

    /// Verify H_20 is orthogonal: H · H^T = 20 · I.
    #[test]
    fn h20_is_orthogonal() {
        let m = 20usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H20_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H20_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H20[{i}]·H20[{j}] = {dot}, expected {expected}");
            }
        }
    }

    /// Verify H_28 is orthogonal: H · H^T = 28 · I.
    #[test]
    fn h28_is_orthogonal() {
        let m = 28usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H28_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H28_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H28[{i}]·H28[{j}] = {dot}, expected {expected}");
            }
        }
    }
}
