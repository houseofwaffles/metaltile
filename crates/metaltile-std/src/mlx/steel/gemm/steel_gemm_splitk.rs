//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel split-K GEMM — #[kernel] DSL vs MLX
//! `metal/steel/gemm/kernels/steel_gemm_splitk.metal`.
//!
//! GEMM that partitions the K dimension across threadgroups so a
//! skinny-M / skinny-N matmul with a very large K still saturates the
//! GPU. It is a **two-kernel** dispatch:
//!
//!   1. `mt_steel_gemm_splitk_*` — each K-split computes a partial
//!      `[M, N]` product over its slice of K and writes it to a
//!      `[n_splits, M, N]` fp32 partials buffer.
//!   2. `mt_steel_gemm_splitk_accum*` — reduces the `n_splits` partial
//!      `[M, N]` matrices into the final `[M, N]` output. The plain
//!      `accum` form is a straight sum; the `axpby` form computes
//!      `α·(Σ partials) + β·C_in` for the fused-bias / residual case.
//!
//! ## How the split-K handoff is expressed
//!
//! The DSL needs no "split-K scheduling primitive" — the partition is
//! just a 3-D grid (`program_id<2>` = K-split index) plus a K-loop
//! whose `[k_start, k_end)` bounds are derived from the split index
//! and a per-split `k_per_split` constexpr. The inter-kernel handoff
//! is an ordinary device buffer: kernel 1 writes the partials, kernel
//! 2 reads them. Two separate `#[kernel]` dispatches, sequenced by the
//! caller — exactly the MLX two-pass pattern.
//!
//! The partials buffer is always **fp32** (the accumulator dtype) so
//! the cross-split sum keeps full precision even for f16 / bf16
//! inputs — mirroring MLX's `AccumType = float`.
//!
//! ## DISPATCH INVARIANTS — split-K kernel
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`.
//! - **Grid: 3-D — `program_id<0>` = N-block, `program_id<1>` = M-block,
//!   `program_id<2>` = K-split index** (`0 ≤ split < n_splits`).
//! - **`m % BM == 0`, `n % BN == 0`.** `k_per_split` is a multiple of
//!   16 and `n_splits * k_per_split == k`. The last split may legally
//!   run past `k`; the K-loop is clamped to `k`.
//! - **`partials` is fp32, length `n_splits * m * n`**, laid out
//!   `[split, M, N]` row-major. The split-K kernel is itself a `T`
//!   kernel for the A / B operands but writes f32 partials — the
//!   `partials` tensor is declared `Tensor<f32>` regardless of `T`.
//! - **`KernelMode::SimdGroup2D`.**
//!
//! ## DISPATCH INVARIANTS — accum kernel
//!
//! - **Elementwise / Grid3D — one thread per `[M, N]` output element.**
//! - **`partials` length `n_splits * m * n` (fp32)**, `out` length
//!   `m * n`. The `axpby` form additionally reads a `c_in` `[M, N]`
//!   operand and two scalar constexprs `alpha` / `beta`.

use metaltile::kernel;

// ── Pass 1 — split-K partial GEMM ───────────────────────────────────────

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the
/// split-K partial GEMM. The outer `macro_rules!` substitutes the
/// literals before the `#[kernel]` body parser runs.
#[rustfmt::skip]
macro_rules! steel_gemm_splitk_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="steel_gemm_splitk",
                subop=$subop,
                class=SteelGemm,
                tol=1e-2,
                kernel_mode=SimdGroup2D,
                bm=$bm,
                bn=$bn,
                tpg=$tpg,
            )
        )]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            mut partials: Tensor<f32>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] k: u32,
            #[constexpr] k_per_split: u32,
        ) {
            // ── Block / simdgroup geometry (identical to steel_gemm_fused) ──
            let bm = $bm;
            let bn = $bn;
            let wm = $wm;
            let wn = $wn;
            let sub_m = bm / wm;
            let sub_n = bn / wn;
            let n_fm = sub_m / 8u32;
            let n_fn = sub_n / 8u32;
            let n_kf = 2u32; // BK = 16 ⇒ two 8×8 K-fragments per K-step.

            let tg_col = program_id::<0>(); // N-block index
            let tg_row = program_id::<1>(); // M-block index
            let split = program_id::<2>(); // K-split index
            let sg_id = simd_group_id();
            let sg_m = sg_id / wn;
            let sg_n = sg_id % wn;
            let lane = simd_lane_id();

            // Apple 8×8 fragment lane mapping.
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;

            let sub_m0 = sg_m * sub_m;
            let sub_n0 = sg_n * sub_n;
            let block_m0 = tg_row * bm;
            let block_n0 = tg_col * bn;

            // ── This split's K-range ──
            // [k_start, k_end) — the last split is clamped to `k`.
            let k_start = split * k_per_split;
            let k_end_raw = k_start + k_per_split;
            let k_end = select(k_end_raw < k, k_end_raw, k);
            // Partial-output base offset for this split: [split, M, N].
            let part_base = split * m * n;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    let m_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;

                    for kb in range(k_start, k_end, 16) {
                        for _kf in range(0, n_kf, 1) {
                            let kf = kb + _kf * 8u32;
                            let sub_a = simdgroup_alloc::<T, 8, 8>();
                            let sub_b = simdgroup_alloc::<T, 8, 8>();

                            simdgroup_elem_store(
                                sub_a,
                                0,
                                load(a[(m_row + fm) * k + kf + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_a,
                                1,
                                load(a[(m_row + fm) * k + kf + fn1]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_b,
                                0,
                                load(b[(kf + fm) * n + n_col + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_b,
                                1,
                                load(b[(kf + fm) * n + n_col + fn1]).cast::<T>(),
                            );

                            simdgroup_matmul(sub_a, sub_b, acc);
                        }
                    }

                    // Write this split's fp32 partial — no cast, the
                    // partials buffer is the f32 accumulator dtype.
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    store(partials[part_base + (m_row + fm) * n + n_col + fn0], r0);
                    store(partials[part_base + (m_row + fm) * n + n_col + fn1], r1);
                }
            }
        }
    };
}

// 64×64×16 / 2×2 — the canonical large-tile shape (4 simdgroups).
steel_gemm_splitk_kernel!(
    mt_steel_gemm_splitk_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape (4 simdgroups) — split-K is most
// useful exactly here: skinny M/N with a large K.
steel_gemm_splitk_kernel!(
    mt_steel_gemm_splitk_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);

// ── Pass 2 — partial-sum reduction ──────────────────────────────────────

/// Split-K accumulation: reduce `n_splits` partial `[M, N]` matrices
/// (fp32) into the final `[M, N]` output. One thread per output
/// element. This is the plain-sum form of MLX's
/// `steel_gemm_splitk_accum`.
#[kernel(
    bench(
        op="steel_gemm_splitk",
        subop="accum",
        class=GenericEmpty,
        tol=1e-3f32,
        kernel_mode=Elementwise,
        mlx="steel_gemm_splitk_accum_{tn}_float32",
        metal_file="steel/gemm/steel_gemm_splitk.metal",
    )
)]
pub fn mt_steel_gemm_splitk_accum<T>(
    partials: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] m: u32,
    #[constexpr] n: u32,
    #[constexpr] n_splits: u32,
) {
    // One thread per [M, N] output element — `program_id::<0>()` is the
    // global flat index, the grid is sized to `m * n` by the dispatch.
    let idx = program_id::<0>();
    let total = m * n;
    // Sum this element across every K-split.
    let mut acc = 0.0f32;
    for s in range(0u32, n_splits, 1u32) {
        acc = acc + load(partials[s * total + idx]);
    }
    store(out[idx], acc.cast::<T>());
}

/// Split-K accumulation, `axpby` form: `out = α·(Σ partials) + β·c_in`.
/// The fused-bias / residual variant of MLX's
/// `steel_gemm_splitk_accum_*_axbpy`. One thread per output element.
#[kernel(
    bench(
        op="steel_gemm_splitk",
        subop="accum_axpby",
        class=GenericEmpty,
        tol=1e-3f32,
        kernel_mode=Elementwise,
        mlx="steel_gemm_splitk_accum_{tn}_float32_axbpy",
        metal_file="steel/gemm/steel_gemm_splitk.metal",
    )
)]
pub fn mt_steel_gemm_splitk_accum_axpby<T>(
    partials: Tensor<f32>,
    c_in: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] m: u32,
    #[constexpr] n: u32,
    #[constexpr] n_splits: u32,
    #[constexpr] alpha: f32,
    #[constexpr] beta: f32,
) {
    let idx = program_id::<0>();
    let total = m * n;
    let mut acc = 0.0f32;
    for s in range(0u32, n_splits, 1u32) {
        acc = acc + load(partials[s * total + idx]);
    }
    // α·(Σ partials) + β·c_in.
    let prev = load(c_in[idx]).cast::<f32>();
    let res = alpha * acc + beta * prev;
    store(out[idx], res.cast::<T>());
}

/// New-syntax benches for the two-kernel split-K steel GEMM.
///
/// Pass 1 (`splitk_*`) — `m = n = 4096`, `k = 4096`, `N_SPLITS = 4`,
/// `k_per_split = k / N_SPLITS = 1024` (multiple of 16). `SimdGroup2D`
/// 3-D dispatch: grid is tile-group counts `(n/BN, m/BM, n_splits)` —
/// `program_id<2>` selects the K-split — with `tpg = [WM*WN*32, 1, 1]`.
/// The `partials` slab is fp32, length `n_splits*m*n` (`[split, M, N]`).
///
/// Pass 2 (`accum` / `accum_axpby`) — `Elementwise`, one thread per
/// `[M, N]` output element: grid `m*n / tpg`. `accum` is a straight sum;
/// `accum_axpby` adds a `c_in` residual scaled by `α` / `β`.
///
/// `bytes_moved` counts the dominant streams (A/B + partials write for
/// pass 1; partials read + out for pass 2). Bench-only — correctness
/// stays on the legacy GPU tests.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;
    const N_SPLITS: u32 = 4;
    /// Per-split K extent (`n_splits * k_per_split == k`, multiple of 16).
    const K_PER_SPLIT: u32 = K / N_SPLITS;
    /// Threads per group for the elementwise accum pass.
    const ACCUM_TPG: u32 = 256;

    // ── Pass 1 — split-K partial GEMM (SimdGroup2D, 3-D grid) ──────────────
    fn pb(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let f32_sz = DType::F32.size_bytes();
        // A / B input streams (full K across all splits) + the fp32 partials.
        let bytes = (m * k + k * n) * sz + N_SPLITS as usize * m * n * f32_sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            .buffer(BenchBuffer::zeros("partials", N_SPLITS as usize * m * n, DType::F32).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("k", K)
            .constexpr("k_per_split", K_PER_SPLIT)
            .with_shape_label(format!(
                "m{M} n{N} k{K} split{N_SPLITS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(N / bn, M / bm, N_SPLITS, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/steel_gemm_splitk/bm64_bn64_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_splitk_64x64x16_2x2(dt: DType) -> BenchSetup {
        pb(mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[bench(name = "mlx/steel_gemm_splitk/bm32_bn32_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_splitk_32x32x16_2x2(dt: DType) -> BenchSetup {
        pb(mt_steel_gemm_splitk_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }

    // ── Pass 2 — partial-sum reduction (Elementwise, one thread / elem) ────
    #[bench(name = "mlx/steel_gemm_splitk/accum", dtypes = [f32, f16, bf16])]
    fn bench_splitk_accum(dt: DType) -> BenchSetup {
        let (m, n) = (M as usize, N as usize);
        let sz = dt.size_bytes();
        let f32_sz = DType::F32.size_bytes();
        // Read every fp32 partial; write the [M, N] output.
        let bytes = N_SPLITS as usize * m * n * f32_sz + m * n * sz;
        BenchSetup::new(mt_steel_gemm_splitk_accum::kernel_ir_for(dt))
            .mode(KernelMode::Elementwise)
            .buffer(BenchBuffer::random("partials", N_SPLITS as usize * m * n, DType::F32))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("n_splits", N_SPLITS)
            .with_shape_label(format!(
                "m{M} n{N} split{N_SPLITS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_1d(m * n, ACCUM_TPG)
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/steel_gemm_splitk/accum_axpby", dtypes = [f32, f16, bf16])]
    fn bench_splitk_accum_axpby(dt: DType) -> BenchSetup {
        let (m, n) = (M as usize, N as usize);
        let sz = dt.size_bytes();
        let f32_sz = DType::F32.size_bytes();
        // Partials read + c_in read + out write.
        let bytes = N_SPLITS as usize * m * n * f32_sz + 2 * m * n * sz;
        BenchSetup::new(mt_steel_gemm_splitk_accum_axpby::kernel_ir_for(dt))
            .mode(KernelMode::Elementwise)
            .buffer(BenchBuffer::random("partials", N_SPLITS as usize * m * n, DType::F32))
            .buffer(BenchBuffer::random("c_in", m * n, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("n_splits", N_SPLITS)
            .constexpr("alpha", 1.0f32)
            .constexpr("beta", 1.0f32)
            .with_shape_label(format!(
                "m{M} n{N} split{N_SPLITS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_1d(m * n, ACCUM_TPG)
            .bytes_moved(bytes as u64)
    }
}
