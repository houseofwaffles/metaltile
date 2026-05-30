//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel tiled GEMM — #[kernel] DSL vs MLX steel/gemm/kernels/steel_gemm_fused.metal
//!
//! Plain row-major `C = A · B` (the `nn` / non-transposed steel-gemm case):
//!   A: [M, K]  B: [K, N]  C: [M, N]
//!
//! Block shapes are instantiated per-(BM, BN); BK is fixed at 16. Each
//! threadgroup owns one BM×BN output block; the BM×BN block is split
//! into per-simdgroup sub-tiles (WM×WN simdgroups), and each sub-tile is
//! a grid of Apple 8×8 simdgroup-matrix fragments. The K dimension is
//! walked in BK=16 steps, two 8×8 K-fragments per step, accumulating into
//! the f32 `acc` fragments via `simdgroup_multiply_accumulate`.
//!
//! Apple 8×8 fragment lane layout (32 lanes, standard steel layout —
//! empirically confirmed by `probe/mma_layout_probe.rs`): each lane owns
//! two elements of the 8×8 fragment, and for **every** operand (A, B and
//! the C/D accumulator) lane element `i` sits at fragment position
//! `(fm, fn_i)` and holds the matrix value at that same `(row, col)`.
//! The earlier revision loaded B with a *transposed* convention
//! (`elem i` at `(fm, fn_i)` holding `B[fn_i, fm]`); that produced a
//! `Bᵀ`-shaped result and is the bug this file fixes.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`. Must be a multiple of 32.
//! - **Grid: 1 threadgroup per `BM×BN` output block** — a 2-D grid,
//!   `program_id<0>` = N-block (column), `program_id<1>` = M-block (row).
//! - **`m % BM == 0`, `n % BN == 0`, `k % 16 == 0`.** All loads are
//!   unconditional — ragged shapes read out of bounds. Callers with
//!   non-multiple shapes must pad (the steel-gemm `align_*` contract).
//! - **`KernelMode::SimdGroup2D`** so `program_id<i>` lowers to the
//!   threadgroup index, not the global thread index.

use metaltile::kernel;

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the fused GEMM.
///
/// Wrapping the **entire** `#[kernel] fn` in this outer `macro_rules!`
/// keeps the proc-macro happy: the compiler substitutes `$bm` / `$bn` /
/// `$wm` / `$wn` *before* the `#[kernel]` body parser runs, so the parser
/// only ever sees concrete `u32` literals — never an un-expanded inner
/// macro call (which would silently emit an empty body; see
/// `docs/developing.md` kernel-authoring hazards).
#[rustfmt::skip]
macro_rules! steel_gemm_fused_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] k: u32,
        ) {
            // ── Block / simdgroup geometry ──
            let bm = $bm;
            let bn = $bn;
            let wm = $wm;
            let wn = $wn;
            // Each simdgroup covers a (bm/wm)×(bn/wn) sub-tile, split into
            // 8×8 fragments → n_fm × n_fn fragments along M / N.
            let sub_m = bm / wm;
            let sub_n = bn / wn;
            let n_fm = sub_m / 8u32;
            let n_fn = sub_n / 8u32;
            let n_kf = 2u32; // BK = 16 ⇒ two 8×8 K-fragments per K-step.

            let tg_col = program_id::<0>(); // N-block index
            let tg_row = program_id::<1>(); // M-block index
            let sg_id = simd_group_id();
            let sg_m = sg_id / wn; // simdgroup row within the block
            let sg_n = sg_id % wn; // simdgroup col within the block
            let lane = simd_lane_id();

            // Apple 8×8 fragment lane mapping — each lane owns elements at
            // (fm, fn0) and (fm, fn1).
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;

            // Origin of this simdgroup's sub-tile inside the BM×BN block.
            let sub_m0 = sg_m * sub_m;
            let sub_n0 = sg_n * sub_n;
            // Absolute (row, col) origin of the block in C / A / B space.
            let block_m0 = tg_row * bm;
            let block_n0 = tg_col * bn;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    // f32 accumulator fragment for this 8×8 output tile.
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    // This fragment's M-row / N-col origin (absolute).
                    let m_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;

                    // ── Walk the full K dimension in BK=16 steps ──
                    for kb in range(0, k, 16) {
                        for _kf in range(0, n_kf, 1) {
                            let kf = kb + _kf * 8u32;
                            let sub_a = simdgroup_alloc::<T, 8, 8>();
                            let sub_b = simdgroup_alloc::<T, 8, 8>();

                            // A fragment: lane elem i at (fm, fn_i) holds
                            //   A[m_row + fm, kf + fn_i].
                            // The `.cast::<T>()` is load-bearing: the codegen
                            // lowers a bare `load` to an `float`-typed value,
                            // and MSL has no implicit `float → bfloat`
                            // conversion — without the cast the bf16
                            // instantiation fails to compile.
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

                            // B fragment: lane elem i at (fm, fn_i) holds
                            //   B[kf + fm, n_col + fn_i].  The fragment row
                            //   index `fm` walks the K dimension here — B is
                            //   loaded in the SAME (non-transposed) layout as
                            //   A and the accumulator, which is what Apple's
                            //   `simdgroup_multiply_accumulate` expects.
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

                            // acc += sub_a · sub_b
                            simdgroup_matmul(sub_a, sub_b, acc);
                        }
                    }

                    // ── Store the 8×8 result tile to C ──
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    store(out[(m_row + fm) * n + n_col + fn0], r0.cast::<T>());
                    store(out[(m_row + fm) * n + n_col + fn1], r1.cast::<T>());
                }
            }
        }
    };
}

// ── Block-shape instantiations ──────────────────────────────────────────
// Each shape mirrors an MLX `steel_gemm_fused` instantiation so the
// bench harness can wire a side-by-side reference (see the MLX
// `instantiate_gemm_shapes_helper` list in `steel_gemm_fused.metal`).
//
// 64×64×16 / 2×2 — the canonical large-tile prefill shape (4 simdgroups).
steel_gemm_fused_kernel!(
    mt_steel_gemm_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape for skinny M or N (4 simdgroups).
steel_gemm_fused_kernel!(
    mt_steel_gemm_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);
// 64×64×16 / 1×2 — large tile, 2 simdgroups (lower occupancy, N-split only).
steel_gemm_fused_kernel!(
    mt_steel_gemm_64x64x16_1x2,
    64u32,
    64u32,
    1u32,
    2u32,
    64u32,
    "bm64_bn64_bk16_wm1_wn2"
);
// 32×64×16 / 1×2 — wide-tile shape (N-heavy block, 2 simdgroups).
steel_gemm_fused_kernel!(
    mt_steel_gemm_32x64x16_1x2,
    32u32,
    64u32,
    1u32,
    2u32,
    64u32,
    "bm32_bn64_bk16_wm1_wn2"
);
// 64×64×16 / 4×2 — large tile at higher occupancy (8 simdgroups, TPG=256).
// ~40% faster than the 2×2 variant on the 4096³ bench (f32 1.4 vs 1.0
// GB/s) — the extra simdgroups hide the device-memory fragment loads.
steel_gemm_fused_kernel!(
    mt_steel_gemm_64x64x16_4x2,
    64u32,
    64u32,
    4u32,
    2u32,
    256u32,
    "bm64_bn64_bk16_wm4_wn2"
);
// 64×32×16 / 1×2 — M-heavy block (transpose of 32×64; 2 simdgroups).
// For M-skewed problems where a tall, narrow output tile keeps the
// per-simdgroup A-fragment reuse high.
steel_gemm_fused_kernel!(
    mt_steel_gemm_64x32x16_1x2,
    64u32,
    32u32,
    1u32,
    2u32,
    64u32,
    "bm64_bn32_bk16_wm1_wn2"
);
// 32×32×16 / 1×2 — small tile, 2 simdgroups (skinny problems, low TPG).
// A lower-occupancy small tile for problems too small to fill a 4-SG
// block; gives the dispatcher a 64-thread option.
steel_gemm_fused_kernel!(
    mt_steel_gemm_32x32x16_1x2,
    32u32,
    32u32,
    1u32,
    2u32,
    64u32,
    "bm32_bn32_bk16_wm1_wn2"
);

/// New-syntax benches for the fused steel GEMM block-shape family.
///
/// Each shape runs the canonical 4096³ prefill problem (`m = n = k =
/// 4096`, divisible by every `BM` / `BN` / 16). `SimdGroup2D` dispatch:
/// the grid is **tile-group counts** `(n/BN, m/BM, 1)` — `program_id<0>`
/// = N-block, `program_id<1>` = M-block — with `tpg = [WM*WN*32, 1, 1]`
/// (one simdgroup per sub-tile). `bytes_moved` counts the three dominant
/// matmul streams (A `m·k`, B `k·n`, C `m·n`). Bench-only — correctness
/// stays on the legacy GPU-correctness tests.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    /// Canonical production GEMM problem size (4096³).
    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;

    /// Build a fused steel-GEMM bench for one block shape.
    /// `bn` / `bm` are the output-block dims; `tpg` = `WM*WN*32`.
    fn fb(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let bytes = (m * k + k * n + m * n) * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("k", K)
            .with_shape_label(format!("m{M} n{N} k{K} {}", crate::bench_types::dtype_label(dt)))
            // SimdGroup2D grid is tile-GROUP counts: (n/BN, m/BM, 1).
            .grid_3d(N / bn, M / bm, 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/steel_gemm_fused/bm64_bn64_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_64x64x16_2x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm32_bn32_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_32x32x16_2x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm64_bn64_bk16_wm1_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_64x64x16_1x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_64x64x16_1x2::kernel_ir_for(dt), 64, 64, 64, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm32_bn64_bk16_wm1_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_32x64x16_1x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_32x64x16_1x2::kernel_ir_for(dt), 32, 64, 64, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm64_bn64_bk16_wm4_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_64x64x16_4x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_64x64x16_4x2::kernel_ir_for(dt), 64, 64, 256, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm64_bn32_bk16_wm1_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_64x32x16_1x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_64x32x16_1x2::kernel_ir_for(dt), 64, 32, 64, dt)
    }
    #[bench(name = "mlx/steel_gemm_fused/bm32_bn32_bk16_wm1_wn2", dtypes = [f32, f16, bf16])]
    fn bench_fused_32x32x16_1x2(dt: DType) -> BenchSetup {
        fb(mt_steel_gemm_32x32x16_1x2::kernel_ir_for(dt), 32, 32, 64, dt)
    }
}

/// New-syntax correctness tests for the fused steel GEMM — ports the
/// `nn` (non-transposed) oracle from
/// `tests/steel_gemm_gpu_correctness.rs`. The kernel computes a plain
/// row-major `C = A · B` (`A:[M,K]`, `B:[K,N]`, `C:[M,N]`); the oracle
/// is a straight triple-loop fp32 matmul over dtype-rounded inputs.
///
/// Small shape per block: `M = 2·BM`, `N = 2·BN` (a 2×2 grid of output
/// blocks, exercising `program_id<0/1>`), `K = 48` (3 BK=16 K-steps).
/// `SimdGroup2D` dispatch — grid is tile-group counts `(N/BN, M/BM, 1)`,
/// `tpg = [WM*WN*32, 1, 1]` — copied from the matching `#[bench]`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    /// Naive fp32 reference: `out[m,n] = Σ_k a[m,k]·b[k,n]`.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += a[mi * k + ki] * b[ki * n + ni];
                }
                out[mi * n + ni] = acc;
            }
        }
        out
    }

    /// Deterministic ramp — mirrors the legacy test's `ramp(n, modulus, offset)`.
    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// Build a fused-GEMM correctness setup for one block shape.
    fn fused_setup(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> TestSetup {
        let (m, n, k) = (bm as usize * 2, bn as usize * 2, 48usize);
        // Dtype-round the inputs so the CPU oracle sees the same
        // load-cast quantization the kernel does.
        let a = unpack_f32(&pack_f32(&ramp(m * k, 19, 7.0), dt), dt);
        let b = unpack_f32(&pack_f32(&ramp(k * n, 23, 9.0), dt), dt);
        let expected = naive_matmul(&a, &b, m, k, n);
        TestSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            // SimdGroup2D grid is tile-GROUP counts: (n/BN, m/BM, 1).
            .grid_3d(n as u32 / bn, m as u32 / bm, 1, [tpg, 1, 1])
    }

    // tol per dtype: f32 5e-3, f16 5e-2, bf16 2e-1 (K=48 matmul magnitudes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fused_64x64x16_2x2(dt: DType) -> TestSetup {
        fused_setup(mt_steel_gemm_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fused_32x32x16_2x2(dt: DType) -> TestSetup {
        fused_setup(mt_steel_gemm_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fused_64x64x16_1x2(dt: DType) -> TestSetup {
        fused_setup(mt_steel_gemm_64x64x16_1x2::kernel_ir_for(dt), 64, 64, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fused_32x64x16_1x2(dt: DType) -> TestSetup {
        fused_setup(mt_steel_gemm_32x64x16_1x2::kernel_ir_for(dt), 32, 64, 64, dt)
    }
}
