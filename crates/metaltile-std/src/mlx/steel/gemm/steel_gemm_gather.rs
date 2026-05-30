//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel gather GEMM — #[kernel] DSL vs MLX
//! `metal/steel/gemm/kernels/steel_gemm_gather.metal`.
//!
//! Row-major `C = A_gathered · B_gathered` where each output row /
//! column is computed from a **gathered** input row / column: an index
//! buffer maps each output row to a (non-contiguous) `A` row, and a
//! second index buffer maps each output column-block to a `B` matrix.
//! This is the MLX `gather_mm` op — the dense-matmul half of a
//! Mixture-of-Experts FFN, where `lhs_indices` routes tokens to expert
//! weight matrices.
//!
//! ## How gather is expressed without a gather-load primitive
//!
//! The DSL has no gather/scatter load, and it does not need one: a
//! gathered tiled matmul is the fused GEMM with one extra integer load
//! that **redirects the row index** before the address arithmetic.
//!
//!   - **`lhs_indices[r]`** — one `u32` per output row. The kernel
//!     reads it once and uses the result, instead of `r`, as the `A`
//!     row index. Because every lane in a simdgroup shares the same
//!     fragment-row `m_row`, the gather index is a per-row scalar — a
//!     single `load` of `lhs_indices[m_row + fm]`, then ordinary
//!     address arithmetic.
//!   - **`rhs_indices[blk]`** — one `u32` per `N`-block, selecting
//!     which `B` matrix (of `[K, N]` each) this output block multiplies
//!     against. The selected matrix base offset is
//!     `rhs_indices[tg_col] * k * n`.
//!
//! Both index buffers are plain `Tensor<u32>` operands; the redirection
//! is ordinary arithmetic. No new codegen primitive is required.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`. Must be a multiple of 32.
//! - **Grid: 1 threadgroup per `BM×BN` output block** — a 2-D grid,
//!   `program_id<0>` = N-block (column), `program_id<1>` = M-block (row).
//! - **`m % BM == 0`, `n % BN == 0`, `k % 16 == 0`.** All loads are
//!   unconditional — ragged shapes read out of bounds.
//! - **`lhs_indices` length `m`** (one gathered `A`-row per output row),
//!   `u32`, each value `< n_a_rows`. **`rhs_indices` length `n/BN`**
//!   (one selected `B`-matrix per N-block), `u32`, each value
//!   `< n_b_mats`. The kernel does no bounds-check — callers must keep
//!   indices in range.
//! - **`KernelMode::SimdGroup2D`** so `program_id<i>` lowers to the
//!   threadgroup index, not the global thread index.

use metaltile::kernel;

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the
/// gather GEMM. The outer `macro_rules!` substitutes the literals
/// before the `#[kernel]` body parser runs — see `steel_gemm_fused.rs`.
#[rustfmt::skip]
macro_rules! steel_gemm_gather_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            lhs_indices: Tensor<u32>,
            rhs_indices: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] k: u32,
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

            // ── Gather: select the B matrix for this N-block ──
            // rhs_indices[tg_col] picks one [K, N] matrix; its base
            // element offset into the flat `b` operand is index * k * n.
            let b_mat = load(rhs_indices[tg_col]);
            let b_base = b_mat * k * n;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    // Output (row, col) — the *destination* row is the
                    // contiguous fragment row; the *source* A row is
                    // gathered through `lhs_indices`.
                    let out_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;
                    // Gathered A-row for this fragment lane's row.
                    let a_row = load(lhs_indices[out_row + fm]);

                    for kb in range(0, k, 16) {
                        for _kf in range(0, n_kf, 1) {
                            let kf = kb + _kf * 8u32;
                            let sub_a = simdgroup_alloc::<T, 8, 8>();
                            let sub_b = simdgroup_alloc::<T, 8, 8>();

                            // A fragment: row redirected through the gather
                            // index `a_row`; column is the ordinary K index.
                            simdgroup_elem_store(
                                sub_a,
                                0,
                                load(a[a_row * k + kf + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_a,
                                1,
                                load(a[a_row * k + kf + fn1]).cast::<T>(),
                            );

                            // B fragment: from the gathered B matrix
                            // (`b_base`), non-transposed layout.
                            simdgroup_elem_store(
                                sub_b,
                                0,
                                load(b[b_base + (kf + fm) * n + n_col + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_b,
                                1,
                                load(b[b_base + (kf + fm) * n + n_col + fn1]).cast::<T>(),
                            );

                            simdgroup_matmul(sub_a, sub_b, acc);
                        }
                    }

                    // Store: the destination row is the *contiguous*
                    // output row, not the gathered A row.
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    store(out[(out_row + fm) * n + n_col + fn0], r0.cast::<T>());
                    store(out[(out_row + fm) * n + n_col + fn1], r1.cast::<T>());
                }
            }
        }
    };
}

// ── Block-shape instantiations ──────────────────────────────────────────
// 64×64×16 / 2×2 — the canonical large-tile shape (4 simdgroups).
steel_gemm_gather_kernel!(
    mt_steel_gemm_gather_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape for skinny M or N (4 simdgroups).
steel_gemm_gather_kernel!(
    mt_steel_gemm_gather_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);

/// New-syntax benches for the gather steel GEMM (MoE `gather_mm`).
///
/// Canonical 4096³ problem. `SimdGroup2D` dispatch: grid is tile-group
/// counts `(n/BN, m/BM, 1)`, `tpg = [WM*WN*32, 1, 1]`. Two index buffers
/// route the gather: `lhs_indices` (length `m`, one gathered A-row per
/// output row) and `rhs_indices` (length `n/BN`, one B-matrix per
/// N-block). `bytes_moved` counts the three dominant matmul streams plus
/// the index reads. Bench-only — correctness stays on the legacy GPU
/// tests.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;

    /// Build a gather steel-GEMM bench. `bn` is the output N-block dim
    /// (so `rhs_indices` has `n/bn` entries); `bm` the M-block dim.
    fn gb(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let n_blocks = n / bn as usize;
        let bytes = (m * k + k * n + m * n) * sz + (m + n_blocks) * DType::U32.size_bytes();
        BenchSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            // Index buffers must be in-range; zeros (gather row/matrix 0)
            // are valid and keep the bench deterministic.
            .buffer(BenchBuffer::zeros("lhs_indices", m, DType::U32))
            .buffer(BenchBuffer::zeros("rhs_indices", n_blocks, DType::U32))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("k", K)
            .with_shape_label(format!("m{M} n{N} k{K} {}", crate::bench_types::dtype_label(dt)))
            .grid_3d(N / bn, M / bm, 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/steel_gemm_gather/bm64_bn64_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_gather_64x64x16_2x2(dt: DType) -> BenchSetup {
        gb(mt_steel_gemm_gather_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[bench(name = "mlx/steel_gemm_gather/bm32_bn32_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_gather_32x32x16_2x2(dt: DType) -> BenchSetup {
        gb(mt_steel_gemm_gather_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
}

/// New-syntax correctness tests for the gather steel GEMM — ports the
/// oracle from `tests/steel_gemm_gather_gpu_correctness.rs`. The kernel
/// computes `out[r,c] = Σ_k a[lhs_indices[r], k] · b_sel[k, c]` where
/// `b_sel` is the `[K, N]` matrix `rhs_indices[c/BN]` stored flat in `b`
/// at offset `idx*K*N`.
///
/// Small shape: `M = 2·BM`, `N = 2·BN`, `K = 48`. Uses a permuted /
/// repeated `lhs_indices` (pulling from a larger A pool, `n_a_rows =
/// M+16`) and a two-matrix rhs select to exercise both gathers in a
/// single dispatch. `SimdGroup2D` grid `(N/BN, M/BM, 1)`, `tpg` copied
/// from the matching `#[bench]`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// Naive gathered fp32 reference (`b` is `n_b_mats` flat [K,N] matrices).
    #[allow(clippy::too_many_arguments)]
    fn naive_gather_matmul(
        a: &[f32],
        b: &[f32],
        lhs: &[u32],
        rhs: &[u32],
        m: usize,
        k: usize,
        n: usize,
        bn: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            let a_row = lhs[r] as usize;
            for c in 0..n {
                let b_base = rhs[c / bn] as usize * k * n;
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += a[a_row * k + ki] * b[b_base + ki * n + c];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    /// Build a gather-GEMM correctness setup. Two B matrices (rhs select)
    /// and a permuted lhs over an `M+16`-row A pool.
    fn gather_setup(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> TestSetup {
        let (m, n, k) = (bm as usize * 2, bn as usize * 2, 48usize);
        let n_a_rows = m + 16;
        let n_b_mats = 2usize;
        // Permuted/repeated lhs over the larger A pool; rhs picks matrix
        // 1 for N-block 0 and matrix 0 for N-block 1.
        let lhs: Vec<u32> = (0..m).map(|r| ((r * 7 + 3) % n_a_rows) as u32).collect();
        let rhs: Vec<u32> = (0..(n / bn as usize)).map(|i| (i % n_b_mats) as u32 ^ 1).collect();
        let a = unpack_f32(&pack_f32(&ramp(n_a_rows * k, 19, 7.0), dt), dt);
        let b = unpack_f32(&pack_f32(&ramp(n_b_mats * k * n, 23, 9.0), dt), dt);
        let expected = naive_gather_matmul(&a, &b, &lhs, &rhs, m, k, n, bn as usize);
        TestSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::from_vec("lhs_indices", u32_bytes(&lhs), DType::U32))
            .input(TestBuffer::from_vec("rhs_indices", u32_bytes(&rhs), DType::U32))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n as u32 / bn, m as u32 / bm, 1, [tpg, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_gather_64x64x16_2x2(dt: DType) -> TestSetup {
        gather_setup(mt_steel_gemm_gather_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_gather_32x32x16_2x2(dt: DType) -> TestSetup {
        gather_setup(mt_steel_gemm_gather_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
}
