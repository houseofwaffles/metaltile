//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel block-masked GEMM — #[kernel] DSL vs MLX
//! `metal/steel/gemm/kernels/steel_gemm_masked.metal`.
//!
//! Plain row-major `C = A · B` (the `nn` case) with **block-level
//! predication**: a per-output-block mask buffer says whether each
//! `BM×BN` output block is computed at all, and a per-operand-block
//! mask says whether each `BM×BK` / `BK×BN` operand tile contributes.
//! Skipped blocks write zeros (output-mask) or simply do not
//! accumulate (op-mask) — this is the MLX `block_masked_gemm`
//! semantics, used by sparse / windowed attention projections where
//! large rectangular regions of the product are known to be zero.
//!
//! ## How predication is expressed without block-conditional dispatch
//!
//! The DSL has no "early-exit a whole threadgroup" primitive, and it
//! does not need one: the masked GEMM is the fused GEMM with two
//! extra reads.
//!
//!   - **Output-block mask** `out_mask[blk]` — one `T` value per
//!     `(M-block, N-block)`. The block index is
//!     `tg_row * n_n_blocks + tg_col`. When the mask is `0` the
//!     threadgroup skips the K loop entirely and stores zeros; the
//!     skip is a single `if` around the accumulation + a `select` on
//!     the stored value, so every thread in the group takes the same
//!     branch (uniform control flow — no divergence cost).
//!   - **Operand-block mask** `op_mask[blk*n_k_blocks + kb_idx]` — one
//!     `T` per `(out-block, K-block)`. A `0` entry zeroes that K-block's
//!     contribution. Implemented as a multiply of the loaded fragment by
//!     the mask scalar, so a `0` mask contributes `0` without a branch.
//!
//! Both masks are plain `Tensor<T>` operands; the indexing is ordinary
//! arithmetic. No new codegen primitive is required — the existing
//! `if`, `select`, `load` and the `steel_gemm_fused` MMA ladder cover
//! it.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`. Must be a multiple of 32.
//! - **Grid: 1 threadgroup per `BM×BN` output block** — a 2-D grid,
//!   `program_id<0>` = N-block (column), `program_id<1>` = M-block (row).
//! - **`m % BM == 0`, `n % BN == 0`, `k % 16 == 0`.** All loads are
//!   unconditional within an un-masked block — ragged shapes read out of
//!   bounds. Callers with non-multiple shapes must pad.
//! - **`out_mask` length `(m/BM)*(n/BN)`**, row-major over
//!   `(M-block, N-block)`. **`op_mask` length `(m/BM)*(n/BN)*(k/16)`**,
//!   row-major over `(out-block, K-block)`. Both are `T`-typed; a value
//!   of `0` masks, any non-zero value scales.
//! - **`KernelMode::SimdGroup2D`** so `program_id<i>` lowers to the
//!   threadgroup index, not the global thread index.

use metaltile::kernel;

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the
/// block-masked GEMM. The outer `macro_rules!` substitutes the literals
/// before the `#[kernel]` body parser runs — see `steel_gemm_fused.rs`
/// for why the entire `#[kernel] fn` must be inside the macro.
#[rustfmt::skip]
macro_rules! steel_gemm_masked_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            out_mask: Tensor<T>,
            op_mask: Tensor<T>,
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

            // ── Block-mask indexing ──
            // n_n_blocks columns of output blocks; this block's flat index.
            let n_n_blocks = n / bn;
            let n_k_blocks = k / 16u32;
            let blk = tg_row * n_n_blocks + tg_col;
            // Output-block mask: 0 ⇒ this whole threadgroup writes zeros.
            let out_active = load(out_mask[blk]).cast::<f32>() != 0.0f32;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    let m_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;

                    // Only accumulate when the output block is active —
                    // uniform branch across the threadgroup.
                    if out_active {
                        // Walk K in BK=16 steps. The K-block index is
                        // `kb / 16` — derived directly, no loop counter.
                        for kb in range(0, k, 16) {
                            // Operand-block mask scalar for this K-block.
                            // 0 zeroes the contribution; non-zero scales it.
                            let kb_idx = kb / 16u32;
                            let opm = load(op_mask[blk * n_k_blocks + kb_idx]).cast::<f32>();
                            for _kf in range(0, n_kf, 1) {
                                let kf = kb + _kf * 8u32;
                                let sub_a = simdgroup_alloc::<T, 8, 8>();
                                let sub_b = simdgroup_alloc::<T, 8, 8>();

                                // A fragment, scaled by the op-mask scalar so
                                // a 0 mask contributes 0 with no branch.
                                let a0 = load(a[(m_row + fm) * k + kf + fn0]).cast::<f32>() * opm;
                                let a1 = load(a[(m_row + fm) * k + kf + fn1]).cast::<f32>() * opm;
                                simdgroup_elem_store(sub_a, 0, a0.cast::<T>());
                                simdgroup_elem_store(sub_a, 1, a1.cast::<T>());

                                // B fragment, non-transposed layout.
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
                    }

                    // Store: an inactive output block writes zeros.
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    let s0 = select(out_active, r0, 0.0f32);
                    let s1 = select(out_active, r1, 0.0f32);
                    store(out[(m_row + fm) * n + n_col + fn0], s0.cast::<T>());
                    store(out[(m_row + fm) * n + n_col + fn1], s1.cast::<T>());
                }
            }
        }
    };
}

// ── Block-shape instantiations ──────────────────────────────────────────
// 64×64×16 / 2×2 — the canonical large-tile shape (4 simdgroups).
steel_gemm_masked_kernel!(
    mt_steel_gemm_masked_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape for skinny M or N (4 simdgroups).
steel_gemm_masked_kernel!(
    mt_steel_gemm_masked_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);

/// New-syntax benches for the block-masked steel GEMM.
///
/// Canonical 4096³ problem. `SimdGroup2D` dispatch: grid is tile-group
/// counts `(n/BN, m/BM, 1)`, `tpg = [WM*WN*32, 1, 1]`. The two `T`-typed
/// masks are sized per the dispatch contract: `out_mask` length
/// `(m/BM)*(n/BN)` and `op_mask` length `(m/BM)*(n/BN)*(k/16)`. Masks are
/// seeded random (any non-zero value scales; the bench exercises the
/// fully-active path with overwhelming probability). `bytes_moved`
/// counts the three dominant matmul streams plus the mask reads.
/// Bench-only — correctness stays on the legacy GPU tests.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;
    const BK: u32 = 16;

    /// Build a block-masked steel-GEMM bench. `bm` / `bn` are the
    /// output-block dims (mask lengths derive from `m/bm`, `n/bn`, `k/16`).
    fn mb(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let n_out_blocks = (m / bm as usize) * (n / bn as usize);
        let n_op_entries = n_out_blocks * (k / BK as usize);
        let bytes = (m * k + k * n + m * n) * sz + (n_out_blocks + n_op_entries) * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            .buffer(BenchBuffer::random("out_mask", n_out_blocks, dt))
            .buffer(BenchBuffer::random("op_mask", n_op_entries, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("k", K)
            .with_shape_label(format!("m{M} n{N} k{K} {}", crate::bench_types::dtype_label(dt)))
            .grid_3d(N / bn, M / bm, 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "mlx/steel_gemm_masked/bm64_bn64_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_masked_64x64x16_2x2(dt: DType) -> BenchSetup {
        mb(mt_steel_gemm_masked_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[bench(name = "mlx/steel_gemm_masked/bm32_bn32_bk16_wm2_wn2", dtypes = [f32, f16, bf16])]
    fn bench_masked_32x32x16_2x2(dt: DType) -> BenchSetup {
        mb(mt_steel_gemm_masked_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
}

/// New-syntax correctness tests for the block-masked steel GEMM — ports
/// the oracle from `tests/steel_gemm_masked_gpu_correctness.rs`. The
/// kernel computes `C = A · B` with block-level predication: an
/// output-block mask zeroes whole `BM×BN` blocks, an operand-block mask
/// scales each `BK`-block's contribution.
///
/// Small shape: `M = 2·BM`, `N = 2·BN`, `K = 48` ⇒ 2×2 output blocks,
/// 3 K-blocks ⇒ `out_mask` length 4, `op_mask` length 12. The setup
/// drives a checkerboard out-mask (zero the (0,1)/(1,0) blocks) and a
/// partial op-mask (drop the middle K-block) so both predication paths
/// fire in one dispatch. `SimdGroup2D` grid `(N/BN, M/BM, 1)`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// Naive masked fp32 reference. `out_mask` is `[m/bm * n/bn]`;
    /// `op_mask` is `[m/bm * n/bn * k/16]`, row-major over (out-block, K-block).
    #[allow(clippy::too_many_arguments)]
    fn naive_masked_matmul(
        a: &[f32],
        b: &[f32],
        out_mask: &[f32],
        op_mask: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bm: usize,
        bn: usize,
    ) -> Vec<f32> {
        let n_n_blocks = n / bn;
        let n_k_blocks = k / 16;
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let blk = (mi / bm) * n_n_blocks + (ni / bn);
                if out_mask[blk] == 0.0 {
                    out[mi * n + ni] = 0.0;
                    continue;
                }
                let mut acc = 0.0f32;
                for kb in 0..n_k_blocks {
                    let opm = op_mask[blk * n_k_blocks + kb];
                    for ki in (kb * 16)..(kb * 16 + 16) {
                        acc += a[mi * k + ki] * opm * b[ki * n + ni];
                    }
                }
                out[mi * n + ni] = acc;
            }
        }
        out
    }

    /// Build a masked-GEMM correctness setup with a checkerboard out-mask
    /// and a middle-K-block-dropped op-mask. The mask buffers are `T`-typed.
    fn masked_setup(kernel: Kernel, bm: u32, bn: u32, tpg: u32, dt: DType) -> TestSetup {
        let (m, n, k) = (bm as usize * 2, bn as usize * 2, 48usize);
        let n_out_blocks = (m / bm as usize) * (n / bn as usize); // 4
        let n_op = n_out_blocks * (k / 16); // 12
        // Checkerboard output mask: keep (0,0) and (1,1), zero the rest.
        let out_mask = vec![1.0f32, 0.0, 0.0, 1.0];
        // Drop the middle K-block of every output block.
        let op_mask: Vec<f32> = (0..n_op).map(|i| if i % 3 == 1 { 0.0 } else { 1.0 }).collect();
        let a = unpack_f32(&pack_f32(&ramp(m * k, 19, 7.0), dt), dt);
        let b = unpack_f32(&pack_f32(&ramp(k * n, 23, 9.0), dt), dt);
        // Masks are dtype-rounded too (they round-trip exactly for 0/1).
        let om = unpack_f32(&pack_f32(&out_mask, dt), dt);
        let opm = unpack_f32(&pack_f32(&op_mask, dt), dt);
        let expected = naive_masked_matmul(&a, &b, &om, &opm, m, k, n, bm as usize, bn as usize);
        TestSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::from_vec("out_mask", pack_f32(&out_mask, dt), dt))
            .input(TestBuffer::from_vec("op_mask", pack_f32(&op_mask, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n as u32 / bn, m as u32 / bm, 1, [tpg, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_masked_64x64x16_2x2(dt: DType) -> TestSetup {
        masked_setup(mt_steel_gemm_masked_64x64x16_2x2::kernel_ir_for(dt), 64, 64, 128, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_masked_32x32x16_2x2(dt: DType) -> TestSetup {
        masked_setup(mt_steel_gemm_masked_32x32x16_2x2::kernel_ir_for(dt), 32, 32, 128, dt)
    }
}
