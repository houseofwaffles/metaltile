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
        #[kernel(
            bench(
                op="steel_gemm_masked",
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
