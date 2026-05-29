//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel segmented GEMM — #[kernel] DSL vs MLX
//! `metal/steel/gemm/kernels/steel_gemm_segmented.metal`.
//!
//! Batched row-major matmul where each batch **segment** sums over a
//! different K-range of a shared `A` / `B`:
//!
//!   C[seg] = A[:, k_start(seg)..k_end(seg)]
//!            · B[k_start(seg)..k_end(seg), :]
//!
//! `A` is `[M, total_K]`, `B` is `[total_K, N]`, and the output is
//! `[n_segments, M, N]` — one `[M, N]` matrix per segment. A
//! `segments` descriptor buffer holds the `(k_start, k_end)` half-open
//! K-range of each segment. This is MLX's `segmented_mm`, the
//! ragged-K batched matmul used by variable-context attention and
//! segment-sum GEMMs.
//!
//! ## How the ragged K-range is expressed
//!
//! The DSL has no "ragged batched matmul" primitive, and it does not
//! need one: a segmented GEMM is the fused GEMM with a **3-D grid**
//! (`program_id<2>` = segment index) and a K-loop whose bounds are
//! read from the `segments` descriptor instead of being a constexpr.
//!
//!   - `segments[2*seg]` / `segments[2*seg + 1]` — the half-open
//!     `[k_start, k_end)` K-range. `k_start` is a multiple of 16 and
//!     `k_end - k_start` is a multiple of 16 (the BK contract); the
//!     K-loop steps `for kb in range(k_start, k_end, 16)`.
//!   - `program_id<2>()` selects the segment; the output base offset is
//!     `seg * m * n`, the A / B operands are shared (offsets keyed by
//!     the actual K index, which already encodes the segment range).
//!
//! Both the descriptor read and the variable loop bounds are ordinary
//! arithmetic over a `Tensor<u32>` operand and a `range(start, end, …)`
//! call — no new codegen primitive is required.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`. Must be a multiple of 32.
//! - **Grid: 3-D — `program_id<0>` = N-block, `program_id<1>` = M-block,
//!   `program_id<2>` = segment index.** One `[M, N]` output per segment.
//! - **`m % BM == 0`, `n % BN == 0`.** Each segment's `k_start` and
//!   `(k_end − k_start)` must be multiples of 16 (the BK contract).
//!   All loads are unconditional — ragged M / N shapes read OOB.
//! - **`segments` length `2 * n_segments`**, `u32`, laid out
//!   `[k_start_0, k_end_0, k_start_1, k_end_1, …]`. **`total_k`** is the
//!   shared A column count / B row count; it is the leading-dimension
//!   stride, *not* a per-segment K extent.
//! - **`KernelMode::SimdGroup2D`** so `program_id<i>` lowers to the
//!   threadgroup index `tid.{x,y,z}`, not the global thread index.

use metaltile::kernel;

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the
/// segmented GEMM. The outer `macro_rules!` substitutes the literals
/// before the `#[kernel]` body parser runs — see `steel_gemm_fused.rs`.
#[rustfmt::skip]
macro_rules! steel_gemm_segmented_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="steel_gemm_segmented",
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
            segments: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] total_k: u32,
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
            let seg = program_id::<2>(); // segment index
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

            // ── Segment K-range from the descriptor ──
            // segments[2*seg .. 2*seg+2) = [k_start, k_end).
            let k_start = load(segments[seg * 2u32]);
            let k_end = load(segments[seg * 2u32 + 1u32]);
            // This segment's output base offset (one [M,N] matrix each).
            let out_base = seg * m * n;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    let m_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;

                    // K-loop over this segment's [k_start, k_end) range.
                    // `total_k` is the shared leading-dimension stride.
                    for kb in range(k_start, k_end, 16) {
                        for _kf in range(0, n_kf, 1) {
                            let kf = kb + _kf * 8u32;
                            let sub_a = simdgroup_alloc::<T, 8, 8>();
                            let sub_b = simdgroup_alloc::<T, 8, 8>();

                            // A: [M, total_k] — column index is the
                            // absolute K (already in the segment range).
                            simdgroup_elem_store(
                                sub_a,
                                0,
                                load(a[(m_row + fm) * total_k + kf + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_a,
                                1,
                                load(a[(m_row + fm) * total_k + kf + fn1]).cast::<T>(),
                            );

                            // B: [total_k, N] — non-transposed layout.
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

                    // Store into this segment's [M, N] output slice.
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    store(out[out_base + (m_row + fm) * n + n_col + fn0], r0.cast::<T>());
                    store(out[out_base + (m_row + fm) * n + n_col + fn1], r1.cast::<T>());
                }
            }
        }
    };
}

// ── Block-shape instantiations ──────────────────────────────────────────
// 64×64×16 / 2×2 — the canonical large-tile shape (4 simdgroups).
steel_gemm_segmented_kernel!(
    mt_steel_gemm_segmented_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape for skinny M or N (4 simdgroups).
steel_gemm_segmented_kernel!(
    mt_steel_gemm_segmented_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);
