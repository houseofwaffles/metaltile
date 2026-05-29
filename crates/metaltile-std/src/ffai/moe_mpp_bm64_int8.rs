//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped int8 BGEMM — `mt_moe_gather_qmm_mma_int8_bm64_mpp`.
//!
//! BM=BN=64, BK=32 int8 variant of `mt_moe_gather_qmm_mma_int4_bm64_mpp`.
//! Runs **4 simdgroups** in a 2×2 warp grid over a 64×64 tile — each SG owns
//! a 32×32 sub-tile and a 32×32×32 `matmul2d`. For long-context prefill the
//! larger tile amortises the int8 dequant across more output.
//!
//! ## int4 → int8 lane mapping (BM=64)
//!
//! W tile size: BN(64) × BK(32) = 2048 elements.
//!
//! - **int4**: 128 lanes × 2 packs/lane × 8 nibbles/pack = 2048 ✓
//!   - pack_id = lane_in_tg*2 + _pi; w_row = pack_id/4; pack_in_row = pack_id%4
//!   - k_off = kb + pack_in_row*8; ws_base = w_row*32 + pack_in_row*8
//!   - Extracts 8 nibbles: `(packed >> (j*4)) & 0xf`
//!
//! - **int8**: 128 lanes × 4 packs/lane × 4 bytes/pack = 2048 ✓
//!   - pack_id = lane_in_tg*4 + _pi; w_row = pack_id/8; pack_in_row = pack_id%8
//!   - k_off = kb + pack_in_row*4; ws_base = w_row*32 + pack_in_row*4
//!   - Extracts 4 bytes: `(packed >> (j*8)) & 0xff`
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(32, 32, 32, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — all dims 32, so the inputs are cooperative
//! tensors (not the direct-input path the `…_bm8` variant needs).
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for `T = bf16`, else `T`. Apple's `matmul2d`
//! mishandles `bfloat` cooperative tensors; `half` losslessly covers
//! bf16's mantissa. Accumulation is fp32.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/64, ceil(M/64), 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2×2 warp grid).
//! - `k_in % 32 == 0`, `n_out % 64 == 0`, `group_size` divides `k_in`.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_bm64_int8_correctness.rs`.

use metaltile::kernel;

/// MPP MoE int8 grouped BGEMM, BM=BN=64 / BK=32, 4 simdgroups (2×2).
/// Signature matches `…_int4_bm64_mpp`.
#[kernel(
    bench(
        op="moe",
        subop="gather_qmm_mma_int8_bm64_mpp",
        class=GenericEmpty,
        tol=5e-2,
        kernel_mode=Reduction,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int8_bm64_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    // 2×2 warp grid: sg_m_base / sg_n_base select this SG's 32×32 sub-tile.
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    // int8: 4 bytes per u32 → k_in / 4 packs per weight row.
    let packs_per_row = k_in / 4u32;
    let groups_per_row = k_in / group_size;
    // X coop-load: 128 lanes × 16 contiguous K = 2048 = BM(64)×TG_LD(32).
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    threadgroup_alloc("Xs", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("Ws", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("OutScratch", 4096, f32); // 4 SG × 32 × 32
    // Descriptor 32×32×32, cooperative-tensor inputs, accumulate.
    coop_tile_setup(
        "gemm",
        32,
        32,
        32, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 64u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 64u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Walk forward to find the first row whose expert differs, clamping
        // sub_end at the tile boundary or at m_total.
        let mut sub_end = 64u32;
        let mut found = 0u32;
        for _ii in range(0u32, 64u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 64u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 64u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 64u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                // Stage X[m_tile_base..+64, kb..kb+32] → Xs. 128 lanes × 16.
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>();
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                // Dequant W → Ws.
                //
                // int8 lane mapping: 128 lanes × 4 packs/lane × 4 bytes/pack
                //   = 2048 = BN(64) × BK(32).
                //
                // pack_id     = lane_in_tg*4 + _pi   (0..511)
                // w_row       = pack_id / 8           (0..63 = BN rows)
                // pack_in_row = pack_id % 8           (0..7 — BK=32 → 8 u32s of 4 bytes)
                //
                // k_off   = kb + pack_in_row*4
                // ws_base = w_row*32 + pack_in_row*4
                //
                // Each pack holds 4 bytes (one per K-element); inner _j in 0..4
                // extracts byte j via (packed >> (j*8)) & 0xff.
                for _pi in range(0u32, 4u32, 1u32) {
                    let pack_id = lane_in_tg * 4u32 + _pi;
                    let w_row = pack_id / 8u32; // 0..63 (BN rows)
                    let pack_in_row = pack_id & 7u32; // 0..7 (BK=32 → 8 packs)
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 4u32
                        + pack_in_row;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_in_row * 4u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let ws_base = w_row * 32u32 + pack_in_row * 4u32;
                    for _j in range(0u32, 4u32, 1u32) {
                        let q = ((packed >> (_j * 8u32)) & 255u32).cast::<f32>();
                        threadgroup_store("Ws", ws_base + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                // Per-SG 32×32 sub-tile views into Xs / Ws (offset by the
                // SG's 32-row span × TG_LD=32). extents<32, 32> = K-inner.
                coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
                coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            // Store this SG's 32×32 fp32 result into its OutScratch slot.
            coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
            threadgroup_barrier();
            // Coop-write OutScratch → out. 128 lanes × 32 = 4096 = BM*BN.
            // Each (mr, nc) lives in SG `(mr/32)*2 + (nc/32)`'s scratch.
            for _e in range(0u32, 32u32, 1u32) {
                let flat = lane_in_tg * 32u32 + _e;
                let mr = flat / 64u32;
                let nc = flat & 63u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
                    let v = threadgroup_load(
                        "OutScratch",
                        src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                    );
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::Op;

    use super::*;
    use crate::bench_types::DType;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int8_bm64_mpp::kernel_ir_for(dt);
            assert_eq!(k.params.len(), 6);
            assert_eq!(k.constexprs.len(), 4);
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn bf16_stages_through_half() {
        let k = mt_moe_gather_qmm_mma_int8_bm64_mpp::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half");
    }
}

/// New-syntax benchmark for the MPP MoE int8 BGEMM (BM=BN=64). Bench-only:
/// correctness covered by the int4 MPP bm64 test + legacy int8 A/B test.
/// `bits=8` → `k_in/4` u32 weight words/row.
///
/// Grid (Reduction, 4 simdgroups per TG): `grid_3d(n_out/64, ceil(m_total/64), 1, [128,1,1])`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_moe_gather_qmm_mma_int8_bm64_mpp;
    use crate::ffai::moe_mpp_shared::{MmaBenchShape, int4_mma_bench};

    #[bench(name = "ffai/moe_mpp/gather_qmm_mma_int8_bm64", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int8_bm64_mpp(dt: DType) -> BenchSetup {
        int4_mma_bench(
            mt_moe_gather_qmm_mma_int8_bm64_mpp::kernel_ir_for(dt),
            MmaBenchShape {
                bits: 8,
                bn: 64,
                bm: 64,
                tpg: 128,
                m_total: 1024,
                n_out: 256,
                k_in: 2048,
                n_experts: 128,
                group_size: 64,
            },
            dt,
        )
    }
}
