//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm8_mpp`.
//!
//! BM=8 sibling of `mt_moe_gather_qmm_mma_int4_bm16_mpp`. Same algorithm
//! and call-site signature; the per-TG row tile shrinks to 8 so the
//! kernel doesn't waste half the MMA tile on zero-padded rows at
//! decode-time MoE shapes (T=1, topK=8 → m_total=8 after gather/permute).
//!
//! Expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics —
//! no `Op::InlineMsl`.
//!
//! ## Direct-input matmul2d
//!
//! The descriptor is `matmul2d_descriptor(8, 32, 16, ta=false, tb=true,
//! tc=false, multiply_accumulate)`. With M=8 the inputs cannot be
//! cooperative tensors (Apple's MPP path constrains the cooperative-tensor
//! descriptor dims), so A and B are passed as **direct** `metal::tensor`
//! views over threadgroup memory — the `direct_inputs` form of
//! `coop_tile_setup` / `coop_tile_load_*` / `coop_tile_run`.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for `T = bf16`, else `T` — Apple's `matmul2d`
//! mishandles `bfloat` operands, and `half` losslessly covers bf16's
//! mantissa. Accumulation is fp32.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/8), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_bm8_correctness.rs`.

use metaltile::kernel;

/// MPP MoE int4 grouped BGEMM, BM=8 / BN=32 / BK=16, one simdgroup,
/// direct-input `matmul2d`. Signature matches `…_bm16_mpp`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4_bm8_mpp<T>(
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
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    // Descriptor 8×32×16, direct-input (M=8 → not a cooperative tensor).
    // direct_inputs=true; A view = [K=16, M=8], B view = [K=16, N=32].
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+8, kb..kb+16] → xs. 32 lanes × 4.
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W → ws. 32 lanes × 2 packs/lane, 8 nibbles/pack.
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32;
                    let pack_col = pack_id % 2u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 8u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let q = ((packed >> (_j * 4u32)) & 15u32).cast::<f32>();
                        threadgroup_store("ws", dst + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            // C [M=8, N=32] row-major → extents N,M = 32,8.
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
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
            let k = mt_moe_gather_qmm_mma_int4_bm8_mpp::kernel_ir_for(dt);
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
        let k = mt_moe_gather_qmm_mma_int4_bm8_mpp::kernel_ir_for(DType::BF16);
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

/// New-syntax correctness test for the MPP MoE int4 BGEMM (BM=8). Shares the
/// per-row-`indices` int4 dequant-then-matmul oracle with the BM=16 sibling;
/// only the m-tile height (BM=8 → ceil(M/8) m-tiles) differs.
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, ceil(m_total/8), 1, [32,1,1])`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_moe_gather_qmm_mma_int4_bm8_mpp;
    use crate::ffai::moe_mpp_shared::{MmaTestShape, int4_indexed_setup};

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int4_bm8_mpp(dt: DType) -> TestSetup {
        // BM=8 → ceil(64/8)=8 m-tiles, BN=32 → 64/32=2 n-tiles.
        int4_indexed_setup(
            mt_moe_gather_qmm_mma_int4_bm8_mpp::kernel_ir_for(dt),
            MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 },
            32, // bn
            8,  // bm
            32, // tpg
            dt,
        )
    }
}

/// New-syntax benchmark for the MPP MoE int4 BGEMM (BM=8). Qwen3.6-A3B-ish.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_moe_gather_qmm_mma_int4_bm8_mpp;
    use crate::ffai::moe_mpp_shared::{MmaBenchShape, int4_mma_bench};

    #[bench(name = "ffai/moe_mpp/gather_qmm_mma_int4_bm8", dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int4_bm8_mpp(dt: DType) -> BenchSetup {
        int4_mma_bench(
            mt_moe_gather_qmm_mma_int4_bm8_mpp::kernel_ir_for(dt),
            MmaBenchShape {
                bits: 4,
                bn: 32,
                bm: 8,
                tpg: 32,
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
