//! MPP-backed MoE grouped int8 BGEMM — `mt_moe_gather_qmm_mma_int8_bm8_mpp`.
//!
//! BM=8 int8 sibling of `mt_moe_gather_qmm_mma_int4_bm8_mpp`. Same algorithm
//! and call-site signature; the weight layout changes from int4 (8 nibbles/u32)
//! to int8 (4 bytes/u32), doubling the number of weight u32s per row but
//! halving the packing inner-loop work per word.
//!
//! ## Direct-input matmul2d
//!
//! Descriptor `matmul2d_descriptor(8, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)`. With M=8 the inputs cannot be cooperative tensors
//! (Apple's MPP path requires at least one of M/N/K ≥ 16 for cooperative
//! tensor descriptors), so A and B are passed as **direct** `metal::tensor`
//! views over threadgroup memory — the `direct_inputs` form.
//!
//! ## int4 → int8 lane mapping (BM=8)
//!
//! W tile size: BN(32) × BK(16) = 512 elements.
//!
//! - **int4**: 32 lanes × 2 packs/lane × 8 nibbles/pack = 512 ✓
//!   - pack_id = lane*2 + _pi; w_row = pack_id/2; pack_col = pack_id%2
//!   - k_off = kb + pack_col*8; dst = w_row*16 + pack_col*8
//!   - Extracts 8 nibbles: `(packed >> (j*4)) & 0xf`
//!
//! - **int8**: 32 lanes × 4 packs/lane × 4 bytes/pack = 512 ✓
//!   - pack_id = lane*4 + _pi; w_row = pack_id/4; pack_col = pack_id%4
//!   - k_off = kb + pack_col*4; dst = w_row*16 + pack_col*4
//!   - Extracts 4 bytes: `(packed >> (j*8)) & 0xff`
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
//! Correctness validated by `tests/moe_gather_qmm_mpp_bm8_int8_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// MPP MoE int8 grouped BGEMM, BM=8 / BN=32 / BK=16, one simdgroup,
/// direct-input `matmul2d`. Signature matches `…_int4_bm8_mpp`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int8_bm8_mpp<T>(
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

    // int8: 4 bytes per u32 → k_in / 4 packs per weight row.
    let packs_per_row = k_in / 4u32;
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

        // Walk forward to find the first row whose expert differs, clamping
        // sub_end at the tile boundary or at m_total.
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

                // Dequant W → ws.
                //
                // int8 lane mapping: 32 lanes × 4 packs/lane × 4 bytes/pack
                //   = 512 = BN(32) × BK(16).
                //
                // pack_id = lane*4 + _pi   (0..127 — covers 32 w_rows × 4 packs/row)
                // w_row   = pack_id / 4    (0..31 = BN rows)
                // pack_col= pack_id % 4    (0..3 — selects which of the 4 u32s in BK)
                //
                // k_off = kb + pack_col*4  (byte-offset of this pack's first element)
                // dst   = w_row*16 + pack_col*4 (flat index into ws threadgroup buf)
                //
                // Each pack holds 4 bytes (one per K-element); inner _j in 0..4
                // extracts byte j via (packed >> (j*8)) & 0xff.
                for _pi in range(0u32, 4u32, 1u32) {
                    let pack_id = lane * 4u32 + _pi;
                    let w_row = pack_id / 4u32;
                    let pack_col = pack_id % 4u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 4u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 4u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 4u32;
                    for _j in range(0u32, 4u32, 1u32) {
                        let q = ((packed >> (_j * 8u32)) & 255u32).cast::<f32>();
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

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int8_bm8_mpp",
        kernel_name: "mt_moe_gather_qmm_mma_int8_bm8_mpp",
        kernel_ir: mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::Op;

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(dt);
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
        let k = mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(DType::BF16);
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
