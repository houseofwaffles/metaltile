//! MPP-backed MoE int8 grouped BGEMM — `mt_moe_gather_qmm_mma_int8_bm16_mpp`.
//!
//! Int8 analogue of `moe_mpp::mt_moe_gather_qmm_mma_int4_bm16_mpp`. Same
//! BM=16 / BN=32 / BK=16 MPP cooperative-tensor tiling; the only change is
//! the weight coop-dequant inner loop:
//!
//!   int4: 32 lanes × 2 packs/lane × 8 nibbles/pack = 512 = BN×BK ✓
//!   int8: 32 lanes × 4 packs/lane × 4 bytes/pack   = 512 = BN×BK ✓
//!
//! Each uint32 holds 4 consecutive unsigned-byte codes in LSB-first order.
//! `packs_per_row = k_in / 4` (4 bytes/u32 vs 8 nibbles/u32 for int4).
//!
//! ## bf16 staging
//!
//! Same `coop_stage(T)` trick as the int4 MPP kernel: bf16 activations are
//! staged through `half` so `mpp::tensor_ops::matmul2d` sees a supported
//! cooperative-tensor dtype. Accumulation is fp32.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — identical to the int4 MPP descriptor; only the
//! threadgroup W tile contents differ.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/16), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`,
//!   `(k_in / 4) % 4 == 0` (i.e. `k_in % 16 == 0`).
//! - macOS 26+ / Metal 4; on older toolchains a zero-write stub is emitted.
//!
//! Correctness: `tests/moe_gather_qmm_mpp_int8_correctness.rs` (cosine ≥ 0.999).

use metaltile::{bench_kernel, kernel};

/// MPP MoE int8 grouped BGEMM, BM=16 / BN=32 / BK=16, one simdgroup.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/4]` (int8
/// packed, 4 bytes/uint32), `scales`/`biases [n_experts, n_out,
/// k_in/group]`, `indices [m_total]` (per-row expert id), `out
/// [m_total, n_out]`.
#[bench_kernel(
    op="moe",
    subop="gather_qmm_mma_int8_bm16_mpp",
    class=GenericEmpty,
    tol=5e-2,
    kernel_mode=Reduction,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int8_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;

    // int8: 4 bytes per u32 → packs_per_row = k_in / 4.
    let packs_per_row = k_in / 4u32;
    let groups_per_row = k_in / group_size;

    // Threadgroup staging tiles. `coop_stage(T)` = half for bf16, else T.
    // `out_scratch` is fp32: `coop_tile_store_c` destination must match the
    // accumulator type.
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32

    // MPP descriptor 16×32×16, ta=false tb=true tc=false, accumulate.
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );

    // Walk the BM=16 rows in contiguous-expert sub-runs (identical to int4 MPP).
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);

        // Find run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }

        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;

            coop_tile_zero("gemm");

            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }

                // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws.
                // int8: 32 lanes × 4 packs/lane × 4 bytes/pack = 512 = BN×BK.
                // Lane assignment:
                //   pack_id = lane * 4 + _pi       (0..127, but we have 32 lanes so 4 iters)
                //   w_row   = pack_id / 4           (0..31: which BN row)
                //   pack_col = pack_id % 4          (0..3: which uint32 in BK=16 slice)
                //   k_off   = kb + pack_col * 4     (byte offset of first element in pack)
                //
                // 32 lanes × 4 iters × 4 bytes/pack = 512 elements = 32 rows × 16 cols ✓
                for _pi in range(0u32, 4u32, 1u32) {
                    let pack_id = lane * 4u32 + _pi;
                    let w_row = pack_id / 4u32; // 0..31 (BN rows)
                    let pack_col = pack_id % 4u32; // 0..3 (BK=16 → 4 packs × 4 bytes)
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 4u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    // k_off = byte offset of the first element in this pack within the row.
                    let k_off = kb + pack_col * 4u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    // Extract 4 unsigned byte codes (LSB-first).
                    let q0 = (packed & 255u32).cast::<f32>();
                    let q1 = ((packed >> 8u32) & 255u32).cast::<f32>();
                    let q2 = ((packed >> 16u32) & 255u32).cast::<f32>();
                    let q3 = ((packed >> 24u32) & 255u32).cast::<f32>();
                    // Write to threadgroup ws at the correct row/col position.
                    let dst = w_row * 16u32 + pack_col * 4u32;
                    threadgroup_store("ws", dst, s * q0 + b);
                    threadgroup_store("ws", dst + 1u32, s * q1 + b);
                    threadgroup_store("ws", dst + 2u32, s * q2 + b);
                    threadgroup_store("ws", dst + 3u32, s * q3 + b);
                }

                threadgroup_barrier();

                // A = xs [M=16, K=16] (ta=false → extents K,M = 16,16).
                // B = ws [N=32, K=16] (tb=true  → extents K,N = 16,32).
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");

                threadgroup_barrier();
            }

            // C [M=16, N=32] row-major → extents N,M = 32,16.
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();

            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::Op;

    use super::*;
    use crate::bench_types::DType;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_moe_gather_qmm_mma_int8_bm16_mpp");
            assert_eq!(k.params.len(), 6);
            assert!(k.params[5].is_output);
            assert_eq!(k.constexprs.len(), 4);
            // No raw inline MSL — the matmul is CoopTile* ops.
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half for matmul2d");
    }

    /// Codegen sanity — the MPP header + descriptor land in the MSL.
    #[test]
    fn codegen_emits_mpp_include() {
        let mut k = mt_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(DType::F32);
        k.name = "mt_moe_gather_qmm_mma_int8_bm16_mpp_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int8_bm16_mpp_f32"));
    }
}
