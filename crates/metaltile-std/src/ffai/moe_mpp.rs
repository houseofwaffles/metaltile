//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm16_mpp`.
//!
//! Routes the per-tile matmul through Apple's MetalPerformancePrimitives
//! `mpp::tensor_ops::matmul2d`. Algorithmically mirrors
//! `mt_moe_gather_qmm_mma_int4_bm16` (BM=16, BN=32, per-TG expert
//! sub-runs, per-row expert dispatch); the inner `simdgroup_matmul`
//! 8×8 frags are replaced by a single `16×32×16` MPP descriptor.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. The `coop_tile_*` ops lower to the
//! `mpp::tensor_ops::matmul2d` cooperative-tensor calls; codegen emits
//! the framework include automatically.
//!
//! ## bf16 staging
//!
//! Apple's `matmul2d` mishandles `bfloat` cooperative tensors, so bf16
//! activations are staged through `half` (10-bit mantissa losslessly
//! covers bf16's 7; accumulation is fp32 regardless). The DSL
//! `coop_stage(T)` form yields `half` for `T = bf16` and `T` otherwise —
//! the kernel stays generic over `T` while its threadgroup tiles and
//! cooperative tensors pick up the staged type.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — `N=32` satisfies Apple's "at least one of
//! M/N/K = 32" rule; `tb=true` reads W in its native `[N, K]` layout;
//! `multiply_accumulate` spans the K loop without an explicit add.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/16), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a
//!   linkable stub.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_correctness.rs`
//! (cosine ≥ 0.999 vs the m1 scalar oracle).

use metaltile::{bench_kernel, kernel};

/// MPP MoE int4 grouped BGEMM, BM=16 / BN=32 / BK=16, one simdgroup.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (int4
/// packed, 8 nibbles/uint32), `scales`/`biases [n_experts, n_out,
/// k_in/group]`, `indices [m_total]` (per-row expert id), `out
/// [m_total, n_out]`.
#[bench_kernel(
    op="moe",
    subop="gather_qmm_mma_int4_bm16_mpp",
    class=GenericEmpty,
    tol=5e-2,
    kernel_mode=Reduction,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4_bm16_mpp<T>(
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

    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;

    // Threadgroup staging tiles. `coop_stage(T)` = half for bf16, else T —
    // the matmul reads these as cooperative tensors. `out_scratch` is
    // fp32: `coop_tile_store_c` requires the destination elem-type to
    // match the accumulator.
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

    // Walk the BM=16 rows in contiguous-expert sub-runs.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);

        // Find the run end — first row whose expert differs (or OOB).
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
                // 32 lanes × 2 packs/lane; 8 nibbles/pack.
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32; // 0..31 (BN rows)
                    let pack_col = pack_id % 2u32; // 0..1 (BK=16 → 2 packs)
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
            let k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_moe_gather_qmm_mma_int4_bm16_mpp");
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
        let k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::BF16);
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
        let mut k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::F32);
        k.name = "mt_moe_gather_qmm_mma_int4_bm16_mpp_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int4_bm16_mpp_f32"));
    }
}
