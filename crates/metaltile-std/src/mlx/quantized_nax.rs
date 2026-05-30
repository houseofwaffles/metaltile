//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_qmm_nax` — production int4 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! This is the MPP (MetalPerformancePrimitives) counterpart of
//! `mt_qmm_mma` (the simdgroup-ladder variant). It mirrors the same
//! algorithm — int4 weights dequantized into threadgroup memory once
//! per K-block, then a per-simdgroup matmul against the fp T X-tile —
//! but replaces the manual 8×8 `simdgroup_matmul` ladder with one
//! cooperative `matmul2d` per SG per K-block.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. Algorithmically identical to
//! `mt_qmm_mma_mpp`; the two co-exist so consumers can pick the
//! `_nax` vs `_mpp` name in their own dispatch tables.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for bf16 (Apple `matmul2d` mishandles
//! `bfloat` cooperative tensors), else `T`. Accumulation is fp32.
//!
//! ## Geometry
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM = WN = 2). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile (1024 outputs/TG).
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile + one 16×16×32 MMA
//!   per K-block (acc-mode `multiply_accumulate`).
//! - **TG row stride = BK + 4 (skew) = 36** — bank-conflict avoidance.
//! - **Group size baked at 64** — Qwen3.6-A3B default.
//! - **`KernelMode::Reduction`**.

use metaltile::kernel;

/// MPP int4 quantized matmul `Out = X · dequant(W)`. Same shape as
/// `mt_qmm_mma_mpp`; both kernels co-exist for naming compatibility.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_qmm_nax<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("Ws", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG × 16 × 16
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let pack_dev = w_pack_row_base + kb / 8u32 + x_k_quad;
        let packed = load(w[pack_dev]);
        let k_off = kb + x_k_quad * 8u32;
        let g = k_off / 64u32;
        let sb_off = sb_base + g;
        let scale = load(scales[sb_off]).cast::<f32>();
        let bias = load(biases[sb_off]).cast::<f32>();
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = ((packed >> (_ni * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", x_ws_base + _ni, scale * nib + bias);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_qmm_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.constexprs.len(), 3);

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn bf16_stages_through_half() {
        let k = mt_qmm_nax::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 must stage as half");
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_qmm_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }
}

/// New-syntax correctness + benchmark for `mt_qmm_nax` (MPP/NAX tensor-core
/// quantized matmul). The dequant-then-matmul math is identical to the
/// `quantized` qmm-MMA family (group_size 64 baked), so this reuses the
/// shared CPU oracle / bench builder rather than restating them.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_qmm_nax;
    use crate::mlx::quantized::kernel_tests::qx_setup;

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_qmm_nax(dt: DType) -> TestSetup {
        qx_setup(
            mt_qmm_nax::kernel_ir_for(dt),
            32,
            64,
            512,
            4,
            64,
            true,
            [2, 1, 1],
            [128, 1, 1],
            dt,
        )
    }
}

/// New-syntax benchmark for `mt_qmm_nax`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_qmm_nax;
    use crate::mlx::quantized::kernel_benches::qmb;

    #[bench(name = "mlx/quantized/qmm_nax", dtypes = [f32, f16, bf16])]
    fn bench_qmm_nax(dt: DType) -> BenchSetup {
        qmb(
            mt_qmm_nax::kernel_ir_for(dt),
            32,
            4096,
            4096,
            4,
            64,
            true,
            [128, 1, 1],
            [128, 1, 1],
            dt,
        )
    }
}
