//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_qmm_mma_mpp_int8` — production int8 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! Int8 counterpart of `mt_qmm_mma_mpp` (int4). Algorithmically identical
//! except the weight tensor packs **4 bytes per u32** (int8) instead of
//! 8 nibbles (int4). Accumulation, staging, and tile geometry are unchanged.
//!
//! ## int4 → int8 W-dequant changes
//!
//! | Property            | int4                          | int8                          |
//! |---------------------|-------------------------------|-------------------------------|
//! | Packs per row       | `k / 8`                       | `k / 4`                       |
//! | Packs per K-block   | 4 (BK=32 ÷ 8 nibbles)        | 8 (BK=32 ÷ 4 bytes)          |
//! | Packs per lane      | 1 (`x_k_quad` selects 1/4)   | 2 (each lane handles 2 packs) |
//! | Byte extraction     | `(packed >> (ni*4)) & 0xF`   | `(packed >> (bi*8)) & 0xFF`  |
//! | Elements per lane   | 8 nibbles from 1 u32          | 8 bytes from 2 u32s           |
//!
//! ## Geometry (unchanged from int4 MPP)
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM = WN = 2). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile (1024 outputs/TG).
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile.
//! - **TG row stride = BK + 4 (skew) = 36** — bank-conflict avoidance.
//! - **Group size baked at 32** — natural int8 group size.
//! - **`KernelMode::Reduction`**.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for bf16 (Apple `matmul2d` mishandles
//! `bfloat` cooperative tensors), else `T`. Accumulation is fp32.

use metaltile::kernel;

/// MPP int8 quantized matmul `Out = X · dequant(W)`. Params:
///   `w [n, k/4]` int8 packed (4 bytes/u32),
///   `scales`/`biases [n, k/group_size]` (T),
///   `x [m, k]` (T), `out [m, n]` (T). group_size = 32.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_qmm_mma_mpp_int8<T>(
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
    // 2×2 warp grid — same as int4 MPP.
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
    // Per-lane coordinates: 128 lanes, 32 rows, 4 column-groups.
    // Each lane handles x_k_base..+8 elements (8 K-elems per lane per K-block).
    let x_m_row = lane_in_tg / 4u32; // 0..32 (= w_row)
    let x_k_quad = lane_in_tg & 3u32; // 0..4  (column group selector)
    let x_k_base = x_k_quad * 8u32; // 0/8/16/24 — base K offset within BK
    let x_ws_base = x_m_row * 36u32 + x_k_base; // shared by Xs / Ws stages
    // int8: 4 bytes per u32 → packs_per_row = k/4.
    let packs_per_row = k / 4u32;
    // Per-row W addressing (N-direction). Same pattern as int4.
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        // Stage X[x_m_base + x_m_row, kb + x_k_base..+8] → Xs.
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        // W dequant (int8): each lane processes 2 u32 packs = 8 bytes = 8 K-elems.
        //
        // Pack layout in W buffer (per row of N):
        //   int4: packs at kb/8 + x_k_quad  (1 pack/lane × 8 nibbles = 8 K-elems)
        //   int8: packs at kb/4 + x_k_quad*2 + _pi  (2 packs/lane × 4 bytes = 8 K-elems)
        //
        // The inner loop _pi = 0..2 steps through the 2 consecutive u32 packs
        // that this lane owns within the current K-block.
        let w_kb_off = kb / 4u32 + x_k_quad * 2u32;
        for _pi in range(0u32, 2u32, 1u32) {
            let pack_dev = w_pack_row_base + w_kb_off + _pi;
            let packed = load(w[pack_dev]);
            // Group index: each pack covers 4 consecutive K-elements.
            // k_off = start K-element for this pack (absolute within W row).
            let k_off = kb + x_k_quad * 8u32 + _pi * 4u32;
            let g = k_off / 32u32; // group_size = 32 (baked)
            let sb_off = sb_base + g;
            let scale = load(scales[sb_off]).cast::<f32>();
            let bias = load(biases[sb_off]).cast::<f32>();
            // Unroll 4 byte extractions: byte = (packed >> (bi*8)) & 0xFF.
            // Writes 4 elements per pack × 2 packs = 8 elements total per lane,
            // matching x_ws_base + _pi*4 + 0..4 within Ws.
            for _bi in range(0u32, 4u32, 1u32) {
                let byte_val = ((packed >> (_bi * 8u32)) & 255u32).cast::<f32>();
                threadgroup_store("Ws", x_ws_base + _pi * 4u32 + _bi, scale * byte_val + bias);
            }
        }
        threadgroup_barrier();
        // Per-SG cooperative matmul — identical to int4 MPP.
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    // Coop-write OutScratch → out. 32 lanes × 8 elems = 256 = 16×16 per SG.
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
            let k = mt_qmm_mma_mpp_int8::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_mma_mpp_int8");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.params[0].name, "w");
            assert_eq!(k.params[1].name, "scales");
            assert_eq!(k.params[2].name, "biases");
            assert_eq!(k.params[3].name, "x");
            assert_eq!(k.params[4].name, "out");
            assert!(k.params[4].is_output);
            assert_eq!(k.constexprs.len(), 3);
            assert_eq!(k.constexprs[0].name.name(), "k");
            assert_eq!(k.constexprs[1].name.name(), "n");
            assert_eq!(k.constexprs[2].name.name(), "gs_per_row");

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileStoreC { .. })));
        }
    }

    /// bf16 must stage through `half` for matmul2d compatibility.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_qmm_mma_mpp_int8::kernel_ir_for(DType::BF16);
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

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_qmm_mma_mpp_int8::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_mma_mpp_int8_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_mma_mpp_int8_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }
}

/// New-syntax correctness + benchmark for `mt_qmm_mma_mpp_int8` (MPP/NAX tensor-core
/// quantized matmul). The dequant-then-matmul math is identical to the
/// `quantized` qmm-MMA family (group_size 32 baked), so this reuses the
/// shared CPU oracle / bench builder rather than restating them.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_qmm_mma_mpp_int8;
    use crate::mlx::quantized::kernel_tests::qx_setup;

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_qmm_mma_mpp_int8(dt: DType) -> TestSetup {
        qx_setup(
            mt_qmm_mma_mpp_int8::kernel_ir_for(dt),
            32,
            64,
            512,
            8,
            32,
            true,
            [2, 1, 1],
            [128, 1, 1],
            dt,
        )
    }
}

/// New-syntax benchmark for `mt_qmm_mma_mpp_int8`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_qmm_mma_mpp_int8;
    use crate::mlx::quantized::kernel_benches::qmb;

    #[bench(name = "mlx/quantized/qmm_mma_mpp_int8", dtypes = [f32, f16, bf16])]
    fn bench_qmm_mma_mpp_int8(dt: DType) -> BenchSetup {
        qmb(
            mt_qmm_mma_mpp_int8::kernel_ir_for(dt),
            32,
            4096,
            4096,
            8,
            32,
            true,
            [128, 1, 1],
            [128, 1, 1],
            dt,
        )
    }
}
