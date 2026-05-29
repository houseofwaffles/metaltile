//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_qmm_mma_mpp` — production int4 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! This is the MPP (MetalPerformancePrimitives) counterpart of
//! `mt_qmm_mma`. It mirrors the same algorithm — int4 weights dequantized
//! into threadgroup memory once per K-block, then a per-simdgroup matmul
//! against the fp T X-tile — but replaces the manual 8×8 `simdgroup_matmul`
//! ladder with one cooperative `matmul2d` per SG per K-block.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`.
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
//!
//! Per-K-block layout (cooperative, all 128 lanes):
//!   1. X-tile coop-load → `Xs[BM × TG_LD=36]`
//!   2. W-tile coop-dequant int4 → `Ws[BN × TG_LD=36]` in fp32 (staged
//!      into `coop_stage(T)` tile — MSL implicitly converts `half = float`)
//!   3. `threadgroup_barrier()`
//!   4. Each SG calls `coop_tile_run("gemm")` — `ct_c` persists across
//!      K-blocks via `multiply_accumulate`.
//!   5. `threadgroup_barrier()`
//!
//! After all K-blocks, each SG stores its 16×16 fp32 ct_c into a per-SG
//! slot of `OutScratch`, then all 32 lanes coop-write it to `out` (cast to T).

use metaltile::kernel;

/// MPP int4 quantized matmul `Out = X · dequant(W)`. Params:
///   `w [n, k/8]` int4 packed (8 nibbles/u32),
///   `scales`/`biases [n, k/group_size]` (T),
///   `x [m, k]` (T), `out [m, n]` (T). group_size = 64.
#[kernel(
    bench(
        op="quantized",
        subop="qmm_mma_mpp",
        class=GenericEmpty,
        tol=5e-2,
        kernel_mode=Reduction,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_qmm_mma_mpp<T>(
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
    // 2×2 warp grid.
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
    // Per-lane coordinates: 128 lanes × 8 X-elems = 1024 = BM*BK = 32*32.
    // Same coordinate doubles as the W (N-direction) row/pack offsets.
    let x_m_row = lane_in_tg / 4u32; // 0..32 (= w_row)
    let x_k_quad = lane_in_tg & 3u32; // 0..4    (= pack_in_row)
    let x_k_base = x_k_quad * 8u32; // 0/8/16/24
    let x_ws_base = x_m_row * 36u32 + x_k_base; // shared by Xs / Ws stages
    let packs_per_row = k / 8u32;
    // wn_plus_wr = w_n_base + w_row (W's N coordinate). Per-row scales /
    // biases base = (n_row) * gs_per_row; per-row pack base = ... * packs_per_row.
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
        // W dequant: 1 u32 pack per lane (8 nibbles, one full K-quad).
        let pack_dev = w_pack_row_base + kb / 8u32 + x_k_quad;
        let packed = load(w[pack_dev]);
        // Group index for these 8 nibbles. Baked-in group_size = 64.
        let k_off = kb + x_k_quad * 8u32;
        let g = k_off / 64u32;
        let sb_off = sb_base + g;
        let scale = load(scales[sb_off]).cast::<f32>();
        let bias = load(biases[sb_off]).cast::<f32>();
        // Unroll 8 nibble extractions: nib = (packed >> (ni*4)) & 15.
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = ((packed >> (_ni * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", x_ws_base + _ni, scale * nib + bias);
        }
        threadgroup_barrier();
        // Per-SG cooperative matmul.
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
            let k = mt_qmm_mma_mpp::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_mma_mpp");
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
        let k = mt_qmm_mma_mpp::kernel_ir_for(DType::BF16);
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
            let mut k = mt_qmm_mma_mpp::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_mma_mpp_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_mma_mpp_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }
}
