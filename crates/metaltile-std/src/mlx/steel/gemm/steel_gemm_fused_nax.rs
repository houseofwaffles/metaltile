//! `mt_steel_gemm_fused_nax` — plain fused GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the `nn` (non-transposed) steel-gemm
//! `C = A · B` where `A: [M, K]`, `B: [K, N]`, `C: [M, N]`, all row-major.
//! Requires the Metal 4 `MetalPerformancePrimitives` framework (macOS 26+)
//! and Apple10+ hardware; runtime-gated via `Context::chip_family()`.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. The `coop_tile_*` ops lower to the
//! `mpp::tensor_ops::matmul2d` cooperative-tensor calls; codegen emits
//! the framework include automatically. This is the cooperative-tensor
//! counterpart of `steel_gemm_fused`.
//!
//! ## bf16 staging
//!
//! Apple's `matmul2d` mishandles `bfloat` cooperative tensors, so bf16
//! activations are staged through `half` (10-bit mantissa losslessly
//! covers bf16's 7; accumulation is fp32 regardless). The DSL
//! `coop_stage(T)` form yields `half` for `T = bf16` and `T` otherwise —
//! the kernel stays generic while its threadgroup tiles and cooperative
//! tensors pick up the staged type.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 16, 32, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — `K=32` satisfies Apple's "at least one of
//! M/N/K = 32" rule. `tb=true` reads `Ws` in its native `[N, K]` layout
//! (the gather/transpose into `Ws` happens at stage time).
//!
//! ## Geometry
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM=WN=2). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile per TG.
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile and runs one
//!   `16×16×32` `matmul2d` per K-block.
//! - **TG-row skew = 4 elems**: `Xs` / `Ws` are `32 × 36` to scatter
//!   32-bank conflicts on the column reads inside `matmul2d`'s frag load.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`** — all loads are
//!   unconditional; ragged shapes read out of bounds. Callers must pad.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index, not the global thread index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_fused_nax_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// NAX fused GEMM `C = A · B`. Params:
///   `a [m, k]`, `b [k, n]`, `out [m, n]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_steel_gemm_fused_nax<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;

    // 2×2 warp grid: sm / sn pick this SG's 16×16 sub-tile.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;

    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;

    // TG row stride = BK + 4 (skew) = 36 — scatter bank conflicts on the
    // column reads the matmul2d frag load performs inside Xs / Ws.
    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("Ws", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG × 16 × 16

    // Cooperative-tensor descriptor 16×16×32, ta=false tb=true tc=false,
    // multiply_accumulate (one MMA per K-block sums into the same C tile).
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");

    // Per-lane stage coordinates: 128 lanes × 8 elems = 1024 = 32 × 32.
    // x_m_row also doubles as w_n_row (Ws is `[N, K]` in the same layout).
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;

    // Shared TG-tile write offset (Xs[m_row][k_base..+8] / Ws[n_row][k_base..+8]).
    let x_ws_base = x_m_row * 36u32 + x_k_base;

    // Per-SG offsets into the shared staging tiles + scratch.
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;

    // The N column this lane gathers from device B (for the transposed Ws).
    let b_n = w_n_base + x_m_row;

    // K-loop: kb = 0..k step BK = 32.
    for kb in range(0u32, k, 32u32) {
        // Stage A[x_m_base + x_m_row, kb + x_k_base..+8] → Xs.
        let a_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let av = load(a[a_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, av);
        }

        // Stage B^T[w_n_base + x_m_row, kb + x_k_base..+8] → Ws.
        // Device read: B[(kb + x_k_base + i) * n + b_n].
        let b_k_base = kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let bv = load(b[(b_k_base + _i) * n + b_n]).cast::<f32>();
            threadgroup_store("Ws", x_ws_base + _i, bv);
        }

        threadgroup_barrier();

        // Per-SG cooperative-tensor matmul.
        // extents: stride-1 along K (inner) = TG_LD = 36, slow = 16.
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");

        threadgroup_barrier();
    }

    // Per-SG fp32 result → OutScratch slot (16×16, packed contiguous).
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();

    // Coop-write OutScratch → out. 32 lanes × 8 elems = 256 = 16 × 16 per SG.
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

inventory::submit! {
    BenchSpec {
        op: "steel_gemm",
        subop: "fused_nax",
        kernel_name: "mt_steel_gemm_fused_nax",
        kernel_ir: mt_steel_gemm_fused_nax::kernel_ir_for,
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
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::Op;

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_steel_gemm_fused_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_fused_nax");
            assert_eq!(k.params.len(), 3);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "out");
            assert!(k.params[2].is_output);
            assert_eq!(k.constexprs.len(), 2);
            assert_eq!(k.constexprs[0].name.name(), "k");
            assert_eq!(k.constexprs[1].name.name(), "n");

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileZero { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileStoreC { .. })));
        }
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_steel_gemm_fused_nax::kernel_ir_for(DType::BF16);
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

    /// Codegen sanity — MPP header + descriptor + the 32×32 geometry.
    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_steel_gemm_fused_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_fused_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_fused_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
        }
    }
}
