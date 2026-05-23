//! `mt_fp_qmm_nax` — fp4 (E2M1) quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the fp4 quantized matmul from MLX
//! `metal/kernels/fp_quantized_nax.metal`. Gated behind the `nax` Cargo
//! feature — the kernel requires the Metal 4 `MetalPerformancePrimitives`
//! framework (macOS 26+).
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`.
//!
//! Fp4 counterpart of `mt_qmm_nax`. Mirrors the same coop-load /
//! coop-dequant / `matmul2d` pattern — packed weights are dequantized
//! into threadgroup memory once per K-block, then per-simdgroup
//! `matmul2d` runs against the fp `T` X-tile — but swaps the int4
//! nibble-dequant for an **fp4 E2M1 codebook lookup**:
//!
//!   - Each 4-bit code is `[sign : 1][exp : 2][mantissa : 1]`.
//!   - The 3-bit magnitude indexes the E2M1 codebook
//!     `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (the nvfp4 levels — see
//!     MLX `fp4.h`). Computed via integer arithmetic on `two_m_int`
//!     (= value × 2) to avoid f32 constants:
//!     - subnormal (exp = 0): `two_m_int = mantissa`, `∈ {0, 1}`
//!     - normal (exp ≥ 1): `two_m_int = (mantissa + 2) · 2^(exp − 1)`,
//!       `∈ {2, 3, 4, 6, 8, 12}`
//!   - The sign bit (`code & 8`) negates the magnitude via `1 − 2·sign`.
//!   - The dequantized value is `scale · sign · two_m_int / 2`. Fp4
//!     quantization is **scale-only** — no per-group bias.
//!
//! 8 fp4 codes pack into one `u32`; one `u32` covers `BK/4 = 8` of a
//! BK=32 row. Per-K-block scale layout uses `GROUP_SIZE = 32` (one
//! scale per BK-block per N-row).
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
//! - **Group size baked at 32** — one scale per BK-block per N-row.
//! - **`KernelMode::Reduction`**.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/fp_quantized_nax_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// MPP fp4 (E2M1) quantized matmul `Out = X · dequant(W)`. Params:
///   `w [n, k/8]` fp4 packed (8 codes/u32),
///   `scales [n, k/group_size]` (T) — group_size = 32, scale-only,
///   `x [m, k]` (T), `out [m, n]` (T).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp_qmm_nax<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
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
        // Stage X[x_m_base + x_m_row, kb + x_k_base..+8] → Xs.
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }

        // W fp4-dequant: 1 u32 pack per lane (8 fp4 codes covering one K-quad).
        let pack_dev = w_pack_row_base + kb / 8u32 + x_k_quad;
        let packed = load(w[pack_dev]);
        // Group index for these 8 codes (group_size = 32).
        let k_off = kb + x_k_quad * 8u32;
        let g = k_off / 32u32;
        let scale = load(scales[sb_base + g]).cast::<f32>();

        // Per-code E2M1 dequant. Decode integer "value × 2" then divide
        // once by the shared 2.0 — saves 7 fp ops vs decoding floats.
        for _ni in range(0u32, 8u32, 1u32) {
            let code = (packed >> (_ni * 4u32)) & 15u32;
            let sign_bit = (code >> 3u32) & 1u32;
            let mag_bits = code & 7u32;
            let exp = mag_bits >> 1u32; // 0..3
            let mant = mag_bits & 1u32; // 0 or 1
            let is_subnormal = exp == 0u32;
            // safe_exp keeps (exp − 1) ≥ 0 for the shift; subnormal branch
            // ignores the result.
            let safe_exp = select(is_subnormal, 1u32, exp);
            let pow2 = 1u32 << (safe_exp - 1u32); // {1, 2, 4}
            let normal_two_m = (mant + 2u32) * pow2;
            let two_m_int = select(is_subnormal, mant, normal_two_m);
            let two_m_f = two_m_int.cast::<f32>();
            // sign_f = 1.0 − 2.0·sign_bit ∈ {+1, −1}. fp32 to avoid u32
            // underflow when sign_bit = 1.
            let sign_f = 1.0f32 - 2.0f32 * sign_bit.cast::<f32>();
            // value = scale · sign · two_m_int / 2.
            let value = scale * sign_f * two_m_f * 0.5f32;
            threadgroup_store("Ws", x_ws_base + _ni, value);
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

inventory::submit! {
    BenchSpec {
        op: "fp_quantized",
        subop: "fp_qmm_nax",
        kernel_name: "mt_fp_qmm_nax",
        kernel_ir: mt_fp_qmm_nax::kernel_ir_for,
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
            let k = mt_fp_qmm_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_fp_qmm_nax");
            assert_eq!(k.params.len(), 4);
            assert_eq!(k.params[0].name, "w");
            assert_eq!(k.params[0].dtype, DType::U32);
            assert_eq!(k.params[1].name, "scales");
            assert_eq!(k.params[2].name, "x");
            assert_eq!(k.params[3].name, "out");
            assert!(k.params[3].is_output);
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
        let k = mt_fp_qmm_nax::kernel_ir_for(DType::BF16);
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
            let mut k = mt_fp_qmm_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_fp_qmm_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_fp_qmm_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }
}
