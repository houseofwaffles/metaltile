//! Golden MSL snapshots for the steel / hadamard / quantized NAX
//! kernels in PR #147.
//!
//! Per @0xClandestine's review on #147: "Would add a test that ensures
//! generated output is as you would expect using this." These kernels
//! are expressed entirely via the `CoopTile*` / DSL IR ops — no
//! `Op::InlineMsl` — and the GPU correctness tests under
//! `tests/steel_*_nax_gpu_correctness.rs` already pin numerical
//! behaviour. The snapshots here pin the EMIT PATH: any change to op
//! lowering, preamble emission, scheduling, or vectorization shows up
//! as a reviewable text diff instead of silent drift through to the
//! per-kernel GPU runs.
//!
//! Same shape as `aura_msl_snapshots.rs` — one representative dtype /
//! bit-width / dim per kernel. The numerical contract for every
//! monomorphization lives in the GPU-correctness tests; this file pins
//! the codegen path.
//!
//! Refresh after an intentional codegen change:
//!   cargo insta test --accept -p metaltile-std --test steel_msl_snapshots
//! Or interactively:
//!   cargo insta review
//!
//! NAX kernels lower through `MetalPerformancePrimitives` cooperative-tensor
//! intrinsics. They compile unconditionally now; runtime dispatch is gated
//! by `skip_unless_apple10` (the M4+ tensor-core test guard).

use insta::assert_snapshot;
use metaltile_codegen::{MslGenerator, msl::MslConfig};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_std::mlx::{
    fp_quantized_nax,
    hadamard_m,
    quantized_nax,
    steel::gemm::{steel_gemm_fused_nax, steel_gemm_gather_nax, steel_gemm_splitk_nax},
};

/// Lower one of the NAX-family kernel IRs to MSL with its declared
/// kernel mode. `kernel_ir_for` returns the bare IR; the dispatch mode
/// (Reduction / Grid3D) lives on the BenchSpec and must be set on the
/// IR before codegen so the reduce-emit vs Grid3D-emit branches lower
/// correctly.
fn steel_msl(kernel_ir: metaltile_core::ir::Kernel, mode: KernelMode) -> String {
    let mut kernel = kernel_ir;
    kernel.mode = mode;
    MslGenerator::new(MslConfig::default()).generate(&kernel).expect("kernel must codegen cleanly")
}

// ── steel_gemm_fused_nax ─────────────────────────────────────────────
//
// Plain fused GEMM backed by `mpp::tensor_ops::matmul2d`. F16 is the
// production dtype on Qwen3.5/3.6 MoE BGEMM gate paths; pin that here.

#[test]
fn steel_gemm_fused_nax_f16_msl() {
    let msl = steel_msl(
        steel_gemm_fused_nax::mt_steel_gemm_fused_nax::kernel_ir_for(DType::F16),
        KernelMode::Reduction,
    );
    assert_snapshot!(msl);
}

// ── steel_gemm_gather_nax ────────────────────────────────────────────
//
// MoE-style gather GEMM — same NAX cooperative-tensor matmul path as
// `fused`, with an additional `gather_idx` permutation on the M axis.

#[test]
fn steel_gemm_gather_nax_f16_msl() {
    let msl = steel_msl(
        steel_gemm_gather_nax::mt_steel_gemm_gather_nax::kernel_ir_for(DType::F16),
        KernelMode::Reduction,
    );
    assert_snapshot!(msl);
}

// ── steel_gemm_splitk_nax (gemm + accum) ─────────────────────────────
//
// Two-phase split-K GEMM: per-split partial outputs (kernel) + cross-
// split f32 reduction (accum). Both phases are DSL-expressed; snapshot
// pins both emit paths.

#[test]
fn steel_gemm_splitk_nax_f16_msl() {
    let msl = steel_msl(
        steel_gemm_splitk_nax::mt_steel_gemm_splitk_nax::kernel_ir_for(DType::F16),
        KernelMode::Reduction,
    );
    assert_snapshot!(msl);
}

#[test]
fn steel_gemm_splitk_nax_accum_f32_msl() {
    let msl = steel_msl(
        steel_gemm_splitk_nax::mt_steel_gemm_splitk_accum_nax::kernel_ir_for(DType::F32),
        KernelMode::Grid3D,
    );
    assert_snapshot!(msl);
}

// ── quantized_nax ────────────────────────────────────────────────────
//
// `mt_qmm_mma_mpp` — quantized matmul over MPP cooperative tensors.
// F16 activation × int4 packed weight is the production Qwen3.6 path.

#[test]
fn quantized_nax_f16_msl() {
    let msl =
        steel_msl(quantized_nax::mt_qmm_nax::kernel_ir_for(DType::F16), KernelMode::Reduction);
    assert_snapshot!(msl);
}

// ── fp_quantized_nax ─────────────────────────────────────────────────
//
// `mt_fp_qmm_nax` — fp4 (E2M1) quantized matmul. Same MPP coop-load /
// `matmul2d` shape as `quantized_nax`, with the FP4 codebook dequant
// replacing int4 nibble · scale + bias. F16 pins the production dtype.

#[test]
fn fp_quantized_nax_f16_msl() {
    let msl = steel_msl(
        fp_quantized_nax::mt_fp_qmm_nax::kernel_ir_for(DType::F16),
        KernelMode::Reduction,
    );
    assert_snapshot!(msl);
}

// ── hadamard_m ───────────────────────────────────────────────────────
//
// Non-power-of-2 Hadamard transform for the M ∈ {12, 20, 28} factors.
// M = 20 covers the middle of the supported range; the per-M sign-bit
// table is constexpr-baked so each M gets a distinct kernel IR.

#[test]
fn hadamard_m_m20_f32_msl() {
    let msl =
        steel_msl(hadamard_m::mt_hadamard_m20::kernel_ir_for(DType::F32), KernelMode::Reduction);
    assert_snapshot!(msl);
}
