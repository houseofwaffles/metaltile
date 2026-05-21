//! MPP (MetalPerformancePrimitives) `matmul2d` smoke kernel.
//!
//! Single simdgroup, 16×32 fp16 → 16×16 fp32 matmul through the new
//! `mpp::tensor_ops::matmul2d` API introduced in Metal 4 (macOS 26+).
//! Proves the metaltile codegen + toolchain stack accepts the MPP
//! header and the Apple-private cooperative-tensor types.
//!
//! Shapes:
//!   A = [M=16, K=32], row-major fp16
//!   B = [K=32, N=16], row-major fp16
//!   C = [M=16, N=16], row-major fp32
//!
//! Geometry: single threadgroup, single simdgroup (32 threads).
//! The matmul2d descriptor enforces "at least one of M/N/K = 32" when
//! both inputs are cooperative tensors — K=32 satisfies that.
//!
//! This kernel is the *first* step toward replacing `mt_qmm_mma` with
//! an MPP-backed variant that taps the NAX hardware path MLX exposes via
//! `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`. The
//! observed MLX `down_proj` ceiling at ~3000 GF on Qwen3.6-A3B sits on
//! this exact API.
//!
//! Built as an IR escape-hatch via `Op::InlineMsl` rather than the
//! `#[kernel]` macro because the macro front-end does not (yet) expose
//! a way to emit raw MSL that references `mpp::` symbols. The codegen
//! preamble auto-detects the `"mpp::"` prefix and emits the required
//! `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
//! gated on `__METAL_VERSION__ >= 400`.
//!
//! Runtime behavior on `gen < 17` GPUs (e.g. M3 and earlier): the metallib
//! still links because the MPP source body is `#if __METAL_VERSION__ >= 400`
//! gated and the `#else` branch performs only a trivial elementwise write.
//! Caller-side dispatch should skip this kernel on unsupported GPU gens —
//! `metaltile-runtime` already routes via `KernelFeatures::needs_mpp`, so
//! downstream callers don't need to gate explicitly.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    ir::{Block, BlockId, Kernel, KernelMode, Op, Param, ParamKind},
    shape::{Dim, Shape},
};

// Inline MSL body. References the kernel parameter names (`A`, `B`, `C`)
// directly — the codegen emits them as `const device half*` / `device float*`
// with the corresponding buffer bindings.
//
// `simd_lane` is also bound by the codegen because we set
// `needs_simd_lane = true` via the MPP-detection path in `features.rs`.
const MPP_MATMUL_SMOKE_SRC: &str = r#"// --- MPP matmul2d smoke (M=16, N=16, K=32, half/half -> float) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/16, /*K=*/32,
    /*ta=*/false, /*tb=*/false, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

// Build `tensor_inline` views over raw device pointers. The tensor_inline
// packed-stride ctor sets strides[0]=1, strides[i]=strides[i-1]*extents[i-1]
// — i.e. extents[0] is the contiguous/inner dim. For row-major buffers,
// pass extents{ inner, outer } = {cols, rows}.
//
//   A row-major [M=16, K=32]  -> extents<int, 32, 16>
//   B row-major [K=32, N=16]  -> extents<int, 16, 32>
//   C row-major [M=16, N=16]  -> extents<int, 16, 16>
// A and B are emitted as `const device half*` by the metaltile codegen
// (read-only param convention). The cooperative_tensor.load() overload
// requires a non-const element type — cast away the const here. Safe
// because MPP's load is a pure-read operation in MSL semantics.
metal::tensor<device half,  metal::extents<int, 32, 16>, metal::tensor_inline> tA(const_cast<device half*>(A), metal::extents<int, 32, 16>{});
metal::tensor<device half,  metal::extents<int, 16, 32>, metal::tensor_inline> tB(const_cast<device half*>(B), metal::extents<int, 16, 32>{});
metal::tensor<device float, metal::extents<int, 16, 16>, metal::tensor_inline> tC(C, metal::extents<int, 16, 16>{});

ct_a.load(tA);
ct_b.load(tB);

// Zero accumulator (mode = multiply_accumulate adds to dst).
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;

gemm_op.run(ct_a, ct_b, ct_c);

ct_c.store(tC);
#else
// Fallback for pre-Metal 4 toolchains: silence the binding usage so the
// metallib still links. Smoke test must be skipped on such targets.
if (simd_lane == 0u) {
    for (uint i = 0; i < 16u * 16u; ++i) {
        C[i] = float(A[0]) * float(B[0]);
    }
}
#endif
"#;

/// Build the [`Kernel`] IR for `mt_mpp_matmul_smoke`.
///
/// The kernel has three buffer params + a single `Op::InlineMsl` op that
/// emits the raw MSL above. The codegen's `analyze` pass detects `"mpp::"`
/// in the source and toggles `needs_mpp = true`, which (in `mod.rs`) emits
/// `#include <MetalPerformancePrimitives/...>` gated on `__METAL_VERSION__ >= 400`.
///
/// Dispatch geometry: 1 threadgroup × 32 threads = one simdgroup.
pub fn kernel_ir() -> Kernel {
    let mut k = Kernel::new("mt_mpp_matmul_smoke");
    k.mode = KernelMode::Elementwise;

    // Params: A [M,K] fp16, B [K,N] fp16, C [M,N] fp32.
    k.params.push(Param {
        name: "A".into(),
        dtype: DType::F16,
        shape: Shape::new([Dim::Known(16), Dim::Known(32)]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "B".into(),
        dtype: DType::F16,
        shape: Shape::new([Dim::Known(32), Dim::Known(16)]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "C".into(),
        dtype: DType::F32,
        shape: Shape::new([Dim::Known(16), Dim::Known(16)]),
        is_output: true,
        kind: ParamKind::Tensor,
    });
    k.return_shapes.push(Shape::new([Dim::Known(16), Dim::Known(16)]));

    // Single InlineMsl op with no SSA inputs/outputs — pure side-effecting
    // write to the bound `C` parameter buffer. (`outputs` and `inputs` lists
    // are both empty because the kernel addresses params by name in the
    // inline source rather than going through SSA values.)
    let mut body = Block::new(BlockId::new(0));
    body.push_op_no_result(Op::InlineMsl {
        source: MPP_MATMUL_SMOKE_SRC.into(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body.clone();
    let mut blocks = BTreeMap::new();
    blocks.insert(BlockId::new(0), body);
    k.blocks = blocks;

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_and_has_three_params() {
        let k = kernel_ir();
        assert_eq!(k.name, "mt_mpp_matmul_smoke");
        assert_eq!(k.params.len(), 3);
        assert_eq!(k.params[0].name, "A");
        assert_eq!(k.params[1].name, "B");
        assert_eq!(k.params[2].name, "C");
        assert!(k.params[2].is_output);
        // Sanity: body has the inline MSL op.
        assert_eq!(k.body.ops.len(), 1);
        assert!(matches!(&k.body.ops[0], Op::InlineMsl { .. }));
    }

    #[test]
    fn codegen_emits_mpp_include() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(
            msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
            "MPP include missing from generated MSL:\n{msl}"
        );
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_mpp_matmul_smoke"));
    }

    /// Developer aid — `cargo test -p metaltile-std --lib -- dump_generated_msl --nocapture`
    /// prints the full generated MSL for inspection. Always passes; gated
    /// behind `--nocapture` for output.
    #[test]
    fn dump_generated_msl() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
