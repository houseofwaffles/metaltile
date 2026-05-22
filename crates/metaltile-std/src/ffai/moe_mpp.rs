//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm16_mpp`.
//!
//! Routes the per-tile matmul through Apple's MetalPerformancePrimitives
//! `mpp::tensor_ops::matmul2d` (the same API MLX uses to reach ~3000 GF
//! on `down_proj` at Qwen3.6-A3B prefill). Algorithmically mirrors
//! `mt_moe_gather_qmm_mma_int4_bm16` (BM=16, BN=32, 2 sub-runs per tile,
//! per-row expert dispatch), but swaps the inner `simdgroup_matmul`
//! 8×8 frags for a single 16×32×16 MPP descriptor.
//!
//! ## Why this kernel exists
//!
//! The simdgroup-matrix BM=16 variant tops out at ~205 GF on Qwen3.6-A3B
//! `down_proj` while MLX's `affine_gather_qmm_rhs_nax` (NAX path on
//! M5 Max / macOS 26+ / gen ≥17) hits ~3000 GF — a 14× gap. The gap
//! cannot be closed from the metaltile DSL alone: `simdgroup_matmul`
//! goes through the MXU but stops short of the NAX scheduler. Only the
//! `mpp::tensor_ops::matmul2d<desc, execution_simdgroup>` API taps that
//! path.
//!
//! Predecessor smoke kernel: `crates/metaltile-std/src/probe/mpp_matmul_smoke.rs`
//! proved the metaltile codegen + toolchain accept the MPP header and
//! the Apple-private cooperative_tensor types via `Op::InlineMsl` (the
//! DSL macro front-end can't represent `mpp::` symbols yet).
//!
//! ## Algorithm
//!
//! Identical row-partitioning + dequant logic to
//! `mt_moe_gather_qmm_mma_int4_bm16`:
//! - Grid: `[N/32, ceil(M/16), 1]`, one threadgroup per output tile
//! - Threadgroup: 32 lanes = 1 simdgroup (MPP `matmul2d` is
//!   `execution_simdgroup`)
//! - Each TG owns a [BM=16, BN=32] output sub-tile of `out`
//! - Up to 16 expert sub-runs walk the 16 rows; production
//!   Qwen3.6-A3B T=1024 × 128 experts ≈ 2 sub-runs
//! - For each sub-run: dequant W[expert, n_tile, K] int4 → T into TG
//!   memory, copy X[m_tile, K] into TG memory, feed both as
//!   `tensor_inline` views to `cooperative_tensor`s, and run the
//!   matmul with `multiply_accumulate` mode across the K loop
//! - K tile width is BK=16 (the descriptor's K dim); we walk K in
//!   chunks of 16 and accumulate in the output cooperative_tensor
//!
//! ## Descriptor choice
//!
//! `matmul2d_descriptor(16, 32, 16, false, true, false, multiply_accumulate)`
//! - M=16, N=32, K=16 — N=32 satisfies Apple's "at least one of
//!   M/N/K = 32" constraint for the cooperative_tensor path
//! - `ta=false` → A is `[M, K]` row-major (X tile)
//! - `tb=true`  → B is `[N, K]` row-major (W tile, "transposed" from the
//!   `C = A·B` perspective: W is stored `[N, K]` natively, same as MLX's
//!   `affine_gather_qmm_rhs_nax` with `transpose=true`)
//! - `tc=false` → C is `[M, N]` row-major (out tile, natural form)
//! - Acc mode `multiply_accumulate` lets us span K in BK=16 steps and
//!   accumulate without an explicit add — descriptor handles it
//!
//! ## Threadgroup memory layout
//!
//! - `xs[16 × 16]` — half/float, X chunk for one K-tile
//! - `ws[32 × 16]` — half/float, dequant'd W chunk for one K-tile
//! - `out_scratch[16 × 32]` — fp32, post-matmul staging for the
//!   `ct_c.store(...)` call (the cooperative_tensor store overload
//!   requires destination elem-type == acc type; we narrow to T on the
//!   coop-write to global)
//!
//! ## Constraints inherited from MLX's NAX path
//!
//! - macOS 26+ / Metal 4 (`__METAL_VERSION__ >= 400`) — codegen
//!   auto-emits the MPP include gated on this. Pre-Metal-4 toolchains
//!   compile a no-op stub so the metallib still links.
//! - At least one of `M`, `N`, `K` in the descriptor must be 32 (Apple
//!   assertion in the cooperative_tensor path).
//! - `tensor_inline` requires packed/contiguous strides — we stage
//!   into TG memory rather than passing arbitrary-stride device views.
//!
//! ## Status
//!
//! First-pass MPP MoE kernel. Correctness validated by
//! `tests/moe_gather_qmm_mpp_correctness.rs` (cosine ≥ 0.999 vs the m1
//! scalar oracle at n_experts=4, T=64, N=64, K=64, group_size=32).
//! Performance characterization on Qwen3.6-A3B production shapes
//! pending bench harness on M2 mini (see
//! `feedback_metaltile_bench_on_m2_mini.md` — never bench on M5 Max).

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind, ValueId},
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

use crate::spec::{BenchDispatch, BenchSpec};

/// Render the inline MSL body for the MoE MPP kernel.
///
/// `t` is the MSL type of the device-side params (`x`, `out`) — `"half"`,
/// `"float"`, or `"bfloat"`. `ts` is the *staging* type used for the
/// threadgroup tiles + MPP cooperative tensors — `"half"` for both `half`
/// and `bfloat` activations, `"float"` for `float`.
///
/// Why the split: Apple's `mpp::tensor_ops::matmul2d` does not handle
/// `bfloat` cooperative tensors correctly (verified on M5 Max). bf16
/// activations are read from device `bfloat`, cast to `half` into the
/// threadgroup tiles, and the matmul runs `half`×`half`→`float`. `half`'s
/// 10-bit mantissa strictly covers `bfloat`'s 7, so the staged operands
/// are lossless and accumulation is fp32 regardless. For `float`/`half`
/// activations `ts == t`. The W buffer is `uint32_t` (packed int4)
/// regardless of `t`.
fn msl_body(t: &str, ts: &str) -> String {
    // Internal name aliases used in the inline body:
    //   x, w, scales, biases, indices, out — kernel params (bound by name)
    //   xs, ws, ys                          — threadgroup arrays (hoisted)
    //   m_total, n_out, k_in, group_size    — constexpr scalars
    //   tgid_x, tgid_y                       — threadgroup position aliases
    //                                          (auto-emitted by the codegen
    //                                          because we push dummy
    //                                          `Op::ProgramId` ops with
    //                                          axis=0,1 before the
    //                                          InlineMsl block)
    //   simd_lane                            — thread index in the SG
    //                                          (auto-bound because the MPP
    //                                          detector turns on
    //                                          needs_simd_lane)
    //
    // The dequant constants (1/16, 1/256, 1/4096) match MLX's nibble
    // unpack: nibble `nib` at packed position `i` gives
    //   (packed >> (i*4)) & 0xf  → fp value q
    //   weight = q * scale + bias
    // Per group, scale and bias are constants for `group_size`
    // consecutive K-positions.
    format!(
        r#"// --- mt_moe_gather_qmm_mma_int4_bm16_mpp body ({t} acc fp32) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 16u;
constexpr uint BN = 32u;
constexpr uint BK = 16u;

const uint n_tile_base = tgid_x * BN;
const uint m_tile_base = tgid_y * BM;
const uint lane        = simd_lane;

const uint packs_per_row  = k_in / 8u;
const uint groups_per_row = k_in / group_size;

// MPP descriptor + cooperative tensors. ct_c persists across K iters
// for cross-iteration accumulation under multiply_accumulate mode.
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/32, /*K=*/16,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<{ts}, {ts}, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<{ts}, {ts}, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

// Walk row sub-runs — same logic as the simdgroup-matrix BM=16 variant.
// Production shape (T=1024, n_experts=128, top_k=8 → 64 rows per expert
// after permute, 4 m-tiles per expert) typically gives 1 sub-run per TG.
uint sub_offset = 0u;
for (uint _sub_iter = 0u; _sub_iter < 16u; _sub_iter++) {{
    uint sub_end = sub_offset;
    uint cur_expert = 0xffffffffu;
    if (sub_offset < BM) {{
        uint cur_row = m_tile_base + sub_offset;
        if (cur_row < m_total) cur_expert = indices[cur_row];
        sub_end = BM;
        bool found = false;
        for (uint ii = 0u; ii < BM; ii++) {{
            uint probe = sub_offset + 1u + ii;
            uint probe_row = m_tile_base + probe;
            if ((probe < BM) && (probe_row < m_total) && !found) {{
                uint e = indices[probe_row];
                if (e != cur_expert) {{ sub_end = probe; found = true; }}
            }}
            if ((probe < BM) && (probe_row >= m_total) && !found) {{
                sub_end = probe;
                found = true;
            }}
        }}
    }}

    bool cur_valid = (cur_expert != 0xffffffffu) && (sub_offset < BM);
    if (cur_valid) {{
        const uint w_expert_base  = cur_expert * n_out * packs_per_row;
        const uint sb_expert_base = cur_expert * n_out * groups_per_row;

        // Zero the accumulator before this sub-run.
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;

        for (uint kb = 0u; kb < k_in; kb += BK) {{
            // -- Stage X[m_tile_base..+BM, kb..kb+BK] -> xs[BM*BK]
            // 32 lanes × 8 elems/lane = 256 = BM*BK. flat = lane*8 + e.
            for (uint e = 0u; e < 8u; e++) {{
                uint flat = lane * 8u + e;
                uint mr = flat / BK;
                uint kc = flat % BK;
                uint gr = m_tile_base + mr;
                // Mask rows outside [sub_offset, sub_end) ∩ m_total to 0
                // — they contribute zero to the matmul.
                bool in_run = (mr >= sub_offset) && (mr < sub_end) && (gr < m_total);
                uint safe_g = in_run ? gr : 0u;
                // Explicit cast: device `x` is `{t}`; for bf16 activations
                // this narrows bfloat → half for the staged matmul. For
                // f32/f16 it is a no-op the compiler elides.
                {ts} xv = ({ts})x[safe_g * k_in + kb + kc];
                xs[mr * BK + kc] = in_run ? xv : ({ts})0;
            }}

            // -- Dequant W[expert, n_tile_base..+BN, kb..kb+BK] -> ws[BN*BK]
            // BN=32, BK=16 → 512 elems = 64 packs (8 nibbles each).
            // 32 lanes × 2 packs/lane = 64 packs.
            for (uint pi = 0u; pi < 2u; pi++) {{
                uint pack_id  = lane * 2u + pi;
                uint w_row    = pack_id / 2u;          // 0..31 (BN rows)
                uint pack_col = pack_id % 2u;          // 0..1 (BK=16 → 2 packs of 8 nibbles)
                uint pack_dev = w_expert_base
                              + (n_tile_base + w_row) * packs_per_row
                              + (kb / 8u)
                              + pack_col;
                uint packed = w[pack_dev];
                uint k_off  = kb + pack_col * 8u;
                uint g      = k_off / group_size;
                uint sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                float s = (float)scales[sb_off];
                float b = (float)biases[sb_off];

                uint hi = packed >> 16u;
                uint dst_row_base = w_row * BK + pack_col * 8u;
                ws[dst_row_base + 0u] = ({ts})(s * (float)((packed >>  0) & 0xfu) + b);
                ws[dst_row_base + 1u] = ({ts})(s * (float)((packed >>  4) & 0xfu) + b);
                ws[dst_row_base + 2u] = ({ts})(s * (float)((packed >>  8) & 0xfu) + b);
                ws[dst_row_base + 3u] = ({ts})(s * (float)((packed >> 12) & 0xfu) + b);
                ws[dst_row_base + 4u] = ({ts})(s * (float)((hi     >>  0) & 0xfu) + b);
                ws[dst_row_base + 5u] = ({ts})(s * (float)((hi     >>  4) & 0xfu) + b);
                ws[dst_row_base + 6u] = ({ts})(s * (float)((hi     >>  8) & 0xfu) + b);
                ws[dst_row_base + 7u] = ({ts})(s * (float)((hi     >> 12) & 0xfu) + b);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Build tensor_inline views over the TG buffers and load
            // into cooperative_tensors. extents are inner-first; with
            //   ta=false, A is row-major [M=16, K=16] → extents{{K=16, M=16}}
            //   tb=true , B is row-major [N=32, K=16] → extents{{K=16, N=32}}
            metal::tensor<threadgroup {ts}, metal::extents<int, 16, 16>, metal::tensor_inline>
                tA(xs, metal::extents<int, 16, 16>{{}});
            metal::tensor<threadgroup {ts}, metal::extents<int, 16, 32>, metal::tensor_inline>
                tB(ws, metal::extents<int, 16, 32>{{}});

            ct_a.load(tA);
            ct_b.load(tB);

            gemm_op.run(ct_a, ct_b, ct_c);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Stage the fp32 accumulator into the per-SG fp32 scratch — the
        // `ct_c.store(...)` overload requires the destination
        // tensor_inline element type to match `ct_c`'s accumulator type
        // (`float`). We can't write directly to the half-typed `ys`
        // buffer here; cast at coop-write below instead. Layout is
        // row-major [M=16, N=32], inner = N = 32.
        metal::tensor<threadgroup float, metal::extents<int, 32, 16>, metal::tensor_inline>
            tY(out_scratch, metal::extents<int, 32, 16>{{}});
        ct_c.store(tY);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Coop-write out_scratch -> out[m_tile_base..+BM, n_tile_base..+BN]
        // with fp32 → {t} narrow + per-row expert mask.
        // 32 lanes × 16 elems = 512 = BM*BN.
        for (uint e = 0u; e < 16u; e++) {{
            uint flat = lane * 16u + e;
            uint mr   = flat / BN;            // 0..15
            uint nc   = flat % BN;            // 0..31
            uint gr   = m_tile_base + mr;
            uint gc   = n_tile_base + nc;
            bool in_run = (mr >= sub_offset) && (mr < sub_end)
                        && (gr < m_total) && (gc < n_out);
            if (in_run) {{
                out[gr * n_out + gc] = ({t})out_scratch[mr * BN + nc];
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    sub_offset = sub_end;
}}
#else
// Pre-Metal-4 fallback: write zeros to keep the metallib linkable but
// fail the correctness test loud-and-clear.
if (simd_lane == 0u) {{
    uint m_tile_base = tgid_y * 16u;
    uint n_tile_base = tgid_x * 32u;
    for (uint r = 0u; r < 16u; r++) {{
        uint gr = m_tile_base + r;
        if (gr >= m_total) continue;
        for (uint c = 0u; c < 32u; c++) {{
            uint gc = n_tile_base + c;
            if (gc >= n_out) continue;
            out[gr * n_out + gc] = ({t})0;
        }}
    }}
}}
#endif
"#
    )
}

/// Build the [`Kernel`] IR for `mt_moe_gather_qmm_mma_int4_bm16_mpp`.
///
/// Dispatched as `KernelMode::Reduction`. Grid is `[N/32, ceil(M/16), 1]`,
/// threadgroup size is `[32, 1, 1]` (1 simdgroup — required by the MPP
/// descriptor's `execution_simdgroup` scope).
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16 | DType::BF16),
        "mt_moe_gather_qmm_mma_int4_bm16_mpp: dtype must be F32, F16, or BF16, got {:?}",
        dt
    );
    let t = match dt {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        _ => unreachable!(),
    };
    // Staging dtype for the threadgroup tiles + MPP cooperative tensors.
    // bf16 stages through `half` because `matmul2d` mishandles `bfloat`
    // cooperative tensors; f32/f16 stage in their own type (ts == t).
    let stage_dt = match dt {
        DType::BF16 => DType::F16,
        other => other,
    };
    let ts = match stage_dt {
        DType::F32 => "float",
        DType::F16 => "half",
        _ => unreachable!(),
    };

    let mut k = Kernel::new("mt_moe_gather_qmm_mma_int4_bm16_mpp");
    k.mode = KernelMode::Reduction;

    // Params — match the bm16 signature exactly:
    //   x       [m_total, k_in]                  T
    //   w       [n_experts, n_out, k_in/8]       u32  (int4-packed)
    //   scales  [n_experts, n_out, k_in/group]   T
    //   biases  [n_experts, n_out, k_in/group]   T
    //   indices [m_total]                        u32
    //   out     [m_total, n_out]                 T
    //
    // Shapes use `Dim::Any` because the codegen emits the C-pointer
    // signature regardless; concrete sizes come from the constexpr
    // scalars (m_total / n_out / k_in / group_size) at dispatch time —
    // same convention as `quantized_mpp::kernel_ir_for`.
    k.params.push(Param {
        name: "x".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "w".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any, Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "scales".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "biases".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "indices".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });
    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // Constexpr scalars (same as the bm16 sibling).
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("m_total"),
        dtype: DType::U32,
        value: None,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n_out"),
        dtype: DType::U32,
        value: None,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("k_in"),
        dtype: DType::U32,
        value: None,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("group_size"),
        dtype: DType::U32,
        value: None,
    });

    // Build the kernel body. We need:
    //  1. Hoist three threadgroup arrays — xs[16*16], ws[32*16], ys[16*32].
    //  2. Force `tgid_y` alias emission via the `Op::Load { src: "tgid_y" }`
    //     direct-identifier form (per `ssm_step` precedent — see
    //     `reduction_preamble_emits_tgid_y_when_used_as_identifier` in
    //     `metaltile-codegen/src/msl/mod.rs`). `tgid_x` is unconditional
    //     in Reduction mode so no hint needed.
    //  3. The InlineMsl payload referencing all of the above by name.
    let mut body = Block::new(BlockId::new(0));

    // Threadgroup allocations. `xs` and `ws` are sized for ONE BK=16
    // K-chunk; `ys` is sized for the BM=16 × BN=32 output stage. Sizes
    // are in elements (not bytes).
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: 16 * 16,
        name: "xs".into(),
    });
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: 32 * 16,
        name: "ws".into(),
    });
    // fp32 staging for the cooperative_tensor `ct_c.store(...)`. The
    // store overload requires destination elem-type == accumulator type
    // (float here). We narrow to `{t}` during the coop-write to global.
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 16 * 32,
        name: "out_scratch".into(),
    });

    // Force `tgid_y` alias emission. InlineMsl body references `tgid_y`
    // by name; codegen's `kernel_uses_program_id_axis(kernel, 1)` check
    // looks at IR ops only, not body text. The direct-identifier load
    // form is the convention (matches `ssm_step` + the codegen
    // regression test).
    body.push_op(
        Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(0),
    );

    // The InlineMsl op contains the full body. inputs/outputs are empty
    // because the body addresses params + TG arrays by name in MSL.
    body.push_op_no_result(Op::InlineMsl {
        source: msl_body(t, ts),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });

    k.body = body.clone();
    let mut blocks = FxHashMap::default();
    blocks.insert(BlockId::new(0), body);
    k.blocks = blocks;

    k
}

// Bench/inventory registration — same pattern as the bm16 sibling.
// `dispatch: Generic` + `shapes: &[]` means `tile bench` skips the
// kernel; correctness lives in tests/.
inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int4_bm16_mpp",
        kernel_name: "mt_moe_gather_qmm_mma_int4_bm16_mpp",
        kernel_ir: kernel_ir_for,
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
    use super::*;

    #[test]
    fn kernel_ir_constructs_f32() {
        let k = kernel_ir_for(DType::F32);
        assert_eq!(k.name, "mt_moe_gather_qmm_mma_int4_bm16_mpp");
        assert_eq!(k.params.len(), 6);
        assert!(k.params[5].is_output);
        assert_eq!(k.constexprs.len(), 4);
        assert!(matches!(k.mode, KernelMode::Reduction));
    }

    #[test]
    fn kernel_ir_constructs_f16() {
        let k = kernel_ir_for(DType::F16);
        // Inline source carries the dtype — quick sanity check.
        let has_half = k.body.ops.iter().any(|op| match op {
            Op::InlineMsl { source, .. } => source.contains("metal::tensor<threadgroup half"),
            _ => false,
        });
        assert!(has_half, "F16 kernel should carry `threadgroup half` tensor view");
    }

    /// Sanity-check that the generated MSL pulls in the MPP header.
    /// Same gate the smoke kernel uses.
    #[test]
    fn codegen_emits_mpp_include() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(
            msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
            "MPP include missing from generated MSL"
        );
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int4_bm16_mpp"));
        // tgid_y alias must be emitted — we depend on it in the
        // inline MSL.
        assert!(msl.contains("tgid_y"), "tgid_y alias missing");
    }

    /// Developer aid — dump the full generated MSL.
    /// `cargo test -p metaltile-std --lib -- moe_mpp::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
