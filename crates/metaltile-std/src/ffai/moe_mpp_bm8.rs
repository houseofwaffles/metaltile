//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm8_mpp`.
//!
//! BM=8 sibling of `mt_moe_gather_qmm_mma_int4_bm16_mpp` and
//! `mt_moe_gather_qmm_mma_int4_bm64_mpp`. Same algorithm and call-site
//! signature, but the per-TG row tile shrinks to 8 so the kernel doesn't
//! waste half the MMA frag on zero-padded rows at decode-time MoE shapes
//! (T=1, topK=8 → m_total=8 after gather/permute).
//!
//! ## Why this kernel exists
//!
//! At decode-time (T=1, B=1) the post-permute MoE input has exactly
//! `m_total = topK` rows — 8 for Qwen3.6-A3B (topK=8). The BM=16 MPP
//! variant pads the second half of the 16-row tile with zeros and still
//! pays full SG dispatch cost on those rows. BM=8 matches the workload
//! exactly: one TG, one sub-run, no padding waste.
//!
//! ## Algorithm
//!
//! Identical row-partitioning + dequant logic to
//! `mt_moe_gather_qmm_mma_int4_bm16_mpp`, with the BM dim halved:
//! - Grid: `[N/32, ceil(M/8), 1]`, one threadgroup per output tile
//! - Threadgroup: 32 lanes = 1 simdgroup (MPP `matmul2d` is
//!   `execution_simdgroup`)
//! - Each TG owns a [BM=8, BN=32] output sub-tile of `out`
//! - Up to 8 expert sub-runs walk the 8 rows; typical decode shape
//!   T=1 × topK=8 = 1 row per expert → 8 sub-runs of 1 row each, but the
//!   common production case (sorted-by-expert post-permute) collapses
//!   to ~1 sub-run when adjacent rows share an expert
//! - K tile width is BK=16; we walk K in chunks of 16 and accumulate
//!   in the output cooperative_tensor under `multiply_accumulate` mode
//!
//! ## Descriptor choice
//!
//! `matmul2d_descriptor(8, 32, 16, false, true, false, multiply_accumulate)`
//! - M=8, N=32, K=16 — N=32 satisfies the simdgroup-scope "at least one
//!   of M, N must be a multiple of 16" rule (M=8 is a multiple of 8;
//!   N=32 is a multiple of 16; K=16 is a multiple of 16).
//! - **A and B are passed as direct `metal::tensor`s over threadgroup
//!   memory, NOT as cooperative tensors.** With both inputs as
//!   cooperative tensors, Apple's MPP impl asserts each of M, N, K must
//!   be exactly 16 or 32 (see `MPPTensorOpsMatMul2dImpl.h:3756-3758` —
//!   `"M must be 16 or 32 if both inputs are cooperative tensors"`),
//!   which rules out M=8. Using a direct `tensor` input bypasses that
//!   strict-cooperative path and the looser simdgroup-scope constraint
//!   applies. Destination remains a cooperative tensor (`ct_c`) so we
//!   can persist fp32 partials across the K-loop under
//!   `multiply_accumulate` mode.
//! - `ta=false` → A is `[M=8, K=16]` row-major (X tile)
//! - `tb=true`  → B is `[N=32, K=16]` row-major (W tile — qmm_t layout)
//! - `tc=false` → C is `[M=8, N=32]` row-major fp32
//! - Acc mode `multiply_accumulate` lets us span K in BK=16 steps and
//!   accumulate without an explicit add — descriptor handles it
//!
//! ## Threadgroup memory layout
//!
//! - `xs[8 × 16]` — half/float/bfloat, X chunk for one K-tile (128 elems)
//! - `ws[32 × 16]` — half/float/bfloat, dequant'd W chunk (512 elems)
//! - `out_scratch[8 × 32]` — fp32 staging for `ct_c.store(...)` (256 elems)
//!
//! All three are well under the M5 Max 32 KB per-TG threadgroup limit.
//!
//! ## Constraints inherited from the bm16/bm64 siblings
//!
//! - macOS 26+ / Metal 4 (`__METAL_VERSION__ >= 400`) — codegen
//!   auto-emits the MPP include gated on this. Pre-Metal-4 toolchains
//!   compile a no-op stub so the metallib still links.
//! - At least one of `M`, `N`, `K` in the descriptor must be 32 (Apple
//!   assertion in the cooperative_tensor path). N=32 satisfies it here.
//! - `tensor_inline` requires packed/contiguous strides — we stage
//!   into TG memory rather than passing arbitrary-stride device views.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind, ValueId},
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

use crate::spec::{BenchDispatch, BenchSpec};

/// Render the inline MSL body for the BM=8 MPP MoE kernel.
///
/// `t` is the MSL type of the device-side params (`x`, `out`) — `"half"`,
/// `"float"`, or `"bfloat"`. `ts` is the *staging* type used for the
/// threadgroup tiles + MPP tensor operands — `"half"` for both `half` and
/// `bfloat` activations, `"float"` for `float`.
///
/// Why the split: Apple's `mpp::tensor_ops::matmul2d` does not handle
/// `bfloat` tensor operands correctly (verified on M5 Max). bf16
/// activations are read from device `bfloat`, cast to `half` into the
/// threadgroup tiles, and the matmul runs `half`×`half`→`float`. `half`'s
/// 10-bit mantissa strictly covers `bfloat`'s 7, so the staged operands
/// are lossless and accumulation is fp32 regardless. For `float`/`half`
/// activations `ts == t`. The W buffer is `uint32_t` (packed int4)
/// regardless of `t`.
fn msl_body(t: &str, ts: &str) -> String {
    // Internal name aliases used in the inline body:
    //   x, w, scales, biases, indices, out — kernel params (bound by name)
    //   xs, ws, out_scratch                — threadgroup arrays (hoisted)
    //   m_total, n_out, k_in, group_size   — constexpr scalars
    //   tgid_x, tgid_y                     — threadgroup position aliases
    //   simd_lane                          — thread index in the SG
    format!(
        r#"// --- mt_moe_gather_qmm_mma_int4_bm8_mpp body ({t} acc fp32) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 8u;
constexpr uint BN = 32u;
constexpr uint BK = 16u;

const uint n_tile_base = tgid_x * BN;
const uint m_tile_base = tgid_y * BM;
const uint lane        = simd_lane;

const uint packs_per_row  = k_in / 8u;
const uint groups_per_row = k_in / group_size;

// MPP descriptor. M=8 forces us off the strict-cooperative-tensor
// path (which requires M,N,K ∈ {{16, 32}}); we pass A and B as direct
// `metal::tensor`s and only the destination as a cooperative tensor
// (`ct_c`), which the simdgroup-scope path accepts at M=8.
// ct_c persists across K iters for cross-iteration accumulation under
// multiply_accumulate mode.
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/8, /*N=*/32, /*K=*/16,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

// Tensor-operand type aliases — re-used inside the K loop to materialise
// tA / tB and to spell the operand types in get_destination_cooperative_tensor.
//   tA: A = [M=8,  K=16] row-major, inner = K = 16 → extents{{16, 8 }}
//   tB: B = [N=32, K=16] row-major, inner = K = 16 → extents{{16, 32}}
using tA_t = metal::tensor<threadgroup {ts}, metal::extents<int, 16, 8 >, metal::tensor_inline>;
using tB_t = metal::tensor<threadgroup {ts}, metal::extents<int, 16, 32>, metal::tensor_inline>;

auto ct_c = gemm_op.template get_destination_cooperative_tensor<tA_t, tB_t, float>();

// Walk row sub-runs — same logic as the BM=16 sibling, just over 8 rows.
// Worst case 8 sub-runs (1 row each, e.g. topK=8 decode with each token
// routed to a unique expert).
uint sub_offset = 0u;
for (uint _sub_iter = 0u; _sub_iter < 8u; _sub_iter++) {{
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
            // 32 lanes × 4 elems/lane = 128 = BM*BK. flat = lane*4 + e.
            for (uint e = 0u; e < 4u; e++) {{
                uint flat = lane * 4u + e;
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
            // 32 lanes × 2 packs/lane = 64 packs. Identical to bm16 path.
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

            // Build tensor_inline views over the TG buffers. Types match
            // the tA_t / tB_t aliases above so the destination
            // cooperative_tensor's operand-type template parameters stay
            // consistent. Pass tA and tB directly to gemm_op.run() — no
            // cooperative tensor intermediate (see descriptor comment
            // above for why).
            tA_t tA(xs, metal::extents<int, 16, 8 >{{}});
            tB_t tB(ws, metal::extents<int, 16, 32>{{}});

            gemm_op.run(tA, tB, ct_c);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Stage the fp32 accumulator into the per-SG fp32 scratch — the
        // `ct_c.store(...)` overload requires the destination
        // tensor_inline element type to match `ct_c`'s accumulator type
        // (`float`). We narrow to `{t}` during the coop-write below.
        // Layout is row-major [M=8, N=32], inner = N = 32.
        metal::tensor<threadgroup float, metal::extents<int, 32, 8>, metal::tensor_inline>
            tY(out_scratch, metal::extents<int, 32, 8>{{}});
        ct_c.store(tY);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Coop-write out_scratch -> out[m_tile_base..+BM, n_tile_base..+BN]
        // with fp32 → {t} narrow + per-row expert mask.
        // 32 lanes × 8 elems = 256 = BM*BN.
        for (uint e = 0u; e < 8u; e++) {{
            uint flat = lane * 8u + e;
            uint mr   = flat / BN;            // 0..7
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
    uint m_tile_base = tgid_y * 8u;
    uint n_tile_base = tgid_x * 32u;
    for (uint r = 0u; r < 8u; r++) {{
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

/// Build the [`Kernel`] IR for `mt_moe_gather_qmm_mma_int4_bm8_mpp`.
///
/// Dispatched as `KernelMode::Reduction`. Grid is `[N/32, ceil(M/8), 1]`,
/// threadgroup size is `[32, 1, 1]` (1 simdgroup — required by the MPP
/// descriptor's `execution_simdgroup` scope).
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16 | DType::BF16),
        "mt_moe_gather_qmm_mma_int4_bm8_mpp: dtype must be F32, F16, or BF16, got {:?}",
        dt
    );
    let t = match dt {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        _ => unreachable!(),
    };
    // Staging dtype for the threadgroup tiles + MPP tensor operands.
    // bf16 stages through `half` because `matmul2d` mishandles `bfloat`
    // tensor operands; f32/f16 stage in their own type (ts == t).
    let stage_dt = match dt {
        DType::BF16 => DType::F16,
        other => other,
    };
    let ts = match stage_dt {
        DType::F32 => "float",
        DType::F16 => "half",
        _ => unreachable!(),
    };

    let mut k = Kernel::new("mt_moe_gather_qmm_mma_int4_bm8_mpp");
    k.mode = KernelMode::Reduction;

    // Params — match the bm16_mpp / bm64_mpp signature exactly (call-site
    // compatible across the family):
    //   x       [m_total, k_in]                  T
    //   w       [n_experts, n_out, k_in/8]       u32  (int4-packed)
    //   scales  [n_experts, n_out, k_in/group]   T
    //   biases  [n_experts, n_out, k_in/group]   T
    //   indices [m_total]                        u32
    //   out     [m_total, n_out]                 T
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

    // Constexpr scalars (same as the bm16_mpp / bm64_mpp siblings).
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

    // Build the kernel body. Hoist three TG arrays — xs[8*16], ws[32*16],
    // out_scratch[8*32 fp32]. Force `tgid_y` alias emission via the
    // direct-identifier `Op::Load { src: "tgid_y" }` form (same pattern
    // as the bm16_mpp sibling — see comments there for the rationale).
    let mut body = Block::new(BlockId::new(0));

    // Threadgroup allocations (sizes in elements, not bytes).
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: 8 * 16, // 128
        name: "xs".into(),
    });
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: 32 * 16, // 512
        name: "ws".into(),
    });
    // fp32 staging for the cooperative_tensor `ct_c.store(...)`. The
    // store overload requires destination elem-type == accumulator type
    // (float here). We narrow to `{t}` during the coop-write to global.
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 8 * 32, // 256
        name: "out_scratch".into(),
    });

    // Force `tgid_y` alias emission. InlineMsl body references `tgid_y`
    // by name; codegen's `kernel_uses_program_id_axis(kernel, 1)` check
    // looks at IR ops only, not body text. The direct-identifier load
    // form is the convention (matches `moe_mpp.rs` + the codegen
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

// Bench/inventory registration — same pattern as the bm16_mpp / bm64_mpp
// siblings. `dispatch: Generic` + `shapes: &[]` means `tile bench` skips
// the kernel; correctness lives in
// `tests/moe_gather_qmm_mpp_bm8_correctness.rs`.
inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int4_bm8_mpp",
        kernel_name: "mt_moe_gather_qmm_mma_int4_bm8_mpp",
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
        assert_eq!(k.name, "mt_moe_gather_qmm_mma_int4_bm8_mpp");
        assert_eq!(k.params.len(), 6);
        assert!(k.params[5].is_output);
        assert_eq!(k.constexprs.len(), 4);
        assert!(matches!(k.mode, KernelMode::Reduction));
    }

    #[test]
    fn kernel_ir_constructs_f16() {
        let k = kernel_ir_for(DType::F16);
        let has_half = k.body.ops.iter().any(|op| match op {
            Op::InlineMsl { source, .. } => source.contains("metal::tensor<threadgroup half"),
            _ => false,
        });
        assert!(has_half, "F16 kernel should carry `threadgroup half` tensor view");
    }

    #[test]
    fn kernel_ir_constructs_bf16() {
        let k = kernel_ir_for(DType::BF16);
        // Device params stay bf16 — call-site buffers are bf16.
        assert_eq!(k.params[0].dtype, DType::BF16, "x param should be bf16");
        assert_eq!(k.params[5].dtype, DType::BF16, "out param should be bf16");
        // ...but the MPP staging path runs in `half`: matmul2d mishandles
        // `bfloat` tensor operands, so xs/ws + the tensor views are emitted
        // as `half`, never `bfloat`.
        let src = k
            .body
            .ops
            .iter()
            .find_map(|op| match op {
                Op::InlineMsl { source, .. } => Some(source.clone()),
                _ => None,
            })
            .expect("InlineMsl body");
        assert!(
            src.contains("metal::tensor<threadgroup half"),
            "bf16 kernel should stage through `threadgroup half` tensor views"
        );
        assert!(
            !src.contains("threadgroup bfloat"),
            "bf16 kernel must not emit `threadgroup bfloat` (matmul2d mishandles it)"
        );
    }

    /// Sanity-check that the generated MSL pulls in the MPP header + the
    /// (8, 32, 16) descriptor.
    #[test]
    fn codegen_emits_mpp_include_and_8x32_geometry() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(
            msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
            "MPP include missing from generated MSL"
        );
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int4_bm8_mpp"));
        // BM=8 + BN=32 geometry must be baked into the inline source.
        assert!(msl.contains("constexpr uint BM = 8u"));
        assert!(msl.contains("constexpr uint BN = 32u"));
        // tgid_y alias must be emitted — we depend on it in the inline MSL.
        assert!(msl.contains("tgid_y"), "tgid_y alias missing");
    }

    /// Developer aid — dump the full generated MSL.
    /// `cargo test -p metaltile-std --lib -- moe_mpp_bm8::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
