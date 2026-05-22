//! `mt_qmm_mma_mpp` — production int4 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! This is the MPP (MetalPerformancePrimitives) counterpart of
//! `mt_qmm_mma`. It mirrors the same algorithm — int4 weights dequantized
//! into threadgroup memory once per K-block, then a per-simdgroup matmul
//! against the fp T X-tile — but replaces the manual 8×8 `simdgroup_matmul`
//! ladder with one cooperative `matmul2d` per SG per K-block. The
//! cooperative-tensor path is what taps the NAX hardware MLX uses for
//! `affine_qmm_t_nax` / `gather_qmm_rhs_nax` (≈3000 GF on Qwen3.6-A3B
//! `down_proj` shapes).
//!
//! Built as an IR escape-hatch via `Op::InlineMsl` rather than the
//! `#[kernel]` macro because the macro front-end does not (yet) expose
//! `mpp::` types. Geometry mirrors `mt_qmm_mma`:
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile (1024 outputs/TG)
//!   Grid: [N/32, M/32, 1]
//!   Per SG: one 16×16×32 `matmul2d` per K-block (acc-mode multiply_accumulate)
//!
//! Per-K-block layout (cooperative, all 128 lanes):
//!   1. X-tile coop-load → Xs[BM × (BK+4)] (skewed for bank-conflict avoidance)
//!   2. W-tile coop-dequant int4 → Ws[BN × (BK+4)] in fp T
//!   3. threadgroup_barrier
//!   4. Each SG calls `gemm_op.run(ct_a, ct_b, ct_c)` where ct_c persists across
//!      K-blocks (matmul2d_descriptor::mode::multiply_accumulate)
//!   5. threadgroup_barrier
//!
//! After all K-blocks, each SG stores its 16×16 fp32 ct_c via
//! `tensor_inline<half, extents<16,16>>` cast over the device `out` pointer
//! (cast to half on store; same final-precision as mt_qmm_mma).
//!
//! The MPP descriptor uses `(M=16, N=16, K=32)` — K=32 satisfies the
//! Apple "at least one of M/N/K = 32" assertion for two-operand
//! cooperative tensors. With `ta=false, tb=true` we read B from a
//! `[BN, BK]` (= W-layout) threadgroup tile, matching the `mt_qmm_mma`
//! dequant W layout exactly.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/qmm_mpp_correctness.rs`.
//!
//! Runtime behavior on `gen < 17` GPUs (M3 and earlier): the MSL body is
//! `#if __METAL_VERSION__ >= 400` gated so the metallib still links cleanly
//! on Metal 3 toolchains, with a no-op `#else` arm. Caller-side dispatch
//! must skip this kernel on unsupported GPU gens — `KernelFeatures::needs_mpp`
//! is the runtime gate; downstream callers route to `mt_qmm_mma` (the
//! non-MPP `simdgroup_matmul` variant) when `needs_mpp` is unsupported.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind},
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

/// Tile geometry — keep in lock-step with the inline MSL below.
pub const BM: u32 = 32;
pub const BN: u32 = 32;
pub const BK: u32 = 32;
/// Threads per group (4 SG × 32 lanes).
pub const TPG: u32 = 128;
/// Threadgroup-mem row skew (matches `mt_qmm_mma` xs_ld_const). 4 elems of
/// padding past BK to scatter 32-bank conflicts on the column reads done
/// inside `matmul2d`'s frag load. Stride = BK + 4 = 36.
pub const TG_SKEW: u32 = 4;
pub const TG_LD: u32 = BK + TG_SKEW; // 36

/// MSL source. References the kernel parameters by name; codegen emits
/// the bindings as `const device {T} *w/scales/biases/x` + `device {T} *out`
/// + `constant uint &k/n/gs_per_row` per the standard `Param` /
///   `ConstExprDecl` signature path.
///
/// Templated on `T` (fp32 / fp16) at metaltile-build time via the per-dtype
/// kernel-IR (`kernel_ir_for(DType)`) — the `{T}` placeholder is rewritten
/// before codegen. Group size baked in at 64 (Qwen3.6-A3B / Qwen3.6-27B-UD
/// default).
const QMM_MMA_MPP_SRC_TEMPLATE: &str = r#"// --- mt_qmm_mma_mpp body (BM=BN=BK=32, TG=128, 4 SGs WM=WN=2) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 32;
constexpr uint BN = 32;
constexpr uint BK = 32;
constexpr uint TG_LD = 36;     // BK + 4 skew
constexpr uint GROUP_SIZE = 64;

// Threadgroup tiles — Xs holds X in (m, k) row-major; Ws holds dequant W in
// (n, k) row-major (qmm_t layout). Skew of 4 elements past BK breaks the
// 32-bank conflict on the column-strided frag loads inside `matmul2d`.
threadgroup {T} Xs[BM * TG_LD];
threadgroup {T} Ws[BN * TG_LD];

// Per-TG output tile origin in (m, n).
const uint m_tile = tgid_y;
const uint n_tile = tgid_x;
const uint lane_in_tg = simd_group * 32u + simd_lane;
// 4 SGs in a 2×2 WM=WN=2 warp grid: sm = simd_group/2, sn = simd_group%2.
// Each SG owns a 16×16 sub-tile at (sm*16, sn*16) inside the 32×32 output.
const uint sm = simd_group / 2u;
const uint sn = simd_group & 1u;

// ── X coop-load mapping ──
// 128 lanes × 8 contiguous K-elems each fill the 1024-elt Xs tile.
// lane_in_tg ∈ 0..128, m_row = lane_in_tg / 4 ∈ 0..32, k_quad = lane_in_tg & 3 ∈ 0..4.
// Per lane writes 8 contiguous {T} into Xs[m_row*TG_LD + k_quad*8 + i] for i in 0..8.
const uint x_m_row  = lane_in_tg / 4u;
const uint x_k_quad = lane_in_tg & 3u;
const uint x_k_base = x_k_quad * 8u;

// ── W coop-dequant mapping ──
// 128 packs / 128 lanes = 1 pack per lane. lane_in_tg = w_row*4 + pack_in_row.
// Each lane dequants 8 nibbles → Ws[w_row*TG_LD + pack_in_row*8 + i].
const uint w_row        = lane_in_tg / 4u;
const uint pack_in_row  = lane_in_tg & 3u;

const uint x_m_base = m_tile * 32u;
const uint w_n_base = n_tile * 32u;
const uint packs_per_row = k / 8u;
const uint sb_base = (w_n_base + w_row) * gs_per_row;
const uint w_pack_row_base = (w_n_base + w_row) * packs_per_row;

// ── Set up MPP matmul: (M=16, N=16, K=32), ta=false, tb=true, tc=false ──
// `tb=true` lets us read B from the [BN, BK] = W-layout threadgroup tile
// without transposing in memory. mode=multiply_accumulate so ct_c persists
// across K-block iterations (we zero it once before the K-loop).
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/16, /*K=*/32,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<{T}, {T}, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<{T}, {T}, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

// Zero accumulator (mode = multiply_accumulate adds to dst on each run()).
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    ct_c[i] = 0.0f;
}

// Per-SG sub-tile origin inside the 32×32 TG tile.
const uint sg_m_base = sm * 16u;
const uint sg_n_base = sn * 16u;

for (uint kb = 0u; kb < k; kb += BK) {
    // ── 1. Coop X load (128 lanes × 8 contiguous K) ──
    const uint x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
    const uint x_ws_base = x_m_row * TG_LD + x_k_base;
    {T} xv0 = ({T})x[x_row_dev_base + 0u];
    {T} xv1 = ({T})x[x_row_dev_base + 1u];
    {T} xv2 = ({T})x[x_row_dev_base + 2u];
    {T} xv3 = ({T})x[x_row_dev_base + 3u];
    {T} xv4 = ({T})x[x_row_dev_base + 4u];
    {T} xv5 = ({T})x[x_row_dev_base + 5u];
    {T} xv6 = ({T})x[x_row_dev_base + 6u];
    {T} xv7 = ({T})x[x_row_dev_base + 7u];
    Xs[x_ws_base + 0u] = xv0;
    Xs[x_ws_base + 1u] = xv1;
    Xs[x_ws_base + 2u] = xv2;
    Xs[x_ws_base + 3u] = xv3;
    Xs[x_ws_base + 4u] = xv4;
    Xs[x_ws_base + 5u] = xv5;
    Xs[x_ws_base + 6u] = xv6;
    Xs[x_ws_base + 7u] = xv7;

    // ── 2. Coop W dequant — 1 pack/lane → 8 fp {T} into Ws ──
    const uint pack_k_off = kb / 8u + pack_in_row;
    const uint pack = w[w_pack_row_base + pack_k_off];
    const uint k_off = kb + pack_in_row * 8u;
    const uint g = k_off / GROUP_SIZE;
    const float s = (float)scales[sb_base + g];
    const float b = (float)biases[sb_base + g];
    // Mask-without-shift trick (matches mt_qmm_mma) — multiply by 1/16^i
    // instead of right-shifting.
    const float s_16    = 0.0625f;
    const float s_256   = 0.00390625f;
    const float s_4096  = 0.000244140625f;
    const uint pack_hi = pack >> 16u;
    const float q0 = (float)(pack    &     15u);
    const float q1 = (float)(pack    &    240u) * s_16;
    const float q2 = (float)(pack    &   3840u) * s_256;
    const float q3 = (float)(pack    &  61440u) * s_4096;
    const float q4 = (float)(pack_hi &     15u);
    const float q5 = (float)(pack_hi &    240u) * s_16;
    const float q6 = (float)(pack_hi &   3840u) * s_256;
    const float q7 = (float)(pack_hi &  61440u) * s_4096;
    const uint ws_base = w_row * TG_LD + pack_in_row * 8u;
    Ws[ws_base + 0u] = ({T})(s * q0 + b);
    Ws[ws_base + 1u] = ({T})(s * q1 + b);
    Ws[ws_base + 2u] = ({T})(s * q2 + b);
    Ws[ws_base + 3u] = ({T})(s * q3 + b);
    Ws[ws_base + 4u] = ({T})(s * q4 + b);
    Ws[ws_base + 5u] = ({T})(s * q5 + b);
    Ws[ws_base + 6u] = ({T})(s * q6 + b);
    Ws[ws_base + 7u] = ({T})(s * q7 + b);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. Build per-SG tensor views over the TG tiles ──
    // ct_a reads A [16, 32] = Xs[sg_m_base..sg_m_base+16, 0..32] (row-major, ld=TG_LD).
    // ct_b reads B [16, 32] = Ws[sg_n_base..sg_n_base+16, 0..32] (row-major, ld=TG_LD)
    //   — with tb=true, MPP treats this as K×N column-major = the qmm_t weight tile.
    // `tensor_inline` packed-stride ctor: strides[0]=1, strides[1]=extents[0]=TG_LD.
    // So extents{TG_LD, 16} means stride[0]=1 along the TG_LD-wide axis (k-inner),
    // stride[1]=TG_LD along the 16-wide axis (m or n). That matches our row-major
    // staging perfectly: contiguous in k, strided by TG_LD across rows.
    threadgroup {T}* xs_sg = Xs + sg_m_base * TG_LD;
    threadgroup {T}* ws_sg = Ws + sg_n_base * TG_LD;
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tA(xs_sg, metal::extents<int, TG_LD, 16>{});
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tB(ws_sg, metal::extents<int, TG_LD, 16>{});

    ct_a.load(tA);
    ct_b.load(tB);

    // ── 4. Run the matmul; ct_c accumulates ──
    gemm_op.run(ct_a, ct_b, ct_c);

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── 5. Store ct_c to global out (cast fp32 → {T}) ──
// Output tile origin for this SG.
const uint out_m_base = m_tile * 32u + sg_m_base;
const uint out_n_base = n_tile * 32u + sg_n_base;
// We need to write a 16×16 fp32 ct_c to a 16×16 {T} sub-tile of out.
// `ct_c.store(...)` writes through a tensor_inline view. For {T}=half we
// store directly; for {T}=float the cast is a no-op. Use a temporary fp32
// stage then cast — simplest cross-dtype path. Stage in a small per-SG TG
// scratch to keep ct_c.store typed as float.
// Per-SG fp32 scratch (4 SG × 256 floats = 4 KB).
threadgroup float OutScratch[4 * 16 * 16];
threadgroup float* sg_scratch = OutScratch + simd_group * (16 * 16);
metal::tensor<threadgroup float, metal::extents<int, 16, 16>, metal::tensor_inline>
    tC(sg_scratch, metal::extents<int, 16, 16>{});
ct_c.store(tC);
// Cooperative store back to device out, casting to {T}. 32 lanes × 8 elems
// covers 16×16=256.
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint lane = simd_lane;
// Map lane → (row, col) in the 16×16 sub-tile: 32 lanes × 8 elems = 256
// outputs. Lane handles row = lane/2, col_base = (lane & 1) * 8 → 8 contig
// cols per lane in one row.
const uint o_row = lane / 2u;
const uint o_col_base = (lane & 1u) * 8u;
#pragma clang loop unroll(full)
for (uint i = 0u; i < 8u; ++i) {
    out[(out_m_base + o_row) * n + (out_n_base + o_col_base + i)] =
        ({T})sg_scratch[o_row * 16u + o_col_base + i];
}
#else
// Pre-Metal-4 fallback — silence the bindings so the metallib still links.
// Correctness test on such targets is the intended failure signal.
if (simd_group == 0u && simd_lane == 0u) {
    const uint o = tgid_y * 32u * n + tgid_x * 32u;
    const uint _gs = gs_per_row; // silence unused-var
    out[o] = ({T})((float)x[0] * (float)scales[0] + (float)biases[0]) * ({T})(w[0] & 15u) * ({T})_gs;
}
#endif
"#;

/// Substitute the `{T}` placeholder for the per-dtype MSL source.
fn substitute_dtype(src: &str, dt: DType) -> String {
    let t = match dt {
        DType::F32 => "float",
        DType::F16 => "half",
        _ => unreachable!("kernel_ir_for asserts dtype before reaching here"),
    };
    src.replace("{T}", t)
}

/// Build the per-dtype [`Kernel`] IR for `mt_qmm_mma_mpp_{T}`.
///
/// Param layout (lock-step with `run_qmm_mma_mpp` in the correctness test):
///   buffer(0) = w        const device uint  *
///   buffer(1) = scales   const device {T}   *
///   buffer(2) = biases   const device {T}   *
///   buffer(3) = x        const device {T}   *
///   buffer(4) = out      device       {T}   *
///   buffer(5) = k        constant     uint  &
///   buffer(6) = n        constant     uint  &
///   buffer(7) = gs_per_row constant   uint  &
///
/// Dispatch geometry: grid `[n/32, m/32, 1]`, threadgroup `[128, 1, 1]`.
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_qmm_mma_mpp only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_qmm_mma_mpp");
    k.mode = KernelMode::Reduction;

    // Buffers. Shapes are nominal — codegen emits the C-pointer signature
    // regardless. Concrete shapes (M, N, K) are passed via the `k`/`n`
    // constexpr scalars at dispatch time.
    k.params.push(Param {
        name: "w".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "scales".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "biases".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "x".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
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

    // Constexpr scalars (passed via setBytes after the buffers).
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("gs_per_row"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // Force `tgid_y` alias emission. InlineMsl source mentions `tgid_y` but
    // the body-text isn't scanned for the alias trigger — codegen only looks
    // at IR ops. Use the `Op::Load { src: "tgid_y" }` direct-identifier form
    // (per `ssm_step` precedent, see `reduction_preamble_emits_tgid_y_when_used_as_identifier`
    // test in `metaltile-codegen/src/msl/mod.rs`). Reduction mode emits
    // `tgid_x` unconditionally, so axis=0 needs no hint.
    use metaltile_core::ir::ValueId;
    let mut body = Block::new(BlockId::new(0));
    body.push_op(
        Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(0),
    );
    body.push_op_no_result(Op::InlineMsl {
        source: substitute_dtype(QMM_MMA_MPP_SRC_TEMPLATE, dt),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body.clone();
    let mut blocks = FxHashMap::default();
    blocks.insert(BlockId::new(0), body);
    k.blocks = blocks;

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_for_f32_and_f16() {
        for dt in [DType::F32, DType::F16] {
            let k = kernel_ir_for(dt);
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
            // Body has `Op::Load { src: "tgid_y" }` (tgid_y alias trigger) + InlineMsl.
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(
                k.body.ops.iter().any(|op| matches!(op, Op::Load { src, .. } if src == "tgid_y"))
            );
        }
    }

    /// Developer aid — `cargo test -p metaltile-std --lib quantized_mpp::tests::dump_generated_msl -- --nocapture`.
    /// Always passes; gated behind `--nocapture` for output.
    #[test]
    fn dump_generated_msl() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = kernel_ir_for(DType::F16);
        k.name = "mt_qmm_mma_mpp_f16".to_string();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
            let mut k = kernel_ir_for(dt);
            // Per-dtype naming convention used by the `tile emit` subcommand.
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_mma_mpp_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_mma_mpp_{suffix}")));
            // Sanity-check the dtype substitution landed.
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            // Sanity-check tgid_y is bound (Reduction mode emits it when
            // `kernel_uses_program_id_axis(1)` returns true, which the
            // synthetic Op::ProgramId { axis: 1 } guarantees).
            assert!(
                msl.contains("tgid_y"),
                "tgid_y must be bound (synthetic ProgramId axis=1 op):\n{msl}"
            );
        }
    }
}
