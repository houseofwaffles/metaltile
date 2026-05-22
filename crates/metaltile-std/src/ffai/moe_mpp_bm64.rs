//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm64_mpp`.
//!
//! BM=BN=64, BK=32 variant of the MPP MoE kernel. Scales the proven
//! `mt_qmm_mma_mpp` 4-SG WM=WN=2 layout (single-expert, BM=BN=BK=32) up to
//! a 64×64 output tile, then wraps it in the per-expert sub-run scheduler
//! from `mt_moe_gather_qmm_mma_int4_bm16_mpp`.
//!
//! ## Why this kernel exists
//!
//! The BM=16 MPP MoE kernel only reaches ~175-205 GF on Qwen3.6-35B-A3B
//! prefill at T=32K B=1 because 1 SG per TG underuses the GPU. MLX's
//! `affine_gather_qmm_rhs_nax` runs at BM=BN=BK=64 with WM=WN=2 (4 SGs per
//! TG) and hits ~3000 GF on the same shape. This kernel closes that gap
//! by mirroring MLX's tile geometry exactly while reusing the MPP DSL
//! escape hatch (`Op::InlineMsl` + `mpp::tensor_ops::matmul2d`).
//!
//! Measured baseline on Qwen3.6-35B-A3B-4bit T=32768 B=1:
//!   MLX:                                              14452 ms prefill
//!   mt_moe_gather_qmm_mma_int4_bm16_mpp_bf16          22399 ms (1.55× slower)
//!
//! Target: match MLX or come within 10%.
//!
//! ## Algorithm
//!
//! - Grid: `[N/64, ceil(M/64), 1]`, one threadgroup per 64×64 output tile.
//! - Threadgroup: 128 lanes = 4 simdgroups (WM=WN=2 — each SG owns a
//!   32×32 sub-tile of the 64×64 output).
//! - Up to 64 expert sub-runs walk the 64 rows in contiguous expert
//!   spans. Production Qwen3.6-A3B T=1024 × 128 experts ≈ 8 rows/expert,
//!   so ~8 sub-runs per TG.
//! - For each sub-run:
//!   1. Coop X load (all 128 lanes): stage X[m_tile..+64, kb..kb+32] → `Xs`
//!      in row-major `[BM=64, TG_LD=36]` (skew=4 for bank-conflict avoidance).
//!   2. Coop W dequant (all 128 lanes): unpack int4 → `{T}` into
//!      `Ws[BN=64, TG_LD=36]` (qmm_t layout: row-major in `[N, K]`).
//!   3. Per-SG `matmul2d<desc=(32,32,32), execution_simdgroup>` over each
//!      SG's 32×32 sub-tile, accumulating into ct_c (mode = multiply_accumulate).
//!   4. After all K-blocks, each SG stores its 32×32 fp32 ct_c into a
//!      per-SG scratch (`OutScratch`), barrier, then a coop-write narrows
//!      to `{T}` and writes to global `out` with per-row sub-run masking.
//!
//! ## Descriptor choice
//!
//! `matmul2d_descriptor(32, 32, 32, false, true, false, multiply_accumulate)`
//! - M=N=K=32 — satisfies Apple's "at least one of M/N/K = 32" constraint
//!   trivially (all three).
//! - `ta=false` → A is `[M=32, K=32]` row-major from `Xs` (the SG's sub-tile).
//! - `tb=true`  → B is `[N=32, K=32]` row-major from `Ws` (qmm_t layout —
//!   same as `mt_qmm_mma_mpp`).
//! - `tc=false` → C is `[M=32, N=32]` row-major fp32.
//!
//! ## Threadgroup memory budget
//!
//! TG_LD = BK + skew = 32 + 4 = 36 elements/row.
//!   Xs:         64 × 36 × sizeof(T)         ≈ 4.6 KB at fp16
//!   Ws:         64 × 36 × sizeof(T)         ≈ 4.6 KB at fp16
//!   OutScratch: 4 × 32 × 32 × sizeof(float) = 16 KB
//!   Total ≈ 25 KB ≤ 32 KB M5 Max per-TG limit. Tight but fits.
//!
//! ## Reference
//!
//! Sibling kernels:
//!   - `crates/metaltile-std/src/mlx/quantized_mpp.rs` (single-expert MPP qmm,
//!     BM=BN=BK=32, 4 SG WM=WN=2 — copy the 4-SG layout pattern from here).
//!   - `crates/metaltile-std/src/ffai/moe_mpp.rs` (per-expert sub-run wrapper
//!     at BM=16 — copy the sub-run scheduling pattern from here).
//!   - `crates/metaltile-std/src/ffai/moe.rs` `mt_moe_gather_qmm_mma_int4_bm16`
//!     (simdgroup-matrix BM=16 MoE — pre-MPP reference for the row mask).
//!
//! MLX upstream: `gather_qmm_rhs_nax_t` in
//! `mlx/backend/metal/kernels/quantized_nax.h` — BM=BN=BK=64, WM=WN=2,
//! SK=32 (inner SG matmul K dim). We simplify by using BK=32 (one matmul
//! per K-block, no inner SK sub-loop) which costs ~2× K-loop overhead but
//! keeps the inline MSL tractable.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind, ValueId},
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

use crate::spec::{BenchDispatch, BenchSpec};

/// Tile geometry — keep in lock-step with the inline MSL below.
pub const BM: u32 = 64;
pub const BN: u32 = 64;
pub const BK: u32 = 32;
/// 4 SGs × 32 lanes (WM=WN=2).
pub const TPG: u32 = 128;
/// Threadgroup-mem row stride. Set to BK (= 32) — no skew, because at
/// BM=BN=64 the per-tile memory budget would otherwise exceed M5 Max's
/// 32 KB per-TG limit (with skew=4 we'd hit 34816 B; without skew we're
/// at ~32.5 KB which still requires the per-SG output scratch reorg
/// below — see kernel_ir_for). If perf later shows bank-conflict
/// pressure on the matmul2d frag loads, revisit (re-add skew + shrink
/// OutScratch by serializing the SG store-then-coop-write phase).
pub const TG_LD: u32 = BK; // 32

/// Render the inline MSL body for the BM=64 MPP MoE kernel.
///
/// `t` is the MSL type of the device-side params (`x`, `out`) — `"half"`,
/// `"float"`, or `"bfloat"`. `ts` is the *staging* type used for the
/// threadgroup tiles + MPP cooperative tensors — `"half"` for both `half`
/// and `bfloat` activations, `"float"` for `float`.
///
/// Why the split: Apple's `mpp::tensor_ops::matmul2d` does not handle
/// `bfloat` cooperative tensors correctly (verified on M5 Max — `bfloat`
/// coop tensors produce garbage while the bit-identical `half` path is
/// correct). bf16 activations are therefore read from device `bfloat`,
/// cast to `half` on the way into the threadgroup tiles, and the matmul
/// runs `half`×`half`→`float`. `half`'s 10-bit mantissa strictly covers
/// `bfloat`'s 7, so this is lossless for the staged operands and the
/// accumulation is fp32 regardless. `out` is narrowed back to `t` on
/// store. For `float`/`half` activations `ts == t`, so the emitted MSL is
/// byte-identical to the pre-split kernel.
///
/// The W buffer is `uint32_t` (packed int4) regardless of `t`. Group size
/// baked in at 64 (Qwen3.6-A3B default).
fn msl_body(t: &str, ts: &str) -> String {
    // Internal name aliases used in the inline body:
    //   x, w, scales, biases, indices, out — kernel params (bound by name)
    //   Xs, Ws, OutScratch                 — threadgroup arrays (hoisted)
    //   m_total, n_out, k_in, group_size   — constexpr scalars
    //   tgid_x, tgid_y                     — threadgroup position aliases
    //   simd_group, simd_lane              — SG-id + lane in SG
    //
    // The dequant uses the same "mask-without-shift" pattern as
    // `mt_qmm_mma_mpp`: nibble at position i = (packed & (0xf << 4i)) * 16^-i.
    format!(
        r#"// --- mt_moe_gather_qmm_mma_int4_bm64_mpp body ({t} acc fp32) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 64u;
constexpr uint BN = 64u;
constexpr uint BK = 32u;
constexpr uint TG_LD = 32u;        // = BK (no skew — see Rust-side comment)
constexpr uint TPG   = 128u;       // 4 SG × 32 lanes

// Per-TG output tile origin (in elements, not tiles).
const uint m_tile_base = tgid_y * BM;
const uint n_tile_base = tgid_x * BN;

// Lane-in-TG and per-SG sub-tile origin. 4 SGs in a 2×2 WM=WN=2 grid:
//   sm = simd_group / 2 ∈ {{0, 1}}, sn = simd_group & 1 ∈ {{0, 1}}.
// Each SG owns a 32×32 sub-tile at (sm*32, sn*32) inside the 64×64.
const uint lane_in_tg = simd_group * 32u + simd_lane;
const uint sm = simd_group / 2u;
const uint sn = simd_group & 1u;
const uint sg_m_base = sm * 32u;
const uint sg_n_base = sn * 32u;

const uint packs_per_row  = k_in / 8u;
const uint groups_per_row = k_in / group_size;

// ── X coop-load mapping ─────────────────────────────────────────────────
// 128 lanes × 16 contiguous K-elems per lane × 1 m-row per lane =
// 128 × 16 = 2048 = BM(=64) × BK(=32). Mapping:
//   m_row   = lane_in_tg / 2 ∈ 0..64
//   k_quad  = lane_in_tg & 1 ∈ 0..2  (each lane writes 16 contiguous K)
//   k_base  = k_quad * 16
const uint x_m_row  = lane_in_tg / 2u;
const uint x_k_quad = lane_in_tg & 1u;
const uint x_k_base = x_k_quad * 16u;

// ── W coop-dequant mapping ─────────────────────────────────────────────
// 128 lanes × 2 packs/lane × 8 nibbles/pack = 2048 = BN(=64) × BK(=32).
//   pack_id     = lane_in_tg * 2 + pi   for pi in 0..2  (256 packs total)
//   w_row       = pack_id / 4 ∈ 0..64   (BK=32 → 4 packs per N-row)
//   pack_in_row = pack_id & 3 ∈ 0..4
// Mask-without-shift dequant constants (same pattern as the MPP qmm path).
constexpr float s_16   = 0.0625f;
constexpr float s_256  = 0.00390625f;
constexpr float s_4096 = 0.000244140625f;

// ── Set up MPP matmul: (M=32, N=32, K=32), ta=false, tb=true ───────────
// `tb=true` lets us read B from the qmm_t [BN, BK] layout without
// transposing. mode=multiply_accumulate so ct_c persists across K-block
// iterations (zero ct_c once before the K-loop per sub-run).
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/32, /*N=*/32, /*K=*/32,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<{ts}, {ts}, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<{ts}, {ts}, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

// Walk row sub-runs — same logic as the BM=16 sibling, just over 64 rows.
// Worst case 64 sub-runs (1 row each); production Qwen3.6-A3B
// T=1024 × 128 experts ≈ 8 rows/expert → ~8 sub-runs per TG.
uint sub_offset = 0u;
for (uint _sub_iter = 0u; _sub_iter < BM; _sub_iter++) {{
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
            // ── 1. Stage X[m_tile_base..+BM, kb..kb+BK] -> Xs[BM*TG_LD]
            //    128 lanes × 16 contiguous K elems each fills the 64×32 tile.
            uint gr_x = m_tile_base + x_m_row;
            // Per-row mask: in [sub_offset, sub_end) AND global row valid.
            bool in_run_x =
                (x_m_row >= sub_offset) && (x_m_row < sub_end) && (gr_x < m_total);
            uint safe_gr_x = in_run_x ? gr_x : 0u;
            uint x_dev_base = safe_gr_x * k_in + kb + x_k_base;
            uint x_ws_base  = x_m_row * TG_LD + x_k_base;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 16u; i++) {{
                // Explicit cast: device `x` is `{t}`; for bf16 activations
                // this narrows bfloat → half for the staged matmul. For
                // f32/f16 it is a no-op the compiler elides.
                {ts} xv = ({ts})x[x_dev_base + i];
                Xs[x_ws_base + i] = in_run_x ? xv : ({ts})0;
            }}

            // ── 2. Dequant W[expert, n_tile_base..+BN, kb..kb+BK] -> Ws
            //    128 lanes × 2 packs/lane = 256 packs = BN(=64) × (BK/8 = 4).
            #pragma clang loop unroll(full)
            for (uint pi = 0u; pi < 2u; pi++) {{
                uint pack_id     = lane_in_tg * 2u + pi;
                uint w_row       = pack_id / 4u;       // 0..63 (BN rows)
                uint pack_in_row = pack_id & 3u;       // 0..3  (BK=32 → 4 packs/N-row)
                uint pack_dev    = w_expert_base
                                 + (n_tile_base + w_row) * packs_per_row
                                 + (kb / 8u)
                                 + pack_in_row;
                uint packed = w[pack_dev];
                uint k_off  = kb + pack_in_row * 8u;
                uint g      = k_off / group_size;
                uint sb_off = sb_expert_base
                            + (n_tile_base + w_row) * groups_per_row
                            + g;
                float s = (float)scales[sb_off];
                float b = (float)biases[sb_off];
                uint pack_hi = packed >> 16u;
                float q0 = (float)(packed   &     15u);
                float q1 = (float)(packed   &    240u) * s_16;
                float q2 = (float)(packed   &   3840u) * s_256;
                float q3 = (float)(packed   &  61440u) * s_4096;
                float q4 = (float)(pack_hi  &     15u);
                float q5 = (float)(pack_hi  &    240u) * s_16;
                float q6 = (float)(pack_hi  &   3840u) * s_256;
                float q7 = (float)(pack_hi  &  61440u) * s_4096;
                uint ws_base = w_row * TG_LD + pack_in_row * 8u;
                Ws[ws_base + 0u] = ({ts})(s * q0 + b);
                Ws[ws_base + 1u] = ({ts})(s * q1 + b);
                Ws[ws_base + 2u] = ({ts})(s * q2 + b);
                Ws[ws_base + 3u] = ({ts})(s * q3 + b);
                Ws[ws_base + 4u] = ({ts})(s * q4 + b);
                Ws[ws_base + 5u] = ({ts})(s * q5 + b);
                Ws[ws_base + 6u] = ({ts})(s * q6 + b);
                Ws[ws_base + 7u] = ({ts})(s * q7 + b);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── 3. Per-SG tensor views over the TG tiles ─────────────────
            // ct_a reads A [32, 32] = Xs[sg_m_base..+32, 0..32] (row-major,
            // ld=TG_LD). ct_b reads B [32, 32] = Ws[sg_n_base..+32, 0..32]
            // (row-major, ld=TG_LD) — tb=true treats this as K×N column-major
            // = the qmm_t weight tile. `tensor_inline` packed-stride ctor:
            // extents{{TG_LD, 32}} → stride[0]=1 along TG_LD (k-inner),
            // stride[1]=TG_LD along 32 (m or n outer).
            threadgroup {ts}* xs_sg = Xs + sg_m_base * TG_LD;
            threadgroup {ts}* ws_sg = Ws + sg_n_base * TG_LD;
            metal::tensor<threadgroup {ts}, metal::extents<int, TG_LD, 32>, metal::tensor_inline>
                tA(xs_sg, metal::extents<int, TG_LD, 32>{{}});
            metal::tensor<threadgroup {ts}, metal::extents<int, TG_LD, 32>, metal::tensor_inline>
                tB(ws_sg, metal::extents<int, TG_LD, 32>{{}});

            ct_a.load(tA);
            ct_b.load(tB);

            gemm_op.run(ct_a, ct_b, ct_c);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // ── 4. Store per-SG 32×32 fp32 ct_c into a per-SG scratch slot,
        //    barrier, then coop-write the 64×64 staged tile to global out
        //    with fp32 → {t} narrow + per-row expert mask.
        threadgroup float* sg_scratch = OutScratch + simd_group * (32u * 32u);
        metal::tensor<threadgroup float, metal::extents<int, 32, 32>, metal::tensor_inline>
            tC(sg_scratch, metal::extents<int, 32, 32>{{}});
        ct_c.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Coop-write OutScratch -> out[m_tile_base..+BM, n_tile_base..+BN].
        // 128 lanes × 32 elems = 4096 = BM*BN. Mapping:
        //   flat = lane_in_tg * 32 + e
        //   mr   = flat / BN ∈ 0..64
        //   nc   = flat % BN ∈ 0..64
        // Find which SG's scratch holds (mr, nc):
        //   ssm = mr / 32 ∈ {{0, 1}}, ssn = nc / 32 ∈ {{0, 1}}
        //   src_sg = ssm * 2 + ssn
        //   sub_mr = mr & 31, sub_nc = nc & 31
        #pragma clang loop unroll(full)
        for (uint e = 0u; e < 32u; e++) {{
            uint flat   = lane_in_tg * 32u + e;
            uint mr     = flat / BN;
            uint nc     = flat & 63u;          // BN=64 → mod via mask
            uint gr     = m_tile_base + mr;
            uint gc     = n_tile_base + nc;
            bool in_run = (mr >= sub_offset) && (mr < sub_end)
                        && (gr < m_total) && (gc < n_out);
            if (in_run) {{
                uint ssm     = mr >> 5u;       // mr / 32
                uint ssn     = nc >> 5u;       // nc / 32
                uint src_sg  = ssm * 2u + ssn;
                uint sub_mr  = mr & 31u;
                uint sub_nc  = nc & 31u;
                float v      = OutScratch[src_sg * (32u * 32u) + sub_mr * 32u + sub_nc];
                out[gr * n_out + gc] = ({t})v;
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    sub_offset = sub_end;
}}
#else
// Pre-Metal-4 fallback: write zeros to keep the metallib linkable but
// fail the correctness test loud-and-clear.
if (simd_group == 0u && simd_lane == 0u) {{
    uint m_tile_base = tgid_y * 64u;
    uint n_tile_base = tgid_x * 64u;
    for (uint r = 0u; r < 64u; r++) {{
        uint gr = m_tile_base + r;
        if (gr >= m_total) continue;
        for (uint c = 0u; c < 64u; c++) {{
            uint gc = n_tile_base + c;
            if (gc >= n_out) continue;
            out[gr * n_out + gc] = ({t})0;
        }}
    }}
}}
#endif
"#,
        t = t,
        ts = ts
    )
}

/// Build the [`Kernel`] IR for `mt_moe_gather_qmm_mma_int4_bm64_mpp`.
///
/// Dispatched as `KernelMode::Reduction`. Grid is `[ceil(N/64), ceil(M/64), 1]`,
/// threadgroup size is `[128, 1, 1]` (4 simdgroups WM=WN=2).
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16 | DType::BF16),
        "mt_moe_gather_qmm_mma_int4_bm64_mpp: dtype must be F32, F16, or BF16, got {:?}",
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

    let mut k = Kernel::new("mt_moe_gather_qmm_mma_int4_bm64_mpp");
    k.mode = KernelMode::Reduction;

    // Params — match the bm16_mpp signature exactly (call-site compatible):
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

    // Constexpr scalars (same as the bm16_mpp sibling).
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

    // Build the kernel body. Hoist three TG arrays — Xs[BM*TG_LD],
    // Ws[BN*TG_LD], OutScratch[4*32*32 fp32]. Force `tgid_y` + `simd_group`
    // alias emission via synthetic ProgramId(axis=1) op (mirrors the
    // bm16_mpp + qmm_mma_mpp pattern).
    let mut body = Block::new(BlockId::new(0));

    // Threadgroup allocations (sizes in elements, not bytes).
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: BM * TG_LD, // 64 * 36 = 2304
        name: "Xs".into(),
    });
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: stage_dt,
        size: BN * TG_LD, // 64 * 36 = 2304
        name: "Ws".into(),
    });
    // 4 SG × 32 × 32 fp32 scratch (16 KB).
    body.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 4 * 32 * 32,
        name: "OutScratch".into(),
    });

    // Force `tgid_y` alias emission. See moe_mpp.rs / quantized_mpp.rs for
    // why: `kernel_uses_program_id_axis(kernel, 1)` only returns true when
    // an `Op::ProgramId { axis: 1 }` op is present in the body.
    body.push_op(Op::ProgramId { axis: 1 }, ValueId::new(0));

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

// Bench/inventory registration — same pattern as the bm16_mpp sibling.
// `dispatch: Generic` + `shapes: &[]` means `tile bench` skips the kernel;
// correctness lives in `tests/moe_gather_qmm_mpp_bm64_correctness.rs`.
inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_mma_int4_bm64_mpp",
        kernel_name: "mt_moe_gather_qmm_mma_int4_bm64_mpp",
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
        assert_eq!(k.name, "mt_moe_gather_qmm_mma_int4_bm64_mpp");
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
        // `bfloat` cooperative tensors, so Xs/Ws + the tensor views are
        // emitted as `half`, never `bfloat`.
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
    /// (32,32,32) descriptor.
    #[test]
    fn codegen_emits_mpp_include_and_64x64_geometry() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(
            msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
            "MPP include missing from generated MSL"
        );
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int4_bm64_mpp"));
        // BM=BN=64 geometry must be baked into the inline source.
        assert!(msl.contains("constexpr uint BM = 64u"));
        assert!(msl.contains("constexpr uint BN = 64u"));
        // tgid_y alias must be emitted — we depend on it in the inline MSL.
        assert!(msl.contains("tgid_y"), "tgid_y alias missing");
    }

    /// Developer aid — dump the full generated MSL.
    /// `cargo test -p metaltile-std --lib -- moe_mpp_bm64::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
