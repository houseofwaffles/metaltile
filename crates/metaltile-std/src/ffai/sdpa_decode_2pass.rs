//! Two-pass SDPA decode — pass 1 emits per-block (max, sum, partial_o)
//! into staging buffers; pass 2 merges across blocks to produce the
//! normalised output.
//!
//! Geometry mirrors MLX `sdpa_vector_2pass`:
//! - pass 1 TG `(BD=32, gqa_factor, 1)` — one simdgroup per Q head
//!   in the GQA group, sharing K/V loads via L1
//! - pass 1 grid `(n_kv_heads, blocks, 1)` — `block_idx = tgid_y`
//!   (MLX uses tid.z; metaltile's DSL doesn't expose tgid_z, so we
//!   collapse batch into y — single-decode is batch=1 anyway)
//! - pass 1 stride: `for i = block_idx; i < N; i += blocks` so adjacent
//!   TGs read adjacent K rows in the same wave (coalesced)
//! - pass 2 TG `(1024, 1, 1)` = 32 sg × 32 lanes, one TG per Q head
//! - pass 2 loop `b < blocks / 32`; **`blocks` MUST be multiple of 32**
//!   (otherwise the reducer silently drops partials)
//!
//! head_dim hardcoded to 128; online softmax in fp32 throughout.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// MLX-derived `blocks` value for chained 2-pass dispatch on Apple
/// M5 Max (architecture char `'s'`). Mirrors the curve in upstream
/// `mlx/backend/metal/scaled_dot_product_attention.cpp`:
///
/// ```text
/// devc == 's':
///   blocks = 64
///   if N > 1024 and n_simds > 4:
///     N <=  8192 → 128
///     N <= 32768 → 256
///     N <= 65536 → 512
///     else       → 1024
/// ```
///
/// `n_simds` = `gqa_factor * q_seq_len`. For single-token decode with
/// Qwen3-class GQA (gqa_factor=4), `n_simds == 4` and the heuristic
/// stays at 64.
///
/// **`blocks` MUST be a multiple of 32** — the pass-2 reducer loops
/// `b < blocks / 32` and silently drops partials otherwise.
pub fn recommended_blocks_m5_max(n_kv: u32, n_simds: u32) -> u32 {
    if n_kv <= 1024 || n_simds <= 4 {
        return 64;
    }
    match n_kv {
        ..=8_192 => 128,
        8_193..=32_768 => 256,
        32_769..=65_536 => 512,
        _ => 1024,
    }
}

#[cfg(test)]
mod heuristic_tests {
    use super::recommended_blocks_m5_max;
    #[test]
    fn matches_upstream_curve_at_known_points() {
        // Qwen3-class single-decode: gqa=4, q_seq=1, n_simds=4.
        // Heuristic gate is n_simds > 4 — stays at the 64 floor.
        for &n in &[256u32, 1024, 4096, 16384, 65_536, 131_072] {
            assert_eq!(recommended_blocks_m5_max(n, 4), 64);
        }
        // Llama-3-70B-class: gqa=8, n_simds=8. Curve fires.
        assert_eq!(recommended_blocks_m5_max(1025, 8), 128);
        assert_eq!(recommended_blocks_m5_max(8_192, 8), 128);
        assert_eq!(recommended_blocks_m5_max(32_768, 8), 256);
        assert_eq!(recommended_blocks_m5_max(65_536, 8), 512);
        assert_eq!(recommended_blocks_m5_max(131_072, 8), 1024);
        assert_eq!(recommended_blocks_m5_max(1024, 8), 64);
        // All outputs divisible by 32 (pass-2 reducer hard constraint).
        for &n in &[256u32, 1025, 8192, 32_768, 65_536, 131_072, 200_000] {
            for &s in &[4u32, 8, 16] {
                assert_eq!(recommended_blocks_m5_max(n, s) % 32, 0);
            }
        }
    }
}

// ── Pass 1: per-block partials, GQA co-load ──────────────────────────────

#[kernel]
pub fn sdpa_decode_2pass_pass1<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    mut partial_o: Tensor<T>,
    mut partial_m: Tensor<f32>,
    mut partial_l: Tensor<f32>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] blocks: u32,
    #[constexpr] scale: f32,
) {
    let kv_head = tgid_x;
    let block_idx = tgid_y;
    let gqa_idx = simd_id;
    let lane = simd_lane;
    let q_head = kv_head * gqa_factor + gqa_idx;
    let d0 = lane * 4u32;

    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;

    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;

    for _t in range(block_idx, n_kv, blocks) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        // Pre-compute all 4 index VIDs BEFORE issuing the loads so the
        // vectorize pass sees 4 consecutive Load ops (no BinOp/Const
        // interleaved between them — the pass `break`s on any non-Load).
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let kv2 = kv_idx + 2u32;
        let kv3 = kv_idx + 3u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
    }

    // Pre-compute store indices AND pre-load the o0..3 accumulators
    // into fresh values so all four Stores land consecutively in IR
    // (vectorize requires consecutive Op::Store with no intervening
    // ops like the implicit Load(__ml_oN) the DSL would otherwise
    // emit just before each store).
    let out_block_off = (q_head * blocks + block_idx) * head_dim + d0;
    let po0 = out_block_off;
    let po1 = out_block_off + 1u32;
    let po2 = out_block_off + 2u32;
    let po3 = out_block_off + 3u32;
    // f32→T narrowing happens implicitly at the MSL Store (`dst[i] = val`),
    // so we don't add a Cast op here — that would introduce an extra
    // rounding step + break the 4-consecutive-Store window vectorize needs.
    let so0 = o0;
    let so1 = o1;
    let so2 = o2;
    let so3 = o3;
    store(partial_o[po0], so0);
    store(partial_o[po1], so1);
    store(partial_o[po2], so2);
    store(partial_o[po3], so3);
    if lane == 0u32 {
        let ml_off = q_head * blocks + block_idx;
        store(partial_m[ml_off], run_max);
        store(partial_l[ml_off], run_sum);
    }
}

// ── Pass 2: 32-sg × 32-lane merge, MLX-style ─────────────────────────────

#[kernel]
pub fn sdpa_decode_2pass_pass2<T>(
    partial_o: Tensor<T>,
    partial_m: Tensor<f32>,
    partial_l: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] blocks: u32,
) {
    let q_head = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let bn = 32u32;
    let block_chunks = blocks / bn;
    let d0 = lane * 4u32;

    let mbase = q_head * blocks;
    let obase = q_head * blocks * head_dim;
    let stride = bn + 1u32;

    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);

    let mut local_max = neg_infinity();
    for b in range(0u32, block_chunks, 1u32) {
        let m_val = load(partial_m[mbase + lane + b * bn]);
        local_max = select(m_val > local_max, m_val, local_max);
    }
    let max_score = simd_max(local_max);

    let mut local_sum = 0.0f32;
    for b in range(0u32, block_chunks, 1u32) {
        let m_val = load(partial_m[mbase + lane + b * bn]);
        let l_val = load(partial_l[mbase + lane + b * bn]);
        let factor = exp(m_val - max_score);
        local_sum = local_sum + factor * l_val;
    }
    let sum_exp = simd_sum(local_sum);

    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    for b in range(0u32, block_chunks, 1u32) {
        let m_val = load(partial_m[mbase + sg + b * bn]);
        let factor = exp(m_val - max_score);
        let po = obase + (sg + b * bn) * head_dim + d0;
        // Pre-compute 4 contiguous indices + issue all 4 loads
        // back-to-back so vectorize collapses them to one float4.
        let po0 = po;
        let po1 = po + 1u32;
        let po2 = po + 2u32;
        let po3 = po + 3u32;
        // Four consecutive Loads → vectorize collapses to one TxN
        // load. Cast each lane to f32 AFTER the loads so the Load run
        // stays uninterrupted; the running accumulators stay f32.
        let p0_raw = load(partial_o[po0]);
        let p1_raw = load(partial_o[po1]);
        let p2_raw = load(partial_o[po2]);
        let p3_raw = load(partial_o[po3]);
        let p0 = p0_raw.cast::<f32>();
        let p1 = p1_raw.cast::<f32>();
        let p2 = p2_raw.cast::<f32>();
        let p3 = p3_raw.cast::<f32>();
        o0 = o0 + factor * p0;
        o1 = o1 + factor * p1;
        o2 = o2 + factor * p2;
        o3 = o3 + factor * p3;
    }

    threadgroup_store("tg_out0", lane * stride + sg, o0);
    threadgroup_store("tg_out1", lane * stride + sg, o1);
    threadgroup_store("tg_out2", lane * stride + sg, o2);
    threadgroup_store("tg_out3", lane * stride + sg, o3);
    threadgroup_barrier();

    let r0 = threadgroup_load("tg_out0", sg * stride + lane);
    let r1 = threadgroup_load("tg_out1", sg * stride + lane);
    let r2 = threadgroup_load("tg_out2", sg * stride + lane);
    let r3 = threadgroup_load("tg_out3", sg * stride + lane);
    let red0 = simd_sum(r0);
    let red1 = simd_sum(r1);
    let red2 = simd_sum(r2);
    let red3 = simd_sum(r3);

    if lane == 0u32 {
        let inv_sum = select(sum_exp > 0.0f32, 1.0f32 / sum_exp, 0.0f32);
        let out_off = q_head * head_dim + sg * 4u32;
        store(out[out_off], (red0 * inv_sum).cast::<T>());
        store(out[out_off + 1u32], (red1 * inv_sum).cast::<T>());
        store(out[out_off + 2u32], (red2 * inv_sum).cast::<T>());
        store(out[out_off + 3u32], (red3 * inv_sum).cast::<T>());
    }
}

// Single registration covering the chained pass1+pass2 pair. Pass 1 is
// the `kernel_name`/`kernel_ir` on the spec; pass 2 is carried inside
// the `SdpaVector2Pass` dispatch variant. MLX reference is single-pass
// `sdpa_vector` at the same shape (MLX's `sdpa_vector_2pass` doesn't
// have a name-stable callable surface in our ref MSL); single-pass is
// the fair head-to-head for the long-N regime this kernel targets.
inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_decode_2pass",
        kernel_name: "sdpa_decode_2pass_pass1",
        kernel_ir: sdpa_decode_2pass_pass1::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: Some(include_str!(concat!(
            env!("OUT_DIR"),
            "/metal/scaled_dot_product_attention.metal"
        ))),
        mlx_pattern: Some("sdpa_vector_{tn}_128_128"),
        shapes: &[],
        dispatch: BenchDispatch::SdpaVector2Pass {
            head_dim: 128,
            n_kv: 4096,
            n_q_heads: 32,
            gqa_factor: 4,
            batch: 1,
            blocks: 32,
            pass2_kernel_name: "sdpa_decode_2pass_pass2",
            pass2_kernel_ir: sdpa_decode_2pass_pass2::kernel_ir_for,
        },
        kernel_mode: Some(KernelMode::Reduction),
    }
}
