//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prefill scaled-dot-product attention — `mt_sdpa_prefill`.
//!
//! Self-attention prefill with online softmax + causal masking. MLX
//! reference is `steel_attention_*_bq32_bk16_bd128_wm4_wn1` (Flash-
//! Attention 2 tile kernel).
//!
//! ## A.1 geometry: K/V tile reuse via threadgroup memory, BQ=4
//!
//! `BQ = 4` queries per TG, `BK = 16` K rows per outer block. tpg = 32
//! lanes (1 simdgroup); each lane owns `head_dim / 32 = 4` head-dim
//! elements. Per-lane Q + per-query softmax state held in registers.
//!
//! Outer loop iterates K-blocks of 16. For each block:
//!   1. Cooperatively load 16 × 128 K rows into `tg_ks` and 16 × 128 V
//!      rows into `tg_vs`.
//!   2. `threadgroup_barrier`.
//!   3. Inner loop: for each k_off in 0..16, for each q_i in 0..4 —
//!      compute the dot product `Q[q_i] · Ks[k_off]` (simd_sum), update
//!      per-query online-softmax (`run_max[q_i]`, `run_sum[q_i]`), and
//!      accumulate `O[q_i] += weight * Vs[k_off]`.
//!
//! ## Bandwidth lift over A.0
//!
//! A.0: per-TG K read 32 times (once per query), 512 TGs total → 16384× T·BD K-loads.
//! A.1: per-TG K read 1 time, 2048 TGs total (BQ=4 → 128 q_tiles × 32 heads) → 2048× T·BD = **8× reduction**.

use metaltile::kernel;

#[kernel]
pub fn mt_sdpa_prefill<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] q_len: u32,
    #[constexpr] k_len: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] n_kv_heads: u32,
    #[constexpr] scale: f32,
) {
    let q_tile = tgid_x;
    let q_head = tgid_y;
    let batch = tgid_z;
    let kv_head = q_head / gqa_factor;
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let d0_load = lane_in_tg;
    let d0 = lane * 4u32;
    let head_dim = 128u32;
    let bq = 32u32;
    let bq_sg = 8u32;
    let bk = 16u32;
    let q_len_off = k_len - q_len;
    let scale_log2 = scale * 1.4426950408889634f32;
    // Batched-prefill layout: q/out [batch, n_q_heads, q_len, head_dim],
    // k/v [batch, n_kv_heads, k_len, head_dim]. Single-batch B=1 collapses
    // to the original `(kv|q_head) * len * head_dim` form.
    let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
    let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
    threadgroup_alloc("tg_ks", 2048, T);
    threadgroup_alloc("tg_vs", 2048, T);
    let q_tile_first = q_tile * bq + sg * bq_sg;
    let q0_row = q_head_row_off + (q_tile_first + 0u32) * head_dim;
    let q1_row = q_head_row_off + (q_tile_first + 1u32) * head_dim;
    let q2_row = q_head_row_off + (q_tile_first + 2u32) * head_dim;
    let q3_row = q_head_row_off + (q_tile_first + 3u32) * head_dim;
    let q4_row = q_head_row_off + (q_tile_first + 4u32) * head_dim;
    let q5_row = q_head_row_off + (q_tile_first + 5u32) * head_dim;
    let q6_row = q_head_row_off + (q_tile_first + 6u32) * head_dim;
    let q7_row = q_head_row_off + (q_tile_first + 7u32) * head_dim;
    let q0_0 = load(q[q0_row + d0]).cast::<f32>() * scale_log2;
    let q0_1 = load(q[q0_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q0_2 = load(q[q0_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q0_3 = load(q[q0_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q1_0 = load(q[q1_row + d0]).cast::<f32>() * scale_log2;
    let q1_1 = load(q[q1_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q1_2 = load(q[q1_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q1_3 = load(q[q1_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q2_0 = load(q[q2_row + d0]).cast::<f32>() * scale_log2;
    let q2_1 = load(q[q2_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q2_2 = load(q[q2_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q2_3 = load(q[q2_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q3_0 = load(q[q3_row + d0]).cast::<f32>() * scale_log2;
    let q3_1 = load(q[q3_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q3_2 = load(q[q3_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q3_3 = load(q[q3_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q4_0 = load(q[q4_row + d0]).cast::<f32>() * scale_log2;
    let q4_1 = load(q[q4_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q4_2 = load(q[q4_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q4_3 = load(q[q4_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q5_0 = load(q[q5_row + d0]).cast::<f32>() * scale_log2;
    let q5_1 = load(q[q5_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q5_2 = load(q[q5_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q5_3 = load(q[q5_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q6_0 = load(q[q6_row + d0]).cast::<f32>() * scale_log2;
    let q6_1 = load(q[q6_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q6_2 = load(q[q6_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q6_3 = load(q[q6_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q7_0 = load(q[q7_row + d0]).cast::<f32>() * scale_log2;
    let q7_1 = load(q[q7_row + d0 + 1u32]).cast::<f32>() * scale_log2;
    let q7_2 = load(q[q7_row + d0 + 2u32]).cast::<f32>() * scale_log2;
    let q7_3 = load(q[q7_row + d0 + 3u32]).cast::<f32>() * scale_log2;
    let q0_abs = q_tile_first + 0u32 + q_len_off;
    let q1_abs = q_tile_first + 1u32 + q_len_off;
    let q2_abs = q_tile_first + 2u32 + q_len_off;
    let q3_abs = q_tile_first + 3u32 + q_len_off;
    let q4_abs = q_tile_first + 4u32 + q_len_off;
    let q5_abs = q_tile_first + 5u32 + q_len_off;
    let q6_abs = q_tile_first + 6u32 + q_len_off;
    let q7_abs = q_tile_first + 7u32 + q_len_off;
    let mut m0 = neg_infinity();
    let mut s0 = 0.0f32;
    let mut o00 = 0.0f32;
    let mut o01 = 0.0f32;
    let mut o02 = 0.0f32;
    let mut o03 = 0.0f32;
    let mut m1 = neg_infinity();
    let mut s1 = 0.0f32;
    let mut o10 = 0.0f32;
    let mut o11 = 0.0f32;
    let mut o12 = 0.0f32;
    let mut o13 = 0.0f32;
    let mut m2 = neg_infinity();
    let mut s2 = 0.0f32;
    let mut o20 = 0.0f32;
    let mut o21 = 0.0f32;
    let mut o22 = 0.0f32;
    let mut o23 = 0.0f32;
    let mut m3 = neg_infinity();
    let mut s3 = 0.0f32;
    let mut o30 = 0.0f32;
    let mut o31 = 0.0f32;
    let mut o32 = 0.0f32;
    let mut o33 = 0.0f32;
    let mut m4 = neg_infinity();
    let mut s4 = 0.0f32;
    let mut o40 = 0.0f32;
    let mut o41 = 0.0f32;
    let mut o42 = 0.0f32;
    let mut o43 = 0.0f32;
    let mut m5 = neg_infinity();
    let mut s5 = 0.0f32;
    let mut o50 = 0.0f32;
    let mut o51 = 0.0f32;
    let mut o52 = 0.0f32;
    let mut o53 = 0.0f32;
    let mut m6 = neg_infinity();
    let mut s6 = 0.0f32;
    let mut o60 = 0.0f32;
    let mut o61 = 0.0f32;
    let mut o62 = 0.0f32;
    let mut o63 = 0.0f32;
    let mut m7 = neg_infinity();
    let mut s7 = 0.0f32;
    let mut o70 = 0.0f32;
    let mut o71 = 0.0f32;
    let mut o72 = 0.0f32;
    let mut o73 = 0.0f32;
    // Causal trim: bound K-block loop at the LAST query of the ENTIRE TG
    // (across all simdgroups), not per-SG. All 4 SGs must execute the same
    // count of `threadgroup_barrier`s or the TG deadlocks.
    let q_tile_last_abs = q_tile * bq + (bq - 1u32) + q_len_off;
    let kb_lim = (q_tile_last_abs / bk) + 1u32;
    let sg_kb_lim = (q7_abs / bk) + 1u32;
    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * bk;
        // Coop load: 128 lanes × bk × 1 element = 128 × 16 = full K-block (BK*BD = 2048).
        // 1/4 the per-lane load work vs the bq=8 / tpg=32 single-SG path.
        for kr in range(0u32, bk, 1u32) {
            let kv_off = kv_row_base + (kb_off + kr) * head_dim + d0_load;
            let kr_off = kr * head_dim;
            // Store K/V in native T (no f32 upcast) so f16/bf16 paths halve
            // TG mem footprint, freeing occupancy (agent #3: 4 KB vs 8 KB
            // each enables more concurrent TGs / SM).
            threadgroup_store("tg_ks", kr_off + d0_load, load(k[kv_off]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + d0_load, load(v[kv_off]).cast::<T>());
        }
        threadgroup_barrier();
        if kb < sg_kb_lim {
            for k_off in range(0u32, bk, 1u32) {
                let k_abs = kb_off + k_off;
                let kr_off = k_off * head_dim;
                let k0 = threadgroup_load("tg_ks", kr_off + d0).cast::<f32>();
                let k1 = threadgroup_load("tg_ks", kr_off + d0 + 1u32).cast::<f32>();
                let k2 = threadgroup_load("tg_ks", kr_off + d0 + 2u32).cast::<f32>();
                let k3 = threadgroup_load("tg_ks", kr_off + d0 + 3u32).cast::<f32>();
                let v0 = threadgroup_load("tg_vs", kr_off + d0).cast::<f32>();
                let v1 = threadgroup_load("tg_vs", kr_off + d0 + 1u32).cast::<f32>();
                let v2 = threadgroup_load("tg_vs", kr_off + d0 + 2u32).cast::<f32>();
                let v3 = threadgroup_load("tg_vs", kr_off + d0 + 3u32).cast::<f32>();
                let r0 = simd_sum(q0_0 * k0 + q0_1 * k1 + q0_2 * k2 + q0_3 * k3);
                let r1 = simd_sum(q1_0 * k0 + q1_1 * k1 + q1_2 * k2 + q1_3 * k3);
                let r2 = simd_sum(q2_0 * k0 + q2_1 * k1 + q2_2 * k2 + q2_3 * k3);
                let r3 = simd_sum(q3_0 * k0 + q3_1 * k1 + q3_2 * k2 + q3_3 * k3);
                let r4 = simd_sum(q4_0 * k0 + q4_1 * k1 + q4_2 * k2 + q4_3 * k3);
                let r5 = simd_sum(q5_0 * k0 + q5_1 * k1 + q5_2 * k2 + q5_3 * k3);
                let r6 = simd_sum(q6_0 * k0 + q6_1 * k1 + q6_2 * k2 + q6_3 * k3);
                let r7 = simd_sum(q7_0 * k0 + q7_1 * k1 + q7_2 * k2 + q7_3 * k3);
                let mk0 = select(k_abs > q0_abs, neg_infinity(), r0);
                let mk1 = select(k_abs > q1_abs, neg_infinity(), r1);
                let mk2 = select(k_abs > q2_abs, neg_infinity(), r2);
                let mk3 = select(k_abs > q3_abs, neg_infinity(), r3);
                let mk4 = select(k_abs > q4_abs, neg_infinity(), r4);
                let mk5 = select(k_abs > q5_abs, neg_infinity(), r5);
                let mk6 = select(k_abs > q6_abs, neg_infinity(), r6);
                let mk7 = select(k_abs > q7_abs, neg_infinity(), r7);
                let nm0 = select(mk0 > m0, mk0, m0);
                let f0 = exp2(m0 - nm0);
                let w0 = exp2(mk0 - nm0);
                s0 = s0 * f0 + w0;
                m0 = nm0;
                o00 = o00 * f0 + w0 * v0;
                o01 = o01 * f0 + w0 * v1;
                o02 = o02 * f0 + w0 * v2;
                o03 = o03 * f0 + w0 * v3;
                let nm1 = select(mk1 > m1, mk1, m1);
                let f1 = exp2(m1 - nm1);
                let w1 = exp2(mk1 - nm1);
                s1 = s1 * f1 + w1;
                m1 = nm1;
                o10 = o10 * f1 + w1 * v0;
                o11 = o11 * f1 + w1 * v1;
                o12 = o12 * f1 + w1 * v2;
                o13 = o13 * f1 + w1 * v3;
                let nm2 = select(mk2 > m2, mk2, m2);
                let f2 = exp2(m2 - nm2);
                let w2 = exp2(mk2 - nm2);
                s2 = s2 * f2 + w2;
                m2 = nm2;
                o20 = o20 * f2 + w2 * v0;
                o21 = o21 * f2 + w2 * v1;
                o22 = o22 * f2 + w2 * v2;
                o23 = o23 * f2 + w2 * v3;
                let nm3 = select(mk3 > m3, mk3, m3);
                let f3 = exp2(m3 - nm3);
                let w3 = exp2(mk3 - nm3);
                s3 = s3 * f3 + w3;
                m3 = nm3;
                o30 = o30 * f3 + w3 * v0;
                o31 = o31 * f3 + w3 * v1;
                o32 = o32 * f3 + w3 * v2;
                o33 = o33 * f3 + w3 * v3;
                let nm4 = select(mk4 > m4, mk4, m4);
                let f4 = exp2(m4 - nm4);
                let w4 = exp2(mk4 - nm4);
                s4 = s4 * f4 + w4;
                m4 = nm4;
                o40 = o40 * f4 + w4 * v0;
                o41 = o41 * f4 + w4 * v1;
                o42 = o42 * f4 + w4 * v2;
                o43 = o43 * f4 + w4 * v3;
                let nm5 = select(mk5 > m5, mk5, m5);
                let f5 = exp2(m5 - nm5);
                let w5 = exp2(mk5 - nm5);
                s5 = s5 * f5 + w5;
                m5 = nm5;
                o50 = o50 * f5 + w5 * v0;
                o51 = o51 * f5 + w5 * v1;
                o52 = o52 * f5 + w5 * v2;
                o53 = o53 * f5 + w5 * v3;
                let nm6 = select(mk6 > m6, mk6, m6);
                let f6 = exp2(m6 - nm6);
                let w6 = exp2(mk6 - nm6);
                s6 = s6 * f6 + w6;
                m6 = nm6;
                o60 = o60 * f6 + w6 * v0;
                o61 = o61 * f6 + w6 * v1;
                o62 = o62 * f6 + w6 * v2;
                o63 = o63 * f6 + w6 * v3;
                let nm7 = select(mk7 > m7, mk7, m7);
                let f7 = exp2(m7 - nm7);
                let w7 = exp2(mk7 - nm7);
                s7 = s7 * f7 + w7;
                m7 = nm7;
                o70 = o70 * f7 + w7 * v0;
                o71 = o71 * f7 + w7 * v1;
                o72 = o72 * f7 + w7 * v2;
                o73 = o73 * f7 + w7 * v3;
            }
        }
        threadgroup_barrier();
    }
    let is0 = select(s0 > 0.0f32, 1.0f32 / s0, 0.0f32);
    let is1 = select(s1 > 0.0f32, 1.0f32 / s1, 0.0f32);
    let is2 = select(s2 > 0.0f32, 1.0f32 / s2, 0.0f32);
    let is3 = select(s3 > 0.0f32, 1.0f32 / s3, 0.0f32);
    store(out[q0_row + d0], o00 * is0);
    store(out[q0_row + d0 + 1u32], o01 * is0);
    store(out[q0_row + d0 + 2u32], o02 * is0);
    store(out[q0_row + d0 + 3u32], o03 * is0);
    store(out[q1_row + d0], o10 * is1);
    store(out[q1_row + d0 + 1u32], o11 * is1);
    store(out[q1_row + d0 + 2u32], o12 * is1);
    store(out[q1_row + d0 + 3u32], o13 * is1);
    store(out[q2_row + d0], o20 * is2);
    store(out[q2_row + d0 + 1u32], o21 * is2);
    store(out[q2_row + d0 + 2u32], o22 * is2);
    store(out[q2_row + d0 + 3u32], o23 * is2);
    store(out[q3_row + d0], o30 * is3);
    store(out[q3_row + d0 + 1u32], o31 * is3);
    store(out[q3_row + d0 + 2u32], o32 * is3);
    store(out[q3_row + d0 + 3u32], o33 * is3);
    let is4 = select(s4 > 0.0f32, 1.0f32 / s4, 0.0f32);
    let is5 = select(s5 > 0.0f32, 1.0f32 / s5, 0.0f32);
    let is6 = select(s6 > 0.0f32, 1.0f32 / s6, 0.0f32);
    let is7 = select(s7 > 0.0f32, 1.0f32 / s7, 0.0f32);
    store(out[q4_row + d0], o40 * is4);
    store(out[q4_row + d0 + 1u32], o41 * is4);
    store(out[q4_row + d0 + 2u32], o42 * is4);
    store(out[q4_row + d0 + 3u32], o43 * is4);
    store(out[q5_row + d0], o50 * is5);
    store(out[q5_row + d0 + 1u32], o51 * is5);
    store(out[q5_row + d0 + 2u32], o52 * is5);
    store(out[q5_row + d0 + 3u32], o53 * is5);
    store(out[q6_row + d0], o60 * is6);
    store(out[q6_row + d0 + 1u32], o61 * is6);
    store(out[q6_row + d0 + 2u32], o62 * is6);
    store(out[q6_row + d0 + 3u32], o63 * is6);
    store(out[q7_row + d0], o70 * is7);
    store(out[q7_row + d0 + 1u32], o71 * is7);
    store(out[q7_row + d0 + 2u32], o72 * is7);
    store(out[q7_row + d0 + 3u32], o73 * is7);
}

/// New-syntax benchmarks for the SDPA-prefill family (vs MLX's
/// `steel_attention_*_bq32_bk16_bd128_wm4_wn1` Flash-Attention-2 tile
/// kernel). Covers the scalar flash variant and both simdgroup-matrix
/// (MMA) variants. All three share the same dispatch contract:
///
/// - **SimdGroup2D mode** — the kernels read `tgid_x`/`tgid_y`/`tgid_z`.
/// - **Grid = (q_len / BQ=32, n_q_heads, batch)** threadGROUP counts,
///   **TPG = 128** (4 simdgroups). `grid_3d` takes group counts, so the
///   first three args are the group counts directly.
/// - `head_dim` is hardcoded 128 in the kernel body; `scale = 1/sqrt(128)`.
///
/// Shape mirrors the legacy `bench(...)`: h=128, n_heads=32, gqa_factor=4
/// (→ 8 KV heads), batch=1, q_len=k_len=512.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_sdpa_prefill;
    use crate::mlx::steel::attn::{
        steel_attention_mma::mt_sdpa_prefill_mma,
        steel_attention_mma_bf16::mt_sdpa_prefill_mma_bf16,
    };

    // SDPA prefill geometry shared by all three variants.
    const HEAD_DIM: usize = 128;
    const N_Q_HEADS: usize = 32;
    const GQA_FACTOR: usize = 4;
    const N_KV_HEADS: usize = N_Q_HEADS / GQA_FACTOR; // 8
    const BATCH: usize = 1;
    const Q_LEN: usize = 512;
    const K_LEN: usize = 512;
    const BQ: u32 = 32;
    const TPG: u32 = 128;

    fn sdpa_b(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let q_elems = BATCH * N_Q_HEADS * Q_LEN * HEAD_DIM;
        let kv_elems = BATCH * N_KV_HEADS * K_LEN * HEAD_DIM;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        BenchSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .buffer(BenchBuffer::random("q", q_elems, dt))
            .buffer(BenchBuffer::random("k", kv_elems, dt))
            .buffer(BenchBuffer::random("v", kv_elems, dt))
            .buffer(BenchBuffer::zeros("out", q_elems, dt).output())
            .constexpr("q_len", Q_LEN as u32)
            .constexpr("k_len", K_LEN as u32)
            .constexpr("gqa_factor", GQA_FACTOR as u32)
            .constexpr("n_q_heads", N_Q_HEADS as u32)
            .constexpr("n_kv_heads", N_KV_HEADS as u32)
            .constexpr("scale", scale)
            // grid_3d takes threadGROUP counts: (q_len / BQ, n_q_heads, batch).
            .grid_3d(Q_LEN as u32 / BQ, N_Q_HEADS as u32, BATCH as u32, [TPG, 1, 1])
            // Q/O read+written once each; K/V read once per q_tile group.
            .bytes_moved(((2 * q_elems + 2 * kv_elems) * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/sdpa/sdpa_prefill", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill(dt: DType) -> BenchSetup {
        sdpa_b(mt_sdpa_prefill::kernel_ir_for(dt), dt)
    }

    #[bench(name = "mlx/sdpa/sdpa_prefill_mma", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill_mma(dt: DType) -> BenchSetup {
        sdpa_b(mt_sdpa_prefill_mma::kernel_ir_for(dt), dt)
    }

    // bf16-emulated MMA variant — the M2-family bf16 routing target. Only
    // meaningful at bf16 (single-Q dd-loop, bf16 MMA frags).
    // dtypes f32/f16/bf16 to match the legacy emission (the bf16-tile kernel
    // is monomorphized over all three input dtypes, same as the legacy spec).
    #[bench(name = "mlx/sdpa/sdpa_prefill_mma_bf16", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill_mma_bf16(dt: DType) -> BenchSetup {
        sdpa_b(mt_sdpa_prefill_mma_bf16::kernel_ir_for(dt), dt)
    }
}

/// New-syntax correctness tests for the SDPA-prefill family — ports the
/// causal-SDPA oracle from `tests/steel_attention_gpu_correctness.rs`.
/// All three variants (`mt_sdpa_prefill` scalar flash, `_mma`, and
/// `_mma_bf16`) share the same dispatch contract and the same reference:
///   `O = softmax(Q·Kᵀ · scale)·V` per Q head, with causal masking
///   (`k_abs ≤ q_abs`, `q_abs = (k_len - q_len) + qi`) and GQA via
///   `kv_head = q_head / gqa_factor`.
///
/// Minimal valid shape: `n_q_heads = n_kv_heads = 4`, `q_len = k_len =
/// 128` (= 4·BQ=32 q-tiles), `head_dim = 128` (hardcoded in the kernel),
/// `gqa_factor = 1`, `scale = 1/sqrt(128)`. `SimdGroup2D` dispatch —
/// grid is threadgroup counts `(q_len/BQ, n_q_heads, batch)` with `tpg =
/// [128, 1, 1]` (4 simdgroups) — copied from the matching `#[bench]`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::mt_sdpa_prefill;
    use crate::{
        mlx::steel::attn::{
            steel_attention_mma::mt_sdpa_prefill_mma,
            steel_attention_mma_bf16::mt_sdpa_prefill_mma_bf16,
        },
        utils::{pack_f32, unpack_f32},
    };

    // Shared shape (single batch). head_dim is hardcoded 128 in the kernel.
    const N_Q_HEADS: usize = 4;
    const N_KV_HEADS: usize = 4;
    const Q_LEN: usize = 128;
    const K_LEN: usize = 128;
    const HEAD_DIM: usize = 128;
    const BQ: u32 = 32;

    /// Deterministic ramp — mirrors the legacy test's `ramp(n, modulus, offset)`.
    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// Naive causal SDPA, single batch. Q/K/V laid out
    /// `[n_heads, q_len, head_dim]` contiguous.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_prefill_causal(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        q_len: usize,
        k_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let q_len_off = k_len - q_len;
        let mut out = vec![0.0f32; n_q_heads * q_len * head_dim];
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = qh * q_len * head_dim;
            let kv_off = kvh * k_len * head_dim;
            for qi in 0..q_len {
                let causal_lim = q_len_off + qi + 1;
                let mut scores = vec![f32::NEG_INFINITY; k_len];
                for (j, s) in scores.iter_mut().enumerate().take(causal_lim) {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + qi * head_dim + d] * k[kv_off + j * head_dim + d];
                    }
                    *s = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores
                    .iter()
                    .map(|&s| if s.is_finite() { (s - m).exp() } else { 0.0 })
                    .collect();
                let total: f32 = e.iter().sum();
                let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, &ej) in e.iter().enumerate() {
                        acc += ej * inv * v[kv_off + j * head_dim + d];
                    }
                    out[q_off + qi * head_dim + d] = acc;
                }
            }
        }
        out
    }

    /// Build a causal-SDPA prefill correctness setup for one variant.
    fn sdpa_setup(kernel: Kernel, dt: DType) -> TestSetup {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        // Dtype-round inputs so the CPU oracle sees the same load-cast
        // quantization the kernel does.
        let q = unpack_f32(&pack_f32(&ramp(N_Q_HEADS * Q_LEN * HEAD_DIM, 17, 8.0), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(N_KV_HEADS * K_LEN * HEAD_DIM, 13, 6.0), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(N_KV_HEADS * K_LEN * HEAD_DIM, 11, 5.0), dt), dt);
        let expected = naive_sdpa_prefill_causal(
            &q, &k, &v, N_Q_HEADS, N_KV_HEADS, Q_LEN, K_LEN, HEAD_DIM, scale,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::SimdGroup2D)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", N_Q_HEADS * Q_LEN * HEAD_DIM, dt))
            .constexpr("q_len", Q_LEN as u32)
            .constexpr("k_len", K_LEN as u32)
            .constexpr("gqa_factor", (N_Q_HEADS / N_KV_HEADS) as u32)
            .constexpr("n_q_heads", N_Q_HEADS as u32)
            .constexpr("n_kv_heads", N_KV_HEADS as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            // grid_3d takes threadGROUP counts: (q_len / BQ, n_q_heads, batch).
            .grid_3d(Q_LEN as u32 / BQ, N_Q_HEADS as u32, 1, [128, 1, 1])
    }

    // tol per dtype: f32 2e-2 (matches the legacy test), f16 5e-2, bf16
    // 2e-1 (online-softmax + matmul drift at head_dim=128).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-2, 5e-2, 2e-1])]
    fn test_sdpa_prefill(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_prefill::kernel_ir_for(dt), dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-2, 5e-2, 2e-1])]
    fn test_sdpa_prefill_mma(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_prefill_mma::kernel_ir_for(dt), dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-2, 5e-2, 2e-1])]
    fn test_sdpa_prefill_mma_bf16(dt: DType) -> TestSetup {
        sdpa_setup(mt_sdpa_prefill_mma_bf16::kernel_ir_for(dt), dt)
    }
}
