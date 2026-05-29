//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prefill SDPA via `simdgroup_multiply_accumulate` (MMA) — `mt_sdpa_prefill_mma`.
//!
//! Mirrors `mt_sdpa_prefill`'s outer geometry (BQ=32, BK=16, BD=128, WM=4,
//! WN=1, tpg=128 = 4 SGs sharing one K/V TG cache) but replaces the per-SG
//! scalar simd_sum dot product with Apple's 8×8 simdgroup matrix MMA
//! fragments. Per K-block per SG:
//!   1. Preload 16 Q fragments (8×8) — persistent across K-blocks.
//!   2. Coop load K + V tiles into TG memory (same as scalar path).
//!   3. Q·K^T → S via 16 d_frags × 2 k_chunks = 32 matmuls per SG.
//!   4. Online softmax: dump S to per-SG TG scratch, each lane reads its
//!      `fm` row, computes row max/sum, scales own O frag elements by m_diff,
//!      writes new P values back into the s_frag elements.
//!   5. P·V → O via 16 d_frags × 2 k_chunks = 32 matmuls per SG.
//!
//! Apple frag lane layout (32 lanes per SG, 8×8 fragments):
//!   `qid = lane/4, fm = (qid & 4) + (lane/2 % 4),
//!    fn0 = (qid & 2)*2 + (lane%2)*2, fn1 = fn0 + 1`
//! Each lane owns 2 elements per frag at positions (fm, fn0) and (fm, fn1).

use metaltile::kernel;

#[kernel(
    bench(
        op="sdpa",
        subop="sdpa_prefill_mma",
        class=SdpaPrefill,
        h=128,
        n_heads=32,
        gqa_factor=4,
        batch=1,
        q_len=512,
        k_len=512,
        bq=32,
        bk=16,
        wm=4,
        wn=1,
        tpg=128,
        tol=2e-2,
        metal_file="steel/attn/steel_attention.metal",
        mlx="steel_attention_float32_bq32_bk16_bd128_wm4_wn1_maskfloat32",
    ),
    // K=8 speculative-decode verify: BQ=32 prefill tile, 75% wasted Q rows.
    bench(
        op="sdpa",
        subop="sdpa_decode_batched_q8",
        class=SdpaBatchedDecode,
        h=128,
        n_kv=4096,
        n_heads=32,
        gqa_factor=4,
        batch_q=8,
        variant=PrefillTile,
        bq=32,
        bk=16,
        wm=4,
        wn=1,
        tpg=128,
        tol=2e-2,
        kernel_mode=SimdGroup2D,
    ),
    // K=16 speculative-decode verify: BQ=32 prefill tile, 50% wasted Q rows.
    bench(
        op="sdpa",
        subop="sdpa_decode_batched_q16",
        class=SdpaBatchedDecode,
        h=128,
        n_kv=4096,
        n_heads=32,
        gqa_factor=4,
        batch_q=16,
        variant=PrefillTile,
        bq=32,
        bk=16,
        wm=4,
        wn=1,
        tpg=128,
        tol=2e-2,
        kernel_mode=SimdGroup2D,
    ),
)]
pub fn mt_sdpa_prefill_mma<T>(
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
    // ── 8×8 frag lane mapping (Apple steel_gemm layout) ──
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    let head_dim = 128u32;
    let bq = 32u32;
    let bq_sg = 8u32;
    let bk = 16u32;
    let q_len_off = k_len - q_len;
    let scale_log2 = scale * 1.4426950408889634f32;
    // Batched-prefill layout (B > 1):
    //   q, out : [batch, n_q_heads,  q_len, head_dim]   row-major
    //   k, v   : [batch, n_kv_heads, k_len, head_dim]   row-major
    // Slab offsets fold `batch` into the global base so Q/K/V/O all
    // pick up the right per-(batch, head) slice. Single-batch B=1
    // collapses to the original `(kv|q_head) * len * head_dim` form.
    let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
    let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
    let q_tile_first = q_tile * bq + sg * bq_sg;
    let q_row_base = q_head_row_off + q_tile_first * head_dim;
    // kv_ld = head_dim + 8 = 136 bank-skew pad on the column-major K^T
    // reads (`tg_ks[fn * kv_ld + fm]` strides by kv_ld across lanes).
    // Median-of-5 sweep (2026-05-19, see selector docstring) confirmed
    // +8 wins on the f16 4-byte-load bank pattern. Real wins: M2 f16
    // 92% → 96% (+4pt, larger than noise), M2 f32 124% → 127% (+3pt),
    // M5 f32 114% → 116% (+2pt). M5 f16 / bf16 are wash (within 0.9-3.7%
    // noise). The mma_bf16 sibling keeps +4 (132) — 8-byte bf16 loads
    // hit a different bank pattern than 4-byte f16 loads, and no
    // kv_ld=136 win on bf16 surfaced larger than noise.
    let kv_ld = 136u32;
    threadgroup_alloc("tg_ks", 2176, T);
    threadgroup_alloc("tg_vs", 2176, T);
    // No softmax scratch — row reduction via simd_shuffle_xor keeps S in regs.
    // ── Preload 16 Q frags (one per d_frag of head_dim=128), pre-scaled ──
    let q_f0 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f0, 0, load(q[q_row_base + fm * head_dim + fn0]).cast::<T>());
    simdgroup_elem_store(q_f0, 1, load(q[q_row_base + fm * head_dim + fn1]).cast::<T>());
    let q_f1 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1, 0, load(q[q_row_base + fm * head_dim + 8u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1, 1, load(q[q_row_base + fm * head_dim + 8u32 + fn1]).cast::<T>());
    let q_f2 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f2, 0, load(q[q_row_base + fm * head_dim + 16u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f2, 1, load(q[q_row_base + fm * head_dim + 16u32 + fn1]).cast::<T>());
    let q_f3 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f3, 0, load(q[q_row_base + fm * head_dim + 24u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f3, 1, load(q[q_row_base + fm * head_dim + 24u32 + fn1]).cast::<T>());
    let q_f4 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f4, 0, load(q[q_row_base + fm * head_dim + 32u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f4, 1, load(q[q_row_base + fm * head_dim + 32u32 + fn1]).cast::<T>());
    let q_f5 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f5, 0, load(q[q_row_base + fm * head_dim + 40u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f5, 1, load(q[q_row_base + fm * head_dim + 40u32 + fn1]).cast::<T>());
    let q_f6 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f6, 0, load(q[q_row_base + fm * head_dim + 48u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f6, 1, load(q[q_row_base + fm * head_dim + 48u32 + fn1]).cast::<T>());
    let q_f7 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f7, 0, load(q[q_row_base + fm * head_dim + 56u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f7, 1, load(q[q_row_base + fm * head_dim + 56u32 + fn1]).cast::<T>());
    let q_f8 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f8, 0, load(q[q_row_base + fm * head_dim + 64u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f8, 1, load(q[q_row_base + fm * head_dim + 64u32 + fn1]).cast::<T>());
    let q_f9 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f9, 0, load(q[q_row_base + fm * head_dim + 72u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f9, 1, load(q[q_row_base + fm * head_dim + 72u32 + fn1]).cast::<T>());
    let q_fa = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fa, 0, load(q[q_row_base + fm * head_dim + 80u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fa, 1, load(q[q_row_base + fm * head_dim + 80u32 + fn1]).cast::<T>());
    let q_fb = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fb, 0, load(q[q_row_base + fm * head_dim + 88u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fb, 1, load(q[q_row_base + fm * head_dim + 88u32 + fn1]).cast::<T>());
    let q_fc = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fc, 0, load(q[q_row_base + fm * head_dim + 96u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fc, 1, load(q[q_row_base + fm * head_dim + 96u32 + fn1]).cast::<T>());
    let q_fd = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fd, 0, load(q[q_row_base + fm * head_dim + 104u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fd, 1, load(q[q_row_base + fm * head_dim + 104u32 + fn1]).cast::<T>());
    let q_fe = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fe, 0, load(q[q_row_base + fm * head_dim + 112u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fe, 1, load(q[q_row_base + fm * head_dim + 112u32 + fn1]).cast::<T>());
    let q_ff = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_ff, 0, load(q[q_row_base + fm * head_dim + 120u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_ff, 1, load(q[q_row_base + fm * head_dim + 120u32 + fn1]).cast::<T>());
    // ── Init 16 O frags to zero ──
    let o_f0 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f0, 0, 0.0f32);
    simdgroup_elem_store(o_f0, 1, 0.0f32);
    let o_f1 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1, 0, 0.0f32);
    simdgroup_elem_store(o_f1, 1, 0.0f32);
    let o_f2 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f2, 0, 0.0f32);
    simdgroup_elem_store(o_f2, 1, 0.0f32);
    let o_f3 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f3, 0, 0.0f32);
    simdgroup_elem_store(o_f3, 1, 0.0f32);
    let o_f4 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f4, 0, 0.0f32);
    simdgroup_elem_store(o_f4, 1, 0.0f32);
    let o_f5 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f5, 0, 0.0f32);
    simdgroup_elem_store(o_f5, 1, 0.0f32);
    let o_f6 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f6, 0, 0.0f32);
    simdgroup_elem_store(o_f6, 1, 0.0f32);
    let o_f7 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f7, 0, 0.0f32);
    simdgroup_elem_store(o_f7, 1, 0.0f32);
    let o_f8 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f8, 0, 0.0f32);
    simdgroup_elem_store(o_f8, 1, 0.0f32);
    let o_f9 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f9, 0, 0.0f32);
    simdgroup_elem_store(o_f9, 1, 0.0f32);
    let o_fa = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fa, 0, 0.0f32);
    simdgroup_elem_store(o_fa, 1, 0.0f32);
    let o_fb = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fb, 0, 0.0f32);
    simdgroup_elem_store(o_fb, 1, 0.0f32);
    let o_fc = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fc, 0, 0.0f32);
    simdgroup_elem_store(o_fc, 1, 0.0f32);
    let o_fd = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fd, 0, 0.0f32);
    simdgroup_elem_store(o_fd, 1, 0.0f32);
    let o_fe = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fe, 0, 0.0f32);
    simdgroup_elem_store(o_fe, 1, 0.0f32);
    let o_ff = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_ff, 0, 0.0f32);
    simdgroup_elem_store(o_ff, 1, 0.0f32);
    // S/P frags (2: cols 0..7 and 8..15) reused per K-block.
    // Separate S (Q·K^T accumulator) and P (softmax output → P·V input) frags.
    // Aliasing the same matrix as both matmul output and next matmul input
    // appears to confuse Apple's register tracker — the prior MMA POC kept
    // them as one and produced silent zero output (see commit history).
    let s_f0 = simdgroup_alloc::<f32, 8, 8>();
    let s_f1 = simdgroup_alloc::<f32, 8, 8>();
    let p_f0 = simdgroup_alloc::<T, 8, 8>();
    let p_f1 = simdgroup_alloc::<T, 8, 8>();
    // K^T and V frags reused per d_frag.
    let kt_a = simdgroup_alloc::<T, 8, 8>();
    let kt_b = simdgroup_alloc::<T, 8, 8>();
    let v_a = simdgroup_alloc::<T, 8, 8>();
    let v_b = simdgroup_alloc::<T, 8, 8>();
    // Per-lane row state (4 lanes share the same fm → redundantly hold
    // identical m_row / s_row, which is fine).
    let mut m_row = neg_infinity();
    let mut s_row = 0.0f32;
    let q_abs = q_tile_first + fm + q_len_off;
    // TG-wide kb_lim so all 4 SGs execute the same barrier count.
    let q_tile_last_abs = q_tile * bq + (bq - 1u32) + q_len_off;
    let kb_lim = (q_tile_last_abs / bk) + 1u32;
    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * bk;
        // ── Coop K/V load (combined): 128 lanes × bk × 1 elem = full K-block.
        for kr in range(0u32, bk, 1u32) {
            let kv_off = kv_row_base + (kb_off + kr) * head_dim + lane_in_tg;
            let kr_off = kr * kv_ld;
            threadgroup_store("tg_ks", kr_off + lane_in_tg, load(k[kv_off]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + lane_in_tg, load(v[kv_off]).cast::<T>());
        }
        threadgroup_barrier();
        // ── S = Q · K^T (32 matmuls per SG: 16 d_frags × 2 k_chunks) ──
        simdgroup_elem_store(s_f0, 0, 0.0f32);
        simdgroup_elem_store(s_f0, 1, 0.0f32);
        simdgroup_elem_store(s_f1, 0, 0.0f32);
        simdgroup_elem_store(s_f1, 1, 0.0f32);
        // K^T frag elem layout: elem[i] = K[k_base + fn_i, d_base + fm]
        //   = tg_ks[(k_chunk_base + fn_i) * head_dim + d_base + fm]
        // Unrolled 16 d_frags × 2 k_chunks below.
        // d=0
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 0u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 0u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f0, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 0u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 0u32 + fm));
        simdgroup_matmul(q_f0, kt_b, s_f1);
        // d=1
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 8u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f1, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 8u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 8u32 + fm));
        simdgroup_matmul(q_f1, kt_b, s_f1);
        // d=2
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 16u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f2, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 16u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 16u32 + fm));
        simdgroup_matmul(q_f2, kt_b, s_f1);
        // d=3
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 24u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f3, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 24u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 24u32 + fm));
        simdgroup_matmul(q_f3, kt_b, s_f1);
        // d=4
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 32u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 32u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f4, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 32u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 32u32 + fm));
        simdgroup_matmul(q_f4, kt_b, s_f1);
        // d=5
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 40u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 40u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f5, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 40u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 40u32 + fm));
        simdgroup_matmul(q_f5, kt_b, s_f1);
        // d=6
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 48u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 48u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f6, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 48u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 48u32 + fm));
        simdgroup_matmul(q_f6, kt_b, s_f1);
        // d=7
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 56u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 56u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f7, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 56u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 56u32 + fm));
        simdgroup_matmul(q_f7, kt_b, s_f1);
        // d=8
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 64u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 64u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f8, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 64u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 64u32 + fm));
        simdgroup_matmul(q_f8, kt_b, s_f1);
        // d=9
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 72u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 72u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_f9, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 72u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 72u32 + fm));
        simdgroup_matmul(q_f9, kt_b, s_f1);
        // d=a
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 80u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 80u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_fa, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 80u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 80u32 + fm));
        simdgroup_matmul(q_fa, kt_b, s_f1);
        // d=b
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 88u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 88u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_fb, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 88u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 88u32 + fm));
        simdgroup_matmul(q_fb, kt_b, s_f1);
        // d=c
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 96u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 96u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_fc, kt_a, s_f0);
        simdgroup_elem_store(kt_b, 0, threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 96u32 + fm));
        simdgroup_elem_store(kt_b, 1, threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 96u32 + fm));
        simdgroup_matmul(q_fc, kt_b, s_f1);
        // d=d
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 104u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 104u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_fd, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 104u32 + fm),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 104u32 + fm),
        );
        simdgroup_matmul(q_fd, kt_b, s_f1);
        // d=e
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 112u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 112u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_fe, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 112u32 + fm),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 112u32 + fm),
        );
        simdgroup_matmul(q_fe, kt_b, s_f1);
        // d=f
        simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 120u32 + fm));
        simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 120u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_ff, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 120u32 + fm),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 120u32 + fm),
        );
        simdgroup_matmul(q_ff, kt_b, s_f1);
        // ── Online softmax, register-only via simd_shuffle_xor row reduce ──
        // S lives in s_f0 (cols 0..7) / s_f1 (cols 8..15). Each lane owns
        // 4 elements: (fm, fn0/fn1) of s_f0 and (fm, 8+fn0/fn1) of s_f1.
        // For an 8-lane row group sharing `fm`, xor-1 pairs adjacent fn-cols
        // and xor-8 pairs quad-groups (Apple 8×8 lane layout) — same row-
        // reduce pattern as MLX `mma.h:200-226`. Replaces the TG-roundtrip
        // softmax of the prior MMA POC (which silent-zeroed: the dump +
        // overwrite-with-P pattern on the same s_frag triggered a WAR
        // hazard between the elem_store-of-P and the next P·V matmul).
        let raw_s00 = simdgroup_elem_load(s_f0, 0) * scale_log2;
        let raw_s01 = simdgroup_elem_load(s_f0, 1) * scale_log2;
        let raw_s10 = simdgroup_elem_load(s_f1, 0) * scale_log2;
        let raw_s11 = simdgroup_elem_load(s_f1, 1) * scale_log2;
        let s00 = select(kb_off + fn0 > q_abs, neg_infinity(), raw_s00);
        let s01 = select(kb_off + fn1 > q_abs, neg_infinity(), raw_s01);
        let s10 = select(kb_off + 8u32 + fn0 > q_abs, neg_infinity(), raw_s10);
        let s11 = select(kb_off + 8u32 + fn1 > q_abs, neg_infinity(), raw_s11);
        let mxa = select(s00 > s01, s00, s01);
        let mxb = select(s10 > s11, s10, s11);
        let lane_max = select(mxa > mxb, mxa, mxb);
        let mxor1 = simd_shuffle_xor(lane_max, 1u32);
        let mx_after1 = select(lane_max > mxor1, lane_max, mxor1);
        let mxor8 = simd_shuffle_xor(mx_after1, 8u32);
        let row_max = select(mx_after1 > mxor8, mx_after1, mxor8);
        let new_m = select(row_max > m_row, row_max, m_row);
        let m_diff = exp2(m_row - new_m);
        let p00 = exp2(s00 - new_m);
        let p01 = exp2(s01 - new_m);
        let p10 = exp2(s10 - new_m);
        let p11 = exp2(s11 - new_m);
        let lane_sum = p00 + p01 + p10 + p11;
        let sxor1 = simd_shuffle_xor(lane_sum, 1u32);
        let sum_after1 = lane_sum + sxor1;
        let sxor8 = simd_shuffle_xor(sum_after1, 8u32);
        let row_sum = sum_after1 + sxor8;
        s_row = s_row * m_diff + row_sum;
        m_row = new_m;
        // Write P into distinct p_f0/p_f1 (not s_f0/s_f1) so P·V matmul reads
        // from a frag that wasn't just the Q·K^T accumulator.
        simdgroup_elem_store(p_f0, 0, p00.cast::<T>());
        simdgroup_elem_store(p_f0, 1, p01.cast::<T>());
        simdgroup_elem_store(p_f1, 0, p10.cast::<T>());
        simdgroup_elem_store(p_f1, 1, p11.cast::<T>());
        // ── Scale all 16 O frags by m_diff ──
        simdgroup_elem_store(o_f0, 0, simdgroup_elem_load(o_f0, 0) * m_diff);
        simdgroup_elem_store(o_f0, 1, simdgroup_elem_load(o_f0, 1) * m_diff);
        simdgroup_elem_store(o_f1, 0, simdgroup_elem_load(o_f1, 0) * m_diff);
        simdgroup_elem_store(o_f1, 1, simdgroup_elem_load(o_f1, 1) * m_diff);
        simdgroup_elem_store(o_f2, 0, simdgroup_elem_load(o_f2, 0) * m_diff);
        simdgroup_elem_store(o_f2, 1, simdgroup_elem_load(o_f2, 1) * m_diff);
        simdgroup_elem_store(o_f3, 0, simdgroup_elem_load(o_f3, 0) * m_diff);
        simdgroup_elem_store(o_f3, 1, simdgroup_elem_load(o_f3, 1) * m_diff);
        simdgroup_elem_store(o_f4, 0, simdgroup_elem_load(o_f4, 0) * m_diff);
        simdgroup_elem_store(o_f4, 1, simdgroup_elem_load(o_f4, 1) * m_diff);
        simdgroup_elem_store(o_f5, 0, simdgroup_elem_load(o_f5, 0) * m_diff);
        simdgroup_elem_store(o_f5, 1, simdgroup_elem_load(o_f5, 1) * m_diff);
        simdgroup_elem_store(o_f6, 0, simdgroup_elem_load(o_f6, 0) * m_diff);
        simdgroup_elem_store(o_f6, 1, simdgroup_elem_load(o_f6, 1) * m_diff);
        simdgroup_elem_store(o_f7, 0, simdgroup_elem_load(o_f7, 0) * m_diff);
        simdgroup_elem_store(o_f7, 1, simdgroup_elem_load(o_f7, 1) * m_diff);
        simdgroup_elem_store(o_f8, 0, simdgroup_elem_load(o_f8, 0) * m_diff);
        simdgroup_elem_store(o_f8, 1, simdgroup_elem_load(o_f8, 1) * m_diff);
        simdgroup_elem_store(o_f9, 0, simdgroup_elem_load(o_f9, 0) * m_diff);
        simdgroup_elem_store(o_f9, 1, simdgroup_elem_load(o_f9, 1) * m_diff);
        simdgroup_elem_store(o_fa, 0, simdgroup_elem_load(o_fa, 0) * m_diff);
        simdgroup_elem_store(o_fa, 1, simdgroup_elem_load(o_fa, 1) * m_diff);
        simdgroup_elem_store(o_fb, 0, simdgroup_elem_load(o_fb, 0) * m_diff);
        simdgroup_elem_store(o_fb, 1, simdgroup_elem_load(o_fb, 1) * m_diff);
        simdgroup_elem_store(o_fc, 0, simdgroup_elem_load(o_fc, 0) * m_diff);
        simdgroup_elem_store(o_fc, 1, simdgroup_elem_load(o_fc, 1) * m_diff);
        simdgroup_elem_store(o_fd, 0, simdgroup_elem_load(o_fd, 0) * m_diff);
        simdgroup_elem_store(o_fd, 1, simdgroup_elem_load(o_fd, 1) * m_diff);
        simdgroup_elem_store(o_fe, 0, simdgroup_elem_load(o_fe, 0) * m_diff);
        simdgroup_elem_store(o_fe, 1, simdgroup_elem_load(o_fe, 1) * m_diff);
        simdgroup_elem_store(o_ff, 0, simdgroup_elem_load(o_ff, 0) * m_diff);
        simdgroup_elem_store(o_ff, 1, simdgroup_elem_load(o_ff, 1) * m_diff);
        // ── O += P · V (32 matmuls per SG: 16 d_frags × 2 k_chunks) ──
        // V frag elem: V_a.elem[i] = V[fm,        d_base + fn_i] = tg_vs[fm * head_dim       + d_base + fn_i]
        //              V_b.elem[i] = V[fm + 8,    d_base + fn_i] = tg_vs[(fm + 8) * head_dim + d_base + fn_i]
        // d=0
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f0);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f0);
        // d=1
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f1);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f1);
        // d=2
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f2);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f2);
        // d=3
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f3);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f3);
        // d=4
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f4);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f4);
        // d=5
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f5);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f5);
        // d=6
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f6);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f6);
        // d=7
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f7);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f7);
        // d=8
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f8);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f8);
        // d=9
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_f9);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_f9);
        // d=a
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_fa);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_fa);
        // d=b
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_fb);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_fb);
        // d=c
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_fc);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_fc);
        // d=d
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_fd);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_fd);
        // d=e
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_fe);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_fe);
        // d=f
        simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn0));
        simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn1));
        simdgroup_matmul(p_f0, v_a, o_ff);
        simdgroup_elem_store(v_b, 0, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn0));
        simdgroup_elem_store(v_b, 1, threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn1));
        simdgroup_matmul(p_f1, v_b, o_ff);
        threadgroup_barrier();
    }
    // ── Final normalize + write O to out ──
    let is_row = select(s_row > 0.0f32, 1.0f32 / s_row, 0.0f32);
    store(
        out[q_row_base + fm * head_dim + fn0],
        (simdgroup_elem_load(o_f0, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + fn1],
        (simdgroup_elem_load(o_f0, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 8u32 + fn0],
        (simdgroup_elem_load(o_f1, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 8u32 + fn1],
        (simdgroup_elem_load(o_f1, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 16u32 + fn0],
        (simdgroup_elem_load(o_f2, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 16u32 + fn1],
        (simdgroup_elem_load(o_f2, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 24u32 + fn0],
        (simdgroup_elem_load(o_f3, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 24u32 + fn1],
        (simdgroup_elem_load(o_f3, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 32u32 + fn0],
        (simdgroup_elem_load(o_f4, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 32u32 + fn1],
        (simdgroup_elem_load(o_f4, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 40u32 + fn0],
        (simdgroup_elem_load(o_f5, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 40u32 + fn1],
        (simdgroup_elem_load(o_f5, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 48u32 + fn0],
        (simdgroup_elem_load(o_f6, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 48u32 + fn1],
        (simdgroup_elem_load(o_f6, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 56u32 + fn0],
        (simdgroup_elem_load(o_f7, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 56u32 + fn1],
        (simdgroup_elem_load(o_f7, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 64u32 + fn0],
        (simdgroup_elem_load(o_f8, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 64u32 + fn1],
        (simdgroup_elem_load(o_f8, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 72u32 + fn0],
        (simdgroup_elem_load(o_f9, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 72u32 + fn1],
        (simdgroup_elem_load(o_f9, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 80u32 + fn0],
        (simdgroup_elem_load(o_fa, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 80u32 + fn1],
        (simdgroup_elem_load(o_fa, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 88u32 + fn0],
        (simdgroup_elem_load(o_fb, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 88u32 + fn1],
        (simdgroup_elem_load(o_fb, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 96u32 + fn0],
        (simdgroup_elem_load(o_fc, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 96u32 + fn1],
        (simdgroup_elem_load(o_fc, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 104u32 + fn0],
        (simdgroup_elem_load(o_fd, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 104u32 + fn1],
        (simdgroup_elem_load(o_fd, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 112u32 + fn0],
        (simdgroup_elem_load(o_fe, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 112u32 + fn1],
        (simdgroup_elem_load(o_fe, 1) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 120u32 + fn0],
        (simdgroup_elem_load(o_ff, 0) * is_row).cast::<T>(),
    );
    store(
        out[q_row_base + fm * head_dim + 120u32 + fn1],
        (simdgroup_elem_load(o_ff, 1) * is_row).cast::<T>(),
    );
}
