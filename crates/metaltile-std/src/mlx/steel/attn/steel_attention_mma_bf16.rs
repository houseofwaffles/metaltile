//! Prefill SDPA via `simdgroup_multiply_accumulate` (MMA) — `mt_sdpa_prefill_mma`.
//!
//! MLX-style dd-loop (steel_attention.h `tile_matmad` body): per K-block, a
//! single Q simdgroup-matrix fragment is reloaded per d_frag iteration,
//! mirroring MLX's `Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[...])` +
//! `tile_matmad(Stile, Qtile, Ktile, Stile)`. We deviate from MLX in two
//! ways that turn out to matter on Apple Silicon (M2 specifically):
//!
//! 1. **No Qs threadgroup staging.** Q is hoisted into 32 thread-private
//!    scalar registers per lane (16 d_frags × {fn0, fn1}) before the kb-loop.
//!    Q is tiny (per-SG = 8 rows × 128 cols = 1KB of device data) and Apple
//!    Metal's L1 keeps it hot anyway; explicit scalars sidestep the TG-load
//!    bank-conflict cost MLX hits at f32 (its Qs lives in the same 32KB TG
//!    budget as Ks/Vs and there isn't headroom for a kv_ld pad on Qs).
//!
//! 2. **Same dtype as input for MMA tiles (no f32 retype of Q/K/V/P).**
//!    Tested all-float32 MMA per MLX `MMATile<float>` pattern: native f32
//!    MMA at bf16 input is faster *per op* on M2 than bf16-MMA, but the
//!    forced f32 frag retype doubles simdgroup-matrix register pressure
//!    and triggers register spilling. Net regression. Keeping T-typed
//!    Q/K^T/V/P frags (matching baseline) + reducing simdgroup-matrix frag
//!    count from 22 → 7 (via single q_frag) does the actual work —
//!    register-pressure relief is what unlocks the bf16 path on M2.
//!    Result on M2: bf16 85% → ~99% vs MLX ref (raw +25% throughput);
//!    f32 131% → ~123-125% (raw -3 to -6%); f16 98% → ~91-94% (raw -3 to
//!    -6%) — mild regression we accept for the bf16 swing. M5 unchanged on
//!    all dtypes (bf16 already ~106% baseline, stays there).
//!
//! Geometry mirrors `mt_sdpa_prefill` (BQ=32, BK=16, BD=128, WM=4, WN=1,
//! tpg=128 = 4 SGs sharing one K/V TG cache). Per K-block per SG:
//!   1. Coop load K, V tiles into TG memory (Q preloaded outside loop).
//!   2. Q·K^T → S via dd-loop: 16 d_frags × 2 k_chunks = 32 matmuls per SG.
//!      Each iter: write q_frag from preloaded scalars; load kt_a/kt_b from
//!      tg_ks; matmul into s_f0/s_f1.
//!   3. Online softmax via simd_shuffle_xor row reduce (no TG roundtrip).
//!   4. O += P · V (32 matmuls per SG, same dd-loop structure).
//!
//! Apple frag lane layout (32 lanes per SG, 8×8 fragments):
//!   `qid = lane/4, fm = (qid & 4) + (lane/2 % 4),
//!    fn0 = (qid & 2)*2 + (lane%2)*2, fn1 = fn0 + 1`
//! Each lane owns 2 elements per frag at positions (fm, fn0) and (fm, fn1).

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_prefill_mma_bf16",
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
)]
#[kernel]
pub fn mt_sdpa_prefill_mma_bf16<T>(
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

    // Batched-prefill layout: q/out [batch, n_q_heads, q_len, head_dim],
    // k/v [batch, n_kv_heads, k_len, head_dim]. Single-batch B=1
    // collapses to the original `(kv|q_head) * len * head_dim` form.
    let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
    let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
    let q_tile_first = q_tile * bq + sg * bq_sg;
    let q_row_base = q_head_row_off + q_tile_first * head_dim;

    // TG memory layout (Q lives in thread-private scalar regs, see preload
    // below — no tg_qs):
    //   tg_ks: BK × kv_ld = 16 × 132 = 2112 T elems (8KB f32, 4KB f16/bf16)
    //   tg_vs: BK × kv_ld = 16 × 132 = 2112 T elems (8KB f32, 4KB f16/bf16)
    // kv_ld=132 = 128+4 pad: skips bank-conflict pattern on the column-major
    // K^T reads (`tg_ks[fn * kv_ld + fm]` strides by kv_ld across lanes).
    let kv_ld = 132u32;
    threadgroup_alloc("tg_ks", 2112, T);
    threadgroup_alloc("tg_vs", 2112, T);

    // ── Init 16 O frags to zero (f32) ──
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

    // S/P frags (2: cols 0..7 and 8..15) reused per K-block. f32 — mirrors MLX.
    let s_f0 = simdgroup_alloc::<f32, 8, 8>();
    let s_f1 = simdgroup_alloc::<f32, 8, 8>();
    let p_f0 = simdgroup_alloc::<T, 8, 8>();
    let p_f1 = simdgroup_alloc::<T, 8, 8>();
    // Q/K^T/V frags reused per d_frag. Single q frag (MLX dd-loop pattern):
    // 7 simdgroup-matrix frags total (q + kt_a + kt_b + v_a + v_b + p_f0 +
    // p_f1) vs original 22 (16 Q preloaded + 6). The register-pressure relief
    // is what closes M2's bf16-MMA gap; Q values themselves are kept hot in
    // thread-private scalar regs (see preload below).
    let q_frag = simdgroup_alloc::<T, 8, 8>();
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

    // ── Preload Q into thread-scalar regs (32 values per lane). ──
    // Single q_frag (vs original 16) drops simdgroup-matrix register pressure;
    // hoisting Q-reads out of the kb-loop avoids redundant device loads per dd.
    // 32 scalars per lane = 32×sizeof(T) bytes thread-private regs (well within
    // budget). Same MMA dtype (T) as baseline — no f32 retype, no cast cost.
    let q_d0_0 = load(q[q_row_base + fm * head_dim + 0u32 + fn0]).cast::<T>();
    let q_d0_1 = load(q[q_row_base + fm * head_dim + 0u32 + fn1]).cast::<T>();
    let q_d1_0 = load(q[q_row_base + fm * head_dim + 8u32 + fn0]).cast::<T>();
    let q_d1_1 = load(q[q_row_base + fm * head_dim + 8u32 + fn1]).cast::<T>();
    let q_d2_0 = load(q[q_row_base + fm * head_dim + 16u32 + fn0]).cast::<T>();
    let q_d2_1 = load(q[q_row_base + fm * head_dim + 16u32 + fn1]).cast::<T>();
    let q_d3_0 = load(q[q_row_base + fm * head_dim + 24u32 + fn0]).cast::<T>();
    let q_d3_1 = load(q[q_row_base + fm * head_dim + 24u32 + fn1]).cast::<T>();
    let q_d4_0 = load(q[q_row_base + fm * head_dim + 32u32 + fn0]).cast::<T>();
    let q_d4_1 = load(q[q_row_base + fm * head_dim + 32u32 + fn1]).cast::<T>();
    let q_d5_0 = load(q[q_row_base + fm * head_dim + 40u32 + fn0]).cast::<T>();
    let q_d5_1 = load(q[q_row_base + fm * head_dim + 40u32 + fn1]).cast::<T>();
    let q_d6_0 = load(q[q_row_base + fm * head_dim + 48u32 + fn0]).cast::<T>();
    let q_d6_1 = load(q[q_row_base + fm * head_dim + 48u32 + fn1]).cast::<T>();
    let q_d7_0 = load(q[q_row_base + fm * head_dim + 56u32 + fn0]).cast::<T>();
    let q_d7_1 = load(q[q_row_base + fm * head_dim + 56u32 + fn1]).cast::<T>();
    let q_d8_0 = load(q[q_row_base + fm * head_dim + 64u32 + fn0]).cast::<T>();
    let q_d8_1 = load(q[q_row_base + fm * head_dim + 64u32 + fn1]).cast::<T>();
    let q_d9_0 = load(q[q_row_base + fm * head_dim + 72u32 + fn0]).cast::<T>();
    let q_d9_1 = load(q[q_row_base + fm * head_dim + 72u32 + fn1]).cast::<T>();
    let q_da_0 = load(q[q_row_base + fm * head_dim + 80u32 + fn0]).cast::<T>();
    let q_da_1 = load(q[q_row_base + fm * head_dim + 80u32 + fn1]).cast::<T>();
    let q_db_0 = load(q[q_row_base + fm * head_dim + 88u32 + fn0]).cast::<T>();
    let q_db_1 = load(q[q_row_base + fm * head_dim + 88u32 + fn1]).cast::<T>();
    let q_dc_0 = load(q[q_row_base + fm * head_dim + 96u32 + fn0]).cast::<T>();
    let q_dc_1 = load(q[q_row_base + fm * head_dim + 96u32 + fn1]).cast::<T>();
    let q_dd_0 = load(q[q_row_base + fm * head_dim + 104u32 + fn0]).cast::<T>();
    let q_dd_1 = load(q[q_row_base + fm * head_dim + 104u32 + fn1]).cast::<T>();
    let q_de_0 = load(q[q_row_base + fm * head_dim + 112u32 + fn0]).cast::<T>();
    let q_de_1 = load(q[q_row_base + fm * head_dim + 112u32 + fn1]).cast::<T>();
    let q_df_0 = load(q[q_row_base + fm * head_dim + 120u32 + fn0]).cast::<T>();
    let q_df_1 = load(q[q_row_base + fm * head_dim + 120u32 + fn1]).cast::<T>();

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

        // Per-dd: write q_frag from preloaded scalars (q_dX_Y), load kt_a/kt_b
        // from tg_ks, then matmul. K^T frag elem layout:
        //   elem[i] = K[k_base + fn_i, d_base + fm]
        //           = tg_ks[(k_chunk_base + fn_i) * kv_ld + d_base + fm]
        // d=0
        simdgroup_elem_store(q_frag, 0, q_d0_0);
        simdgroup_elem_store(q_frag, 1, q_d0_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 0u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 0u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 0u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 0u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=1
        simdgroup_elem_store(q_frag, 0, q_d1_0);
        simdgroup_elem_store(q_frag, 1, q_d1_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 8u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 8u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 8u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 8u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=2
        simdgroup_elem_store(q_frag, 0, q_d2_0);
        simdgroup_elem_store(q_frag, 1, q_d2_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 16u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 16u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 16u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 16u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=3
        simdgroup_elem_store(q_frag, 0, q_d3_0);
        simdgroup_elem_store(q_frag, 1, q_d3_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 24u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 24u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 24u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 24u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=4
        simdgroup_elem_store(q_frag, 0, q_d4_0);
        simdgroup_elem_store(q_frag, 1, q_d4_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 32u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 32u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 32u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 32u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=5
        simdgroup_elem_store(q_frag, 0, q_d5_0);
        simdgroup_elem_store(q_frag, 1, q_d5_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 40u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 40u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 40u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 40u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=6
        simdgroup_elem_store(q_frag, 0, q_d6_0);
        simdgroup_elem_store(q_frag, 1, q_d6_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 48u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 48u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 48u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 48u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=7
        simdgroup_elem_store(q_frag, 0, q_d7_0);
        simdgroup_elem_store(q_frag, 1, q_d7_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 56u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 56u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 56u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 56u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=8
        simdgroup_elem_store(q_frag, 0, q_d8_0);
        simdgroup_elem_store(q_frag, 1, q_d8_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 64u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 64u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 64u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 64u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=9
        simdgroup_elem_store(q_frag, 0, q_d9_0);
        simdgroup_elem_store(q_frag, 1, q_d9_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 72u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 72u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 72u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 72u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=a
        simdgroup_elem_store(q_frag, 0, q_da_0);
        simdgroup_elem_store(q_frag, 1, q_da_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 80u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 80u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 80u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 80u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=b
        simdgroup_elem_store(q_frag, 0, q_db_0);
        simdgroup_elem_store(q_frag, 1, q_db_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 88u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 88u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 88u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 88u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=c
        simdgroup_elem_store(q_frag, 0, q_dc_0);
        simdgroup_elem_store(q_frag, 1, q_dc_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 96u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 96u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 96u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 96u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=d
        simdgroup_elem_store(q_frag, 0, q_dd_0);
        simdgroup_elem_store(q_frag, 1, q_dd_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 104u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 104u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 104u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 104u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=e
        simdgroup_elem_store(q_frag, 0, q_de_0);
        simdgroup_elem_store(q_frag, 1, q_de_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 112u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 112u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 112u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 112u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);
        // d=f
        simdgroup_elem_store(q_frag, 0, q_df_0);
        simdgroup_elem_store(q_frag, 1, q_df_1);
        simdgroup_elem_store(
            kt_a,
            0,
            threadgroup_load("tg_ks", fn0 * kv_ld + 120u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_a,
            1,
            threadgroup_load("tg_ks", fn1 * kv_ld + 120u32 + fm).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(q_frag, kt_a, s_f0);
        simdgroup_elem_store(
            kt_b,
            0,
            threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 120u32 + fm).cast::<T>(),
        );
        simdgroup_elem_store(
            kt_b,
            1,
            threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 120u32 + fm).cast::<T>(),
        );
        simdgroup_matmul(q_frag, kt_b, s_f1);

        // ── Online softmax, register-only via simd_shuffle_xor row reduce ──
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

        // P frag is f32 (MLX MMATile<float> pattern); P·V runs as float MMA.
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
        // V frag elem: V_a.elem[i] = V[fm,        d_base + fn_i] = tg_vs[fm * kv_ld       + d_base + fn_i]
        //              V_b.elem[i] = V[fm + 8,    d_base + fn_i] = tg_vs[(fm + 8) * kv_ld + d_base + fn_i]
        // d=0
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f0);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f0);
        // d=1
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f1);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f1);
        // d=2
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f2);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f2);
        // d=3
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f3);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f3);
        // d=4
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f4);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f4);
        // d=5
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f5);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f5);
        // d=6
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f6);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f6);
        // d=7
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f7);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f7);
        // d=8
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_f8);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f8);
        // d=9
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_f9);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_f9);
        // d=a
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_fa);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_fa);
        // d=b
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_fb);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_fb);
        // d=c
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_fc);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_fc);
        // d=d
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn1).cast::<T>(),
        );
        simdgroup_barrier_mem_none();
        simdgroup_matmul(p_f0, v_a, o_fd);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_fd);
        // d=e
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_fe);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f1, v_b, o_fe);
        // d=f
        simdgroup_elem_store(
            v_a,
            0,
            threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_a,
            1,
            threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn1).cast::<T>(),
        );
        simdgroup_matmul(p_f0, v_a, o_ff);
        simdgroup_elem_store(
            v_b,
            0,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn0).cast::<T>(),
        );
        simdgroup_elem_store(
            v_b,
            1,
            threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn1).cast::<T>(),
        );
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
