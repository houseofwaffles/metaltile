//! Decode-form scaled dot-product attention — `mt_sdpa_vector`.
//!
//! Mirrors MLX `sdpa_vector<T, D, V=D>` semantically (online softmax over
//! `n_kv` positions, GQA via `gqa_factor`) but simplified:
//!   - no optional mask / sinks / causal / transposed-query paths
//!   - single Q (q_seq=1) — decode step only
//!   - one threadgroup per Q head (no batch axis dispatch)
//!   - one simdgroup (32 threads) per threadgroup; serial over `n_kv`
//!     positions instead of MLX's `BN=32` simdgroups in parallel
//!
//! Each thread holds `head_dim / 32` elements of the query and value
//! accumulator (so `head_dim` must be a multiple of 32). The dot product
//! across `head_dim` reduces via `simd_sum` within the simdgroup.
//!
//! `mt_sdpa_vector` closes the algorithmic-validation gap for FFAI's
//! `sdpa_decode` (which is a specialisation of this online-softmax
//! pattern) by giving MLX a side-by-side correctness reference.
//!
//! For now `head_dim` is hardcoded to 128 — the kernel unrolls 4
//! quartile-loads per thread. Wider head dims would either need a
//! runtime loop or a separate template instantiation. Future work.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_vector",
    class=SdpaVector,
    h=128,        // head_dim
    n_kv=1024,
    n_heads=32,   // n_q_heads
    gqa_factor=4, // GQA: 32 Q heads grouped onto 8 KV heads
    batch=1,
    tpg=32,       // one simdgroup
    tol=1e-3,
    // MLX `sdpa_vector` template instantiations live in
    // scaled_dot_product_attention.metal (same file as mt_sdpa's reference).
    metal_file="scaled_dot_product_attention.metal",
)]
#[kernel]
pub fn mt_sdpa_vector<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    // One threadgroup per Q head; tid is the simd lane (0..32).
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let lane = tid;
    let d_base = lane * 4u32; // head_dim=128, 4 quartiles per thread

    let q_off = q_head * head_dim;
    let kv_head_slab = kv_head * n_kv * head_dim;

    // Load this thread's 4 query elements and pre-scale.
    let q0 = load(q[q_off + d_base + 0u32]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d_base + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d_base + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d_base + 3u32]).cast::<f32>() * scale;

    let mut max_score = neg_infinity();
    let mut sum_exp = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;

    // Serial over n_kv positions. Each iteration: partial dot in this
    // lane's quartile, simd_sum to get the full score, online-softmax
    // update, per-quartile accumulate of v.
    for kv in range(0u32, n_kv, 1u32) {
        let k_off = kv_head_slab + kv * head_dim;

        let partial = q0 * load(k[k_off + d_base + 0u32]).cast::<f32>()
            + q1 * load(k[k_off + d_base + 1u32]).cast::<f32>()
            + q2 * load(k[k_off + d_base + 2u32]).cast::<f32>()
            + q3 * load(k[k_off + d_base + 3u32]).cast::<f32>();
        let score = simd_sum(partial);

        let new_max = select(score > max_score, score, max_score);
        let factor = exp(max_score - new_max);
        let exp_s = exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_s;

        o0 = o0 * factor + exp_s * load(v[k_off + d_base + 0u32]).cast::<f32>();
        o1 = o1 * factor + exp_s * load(v[k_off + d_base + 1u32]).cast::<f32>();
        o2 = o2 * factor + exp_s * load(v[k_off + d_base + 2u32]).cast::<f32>();
        o3 = o3 * factor + exp_s * load(v[k_off + d_base + 3u32]).cast::<f32>();
    }

    let inv_sum = 1.0f32 / sum_exp;
    let out_off = q_head * head_dim;
    store(out[out_off + d_base + 0u32], (o0 * inv_sum).cast::<T>());
    store(out[out_off + d_base + 1u32], (o1 * inv_sum).cast::<T>());
    store(out[out_off + d_base + 2u32], (o2 * inv_sum).cast::<T>());
    store(out[out_off + d_base + 3u32], (o3 * inv_sum).cast::<T>());
}
