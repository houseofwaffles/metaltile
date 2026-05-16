//! Naive single-Q SDPA decode with online softmax (FFAI-specific).
//! Each thread owns one output element (q_head, d). Walks all KV
//! positions; for each, computes the full dot(q[q_head], k[kv_head, t])
//! per thread (wasteful but trivially correct). Maintains per-thread
//! (max, sum, output_d) state via online softmax.
//!
//! K and V cache layout: `[n_kv_heads, kv_stride, head_dim]` where
//! kv_stride is the physical capacity (`maxSeq`) and `n_kv` is the
//! number of currently filled positions (the loop bound). Decoupling
//! the two lets the cache be pre-allocated to maxSeq while only
//! attending to filled positions.
//!
//! GQA: `kv_head = q_head / heads_per_group`.
//!
//! Different from upstream `mt_sdpa`:
//!   - generic head_dim (upstream is hardcoded to 128)
//!   - GQA via `heads_per_group` constexpr (upstream assumes Q heads = KV heads)
//!   - single-token decode form, one thread per (q_head, d)

use metaltile::kernel;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn ffai_sdpa_decode<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let idx = program_id::<0>();
    let q_head = idx / head_dim;
    let d = idx - q_head * head_dim;
    let kv_head = q_head / heads_per_group;
    let q_off = q_head * head_dim;
    let head_slab = kv_head * kv_stride * head_dim;

    let mut m = neg_infinity();
    let mut s = 0.0f32;
    let mut o = 0.0f32;

    for _t in range(0u32, n_kv, 1u32) {
        let k_base = head_slab + _t * head_dim;
        let mut score = 0.0f32;
        for j in range(0u32, head_dim, 1u32) {
            score = score
                + load(q[q_off + j]).cast::<f32>()
                * load(k[k_base + j]).cast::<f32>();
        }
        score = score * scale;

        let new_m = select(score > m, score, m);
        let factor = exp(m - new_m);
        let weight = exp(score - new_m);
        s = s * factor + weight;

        let v_idx = k_base + d;
        o = o * factor + weight * load(v[v_idx]).cast::<f32>();
        m = new_m;
    }

    let final_out = o / s;
    store(out[idx], final_out.cast::<T>());
}

inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "ffai_sdpa_decode",
        kernel_name: "ffai_sdpa_decode",
        kernel_ir: ffai_sdpa_decode::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: None,
    }
}
