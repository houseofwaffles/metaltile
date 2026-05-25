//! Multi-query SDPA for `head_dim == 256`. The d=256 sibling of
//! `ffai_sdpa_multi`, generalising `ffai_sdpa_decode_d256` with a
//! query dimension. One threadgroup per (query, q_head) attends a
//! shared K/V cache in a single dispatch, TPG=1024 online softmax,
//! same multi-query causal / full semantics as the d=128 reference
//! kernel (`sdpa_multi.rs`).
//!
//! Two attention modes select via the `causal` uniform:
//!
//!   - `causal == 0`, every query attends `[0, base_kv + n_query)`,
//!     full / bidirectional over the cached prefix plus the whole
//!     block.
//!   - `causal == 1`, query `r` attends `[0, base_kv + r + 1)`,
//!     causal within the block, the prefix always fully visible.
//!
//! Needed for Qwen3.6-A3B's full-attention layers (`head_dim=256`),
//! whose prefill path in FFAI (`Qwen35AttentionMixer.forwardMany`)
//! currently falls back to a per-token `Ops.sdpaDecode` T-loop because
//! `Ops.sdpaMulti` only accepts `head_dim == 128`. With this kernel
//! wired, the whole prefill block fans out across one dispatch.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel, STRICT threadgroup geometry, same
//! machine-freeze hazard as `ffai_sdpa_decode_d256` and
//! `ffai_sdpa_multi`. Consumers MUST encode these as preconditions
//! in their wrappers.
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Hard. A TPG
//!   below 32 makes `n_simd = TPG / 32 = 0`, turning the K walk
//!   `range(sg, n_kv, 0)` into an infinite GPU loop (the freeze).
//! - **`head_dim == 256`.** Each lane owns 8 consecutive Q/K/V
//!   elements at `lane*8 + {0..7}`, indexed unconditionally.
//! - **Grid: 1 threadgroup per (query, q_head).** `tgid_x` ranges
//!   `[0, n_q_heads * n_query)`, decoded `query = tgid / n_q_heads`,
//!   `q_head = tgid % n_q_heads`. Wrapper dispatches
//!   `grid = (n_q_heads * n_query * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`n_q_heads % heads_per_group == 0`** for integer GQA fan-out.
//! - **`base_kv + n_query <= kv_stride`**, the kernel never walks
//!   past the cache's allocated depth.
//!
//! K/V cache layout `[n_kv_heads, kv_stride, head_dim]`, Q and `out`
//! layout `[n_query, n_q_heads, head_dim]`. Online softmax runs in
//! fp32 throughout (storage stays in T).
//!
//! ## Why 2-phase output reduction
//!
//! `sdpa_multi` (d=128) keeps all 4 per-lane output dims in 4 tg_outN
//! buffers of `n_lanes * (n_simd + 1) = 32 * 33 = 1056` floats each
//! (`+1` is bank-conflict padding). 4 × 1056 = 4 224 floats = ~16 KB.
//!
//! d=256 has 8 per-lane output dims. 8 × 1056 = 8 448 floats = ~33 KB,
//! over Apple's per-kernel threadgroup-memory cap. We split into two
//! halves (dims 0..3 then 4..7) and reuse the same 4 tg_out buffers
//! across both phases. Two extra barriers, same ~16 KB allocation.
//! This mirrors `ffai_sdpa_decode_d256` exactly.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_multi_d256",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_sdpa_multi_d256<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] causal: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // KV positions this query attends. causal: prefix + the block up to
    // and including this query. full: prefix + the entire block.
    let n_kv = select(causal == 1u32, base_kv + query_idx + 1u32, base_kv + n_query);
    // Two-phase reduction: 4 tg_outN buffers, reused for dims (0..3)
    // then (4..7). 1056 = n_lanes * (n_simd + 1) with bank padding.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 8u32;
    // Pre-scale this lane's 8-element Q stripe once, K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let q4 = load(q[q_off + d0 + 4u32]).cast::<f32>() * scale;
    let q5 = load(q[q_off + d0 + 5u32]).cast::<f32>() * scale;
    let q6 = load(q[q_off + d0 + 6u32]).cast::<f32>() * scale;
    let q7 = load(q[q_off + d0 + 7u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut o4 = 0.0f32;
    let mut o5 = 0.0f32;
    let mut o6 = 0.0f32;
    let mut o7 = 0.0f32;
    // Each simdgroup walks every ns-th KV position. simd_sum reduces the
    // per-lane stripe dot product into the full score, online softmax
    // updates the running (max, sum), V accumulates into fp32 registers.
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv0 + 1u32]).cast::<f32>();
        let k2 = load(k[kv0 + 2u32]).cast::<f32>();
        let k3 = load(k[kv0 + 3u32]).cast::<f32>();
        let k4 = load(k[kv0 + 4u32]).cast::<f32>();
        let k5 = load(k[kv0 + 5u32]).cast::<f32>();
        let k6 = load(k[kv0 + 6u32]).cast::<f32>();
        let k7 = load(k[kv0 + 7u32]).cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3 + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv0 + 1u32]).cast::<f32>();
        let v2 = load(v[kv0 + 2u32]).cast::<f32>();
        let v3 = load(v[kv0 + 3u32]).cast::<f32>();
        let v4 = load(v[kv0 + 4u32]).cast::<f32>();
        let v5 = load(v[kv0 + 5u32]).cast::<f32>();
        let v6 = load(v[kv0 + 6u32]).cast::<f32>();
        let v7 = load(v[kv0 + 7u32]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
        o6 = o6 * factor + weight * v6;
        o7 = o7 * factor + weight * v7;
    }
    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // ── Cross-simdgroup output reduction — phase 1 (dims 0..3) ─────
    // Transpose layout: idx = lane * stride + sg. After barrier, sg=0
    // lanes read back ns partials each for their own (lane, d) slot.
    // Padded stride (ns + 1) avoids bank conflicts (see sdpa_decode).
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
            so3 = so3 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
        store(out[out_off + 3u32], so3.cast::<T>());
    }
    threadgroup_barrier();
    // ── Cross-simdgroup output reduction — phase 2 (dims 4..7) ─────
    threadgroup_store("tg_out0", idx, o4 * rescale);
    threadgroup_store("tg_out1", idx, o5 * rescale);
    threadgroup_store("tg_out2", idx, o6 * rescale);
    threadgroup_store("tg_out3", idx, o7 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so4 = 0.0f32;
        let mut so5 = 0.0f32;
        let mut so6 = 0.0f32;
        let mut so7 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so4 = so4 + threadgroup_load("tg_out0", ri);
            so5 = so5 + threadgroup_load("tg_out1", ri);
            so6 = so6 + threadgroup_load("tg_out2", ri);
            so7 = so7 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 4u32], so4.cast::<T>());
        store(out[out_off + 5u32], so5.cast::<T>());
        store(out[out_off + 6u32], so6.cast::<T>());
        store(out[out_off + 7u32], so7.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_multi_d256;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_multi_d256::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_multi_d256 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_multi_d256"),
                "MSL for {dt:?} should declare ffai_sdpa_multi_d256:\n{src}",
            );
        }
    }
}
