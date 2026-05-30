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

use metaltile::kernel;

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
    stack_alloc("qs", 8u32, "f32");
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
    }
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    stack_alloc("os", 8u32, "f32");
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("os", _i, 0.0f32);
    }
    stack_alloc("ks", 8u32, "f32");
    // Each simdgroup walks every ns-th KV position. simd_sum reduces the
    // per-lane stripe dot product into the full score, online softmax
    // updates the running (max, sum), V accumulates into fp32 registers.
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        for _i in range(0u32, 8u32, 1u32) {
            stack_store("ks", _i, load(k[kv0 + _i]).cast::<f32>());
        }
        let mut partial = 0.0f32;
        for _i in range(0u32, 8u32, 1u32) {
            partial = partial + stack_load("qs", _i) * stack_load("ks", _i);
        }
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        for _i in range(0u32, 8u32, 1u32) {
            stack_store(
                "os",
                _i,
                stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
            );
        }
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
    threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("os", 2u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("os", 3u32) * rescale);
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
    threadgroup_store("tg_out0", idx, stack_load("os", 4u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("os", 5u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("os", 6u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("os", 7u32) * rescale);
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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_multi_d256;
    use crate::utils::{pack_f32, unpack_f32};

    // Per (query, q_head): softmax(Q·Kᵀ·scale)·V over the attended KV
    // range. The online-softmax multi-query kernel produces the same
    // result as dense softmax-attention. Q/out layout
    // `[n_query, n_q_heads, head_dim]`, K/V layout
    // `[n_kv_heads, kv_stride, head_dim]`. `causal=false` → full range.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_multi(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base_kv: usize,
        n_query: usize,
        kv_stride: usize,
        causal: bool,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
            let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
            for qh in 0..n_q_heads {
                let kvh = qh / gqa;
                let q_off = (r * n_q_heads + qh) * head_dim;
                let kv_slab = kvh * kv_stride * head_dim;
                let mut scores = vec![0.0f32; n_kv];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_off = kv_slab + t * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    *score = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t, s) in scores.iter().enumerate() {
                        acc += *s * inv * v[kv_slab + t * head_dim + d];
                    }
                    out[q_off + d] = acc;
                }
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    fn multi_d256_setup(dt: DType, causal: bool) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 256usize);
        let (base_kv, n_query) = (56usize, 8usize);
        let kv_stride = base_kv + n_query; // 64
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa_multi(
            &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, causal, scale,
        );

        TestSetup::new(ffai_sdpa_multi_d256::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_query * n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("causal", u32::from(causal))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
    }

    // Full (bidirectional) attention over the d256 two-phase reduction.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_ffai_sdpa_multi_d256(dt: DType) -> TestSetup { multi_d256_setup(dt, false) }

    // Causal: query row `r` attends `[0, base_kv + r + 1)`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_ffai_sdpa_multi_d256_causal(dt: DType) -> TestSetup { multi_d256_setup(dt, true) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_multi_d256;

    #[bench(name = "ffai/sdpa_multi_d256", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_multi_d256(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 256usize);
        let (base_kv, n_query) = (4096usize, 8usize);
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv = base_kv + n_query;
        let bytes = (2 * n_query * n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_multi_d256::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_query * n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("causal", 0u32)
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
