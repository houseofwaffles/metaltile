//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-token SDPA decode with GQA, threadgroup-stride K walk, and
//! cross-simdgroup online-softmax reduction — FFAI's production decode
//! kernel.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! attention output (best case) or pins the GPU in an infinite loop
//! (worst case — this was the trigger for FFAI post-mortem
//! 2026-05-19). Consumers MUST encode these as preconditions in
//! their wrappers.
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Hard.
//!   Smaller TPG produces `n_simd = TPG / 32 = 0` for `TPG < 32`,
//!   making the per-token K walk `for _t in range(sg, n_kv, 0)` an
//!   infinite GPU loop. This is the freeze mode that took down the
//!   dev machine.
//! - **`head_dim == 128`.** Each lane owns 4 consecutive Q/K/V
//!   elements (128 / 32 = 4) and indexes them unconditionally at
//!   `lane * 4 + {0..3}`. Other head dims OOB-read. Specializations
//!   for 64 (smaller Qwen/Llama) and 256 (Gemma) are queued — same
//!   pattern with `head_dim / 32` elements per lane.
//! - **Grid: 1 threadgroup per q_head** (1D grid, `tgid_x = q_head`).
//!   Wrapper uses `grid = (nQHeads * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`nQHeads % nKVHeads == 0`** so GQA fan-out is integer.
//! - **`n_kv ≤ kv_stride`.** Cache is pre-allocated to `kv_stride`
//!   (maxSeq); the kernel walks `[0, n_kv)` only — passing
//!   `n_kv > kv_stride` reads past the cache.
//!
//! ## Layout
//!
//! K/V cache shape `[n_kv_heads, kv_stride, head_dim]` where
//! `kv_stride` is the pre-allocated maxSeq capacity and `n_kv` is
//! the currently-filled prefix. The kernel walks `[0, n_kv)` only.
//! GQA: `kv_head = q_head / heads_per_group`. Set
//! `heads_per_group = 1` to disable GQA.
//!
//! The dispatch + walk pattern mirrors `mlx/sdpa_vector.rs`
//! (mt_sdpa_vector). The two kernels are intentionally kept separate
//! rather than unified: `mt_sdpa_vector` is a faithful port of MLX's
//! `sdpa_vector` template, instantiated against MLX's source as the
//! `tile bench` reference. Adding FFAI-specific surface area
//! (`kv_stride`, `heads_per_group`, `sink_end`, `window_start`) to it
//! would break that 1:1 charter and the per-shape MSL diffing
//! invariant the bench harness relies on. Edits to either kernel must
//! stay aware of the other — bandwidth fixes on `mt_sdpa_vector`
//! (e.g. the `tg_out` occupancy collapse in PR #43) should be ported
//! here too, and vice versa. The differences are FFAI-specific:
//!
//! * `kv_stride` decoupled from `n_kv` (cache pre-allocated to
//!   `maxSeq`; loop bound is the filled prefix `n_kv`).
//! * `heads_per_group` parameter name (instead of `gqa_factor`).
//! * Sliding-window + sink-token specialization via the
//!   `sink_end` / `window_start` constexprs. Both default to zero in
//!   the dense path (the sink loop bound is zero so its body never
//!   emits a load; the window loop starts at `sg + 0`, identical to
//!   the original walk). When set, the kernel skips the masked range
//!   `[sink_end, window_start)` at the loop-bound level — no
//!   per-position branching, no simdgroup divergence. MLX's
//!   `sdpa_vector` mask path (sdpa_vector.h:7-13) gates per position
//!   inside the strided walk, so it still iterates every KV slot;
//!   this shrinks the iteration count itself.
//!
//! Caller contract for the sparse path: `window_start >= sink_end`
//! (otherwise the two passes overlap and the online softmax
//! double-counts the intersection) and `window_start <= n_kv`. Both
//! are constexprs so they're host-side checked, not validated in the
//! kernel.
//!
//! * Learned per-head attention sink (`has_sink` / `sink_logit`
//!   constexprs) — GPT-OSS-20B. Distinct from the `sink_end`
//!   sink-*token* range above: this is a single learned logit per
//!   head that joins the softmax denominator as a virtual key with
//!   value 0. It contributes `exp(sink_logit - g_max)` to the running
//!   sum and nothing to the output accumulator. The grid is one
//!   threadgroup per q_head, so the host passes the routed head's
//!   sink as a scalar constexpr (same as `scale`). `has_sink == 0`
//!   masks the term out — the dense / sliding-window paths stay
//!   bit-identical to the pre-sink kernel. Folding it on-GPU removes
//!   the per-layer host-side post-hoc softmax rescale GPT-OSS-20B
//!   otherwise pays.
//!
//! Online-softmax math runs in fp32 throughout (storage stays in T)
//! to avoid catastrophic cancellation in the `exp(max_old - max_new)`
//! rescale at long context.

use metaltile::kernel;

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
    #[constexpr] sink_end: u32,
    #[constexpr] window_start: u32,
    // Learned per-head attention sink (GPT-OSS-20B). When `has_sink == 1`
    // the softmax denominator gains a virtual key with score
    // `sink_logit` and value 0 — the sink contributes
    // `exp(sink_logit - g_max)` to the running sum but nothing to the
    // output accumulator. The grid is one threadgroup per q_head, so
    // the host passes the routed head's sink as a scalar constexpr,
    // exactly like `scale`. `has_sink == 0` masks the term out, so the
    // dense / sliding-window paths are bit-identical to the pre-sink
    // kernel. Distinct from the `sink_end` sink-*token* path above:
    // this is a learned per-head logit, not a position range.
    #[constexpr] has_sink: u32,
    #[constexpr] sink_logit: f32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // Cross-simdgroup reduction storage: 32 slots for max/sum (one per
    // simdgroup), 1024+32 slots × 4 quartiles for the output transpose.
    // The +32 padding avoids 32-way threadgroup-memory bank conflicts in
    // the final per-lane sum across simdgroups: without padding, sg-0's
    // sweep (lane*ns + g, ns=32) hits the same bank every iteration
    // because 32 * 4 bytes lines up exactly with Apple's 32-bank layout.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;
    // Pre-scale this lane's 4-element Q quartile once; K/V are streamed.
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
    // Each simdgroup walks every ns-th KV position. simd_sum reduces
    // the per-lane quartile dot product into the full score; online
    // softmax updates the running (max, sum) tuple; V is accumulated
    // into per-lane fp32 registers.
    //
    // Sink-token pass: walks `[0, sink_end)`. When `sink_end == 0` the
    // loop bound collapses to `range(sg, 0, ns)`, no iterations emit.
    // The body is intentionally a copy of the window pass — the
    // `#[kernel]` proc-macro does not expand `macro_rules!`
    // invocations inside the function body (the AST handed to the
    // body parser keeps the `!` call opaque), so a shared-body macro
    // produces an empty MSL kernel. Keep the two bodies bit-identical
    // by hand and rely on the simdgroup-stride-walk invariant: each
    // visited KV position contributes once across both passes as long
    // as the caller honors `window_start >= sink_end`.
    for _t in range(sg, sink_end, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
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
    // Window pass: walks `[window_start, n_kv)`. Dense path sets
    // `window_start = 0`, giving the original `range(sg, n_kv, ns)`
    // walk back; sliding window passes `n_kv - W` (or
    // `max(sink_end, n_kv - W)` when sinks are active).
    //
    // Pre-compute index VIDs BEFORE issuing loads — vectorize wants
    // 4 consecutive Op::Load with no BinOp/Const interleaved, so the
    // kv_idx+1/2/3 adds need to happen up here.
    for _t in range(sg + window_start, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
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
    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_raw = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        // Fold the learned sink logit into the cross-simdgroup max:
        // the sink is a virtual key, so the global softmax max must
        // also cover its score. Carry it on lane 0 only (combined with
        // that lane's real max, never replacing it) so simd_max sees
        // it exactly once. Masked to -inf when `has_sink == 0`.
        let sink_max = select(has_sink > 0u32, sink_logit, neg_infinity());
        let g_max_in =
            select(lane == 0u32, select(g_max_raw > sink_max, g_max_raw, sink_max), g_max_raw);
        let g_max = simd_max(g_max_in);
        // Each simdgroup's partial sum was computed against its own
        // `tg_max[lane]` (the *raw* per-simdgroup max), so the rescale
        // factor must use `g_max_raw`, not the sink-combined `g_max_in`.
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_raw - g_max), 0.0f32);
        // The sink contributes `exp(sink_logit - g_max)` to the
        // denominator (value 0 → nothing to the output accumulator).
        // Accumulate it on lane 0 so it counts exactly once in the
        // simd_sum. Zero when `has_sink == 0`.
        let sink_sum = select(has_sink > 0u32, exp(sink_logit - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in + select(lane == 0u32, sink_sum, 0.0f32));
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    // ── Cross-simdgroup reduction: outputs ──────────────────────────
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    // Guard against `n_kv == 0`: no K positions visited → run_max stays
    // -inf, g_sum stays 0, naive `exp(-inf - -inf) / 0 = NaN`. Decode
    // should never be called with an empty cache in practice (the
    // current decode step itself contributes at least one position),
    // but the guard keeps the kernel side-effect-safe regardless.
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // Transpose-then-reduce: write per-(lane, sg), read per-(sg, lane).
    // Stride `ns + 1` (33 for ns=32) so adjacent lanes hit different
    // threadgroup-memory banks during sg-0's sweep — without the +1,
    // every lane in the sweep targets the same bank because 32 * 4
    // bytes lines up exactly with Apple's 32-bank layout. Padding by
    // 1 puts each lane on a distinct bank, eliminating a 32-way
    // conflict on the final per-lane sum.
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
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_decode;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_decode codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_decode"),
                "MSL for {dt:?} should declare the kernel function:\n{src}",
            );
        }
    }

    #[test]
    fn codegen_uses_threadgroup_reduction_primitives() {
        // Pin the cross-simdgroup reduction structure: should reference
        // the simdgroup intrinsics, threadgroup memory, the simd_sum +
        // simd_max reductions, and a threadgroup-scoped barrier. If
        // any of these regress, the kernel collapsed back to the
        // per-thread placeholder OR (for the barrier scope) silently
        // lost cross-simdgroup ordering guarantees.
        let src = msl_for(DType::F32);
        for tok in &[
            "simd_group", // emitted from DSL `simd_id`
            "simd_lane",
            "threadgroup_barrier",
            "mem_threadgroup", // barrier must include threadgroup-memory scope
            "simd_sum",
            "simd_max",
        ] {
            assert!(src.contains(tok), "MSL missing `{tok}`:\n{src}");
        }
    }

    #[test]
    fn codegen_emits_kv_stride_indexing() {
        // The kernel's GQA path is `kv_head * kv_stride * head_dim`; if
        // this disappears the cache layout got broken.
        let src = msl_for(DType::F32);
        assert!(
            src.contains("kv_stride") || src.contains("v_kv_stride"),
            "MSL should reference kv_stride constexpr:\n{src}",
        );
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_decode;
    use crate::utils::{pack_f32, unpack_f32};

    /// Dense triple-loop SDPA reference: `O = softmax(Q·Kᵀ·scale)·V` per
    /// Q head, GQA via `kv_head = q_head / heads_per_group`, fp32. K/V are
    /// laid out `[n_kv_heads, kv_stride, head_dim]`; the loop walks the
    /// filled prefix `[0, n_kv)`.
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv: usize,
        kv_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_q_heads * head_dim];
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = qh * head_dim;
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
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode(dt: DType) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (8usize, 4usize, 128usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Dtype-round the inputs so the CPU oracle sees the same
        // load-cast quantization the kernel does.
        let q = unpack_f32(&pack_f32(&ramp(n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected =
            naive_sdpa(&q, &k, &v, n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, scale);

        TestSetup::new(ffai_sdpa_decode::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("sink_end", 0u32)
            .constexpr("window_start", 0u32)
            .constexpr("has_sink", 0u32)
            .constexpr("sink_logit", 0.0f32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
    }

    /// Deterministic ramp generator for test inputs: `start + i*step`,
    /// wrapped to a small range so dtype rounding stays well-conditioned.
    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }
}

/// New-syntax benchmark for `ffai_sdpa_decode` — an MLX-less production
/// decode kernel (`class=GenericEmpty`). Mirrors the legacy perf-bench
/// shape (Qwen3-class GQA, common 4096 decode context).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_decode;

    #[bench(name = "ffai/sdpa_decode", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode(dt: DType) -> BenchSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 128usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
                * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("sink_end", 0u32)
            .constexpr("window_start", 0u32)
            .constexpr("has_sink", 0u32)
            .constexpr("sink_logit", 0.0f32)
            .constexpr("scale", scale)
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
