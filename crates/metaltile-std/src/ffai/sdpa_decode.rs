//! Single-token SDPA decode with GQA, threadgroup-stride K walk, and
//! cross-simdgroup online-softmax reduction — FFAI's production decode
//! kernel.
//!
//! Layout assumptions:
//!   * `head_dim == 128` (one threadgroup is 32 simdgroups × 32 lanes;
//!     each lane owns 4 consecutive Q/K/V elements). Other head dims
//!     are queued for follow-up specializations — common targets
//!     include 64 (smaller Qwen/Llama variants) and 256 (some Gemma
//!     configurations). The right shape for each is the same pattern
//!     with `head_dim / 32` elements per lane.
//!   * K/V cache shape `[n_kv_heads, kv_stride, head_dim]` where
//!     `kv_stride` is the pre-allocated maxSeq capacity and `n_kv` is
//!     the currently-filled prefix. The kernel walks `[0, n_kv)` only.
//!   * GQA: `kv_head = q_head / heads_per_group`. Set
//!     `heads_per_group = 1` to disable GQA.
//!
//! Dispatch: one threadgroup per Q head (1D grid, tgid_x = q_head),
//! 1024 threads (32 simdgroups × 32 lanes).
//!
//! The dispatch + walk pattern mirrors `mlx/sdpa_vector.rs`
//! (mt_sdpa_vector). The differences are FFAI-specific:
//!
//! * `kv_stride` decoupled from `n_kv` (cache pre-allocated to
//!   `maxSeq`; loop bound is the filled prefix `n_kv`).
//! * `heads_per_group` parameter name (instead of `gqa_factor`).
//!
//! Online-softmax math runs in fp32 throughout (storage stays in T)
//! to avoid catastrophic cancellation in the `exp(max_old - max_new)`
//! rescale at long context.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn sdpa_decode<T>(
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
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        // Pre-compute index VIDs BEFORE issuing loads — vectorize wants
        // 4 consecutive Op::Load with no BinOp/Const interleaved, so
        // the kv_idx+1/2/3 adds need to happen up here.
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

inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_decode",
        kernel_name: "sdpa_decode",
        kernel_ir: sdpa_decode::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::sdpa_decode;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = sdpa_decode::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("sdpa_decode codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void sdpa_decode"),
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
