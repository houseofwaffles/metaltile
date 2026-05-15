//! Scaled dot-product attention (decode mode) benchmark — #[kernel] DSL vs MLX
//!   metal/scaled_dot_product_attention.metal  (MLX, Apache-2.0)
//!
//! Single query token attending over a KV cache (decode / prefill latency).
//!
//! MLX kernel: sdpa_vector_float_128_128 / sdpa_vector_half_128_128
//!   (scaled_dot_product_attention.metal)
//!   Params: (queries, keys, values, out, gqa_factor, N, k_head_str, k_seq_str,
//!            v_head_str, v_seq_str, scale) — slots [0..10]
//!   Function constants (all false): 20=has_mask 21=query_transposed 22=do_causal
//!                                    23=bool_mask 24=float_mask 25=has_sinks
//!   Grid: [H, 1, 1] × [1024, 1, 1]  (32 simdgroups × 32 lanes per head)
//!   Algorithm: Online softmax over all N_kv tokens; SIMD-group inter-merge via
//!              threadgroup memory. Each lane holds EPT=4 Q/K/V elements (D=128).
//!
//! MetalTile: mt_sdpa — single-simdgroup (32 threads) decode SDPA.
//!   Generic over T (f32/f16/bf16); intermediate math always in f32.
//!   Grid: [H, 1, 1] × [MT_TPG, 1, 1]
//!   KernelMode::Reduction
//!
//! Note: MT uses 8 simdgroups per head (256 threads) vs MLX's 32 simdgroups per
//! head (1024 threads). Fewer groups = less parallelism but simpler merge.

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{
        DEFAULT_MIN_COSINE_SIM,
        DType,
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        check_equiv_with,
        dtype_tol,
        to_gbps,
    },
    runner::{CompiledKernel, GpuBuffer, GpuRunner},
};

static SRC: &str = include_str!("../metal/scaled_dot_product_attention.metal");

const BENCH: OpBench = OpBench::new("sdpa_f32", "GB/s");
const BENCH_F16: OpBench = OpBench::new("sdpa_f16", "GB/s");
// (H, N_kv, D)  —  D must be 128 (4 elements/lane × 32 lanes)
const SHAPES: &[(usize, usize, usize)] = &[(8, 2048, 128), (32, 4096, 128)];
const MT_TPG: usize = 256;
const REF_NAME: &str = "sdpa_vector_float_128_128";
const REF_NAME_F16: &str = "sdpa_vector_float16_t_128_128";
// function constants: 20=has_mask 21=query_transposed 22=do_causal
//                     23=bool_mask 24=float_mask 25=has_sinks → all false
const REF_FCS: &[(usize, bool)] =
    &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];

// ── DSL kernel ───────────────────────────────────────────────────────────────

/// Multi-simdgroup decode SDPA for D=128. Generic over dtype T.
///
/// n_simd SIMD groups each handle every n_simd-th K/V token.
/// Within each group, 32 lanes hold EPT=4 Q/K/V elements (D=128).
/// Online softmax per-group, then cross-group merge via threadgroup memory
/// (each group writes per-lane partial outputs, sg=0 sums them per lane).
///
/// Dispatch: [H, 1, 1] × [256, 1, 1]  (8 simdgroups × 32 lanes per head)
#[kernel]
pub fn mt_sdpa<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_kv: u32,
    #[constexpr] scale: f32,
) {
    let head = program_id::<0>();
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // Threadgroup storage: per-group (max, sum) + per-lane×per-group outputs
    // layout: tg_max[ns], tg_sum[ns], tg_out[lane * ns + sg][4 elems]
    threadgroup_alloc("tg_max", 8);
    threadgroup_alloc("tg_sum", 8);
    threadgroup_alloc("tg_out0", 256);
    threadgroup_alloc("tg_out1", 256);
    threadgroup_alloc("tg_out2", 256);
    threadgroup_alloc("tg_out3", 256);

    // base offsets (D = 128, EPT = 4 elements per lane)
    let q_off = head * 128u32;
    let kv_base = head * n_kv * 128u32;
    let d0 = lane * 4u32;

    // Load scaled query elements (4 per lane = 128 total per head)
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

    // Each simdgroup strides over tokens: sg, sg+ns, sg+2*ns, ...
    for _t in range(sg, n_kv, ns) {
        let base = kv_base + _t * 128u32;

        let partial = q0 * load(k[base + d0]).cast::<f32>()
            + q1 * load(k[base + d0 + 1u32]).cast::<f32>()
            + q2 * load(k[base + d0 + 2u32]).cast::<f32>()
            + q3 * load(k[base + d0 + 3u32]).cast::<f32>();
        let score = simd_sum(partial);

        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;

        o0 = o0 * factor + weight * load(v[base + d0]).cast::<f32>();
        o1 = o1 * factor + weight * load(v[base + d0 + 1u32]).cast::<f32>();
        o2 = o2 * factor + weight * load(v[base + d0 + 2u32]).cast::<f32>();
        o3 = o3 * factor + weight * load(v[base + d0 + 3u32]).cast::<f32>();
    }

    // Cross-simdgroup merge: store per-group (max, sum) from lane 0 only (values are lane-invariant)
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();

    // Simdgroup 0 finds global max and rescales sums
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

    // Each lane rescales its per-group output and stores to tg_out[lane * ns + sg]
    let rescale = exp(run_max - g_max) / g_sum;
    let idx = lane * ns + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();

    // Simdgroup 0: each lane reads all ns groups' contributions for this lane and sums them.
    // Since there are 32 lanes and ns groups, each lane handles one group's data and
    // we accumulate via simd_sum (which sums across all 32 lanes).
    // But we need per-lane sums, not global sums. So we use a different approach:
    // Each lane loops over ns groups, summing its own 4 output elements.
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * ns + _g;
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

fn sdpa_msl_for(dt: DType) -> Result<String, String> {
    let mut k = mt_sdpa::kernel_ir_for(dt);
    k.mode = KernelMode::Reduction;
    MslGenerator::default()
        .generate(&k)
        .map_err(|e| format!("sdpa codegen ({dt:?}): {e}"))
        .and_then(|msl| if msl.trim().is_empty() { Err("empty MSL".into()) } else { Ok(msl) })
}

// ── CPU reference ─────────────────────────────────────────────────────────────

fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32], h: usize, n: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; h * d];
    for head in 0..h {
        let q_base = head * d;
        let kv_base = head * n * d;
        let mut scores = vec![0.0f32; n];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..n {
            let base = kv_base + t * d;
            let qk: f32 = (0..d).map(|e| q[q_base + e] * k[base + e]).sum::<f32>() * scale;
            scores[t] = qk;
            max_score = max_score.max(qk);
        }
        let mut sum = 0.0f32;
        let mut o = vec![0.0f32; d];
        for t in 0..n {
            let w = (scores[t] - max_score).exp();
            sum += w;
            let base = kv_base + t * d;
            for e in 0..d {
                o[e] += w * v[base + e];
            }
        }
        let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
        for e in 0..d {
            out[q_base + e] = o[e] * inv;
        }
    }
    out
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_sdpa_vector(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner.compile_with_bool_constants(SRC, REF_NAME, REF_FCS).ok();

    let msl = sdpa_msl_for(DType::F32).ok();
    let mk = msl.as_ref().and_then(|m| {
        runner.compile(m, "mt_sdpa").inspect_err(|e| eprintln!("[mt_sdpa f32] compile: {e}")).ok()
    });

    let mut results = Vec::new();
    for &(h, n, d) in SHAPES {
        assert_eq!(d, 128, "mt_sdpa is fixed at D=128");
        let scale = 1.0_f32 / (d as f32).sqrt();

        // Correctness check on a small shape
        let equiv = mk.as_ref().map(|mk| {
            let ch = 2usize;
            let cn = 64usize;
            let cq: Vec<f32> = (0..ch * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let ck: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
            let cv: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let ref_out = cpu_sdpa(&cq, &ck, &cv, ch, cn, d);
            let q_b = runner.buffer_f32(&cq);
            let k_b = runner.buffer_f32(&ck);
            let v_b = runner.buffer_f32(&cv);
            let out_b = runner.buffer_zeros(ch * d * 4);
            let n_b = runner.buffer_u32(cn as u32);
            let sc_b = runner.buffer_f32_scalar(scale);
            dispatch_mt_once(runner, mk, &q_b, &k_b, &v_b, &out_b, &n_b, &sc_b, ch, &ref_out, d)
        });

        let vals: Vec<f32> = (0..h * n * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let q_buf = runner.buffer_f32(&vals[..h * d]);
        let k_buf = runner.buffer_f32(&vals[..h * n * d]);
        let v_buf = runner.buffer_f32(&vals[..h * n * d]);
        let n_buf = runner.buffer_u32(n as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);
        let gqa = runner.buffer_i32(1i32);
        let n_i32 = runner.buffer_i32(n as i32);
        let khs = runner.buffer_u64((n * d) as u64);
        let kss = runner.buffer_u64(d as u64);
        let vhs = runner.buffer_u64((n * d) as u64);
        let vss = runner.buffer_u64(d as u64);

        let bytes = (h * n * d * 4 * 2 + h * d * 4 * 2) as f64;

        let ref_perf = rk.as_ref().and_then(|rk| {
            let out_b = runner.buffer_zeros(h * d * 4);
            let st = runner.bench(
                rk,
                &[&q_buf, &k_buf, &v_buf, &out_b, &gqa, &n_i32, &khs, &kss, &vhs, &vss, &sc_buf],
                [h, 1, 1],
                [1024, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let mt_perf = mk.as_ref().and_then(|mk| {
            let out_b = runner.buffer_zeros(h * d * 4);
            let st = runner.bench(
                mk,
                &[&q_buf, &k_buf, &v_buf, &out_b, &n_buf, &sc_buf],
                [h, 1, 1],
                [MT_TPG, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let shape = format!("H={h} N={n} D={d} f32");
        let result = if let Some(mt_perf) = mt_perf {
            BENCH.implemented(shape, ref_perf, mt_perf, equiv.expect("mk Some → equiv Some"))
        } else {
            BENCH.nyi(shape, ref_perf)
        };
        results.push(result);
    }
    results
}

/// Dispatch mt_sdpa (multi-simdgroup DSL kernel, MT_TPG threads) once and compare.
fn dispatch_mt_once(
    runner: &GpuRunner,
    mk: &CompiledKernel,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    out: &GpuBuffer,
    n_buf: &GpuBuffer,
    sc_buf: &GpuBuffer,
    h: usize,
    ref_out: &[f32],
    d: usize,
) -> EquivResult {
    runner.measure(mk, &[q, k, v, out, n_buf, sc_buf], [h, 1, 1], [MT_TPG, 1, 1], 0, 1);
    let mt_vals = runner.read_f32_slice(out, h * d);
    check_equiv_with(ref_out, &mt_vals, EquivTolerance::new(1e-3, DEFAULT_MIN_COSINE_SIM))
}

/// F16 decode SDPA bench using the generic DSL `mt_sdpa<T>` kernel.
pub fn bench_sdpa_vector_f16(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner
        .compile_with_bool_constants(SRC, REF_NAME_F16, REF_FCS)
        .inspect_err(|e| eprintln!("[{REF_NAME_F16}] compile: {e}"))
        .ok();

    let msl = sdpa_msl_for(DType::F16).ok();
    let mk = msl.as_ref().and_then(|m| {
        runner.compile(m, "mt_sdpa").inspect_err(|e| eprintln!("[mt_sdpa f16] compile: {e}")).ok()
    });

    let mut results = Vec::new();
    for &(h, n, d) in SHAPES {
        let scale = 1.0_f32 / (d as f32).sqrt();
        let tol = EquivTolerance::new(dtype_tol(DType::F16), DEFAULT_MIN_COSINE_SIM);

        // Correctness check on a small shape with f16 data
        let equiv = mk.as_ref().map(|mk| {
            let ch = 2usize;
            let cn = 64usize;
            let cq: Vec<f32> = (0..ch * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let ck_: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
            let cv: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let ref_out = cpu_sdpa(&cq, &ck_, &cv, ch, cn, d);
            let q_b = runner.buffer_f16(&f32_slice_to_f16(&cq));
            let k_b = runner.buffer_f16(&f32_slice_to_f16(&ck_));
            let v_b = runner.buffer_f16(&f32_slice_to_f16(&cv));
            let out_b = runner.buffer_zeros(ch * d * 2);
            let n_b = runner.buffer_u32(cn as u32);
            let sc_b = runner.buffer_f32_scalar(scale);
            runner.measure(
                mk,
                &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b],
                [ch, 1, 1],
                [MT_TPG, 1, 1],
                0,
                1,
            );
            let mt_vals = runner.read_f16_slice(&out_b, ch * d);
            check_equiv_with(&ref_out, &mt_vals, tol)
        });

        let vals_f32: Vec<f32> = (0..h * n * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let q_f16 = f32_slice_to_f16(&vals_f32[..h * d]);
        let kv_f16 = f32_slice_to_f16(&vals_f32[..h * n * d]);
        let q_buf = runner.buffer_f16(&q_f16);
        let k_buf = runner.buffer_f16(&kv_f16);
        let v_buf = runner.buffer_f16(&kv_f16);
        let n_buf = runner.buffer_u32(n as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);
        let gqa = runner.buffer_i32(1i32);
        let n_i32 = runner.buffer_i32(n as i32);
        let khs = runner.buffer_u64((n * d) as u64);
        let kss = runner.buffer_u64(d as u64);
        let vhs = runner.buffer_u64((n * d) as u64);
        let vss = runner.buffer_u64(d as u64);

        let bytes = (h * n * d * 2 * 2 + h * d * 2 * 2) as f64;

        let ref_perf = rk.as_ref().and_then(|rk| {
            let out_b = runner.buffer_zeros(h * d * 2);
            let st = runner.bench(
                rk,
                &[&q_buf, &k_buf, &v_buf, &out_b, &gqa, &n_i32, &khs, &kss, &vhs, &vss, &sc_buf],
                [h, 1, 1],
                [1024, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let mt_perf = mk.as_ref().and_then(|mk| {
            let out_b = runner.buffer_zeros(h * d * 2);
            let st = runner.bench(
                mk,
                &[&q_buf, &k_buf, &v_buf, &out_b, &n_buf, &sc_buf],
                [h, 1, 1],
                [MT_TPG, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let shape = format!("H={h} N={n} D={d} f16");
        let result = if let Some(mt_perf) = mt_perf {
            BENCH_F16.implemented(shape, ref_perf, mt_perf, equiv.expect("mk Some → equiv Some"))
        } else {
            BENCH_F16.nyi(shape, ref_perf)
        };
        results.push(result);
    }
    results
}

fn f32_slice_to_f16(src: &[f32]) -> Vec<u16> { src.iter().copied().map(f32_to_f16_bits).collect() }

fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_sdpa_msl_generates() {
        for dt in [DType::F32, DType::F16] {
            let msl = sdpa_msl_for(dt).expect("codegen failed");
            assert!(!msl.trim().is_empty());
            assert!(msl.contains("mt_sdpa"), "kernel name missing");
            assert!(
                msl.contains("simd_sum") || msl.contains("simd_reduce"),
                "simd reduction missing"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_sdpa_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        for dt in [DType::F32, DType::F16] {
            let msl = sdpa_msl_for(dt).expect("codegen failed");
            runner
                .compile(&msl, "mt_sdpa")
                .unwrap_or_else(|e| panic!("mt_sdpa {dt:?} compile error: {e}\nMSL:\n{msl}"));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_sdpa_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = sdpa_msl_for(DType::F32).expect("codegen");
        let mk = runner.compile(&msl, "mt_sdpa").expect("compile");
        let h = 2;
        let n = 64;
        let d = 128;
        let scale = 1.0_f32 / (d as f32).sqrt();
        let q: Vec<f32> = (0..h * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let k: Vec<f32> = (0..h * n * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let v: Vec<f32> = (0..h * n * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let ref_out = cpu_sdpa(&q, &k, &v, h, n, d);
        let q_b = runner.buffer_f32(&q);
        let k_b = runner.buffer_f32(&k);
        let v_b = runner.buffer_f32(&v);
        let out_b = runner.buffer_zeros(h * d * 4);
        let n_b = runner.buffer_u32(n as u32);
        let sc_b = runner.buffer_f32_scalar(scale);
        runner.measure(
            &mk,
            &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b],
            [h, 1, 1],
            [MT_TPG, 1, 1],
            0,
            1,
        );
        let mt_out = runner.read_f32_slice(&out_b, h * d);
        for (i, (&r, &m)) in ref_out.iter().zip(mt_out.iter()).enumerate() {
            assert!((r - m).abs() < 1e-3, "mismatch[{i}]: ref={r} mt={m}");
        }
    }
}
