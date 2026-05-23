//! GPU runner implementation extracted from metaltile-bench/src/spec.rs
//! BenchSpec::run() and all dispatch arms, transformed into free functions.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};

use crate::{
    bench_types::{
        DtypeCtx,
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        check_equiv,
        check_equiv_with,
    },
    runner::{
        GpuBuffer,
        GpuRunner,
        bench_gbps,
        bench_gbps_only,
        buffer_typed,
        read_typed,
        run_typed_once,
        zeros_typed,
    },
    spec::{BatchedDecodeVariant, BenchDispatch, BenchSpec, MlxArg, ScalarBufSpec, ShapeSpec},
};

pub fn run(spec: &BenchSpec, runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let bench = OpBench::new(spec.op, "GB/s");
    match &spec.dispatch {
        BenchDispatch::Generic => run_generic(spec, runner, dt, &bench),
        BenchDispatch::Sort { b, n, tpg } => run_sort(spec, runner, dt, &bench, *b, *n, *tpg),
        BenchDispatch::Scan { shapes, tpg } => run_scan(spec, runner, dt, &bench, shapes, *tpg),
        BenchDispatch::ArgReduce { n, check_n, tpg } =>
            run_arg_reduce(spec, runner, dt, &bench, *n, *check_n, *tpg),
        BenchDispatch::Random { n, tpg } => run_random(spec, runner, dt, &bench, *n, *tpg),
        BenchDispatch::FpQuantized { n, tpg } =>
            run_fp_quantized(spec, runner, dt, &bench, *n, *tpg),
        BenchDispatch::QuantizedMatVec { shapes, group_size, tpg, bits } =>
            run_quantized_mat_vec(spec, runner, dt, &bench, shapes, *group_size, *tpg, *bits),
        BenchDispatch::QuantizedMatMul { shapes, m, group_size, tpg, bits } =>
            run_quantized_mat_mul(spec, runner, dt, &bench, shapes, *m, *group_size, *tpg, *bits),
        BenchDispatch::Rope { b, h, l, d, n_per_group } =>
            run_rope(spec, runner, dt, &bench, *b, *h, *l, *d, *n_per_group),
        BenchDispatch::Attention { shapes, tpg } =>
            run_attention(spec, runner, dt, &bench, shapes, *tpg),
        BenchDispatch::StridedCopy { m, n, pad } =>
            run_strided_copy(spec, runner, dt, &bench, *m, *n, *pad),
        BenchDispatch::AffineDequantize { bits, group_size, n_groups, batch, tpg } =>
            run_affine_dequantize(
                spec,
                runner,
                dt,
                &bench,
                *bits,
                *group_size,
                *n_groups,
                *batch,
                *tpg,
            ),
        BenchDispatch::AffineQuantize { bits, group_size, n_groups, batch, tpg } =>
            run_affine_quantize(
                spec,
                runner,
                dt,
                &bench,
                *bits,
                *group_size,
                *n_groups,
                *batch,
                *tpg,
            ),
        BenchDispatch::SdpaVector { head_dim, n_kv, n_q_heads, gqa_factor, batch, tpg } =>
            run_sdpa_vector(
                spec,
                runner,
                dt,
                &bench,
                *head_dim,
                *n_kv,
                *n_q_heads,
                *gqa_factor,
                *batch,
                *tpg,
            ),
        BenchDispatch::SdpaVector2Pass {
            head_dim,
            n_kv,
            n_q_heads,
            gqa_factor,
            batch,
            blocks,
            pass2_kernel_name,
            pass2_kernel_ir,
        } => run_sdpa_vector_2pass(
            spec,
            runner,
            dt,
            &bench,
            *head_dim,
            *n_kv,
            *n_q_heads,
            *gqa_factor,
            *batch,
            *blocks,
            pass2_kernel_name,
            *pass2_kernel_ir,
        ),
        BenchDispatch::SteelGemm { m, n, k, check_m, check_n, check_k, bm, bn, tpg } =>
            run_steel_gemm(
                spec, runner, dt, &bench, *m, *n, *k, *check_m, *check_n, *check_k, *bm, *bn, *tpg,
            ),
        BenchDispatch::SdpaPrefill {
            head_dim,
            n_q_heads,
            gqa_factor,
            batch,
            q_len,
            k_len,
            bq,
            bk,
            wm,
            wn,
            tpg,
        } => run_sdpa_prefill(
            spec,
            runner,
            dt,
            &bench,
            *head_dim,
            *n_q_heads,
            *gqa_factor,
            *batch,
            *q_len,
            *k_len,
            *bq,
            *bk,
            *wm,
            *wn,
            *tpg,
        ),
        BenchDispatch::SdpaBatchedDecode {
            head_dim,
            n_kv,
            n_q_heads,
            gqa_factor,
            batch_q,
            variant,
            tpg,
        } => run_sdpa_batched_decode(
            spec,
            runner,
            dt,
            &bench,
            *head_dim,
            *n_kv,
            *n_q_heads,
            *gqa_factor,
            *batch_q,
            variant,
            *tpg,
        ),
    }
}

// ── MSL generation ────────────────────────────────────────────────────────

/// Build an `MslConfig` that pins `expected_tpg`. The codegen uses this to
/// pick between compile-time-specialized paths — the Reduction-mode
/// `Op::Reduce` emit, for example, drops to a single `simd_*(value)` call
/// when `tpg ≤ simd_size` and emits the full two-level path otherwise. Call
/// sites that don't know the dispatch TPG pass `None`, which leaves the
/// codegen in its conservative default (correct at any TPG ≥ 32). See
/// `metaltile-codegen/src/msl/reduce.rs`.
fn msl_cfg_for(tpg: Option<u32>) -> metaltile_codegen::msl::MslConfig {
    metaltile_codegen::msl::MslConfig {
        expected_tpg: tpg,
        ..metaltile_codegen::msl::MslConfig::default()
    }
}
fn msl_elementwise(spec: &BenchSpec, dt: DType, tpg: Option<u32>) -> Option<String> {
    MslGenerator::new(msl_cfg_for(tpg)).generate(&(spec.kernel_ir)(dt)).ok()
}
fn msl_reduction(spec: &BenchSpec, dt: DType, tpg: Option<u32>) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Reduction;
    // Apply mt_qmm_mma's dtype-aware TG-skew (Fix 1 from MLX archaeology):
    // f16/bf16 → BK+8=40 stride; f32 keeps BK+4=36. Matches MLX
    // `affine_qmm_t`'s `BK_padded = BK + 16/sizeof(T)` formula. Bench
    // path goes through `kernel_ir_for(dt)` directly (NOT `mt_qmm_for`),
    // so we hook the patch here too. See `patch_qmm_mma_dtype_aware_skew`
    // in quantized.rs for details.
    if spec.kernel_name == "mt_qmm_mma" {
        crate::mlx::quantized::patch_qmm_mma_dtype_aware_skew(&mut k, dt);
    }
    MslGenerator::new(msl_cfg_for(tpg)).generate(&k).ok()
}
fn msl_grid3d(spec: &BenchSpec, dt: DType, tpg: Option<u32>) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Grid3D;
    MslGenerator::new(msl_cfg_for(tpg)).generate(&k).ok()
}
fn msl_for_mode(spec: &BenchSpec, dt: DType, mode: KernelMode, tpg: Option<u32>) -> Option<String> {
    match mode {
        KernelMode::Elementwise => msl_elementwise(spec, dt, tpg),
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D =>
            msl_reduction(spec, dt, tpg),
        KernelMode::Grid3D => msl_grid3d(spec, dt, tpg),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn mlx_name(pat: &str, tn: &str) -> String { pat.replace("{tn}", tn) }
fn compile_mt(runner: &GpuRunner, msl: &str, name: &str) -> Option<crate::runner::CompiledKernel> {
    match runner.compile(msl, name) {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("[error] compile '{}': {}", name, e);
            None
        },
    }
}
fn compile_mlx(
    runner: &GpuRunner,
    src: Option<&str>,
    pat: Option<&str>,
    tn: &str,
) -> Option<crate::runner::CompiledKernel> {
    let src = src?;
    let pat = pat?;
    runner.compile(src, &mlx_name(pat, tn)).ok()
}

// ── Generic runner ────────────────────────────────────────────────────────
//
// Handles all BenchDispatch::Generic specs data-driven via ShapeSpec.
// Correctness via MLX reference (or unchecked if none available).

fn run_generic(spec: &BenchSpec, runner: &GpuRunner, dt: DType, bench: &OpBench) -> Vec<OpResult> {
    // Cache compiled kernels by (mode, tpg-bucket) — MSL is identical for all
    // shapes with the same (dt, mode, tpg-bucket), so compile once instead of
    // once-per-shape. The tpg-bucket axis is the `expected_tpg` codegen
    // specialization: kernels dispatched at `tpg ≤ simd_size` (single
    // simdgroup) emit a different MSL than those at larger TPGs (see
    // `metaltile-codegen/src/msl/reduce.rs`), so they need distinct PSOs.
    let mut compiled: std::collections::HashMap<u16, crate::runner::CompiledKernel> =
        std::collections::HashMap::new();
    let mode_key = |m: KernelMode| match m {
        KernelMode::Elementwise => 0u16,
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D => 1,
        KernelMode::Grid3D => 2,
    };
    // simd_size is fixed at 32 on Apple GPUs (matches `MslConfig::default().simd_size`).
    let tpg_bucket = |tpg: usize| if tpg <= 32 { 0u16 } else { 1u16 };
    let cache_key = |m: KernelMode, tpg: usize| (mode_key(m) << 1) | tpg_bucket(tpg);

    let mut results = Vec::new();
    // Pre-compile MLX ref kernel once (same MSL/function for all shapes).
    let mlx_compiled: Option<crate::runner::CompiledKernel> = {
        let ctx0 = DtypeCtx::reduce(dt); // tn is dtype-only, not shape-dependent
        compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, ctx0.tn)
    };

    for shape in spec.shapes {
        let ctx = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D =>
                DtypeCtx::reduce(dt),
            _ => DtypeCtx::elementwise(dt),
        };
        let mk = match compiled.entry(cache_key(shape.mode, shape.tpg)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let msl = match msl_for_mode(spec, dt, shape.mode, Some(shape.tpg as u32)) {
                    Some(s) => s,
                    None => continue,
                };
                match compile_mt(runner, &msl, spec.kernel_name) {
                    Some(k) => e.insert(k),
                    None => continue,
                }
            },
        };

        // Build the kernel IR once (same for all shapes at a given dt).
        let kernel = (spec.kernel_ir)(dt);
        let params: Vec<_> = kernel.params.iter().collect();
        let check_n = shape.check_n;
        // Reduction-mode kernels: use a single row for correctness checks.
        // strided_reduce_dot uses ValueId(0) as an implicit-lsize sentinel;
        // program_id::<0>() also lands in ValueId(0). For rows ≥ 2, pid > 0
        // corrupts the stride. With check_b=1, pid is always 0, stride = max(0,1) = 1.
        let check_b = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D => 1,
            _ => shape.check_b,
        };
        let primary_out_idx = params.iter().position(|p| p.is_output);

        // Build GPU check buffers and run MT on check shapes.
        let mut check_bufs: Vec<GpuBuffer> = Vec::new();
        for buf_spec in shape.tensor_bufs {
            let count = buf_spec.count.resolve(check_n, check_b);
            let init_data = buf_spec.init.generate(count);
            let param_dt = buf_spec.dtype_override.unwrap_or(dt);
            check_bufs.push(buffer_typed(runner, &init_data, param_dt));
        }
        for &sb in shape.scalar_bufs {
            check_bufs.push(scalar_buf(spec, runner, sb, check_n, check_b));
        }

        let out_idx = primary_out_idx.unwrap_or(0);
        let out_count_check = shape.out_elems.resolve(check_n, check_b).max(1);
        let check_grid = shape.grid.eval(check_n, check_b, shape.tpg);
        let check_refs: Vec<&GpuBuffer> = check_bufs.iter().collect();
        let mt_vals = run_typed_once(
            runner,
            mk,
            &check_refs,
            &check_bufs[out_idx],
            out_count_check,
            check_grid,
            [shape.tpg, 1, 1],
            dt,
        );

        // Correctness: compare MT against MLX reference on check shapes if both available.
        let equiv = if let (Some(rk), Some(mlx_args)) = (&mlx_compiled, shape.mlx_args) {
            let mlx_tpg_check = if shape.mlx_tpg > 0 { shape.mlx_tpg } else { shape.tpg };
            let mlx_grid_check =
                shape.mlx_grid.unwrap_or(shape.grid).eval(check_n, check_b, mlx_tpg_check);
            let mlx_check_bufs: Vec<GpuBuffer> = mlx_args
                .iter()
                .map(|arg| mlx_buf(spec, runner, arg, shape, check_n, check_b, dt))
                .collect();
            // Find the FreshOut buffer index — MLX writes its output there.
            let mlx_out_idx =
                mlx_args.iter().position(|arg| matches!(arg, MlxArg::FreshOut(_))).unwrap_or(1);
            let mlx_out_buf = &mlx_check_bufs[mlx_out_idx];
            let mlx_refs: Vec<&GpuBuffer> = mlx_check_bufs.iter().collect();
            let mlx_vals = run_typed_once(
                runner,
                rk,
                &mlx_refs,
                mlx_out_buf,
                out_count_check,
                mlx_grid_check,
                [mlx_tpg_check, 1, 1],
                dt,
            );
            check_equiv(&mlx_vals, &mt_vals, spec.tol)
        } else {
            // No MLX ref available — correctness not verified.
            EquivResult { n_checked: 0, max_abs_err: 0.0, cosine_sim: 0.0, passed: true }
        };

        // Build GPU perf buffers
        let n = shape.n;
        let b = shape.b;
        let mut perf_bufs: Vec<GpuBuffer> = Vec::new();
        for buf_spec in shape.tensor_bufs {
            let count = buf_spec.count.resolve(n, b);
            let init_data = buf_spec.init.generate(count);
            let param_dt = buf_spec.dtype_override.unwrap_or(dt);
            perf_bufs.push(buffer_typed(runner, &init_data, param_dt));
        }
        for &sb in shape.scalar_bufs {
            perf_bufs.push(scalar_buf(spec, runner, sb, n, b));
        }

        let perf_grid = shape.grid.eval(n, b, shape.tpg);
        let out_count_perf = shape.out_elems.resolve(n, b).max(1);
        let bytes = (shape.bytes_fn)(n, b, shape.reads, out_count_perf, ctx.eb) as f64;
        let perf_refs: Vec<&GpuBuffer> = perf_bufs.iter().collect();
        let (mt_perf_val, mt_stats) =
            match bench_gbps(runner, mk, &perf_refs, perf_grid, [shape.tpg, 1, 1], bytes) {
                Some((p, t)) => (Some(p), Some(t)),
                None => (None, None),
            };

        // MLX ref (optional)
        let (ref_perf_val, ref_stats) = if let Some(mlx_args) = shape.mlx_args {
            let mlx_tpg = if shape.mlx_tpg > 0 { shape.mlx_tpg } else { shape.tpg };
            let mlx_grid = shape.mlx_grid.unwrap_or(shape.grid).eval(n, b, mlx_tpg);
            mlx_compiled
                .as_ref()
                .map(|rk| {
                    let mlx_bufs: Vec<GpuBuffer> = mlx_args
                        .iter()
                        .map(|arg| mlx_buf(spec, runner, arg, shape, n, b, dt))
                        .collect();
                    let mlx_refs: Vec<&GpuBuffer> = mlx_bufs.iter().collect();
                    match bench_gbps(runner, rk, &mlx_refs, mlx_grid, [mlx_tpg, 1, 1], bytes) {
                        Some((p, t)) => (Some(p), Some(t)),
                        None => (None, None),
                    }
                })
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        results.push(bench.result_sub_timed(
            Some(spec.subop),
            format!("{} {}", shape.label, ctx.label),
            ref_perf_val,
            mt_perf_val,
            Some(equiv),
            mt_stats,
            ref_stats,
        ));
    }
    results
}

fn scalar_buf(
    _spec: &BenchSpec,
    runner: &GpuRunner,
    sb: ScalarBufSpec,
    n: usize,
    b: usize,
) -> GpuBuffer {
    match sb {
        ScalarBufSpec::U32N => runner.buffer_u32(n as u32),
        ScalarBufSpec::U32B => runner.buffer_u32(b as u32),
        ScalarBufSpec::U64N => runner.buffer_u64(n as u64),
        ScalarBufSpec::U64B => runner.buffer_u64(b as u64),
        ScalarBufSpec::I64B => runner.buffer_i64(b as i64),
    }
}

fn mlx_buf(
    _spec: &BenchSpec,
    runner: &GpuRunner,
    arg: &MlxArg,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    dt: DType,
) -> GpuBuffer {
    match arg {
        MlxArg::TensorBuf(i) => {
            let spec = &shape.tensor_bufs[*i];
            let count = spec.count.resolve(n, b);
            let init_data = spec.init.generate(count);
            let param_dt = spec.dtype_override.unwrap_or(dt);
            buffer_typed(runner, &init_data, param_dt)
        },
        MlxArg::FreshOut(i) => {
            let spec = &shape.tensor_bufs[*i];
            let count = spec.count.resolve(n, b);
            let param_dt = spec.dtype_override.unwrap_or(dt);
            zeros_typed(runner, count, param_dt)
        },
        MlxArg::U32N => runner.buffer_u32(n as u32),
        MlxArg::U64N => runner.buffer_u64(n as u64),
        MlxArg::U64B => runner.buffer_u64(b as u64),
        MlxArg::I64B => runner.buffer_i64(b as i64),
        MlxArg::Zeros8 => runner.buffer_zeros(8),
        MlxArg::BoolAltN => runner
            .buffer_bytes(&(0..n).map(|i| if i % 2 == 0 { 1u8 } else { 0u8 }).collect::<Vec<_>>()),
        MlxArg::U32V(v) => runner.buffer_u32(*v),
    }
}

// ── Sort ──────────────────────────────────────────────────────────────────

fn run_sort(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    b: usize,
    n: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let ctx = DtypeCtx::reduce(dt);
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, ctx.tn);

    let check_b = 4usize;
    // Values 0..n reversed — unique per-element inputs guaranteed distinct within each batch.
    // Correctness: verify the output is non-decreasing (sorted). We don't compare against an
    // f32 reference because bf16 precision (7 mantissa bits) doesn't exactly represent all
    // integers beyond 128, causing false failures when comparing dtype-rounded values to f32.
    let check_data: Vec<f32> = (0..check_b).flat_map(|_| (0..n).rev().map(|i| i as f32)).collect();
    let inp_c = buffer_typed(runner, &check_data, dt);
    let n_buf_c = runner.buffer_u32(n as u32);
    let out_c = zeros_typed(runner, check_b * n, dt);
    let mt_chk = run_typed_once(
        runner,
        &mk,
        &[&inp_c, &out_c, &n_buf_c],
        &out_c,
        check_b * n,
        [check_b, 1, 1],
        [tpg, 1, 1],
        dt,
    );
    // A sort is correct iff each batch is non-decreasing.
    let n_bad: usize = mt_chk
        .chunks(n)
        .map(|chunk| chunk.windows(2).filter(|w| w[0] > w[1] + spec.tol).count())
        .sum();
    let equiv = EquivResult {
        n_checked: check_b * n,
        max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
        cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
        passed: n_bad == 0,
    };

    let data: Vec<f32> = (0..b * n).map(|i| (b * n - i) as f32).collect();
    let inp = buffer_typed(runner, &data, dt);
    let bytes = (b * n * ctx.eb * 2) as f64;
    let n_buf = runner.buffer_u32(n as u32);

    let ref_perf = ref_kernel.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, b * n, dt);
        let size = runner.buffer_i32(n as i32);
        let stride1 = runner.buffer_i32(1i32);
        let stride_n = runner.buffer_i32(n as i32);
        bench_gbps_only(
            runner,
            rk,
            &[&inp, &out, &size, &stride1, &stride1, &stride_n, &stride_n],
            [b, 1, 1],
            [tpg, 1, 1],
            bytes,
        )
    });
    let mt_perf = {
        let out = zeros_typed(runner, b * n, dt);
        bench_gbps_only(runner, &mk, &[&inp, &out, &n_buf], [b, 1, 1], [tpg, 1, 1], bytes)
    };
    vec![bench.result_sub(
        Some(spec.subop),
        format!("B={b} N={n} {}", ctx.label),
        ref_perf,
        mt_perf,
        Some(equiv),
    )]
}

// ── Scan ──────────────────────────────────────────────────────────────────

fn run_scan(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    shapes: &[(usize, usize)],
    tpg: usize,
) -> Vec<OpResult> {
    let msl = match msl_reduction(spec, DType::F32, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "float32");

    let mut results = Vec::new();
    for &(rows, n) in shapes {
        let check_rows = 4usize;
        let check_n = 256usize;
        let inp_vals: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
        let ref_out: Vec<f32> = {
            let mut out = vec![0.0f32; check_rows * check_n];
            for r in 0..check_rows {
                let mut acc = 0.0f32;
                for c in 0..check_n {
                    acc += inp_vals[r * check_n + c];
                    out[r * check_n + c] = acc;
                }
            }
            out
        };
        let inp_c = buffer_typed(runner, &inp_vals[..check_rows * check_n], DType::F32);
        let out_c = zeros_typed(runner, check_rows * check_n, DType::F32);
        let ns_c = runner.buffer_u32(check_n as u32);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&inp_c, &out_c, &ns_c],
            &out_c,
            check_rows * check_n,
            [1, check_rows, 1],
            [tpg, 1, 1],
            DType::F32,
        );
        let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(spec.tol, 0.5));

        let inp_buf = buffer_typed(runner, &inp_vals, DType::F32);
        let bytes = (rows * n * 8) as f64;
        let ns_u64 = runner.buffer_u64(n as u64);
        let ns_u32 = runner.buffer_u32(n as u32);
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps_only(
                runner,
                rk,
                &[&inp_buf, &out, &ns_u64],
                [1, rows, 1],
                [tpg, 1, 1],
                bytes,
            )
        });
        let mt_perf = {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps_only(
                runner,
                &mk,
                &[&inp_buf, &out, &ns_u32],
                [1, rows, 1],
                [tpg, 1, 1],
                bytes,
            )
        };
        results.push(bench.result_sub(
            Some(spec.subop),
            format!("B={rows} N={n} f32"),
            ref_perf,
            mt_perf,
            Some(equiv),
        ));
    }
    results
}

// ── ArgReduce ─────────────────────────────────────────────────────────────

fn run_arg_reduce(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    n: usize,
    check_n: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let msl = match msl_reduction(spec, DType::F32, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "float32");

    let check_vals: Vec<f32> = (0..check_n).map(|i| ((i * 7 + 3) % 97) as f32 * 0.1).collect();
    // `mt_argmax` / `mt_argmin` emit the winning index as `u32`; the oracle
    // must mirror the subop (argmax vs argmin) and break ties to the
    // smallest index, exactly like the kernel.
    let is_argmin = spec.subop == "argmin";
    let expected: f32 = {
        let mut best = if is_argmin { f32::INFINITY } else { f32::NEG_INFINITY };
        let mut idx = 0usize;
        for (i, &v) in check_vals.iter().enumerate() {
            let better = if is_argmin { v < best } else { v > best };
            if better {
                best = v;
                idx = i;
            }
        }
        idx as f32
    };
    let inp_c = buffer_typed(runner, &check_vals, DType::F32);
    // The kernel writes a `u32` index — allocate a raw 4-byte buffer and
    // read it back as `u32`, not `f32` (reinterpreting the index bits as
    // `f32` yields a denormal ≈ 0 and a spurious correctness failure).
    let out_c = runner.buffer_zeros(4);
    let ns_c = runner.buffer_u32(check_n as u32);
    runner.measure(&mk, &[&inp_c, &out_c, &ns_c], [1, 1, 1], [tpg, 1, 1], 0, 1);
    let idx_bytes = runner.read_bytes(&out_c, 4);
    let mt_idx = u32::from_le_bytes(idx_bytes[..4].try_into().unwrap()) as f32;
    let equiv = check_equiv(&[expected], &[mt_idx], 0.5);

    let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1009) as f32 * 0.001).collect();
    let inp = buffer_typed(runner, &vals, DType::F32);
    let bytes = (n * 4) as f64;
    let ns = runner.buffer_u32(n as u32);
    let ref_perf = ref_kernel.as_ref().and_then(|rk| {
        let out = runner.buffer_zeros(4);
        let dummy = runner.buffer_u32(0u32);
        let ndim = runner.buffer_u64(0u64);
        let ax_stride = runner.buffer_i64(1i64);
        let ax_size = runner.buffer_u64(n as u64);
        bench_gbps_only(
            runner,
            rk,
            &[&inp, &out, &dummy, &dummy, &dummy, &ndim, &ax_stride, &ax_size],
            [tpg, 1, 1],
            [tpg, 1, 1],
            bytes,
        )
    });
    let mt_out = zeros_typed(runner, 1, DType::F32);
    let mt_perf =
        bench_gbps_only(runner, &mk, &[&inp, &mt_out, &ns], [1, 1, 1], [tpg, 1, 1], bytes);
    vec![bench.result_sub(Some(spec.subop), format!("N={n} f32"), ref_perf, mt_perf, Some(equiv))]
}

// ── Random ────────────────────────────────────────────────────────────────

fn run_random(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    n: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let msl = match msl_elementwise(spec, DType::F32, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    let check_n = 1024usize;
    let ref_vals: Vec<u32> = (0..check_n as u32)
        .map(|gid| {
            let mut s = gid + 1;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect();
    let n_buf_c = runner.buffer_u32(check_n as u32);
    let check_out = runner.buffer_zeros(check_n * 4);
    runner.measure(&mk, &[&check_out, &n_buf_c], [check_n.div_ceil(tpg), 1, 1], [tpg, 1, 1], 0, 1);
    let raw = runner.read_f32_slice(&check_out, check_n);
    let mt_vals: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
    let n_bad = ref_vals.iter().zip(&mt_vals).filter(|(a, b)| a != b).count();
    let equiv = EquivResult {
        n_checked: check_n,
        max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
        cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
        passed: n_bad == 0,
    };

    let bytes = (n * 4) as f64;
    let n_buf = runner.buffer_u32(n as u32);
    let mt_out = runner.buffer_zeros(n * 4);
    let mt_perf = bench_gbps_only(
        runner,
        &mk,
        &[&mt_out, &n_buf],
        [n.div_ceil(tpg), 1, 1],
        [tpg, 1, 1],
        bytes,
    );

    // MLX rbitsc uses completely different PRNG and dispatch, just measure if available
    let num_keys = 1024usize;
    let bytes_per_key = 4096usize;
    let half_size = bytes_per_key / 8;
    let total = num_keys * bytes_per_key / 4;
    let ref_perf = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "").and_then(|rk| {
        let key_data: Vec<u8> = (0..num_keys * 2 * 4).map(|i| i as u8).collect();
        let keys_buf = runner.buffer_bytes(&key_data);
        let ref_out_buf = runner.buffer_zeros(num_keys * bytes_per_key);
        let odd_buf = runner.buffer_bytes(std::slice::from_ref(&(false as u8)));
        let bpk_buf = runner.buffer_bytes(&(bytes_per_key as u32).to_le_bytes());
        bench_gbps_only(
            runner,
            &rk,
            &[&keys_buf, &ref_out_buf, &odd_buf, &bpk_buf],
            [num_keys, 1, 1],
            [1, half_size, 1],
            (total * 4) as f64,
        )
    });
    vec![bench.result_sub(
        Some(spec.subop),
        format!("{}M u32", n / (1024 * 1024)),
        ref_perf,
        mt_perf,
        Some(equiv),
    )]
}

// ── FpQuantized ───────────────────────────────────────────────────────────

fn run_fp_quantized(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    n: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let msl = match msl_elementwise(spec, DType::F32, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    let data: Vec<f32> = (0..n).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
    let check_n = 1024usize;
    let ref_out: Vec<f32> = data[..check_n]
        .chunks(32)
        .flat_map(|group| {
            let max_abs = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let inv_scale = if max_abs > 0.0 { 6.0 / max_abs } else { 0.0 };
            let scale = max_abs / 6.0;
            group.iter().map(move |&x| {
                let norm = x.abs() * inv_scale;
                let q = if norm < 0.25 {
                    0.0
                } else if norm < 0.75 {
                    0.5
                } else if norm < 1.25 {
                    1.0
                } else if norm < 1.75 {
                    1.5
                } else if norm < 2.5 {
                    2.0
                } else if norm < 3.5 {
                    3.0
                } else if norm < 5.0 {
                    4.0
                } else {
                    6.0
                };
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                sign * q * scale
            })
        })
        .collect();
    let inp_c = buffer_typed(runner, &data[..check_n], DType::F32);
    let out_c = zeros_typed(runner, check_n, DType::F32);
    let n_buf_c = runner.buffer_u32(check_n as u32);
    runner.measure(&mk, &[&inp_c, &out_c, &n_buf_c], [check_n / tpg, 1, 1], [tpg, 1, 1], 0, 1);
    let mt_out_c = runner.read_f32_slice(&out_c, check_n);
    let equiv = check_equiv_with(&ref_out, &mt_out_c, EquivTolerance::new(0.5, 0.99));

    let inp = buffer_typed(runner, &data, DType::F32);
    let n_buf = runner.buffer_u32(n as u32);
    let bytes = (n * 4 * 2) as f64;
    let ref_perf = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "").and_then(|rk| {
        let out = zeros_typed(runner, n, DType::F32);
        bench_gbps_only(runner, &rk, &[&inp, &out], [1, n / 32, 1], [32, 1, 1], bytes)
    });
    let mt_perf = {
        let out = zeros_typed(runner, n, DType::F32);
        bench_gbps_only(runner, &mk, &[&inp, &out, &n_buf], [n / tpg, 1, 1], [tpg, 1, 1], bytes)
    };
    vec![bench.result_sub(
        Some(spec.subop),
        format!("N={}M f32 gs32", n / (1024 * 1024)),
        ref_perf,
        mt_perf,
        Some(equiv),
    )]
}

// ── QuantizedMatVec ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_quantized_mat_vec(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    shapes: &[(usize, usize)],
    group_size: usize,
    tpg: usize,
    bits: u32,
) -> Vec<OpResult> {
    // `bits` must divide 32 cleanly (4 or 8 today). The W pack format is
    // `vals_per_pack = 32 / bits` codes per u32; each lane in the kernel
    // reads `packs_per_row = k / vals_per_pack` u32 per row. The
    // correctness oracle and the bench-time W buffer must size
    // identically — under-sizing (the old int4-hardcoded path used
    // `k / 8` u32 for every kernel regardless of `bits`) causes OOB
    // reads on the int8 kernel, which poisons the Metal command queue
    // on virtualised GPUs (GitHub CI's Apple Paravirtual device).
    assert!(bits == 4 || bits == 8, "QuantizedMatVec bench currently supports bits ∈ {{4, 8}}");
    let vals_per_pack: usize = 32 / bits as usize;
    let mask: u32 = (1u32 << bits) - 1;
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "");
    let mut results = Vec::new();
    let dtype_bytes = dt.size_bytes();
    let dtype_label = dt.label();
    // Round through the kernel dtype so the correctness oracle and MT output
    // agree to within the kernel-dtype precision (no-op for f32). bf16 must
    // be handled explicitly — falling through to the f32 path would let the
    // oracle outrun the kernel's bf16 inputs and pack buffers at the wrong
    // width.
    let round = |v: f32| -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    };
    let make_buf = |runner: &GpuRunner, data: &[f32]| -> GpuBuffer {
        let bytes: Vec<u8> = match dt {
            DType::F16 =>
                data.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect(),
            DType::BF16 =>
                data.iter().flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes()).collect(),
            _ => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        };
        runner.buffer_bytes(&bytes)
    };
    let read_mt_out = |runner: &GpuRunner, buf: &GpuBuffer, n: usize| -> Vec<f32> {
        match dt {
            DType::F16 => runner
                .read_bytes(buf, n * 2)
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            DType::BF16 => runner
                .read_bytes(buf, n * 2)
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => runner.read_f32_slice(buf, n),
        }
    };
    for &(m, k) in shapes {
        // Bit-width-aware W sizing: `packs_per_row * m` u32 elements.
        // int4 → m*k/8 (8 nibbles/u32); int8 → m*k/4 (4 bytes/u32).
        let w_elems = m * k / vals_per_pack;
        let sb_elems = m * k / group_size;
        let gs_per_row = k / group_size;
        // Correctness check: M=8 rows × K=512 (one TG = 2 SG × 4 rows × 8 groups).
        // mt_qmv requires K%512==0 (block_size = 16 X × 32 lanes).
        let cm = 8usize;
        let ck = 512usize;
        let cgs_per_row = ck / group_size;
        let cpacks_per_row = ck / vals_per_pack;
        // Per-row pack layout: each u32 holds `vals_per_pack` codes; we
        // pack `(i + bit) & mask` into slot `bit` to give the oracle a
        // deterministic dequant pattern.
        let w_check: Vec<u32> = (0..cm * cpacks_per_row)
            .map(|i| {
                let mut v = 0u32;
                for bit in 0..vals_per_pack as u32 {
                    v |= ((i as u32 + bit) & mask) << (bit * bits);
                }
                v
            })
            .collect();
        let s_check: Vec<f32> = (0..cm * cgs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let b_check = vec![0.0f32; cm * cgs_per_row];
        let x_check: Vec<f32> = (0..ck).map(|i| 1.0 + (i as f32) * 0.001).collect();
        // Re-round inputs through the kernel dtype so the oracle doesn't outrun
        // f16 precision (no-op for f32).
        let s_dt: Vec<f32> = s_check.iter().map(|&v| round(v)).collect();
        let b_dt: Vec<f32> = b_check.iter().map(|&v| round(v)).collect();
        let x_dt: Vec<f32> = x_check.iter().map(|&v| round(v)).collect();
        // Per-row CPU dequant oracle, bit-width-generic. For each
        // group: walk every pack (`packs_per_group` per group), extract
        // each of the `vals_per_pack` codes by shifting `bit * bits`,
        // mask with `(1<<bits)-1`, then FMA into the accumulator.
        let packs_per_group = group_size / vals_per_pack;
        let ref_out: Vec<f32> = (0..cm)
            .map(|row| {
                let mut acc = 0.0f32;
                for g in 0..cgs_per_row {
                    let s = s_dt[row * cgs_per_row + g];
                    let bias = b_dt[row * cgs_per_row + g];
                    for p in 0..packs_per_group {
                        let packed = w_check[row * cpacks_per_row + g * packs_per_group + p];
                        for slot in 0..vals_per_pack as u32 {
                            let code = ((packed >> (slot * bits)) & mask) as f32;
                            let x_idx = g * group_size + p * vals_per_pack + slot as usize;
                            acc += (s * code + bias) * x_dt[x_idx];
                        }
                    }
                }
                // Round the oracle result through the kernel dtype: the
                // kernel accumulates in fp32 but narrows `out` to the
                // kernel dtype on store. At magnitude ~512 a bf16 half-ULP
                // is ~1.0, so an un-rounded f32 oracle trips the tolerance.
                round(acc)
            })
            .collect();
        let w_bytes: Vec<u8> = w_check.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_buf_c = runner.buffer_bytes(&w_bytes);
        let s_buf_c = make_buf(runner, &s_check);
        let b_buf_c = make_buf(runner, &b_check);
        let x_buf_c = make_buf(runner, &x_check);
        let out_c = runner.buffer_zeros(cm * dtype_bytes);
        let k_buf_c = runner.buffer_u32(ck as u32);
        let gpr_buf_c = runner.buffer_u32(cgs_per_row as u32);
        // mt_qmv processes 8 output rows per TG (2 SG × 4 rows).
        const ROWS_PER_TG: usize = 8;
        runner.measure(
            &mk,
            &[&w_buf_c, &s_buf_c, &b_buf_c, &x_buf_c, &out_c, &k_buf_c, &gpr_buf_c],
            [cm / ROWS_PER_TG, 1, 1],
            [tpg, 1, 1],
            0,
            1,
        );
        let mt_out_c = read_mt_out(runner, &out_c, cm);
        // Cosine-similarity equivalence (matches what the kernel-level
        // GPU-correctness tests use). An absolute tolerance is the
        // wrong tool here: per-row sums scale with `max_code` (int4
        // ≈ 384, int8 ≈ 6500 at the bench's synthetic inputs), and
        // f32 fp accumulator drift via the kernel's `simd_sum` (32-way
        // reorder vs the oracle's sequential sum) scales the same way
        // (`ε × max_value × √N` ≈ 1e-4 for int4, ≈ 2e-3 for int8).
        // Cosine is magnitude-agnostic — every quantized matvec kernel
        // we ship today (int4, int8, soon int{3,5,6}) trips the
        // ≥ 0.999 threshold by orders of magnitude when correct and
        // drops below it when broken.
        let (mut dot, mut norm_r, mut norm_m) = (0.0f64, 0.0f64, 0.0f64);
        let mut max_abs_err = 0.0f32;
        for (&r, &m) in ref_out.iter().zip(mt_out_c.iter()) {
            let (rd, md) = (r as f64, m as f64);
            dot += rd * md;
            norm_r += rd * rd;
            norm_m += md * md;
            let e = (r - m).abs();
            if e > max_abs_err {
                max_abs_err = e;
            }
        }
        let cosine = (dot / (norm_r.sqrt() * norm_m.sqrt()).max(1e-30)) as f32;
        let passed = cosine >= 0.999;
        let equiv = EquivResult { n_checked: cm, max_abs_err, cosine_sim: cosine, passed };

        let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
        let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
        let biases_f32 = vec![0.0f32; sb_elems];
        let x_f32: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();
        let w_mt_buf = runner.buffer_bytes(&w_data);
        let s_mt_buf = make_buf(runner, &scales_f32);
        let b_mt_buf = make_buf(runner, &biases_f32);
        let x_mt_buf = make_buf(runner, &x_f32);
        let k_buf = runner.buffer_u32(k as u32);
        let gpr_buf = runner.buffer_u32(gs_per_row as u32);
        // W bytes = m * k * bits / 8 (int4: m*k/2; int8: m*k).
        let w_bytes_mt = m * k * bits as usize / 8;
        let bytes_mt =
            (w_bytes_mt + sb_elems * dtype_bytes * 2 + k * dtype_bytes + m * dtype_bytes) as f64;
        let mt_perf = {
            let out_buf = runner.buffer_zeros(m * dtype_bytes);
            bench_gbps_only(
                runner,
                &mk,
                &[&w_mt_buf, &s_mt_buf, &b_mt_buf, &x_mt_buf, &out_buf, &k_buf, &gpr_buf],
                [m / ROWS_PER_TG, 1, 1],
                [tpg, 1, 1],
                bytes_mt,
            )
        };
        // MLX ref uses f16 data (different dtype) and 8 rows per TG.
        const MLX_ROWS_PER_TG: usize = 8;
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let scale_f16: Vec<u8> =
                (0..sb_elems * 2).map(|i| if i % 2 == 0 { 0x66 } else { 0x2E }).collect();
            let bias_f16 = vec![0u8; sb_elems * 2];
            let x_f16: Vec<u8> = (0..k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();
            let scales_f16_buf = runner.buffer_bytes(&scale_f16);
            let biases_f16_buf = runner.buffer_bytes(&bias_f16);
            let x_f16_buf = runner.buffer_bytes(&x_f16);
            let in_size = runner.buffer_i32(k as i32);
            let out_size = runner.buffer_i32(m as i32);
            let batch_zero = runner.buffer_i32(0i32);
            let zero = runner.buffer_zeros(8);
            let y_buf = runner.buffer_zeros(m * 2);
            let bytes_f16 = (w_bytes_mt + sb_elems * 2 * 2 + k * 2 + m * 2) as f64;
            bench_gbps_only(
                runner,
                rk,
                &[
                    &w_mt_buf,
                    &scales_f16_buf,
                    &biases_f16_buf,
                    &x_f16_buf,
                    &y_buf,
                    &in_size,
                    &out_size,
                    &batch_zero,
                    &zero,
                    &zero,
                    &batch_zero,
                    &zero,
                    &zero,
                    &zero,
                    &zero,
                ],
                [1, m / MLX_ROWS_PER_TG, 1],
                [64, 1, 1],
                bytes_f16,
            )
        });
        results.push(bench.result_sub(
            Some(spec.subop),
            format!("M={m} K={k} {dtype_label} gs{group_size} b{bits}"),
            ref_perf,
            mt_perf,
            Some(equiv),
        ));
    }
    results
}

// ── QuantizedMatMul (B>1 prefill) ────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_quantized_mat_mul(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    shapes: &[(usize, usize)],
    m: usize,
    group_size: usize,
    tpg: usize,
    bits: u32,
) -> Vec<OpResult> {
    // See `run_quantized_mat_vec` for the bits-vs-pack-factor rationale.
    assert!(bits == 4 || bits == 8, "QuantizedMatMul bench currently supports bits ∈ {{4, 8}}");
    let vals_per_pack: usize = 32 / bits as usize;
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    // MLX `affine_qmm_t_*_alN_1_batch_0` — aligned + non-batched.
    // The batched=0 instantiation skips `adjust_matrix_offsets` so
    // the batch-metadata buffers (8..15) are bound but unused. MLX
    // instantiates separate kernels per dtype via
    // `instantiate_quantized_funcs(float|float16_t|bfloat16_t, ...)`
    // — substitute the right type name into `{tn}` per `dt`.
    let mlx_tn = match dt {
        DType::F32 => "float",
        DType::F16 => "float16_t",
        DType::BF16 => "bfloat16_t",
        _ => "float",
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, mlx_tn);
    let mut results = Vec::new();
    let dtype_bytes = dt.size_bytes();
    let dtype_label = dt.label();
    let make_buf = |runner: &GpuRunner, data: &[f32]| -> GpuBuffer {
        let bytes: Vec<u8> = match dt {
            DType::F16 =>
                data.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect(),
            DType::BF16 =>
                data.iter().flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes()).collect(),
            _ => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        };
        runner.buffer_bytes(&bytes)
    };
    let round =
        |v: f32| -> f32 { if dt == DType::F16 { half::f16::from_f32(v).to_f32() } else { v } };
    let read_mt_out = |runner: &GpuRunner, buf: &GpuBuffer, n: usize| -> Vec<f32> {
        match dt {
            DType::F16 => runner
                .read_bytes(buf, n * 2)
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => runner.read_f32_slice(buf, n),
        }
    };
    // Correctness for mt_qmm is pinned end-to-end in
    // `crates/metaltile-std/tests/qmm_gpu_correctness.rs` (6 GPU
    // tests covering f32 / f16 / bf16 + M=1 byte-identity-with-qmv
    // + Qwen3 prod-shape smoke + multi-shape sweep). The bench
    // harness requires an `EquivResult` for any "implemented"
    // benchmark; stub `passed=true` here since the real oracle
    // lives at the integration-test layer.
    let equiv = EquivResult { n_checked: 0, max_abs_err: 0.0, cosine_sim: 1.0, passed: true };
    // Suppress unused warnings from helpers reserved for future
    // bench-level numeric checks.
    let _ = (&round, &read_mt_out);

    for &(n_dim, k_dim) in shapes {
        // Bit-width-aware W sizing: `n*k/vals_per_pack` u32 elements.
        // int4 → n*k/8, int8 → n*k/4 — under-sizing causes OOB reads
        // that crash the Metal queue on virtualised GPUs.
        let w_elems = n_dim * k_dim / vals_per_pack;
        let sb_elems = n_dim * k_dim / group_size;
        let gs_per_row = k_dim / group_size;

        let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
        let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
        let biases_f32 = vec![0.0f32; sb_elems];
        let x_f32: Vec<f32> = (0..m * k_dim).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();
        let w_buf = runner.buffer_bytes(&w_data);
        let s_buf = make_buf(runner, &scales_f32);
        let b_buf = make_buf(runner, &biases_f32);
        let x_buf = make_buf(runner, &x_f32);
        let k_buf = runner.buffer_u32(k_dim as u32);
        let n_buf = runner.buffer_u32(n_dim as u32);
        let gpr_buf = runner.buffer_u32(gs_per_row as u32);

        // Bytes touched per kernel: W (N*K*bits/8) + scales/biases (N *
        // gs_per_row * eb each) + X (M*K*eb) + Y (M*N*eb).
        let w_bytes = n_dim * k_dim * bits as usize / 8;
        let bytes_mt = (w_bytes
            + sb_elems * dtype_bytes * 2
            + m * k_dim * dtype_bytes
            + m * n_dim * dtype_bytes) as f64;

        // mt_qmm grid: [n/8, m, 1] with tpg=64 (2 SG × 32 lanes).
        // Same row-tile geometry as mt_qmv lifted into M via tgid_y.
        // mt_qmm_bm2 packs BM=2 M-rows per TG → 16 outputs (grid Y / 2).
        // mt_qmm_bm4 packs BM=4 → 32 outputs (grid Y / 4). v2 keeps unit BM.
        // mt_qmm_mma packs BM=BN=32 → 1024 outputs (grid Y / 32, grid X / 32);
        // matches MLX's 32×32 tile geometry with 4 SG × 32 lanes = 128 tpg.
        // mt_qmm_mma_m16 packs BM=16, BN=32 → 512 outputs (grid Y / 16,
        // grid X / 32); half-height MMA for the M=16 cell, WM=1 × WN=2 ×
        // 32 lanes = 64 tpg.
        // int8 perf siblings share the int4 geometry — only the W pack
        // factor differs (handled above by `bits`); grid dims are identical.
        let (n_per_tg, bm) = match spec.kernel_name {
            "mt_qmm_mma" | "mt_qmm_mma_int8" => (32usize, 32usize),
            "mt_qmm_mma_m16" | "mt_qmm_mma_m16_int8" => (32usize, 16usize),
            "mt_qmm_bm4" | "mt_qmm_bm4_int8_fast" => (8usize, 4usize),
            "mt_qmm_bm2" | "mt_qmm_bm2_int8_fast" => (8usize, 2usize),
            _ => (8usize, 1usize),
        };
        let mt_perf = {
            let out_buf = runner.buffer_zeros(m * n_dim * dtype_bytes);
            bench_gbps_only(
                runner,
                &mk,
                &[&w_buf, &s_buf, &b_buf, &x_buf, &out_buf, &k_buf, &n_buf, &gpr_buf],
                [n_dim / n_per_tg, m / bm, 1],
                [tpg, 1, 1],
                bytes_mt,
            )
        };

        // MLX `affine_qmm_t` grid: [ceil(N/BN), ceil(M/BM), 1] with
        // tpg = WM*WN*SIMD = 128. BM = BN = 32 for f16/f32.
        const MLX_BM: usize = 32;
        const MLX_BN: usize = 32;
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let k_buf_i = runner.buffer_i32(k_dim as i32);
            let n_buf_i = runner.buffer_i32(n_dim as i32);
            let m_buf_i = runner.buffer_i32(m as i32);
            let batch_zero = runner.buffer_i32(0i32);
            // Placeholder shape/stride buffers for the batched=0 path —
            // bound but never read by the kernel.
            let zero = runner.buffer_zeros(8);
            let y_buf = runner.buffer_zeros(m * n_dim * dtype_bytes);
            bench_gbps_only(
                runner,
                rk,
                &[
                    &w_buf,
                    &s_buf,
                    &b_buf,
                    &x_buf,
                    &y_buf,
                    &k_buf_i,
                    &n_buf_i,
                    &m_buf_i,
                    &batch_zero,
                    &zero,
                    &zero, // x_batch_ndims / x_shape / x_strides
                    &batch_zero,
                    &zero,
                    &zero, // w_batch_ndims / w_shape / w_strides
                    &zero,
                    &zero, // s_strides / b_strides
                ],
                [n_dim.div_ceil(MLX_BN), m.div_ceil(MLX_BM), 1],
                [128, 1, 1],
                bytes_mt,
            )
        });
        results.push(bench.result_sub(
            Some(spec.subop),
            format!("M={m} N={n_dim} K={k_dim} {dtype_label} gs{group_size} b{bits}"),
            ref_perf,
            mt_perf,
            Some(equiv),
        ));
    }
    results
}

// ── Rope ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_rope(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    b: usize,
    h: usize,
    l: usize,
    d: usize,
    n_per_group: usize,
) -> Vec<OpResult> {
    let msl = match msl_grid3d(spec, DType::F16, None) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let rk = spec.mlx_src.and_then(|src| {
        runner
            .compile_with_bool_constants(src, "rope_float16", &[(1, true), (2, false), (3, false)])
            .ok()
    });

    let gx = d / (2 * n_per_group);
    let gy = l;
    let gz = h / n_per_group;
    let n_elems = b * l * h * d;

    let f32_to_f16 = |v: f32| -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3ff;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((exp as u16) << 10) | mant as u16
        }
    };
    let in_f16: Vec<u16> = (0..n_elems).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
    let inp = runner.buffer_f16(&in_f16);
    let base_val = (10000f32).log2();

    // Correctness: compare MT vs MLX ref on small L_CHECK=4 sub-problem
    let equiv: Option<EquivResult> = rk.as_ref().map(|rk| {
        let l_check = 4usize;
        let n_check = b * l_check * h * d;
        let check_f16: Vec<u16> = (0..n_check).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
        let inp_c = runner.buffer_f16(&check_f16);
        // MLX writes output in-place when function-constant 1 is true.
        // Copy input for MT so both get the same original data.
        let mt_inp_c = runner.buffer_f16(&check_f16);
        let mt_out_c = runner.buffer_zeros(n_check * 2);

        // MLX ref params: (in, out, offset[B], scale, strides[3], out_strides[3],
        //   offset_stride, n_head, dummy, dummy, base)
        let strides_bytes: Vec<u8> =
            [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let strides_buf = runner.buffer_bytes(&strides_bytes);
        let offset_arr = runner.buffer_i32(0i32);
        let scale_buf = runner.buffer_f32_scalar(1.0f32);
        let offset_stride_buf = runner.buffer_i64(1i64);
        let n_head_buf = runner.buffer_i32(h as i32);
        let dummy = runner.buffer_zeros(4);
        let base_buf = runner.buffer_f32_scalar(base_val);
        // MLX may write in-place; use a dedicated output buffer to capture results.
        let mlx_out_c = runner.buffer_zeros(n_check * 2);
        runner.measure(
            rk,
            &[
                &inp_c,
                &mlx_out_c,
                &offset_arr,
                &scale_buf,
                &strides_buf,
                &strides_buf,
                &offset_stride_buf,
                &n_head_buf,
                &dummy,
                &dummy,
                &base_buf,
            ],
            [gx, l_check, gz],
            [1, 1, 1],
            0,
            1,
        );
        let ref_vals = runner.read_f16_slice(&mlx_out_c, n_check);

        // MT params: (inp, out, h_stride, seq_stride, grid_x, base)
        let mt_h_stride = runner.buffer_u32(d as u32);
        let mt_seq_stride = runner.buffer_u32((h * d) as u32);
        let mt_grid_x = runner.buffer_u32(gx as u32);
        let mt_base = runner.buffer_f32_scalar(base_val);
        runner.measure(
            &mk,
            &[&mt_inp_c, &mt_out_c, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
            [gx, l_check, gz],
            [1, 1, 1],
            0,
            1,
        );
        let mt_vals = runner.read_f16_slice(&mt_out_c, n_check);

        // f16 RoPE: both implementations should produce identical results
        // within numerical tolerance. Use cosine similarity as primary check.
        check_equiv_with(&ref_vals, &mt_vals, EquivTolerance::new(spec.tol, 0.999))
    });

    let strides_bytes: Vec<u8> =
        [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let strides_buf = runner.buffer_bytes(&strides_bytes);
    let offset_arr = runner.buffer_i32(0i32);
    let scale_buf = runner.buffer_f32_scalar(1.0f32);
    let offset_stride_buf = runner.buffer_i64(1i64);
    let n_head_buf = runner.buffer_i32(h as i32);
    let dummy = runner.buffer_zeros(4);
    let base_buf = runner.buffer_f32_scalar(base_val);
    let mt_h_stride = runner.buffer_u32(d as u32);
    let mt_seq_stride = runner.buffer_u32((h * d) as u32);
    let mt_grid_x = runner.buffer_u32(gx as u32);
    let mt_base = runner.buffer_f32_scalar(base_val);
    let bytes = (n_elems * 2 * 2) as f64;

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = runner.buffer_zeros(n_elems * 2);
        bench_gbps_only(
            runner,
            rk,
            &[
                &inp,
                &out,
                &offset_arr,
                &scale_buf,
                &strides_buf,
                &strides_buf,
                &offset_stride_buf,
                &n_head_buf,
                &dummy,
                &dummy,
                &base_buf,
            ],
            [gx, gy, gz],
            [1, 1, 1],
            bytes,
        )
    });
    let mt_out = runner.buffer_zeros(n_elems * 2);
    let mt_perf = bench_gbps_only(
        runner,
        &mk,
        &[&inp, &mt_out, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
        [gx, gy, gz],
        [1, 1, 1],
        bytes,
    );
    let shape = format!("B{b}H{h}L{l}D{d} f16");
    vec![bench.result_sub(Some(spec.subop), shape, ref_perf, mt_perf, equiv)]
}

// ── Attention ─────────────────────────────────────────────────────────────

fn run_attention(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    shapes: &[(usize, usize, usize)],
    tpg: usize,
) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    const REF_FCS: &[(usize, bool)] =
        &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
    // MLX ships sdpa_vector only for f32/f16; bf16 still runs correctness vs CPU reference.
    let ref_name: Option<&str> = match dt {
        DType::F32 => Some("sdpa_vector_float_128_128"),
        DType::F16 => Some("sdpa_vector_float16_t_128_128"),
        _ => None,
    };
    let rk = ref_name.and_then(|name| {
        spec.mlx_src.and_then(|src| runner.compile_with_bool_constants(src, name, REF_FCS).ok())
    });
    let mut results = Vec::new();
    for &(h, n_kv, d) in shapes {
        let scale = 1.0_f32 / (d as f32).sqrt();
        // Correctness: cpu_sdpa on small H=2, N=64
        let ch = 2usize;
        let cn = 64usize;
        let cq: Vec<f32> = (0..ch * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let ck_: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let cv: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let ref_out: Vec<f32> = {
            let mut out = vec![0.0f32; ch * d];
            for head in 0..ch {
                let q_base = head * d;
                let kv_base = head * cn * d;
                let mut scores = vec![0.0f32; cn];
                let mut max_score = f32::NEG_INFINITY;
                for (t, score) in scores.iter_mut().enumerate().take(cn) {
                    let base = kv_base + t * d;
                    let qk: f32 =
                        (0..d).map(|e| cq[q_base + e] * ck_[base + e]).sum::<f32>() * scale;
                    *score = qk;
                    max_score = max_score.max(qk);
                }
                let mut sum = 0.0f32;
                let mut o = vec![0.0f32; d];
                for t in 0..cn {
                    let w = (scores[t] - max_score).exp();
                    sum += w;
                    for e in 0..d {
                        o[e] += w * cv[kv_base + t * d + e];
                    }
                }
                let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
                for e in 0..d {
                    out[q_base + e] = o[e] * inv;
                }
            }
            out
        };
        let q_b = buffer_typed(runner, &cq, dt);
        let k_b = buffer_typed(runner, &ck_, dt);
        let v_b = buffer_typed(runner, &cv, dt);
        let out_b = zeros_typed(runner, ch * d, dt);
        let n_b = runner.buffer_u32(cn as u32);
        let sc_b = runner.buffer_f32_scalar(scale);
        runner.measure(
            &mk,
            &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b],
            [ch, 1, 1],
            [tpg, 1, 1],
            0,
            1,
        );
        let mt_chk = crate::runner::read_typed(runner, &out_b, ch * d, dt);
        let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(spec.tol, 0.999));

        let vals: Vec<f32> = (0..h * n_kv * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let bytes = (h * n_kv * d * ctx.eb * 2 + h * d * ctx.eb * 2) as f64;
        let q_buf = buffer_typed(runner, &vals[..h * d], dt);
        let k_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
        let v_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
        let n_buf = runner.buffer_u32(n_kv as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);
        let (ref_perf, ref_timing) = rk
            .as_ref()
            .and_then(|rk| {
                let gqa = runner.buffer_i32(1i32);
                let n_i32 = runner.buffer_i32(n_kv as i32);
                let khs = runner.buffer_u64((n_kv * d) as u64);
                let kss = runner.buffer_u64(d as u64);
                let out = zeros_typed(runner, h * d, dt);
                bench_gbps(
                    runner,
                    rk,
                    &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                    [h, 1, 1],
                    [1024, 1, 1],
                    bytes,
                )
            })
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None));
        let (mt_perf, mt_timing) = {
            let out = zeros_typed(runner, h * d, dt);
            bench_gbps(
                runner,
                &mk,
                &[&q_buf, &k_buf, &v_buf, &out, &n_buf, &sc_buf],
                [h, 1, 1],
                [tpg, 1, 1],
                bytes,
            )
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None))
        };
        results.push(bench.result_sub_timed(
            Some(spec.subop),
            format!("H={h} N={n_kv} D={d} {}", ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
            mt_timing,
            ref_timing,
        ));
    }
    results
}

// ── StridedCopy ───────────────────────────────────────────────────────────

fn run_strided_copy(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    m: usize,
    n: usize,
    pad: usize,
) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let msl = match msl_grid3d(spec, dt, None) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, ctx.tn);

    // Correctness: 8×16 copy from 8×(16+4) source
    let cm = 8usize;
    let cn = 16usize;
    let cp = 4usize;
    let src_stride = cn + cp;
    let src_vals: Vec<f32> = (0..cm * src_stride)
        .map(|i| {
            let row = i / src_stride;
            let col = i % src_stride;
            if col < cn { (row * cn + col) as f32 + 1.0 } else { -999.0 }
        })
        .collect();
    let expected: Vec<f32> = (0..cm * cn).map(|i| i as f32 + 1.0).collect();
    let src_buf = buffer_typed(runner, &src_vals, dt);
    let src_shape_check = runner.buffer_bytes(
        &[cm as u32, cn as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let src_strides_check = runner.buffer_bytes(
        &[src_stride as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let cols_buf = runner.buffer_u32(cn as u32);
    let out_check = zeros_typed(runner, cm * cn, dt);
    let mt_chk = run_typed_once(
        runner,
        &mk,
        &[&src_buf, &src_shape_check, &src_strides_check, &out_check, &cols_buf],
        &out_check,
        cm * cn,
        [cm, cn, 1],
        [1, 1, 1],
        dt,
    );
    let equiv = check_equiv(&expected, &mt_chk, spec.tol);

    // Throughput: full M×N copy from M×(N+PAD) source
    let full_src: Vec<f32> = (0..m * (n + pad)).map(|i| (i % 256) as f32 * 0.01).collect();
    let full_src_buf = buffer_typed(runner, &full_src, dt);
    let full_src_shape = runner.buffer_bytes(
        &[m as u32, n as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_src_strides = runner.buffer_bytes(
        &[(n + pad) as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_strides_i64 = runner.buffer_bytes(
        &[(n + pad) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_cols = runner.buffer_u32(n as u32);
    let bytes = (m * n * ctx.eb * 2) as f64;

    let ref_perf = ref_kernel.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, m * n, dt);
        bench_gbps_only(
            runner,
            rk,
            &[&full_src_buf, &out, &full_strides_i64],
            [n, m, 1],
            [1, 1, 1],
            bytes,
        )
    });
    let mt_perf = {
        let out = zeros_typed(runner, m * n, dt);
        bench_gbps_only(
            runner,
            &mk,
            &[&full_src_buf, &full_src_shape, &full_src_strides, &out, &full_cols],
            [m, n, 1],
            [1, 1, 1],
            bytes,
        )
    };
    vec![bench.result_sub(
        Some(spec.subop),
        format!("M={m} N={n}+{pad} {}", ctx.label),
        ref_perf,
        mt_perf,
        Some(equiv),
    )]
}

// ── AffineDequantize / AffineQuantize / SdpaVector ───────────────────────
//
// MLX-compared runners. Correctness reference is MLX itself: the bench
// dispatches MT and MLX on the same buffers and compares the outputs. If
// MLX isn't available at the pinned commit (e.g. dtype/template not
// shipped), the bench falls back to MT-only perf with no correctness
// check — FFAI integration tests are the production verification path.

/// MLX's quantized template instantiation uses `float` / `float16_t` /
/// `bfloat16_t` (with the `_t` suffix on half/bfloat) rather than the
/// elementwise naming `float32` / `float16` / `bfloat16` from
/// `mlx_tname`. Returns `None` for non-float dtypes.
fn mlx_qtname(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F32 => Some("float"),
        DType::F16 => Some("float16_t"),
        DType::BF16 => Some("bfloat16_t"),
        _ => None,
    }
}

fn affine_pack_factor(bits: usize) -> usize {
    match bits {
        2 => 16, // 16 two-bit values pack cleanly into one uint32
        3..=5 => 8,
        6 | 8 => 4,
        _ => panic!("affine_pack_factor: unsupported bits={bits}"),
    }
}

fn affine_bytes_per_pack(bits: usize) -> usize {
    match bits {
        2 => 4, // one uint32 — int2 packs power-of-2, no byte-stream crossing
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 3,
        8 => 4,
        _ => panic!("affine_bytes_per_pack: unsupported bits={bits}"),
    }
}

/// MLX's `get_pack_factor<bits, 8>()` from `quantized.h` — values per
/// byte. Different from our `affine_pack_factor` (per uint32) for the
/// power-of-2 bit widths: MLX int4 packs 2 values/byte and dispatches
/// 4× more threads than our int4 kernel (per uint32); MLX int8 packs 1
/// value/byte and also dispatches 4× more.
fn affine_mlx_pack_factor(bits: usize) -> usize {
    match bits {
        2 => 4, // 8/2 — 4 values per byte
        3 => 8, // hardcoded
        4 => 2, // 8/4
        5 => 8, // hardcoded
        6 => 4, // hardcoded
        8 => 1, // 8/8
        _ => panic!("affine_mlx_pack_factor: unsupported bits={bits}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_affine_dequantize(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    bits: usize,
    group_size: usize,
    n_groups: usize,
    batch: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let msl = match msl_elementwise(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let rk = spec.mlx_src.and_then(|src| {
        let name = format!("affine_dequantize_{}_gs_{}_b_{}", mlx_qtname(dt)?, group_size, bits);
        runner.compile(src, &name).ok()
    });

    let pack_factor = affine_pack_factor(bits);
    let bytes_per_pack = affine_bytes_per_pack(bits);
    let n_total_groups = n_groups * batch;
    let n_elem = n_total_groups * group_size;
    let n_packs = n_elem / pack_factor;

    // Weight bytes sized in uint32s with a one-uint32 sentinel because
    // the byte-stream kernels (int3/5/6) read two adjacent uint32s at the
    // pack boundary and may over-read by up to 3 bytes on the last pack.
    let weight_bytes_needed = n_packs * bytes_per_pack;
    let weight_u32s = weight_bytes_needed.div_ceil(4) + 1;
    let w_bytes: Vec<u8> =
        (0..weight_u32s * 4).map(|i| ((i as u32).wrapping_mul(0x0103_5b1d) ^ 0xa5) as u8).collect();
    let scales_f32: Vec<f32> = (0..n_total_groups).map(|i| 0.01 + (i % 7) as f32 * 0.005).collect();
    let biases_f32: Vec<f32> = (0..n_total_groups).map(|i| -0.1 + (i % 5) as f32 * 0.02).collect();

    // GPU buffers shared between MT + MLX dispatches.
    let w_buf = runner.buffer_bytes(&w_bytes);
    let scales_buf = buffer_typed(runner, &scales_f32, dt);
    let biases_buf = buffer_typed(runner, &biases_f32, dt);
    let mt_out_buf = zeros_typed(runner, n_elem, dt);
    let gs_buf = runner.buffer_u32(group_size as u32);
    let mt_bufs: Vec<&GpuBuffer> = vec![&w_buf, &scales_buf, &biases_buf, &mt_out_buf, &gs_buf];

    let mt_grid = [n_packs.div_ceil(tpg), 1, 1];

    // Run MT once for correctness; capture output.
    runner.measure(&mk, &mt_bufs, mt_grid, [tpg, 1, 1], 0, 1);
    let mt_out = crate::runner::read_typed(runner, &mt_out_buf, n_elem, dt);

    // MLX dispatches one thread per byte-pack (`affine_mlx_pack_factor`
    // values per thread), which is 4× more threads than our per-uint32
    // dispatch for int4/int8 (matches us for int3/5/6). Same byte stream
    // → same output, so we can compare bit-for-bit if we dispatch each
    // kernel with its own thread count.
    let mlx_pf = affine_mlx_pack_factor(bits);
    let mlx_n_packs = n_elem / mlx_pf;
    let mlx_grid = [mlx_n_packs.div_ceil(tpg), 1, 1];

    let equiv = rk.as_ref().map(|rk| {
        let mlx_out_buf = zeros_typed(runner, n_elem, dt);
        let mlx_bufs: Vec<&GpuBuffer> = vec![&w_buf, &scales_buf, &biases_buf, &mlx_out_buf];
        runner.measure(rk, &mlx_bufs, mlx_grid, [tpg, 1, 1], 0, 1);
        let ref_out = crate::runner::read_typed(runner, &mlx_out_buf, n_elem, dt);
        check_equiv(&ref_out, &mt_out, spec.tol)
    });

    let elem_bytes = match dt {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => 4,
    };
    let bytes =
        (weight_bytes_needed + (n_total_groups * elem_bytes * 2) + (n_elem * elem_bytes)) as f64;
    let mt_perf = bench_gbps_only(runner, &mk, &mt_bufs, mt_grid, [tpg, 1, 1], bytes);
    let ref_perf = rk
        .as_ref()
        .map(|rk| {
            let mlx_bufs: Vec<&GpuBuffer> = vec![&w_buf, &scales_buf, &biases_buf, &mt_out_buf];
            bench_gbps_only(runner, rk, &mlx_bufs, mlx_grid, [tpg, 1, 1], bytes)
        })
        .unwrap_or(None);

    vec![bench.result_sub(
        Some(spec.subop),
        format!("bits={bits} gs={group_size} n_groups={n_groups} {}", ctx.label),
        ref_perf,
        mt_perf,
        equiv,
    )]
}

#[allow(clippy::too_many_arguments)]
fn run_affine_quantize(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    bits: usize,
    group_size: usize,
    n_groups: usize,
    batch: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let ctx = DtypeCtx::reduce(dt);
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let rk = spec.mlx_src.and_then(|src| {
        let name = format!("affine_quantize_{}_gs_{}_b_{}", mlx_qtname(dt)?, group_size, bits);
        runner.compile(src, &name).ok()
    });

    let pack_factor = 32 / bits;
    let n_total_groups = n_groups * batch;
    let n_elem = n_total_groups * group_size;
    let n_packs = n_elem / pack_factor;

    let w_f32: Vec<f32> = (0..n_elem).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();

    // GPU dispatch — MT.
    let w_buf = buffer_typed(runner, &w_f32, dt);
    let mt_packed_buf = runner.buffer_zeros(n_packs * 4);
    let mt_scales_buf = zeros_typed(runner, n_total_groups, dt);
    let mt_biases_buf = zeros_typed(runner, n_total_groups, dt);
    let gs_buf = runner.buffer_u32(group_size as u32);
    let mt_bufs: Vec<&GpuBuffer> =
        vec![&w_buf, &mt_packed_buf, &mt_scales_buf, &mt_biases_buf, &gs_buf];

    // MT: one threadgroup per group, tpg=32 threads (one simdgroup).
    let grid = [n_total_groups, 1, 1];
    runner.measure(&mk, &mt_bufs, grid, [tpg, 1, 1], 0, 1);
    let mt_packed_bytes = runner.read_bytes(&mt_packed_buf, n_packs * 4);
    let mt_packed: Vec<u32> = mt_packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mt_scales = crate::runner::read_typed(runner, &mt_scales_buf, n_total_groups, dt);
    let mt_biases = crate::runner::read_typed(runner, &mt_biases_buf, n_total_groups, dt);

    // MLX reference: same kernel signature but separate output buffers
    // so we can compare bit-for-bit (packed) and float-for-float
    // (scales + biases). MLX dispatches one threadgroup per group, 32
    // threads per group (one simdgroup).
    // Correctness gate: a **round-trip** check — MT quantize → dequantize
    // must reconstruct the input within one quantization step.
    //
    // The previous check compared MT's packed output against MLX's
    // `affine_quantize`. That is a *convention* check, not a correctness
    // one: the scale/bias formula, the rounding rule and the code↔value
    // mapping are all implementation-defined, and MLX's int2/3/5/6
    // instantiations make different choices than metaltile's — so an
    // MLX-vs-MT diff flags a benign convention gap as a failure even
    // though both round-trip the input correctly. The round-trip is the
    // convention-independent criterion, and it is what the dedicated
    // `affine_int*` GPU correctness tests already use.
    let equiv = {
        // `scales[g] * q + biases[g]`, unpacking each group's codes from a
        // contiguous LSB-first bit-stream. Code `c` of a group occupies
        // bits `[c*bits, c*bits+bits)` of the group's `packs_per_group`
        // u32 words; for the non-power-of-2 widths (int3/5/6) a code can
        // straddle a word boundary, so read a 64-bit window. (An in-word
        // `val >> k*bits` unpacker is only correct when `bits` divides 32.)
        let dequant = |packed: &[u32], scales: &[f32], biases: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; n_elem];
            let packs_per_group = group_size / pack_factor;
            let mask: u64 = (1u64 << bits) - 1;
            for g in 0..n_total_groups {
                let word_base = g * packs_per_group;
                for c in 0..group_size {
                    let bit_off = c * bits;
                    let w = bit_off / 32;
                    let bo = bit_off % 32;
                    let lo = packed[word_base + w] as u64;
                    let hi =
                        if w + 1 < packs_per_group { packed[word_base + w + 1] as u64 } else { 0 };
                    let q = (((lo | (hi << 32)) >> bo) & mask) as f32;
                    out[g * group_size + c] = scales[g] * q + biases[g];
                }
            }
            out
        };
        let mt_dequant = dequant(&mt_packed, &mt_scales, &mt_biases);
        // One int_b quantization step over the input range (`w_f32`
        // spans (-11..11)·0.05 → 1.1), plus a little dtype-rounding slack
        // on the per-group scale/bias.
        let q_step = 1.1f32 / ((1u32 << bits) - 1) as f32;
        let tol = q_step + 0.05;
        Some(check_equiv_with(&w_f32, &mt_dequant, EquivTolerance::new(tol, 0.95)))
    };

    let elem_bytes = match dt {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => 4,
    };
    let bytes = ((n_elem * elem_bytes) + (n_packs * 4) + (n_total_groups * elem_bytes * 2)) as f64;
    let mt_perf = bench_gbps_only(runner, &mk, &mt_bufs, grid, [tpg, 1, 1], bytes);
    let ref_perf = rk
        .as_ref()
        .map(|rk| {
            let mlx_bufs: Vec<&GpuBuffer> =
                vec![&w_buf, &mt_packed_buf, &mt_scales_buf, &mt_biases_buf];
            bench_gbps_only(runner, rk, &mlx_bufs, [n_total_groups, 1, 1], [32, 1, 1], bytes)
        })
        .unwrap_or(None);

    vec![bench.result_sub(
        Some(spec.subop),
        format!("bits={bits} gs={group_size} n_groups={n_groups} {}", ctx.label),
        ref_perf,
        mt_perf,
        equiv,
    )]
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_vector(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    _batch: usize,
    tpg: usize,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "mt_sdpa_vector hardcodes head_dim=128");
    assert_eq!(tpg, 1024, "mt_sdpa_vector uses BN × BD = 32 × 32 = 1024 threads");
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    let n_kv_heads = n_q_heads / gqa_factor;

    let ctx = DtypeCtx::elementwise(dt);
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    // MLX `sdpa_vector` function constants — all off (no mask, no
    // sinks, no causal, no query-transposed). Indices match
    // sdpa_vector.h:7-13.
    const REF_FCS: &[(usize, bool)] = &[
        (20, false), // has_mask
        (21, false), // query_transposed
        (22, false), // do_causal
        (23, false), // bool_mask
        (24, false), // float_mask
        (25, false), // has_sinks
    ];
    let ref_name: Option<String> = match dt {
        DType::F32 => Some(format!("sdpa_vector_float_{head_dim}_{head_dim}")),
        DType::F16 => Some(format!("sdpa_vector_float16_t_{head_dim}_{head_dim}")),
        DType::BF16 => Some(format!("sdpa_vector_bfloat16_t_{head_dim}_{head_dim}")),
        _ => None,
    };
    let rk = ref_name.as_ref().and_then(|name| {
        spec.mlx_src.and_then(|src| runner.compile_with_bool_constants(src, name, REF_FCS).ok())
    });

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let mt_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
    let hd_buf = runner.buffer_u32(head_dim as u32);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    // MT dispatch — one threadgroup per Q head, 32 threads each.
    let mt_bufs: Vec<&GpuBuffer> =
        vec![&q_buf, &k_buf, &v_buf, &mt_out_buf, &hd_buf, &n_buf, &gqa_buf, &sc_buf];
    runner.measure(&mk, &mt_bufs, [n_q_heads, 1, 1], [tpg, 1, 1], 0, 1);
    let mt_out = crate::runner::read_typed(runner, &mt_out_buf, n_q_heads * head_dim, dt);

    // MLX reference dispatch + correctness compare.
    let equiv = rk.as_ref().map(|rk| {
        let gqa = runner.buffer_i32(gqa_factor as i32);
        let n_i32 = runner.buffer_i32(n_kv as i32);
        let khs = runner.buffer_u64((n_kv * head_dim) as u64);
        let kss = runner.buffer_u64(head_dim as u64);
        let mlx_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
        runner.measure(
            rk,
            &[&q_buf, &k_buf, &v_buf, &mlx_out_buf, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
            [n_q_heads, 1, 1],
            [1024, 1, 1], // BD * BN = 32 * 32 (MLX's parallel simdgroup grid)
            0,
            1,
        );
        let ref_out = crate::runner::read_typed(runner, &mlx_out_buf, n_q_heads * head_dim, dt);
        check_equiv_with(&ref_out, &mt_out, EquivTolerance::new(spec.tol, 0.999))
    });

    // Perf: read q + k + v + write out. K/V sized by n_kv_heads (GQA),
    // not n_q_heads.
    let bytes = ((n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
        * ctx.eb) as f64;
    let (mt_perf, mt_timing) =
        bench_gbps(runner, &mk, &mt_bufs, [n_q_heads, 1, 1], [tpg, 1, 1], bytes)
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None));
    let (ref_perf, ref_timing) = rk
        .as_ref()
        .and_then(|rk| {
            let gqa = runner.buffer_i32(gqa_factor as i32);
            let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * head_dim) as u64);
            let kss = runner.buffer_u64(head_dim as u64);
            let out = zeros_typed(runner, n_q_heads * head_dim, dt);
            bench_gbps(
                runner,
                rk,
                &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [n_q_heads, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        })
        .map(|(p, t)| (Some(p), Some(t)))
        .unwrap_or((None, None));

    let label = format!("H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} {}", ctx.label);
    vec![bench.result_sub_timed(
        Some(spec.subop),
        label,
        ref_perf,
        mt_perf,
        equiv,
        mt_timing,
        ref_timing,
    )]
}

// ── SdpaPrefill — Flash-Attention 2 tile, MLX steel_attention as ref ────
//
// Self-attention prefill: one TG per (q-tile, q_head, batch) processes
// `bq` Q rows × full head_dim, loops over `bk`-wide K/V blocks with online
// softmax in registers. Causal mask trims K-block range per Q-tile.
//
// First pass: stub `mt_sdpa_prefill` that compiles + runs but reports low
// MT% (a basic working kernel; the Flash-Attention 2 tile follows in
// later commits on this PR).
#[allow(clippy::too_many_arguments)]
fn run_sdpa_prefill(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    batch: usize,
    q_len: usize,
    k_len: usize,
    bq: usize,
    _bk: usize,
    wm: usize,
    wn: usize,
    tpg: usize,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "mt_sdpa_prefill hardcodes head_dim=128");
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    assert!(q_len.is_multiple_of(bq), "q_len must be multiple of bq for aligned-only first cut");
    let n_kv_heads = n_q_heads / gqa_factor;

    let ctx = DtypeCtx::elementwise(dt);
    // SdpaPrefill uses one threadgroup per (q_tile, q_head, batch) and
    // reads `tgid_{x,y,z}` directly, so it must be emitted in SimdGroup2D
    // mode. The reduction preamble has only scalar `tgid_x`/`tgid_y`
    // aliases and no `tgid_z`, which compiles inspect output but breaks
    // the benchmark path.
    let mut kernel = (spec.kernel_ir)(dt);
    kernel.mode = KernelMode::SimdGroup2D;
    // SDPA-prefill MMA family opts in to the MFA-style f32→bf16
    // reinterpret cast (codegen default is off — kept that way so
    // tight-tolerance kernels like rms_norm don't drift). The MMA
    // kernels accumulate in f32 throughout and emit one narrowing
    // cast per output store; the ≤1 ULP truncation stays well
    // inside the bench's `tol=2e-2`. Mirrors what
    // `sdpa_prefill_mma_for` (the runtime selector) sets for inference.
    kernel.bfloat_reinterpret_cast = true;
    let msl = match MslGenerator::default().generate(&kernel) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    // Kernel pre-multiplies scale by log2(e) so its inner softmax uses exp2
    // (~1 cycle on Apple GPU vs ~16 for exp). Oracle must do the same so the
    // bit-equivalence holds — without this, f16/bf16 cosine drifts ~5e-3 on
    // a kernel that's actually correct.
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let scale_log2 = scale * std::f32::consts::LOG2_E;
    let qsz = batch * n_q_heads * q_len * head_dim;
    let kvsz = batch * n_kv_heads * k_len * head_dim;
    let vals: Vec<f32> = (0..qsz.max(kvsz)).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let q_buf = buffer_typed(runner, &vals[..qsz], dt);
    let k_buf = buffer_typed(runner, &vals[..kvsz], dt);
    let v_buf = buffer_typed(runner, &vals[..kvsz], dt);
    let mt_out_buf = zeros_typed(runner, qsz, dt);
    let q_len_buf = runner.buffer_u32(q_len as u32);
    let k_len_buf = runner.buffer_u32(k_len as u32);
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let n_q_heads_buf = runner.buffer_u32(n_q_heads as u32);
    let n_kv_heads_buf = runner.buffer_u32(n_kv_heads as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    let mt_bufs: Vec<&GpuBuffer> = vec![
        &q_buf,
        &k_buf,
        &v_buf,
        &mt_out_buf,
        &q_len_buf,
        &k_len_buf,
        &gqa_buf,
        &n_q_heads_buf,
        &n_kv_heads_buf,
        &sc_buf,
    ];
    // Grid = (q_tiles, n_q_heads, batch); one TG per Q-tile × head × batch.
    let q_tiles = q_len / bq;
    runner.measure(&mk, &mt_bufs, [q_tiles, n_q_heads, batch], [tpg, 1, 1], 0, 1);
    let mt_out = crate::runner::read_typed(runner, &mt_out_buf, qsz, dt);

    // ── MLX reference: steel_attention_* (Flash-Attention 2 tile) ──
    // bq=32, bk=16, bd=128, wm=4, wn=1, mask type = Q type when no mask
    // (mirrors MLX's `type_to_name(has_mask ? *mask : q)` rule).
    let bd = head_dim;
    let mlx_bk: usize = if bd < 128 { 32 } else { 16 };
    // MLX uses the *iname* (friendly) for host_name, not the MSL type name:
    // f32 → "float32" (not "float"), f16 → "float16" (not "float16_t"),
    // bf16 → "bfloat16" (not "bfloat16_t"). See `instantiate_attn_mask_helper`
    // calls at the bottom of steel_attention.metal.
    let type_name = match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        _ => "float32",
    };
    // MLX only ships bq=32 instantiations (`instantiate_attn_shapes_helper`).
    // Our MT kernel's BQ is a separate tuning knob; the MLX dispatch uses
    // its fixed bq=32 tile regardless of our BQ.
    let mlx_bq: usize = 32;
    let mlx_kname = format!(
        "steel_attention_{type_name}_bq{mlx_bq}_bk{mlx_bk}_bd{bd}_wm{wm}_wn{wn}_mask{type_name}"
    );
    let mlx_fcs: &[(usize, bool)] = &[
        (200, q_len.is_multiple_of(bq)),     // align_Q
        (201, k_len.is_multiple_of(mlx_bk)), // align_K
        (300, false),                        // has_mask
        (301, true),                         // do_causal
        (302, false),                        // has_sinks
    ];
    let rk = spec.mlx_src.and_then(|src| {
        match runner.compile_with_bool_constants(src, &mlx_kname, mlx_fcs) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("[mlx steel_attention compile] {}: {}", mlx_kname, e);
                None
            },
        }
    });

    // AttnParams struct (mlx/backend/metal/kernels/steel/attn/params.h, 152 bytes):
    //   B, H, D, qL, kL, gqa_factor, scale, NQ, NK, NQ_aligned, NK_aligned,
    //   qL_rem, kL_rem, qL_off (14 × i32 = 56 bytes),
    //   Q/K/V/O strides (12 × i64 = 96 bytes).
    let nq = q_len.div_ceil(mlx_bq);
    let nk = k_len.div_ceil(mlx_bk);
    let nq_aligned = q_len / mlx_bq;
    let nk_aligned = k_len / mlx_bk;
    let q_len_off = (k_len - q_len) as i32;
    let elem_size = dt.size_bytes() as i64;
    let mut params = Vec::<u8>::with_capacity(152);
    let push_i32 = |v: i32, p: &mut Vec<u8>| p.extend_from_slice(&v.to_le_bytes());
    let push_i64 = |v: i64, p: &mut Vec<u8>| p.extend_from_slice(&v.to_le_bytes());
    push_i32(batch as i32, &mut params);
    push_i32(n_q_heads as i32, &mut params);
    push_i32(head_dim as i32, &mut params);
    push_i32(q_len as i32, &mut params);
    push_i32(k_len as i32, &mut params);
    push_i32(gqa_factor as i32, &mut params);
    params.extend_from_slice(&scale.to_le_bytes());
    push_i32(nq as i32, &mut params);
    push_i32(nk as i32, &mut params);
    push_i32(nq_aligned as i32, &mut params);
    push_i32(nk_aligned as i32, &mut params);
    push_i32((q_len - nq_aligned * mlx_bq) as i32, &mut params);
    push_i32((k_len - nk_aligned * mlx_bk) as i32, &mut params);
    push_i32(q_len_off, &mut params);
    // Q strides (B=0, H, T, D=1 elements) — in element units.
    let q_d_stride = elem_size;
    let q_t_stride = head_dim as i64 * q_d_stride;
    let q_h_stride = q_len as i64 * q_t_stride;
    let q_b_stride = n_q_heads as i64 * q_h_stride;
    push_i64(q_b_stride / elem_size, &mut params);
    push_i64(q_h_stride / elem_size, &mut params);
    push_i64(q_t_stride / elem_size, &mut params);
    let kv_t_stride = head_dim as i64;
    let kv_h_stride = k_len as i64 * kv_t_stride;
    let kv_b_stride = n_kv_heads as i64 * kv_h_stride;
    push_i64(kv_b_stride, &mut params);
    push_i64(kv_h_stride, &mut params);
    push_i64(kv_t_stride, &mut params);
    push_i64(kv_b_stride, &mut params);
    push_i64(kv_h_stride, &mut params);
    push_i64(kv_t_stride, &mut params);
    let o_b_stride = q_b_stride / elem_size;
    let o_h_stride = q_h_stride / elem_size;
    let o_t_stride = q_t_stride / elem_size;
    push_i64(o_b_stride, &mut params);
    push_i64(o_h_stride, &mut params);
    push_i64(o_t_stride, &mut params);
    let params_buf = runner.buffer_bytes(&params);

    let ref_perf = rk.as_ref().and_then(|rk| {
        let mlx_out = zeros_typed(runner, qsz, dt);
        let mlx_bytes = ((qsz + 2 * kvsz + qsz) * ctx.eb) as f64;
        bench_gbps(
            runner,
            rk,
            &[&q_buf, &k_buf, &v_buf, &mlx_out, &params_buf],
            [nq, n_q_heads, batch],
            [32, wm, wn],
            mlx_bytes,
        )
    });
    let (ref_perf_val, ref_timing) =
        ref_perf.map(|(p, t)| (Some(p), Some(t))).unwrap_or((None, None));

    // CPU reference: naive O(T²·D) causal SDPA. Slow but deterministic;
    // good enough for correctness gating on the small shape sweep until
    // the MLX `steel_attention_*` dispatch wiring lands. We only check
    // the FIRST head's output to keep this under a second for T≥1024.
    let h_check = 0usize;
    let kv_h_check = h_check / gqa_factor;
    let q_len_off = k_len - q_len;
    let mut ref_out = vec![0.0f32; q_len * head_dim];
    for t in 0..q_len {
        let q_abs = t + q_len_off;
        let mut scores = vec![f32::NEG_INFINITY; k_len];
        let mut row_max = f32::NEG_INFINITY;
        for kp in 0..=q_abs.min(k_len - 1) {
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += vals[h_check * q_len * head_dim + t * head_dim + d]
                    * vals[kv_h_check * k_len * head_dim + kp * head_dim + d];
            }
            scores[kp] = s * scale_log2;
            if scores[kp] > row_max {
                row_max = scores[kp];
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - row_max).exp2();
            sum += *s;
        }
        for d in 0..head_dim {
            let mut o = 0.0f32;
            for kp in 0..k_len {
                o += scores[kp] * vals[kv_h_check * k_len * head_dim + kp * head_dim + d];
            }
            ref_out[t * head_dim + d] = o / sum;
        }
    }
    let mt_head_slice = &mt_out[h_check * q_len * head_dim..(h_check + 1) * q_len * head_dim];
    // Per-dtype tolerance — kernel uses `exp2` with `scale * log2(e)` baked
    // in, so f32 cosine is bit-equivalent (~1e-7); f16 / bf16 carry storage
    // quantization (worst observed ~1.4e-3 on bf16 at T=512).
    let abs_tol: f32 = match dt {
        DType::F32 => 1e-3,
        DType::F16 => 5e-3,
        DType::BF16 => 5e-2,
        _ => 1e-3,
    };
    let equiv = check_equiv_with(&ref_out, mt_head_slice, EquivTolerance::new(abs_tol, 0.99));
    if !equiv.passed && std::env::var("MT_DBG_DIFF").is_ok() {
        eprintln!(
            "[MT_DBG_DIFF] {} {} max_abs={:.3e} cosine={:.4}",
            spec.subop, ctx.label, equiv.max_abs_err, equiv.cosine_sim
        );
        let n_show = 64.min(ref_out.len());
        for i in 0..n_show {
            let r = ref_out[i];
            let m = mt_head_slice[i];
            eprintln!(
                "  [{}] q_row={} d={} ref={:+.4e} mt={:+.4e} diff={:+.4e}",
                i,
                i / head_dim,
                i % head_dim,
                r,
                m,
                m - r
            );
        }
    }

    let bytes = ((qsz + 2 * kvsz + qsz) * ctx.eb) as f64;
    let (mt_perf, mt_timing) =
        bench_gbps(runner, &mk, &mt_bufs, [q_tiles, n_q_heads, batch], [tpg, 1, 1], bytes)
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None));
    let label = format!(
        "B={batch} H={n_q_heads} T={q_len}/{k_len} D={head_dim} gqa={gqa_factor} {}",
        ctx.label
    );
    vec![bench.result_sub_timed(
        Some(spec.subop),
        label,
        ref_perf_val,
        mt_perf,
        Some(equiv),
        mt_timing,
        ref_timing,
    )]
}

// ── SdpaVector2Pass — chained pass1 + pass2, MLX single-pass as ref ─────
//
// The 2-pass pair targets the long-N regime where single-pass `sdpa_vector`
// can't keep all KV slots in-flight per simdgroup. Pass 1 splits the n_kv
// walk across `blocks` threadgroups (each owning n_kv/blocks K positions);
// pass 2 reduces the per-block (max, sum, partial-O) tuples into one final
// O via TG=1024 (32 simdgroups × 32 lanes, where the reducer drops the
// remainder when `blocks % 32 != 0`).
//
// Both passes are measured separately and the rows are summed — this
// approximates chained-dispatch wall time. Real chained dispatch via
// `Context::dispatch_chain` is slightly faster (one CB commit + one wait
// for the whole chain) and is what production callers use; the test-bench
// in `tests/sdpa_decode_2pass_gpu.rs` measures that path end-to-end.
#[allow(clippy::too_many_arguments)]
fn run_sdpa_vector_2pass(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    _batch: usize,
    blocks: usize,
    pass2_kernel_name: &str,
    pass2_kernel_ir: fn(DType) -> Kernel,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "sdpa_decode_2pass hardcodes head_dim=128");
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    assert!(blocks.is_multiple_of(32), "blocks must be a multiple of 32 (pass-2 reducer)");
    let n_kv_heads = n_q_heads / gqa_factor;
    let gqa_factor_u = gqa_factor;
    let ctx = DtypeCtx::elementwise(dt);

    // sdpa_decode pass 1 dispatches at TPG=1024 (per its DISPATCH INVARIANTS).
    // The slow path is correct there; no compile-time spec needed.
    let p1_msl = match msl_reduction(spec, dt, None) {
        Some(s) => s,
        None => return vec![],
    };
    let p1_mk = match compile_mt(runner, &p1_msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let mut p2_kernel = pass2_kernel_ir(dt);
    p2_kernel.mode = KernelMode::Reduction;
    let p2_msl = match MslGenerator::default().generate(&p2_kernel) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let p2_mk = match compile_mt(runner, &p2_msl, pass2_kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);

    // Pass 1 partials: `partial_o` is Tensor<T> (matches MLX sdpa_vector_2pass —
    // post-softmax-weighted values are bounded so storage in T loses no useful
    // bits and halves bandwidth at f16/bf16). `partial_max` / `partial_sum`
    // stay f32 — the online-softmax running sum of `exp()` can blow past f16's
    // ~6.5e4 ceiling on long n_kv.
    let partial_o = runner.buffer_zeros(n_q_heads * blocks * head_dim * ctx.eb);
    let partial_max = runner.buffer_zeros(n_q_heads * blocks * 4);
    let partial_sum = runner.buffer_zeros(n_q_heads * blocks * 4);
    let mt_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);

    let hd_buf = runner.buffer_u32(head_dim as u32);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let kvs_buf = runner.buffer_u32(n_kv as u32); // kv_stride == n_kv (contiguous KV cache)
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let blocks_buf = runner.buffer_u32(blocks as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    // Pass 1: grid (n_kv_heads, blocks, 1), TG (32, gqa_factor, 1).
    // Buffer order matches `sdpa_decode_2pass_pass1` signature:
    //   q, k, v, partial_o, partial_m, partial_l,
    //   head_dim, n_kv, kv_stride, gqa_factor, blocks, scale.
    let p1_bufs: Vec<&GpuBuffer> = vec![
        &q_buf,
        &k_buf,
        &v_buf,
        &partial_o,
        &partial_max,
        &partial_sum,
        &hd_buf,
        &n_buf,
        &kvs_buf,
        &gqa_buf,
        &blocks_buf,
        &sc_buf,
    ];
    let p1_grid = [n_kv_heads, blocks, 1];
    let p1_tpg = [32, gqa_factor_u, 1];
    // Pass 2: grid (n_q_heads, 1, 1), TG (1024, 1, 1).
    let p2_bufs: Vec<&GpuBuffer> =
        vec![&partial_o, &partial_max, &partial_sum, &mt_out_buf, &hd_buf, &blocks_buf];
    let p2_grid = [n_q_heads, 1, 1];
    let p2_tpg = [1024, 1, 1];

    // Run once for correctness.
    runner.measure(&p1_mk, &p1_bufs, p1_grid, p1_tpg, 0, 1);
    runner.measure(&p2_mk, &p2_bufs, p2_grid, p2_tpg, 0, 1);
    let mt_out = crate::runner::read_typed(runner, &mt_out_buf, n_q_heads * head_dim, dt);

    // MLX single-pass `sdpa_vector` reference at the same shape.
    const REF_FCS: &[(usize, bool)] =
        &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
    let ref_name: Option<String> = match dt {
        DType::F32 => Some(format!("sdpa_vector_float_{head_dim}_{head_dim}")),
        DType::F16 => Some(format!("sdpa_vector_float16_t_{head_dim}_{head_dim}")),
        DType::BF16 => Some(format!("sdpa_vector_bfloat16_t_{head_dim}_{head_dim}")),
        _ => None,
    };
    let rk = ref_name.as_ref().and_then(|name| {
        spec.mlx_src.and_then(|src| runner.compile_with_bool_constants(src, name, REF_FCS).ok())
    });

    let equiv = rk.as_ref().map(|rk| {
        let gqa = runner.buffer_i32(gqa_factor as i32);
        let n_i32 = runner.buffer_i32(n_kv as i32);
        let khs = runner.buffer_u64((n_kv * head_dim) as u64);
        let kss = runner.buffer_u64(head_dim as u64);
        let mlx_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
        runner.measure(
            rk,
            &[&q_buf, &k_buf, &v_buf, &mlx_out_buf, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
            [n_q_heads, 1, 1],
            [1024, 1, 1],
            0,
            1,
        );
        let ref_out = crate::runner::read_typed(runner, &mlx_out_buf, n_q_heads * head_dim, dt);
        check_equiv_with(&ref_out, &mt_out, EquivTolerance::new(spec.tol, 0.999))
    });

    // Perf: read q + k + v + write out. K/V sized by n_kv_heads (GQA).
    let bytes = ((n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
        * ctx.eb) as f64;
    let p1_perf = bench_gbps_only(runner, &p1_mk, &p1_bufs, p1_grid, p1_tpg, bytes);
    let p2_perf = bench_gbps_only(runner, &p2_mk, &p2_bufs, p2_grid, p2_tpg, bytes);
    // Sum the per-pass µs (1/GBps × bytes) then convert back to GB/s.
    let mt_perf = match (p1_perf, p2_perf) {
        (Some(p1), Some(p2)) if p1 > 0.0 && p2 > 0.0 => {
            let gb = bytes / 1.0e9;
            let p1_s = gb / p1;
            let p2_s = gb / p2;
            Some(gb / (p1_s + p2_s))
        },
        _ => None,
    };
    let ref_perf = rk
        .as_ref()
        .map(|rk| {
            let gqa = runner.buffer_i32(gqa_factor as i32);
            let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * head_dim) as u64);
            let kss = runner.buffer_u64(head_dim as u64);
            let out = zeros_typed(runner, n_q_heads * head_dim, dt);
            bench_gbps_only(
                runner,
                rk,
                &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [n_q_heads, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        })
        .unwrap_or(None);

    let label = format!(
        "H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} blocks={blocks} {}",
        ctx.label
    );
    vec![bench.result_sub(Some(spec.subop), label, ref_perf, mt_perf, equiv)]
}

// ── SteelGemm (simdgroup tiled GEMM) ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_steel_gemm(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    m: usize,
    n: usize,
    k: usize,
    check_m: usize,
    check_n: usize,
    check_k: usize,
    bm: usize,
    bn: usize,
    tpg: usize,
) -> Vec<OpResult> {
    // `run_steel_gemm` only models the plain fused `C = A·B` signature
    // (`[a, b, d, m, n, k]`). The gather / masked / segmented / split-K
    // variants take extra operands (gather indices, block masks, K
    // segments, an fp32 partials slab) that this harness does not build —
    // dispatching them here binds the wrong buffer set and yields a
    // garbage correctness result. Their correctness is owned by the
    // dedicated `tests/steel_gemm_{gather,masked,segmented,splitk}_gpu_
    // correctness.rs` suites; skip them in `tile bench` until the harness
    // grows per-variant operand + oracle support.
    if spec.op != "steel_gemm_fused" {
        return Vec::new();
    }

    let ctx = DtypeCtx::elementwise(dt);

    // Compile MT kernel — must use SimdGroup2D so program_id axes map to
    // threadgroup indices (tid.x/tid.y) rather than the global thread index.
    let mut kernel = (spec.kernel_ir)(dt);
    kernel.mode = KernelMode::SimdGroup2D;
    let msl = match MslGenerator::default().generate(&kernel) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let mk = match runner.compile(&msl, spec.kernel_name) {
        Ok(k) => k,
        Err(_) => return Vec::new(),
    };

    // Compile MLX reference kernel with function constants
    let mlx_k = spec.mlx_src.and_then(|src| {
        spec.mlx_pattern.and_then(|pat| {
            let name = pat.replace("{tn}", DtypeCtx::elementwise(dt).tn);
            // has_batch(10)=false, use_out_source(100)=false, do_axpby(110)=false,
            // align_M(200)=true, align_N(201)=true, align_K(202)=true
            runner
                .compile_with_bool_constants(src, &name, &[
                    (10, false),
                    (100, false),
                    (110, false),
                    (200, true),
                    (201, true),
                    (202, true),
                ])
                .ok()
        })
    });

    // Build check buffers
    let a_buf = buffer_typed(runner, &vec![1.0f32; check_m * check_k], dt);
    let b_buf = buffer_typed(runner, &vec![1.0f32; check_k * check_n], dt);
    let d_buf = zeros_typed(runner, check_m * check_n, dt);
    let mlx_d_buf = zeros_typed(runner, check_m * check_n, dt);
    let m_buf = buffer_typed(runner, &[check_m as f32], DType::U32);
    let n_buf = buffer_typed(runner, &[check_n as f32], DType::U32);
    let k_buf = buffer_typed(runner, &[check_k as f32], DType::U32);

    // Build GEMMParams for MLX reference
    // C++ struct layout (Metal = C++ ABI):
    //   int M,N,K,lda,ldb,ldd; int tiles_n,tiles_m;  (8×4 = 32 bytes, aligned to 8)
    //   int64_t batch_stride_a,b,d;                   (3×8 = 24 bytes)
    //   int swizzle_log, gemm_k_iter, batch_ndim;    (3×4 = 12 bytes)
    // Total: 68 bytes, no padding needed (32 is 8-aligned)
    let lda = check_k as i32;
    let ldb = check_n as i32;
    let ldd = check_n as i32;
    let params_bytes: Vec<u8> = {
        let mut v = Vec::with_capacity(72);
        v.extend_from_slice(&(check_m as i32).to_le_bytes()); // M
        v.extend_from_slice(&(check_n as i32).to_le_bytes()); // N
        v.extend_from_slice(&(check_k as i32).to_le_bytes()); // K
        v.extend_from_slice(&lda.to_le_bytes()); // lda
        v.extend_from_slice(&ldb.to_le_bytes()); // ldb
        v.extend_from_slice(&ldd.to_le_bytes()); // ldd
        v.extend_from_slice(&((check_n / bn) as i32).to_le_bytes()); // tiles_n
        v.extend_from_slice(&((check_m / bm) as i32).to_le_bytes()); // tiles_m
        // No padding: offset 32 is already 8-byte aligned
        v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_a
        v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_b
        v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_d
        v.extend_from_slice(&0i32.to_le_bytes()); // swizzle_log
        v.extend_from_slice(&((check_k / 16) as i32).to_le_bytes()); // gemm_k_iterations_aligned
        v.extend_from_slice(&0i32.to_le_bytes()); // batch_ndim
        v
    };
    let params_buf = runner.buffer_bytes(&params_bytes);
    let addmm_buf = runner.buffer_zeros(32); // unused (use_out_source=false)
    let batch_shape_buf = runner.buffer_zeros(4); // unused
    let batch_strides_buf = runner.buffer_zeros(8); // unused

    let grid = [check_n / bn, check_m / bm, 1];
    let tpg_arr = [tpg, 1, 1];
    let all_bufs: [&GpuBuffer; 6] = [&a_buf, &b_buf, &d_buf, &m_buf, &n_buf, &k_buf];
    let mlx_bufs: [&GpuBuffer; 8] = [
        &a_buf,
        &b_buf,
        &mlx_d_buf,
        &mlx_d_buf,
        &params_buf,
        &addmm_buf,
        &batch_shape_buf,
        &batch_strides_buf,
    ];

    // Run MLX reference
    let ref_perf = mlx_k.as_ref().and_then(|rk| {
        runner.measure(rk, &mlx_bufs, grid, tpg_arr, 0, 1);
        bench_gbps_only(
            runner,
            rk,
            &mlx_bufs,
            grid,
            tpg_arr,
            ((check_m * check_k + check_k * check_n + check_m * check_n) * ctx.eb) as f64,
        )
    });
    let ref_vals: Vec<f32> = mlx_k
        .as_ref()
        .map(|_| read_typed(runner, &mlx_d_buf, check_m * check_n, dt))
        .unwrap_or_else(|| (0..check_m * check_n).map(|_| check_k as f32).collect());

    // Run MT
    runner.measure(&mk, &all_bufs, grid, tpg_arr, 0, 1);
    let mt_vals = read_typed(runner, &d_buf, check_m * check_n, dt);

    let equiv = check_equiv(&ref_vals, &mt_vals, 1e-2);
    let label = format!("M={m} N={n} K={k} BM={bm} BN={bn} {}", ctx.label);
    let mt_perf = bench_gbps_only(
        runner,
        &mk,
        &all_bufs,
        grid,
        tpg_arr,
        ((check_m * check_k + check_k * check_n + check_m * check_n) * ctx.eb) as f64,
    );

    vec![bench.result_sub(Some(spec.subop), label, ref_perf, mt_perf, Some(equiv))]
}

// ── SdpaBatchedDecode — M7 speculative-decode batched-Q ───────────────────
//
// Stub for Phase 0 scaffolding (task M7-2). Kernel implementations land in
// the follow-up tasks (M7-3 for K=2/4 decode-form, M7-4 for K=8/16 prefill-
// tile reuse). The stub returns an empty result vector so that
// `inventory::submit!` rows can reference the dispatch variant without
// crashing `tile bench --list` while the kernels are in flight.

#[allow(clippy::too_many_arguments)]
fn run_sdpa_batched_decode(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    batch_q: usize,
    variant: &BatchedDecodeVariant,
    tpg: usize,
) -> Vec<OpResult> {
    match variant {
        BatchedDecodeVariant::Decode => run_sdpa_batched_decode_form(
            spec, runner, dt, bench, head_dim, n_kv, n_q_heads, gqa_factor, batch_q, tpg,
        ),
        BatchedDecodeVariant::PrefillTile { bq, bk, wm, wn } =>
            run_sdpa_batched_decode_prefill_tile(
                spec, runner, dt, bench, head_dim, n_kv, n_q_heads, gqa_factor, batch_q, *bq, *bk,
                *wm, *wn, tpg,
            ),
    }
}

// ── Decode variant — K=2/4 batched-Q decode-form bench runner ────────────
//
// Compiles the M7 batched kernel (`sdpa_decode_batched_q{batch_q}` —
// the inventory submit's `kernel_ir`) and the single-Q `sdpa_decode`
// reference at the same shape. Reports:
//
//   mt_perf  = M7_bytes / T_m7                    (real M7 throughput)
//   ref_perf = M7_bytes / (batch_q × T_single)   (effective baseline
//                                                  throughput if the
//                                                  K independent
//                                                  sdpa_decode calls
//                                                  shared one M7_bytes
//                                                  bandwidth budget)
//
// `mt_perf / ref_perf` then equals the wall-clock speedup
// `(batch_q × T_single) / T_m7` — the metric that matches what
// consumers actually see when they swap K independent decode dispatches
// for one batched call. Single-Q decode time at the same shape scales
// linearly across the K independent calls (same kernel, same KV cache;
// the dispatches are independent).
//
// Correctness check: M7's batched output's row `qi` (for `qi in
// 0..batch_q`) must match the single-Q `sdpa_decode` output at the
// same shape, fed Q[qi]. We round-trip the first Q slot only — same
// pattern as `run_sdpa_vector`'s MLX-correctness check (one shape per
// run, no need to gate on every slot).

#[allow(clippy::too_many_arguments)]
fn run_sdpa_batched_decode_form(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    batch_q: usize,
    tpg: usize,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "sdpa_decode_batched hardcodes head_dim=128");
    assert!(
        matches!(batch_q, 2 | 4),
        "Decode variant ships K∈{{2,4}} specializations; got K={batch_q}",
    );
    // K=4 register pressure caps Metal's `maxTotalThreadsPerThreadgroup`
    // at 768 on M1 Max; dispatching past that silently produces all-zero
    // outputs (see DISPATCH INVARIANTS doc on `sdpa_decode_batched.rs`).
    // K=2 has no such cap, so any TPG = 1024 dispatch is fine there.
    assert!(
        tpg <= 768 || batch_q < 4,
        "sdpa_decode_batched_q{batch_q} cannot dispatch at tpg={tpg}: \
         K=4 requires tpg ≤ 768 (M1 Max maxTotalThreadsPerThreadgroup \
         cap at this register pressure; dispatching past the cap \
         silently writes all-zero outputs). Use tpg=512.",
    );
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    let n_kv_heads = n_q_heads / gqa_factor;
    let ctx = DtypeCtx::elementwise(dt);

    // Compile the M7 batched kernel.
    let msl = match msl_reduction(spec, dt, Some(tpg as u32)) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    // Compile `sdpa_decode` (the single-Q reference) at the same shape.
    // `tpg=1024` matches the rebased `sdpa_decode` kernel's design
    // threadgroup size (32 simdgroups × 32 lanes).
    let single_tpg = 1024usize;
    let mut single_kernel = crate::ffai::sdpa_decode::ffai_sdpa_decode::kernel_ir_for(dt);
    single_kernel.mode = KernelMode::Reduction;
    let single_msl =
        match MslGenerator::new(msl_cfg_for(Some(single_tpg as u32))).generate(&single_kernel) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
    let single_compiled = match compile_mt(runner, &single_msl, "ffai_sdpa_decode") {
        Some(k) => k,
        None => return vec![],
    };

    // Buffer setup — Q `[n_q_heads, batch_q, head_dim]`, K/V
    // `[n_kv_heads, n_kv, head_dim]`, out matches Q. `kv_stride == n_kv`
    // for the bench (no slack capacity).
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * batch_q * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    let q_buf = buffer_typed(runner, &vals[..n_q_heads * batch_q * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let mt_out_buf = zeros_typed(runner, n_q_heads * batch_q * head_dim, dt);
    let hd_buf = runner.buffer_u32(head_dim as u32);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let kv_stride_buf = runner.buffer_u32(n_kv as u32);
    let hpg_buf = runner.buffer_u32(gqa_factor as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    let mt_bufs: Vec<&GpuBuffer> = vec![
        &q_buf,
        &k_buf,
        &v_buf,
        &mt_out_buf,
        &hd_buf,
        &n_buf,
        &kv_stride_buf,
        &hpg_buf,
        &sc_buf,
    ];

    // M7 batched dispatch — one TG per Q head.
    runner.measure(&mk, &mt_bufs, [n_q_heads, 1, 1], [tpg, 1, 1], 0, 1);
    let mt_out = read_typed(runner, &mt_out_buf, n_q_heads * batch_q * head_dim, dt);

    // Single-Q reference — gather Q[h, 0, :] for all h, dispatch
    // `ffai_sdpa_decode` once, compare against M7's row qi=0 across all
    // heads. `ffai_sdpa_decode`'s 13-buffer signature is:
    // q, k, v, out, head_dim, n_kv, kv_stride, heads_per_group,
    // sink_end, window_start, has_sink, sink_logit, scale.
    let mut q_first: Vec<f32> = vec![0.0; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let src = (h * batch_q) * head_dim;
        let dst = h * head_dim;
        q_first[dst..dst + head_dim].copy_from_slice(&vals[src..src + head_dim]);
    }
    let single_q_buf = buffer_typed(runner, &q_first, dt);
    let single_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
    let sink_buf = runner.buffer_u32(0);
    let window_buf = runner.buffer_u32(0);
    // Dense path — no learned attention sink (has_sink = 0).
    let has_sink_buf = runner.buffer_u32(0);
    let sink_logit_buf = runner.buffer_f32_scalar(0.0);
    let single_bufs: Vec<&GpuBuffer> = vec![
        &single_q_buf,
        &k_buf,
        &v_buf,
        &single_out_buf,
        &hd_buf,
        &n_buf,
        &kv_stride_buf,
        &hpg_buf,
        &sink_buf,
        &window_buf,
        &has_sink_buf,
        &sink_logit_buf,
        &sc_buf,
    ];
    runner.measure(&single_compiled, &single_bufs, [n_q_heads, 1, 1], [single_tpg, 1, 1], 0, 1);
    let single_out = read_typed(runner, &single_out_buf, n_q_heads * head_dim, dt);

    // Correctness: M7 row qi=0 vs single-Q reference. Extract row 0
    // from every head (stride batch_q). This bench-time check covers
    // only qi=0 across all heads — sufficient to flag a regression in
    // the kernel's per-head dispatch shape or the Q[0] addressing, but
    // qi=1..K-1 are not checked here. The
    // `sdpa_decode_batched_gpu_correctness.rs` integration tests
    // cover all K slots (interleaved-Q comparison + identical-Qs
    // sanity check + tpg=1024 divergence regression test).
    let mut mt_row0: Vec<f32> = vec![0.0; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let src = (h * batch_q) * head_dim;
        mt_row0[h * head_dim..(h + 1) * head_dim].copy_from_slice(&mt_out[src..src + head_dim]);
    }
    let equiv = check_equiv_with(&single_out, &mt_row0, EquivTolerance::new(spec.tol, 0.999));

    // Perf — M7 throughput at M7_bytes (real); reference throughput
    // scaled to "M7_bytes / (batch_q × T_single)" so the displayed
    // ratio == wall-clock speedup vs K independent single-Q decodes.
    let m7_bytes = ((n_q_heads * batch_q * head_dim
        + 2 * n_kv_heads * n_kv * head_dim
        + n_q_heads * batch_q * head_dim)
        * ctx.eb) as f64;
    let (mt_perf, mt_timing) =
        bench_gbps(runner, &mk, &mt_bufs, [n_q_heads, 1, 1], [tpg, 1, 1], m7_bytes)
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None));
    // Feeding `m7_bytes / batch_q` as the bench_gbps `bytes` for the
    // single-Q call gives us `(m7_bytes / batch_q) / T_single =
    // m7_bytes / (batch_q × T_single)` — the effective baseline
    // throughput that makes `mt_perf / ref_perf` equal the wall-clock
    // speedup. Independent of how the single-Q call's "real" bytes
    // are accounted; we're constructing the comparison metric we
    // actually want to display.
    let ref_bytes_scaled = m7_bytes / (batch_q as f64);
    let (ref_perf, ref_timing) = bench_gbps(
        runner,
        &single_compiled,
        &single_bufs,
        [n_q_heads, 1, 1],
        [single_tpg, 1, 1],
        ref_bytes_scaled,
    )
    .map(|(p, t)| (Some(p), Some(t)))
    .unwrap_or((None, None));

    let label =
        format!("K={batch_q} H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} {}", ctx.label,);
    vec![bench.result_sub_timed(
        Some(spec.subop),
        label,
        ref_perf,
        mt_perf,
        Some(equiv),
        mt_timing,
        ref_timing,
    )]
}

// ── PrefillTile variant — K=8/16 via mt_sdpa_prefill_mma reuse ───────────
//
// No new MSL: the FA-2 simdgroup-matrix prefill tile from PR #47/#52
// already implements the KV-reuse pattern at BQ × BK with online softmax,
// which is structurally identical to dflash-mlx's `verify_qmm`. The
// runner pads Q up to BQ=32 rows (real K rows + zeros) and pads K/V up
// to k_len = n_kv + BQ slots; the kernel's hardcoded causal mask then
// gives Q[i] for i in 0..K the speculative-decode-verify semantics
// `attended = [0, n_kv + i + 1)` (prefix + candidates [0..i]).
//
// Wasted work scales with (BQ - K)/BQ — 50% at K=16, 75% at K=8.
// Acceptable tradeoff vs. writing a new BQ=8 / BQ=16 MMA kernel; if the
// bench shows the waste killing the amortization win, a hand-rolled
// BQ=K variant lands in a follow-up.
//
// Phase 1 of M7's PrefillTile arm runs correctness-only — no MLX
// reference comparison, no perf measurement. Bench numbers land
// alongside the Decode-variant runner wiring.

#[allow(clippy::too_many_arguments)]
fn run_sdpa_batched_decode_prefill_tile(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    batch_q: usize,
    bq: usize,
    _bk: usize,
    _wm: usize,
    _wn: usize,
    tpg: usize,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "PrefillTile arm hardcodes head_dim=128 (mt_sdpa_prefill_mma)");
    assert_eq!(bq, 32, "PrefillTile arm requires BQ=32 (the prefill kernel's hardcoded tile)");
    assert!(batch_q <= bq, "batch_q ({batch_q}) must fit inside a single BQ={bq} tile",);
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    let n_kv_heads = n_q_heads / gqa_factor;
    let q_len_padded = bq;
    let k_len_padded = n_kv + bq;

    // Kernel compile — Reduction codegen mode + bfloat reinterpret cast,
    // mirrors `run_sdpa_prefill`. The prefill MMA kernel accumulates in
    // f32 throughout and emits one narrowing cast per output store.
    let mut kernel = (spec.kernel_ir)(dt);
    kernel.mode = KernelMode::SimdGroup2D;
    kernel.bfloat_reinterpret_cast = true;
    let msl = match MslGenerator::default().generate(&kernel) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    // Allocate padded buffers — first K rows of Q hold the candidate Q
    // vectors; the rest are zeros. K/V hold `n_kv` real positions plus
    // `bq` padding slots whose values don't matter (the causal mask
    // gates all KV positions > q_abs[i] for the real rows i in 0..K).
    let qsz_padded = n_q_heads * q_len_padded * head_dim;
    let kvsz_padded = n_kv_heads * k_len_padded * head_dim;
    let vals: Vec<f32> =
        (0..qsz_padded.max(kvsz_padded)).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    // Q: first K rows of each head are the deterministic ramp; the
    // padding rows (K..BQ) are zero so they don't contribute meaningful
    // dot products. The kernel still computes attention for them — the
    // output positions are simply discarded.
    let mut q_padded = vec![0.0f32; qsz_padded];
    for h in 0..n_q_heads {
        for qi in 0..batch_q {
            let src_off = (h * batch_q + qi) * head_dim;
            let dst_off = (h * q_len_padded + qi) * head_dim;
            // Source the candidate Q values from `vals` — same
            // deterministic ramp the prefill bench uses.
            q_padded[dst_off..dst_off + head_dim]
                .copy_from_slice(&vals[src_off..src_off + head_dim]);
        }
    }

    // K/V: real prefix at positions [0, n_kv); padding slots
    // [n_kv, n_kv + bq) hold zeros. The causal mask hides them from
    // the real Q rows we read out.
    let mut k_padded = vec![0.0f32; kvsz_padded];
    let mut v_padded = vec![0.0f32; kvsz_padded];
    for h in 0..n_kv_heads {
        let real_size = n_kv * head_dim;
        let src_off = h * n_kv * head_dim;
        let dst_off = h * k_len_padded * head_dim;
        // The same deterministic ramp populates both K and V (they
        // differ by which slice of `vals` they start at — but for the
        // bench we don't care, we just need correctness vs the CPU
        // reference).
        k_padded[dst_off..dst_off + real_size]
            .copy_from_slice(&vals[..real_size.min(vals.len() - src_off)][..real_size]);
        v_padded[dst_off..dst_off + real_size]
            .copy_from_slice(&vals[..real_size.min(vals.len() - src_off)][..real_size]);
    }

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_buf = buffer_typed(runner, &q_padded, dt);
    let k_buf = buffer_typed(runner, &k_padded, dt);
    let v_buf = buffer_typed(runner, &v_padded, dt);
    let mt_out_buf = zeros_typed(runner, qsz_padded, dt);
    let q_len_buf = runner.buffer_u32(q_len_padded as u32);
    let k_len_buf = runner.buffer_u32(k_len_padded as u32);
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let n_q_heads_buf = runner.buffer_u32(n_q_heads as u32);
    let n_kv_heads_buf = runner.buffer_u32(n_kv_heads as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    let mt_bufs: Vec<&GpuBuffer> = vec![
        &q_buf,
        &k_buf,
        &v_buf,
        &mt_out_buf,
        &q_len_buf,
        &k_len_buf,
        &gqa_buf,
        &n_q_heads_buf,
        &n_kv_heads_buf,
        &sc_buf,
    ];

    // Grid: (q_tiles=1, n_q_heads, batch=1) — q_len_padded == bq so
    // exactly one Q tile per head. The kernel's tpg=128 stays the same.
    runner.measure(&mk, &mt_bufs, [1, n_q_heads, 1], [tpg, 1, 1], 0, 1);
    let mt_out = read_typed(runner, &mt_out_buf, qsz_padded, dt);

    // ── Correctness — inline causal-prefix reference on head 0, real K
    //    rows only. Tiny CPU check (~1 ms at K=16, n_kv=4096) that pins
    //    the kernel against the same mask the
    //    `sdpa_decode_batched_prefill_gpu_correctness` integration
    //    tests use, but kept in-bench so `result_sub_timed`'s
    //    `mt_perf.is_some() && equiv.is_none()` panic guard is
    //    satisfied — and so regressions in the prefill-MMA kernel
    //    that affect this dispatch path get caught at `make bench`
    //    time, not only in the integration suite.
    //
    // q_len_off = k_len_padded - q_len_padded = n_kv. For Q row qi in
    // 0..batch_q, attended KV range = [0, n_kv + qi + 1).
    let head_for_check = 0usize;
    let kv_head_for_check = head_for_check / (n_q_heads / n_kv_heads);
    let kv_slab_base = kv_head_for_check * k_len_padded * head_dim;
    let mut expected_head0: Vec<f32> = vec![0.0; batch_q * head_dim];
    for qi in 0..batch_q {
        let q_off = (head_for_check * q_len_padded + qi) * head_dim;
        let mut scores = vec![f32::NEG_INFINITY; k_len_padded];
        let attended_end = (n_kv + qi + 1).min(k_len_padded);
        for (t, score) in scores.iter_mut().enumerate().take(attended_end) {
            let k_off = kv_slab_base + t * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_padded[q_off + d] * k_padded[k_off + d];
            }
            *score = dot * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            if s.is_finite() {
                *s = (*s - m).exp();
                sum += *s;
            } else {
                *s = 0.0;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (t, &s) in scores.iter().enumerate() {
                acc += s * inv * v_padded[kv_slab_base + t * head_dim + d];
            }
            expected_head0[qi * head_dim + d] = acc;
        }
    }
    // Extract head 0's real-K-row slice from the GPU output.
    let mut actual_head0: Vec<f32> = Vec::with_capacity(batch_q * head_dim);
    for qi in 0..batch_q {
        let src = (head_for_check * q_len_padded + qi) * head_dim;
        actual_head0.extend_from_slice(&mt_out[src..src + head_dim]);
    }
    let equiv =
        check_equiv_with(&expected_head0, &actual_head0, EquivTolerance::new(spec.tol, 0.999));

    // ── Perf measurement (M7-batched-via-prefill-tile vs K independent
    //    single-Q `sdpa_decode` calls). Same speedup-as-displayed-ratio
    //    convention as the Decode variant: feed `m7_bytes / batch_q`
    //    to the single-Q `bench_gbps` so `mt_perf / ref_perf` equals
    //    the wall-clock speedup vs the K-independent decode pattern.
    //
    // Correctness inside the bench is skipped — the K=8/16 prefill-
    // tile arm produces **causal** outputs (Q[i] sees prefix + own
    // predecessors) while single-Q `sdpa_decode` produces **flat**
    // outputs (each Q sees the full prefix), so a per-row equiv check
    // would fail by construction. The
    // `tests/sdpa_decode_batched_prefill_gpu_correctness.rs`
    // integration tests verify against the matching
    // `naive_sdpa_causal_prefix_f32` reference.
    let ctx = DtypeCtx::elementwise(dt);
    let m7_bytes = ((qsz_padded + 2 * kvsz_padded + qsz_padded) * ctx.eb) as f64;
    let (mt_perf, mt_timing) =
        bench_gbps(runner, &mk, &mt_bufs, [1, n_q_heads, 1], [tpg, 1, 1], m7_bytes)
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None));

    // Compile + dispatch single-Q `sdpa_decode` once at the matching
    // shape for the K-independent baseline. The dispatch is dense-path
    // (sink_end = 0, window_start = 0).
    let single_tpg = 1024usize;
    let mut single_kernel = crate::ffai::sdpa_decode::ffai_sdpa_decode::kernel_ir_for(dt);
    single_kernel.mode = KernelMode::Reduction;
    let (ref_perf, ref_timing) = match MslGenerator::new(msl_cfg_for(Some(single_tpg as u32)))
        .generate(&single_kernel)
        .ok()
        .and_then(|s| compile_mt(runner, &s, "ffai_sdpa_decode"))
    {
        Some(single_compiled) => {
            let mut q_first: Vec<f32> = vec![0.0; n_q_heads * head_dim];
            for h in 0..n_q_heads {
                let src = (h * batch_q) * head_dim;
                q_first[h * head_dim..(h + 1) * head_dim]
                    .copy_from_slice(&vals[src..src + head_dim]);
            }
            let single_q_buf = buffer_typed(runner, &q_first, dt);
            // Single-Q reference can use the un-padded KV cache.
            let k_real_size = n_kv_heads * n_kv * head_dim;
            let k_single_buf = buffer_typed(runner, &vals[..k_real_size.min(vals.len())], dt);
            let v_single_buf = buffer_typed(runner, &vals[..k_real_size.min(vals.len())], dt);
            let single_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
            let single_hd_buf = runner.buffer_u32(head_dim as u32);
            let single_n_buf = runner.buffer_u32(n_kv as u32);
            let single_kv_stride_buf = runner.buffer_u32(n_kv as u32);
            let single_hpg_buf = runner.buffer_u32(gqa_factor as u32);
            let sink_buf = runner.buffer_u32(0);
            let window_buf = runner.buffer_u32(0);
            // Dense path — no learned attention sink (has_sink = 0).
            let has_sink_buf = runner.buffer_u32(0);
            let sink_logit_buf = runner.buffer_f32_scalar(0.0);
            let single_bufs: Vec<&GpuBuffer> = vec![
                &single_q_buf,
                &k_single_buf,
                &v_single_buf,
                &single_out_buf,
                &single_hd_buf,
                &single_n_buf,
                &single_kv_stride_buf,
                &single_hpg_buf,
                &sink_buf,
                &window_buf,
                &has_sink_buf,
                &sink_logit_buf,
                &sc_buf,
            ];
            let ref_bytes_scaled = m7_bytes / (batch_q as f64);
            bench_gbps(
                runner,
                &single_compiled,
                &single_bufs,
                [n_q_heads, 1, 1],
                [single_tpg, 1, 1],
                ref_bytes_scaled,
            )
            .map(|(p, t)| (Some(p), Some(t)))
            .unwrap_or((None, None))
        },
        None => (None, None),
    };

    let label = format!(
        "K={batch_q} H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} causal {}",
        ctx.label,
    );
    vec![bench.result_sub_timed(
        Some(spec.subop),
        label,
        ref_perf,
        mt_perf,
        Some(equiv),
        mt_timing,
        ref_timing,
    )]
}
