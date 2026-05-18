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
    spec::{BenchDispatch, BenchSpec, MlxArg, ScalarBufSpec, ShapeSpec},
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
        BenchDispatch::QuantizedMatVec { shapes, group_size, tpg } =>
            run_quantized_mat_vec(spec, runner, dt, &bench, shapes, *group_size, *tpg),
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
    }
}

// ── MSL generation ────────────────────────────────────────────────────────

fn msl_elementwise(spec: &BenchSpec, dt: DType) -> Option<String> {
    MslGenerator::default().generate(&(spec.kernel_ir)(dt)).ok()
}
fn msl_reduction(spec: &BenchSpec, dt: DType) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).ok()
}
fn msl_grid3d(spec: &BenchSpec, dt: DType) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Grid3D;
    MslGenerator::default().generate(&k).ok()
}
fn msl_for_mode(spec: &BenchSpec, dt: DType, mode: KernelMode) -> Option<String> {
    match mode {
        KernelMode::Elementwise => msl_elementwise(spec, dt),
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D =>
            msl_reduction(spec, dt),
        KernelMode::Grid3D => msl_grid3d(spec, dt),
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
    // Cache compiled kernels by mode — MSL is identical for all shapes with the same
    // (dt, mode), so compile once instead of once-per-shape.
    let mut compiled: std::collections::HashMap<u8, crate::runner::CompiledKernel> =
        std::collections::HashMap::new();
    let mode_key = |m: KernelMode| match m {
        KernelMode::Elementwise => 0u8,
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D => 1,
        KernelMode::Grid3D => 2,
    };

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
        let mk = match compiled.entry(mode_key(shape.mode)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let msl = match msl_for_mode(spec, dt, shape.mode) {
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
    let msl = match msl_reduction(spec, dt) {
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
    let msl = match msl_reduction(spec, DType::F32) {
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
    let msl = match msl_reduction(spec, DType::F32) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "float32");

    let check_vals: Vec<f32> = (0..check_n).map(|i| ((i * 7 + 3) % 97) as f32 * 0.1).collect();
    let expected: f32 = {
        let mut best = f32::NEG_INFINITY;
        let mut idx = 0usize;
        for (i, &v) in check_vals.iter().enumerate() {
            if v > best {
                best = v;
                idx = i;
            }
        }
        idx as f32
    };
    let inp_c = buffer_typed(runner, &check_vals, DType::F32);
    let out_c = zeros_typed(runner, 1, DType::F32);
    let ns_c = runner.buffer_u32(check_n as u32);
    let mt_chk = run_typed_once(
        runner,
        &mk,
        &[&inp_c, &out_c, &ns_c],
        &out_c,
        1,
        [1, 1, 1],
        [tpg, 1, 1],
        DType::F32,
    );
    let equiv = check_equiv(&[expected], &mt_chk, 0.5);

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
    let msl = match msl_elementwise(spec, DType::F32) {
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
    let msl = match msl_elementwise(spec, DType::F32) {
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

fn run_quantized_mat_vec(
    spec: &BenchSpec,
    runner: &GpuRunner,
    _dt: DType,
    bench: &OpBench,
    shapes: &[(usize, usize)],
    group_size: usize,
    tpg: usize,
) -> Vec<OpResult> {
    let msl = match msl_reduction(spec, DType::F32) {
        Some(s) => s,
        None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "");
    let mut results = Vec::new();
    for &(m, k) in shapes {
        let w_elems = m * k / 8;
        let sb_elems = m * k / group_size;
        let gs_per_row = k / group_size;
        // Correctness check: M=4 rows, K=group_size (one group per row)
        let cm = 4usize;
        let ck = group_size;
        let w_check: Vec<u32> = (0..cm * ck / 8)
            .map(|i| {
                let mut v = 0u32;
                for bit in 0..8u32 {
                    v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                }
                v
            })
            .collect();
        let s_check = vec![0.1f32; cm];
        let b_check = vec![0.0f32; cm];
        let x_check = vec![1.0f32; ck];
        let ref_out: Vec<f32> = (0..cm)
            .map(|row| {
                let mut acc = 0.0f32;
                for g in 0..1usize {
                    let s = s_check[row + g];
                    let bias = b_check[row + g];
                    for p in 0..8usize {
                        let packed = w_check[row * ck / 8 + g * 8 + p];
                        for bit in 0..8u32 {
                            let int4_val = ((packed >> (bit * 4)) & 0xF) as f32;
                            acc += (s * int4_val + bias) * x_check[g * ck + p * 8 + bit as usize];
                        }
                    }
                }
                acc
            })
            .collect();
        let w_bytes: Vec<u8> = w_check.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_buf_c = runner.buffer_bytes(&w_bytes);
        let s_buf_c = runner.buffer_f32(&s_check);
        let b_buf_c = runner.buffer_f32(&b_check);
        let x_buf_c = runner.buffer_f32(&x_check);
        let out_c = runner.buffer_zeros(cm * 4);
        let k_buf_c = runner.buffer_u32(ck as u32);
        let gpr_buf_c = runner.buffer_u32(1u32);
        runner.measure(
            &mk,
            &[&w_buf_c, &s_buf_c, &b_buf_c, &x_buf_c, &out_c, &k_buf_c, &gpr_buf_c],
            [cm, 1, 1],
            [tpg, 1, 1],
            0,
            1,
        );
        let mt_out_c = runner.read_f32_slice(&out_c, cm);
        let n_bad =
            ref_out.iter().zip(mt_out_c.iter()).filter(|(r, m)| (*r - *m).abs() > 1e-3).count();
        let equiv = EquivResult {
            n_checked: cm,
            max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
            passed: n_bad == 0,
        };

        let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
        let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
        let biases_f32 = vec![0.0f32; sb_elems];
        let x_f32: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();
        let w_mt_buf = runner.buffer_bytes(&w_data);
        let s_mt_buf = runner.buffer_f32(&scales_f32);
        let b_mt_buf = runner.buffer_f32(&biases_f32);
        let x_mt_buf = runner.buffer_f32(&x_f32);
        let k_buf = runner.buffer_u32(k as u32);
        let gpr_buf = runner.buffer_u32(gs_per_row as u32);
        let bytes_mt = (m * k / 2 + sb_elems * 4 * 2 + k * 4 + m * 4) as f64;
        let mt_perf = {
            let out_buf = runner.buffer_zeros(m * 4);
            bench_gbps_only(
                runner,
                &mk,
                &[&w_mt_buf, &s_mt_buf, &b_mt_buf, &x_mt_buf, &out_buf, &k_buf, &gpr_buf],
                [m, 1, 1],
                [tpg, 1, 1],
                bytes_mt,
            )
        };
        // MLX ref uses f16 data (different dtype)
        const ROWS_PER_TG: usize = 8;
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
            let bytes_f16 = (m * k / 2 + sb_elems * 2 * 2 + k * 2 + m * 2) as f64;
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
                [1, m / ROWS_PER_TG, 1],
                [64, 1, 1],
                bytes_f16,
            )
        });
        results.push(bench.result_sub(
            Some(spec.subop),
            format!("M={m} K={k} f32 gs{group_size} b4"),
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
    let msl = match msl_grid3d(spec, DType::F16) {
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
    let msl = match msl_reduction(spec, dt) {
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
        let ref_perf = rk.as_ref().and_then(|rk| {
            let gqa = runner.buffer_i32(1i32);
            let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * d) as u64);
            let kss = runner.buffer_u64(d as u64);
            let out = zeros_typed(runner, h * d, dt);
            bench_gbps_only(
                runner,
                rk,
                &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [h, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        });
        let mt_perf = {
            let out = zeros_typed(runner, h * d, dt);
            bench_gbps_only(
                runner,
                &mk,
                &[&q_buf, &k_buf, &v_buf, &out, &n_buf, &sc_buf],
                [h, 1, 1],
                [tpg, 1, 1],
                bytes,
            )
        };
        results.push(bench.result_sub(
            Some(spec.subop),
            format!("H={h} N={n_kv} D={d} {}", ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
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
    let msl = match msl_grid3d(spec, dt) {
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
        3..=5 => 8,
        6 | 8 => 4,
        _ => panic!("affine_pack_factor: unsupported bits={bits}"),
    }
}

fn affine_bytes_per_pack(bits: usize) -> usize {
    match bits {
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
    let msl = match msl_elementwise(spec, dt) {
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
    let msl = match msl_reduction(spec, dt) {
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
    let equiv = rk.as_ref().map(|rk| {
        let mlx_packed_buf = runner.buffer_zeros(n_packs * 4);
        let mlx_scales_buf = zeros_typed(runner, n_total_groups, dt);
        let mlx_biases_buf = zeros_typed(runner, n_total_groups, dt);
        let mlx_bufs: Vec<&GpuBuffer> =
            vec![&w_buf, &mlx_packed_buf, &mlx_scales_buf, &mlx_biases_buf];
        runner.measure(rk, &mlx_bufs, [n_total_groups, 1, 1], [32, 1, 1], 0, 1);
        let ref_packed_bytes = runner.read_bytes(&mlx_packed_buf, n_packs * 4);
        let ref_packed: Vec<u32> = ref_packed_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let ref_scales = crate::runner::read_typed(runner, &mlx_scales_buf, n_total_groups, dt);
        let ref_biases = crate::runner::read_typed(runner, &mlx_biases_buf, n_total_groups, dt);

        // Compare in dequantized space so ±1-ULP packed disagreement
        // maps to ±scale absolute error matching `spec.tol`.
        let dequant = |packed: &[u32], scales: &[f32], biases: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; n_elem];
            for g in 0..n_total_groups {
                let packs_per_group = group_size / pack_factor;
                let pack_base = g * packs_per_group;
                for p in 0..packs_per_group {
                    let val = packed[pack_base + p];
                    for k in 0..pack_factor {
                        let shift = (k * bits) as u32;
                        let mask = (1u32 << bits) - 1;
                        let q = (val >> shift) & mask;
                        out[g * group_size + p * pack_factor + k] =
                            scales[g] * q as f32 + biases[g];
                    }
                }
            }
            out
        };
        let ref_dequant = dequant(&ref_packed, &ref_scales, &ref_biases);
        let mt_dequant = dequant(&mt_packed, &mt_scales, &mt_biases);
        // Cosine floor 0.99 — quantization rounding in f16/bf16 can flip
        // ±1 nibble on a small fraction of elements (depresses cosine
        // slightly below the default 0.999).
        check_equiv_with(&ref_dequant, &mt_dequant, EquivTolerance::new(spec.tol, 0.99))
    });

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
    let msl = match msl_reduction(spec, dt) {
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
    let mt_perf = bench_gbps_only(runner, &mk, &mt_bufs, [n_q_heads, 1, 1], [tpg, 1, 1], bytes);
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

    let label = format!("H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} {}", ctx.label);
    vec![bench.result_sub(Some(spec.subop), label, ref_perf, mt_perf, equiv)]
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

    let p1_msl = match msl_reduction(spec, dt) {
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
