//! GPU runner implementation extracted from metaltile-bench/src/spec.rs
//! BenchSpec::run() and all dispatch arms, transformed into free functions.

use std::collections::BTreeMap;

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    constexpr::ConstExprValues,
    dtype::DType,
    ir::{KernelMode, Op},
};
use metaltile_interp::{Interpreter, TensorData};
use metaltile_std::{
    bench_types::{
        DtypeCtx,
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        check_equiv,
        check_equiv_with,
    },
    spec::{BenchDispatch, BenchSpec, MlxArg, ScalarBufSpec, ShapeSpec},
};

use crate::{
    measure::{bench_gbps, buffer_typed, run_typed_once, zeros_typed},
    runner::{GpuBuffer, GpuRunner},
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
        KernelMode::Reduction | KernelMode::Tile2D => msl_reduction(spec, dt),
        KernelMode::Grid3D => msl_grid3d(spec, dt),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn mlx_name(pat: &str, tn: &str) -> String { pat.replace("{tn}", tn) }
fn compile_mt(runner: &GpuRunner, msl: &str, name: &str) -> Option<crate::runner::CompiledKernel> {
    runner.compile(msl, name).ok()
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
fn td(dt: DType, shape: &[usize], data: &[f32]) -> TensorData {
    let mut td = TensorData::zeros(shape, dt);
    for (i, &v) in data.iter().enumerate() {
        td.write_scalar(i, v as f64);
    }
    td
}
fn constexprs(vals: &[(&str, usize)]) -> ConstExprValues {
    let mut cv = ConstExprValues::new();
    for (k, v) in vals {
        cv.insert(k.to_string(), *v);
    }
    cv
}
fn interp(
    kernel: &metaltile_core::ir::Kernel,
    inputs: BTreeMap<String, TensorData>,
    cv: ConstExprValues,
    mode: InterpMode,
) -> Option<BTreeMap<String, Vec<f32>>> {
    let mut interp = Interpreter::new(inputs, cv);
    let result = match mode {
        InterpMode::Elementwise(n) => interp.run_grid(kernel, n),
        InterpMode::Reduction(rows) => interp.run_grid_reduction(kernel, rows),
        InterpMode::Grid3D(x, y, z) => interp.run_grid_3d(kernel, x, y, z),
    };
    let result = match result {
        Ok(r) => r,
        Err(_) => return None,
    };
    let mut out = BTreeMap::new();
    for (name, td) in &result.outputs {
        out.insert(
            name.clone(),
            (0..td.num_elements()).map(|i| td.read_scalar(i) as f32).collect(),
        );
    }
    Some(out)
}

// ── Generic runner ────────────────────────────────────────────────────────
//
// Handles all BenchDispatch::Generic specs data-driven via ShapeSpec.
// Correctness via interpreter; perf via GPU.

fn run_generic(spec: &BenchSpec, runner: &GpuRunner, dt: DType, bench: &OpBench) -> Vec<OpResult> {
    // Cache compiled kernels by mode — MSL is identical for all shapes with the same
    // (dt, mode), so compile once instead of once-per-shape.
    let mut compiled: std::collections::HashMap<u8, crate::runner::CompiledKernel> =
        std::collections::HashMap::new();
    let mode_key = |m: KernelMode| match m {
        KernelMode::Elementwise => 0u8,
        KernelMode::Reduction | KernelMode::Tile2D => 1,
        KernelMode::Grid3D => 2,
    };

    let mut results = Vec::new();
    // Build the kernel IR once (same for all shapes at a given dt).
    let kernel = (spec.kernel_ir)(dt);
    // Pre-compile MLX ref kernel once (same MSL/function for all shapes).
    let mlx_compiled: Option<crate::runner::CompiledKernel> = {
        let ctx0 = DtypeCtx::reduce(dt); // tn is dtype-only, not shape-dependent
        compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, ctx0.tn)
    };
    // Pre-collect float literals from IR (same for all shapes at a given dt).
    let float_literal_pairs: Vec<(String, f32)> = {
        let mut srcs = Vec::new();
        for block in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for op in &block.ops {
                if let Op::Load { src, indices, .. } = op
                    && indices.is_empty()
                    && let Some(prefix) = src.strip_suffix('f')
                    && let Ok(v) = prefix.parse::<f64>()
                {
                    srcs.push((src.clone(), v as f32));
                }
            }
        }
        srcs
    };

    for shape in spec.shapes {
        let ctx = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D => DtypeCtx::reduce(dt),
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
        let params: Vec<_> = kernel.params.iter().collect();
        let check_n = shape.check_n;
        // Reduction-mode kernels: use a single row for correctness checks.
        // strided_reduce_dot uses ValueId(0) as an implicit-lsize sentinel;
        // program_id::<0>() also lands in ValueId(0). For rows ≥ 2, pid > 0
        // corrupts the stride. With check_b=1, pid is always 0, stride = max(0,1) = 1.
        let check_b = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D => 1,
            _ => shape.check_b,
        };

        // Build interpreter inputs
        let mut inp_map = BTreeMap::new();
        for (i, buf_spec) in shape.tensor_bufs.iter().enumerate() {
            let Some(param) = params.get(i) else { break };
            let count = buf_spec.count.resolve(check_n, check_b);
            let init_data = buf_spec.init.generate(count);
            let param_dt = buf_spec.dtype_override.unwrap_or(dt);
            inp_map.insert(param.name.clone(), td(param_dt, &[count], &init_data));
        }
        let cv_pairs: Vec<(&str, usize)> =
            shape.cexprs.iter().map(|(k, d)| (*k, d.resolve(check_n, check_b))).collect();
        let cv = constexprs(&cv_pairs);
        // Constexpr params are loaded via Op::Load { src: name } in the IR,
        // so they must also appear in inp_map as 1-element scalar tensors.
        for (name, val) in &cv_pairs {
            inp_map.insert(name.to_string(), td(DType::F32, &[1], &[*val as f32]));
        }
        // GPU built-ins and MSL special constants used as Op::Load { src: name, indices:[] }.
        // In single-threaded CPU interpretation: 1 thread per threadgroup.
        for (name, val) in [
            ("tid", 0.0f32),
            ("lsize", 1.0),
            ("tgid_x", 0.0),
            ("tgid_y", 0.0),
            ("simd_lane", 0.0),
            ("simd_id", 0.0),
            ("n_simd", 1.0),
            ("-INFINITY", f32::NEG_INFINITY),
            ("INFINITY", f32::INFINITY),
        ] {
            inp_map.entry(name.to_string()).or_insert_with(|| td(DType::F32, &[1], &[val]));
        }
        // Inject float literals collected from IR before the loop.
        for (src, val) in &float_literal_pairs {
            inp_map.entry(src.clone()).or_insert_with(|| td(DType::F32, &[1], &[*val]));
        }
        let interp_mode = match shape.mode {
            KernelMode::Elementwise | KernelMode::Grid3D => InterpMode::Elementwise(check_n),
            _ => InterpMode::Reduction(check_b.max(1)),
        };
        let interp_out = match interp(&kernel, inp_map, cv, interp_mode) {
            Some(o) => o,
            None => continue,
        };
        let primary_out_idx = params.iter().position(|p| p.is_output);
        let primary_out_name = match primary_out_idx.and_then(|i| params.get(i)) {
            Some(p) => p.name.clone(),
            None => continue,
        };
        let interp_vals = match interp_out.get(&primary_out_name) {
            Some(v) => v.clone(),
            None => continue,
        };

        // Build GPU check buffers
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
        let equiv = check_equiv(&interp_vals, &mt_vals, spec.tol);

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
        let mt_perf = bench_gbps(runner, mk, &perf_refs, perf_grid, [shape.tpg, 1, 1], bytes);

        // MLX ref (optional)
        let ref_perf = if let Some(mlx_args) = shape.mlx_args {
            let mlx_tpg = if shape.mlx_tpg > 0 { shape.mlx_tpg } else { shape.tpg };
            let mlx_grid = shape.mlx_grid.unwrap_or(shape.grid).eval(n, b, mlx_tpg);
            mlx_compiled.as_ref().and_then(|rk| {
                let mlx_bufs: Vec<GpuBuffer> = mlx_args
                    .iter()
                    .map(|arg| mlx_buf(spec, runner, arg, shape, n, b, dt))
                    .collect();
                let mlx_refs: Vec<&GpuBuffer> = mlx_bufs.iter().collect();
                bench_gbps(runner, rk, &mlx_refs, mlx_grid, [mlx_tpg, 1, 1], bytes)
            })
        } else {
            None
        };

        results.push(bench.result_sub(
            Some(spec.subop),
            format!("{} {}", shape.label, ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
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
    _dt: DType,
    bench: &OpBench,
    b: usize,
    n: usize,
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

    let check_b = 4usize;
    let check_data: Vec<f32> = (0..check_b * n).map(|i| (check_b * n - i) as f32).collect();
    let ref_out = {
        let mut out = check_data.clone();
        for chunk in out.chunks_mut(n) {
            chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        out
    };
    let inp_c = buffer_typed(runner, &check_data, DType::F32);
    let n_buf_c = runner.buffer_u32(n as u32);
    let out_c = zeros_typed(runner, check_b * n, DType::F32);
    let mt_chk = run_typed_once(
        runner,
        &mk,
        &[&inp_c, &out_c, &n_buf_c],
        &out_c,
        check_b * n,
        [check_b, 1, 1],
        [tpg, 1, 1],
        DType::F32,
    );
    let n_bad = ref_out.iter().zip(&mt_chk).filter(|(a, b)| a != b).count();
    let equiv = EquivResult {
        n_checked: check_b * n,
        max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
        cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
        passed: n_bad == 0,
    };

    let data: Vec<f32> = (0..b * n).map(|i| (b * n - i) as f32).collect();
    let inp = buffer_typed(runner, &data, DType::F32);
    let bytes = (b * n * 4 * 2) as f64;
    let n_buf = runner.buffer_u32(n as u32);

    let ref_perf = ref_kernel.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, b * n, DType::F32);
        let size = runner.buffer_i32(n as i32);
        let stride1 = runner.buffer_i32(1i32);
        let stride_n = runner.buffer_i32(n as i32);
        bench_gbps(
            runner,
            rk,
            &[&inp, &out, &size, &stride1, &stride1, &stride_n, &stride_n],
            [b, 1, 1],
            [tpg, 1, 1],
            bytes,
        )
    });
    let mt_perf = {
        let out = zeros_typed(runner, b * n, DType::F32);
        bench_gbps(runner, &mk, &[&inp, &out, &n_buf], [b, 1, 1], [tpg, 1, 1], bytes)
    };
    vec![bench.result_sub(
        Some(spec.subop),
        format!("B={b} N={n} f32"),
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
        let equiv = check_equiv(&ref_out, &mt_chk, spec.tol);

        let inp_buf = buffer_typed(runner, &inp_vals, DType::F32);
        let bytes = (rows * n * 8) as f64;
        let ns_u64 = runner.buffer_u64(n as u64);
        let ns_u32 = runner.buffer_u32(n as u32);
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps(runner, rk, &[&inp_buf, &out, &ns_u64], [1, rows, 1], [tpg, 1, 1], bytes)
        });
        let mt_perf = {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps(runner, &mk, &[&inp_buf, &out, &ns_u32], [1, rows, 1], [tpg, 1, 1], bytes)
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
        bench_gbps(
            runner,
            rk,
            &[&inp, &out, &dummy, &dummy, &dummy, &ndim, &ax_stride, &ax_size],
            [tpg, 1, 1],
            [tpg, 1, 1],
            bytes,
        )
    });
    let mt_out = zeros_typed(runner, 1, DType::F32);
    let mt_perf = bench_gbps(runner, &mk, &[&inp, &mt_out, &ns], [1, 1, 1], [tpg, 1, 1], bytes);
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
    let mt_perf =
        bench_gbps(runner, &mk, &[&mt_out, &n_buf], [n.div_ceil(tpg), 1, 1], [tpg, 1, 1], bytes);

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
        bench_gbps(
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
        bench_gbps(runner, &rk, &[&inp, &out], [1, n / 32, 1], [32, 1, 1], bytes)
    });
    let mt_perf = {
        let out = zeros_typed(runner, n, DType::F32);
        bench_gbps(runner, &mk, &[&inp, &out, &n_buf], [n / tpg, 1, 1], [tpg, 1, 1], bytes)
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
            bench_gbps(
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
            bench_gbps(
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
    let equiv = rk.as_ref().map(|rk| {
        let mk = &mk;
        let l_check = 4usize;
        let n_check = b * l_check * h * d;
        let check_f16: Vec<u16> = (0..n_check).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
        let inp_c = runner.buffer_f16(&check_f16);
        let ref_out_c = runner.buffer_zeros(n_check * 2);
        let mt_out_c = runner.buffer_zeros(n_check * 2);

        // MLX ref params: (in, out, offset[B], scale, strides[3], out_strides[3], offset_stride, n_head, dummy, dummy, base)
        let strides_bytes: Vec<u8> =
            [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let strides_buf = runner.buffer_bytes(&strides_bytes);
        let offset_arr = runner.buffer_i32(0i32);
        let scale_buf = runner.buffer_f32_scalar(1.0f32);
        let offset_stride_buf = runner.buffer_i64(1i64);
        let n_head_buf = runner.buffer_i32(h as i32);
        let dummy = runner.buffer_zeros(4);
        let base_buf = runner.buffer_f32_scalar(base_val);
        runner.measure(
            rk,
            &[
                &inp_c,
                &ref_out_c,
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
        let ref_vals = runner.read_f16_slice(&ref_out_c, n_check);

        // MT params: (inp, out, h_stride, seq_stride, grid_x, base)
        let mt_h_stride = runner.buffer_u32(d as u32);
        let mt_seq_stride = runner.buffer_u32((h * d) as u32);
        let mt_grid_x = runner.buffer_u32(gx as u32);
        let mt_base = runner.buffer_f32_scalar(base_val);
        runner.measure(
            mk,
            &[&inp_c, &mt_out_c, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
            [gx, l_check, gz],
            [1, 1, 1],
            0,
            1,
        );
        let mt_vals = runner.read_f16_slice(&mt_out_c, n_check);
        check_equiv(&ref_vals, &mt_vals, spec.tol)
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
        bench_gbps(
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
    let mt_perf = bench_gbps(
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
    let ref_name = match dt {
        DType::F32 => "sdpa_vector_float_128_128",
        DType::F16 => "sdpa_vector_float16_t_128_128",
        _ => return vec![],
    };
    let rk = spec
        .mlx_src
        .and_then(|src| runner.compile_with_bool_constants(src, ref_name, REF_FCS).ok());
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
        let (q_b, k_b, v_b, out_b, n_b, sc_b) = if dt == DType::F32 {
            let q_b = buffer_typed(runner, &cq, dt);
            let k_b = buffer_typed(runner, &ck_, dt);
            let v_b = buffer_typed(runner, &cv, dt);
            let out_b = zeros_typed(runner, ch * d, dt);
            let n_b = runner.buffer_u32(cn as u32);
            let sc_b = runner.buffer_f32_scalar(scale);
            (q_b, k_b, v_b, out_b, n_b, sc_b)
        } else {
            let f32_to_f16 = |v: f32| -> u16 {
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
                sign | ((exp as u16) << 10) | (mant16 as u16)
            };
            let q_f16: Vec<u16> = cq.iter().copied().map(f32_to_f16).collect();
            let k_f16: Vec<u16> = ck_.iter().copied().map(f32_to_f16).collect();
            let v_f16: Vec<u16> = cv.iter().copied().map(f32_to_f16).collect();
            let q_b = runner.buffer_f16(&q_f16);
            let k_b = runner.buffer_f16(&k_f16);
            let v_b = runner.buffer_f16(&v_f16);
            let out_b = runner.buffer_zeros(ch * d * 2);
            let n_b = runner.buffer_u32(cn as u32);
            let sc_b = runner.buffer_f32_scalar(scale);
            (q_b, k_b, v_b, out_b, n_b, sc_b)
        };
        runner.measure(
            &mk,
            &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b],
            [ch, 1, 1],
            [tpg, 1, 1],
            0,
            1,
        );
        let mt_chk = if dt == DType::F32 {
            runner.read_f32_slice(&out_b, ch * d)
        } else {
            runner.read_f16_slice(&out_b, ch * d)
        };
        let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(spec.tol, 0.999));

        let vals: Vec<f32> = (0..h * n_kv * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let bytes = (h * n_kv * d * ctx.eb * 2 + h * d * ctx.eb * 2) as f64;
        let (q_buf, k_buf, v_buf, n_buf, sc_buf) = if dt == DType::F32 {
            let qb = buffer_typed(runner, &vals[..h * d], dt);
            let kb = buffer_typed(runner, &vals[..h * n_kv * d], dt);
            let vb = buffer_typed(runner, &vals[..h * n_kv * d], dt);
            let nb = runner.buffer_u32(n_kv as u32);
            let sb = runner.buffer_f32_scalar(scale);
            (qb, kb, vb, nb, sb)
        } else {
            let f32_to_f16 = |v: f32| -> u16 {
                let x = v.to_bits();
                let sign = ((x >> 31) as u16) << 15;
                let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
                let mant16 = (x & 0x7F_FFFF) >> 13;
                if exp <= 0 {
                    return sign;
                }
                if exp >= 31 {
                    return sign | 0x7C00;
                }
                sign | ((exp as u16) << 10) | (mant16 as u16)
            };
            let qb = runner
                .buffer_f16(&vals[..h * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>());
            let kb = runner.buffer_f16(
                &vals[..h * n_kv * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
            );
            let vb = runner.buffer_f16(
                &vals[..h * n_kv * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
            );
            let nb = runner.buffer_u32(n_kv as u32);
            let sb = runner.buffer_f32_scalar(scale);
            (qb, kb, vb, nb, sb)
        };
        let ref_perf = rk.as_ref().and_then(|rk| {
            let gqa = runner.buffer_i32(1i32);
            let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * d) as u64);
            let kss = runner.buffer_u64(d as u64);
            let out = if dt == DType::F32 {
                zeros_typed(runner, h * d, dt)
            } else {
                runner.buffer_zeros(h * d * 2)
            };
            bench_gbps(
                runner,
                rk,
                &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [h, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        });
        let mt_perf = {
            let out = if dt == DType::F32 {
                zeros_typed(runner, h * d, dt)
            } else {
                runner.buffer_zeros(h * d * 2)
            };
            bench_gbps(
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
        bench_gbps(
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
        bench_gbps(
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

#[allow(dead_code)]
enum InterpMode {
    Elementwise(usize),
    Reduction(usize),
    Grid3D(usize, usize, usize),
}
