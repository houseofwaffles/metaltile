//! `tile test` — Correctness-only checks: interpreter ↔ GPU.
//!
//! No perf measurement, no SLC flush, no timing iterations.
//! Uses small shapes for speed. Roughly 20× faster than full bench.

use std::collections::BTreeMap;

use metaltile::core::{
    constexpr::ConstExprValues,
    dtype::DType,
    ir::{KernelMode, Op},
};
use metaltile_codegen::msl::MslGenerator;
use metaltile_interp::{Interpreter, TensorData};
use metaltile_std::{
    bench_types::{EquivResult, EquivTolerance, check_equiv_with, dtype_tol, dtype_tol_reduce},
    spec::BenchSpec,
};

use crate::{
    flag_val,
    matches_filter,
    measure::{buffer_typed, run_typed_once},
    runner::GpuRunner,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &[String]) {
    let filter = flag_val(args, "--filter").or_else(|| flag_val(args, "-f"));
    let dtypes_arg = flag_val(args, "--dtypes");
    let verbose = args.iter().any(|a| a == "-v" || a == "-vv");
    let fail_fast = args.iter().any(|a| a == "--fail-fast");

    let dtypes_filter: Option<Vec<DType>> = dtypes_arg.as_ref().map(|s| {
        s.split(',')
            .filter_map(|t| match t.trim() {
                "f32" => Some(DType::F32),
                "f16" => Some(DType::F16),
                "bf16" => Some(DType::BF16),
                _ => None,
            })
            .collect()
    });

    let runner = match GpuRunner::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(e, Style::new().fg(Color::BrightWhite)),
            );
            std::process::exit(1);
        },
    };

    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.op, s.subop));

    let mut passed = 0u32;
    let mut failed = 0u32;
    let names = collect_unique_names(&specs);

    for (name, (mode, dtypes_available)) in &names {
        if !matches_filter(filter.as_deref(), name) {
            continue;
        }

        let dtypes_to_check: Vec<DType> = match &dtypes_filter {
            Some(df) => dtypes_available.iter().filter(|dt| df.contains(dt)).copied().collect(),
            None => dtypes_available.clone(),
        };

        for (i, &dt) in dtypes_to_check.iter().enumerate() {
            let op_name = if i == 0 {
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold())
            } else {
                paint_stdout(format!("{:<20}", ""), Style::new().fg(Color::Cyan))
            };
            let dtype_str = paint_stdout(dtype_label(dt), Style::new().fg(Color::BrightBlack));

            // Find the spec for this name + dt
            let Some(spec) =
                specs.iter().find(|s| s.kernel_name == *name && s.dtypes.contains(&dt))
            else {
                continue;
            };

            match test_one_kernel(spec, &runner, dt, *mode) {
                Ok(equiv) => {
                    passed += 1;
                    let max_err_str = if equiv.max_abs_err < 1e-5 {
                        String::new()
                    } else {
                        format!("  {:.1e}", equiv.max_abs_err)
                    };
                    eprintln!(
                        "  {}  {}   {} {}",
                        op_name,
                        dtype_str,
                        paint_stdout("✓", Style::new().fg(Color::Green).bold()),
                        paint_stdout(&max_err_str, Style::new().fg(Color::BrightBlack)),
                    );
                    if verbose {
                        eprintln!(
                            "           n_checked={} cosine={:.6}",
                            equiv.n_checked, equiv.cosine_sim,
                        );
                    }
                },
                Err(err_msg) => {
                    failed += 1;
                    eprintln!(
                        "  {}  {}   {}  {}",
                        op_name,
                        dtype_str,
                        paint_stderr("✗", Style::new().fg(Color::Red).bold()),
                        paint_stderr(&err_msg, Style::new().fg(Color::BrightWhite)),
                    );
                    if fail_fast {
                        eprintln!();
                        eprintln!(
                            "  {} {}  ·  {}",
                            paint_stdout(
                                format!("{passed} passed"),
                                Style::new().fg(Color::Green).bold()
                            ),
                            paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()),
                            paint_stderr(
                                format!("{failed} failed"),
                                Style::new().fg(Color::Red).bold()
                            ),
                        );
                        std::process::exit(1);
                    }
                },
            }
        }
    }

    // Summary
    eprintln!();
    if failed > 0 {
        eprintln!(
            "  {} {}  ·  {}",
            paint_stdout(format!("{passed} passed"), Style::new().fg(Color::Green).bold()),
            paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()),
            paint_stderr(format!("{failed} failed"), Style::new().fg(Color::Red).bold()),
        );
        std::process::exit(1);
    } else {
        eprintln!(
            "  {}",
            paint_stdout(format!("{passed} passed"), Style::new().fg(Color::Green).bold()),
        );
    }
}

/// (mode, available_dtypes) per unique kernel name.
fn collect_unique_names(specs: &[&BenchSpec]) -> BTreeMap<&'static str, (KernelMode, Vec<DType>)> {
    let mut map: BTreeMap<&str, (KernelMode, Vec<DType>)> = BTreeMap::new();
    for spec in specs {
        let mode = first_mode(spec);
        let entry = map.entry(spec.kernel_name).or_insert_with(|| (mode, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }
    map
}

fn first_mode(spec: &BenchSpec) -> KernelMode {
    match &spec.dispatch {
        metaltile_std::spec::BenchDispatch::Generic =>
            spec.shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise),
        metaltile_std::spec::BenchDispatch::Sort { .. }
        | metaltile_std::spec::BenchDispatch::Scan { .. }
        | metaltile_std::spec::BenchDispatch::ArgReduce { .. }
        | metaltile_std::spec::BenchDispatch::QuantizedMatVec { .. }
        | metaltile_std::spec::BenchDispatch::Attention { .. } => KernelMode::Reduction,
        metaltile_std::spec::BenchDispatch::Random { .. }
        | metaltile_std::spec::BenchDispatch::FpQuantized { .. } => KernelMode::Elementwise,
        metaltile_std::spec::BenchDispatch::Rope { .. }
        | metaltile_std::spec::BenchDispatch::StridedCopy { .. } => KernelMode::Grid3D,
    }
}

fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "?",
    }
}

/// Run a single correctness check for one kernel × dtype.
/// Returns Ok(EquivResult) on success, Err(message) on failure.
fn test_one_kernel(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    mode: KernelMode,
) -> Result<EquivResult, String> {
    match &spec.dispatch {
        metaltile_std::spec::BenchDispatch::Generic => test_generic(spec, runner, dt, mode),
        // Complex dispatches: run the full benchmark path but just extract the equiv result.
        // This is less efficient than a correctness-only path but avoids duplicating
        // hundreds of lines of kernel-specific setup code.
        _ => {
            // Run the full spec — but only once, grab the equiv result.
            let results = crate::run_spec::run(spec, runner, dt);
            let equiv = results
                .first()
                .and_then(|r| r.equiv().copied())
                .ok_or_else(|| "no correctness result".to_string())?;
            Ok(equiv)
        },
    }
}

/// Correctness check for Generic dispatch (Unary, Binary, RowReduce, etc.).
fn test_generic(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    mode: KernelMode,
) -> Result<EquivResult, String> {
    let kernel = (spec.kernel_ir)(dt);
    let tol = match mode {
        KernelMode::Reduction | KernelMode::Tile2D => dtype_tol_reduce(dt),
        _ => dtype_tol(dt),
    };

    // Compile MSL
    let mut k = kernel.clone();
    k.mode = mode;
    let msl = MslGenerator::default().generate(&k).map_err(|e| format!("MSL gen failed: {e}"))?;
    let compiled =
        runner.compile(&msl, spec.kernel_name).map_err(|e| format!("compile failed: {e}"))?;

    // Use the first shape spec for correctness dimensions.
    let shape = spec.shapes.first().ok_or_else(|| "no shapes defined".to_string())?;

    let check_n = shape.check_n;
    let check_b = match mode {
        KernelMode::Reduction | KernelMode::Tile2D => 1,
        _ => shape.check_b,
    };

    // Build interpreter inputs
    let params: Vec<_> = kernel.params.iter().collect();
    let mut inp_map = BTreeMap::new();
    for (i, buf_spec) in shape.tensor_bufs.iter().enumerate() {
        let Some(param) = params.get(i) else { break };
        let count = buf_spec.count.resolve(check_n, check_b);
        let init_data = buf_spec.init.generate(count);
        let param_dt = buf_spec.dtype_override.unwrap_or(dt);
        inp_map.insert(param.name.clone(), tensor_data(param_dt, &[count], &init_data));
    }
    let cv_pairs: Vec<(&str, usize)> =
        shape.cexprs.iter().map(|(k, d)| (*k, d.resolve(check_n, check_b))).collect();
    let cv = constexprs(&cv_pairs);
    for (name, val) in &cv_pairs {
        inp_map.insert(name.to_string(), tensor_data(DType::F32, &[1], &[*val as f32]));
    }
    // GPU built-ins
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
        inp_map.entry(name.to_string()).or_insert_with(|| tensor_data(DType::F32, &[1], &[val]));
    }
    // Float literals
    for block in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
        for op in &block.ops {
            if let Op::Load { src, indices, .. } = op
                && indices.is_empty()
                && let Some(prefix) = src.strip_suffix('f')
                && let Ok(v) = prefix.parse::<f64>()
            {
                inp_map
                    .entry(src.clone())
                    .or_insert_with(|| tensor_data(DType::F32, &[1], &[v as f32]));
            }
        }
    }

    // Run interpreter
    let mut interp = Interpreter::new(inp_map, cv);
    let interp_out = match mode {
        KernelMode::Elementwise | KernelMode::Grid3D =>
            interp.run_grid(&k, check_n).map_err(|e| format!("interp failed: {e}"))?,
        _ => interp
            .run_grid_reduction(&k, check_b.max(1))
            .map_err(|e| format!("interp failed: {e}"))?,
    };

    let primary_out_idx = params.iter().position(|p| p.is_output);
    let primary_out_name = match primary_out_idx.and_then(|i| params.get(i)) {
        Some(p) => &p.name,
        None => return Err("no output parameter".to_string()),
    };
    let interp_vals = interp_out
        .outputs
        .get(primary_out_name)
        .map(|td| (0..td.num_elements()).map(|i| td.read_scalar(i) as f32).collect::<Vec<_>>())
        .ok_or_else(|| format!("output '{}' not found", primary_out_name))?;

    // Build GPU check buffers
    let mut check_bufs = Vec::new();
    for buf_spec in shape.tensor_bufs {
        let count = buf_spec.count.resolve(check_n, check_b);
        let init_data = buf_spec.init.generate(count);
        let param_dt = buf_spec.dtype_override.unwrap_or(dt);
        check_bufs.push(buffer_typed(runner, &init_data, param_dt));
    }
    // Scalar buffers (using the same approach as BenchSpec)
    for &sb in shape.scalar_bufs {
        check_bufs.push(scalar_buf(runner, sb, check_n, check_b));
    }

    let out_idx = primary_out_idx.unwrap_or(0);
    let out_count = shape.out_elems.resolve(check_n, check_b).max(1);
    let grid = shape.grid.eval(check_n, check_b, shape.tpg);
    let refs: Vec<&crate::runner::GpuBuffer> = check_bufs.iter().collect();
    let mt_vals = run_typed_once(
        runner,
        &compiled,
        &refs,
        &check_bufs[out_idx],
        out_count,
        grid,
        [shape.tpg, 1, 1],
        dt,
    );

    let equiv = check_equiv_with(&interp_vals, &mt_vals, EquivTolerance::new(tol, 0.999));
    if equiv.passed {
        Ok(equiv)
    } else {
        Err(format!(
            "max_err={:.2e}  cosine={:.4}  (tol={:.2e})",
            equiv.max_abs_err, equiv.cosine_sim, tol
        ))
    }
}

fn tensor_data(dt: DType, shape: &[usize], data: &[f32]) -> TensorData {
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

fn scalar_buf(
    runner: &GpuRunner,
    sb: metaltile_std::spec::ScalarBufSpec,
    n: usize,
    b: usize,
) -> crate::runner::GpuBuffer {
    match sb {
        metaltile_std::spec::ScalarBufSpec::U32N => runner.buffer_u32(n as u32),
        metaltile_std::spec::ScalarBufSpec::U32B => runner.buffer_u32(b as u32),
        metaltile_std::spec::ScalarBufSpec::U64N => runner.buffer_u64(n as u64),
        metaltile_std::spec::ScalarBufSpec::U64B => runner.buffer_u64(b as u64),
        metaltile_std::spec::ScalarBufSpec::I64B => runner.buffer_i64(b as i64),
    }
}
