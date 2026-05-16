//! `tile build` — Compile all registered kernels to MSL and report errors.
//!
//! No GPU required. Fast enough for pre-push CI.

use std::collections::BTreeMap;

use metaltile_codegen::{
    TileSchedule,
    msl::{MslConfig, MslGenerator},
};
use metaltile_core::ir::KernelMode;
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    flag_val,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &[String]) {
    let filter = flag_val(args, "--filter").or_else(|| flag_val(args, "-f"));
    let dtypes_arg = flag_val(args, "--dtypes");
    let verbose = args.iter().any(|a| a == "-v" || a == "-vv");

    // Parse --dtypes list
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

    // Collect unique kernel specs.
    let mut kernels: BTreeMap<&str, (&BenchSpec, Vec<DType>)> = BTreeMap::new();
    for spec in inventory::iter::<BenchSpec> {
        let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }

    let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
    sorted.sort_unstable_by_key(|(name, _)| *name);

    let mut ok = 0u32;
    let mut errors = 0u32;

    for (name, (spec, dtypes)) in &sorted {
        if !matches_filter(filter.as_deref(), name) {
            continue;
        }

        // Filter dtypes if --dtypes was specified.
        let dtypes_to_check: Vec<DType> = match &dtypes_filter {
            Some(df) => dtypes.iter().filter(|dt| df.contains(dt)).copied().collect(),
            None => dtypes.clone(),
        };

        // Determine the kernel mode from dispatch.
        let mode = first_mode(spec, dtypes);

        let mut dtypes_ok = Vec::new();
        let mut dtypes_err = Vec::new();
        for &dt in &dtypes_to_check {
            let mut k = (spec.kernel_ir)(dt);
            k.mode = mode;

            let generator: MslGenerator = if matches!(mode, KernelMode::Tile2D) {
                MslGenerator::new(MslConfig {
                    tile_schedule: TileSchedule::default(),
                    use_simd_matrix: true,
                    ..MslConfig::default()
                })
            } else {
                MslGenerator::default()
            };

            match generator.generate(&k) {
                Ok(msl) => {
                    dtypes_ok.push(dt);
                    if verbose {
                        println!("// ══ {} {} ══\n{}", name, dtype_label(dt), msl,);
                    }
                },
                Err(e) => {
                    dtypes_err.push((dt, e.to_string()));
                    errors += 1;
                },
            }
        }

        if !dtypes_err.is_empty() {
            for (dt, err_msg) in &dtypes_err {
                eprintln!(
                    "  {}  {}   {}",
                    paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                    paint_stdout(dtype_label(*dt), Style::new().fg(Color::Blue).bold()),
                    paint_stderr("✗", Style::new().fg(Color::Red).bold()),
                );
                for line in err_msg.lines() {
                    eprintln!(
                        "                       {}",
                        paint_stderr(line, Style::new().fg(Color::BrightWhite)),
                    );
                }
            }
        } else if !dtypes_ok.is_empty() {
            ok += 1;
            let dtype_str =
                dtypes_ok.iter().map(|dt| dtype_label(*dt)).collect::<Vec<_>>().join("/");
            eprintln!(
                "  {}  {}   {}",
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                paint_stdout(&dtype_str, Style::new().fg(Color::Blue).bold()),
                paint_stdout("✓", Style::new().fg(Color::Green).bold()),
            );
        }
    }

    // Summary
    eprintln!();
    let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
    if errors > 0 {
        eprintln!(
            "  {} {sep} {}",
            paint_stdout(format!("{ok} ok"), Style::new().fg(Color::Green).bold()),
            paint_stderr(
                format!("{errors} error{}", if errors == 1 { "" } else { "s" }),
                Style::new().fg(Color::Red).bold()
            ),
        );
        std::process::exit(1);
    } else {
        eprintln!("  {}", paint_stdout(format!("{ok} ok"), Style::new().fg(Color::Green).bold()),);
    }
}

fn first_mode(spec: &BenchSpec, _dtypes: &[DType]) -> KernelMode {
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
