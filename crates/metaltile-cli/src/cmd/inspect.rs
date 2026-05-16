//! `tile inspect` — Print IR and/or MSL for kernels.
//!
//! Auto-discovers kernels via inventory — no hardcoded list.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL (default)
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> -o /tmp/out      # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk

use std::collections::BTreeMap;

use metaltile_codegen::{
    TileSchedule,
    msl::{MslConfig, MslGenerator},
    passes::{self, Pass},
};
use metaltile_core::ir::KernelMode;
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    flag_present,
    flag_val,
    matches_filter,
    positional,
    term::{Color, Style, paint_stdout},
};

pub fn run(args: &[String]) {
    let dir = flag_val(args, "--dir").or_else(|| flag_val(args, "-o"));
    let filter = flag_val(args, "--filter").or_else(|| positional(args));
    let all_flag = flag_present(args, "--all");
    let ir_flag = flag_present(args, "--ir");
    let pass_arg = flag_val(args, "--pass");

    // Collect all specs and group by kernel_name.
    let mut kernels: BTreeMap<&str, (&BenchSpec, Vec<DType>)> = BTreeMap::new();
    for spec in inventory::iter::<BenchSpec> {
        let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }

    if kernels.is_empty() {
        eprintln!("No kernels registered.");
        return;
    }

    let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
    sorted.sort_unstable_by_key(|(name, _)| *name);

    // --all flag: dump every kernel
    if all_flag {
        for (name, (spec, dtypes)) in &sorted {
            let dt = dtypes.first().copied().unwrap_or(DType::F32);
            if ir_flag {
                let k = (spec.kernel_ir)(dt);
                if let Some(ref d) = dir {
                    let path = format!("{}/{}.ir", d, name);
                    std::fs::create_dir_all(d).expect("failed to create output directory");
                    std::fs::write(&path, format!("{k}")).expect("write failed");
                    println!("wrote {path}");
                } else {
                    println!("{k}");
                }
            } else {
                let msl = generate_msl(spec, dtypes);
                if let Some(ref d) = dir {
                    let path = format!("{}/{}.metal", d, name);
                    std::fs::create_dir_all(d).expect("failed to create output directory");
                    std::fs::write(&path, &msl).expect("write failed");
                    println!("wrote {path}");
                } else {
                    let mode_str = mode_label(first_mode(spec, dtypes));
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("// kernel: {}  mode: {}", name, mode_str);
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("{}", msl);
                }
            }
        }
        return;
    }

    // No filter: list all kernels
    let Some(filter) = &filter else {
        println!(
            "{}",
            paint_stdout("Kernels registered:", Style::new().fg(Color::BrightBlack).bold()),
        );
        println!();
        for (name, (spec, dtypes)) in &sorted {
            let dtype_str = dtypes.iter().map(|dt| dtype_label(*dt)).collect::<Vec<_>>().join("/");
            let mode_str = mode_label(first_mode(spec, dtypes));
            println!(
                "  {}   {}   {dtype_str}",
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                paint_stdout(mode_str, Style::new().fg(Color::BrightBlack)),
            );
        }
        println!(
            "\n  {}",
            paint_stdout(format!("{} kernels", sorted.len()), Style::new().fg(Color::BrightBlack),),
        );
        println!(
            "  {}",
            paint_stdout(
                "Run 'tile inspect <kernel>' to see MSL.",
                Style::new().fg(Color::BrightBlack),
            ),
        );
        return;
    };

    // Filter by kernel name
    let matched: Vec<_> =
        sorted.iter().filter(|(name, _)| matches_filter(Some(filter), name)).collect();

    if matched.is_empty() {
        eprintln!(
            "{} {}",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
            paint_stdout(
                format!("no kernel matched '{filter}'"),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        eprintln!(
            "\n{} {}",
            paint_stdout("Available:", Style::new().fg(Color::BrightBlack)),
            paint_stdout(
                sorted.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", "),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        std::process::exit(1);
    }

    for (name, (spec, dtypes)) in &matched {
        let dt = dtypes.first().copied().unwrap_or(DType::F32);

        if ir_flag {
            // Print raw IR via Display impl
            let k = (spec.kernel_ir)(dt);
            if let Some(ref d) = dir {
                let path = format!("{}/{}.ir", d, name);
                std::fs::create_dir_all(d).expect("failed to create output directory");
                std::fs::write(&path, format!("{k}")).expect("write failed");
                println!("wrote {path}");
            } else {
                println!("{k}");
            }
        } else if let Some(pass) = &pass_arg {
            // --pass flag: print IR after a specific pass (or 'all' for every stage)
            let mut k = (spec.kernel_ir)(dt);
            let mode = first_mode(spec, dtypes);
            k.mode = mode;

            match pass.as_str() {
                "all" => {
                    println!("// ── BEFORE PASSES ───────────────────────────");
                    println!("{k}");
                    run_all_passes_and_print(&mut k);
                },
                name => {
                    let pass_obj: Box<dyn Pass> = match name {
                        "type_check" => Box::new(passes::type_check::TypeCheckPass),
                        "const_fold" => Box::new(passes::const_fold::ConstFoldPass::new()),
                        "tile_lowering" =>
                            Box::new(passes::tile_lowering::TileLoweringPass::default()),
                        "fusion" => Box::new(passes::fusion::FusionPass),
                        "schedule" => Box::new(passes::schedule::SchedulePass::default()),
                        "vectorize" => Box::new(passes::vectorize::VectorizePass),
                        _ => {
                            eprintln!(
                                "Unknown pass: {name}. Valid: type_check, const_fold, tile_lowering, fusion, schedule, vectorize, all"
                            );
                            return;
                        },
                    };
                    if let Err(e) = pass_obj.run(&mut k) {
                        eprintln!("Pass {name} failed: {e}");
                        return;
                    }
                    println!("// ── AFTER {name} ────────────────────────");
                    println!("{k}");
                },
            }
        } else {
            // Default: print MSL
            let msl = generate_msl(spec, dtypes);
            if let Some(ref d) = dir {
                let path = format!("{}/{}.metal", d, name);
                std::fs::create_dir_all(d).expect("failed to create output directory");
                std::fs::write(&path, &msl).expect("write failed");
                println!("wrote {path}");
            } else {
                let mode_str = mode_label(first_mode(spec, dtypes));
                println!("// ═══════════════════════════════════════════════════════");
                println!("// kernel: {}  mode: {}", name, mode_str);
                println!("// ═══════════════════════════════════════════════════════");
                println!("{}", msl);
            }
        }
    }
}

/// Run all compilation passes and print IR after each stage.
fn run_all_passes_and_print(k: &mut metaltile_core::ir::Kernel) {
    let passes: Vec<(&str, Box<dyn Pass>)> = vec![
        ("type_check", Box::new(passes::type_check::TypeCheckPass)),
        ("const_fold", Box::new(passes::const_fold::ConstFoldPass::new())),
        ("tile_lowering", Box::new(passes::tile_lowering::TileLoweringPass::default())),
        ("fusion", Box::new(passes::fusion::FusionPass)),
        ("schedule", Box::new(passes::schedule::SchedulePass::default())),
        ("vectorize", Box::new(passes::vectorize::VectorizePass)),
    ];

    for (name, pass) in &passes {
        if let Err(e) = pass.run(k) {
            println!("\n// ── AFTER {name} ──────── ERROR ──");
            println!("// {e}");
            return;
        }
        println!("\n// ── AFTER {name} ────────────────────────");
        println!("{k}");
    }

    // Generate final MSL
    let generator = MslGenerator::default();
    match generator.generate(k) {
        Ok(msl) => {
            println!("\n// ── FINAL MSL ───────────────────────────────");
            println!("{msl}");
        },
        Err(e) => {
            println!("\n// ── MSL ERROR ───────────────────────────────");
            println!("// {e}");
        },
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

fn mode_label(mode: KernelMode) -> &'static str {
    match mode {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Tile2D => "Tile2D",
        KernelMode::Grid3D => "Grid3D",
    }
}

fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        _ => "?",
    }
}

fn generate_msl(spec: &BenchSpec, dtypes: &[DType]) -> String {
    let dt = dtypes.first().copied().unwrap_or(DType::F32);
    let mut k = (spec.kernel_ir)(dt);
    let mode = first_mode(spec, dtypes);
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

    generator.generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"))
}
