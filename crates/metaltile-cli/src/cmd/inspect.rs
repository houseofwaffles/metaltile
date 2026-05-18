//! `tile inspect` — Print IR and/or MSL for kernels.
//!
//! Auto-discovers kernels via inventory — no hardcoded list.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL (default)
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> --stats          # print per-pass op-count table
//!   tile inspect <kernel> -o /tmp/out      # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk

use std::collections::BTreeMap;

use metaltile_codegen::{
    TileSchedule,
    msl::{MslConfig, MslGenerator},
};
use metaltile_core::ir::KernelMode;
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    flag_present,
    flag_val,
    kernel_utils::{dtype_label, effective_mode},
    matches_filter,
    positional,
    term::{Color, Style, paint_stdout},
};

pub fn help() {
    eprintln!("tile inspect — Print IR and/or MSL for registered kernels");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  tile inspect                    List all registered kernels");
    eprintln!("  tile inspect <kernel>           Print final MSL (default)");
    eprintln!("  tile inspect <kernel> --ir      Print raw IR before any passes");
    eprintln!("  tile inspect <kernel> --stats   Print per-pass op-count table");
    eprintln!("  tile inspect <kernel> --pass <name|all>  Print IR after a specific pass");
    eprintln!("  tile inspect --all -o <dir>     Dump all kernels to .metal files");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --ir               Print raw IR");
    eprintln!("  --stats            Print per-pass op-count reduction table");
    eprintln!("  --pass <name>      Print IR after named pass; use 'all' for every stage");
    eprintln!("  --all              Process all kernels");
    eprintln!("  --dir, -o <path>   Write output files to <path> instead of stdout");
    eprintln!("  --filter <name>    Filter kernels by name substring");
}

pub fn run(args: &[String]) {
    let dir = flag_val(args, "--dir").or_else(|| flag_val(args, "-o"));
    let filter = flag_val(args, "--filter").or_else(|| positional(args));
    let all_flag = flag_present(args, "--all");
    let ir_flag = flag_present(args, "--ir");
    let stats_flag = flag_present(args, "--stats");
    let pass_arg = flag_val(args, "--pass");
    let dtype_override: Option<DType> = flag_val(args, "--dtype").and_then(|s| match s.as_str() {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        "i32" => Some(DType::I32),
        "u32" => Some(DType::U32),
        _ => None,
    });

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
                    let mode_str = mode_label(effective_mode(spec));
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
            let mode_str = mode_label(effective_mode(spec));
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
        let dt = dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));

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
        } else if stats_flag {
            let mut k = (spec.kernel_ir)(dt);
            k.mode = effective_mode(spec);
            let generator = make_generator(effective_mode(spec));
            match generator.generate_with_stats(&k) {
                Ok((_, stats)) => print_stats_table(&stats),
                Err(e) => eprintln!("error: {e}"),
            }
        } else if let Some(pass) = &pass_arg {
            // --pass flag: print IR after a specific pass (or 'all' for every stage)
            let mut k = (spec.kernel_ir)(dt);
            let mode = effective_mode(spec);
            k.mode = mode;

            match pass.as_str() {
                "all" => {
                    println!("// ── BEFORE PASSES ───────────────────────────");
                    println!("{k}");
                    run_all_passes_and_print(&mut k);
                },
                name => match metaltile_codegen::passes::PassRegistry::get(name) {
                    Some(pass_obj) => {
                        if let Err(e) = pass_obj.run(&mut k) {
                            eprintln!("Pass {name} failed: {e}");
                            return;
                        }
                        println!("// ── AFTER {name} ────────────────────────");
                        println!("{k}");
                    },
                    None => {
                        let valid: Vec<_> = metaltile_codegen::passes::PassRegistry::names();
                        eprintln!("Unknown pass: {name}. Valid: {} all", valid.join(", "));
                        return;
                    },
                },
            }
        } else {
            // Default: print MSL
            let eff_dt =
                dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));
            let msl = generate_msl_dt(spec, eff_dt);
            if let Some(ref d) = dir {
                let path = format!("{}/{}.metal", d, name);
                std::fs::create_dir_all(d).expect("failed to create output directory");
                std::fs::write(&path, &msl).expect("write failed");
                println!("wrote {path}");
            } else {
                let mode_str = mode_label(effective_mode(spec));
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
    let passes = metaltile_codegen::passes::PassRegistry::standard_with_names();

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

fn mode_label(mode: KernelMode) -> &'static str {
    match mode {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Tile2D => "Tile2D",
        KernelMode::SimdGroup2D => "SimdGroup",
        KernelMode::Grid3D => "Grid3D",
    }
}

fn make_generator(mode: KernelMode) -> MslGenerator {
    if matches!(mode, KernelMode::Tile2D) {
        MslGenerator::new(MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: true,
            ..MslConfig::default()
        })
    } else {
        MslGenerator::default()
    }
}

fn generate_msl(spec: &BenchSpec, dtypes: &[DType]) -> String {
    generate_msl_dt(spec, dtypes.first().copied().unwrap_or(DType::F32))
}

fn generate_msl_dt(spec: &BenchSpec, dt: DType) -> String {
    let mut k = (spec.kernel_ir)(dt);
    let mode = effective_mode(spec);
    k.mode = mode;
    make_generator(mode).generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"))
}

fn print_stats_table(stats: &[metaltile_codegen::passes::PassStats]) {
    println!(
        "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
        "pass", "ops_before", "ops_after", "delta", "time_us"
    );
    println!("{:-<20}  {:->10}  {:->9}  {:->6}  {:->7}", "", "", "", "", "");
    for s in stats {
        let delta = s.ops_after as isize - s.ops_before as isize;
        let delta_str = if delta == 0 { "  +0".to_string() } else { format!("{:>+4}", delta) };
        println!(
            "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
            s.name, s.ops_before, s.ops_after, delta_str, s.wall_us
        );
    }
}
