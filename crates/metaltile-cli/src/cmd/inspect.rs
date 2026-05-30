//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile inspect` — Print IR and/or MSL for kernels.
//!
//! Auto-discovers kernels via the `#[bench]` inventory — no hardcoded list.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL (default)
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> --stats          # print per-pass op-count table
//!   tile inspect <kernel> -o /tmp/out      # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk

use std::{collections::BTreeMap, str::FromStr};

use metaltile_codegen::generator_for_mode;
use metaltile_core::{all_benches, bench::KernelBench};
use metaltile_std::bench_types::DType;

use crate::{
    CliError,
    InspectArgs,
    matches_filter,
    term::{Color, Style, paint_stdout},
};

/// One registered kernel: the `#[bench]` that carries its IR + the union of
/// dtypes it is benched at.
type InspectKernel = (&'static dyn KernelBench, Vec<DType>);

pub fn run(args: &InspectArgs) -> Result<(), CliError> {
    let filter_val = args.filter.as_ref().or(args.kernel.as_ref());
    let _span = tracing::info_span!(
        "inspect",
        filter = ?filter_val,
        ir = args.ir,
        stats = args.stats,
    )
    .entered();
    let dir = &args.dir;
    // filter is either --filter flag or the positional kernel name
    let filter = filter_val;
    let all_flag = args.all;
    let ir_flag = args.ir;
    let stats_flag = args.stats;
    let pass_arg = &args.pass;
    let dtype_override: Option<DType> = args.dtype.as_deref().and_then(|s| DType::from_str(s).ok());

    // Collect all #[bench]-registered kernels, grouped by IR kernel name. The
    // bench's `setup(dt)` carries the kernel IR (mode applied via `.mode()`)
    // and the threadgroup geometry — everything inspect needs — so this no
    // longer depends on the legacy `BenchSpec` inventory.
    let mut kernels: BTreeMap<String, InspectKernel> = BTreeMap::new();
    for entry in all_benches() {
        let b = entry.bench();
        let Some(&first_dt) = b.dtypes().first() else { continue };
        let name = b.setup(first_dt).kernel().name.to_string();
        let e = kernels.entry(name).or_insert((b, Vec::new()));
        for &dt in b.dtypes() {
            if !e.1.contains(&dt) {
                e.1.push(dt);
            }
        }
    }

    if kernels.is_empty() {
        eprintln!("No kernels registered.");
        return Ok(());
    }

    let mut sorted: Vec<(String, InspectKernel)> = kernels.into_iter().collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    // --all flag: dump every kernel
    if all_flag {
        for (name, (b, dtypes)) in &sorted {
            let dt = dtypes.first().copied().unwrap_or(DType::F32);
            if ir_flag {
                let k = b.setup(dt).kernel().clone();
                if let Some(d) = dir {
                    let path = format!("{}/{}.ir", d, name);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                    println!("wrote {path}");
                } else {
                    println!("{k}");
                }
            } else {
                let msl = generate_msl_dt(*b, name, dt);
                if let Some(d) = dir {
                    let path = format!("{}/{}.metal", d, name);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, &msl).map_err(CliError::Io)?;
                    println!("wrote {path}");
                } else {
                    let mode_str = b.setup(dt).kernel().mode.to_string();
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("// kernel: {}  mode: {}", name, mode_str);
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("{}", msl);
                }
            }
        }
        return Ok(());
    }

    // No filter: list all kernels
    let Some(filter) = filter else {
        eprintln!("{}", paint_stdout("tile inspect", Style::new().fg(Color::Cyan).bold()),);
        eprintln!();
        for (name, (b, dtypes)) in &sorted {
            let dtype_str = dtypes.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/");
            let dt0 = dtypes.first().copied().unwrap_or(DType::F32);
            let mode_str = b.setup(dt0).kernel().mode.to_string();
            eprintln!(
                "  {}   {}   {dtype_str}",
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                paint_stdout(mode_str, Style::new().fg(Color::BrightBlack)),
            );
        }
        let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
        eprintln!();
        eprintln!(
            "  {} {sep} {}",
            paint_stdout(format!("{} kernels", sorted.len()), Style::new().fg(Color::BrightBlack)),
            paint_stdout("<kernel> for MSL", Style::new().fg(Color::BrightBlack)),
        );
        return Ok(());
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
                sorted.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        return Err(CliError::Other(format!("no kernel matched '{filter}'")));
    }

    for (name, (b, dtypes)) in &matched {
        let dt = dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));

        if ir_flag {
            // Print raw IR via Display impl
            let k = b.setup(dt).kernel().clone();
            if let Some(d) = dir {
                let path = format!("{}/{}.ir", d, name);
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                println!("wrote {path}");
            } else {
                println!("{k}");
            }
        } else if stats_flag {
            let setup = b.setup(dt);
            let k = setup.kernel().clone();
            let expected_tpg = Some(setup.grid().tpg[0]);
            let generator = generator_for_mode(k.mode, expected_tpg);
            match generator.generate_with_stats(&k) {
                Ok((_, stats)) => print_stats_table(&stats),
                Err(e) => eprintln!("error: {e}"),
            }
        } else if let Some(pass) = pass_arg {
            // --pass flag: print IR after a specific pass (or 'all' for every stage)
            let mut k = b.setup(dt).kernel().clone();

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
                            return Ok(());
                        }
                        println!("// ── AFTER {name} ────────────────────────");
                        println!("{k}");
                    },
                    None => {
                        let valid: Vec<_> = metaltile_codegen::passes::PassRegistry::names();
                        eprintln!("Unknown pass: {name}. Valid: {} all", valid.join(", "));
                        return Ok(());
                    },
                },
            }
        } else {
            // Default: print MSL
            let eff_dt =
                dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));
            let msl = generate_msl_dt(*b, name, eff_dt);
            if let Some(d) = dir {
                let path = format!("{}/{}.metal", d, name);
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, &msl).map_err(CliError::Io)?;
                println!("wrote {path}");
            } else {
                let mode_str = b.setup(eff_dt).kernel().mode.to_string();
                println!("// ═══════════════════════════════════════════════════════");
                println!("// kernel: {}  mode: {}", name, mode_str);
                println!("// ═══════════════════════════════════════════════════════");
                println!("{}", msl);
            }
        }
    }
    Ok(())
}

/// Run all compilation passes and print IR after each stage.
fn run_all_passes_and_print(k: &mut metaltile_core::ir::Kernel) {
    use metaltile_codegen::msl::MslGenerator;

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

/// Generate the final MSL a kernel emits, at dtype `dt`. Mirrors the bench /
/// emit path: the IR comes from the `#[bench]` setup (mode already applied),
/// and `mt_qmm_mma` gets the same dtype-aware-skew patch the bench compiles.
fn generate_msl_dt(b: &dyn KernelBench, name: &str, dt: DType) -> String {
    let setup = b.setup(dt);
    let mut k = setup.kernel().clone();
    if name == "mt_qmm_mma" {
        metaltile_std::mlx::quantized::patch_qmm_mma_dtype_aware_skew(&mut k, dt);
    }
    let expected_tpg = Some(setup.grid().tpg[0]);
    generator_for_mode(k.mode, expected_tpg)
        .generate(&k)
        .unwrap_or_else(|e| format!("// ERROR: {e}\n"))
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
