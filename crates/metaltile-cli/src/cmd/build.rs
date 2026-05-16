//! `tile build` — Compile all registered kernels.
//!
//! Default behavior is a compile-check (codegen MSL, report errors,
//! no I/O). With `--emit <list> --out <dir>` it also writes artifacts:
//!
//!   --emit msl       Write per-kernel `<dir>/Resources/kernels/<name>.metal`
//!   --emit metallib  Compile + write `<dir>/Resources/kernels.metallib`
//!                    (implies msl)
//!   --emit swift     Write `<dir>/Generated/MetalTileKernels.swift`
//!                    dispatch wrappers
//!   --emit ir        Write `<dir>/Resources/manifest.json` IR descriptor
//!   --emit all       Shorthand for msl,metallib,swift,ir
//!
//! Multiple kinds may be combined via comma: `--emit msl,swift,ir`.
//!
//! The output layout intentionally matches a Swift Package's
//! `Sources/<Target>/{Resources,Generated}/` convention so a consumer
//! can point `--out` at their target directory and have the artifacts
//! land in the right place for SwiftPM resource bundling.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use metaltile_codegen::{
    TileSchedule,
    emit::{
        self,
        compile_metallib,
        dtype_suffix,
        write_manifest,
        write_msl,
        write_swift_wrappers,
    },
    msl::{MslConfig, MslGenerator},
};
use metaltile_core::ir::{Kernel, KernelMode};
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    flag_val,
    kernel_utils::{dtype_label, first_mode},
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EmitKind {
    Msl,
    Metallib,
    Swift,
    Ir,
}

fn parse_emit_list(raw: &str) -> Result<BTreeSet<EmitKind>, String> {
    let mut kinds = BTreeSet::new();
    for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match tok {
            "msl" => {
                kinds.insert(EmitKind::Msl);
            },
            "metallib" => {
                kinds.insert(EmitKind::Metallib);
                kinds.insert(EmitKind::Msl); // metallib needs the .metal source files on disk
            },
            "swift" => {
                kinds.insert(EmitKind::Swift);
            },
            "ir" => {
                kinds.insert(EmitKind::Ir);
            },
            "all" => {
                kinds.insert(EmitKind::Msl);
                kinds.insert(EmitKind::Metallib);
                kinds.insert(EmitKind::Swift);
                kinds.insert(EmitKind::Ir);
            },
            other => return Err(format!("unknown --emit kind '{other}'")),
        }
    }
    Ok(kinds)
}

pub fn run(args: &[String]) {
    let filter = flag_val(args, "--filter").or_else(|| flag_val(args, "-f"));
    let dtypes_arg = flag_val(args, "--dtypes");
    let verbose = args.iter().any(|a| a == "-v" || a == "-vv");
    let emit_arg = flag_val(args, "--emit");
    let out_arg = flag_val(args, "--out").or_else(|| flag_val(args, "-o"));
    let sdk = flag_val(args, "--sdk").unwrap_or_else(|| "macosx".to_string());

    let emit_kinds: BTreeSet<EmitKind> = match emit_arg.as_deref() {
        None => BTreeSet::new(),
        Some(raw) => match parse_emit_list(raw) {
            Ok(k) => k,
            Err(e) => {
                eprintln!(
                    "  {} {}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                    paint_stderr(e, Style::new().fg(Color::BrightWhite)),
                );
                eprintln!("  valid kinds: msl, metallib, swift, ir, all");
                std::process::exit(1);
            },
        },
    };

    let out_root: Option<PathBuf> = match (&emit_kinds.is_empty(), &out_arg) {
        (true, _) => None,
        (false, Some(p)) => Some(PathBuf::from(p)),
        (false, None) => {
            eprintln!(
                "  {} {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                paint_stderr("--emit requires --out <dir>", Style::new().fg(Color::BrightWhite)),
            );
            std::process::exit(1);
        },
    };

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

    // Per-output collectors for the emit step.
    let kernels_dir = out_root.as_ref().map(|r| r.join("Resources").join("kernels"));
    if let Some(dir) = &kernels_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!(
                "  {} create {}: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                dir.display(),
                e
            );
            std::process::exit(1);
        }
    }
    let mut emitted_kernels: Vec<Kernel> = Vec::new();
    let mut emitted_paths: Vec<PathBuf> = Vec::new();

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

        // Determine the kernel mode. Explicit override beats inference.
        let mode = spec.kernel_mode.unwrap_or_else(|| first_mode(spec));

        let mut dtypes_ok = Vec::new();
        let mut dtypes_err = Vec::new();
        for &dt in &dtypes_to_check {
            let mut k = (spec.kernel_ir)(dt);
            k.mode = mode;
            // Monomorphize per-dtype name (e.g. `mt_add` → `mt_add_f32`),
            // unless the spec is already dtype-specialized (e.g. `mt_argmax_f32`).
            k.name = monomorphized_name(spec.kernel_name, dt, dtypes.len());

            let generator = msl_generator_for(mode);

            // Compile-check via generate.
            let msl_result = generator.generate(&k);
            match msl_result {
                Ok(_) => {
                    dtypes_ok.push(dt);
                },
                Err(e) => {
                    dtypes_err.push((dt, format!("{e:?}")));
                    errors += 1;
                    continue;
                },
            }

            // Emit on success.
            if let Some(dir) = &kernels_dir
                && emit_kinds.contains(&EmitKind::Msl)
            {
                match write_msl(&k, dir, &generator) {
                    Ok(path) => emitted_paths.push(path),
                    Err(e) => {
                        eprintln!(
                            "  {} emit msl for {}: {}",
                            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                            k.name,
                            e
                        );
                        std::process::exit(1);
                    },
                }
            }
            if !emit_kinds.is_empty() {
                emitted_kernels.push(k.clone());
            }

            if verbose {
                if let Ok(msl) = generator.generate(&k) {
                    println!("// ══ {} {} ══\n{}", k.name, dtype_label(dt), msl);
                }
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

    // ─── Emit pass (manifest, Swift wrappers, metallib) ─────────────────
    if let Some(out) = &out_root {
        emit_artifacts(out, &emit_kinds, &emitted_kernels, &emitted_paths, &sdk);
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

/// Build the per-dtype monomorphized kernel symbol name.
///
/// `mt_add` (2+ dtypes) → `mt_add_f32` / `mt_add_f16` / `mt_add_bf16`
/// `mt_argmax_f32` (1 dtype, name already has suffix) → `mt_argmax_f32`
fn monomorphized_name(base: &str, dt: DType, n_dtypes: usize) -> String {
    let suffix = dtype_suffix(dt);
    if n_dtypes == 1 && base.ends_with(&format!("_{suffix}")) {
        base.to_string()
    } else {
        format!("{base}_{suffix}")
    }
}

fn msl_generator_for(mode: KernelMode) -> MslGenerator {
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

fn emit_artifacts(
    out_root: &Path,
    kinds: &BTreeSet<EmitKind>,
    kernels: &[Kernel],
    metal_files: &[PathBuf],
    sdk: &str,
) {
    let resources_dir = out_root.join("Resources");
    let generated_dir = out_root.join("Generated");

    if kinds.contains(&EmitKind::Ir)
        && let Err(e) = std::fs::create_dir_all(&resources_dir).and_then(|_| {
            write_manifest(kernels, &resources_dir.join("manifest.json"))
                .map_err(|e| std::io::Error::other(e.to_string()))
        })
    {
        eprintln!(
            "  {} write manifest.json: {}",
            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
            e
        );
        std::process::exit(1);
    }

    if kinds.contains(&EmitKind::Swift) {
        if let Err(e) = std::fs::create_dir_all(&generated_dir) {
            eprintln!(
                "  {} create {}: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                generated_dir.display(),
                e
            );
            std::process::exit(1);
        }
        let path = generated_dir.join("MetalTileKernels.swift");
        if let Err(e) = write_swift_wrappers(kernels, &path) {
            eprintln!(
                "  {} write {}: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                path.display(),
                e
            );
            std::process::exit(1);
        }
    }

    if kinds.contains(&EmitKind::Metallib) {
        let metallib_path = resources_dir.join("kernels.metallib");
        let air_dir = std::env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target"))
            .join("tile-build-air");
        if let Err(e) = compile_metallib(metal_files, &metallib_path, sdk, &air_dir) {
            eprintln!(
                "  {} compile metallib: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                e
            );
            std::process::exit(1);
        }
    }
    let _ = emit::dtype_suffix; // anchor public API surface for re-export checks
}
