//! `tile bench` — Benchmark suite: MetalTile vs MLX reference.

use std::collections::HashMap;

use metaltile_codegen::passes::{
    self,
    occupancy::{self, Bottleneck},
};
use metaltile_std::{
    bench_types::{CorrectnessStatus, OpResult, set_result_reporter, validate_results},
    run_spec::run as run_spec,
    runner::GpuRunner,
    spec::BenchSpec,
};
use serde_json::Value;

use crate::{
    BenchArgs,
    cmd::diff as diff_cmd,
    git,
    matches_filter,
    suite_printer::{ProfileRow, SuitePrinter},
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &BenchArgs) -> Result<(), crate::CliError> {
    let _span =
        tracing::info_span!("bench", filter = ?args.filter, verbose = args.verbose).entered();
    let json_out = &args.json;
    let filter = &args.filter;
    let verbose = args.verbose;

    // Refuse to bench on a dirty tree: a stale `target/` binary against
    // a dirty source tree silently decouples the numbers from any
    // commit SHA we'd record in a snapshot. `working_tree_dirty()`
    // returns None outside a repo — skip the check there.
    if !args.allow_dirty
        && let Some(true) = git::working_tree_dirty()
    {
        let files = git::list_dirty_files();
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                "working tree has uncommitted changes; bench numbers \
                 would not tie back to a clean commit.",
                Style::new().fg(Color::BrightWhite),
            ),
        );
        if !files.is_empty() {
            let preview: Vec<&str> = files.iter().take(8).map(String::as_str).collect();
            let overflow = if files.len() > 8 {
                format!(" (+{} more)", files.len() - 8)
            } else {
                String::new()
            };
            eprintln!(
                "  {} {}{}",
                paint_stderr("Dirty:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(preview.join(", "), Style::new().fg(Color::BrightWhite)),
                overflow,
            );
        }
        eprintln!(
            "  {} {}",
            paint_stderr("Override:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stderr(
                "re-run with --allow-dirty to bench anyway.",
                Style::new().fg(Color::BrightBlack),
            ),
        );
        return Err(crate::CliError::Other("uncommitted changes".into()));
    }

    let runner = match GpuRunner::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(&e, Style::new().fg(Color::BrightWhite)),
            );
            return Err(crate::CliError::GpuInit(e));
        },
    };

    // Banner — single compact line.
    println!(
        "{} {}",
        paint_stdout("tile bench", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(format!("· {}", runner.device_name), Style::new().fg(Color::BrightBlack)),
    );

    // Run all ops, optionally narrowed to a single substring filter.
    let mut all: Vec<OpResult> = Vec::new();
    let mut matched_filter = false;

    // When -v, compute occupancy/register profile for each op+dtype (CPU-only, fast).
    let profile_map: Option<HashMap<(String, String), ProfileRow>> =
        if verbose > 0 { Some(compute_profiles(filter.as_deref())) } else { None };

    let mut printer = SuitePrinter::new(true);
    printer.set_verbose(verbose);
    if let Some(m) = &profile_map {
        printer.set_profile_map(m.clone());
    }
    {
        let mut report = |result: &OpResult| {
            if matches_filter(filter.as_deref(), result.op()) {
                printer.print_batch(std::slice::from_ref(result));
            }
        };
        let _reporter = set_result_reporter(&mut report);

        // All ops — inventory-registered via #[bench_kernel]
        {
            let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
            specs.sort_unstable_by_key(|s| (s.op, s.subop));
            for spec in specs {
                if matches_filter(filter.as_deref(), spec.op) {
                    matched_filter = true;
                    for &dt in spec.dtypes {
                        let _kspan =
                            tracing::debug_span!("kernel", op = spec.op, dtype = %dt).entered();
                        tracing::debug!(op = spec.op, dtype = %dt, "running benchmark");
                        runner.flush_slc();
                        all.extend(run_spec(spec, &runner, dt));
                    }
                }
            }
        }
    }

    if all.is_empty() {
        if let Some(pattern) = &filter {
            if matched_filter {
                eprintln!(
                    "{} {}",
                    paint_stderr("[error]", Style::new().fg(Color::Red).bold()),
                    paint_stderr(
                        format!(
                            "Kernel matched --filter {pattern:?} but all shapes failed to compile or run"
                        ),
                        Style::new().fg(Color::BrightWhite),
                    ),
                );
            } else {
                eprintln!(
                    "{} {}",
                    paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                    paint_stderr(
                        format!("No benchmarks matched --filter {pattern:?}"),
                        Style::new().fg(Color::BrightWhite),
                    ),
                );
            }
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr("No benchmarks ran", Style::new().fg(Color::BrightWhite)),
            );
        }
        return Ok(());
    }

    validate_results(&all).unwrap_or_else(|err| panic!("{err}"));
    printer.finish();

    // Counters.
    let impl_count = all.iter().filter(|r| r.mt_perf().is_some()).count();
    let nyi_count = all.iter().filter(|r| r.mt_perf().is_none()).count();
    let checked_count = all.iter().filter(|r| r.equiv().is_some()).count();
    let equiv_pass = all
        .iter()
        .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }))
        .count();
    let equiv_fail = all
        .iter()
        .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Failed { .. }))
        .count();
    let unchecked: Vec<String> = all
        .iter()
        .filter(|r| r.is_unchecked())
        .map(|r| format!("{} [{}]", r.op(), r.shape()))
        .collect();
    let avg_pct: Option<f64> = {
        let valid: Vec<f64> = all.iter().filter_map(|r| r.pct()).collect();
        if valid.is_empty() { None } else { Some(valid.iter().sum::<f64>() / valid.len() as f64) }
    };

    // Summary — compact single line (or two if unchecked).
    let mut parts: Vec<String> = Vec::new();
    let sep = format!("  {}  ", paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()));

    parts.push(format!(
        "{} impl",
        paint_stdout(impl_count.to_string(), Style::new().fg(Color::Green).bold()),
    ));
    if nyi_count > 0 {
        parts.push(format!(
            "{} NYI",
            paint_stdout(nyi_count.to_string(), Style::new().fg(Color::Yellow).bold()),
        ));
    }
    if let Some(p) = avg_pct {
        parts.push(format!("avg {}", paint_stdout(format!("{p:.0}% MT"), pct_style(p)),));
    }
    if checked_count > 0 {
        let corr_style = if equiv_fail == 0 {
            Style::new().fg(Color::Green).bold()
        } else {
            Style::new().fg(Color::Yellow).bold()
        };
        parts.push(format!(
            "{} correct",
            paint_stdout(format!("{equiv_pass}/{checked_count}"), corr_style),
        ));
    }
    if !unchecked.is_empty() {
        parts.push(format!(
            "{} unchecked",
            paint_stdout(unchecked.len().to_string(), Style::new().fg(Color::Yellow).bold()),
        ));
    }

    println!("\n  {}", parts.join(&sep));

    if equiv_fail > 0 {
        println!(
            "  {} {}",
            paint_stdout("Failures:", Style::new().fg(Color::Red).bold()),
            paint_stdout(equiv_fail.to_string(), Style::new().fg(Color::Red).bold()),
        );
    }
    if !unchecked.is_empty() {
        println!(
            "  {} {}",
            paint_stdout("Unchecked:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(unchecked.join(", "), Style::new().fg(Color::Yellow).bold()),
        );
    }
    println!();

    if !args.no_diff {
        try_auto_diff(&runner.device_name, &all, args.baseline_ref.as_deref());
    }

    if let Some(path) = json_out {
        save_json(&runner.device_name, &all, path);
    }

    if equiv_fail > 0 {
        return Err(crate::CliError::Other(format!("{equiv_fail} correctness check(s) failed")));
    }
    Ok(())
}

/// Resolve a baseline file from the target branch and diff the
/// just-finished bench against it. Best-effort: any failure (no git
/// repo, no resolved ref, no baseline file at that ref, etc.) logs a
/// one-line skip note and returns. Never aborts the bench.
fn try_auto_diff(device: &str, results: &[OpResult], baseline_ref_override: Option<&str>) {
    let slug = chip_slug(device);
    let baseline_path = format!("baselines/{slug}.json");

    let candidates: Vec<&str> = match baseline_ref_override {
        Some(r) => vec![r],
        None => vec!["origin/dev", "upstream/dev", "dev"],
    };
    let Some(reference) = git::resolve_baseline_ref(&candidates) else {
        log_skip(&format!(
            "baseline auto-diff: no target-branch ref ({}) — skipping",
            candidates.join("/")
        ));
        return;
    };
    let Some(sha) = git::merge_base_with(&reference) else {
        log_skip(&format!("baseline auto-diff: merge-base HEAD..{reference} failed — skipping"));
        return;
    };
    let Some(content) = git::show_file_at(&sha, &baseline_path) else {
        log_skip(&format!(
            "baseline auto-diff: no {baseline_path} at {reference} ({}…) — skipping",
            sha.chars().take(7).collect::<String>()
        ));
        return;
    };

    let baseline_json: Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            log_skip(&format!(
                "baseline auto-diff: {baseline_path} at {reference} is not valid JSON ({e}) — skipping"
            ));
            return;
        },
    };
    let Some(baseline_rows) = baseline_json.get("results").and_then(|v| v.as_array()).cloned()
    else {
        log_skip(&format!(
            "baseline auto-diff: {baseline_path} at {reference} has no 'results' array — skipping"
        ));
        return;
    };

    let current_rows: Vec<Value> = results.iter().map(result_to_value).collect();

    let short_sha: String = sha.chars().take(7).collect();
    let heading = format!("tile bench · diff vs {reference} @ {short_sha} ({baseline_path})");
    let opts = diff_cmd::RenderOpts {
        heading: Some(&heading),
        sort: "regression",
        ..diff_cmd::RenderOpts::default()
    };
    let outcome = diff_cmd::render(&baseline_rows, &current_rows, &opts);
    if outcome.total_rows == 0 {
        log_skip(&format!(
            "baseline auto-diff: no overlapping rows with {baseline_path} at {reference}"
        ));
    }
}

/// Lowercase + collapse whitespace runs into a single dash, dropping
/// any character that isn't alphanumeric or `-`. Yields slugs like
/// `apple-m5-max` from `Apple M5 Max`, matching the naming convention
/// established by `baselines/apple-m5-max.json`.
fn chip_slug(device: &str) -> String {
    let mut out = String::with_capacity(device.len());
    let mut prev_dash = false;
    for ch in device.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            out.push(lowered);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn result_to_value(r: &OpResult) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("op".into(), Value::from(r.op()));
    if let Some(sub) = r.subop() {
        obj.insert("subop".into(), Value::from(sub));
    }
    obj.insert("shape".into(), Value::from(r.shape()));
    obj.insert("metric".into(), Value::from(r.metric()));
    obj.insert("ref".into(), r.ref_perf().map(Value::from).unwrap_or(Value::Null));
    obj.insert("mt".into(), r.mt_perf().map(Value::from).unwrap_or(Value::Null));
    Value::Object(obj)
}

fn log_skip(msg: &str) {
    eprintln!("  {}", paint_stderr(msg, Style::new().fg(Color::BrightBlack)));
}

fn save_json(device: &str, results: &[OpResult], path: &str) {
    use std::io::Write;
    let s = summarize(results);
    let mut out = String::new();
    out.push_str(&format!(
        "{{\"device\":{:?},\"summary\":{{\"total\":{},\"implemented\":{},\"correct\":{},\"unchecked\":{}}},\"results\":[\n",
        device, s.total, s.implemented, s.correct, s.unchecked,
    ));
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        out.push_str(&format!(
            "  {}{}\n",
            format_result_row(r.op(), r.subop(), r.shape(), r.metric(), r.ref_perf(), r.mt_perf()),
            comma
        ));
    }
    out.push_str("]}");
    match std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(".".as_ref()))
        .and_then(|_| std::fs::File::create(path))
        .and_then(|mut f| f.write_all(out.as_bytes()))
    {
        Ok(()) => println!(
            "  {} {}",
            paint_stdout("Saved →", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(path, Style::new().fg(Color::BrightWhite)),
        ),
        Err(e) => eprintln!(
            "  {} {}",
            paint_stderr("save failed:", Style::new().fg(Color::Red).bold()),
            paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
        ),
    }
}

/// Aggregate counts mirroring the terminal banner. Persisted alongside
/// the per-row results in the JSON so CI and dashboards can consume
/// kernel-correctness as a single signal without re-parsing every row.
struct Summary {
    total: usize,
    implemented: usize,
    correct: usize,
    unchecked: usize,
}

fn summarize(results: &[OpResult]) -> Summary {
    Summary {
        total: results.len(),
        implemented: results.iter().filter(|r| r.mt_perf().is_some()).count(),
        correct: results
            .iter()
            .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }))
            .count(),
        unchecked: results.iter().filter(|r| r.is_unchecked()).count(),
    }
}

/// Format one bench result as a single-line JSON object. The `subop` field is
/// emitted only when present, keeping the schema additive — existing consumers
/// that only read `op`/`shape`/`metric`/`ref`/`mt` are unaffected.
fn format_result_row(
    op: &str,
    subop: Option<&str>,
    shape: &str,
    metric: &str,
    ref_perf: Option<f64>,
    mt_perf: Option<f64>,
) -> String {
    match subop {
        Some(s) => format!(
            "{{\"op\":{:?},\"subop\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
            op,
            s,
            shape,
            metric,
            json_f(ref_perf),
            json_f(mt_perf),
        ),
        None => format!(
            "{{\"op\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
            op,
            shape,
            metric,
            json_f(ref_perf),
            json_f(mt_perf),
        ),
    }
}

fn json_f(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "null".into())
}

fn pct_style(pct: f64) -> Style {
    if pct >= 90.0 {
        Style::new().fg(Color::Green).bold()
    } else if pct >= 60.0 {
        Style::new().fg(Color::Yellow).bold()
    } else {
        Style::new().fg(Color::Red).bold()
    }
}

// ── Profile helper for -v / -vv ───────────────────────────────────────

/// Compile-time profile for each op (first dtypes entry, usually f32).
/// Runs the standard optimization pipeline + liveness + occupancy estimate.
fn compute_profiles(filter: Option<&str>) -> HashMap<(String, String), ProfileRow> {
    // Key: (op_display, dtype_label), e.g. ("unary (acos)", "f32")
    let mut map = HashMap::new();
    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.op, s.subop));
    for spec in specs {
        if !matches_filter(filter, spec.op) {
            continue;
        }
        let op_display = if spec.subop.is_empty() {
            spec.op.to_string()
        } else {
            format!("{} ({})", spec.op, spec.subop)
        };
        for &dt in spec.dtypes {
            let mut k = (spec.kernel_ir)(dt);
            k.mode = spec.dispatch.default_mode(spec.shapes);
            if passes::run_passes(&mut k, &passes::standard_pipeline()).is_err() {
                continue;
            }
            let reg_est = passes::register_estimate::estimate_registers(&k);
            let candidates: Vec<(u32, Option<u32>)> =
                [64u32, 128, 256, 512, 1024].iter().map(|&s| (s, None)).collect();
            let (occ_pct, bottleneck) =
                if let Some((_tg, est)) = occupancy::best_threadgroup_size(&k, &candidates) {
                    (est.occupancy_pct, est.bottleneck)
                } else {
                    continue;
                };
            let bottleneck_label = match bottleneck {
                Bottleneck::ThreadLimited => "thread-limited",
                Bottleneck::RegisterLimited => "register-limited",
                Bottleneck::MemoryLimited => "tgmem-limited",
                _ => "unknown",
            };
            let dtype_label = metaltile_std::bench_types::dtype_label(dt).to_string();
            map.insert((op_display.clone(), dtype_label), ProfileRow {
                occ_pct,
                regs_per_thread: reg_est.regs_per_thread,
                bottleneck: bottleneck_label,
            });
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use metaltile_std::bench_types::{EquivResult, OpBench};

    use super::*;

    fn pass_equiv() -> EquivResult {
        EquivResult { n_checked: 1, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }
    }

    fn fail_equiv() -> EquivResult {
        EquivResult { n_checked: 1, max_abs_err: 1e3, cosine_sim: 0.5, passed: false }
    }

    #[test]
    fn summary_counts_per_category() {
        let b = OpBench::new("op_a", "GB/s");
        let implemented_correct = b.result("shape_a", Some(100.0), Some(95.0), Some(pass_equiv()));
        let implemented_wrong = b.result("shape_b", Some(100.0), Some(40.0), Some(fail_equiv()));
        let nyi = b.result("shape_c", Some(100.0), None, None);

        let s = summarize(&[implemented_correct, implemented_wrong, nyi]);
        assert_eq!(s.total, 3);
        assert_eq!(s.implemented, 2); // _correct + _wrong
        assert_eq!(s.correct, 1); // only _correct
        assert_eq!(s.unchecked, 0); // both implemented rows had equiv
    }

    #[test]
    fn summary_counts_unchecked_rows() {
        // An implemented result without an equiv check is an "unchecked"
        // row — pinned via panic in OpBench::result_sub, but the
        // counting path matters for older snapshots loaded via `tile
        // snap --from`.
        let b = OpBench::new("op", "GB/s");
        let r = b.result("shape", Some(100.0), Some(95.0), Some(pass_equiv()));
        // is_unchecked() is false here — sanity check the summary
        // doesn't double-count.
        let s = summarize(&[r]);
        assert_eq!(s.implemented, 1);
        assert_eq!(s.correct, 1);
        assert_eq!(s.unchecked, 0);
    }

    #[test]
    fn summary_on_empty_input_is_all_zero() {
        let s = summarize(&[]);
        assert_eq!(s.total, 0);
        assert_eq!(s.implemented, 0);
        assert_eq!(s.correct, 0);
        assert_eq!(s.unchecked, 0);
    }

    #[test]
    fn json_f_formats_finite_and_none() {
        assert_eq!(json_f(Some(12.345)), "12.345");
        assert_eq!(json_f(Some(0.0)), "0.000");
        assert_eq!(json_f(None), "null");
    }

    #[test]
    fn row_without_subop_matches_legacy_schema() {
        // Pre-existing consumers rely on this exact key set + ordering.
        let row = format_result_row(
            "rms_norm",
            None,
            "B=1024 N=4096 f32",
            "GB/s",
            Some(323.9),
            Some(325.6),
        );
        assert_eq!(
            row,
            r#"{"op":"rms_norm","shape":"B=1024 N=4096 f32","metric":"GB/s","ref":323.900,"mt":325.600}"#,
        );
        assert!(!row.contains("\"subop\""));
    }

    #[test]
    fn row_with_subop_emits_disambiguated_field() {
        // The motivating bug: many `unary` subops collapse to identical
        // (op, shape) tuples in the legacy schema. The `subop` field
        // disambiguates them without breaking schema additively.
        let row =
            format_result_row("unary", Some("sin"), "N=64M f32", "GB/s", Some(544.8), Some(114.5));
        assert_eq!(
            row,
            r#"{"op":"unary","subop":"sin","shape":"N=64M f32","metric":"GB/s","ref":544.800,"mt":114.500}"#,
        );
    }

    #[test]
    fn row_handles_missing_perf_values() {
        let row = format_result_row("sdpa", Some("sdpa_vector"), "H=8 N=2048", "GB/s", None, None);
        assert!(row.contains(r#""ref":null"#));
        assert!(row.contains(r#""mt":null"#));
        assert!(row.contains(r#""subop":"sdpa_vector""#));
    }

    #[test]
    fn row_quotes_strings_containing_special_chars() {
        // Shape strings sometimes embed `=`, spaces, and parens; ensure they
        // round-trip via Debug-quoting so the row is valid JSON.
        let row = format_result_row("foo", None, "k=2 (warm)", "GB/s", Some(1.0), Some(2.0));
        let parsed: serde_json::Value = serde_json::from_str(&row).unwrap();
        assert_eq!(parsed["shape"], "k=2 (warm)");
    }

    // The slug must match the filenames committed under `baselines/`
    // (e.g. `apple-m5-max.json`) so auto-diff can find them.
    #[test]
    fn chip_slug_matches_apple_m5_max_filename() {
        assert_eq!(chip_slug("Apple M5 Max"), "apple-m5-max");
    }

    #[test]
    fn chip_slug_collapses_runs_of_punctuation() {
        // Hypothetical messy device string — make sure we don't emit
        // double dashes or trailing dashes.
        assert_eq!(chip_slug("  Apple  --M1 (Pro)  "), "apple-m1-pro");
        assert_eq!(chip_slug("Apple_M2_Max!"), "apple-m2-max");
    }

    #[test]
    fn result_to_value_emits_legacy_schema() {
        let b = OpBench::new("rms_norm", "GB/s");
        let r = b.result("B=1024 N=4096 f32", Some(323.9), Some(325.6), Some(pass_equiv()));
        let v = result_to_value(&r);
        assert_eq!(v["op"], "rms_norm");
        assert_eq!(v["shape"], "B=1024 N=4096 f32");
        assert_eq!(v["metric"], "GB/s");
        assert_eq!(v["ref"], 323.9);
        assert_eq!(v["mt"], 325.6);
        // No subop here, so the field should be absent (matches save_json).
        assert!(v.get("subop").is_none());
    }

    #[test]
    fn result_to_value_null_perfs_round_trip_through_diff() {
        // NYI result — both perfs are None. The diff `build_result_map`
        // pulls f64::unwrap_or(0.0), so the row goes in as zeros which
        // is the established contract.
        let b = OpBench::new("sdpa", "GB/s");
        let r = b.nyi("H=8 N=2048", None);
        let v = result_to_value(&r);
        assert!(v["ref"].is_null());
        assert!(v["mt"].is_null());
    }
}
