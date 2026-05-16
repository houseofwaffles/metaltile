//! `tile diff` — Compare bench results against a saved baseline.
//!
//! Usage:
//!   tile diff .tile-snapshots/m4max.json                          # run bench then diff
//!   tile diff .tile-snapshots/m4max.json run.json                 # diff two files
//!   tile diff .tile-snapshots/m4max.json -f softmax
//!   tile diff .tile-snapshots/m4max.json --threshold 3

use std::{collections::HashMap, process::Command};

use serde_json::Value;

use crate::{
    flag_val,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(Debug, PartialEq, Eq, Hash)]
struct RowKey {
    op: String,
    shape: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DeltaKind {
    Regression,
    Improvement,
    Unchanged,
    New,
    Removed,
}

#[derive(Debug)]
struct DiffRow {
    op: String,
    shape: String,
    baseline_pct: Option<f64>,
    current_pct: Option<f64>,
    delta_pct: Option<f64>,
    kind: DeltaKind,
}

/// Collect positional arguments, skipping flag names and their values.
fn positionals(args: &[String]) -> Vec<String> {
    const VALUE_FLAGS: &[&str] = &["--filter", "-f", "--threshold", "--sort", "--json", "-o"];
    let mut result = Vec::new();
    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg.starts_with('-') {
            if VALUE_FLAGS.contains(&arg.as_str()) {
                skip_next = true;
            }
        } else {
            result.push(arg.clone());
        }
    }
    result
}

pub fn run(args: &[String]) {
    let pos = positionals(args);
    let baseline_path = pos.first().cloned();
    let current_path = pos.get(1).cloned();
    let filter = flag_val(args, "--filter").or_else(|| flag_val(args, "-f"));
    let threshold_str = flag_val(args, "--threshold");
    let threshold: f64 = threshold_str.as_deref().and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let sort = flag_val(args, "--sort").unwrap_or_else(|| "name".to_string());
    let only_regressions = args.iter().any(|a| a == "--only-regressions");
    let only_improvements = args.iter().any(|a| a == "--only-improvements");

    let baseline_path = match baseline_path {
        Some(p) => p,
        None => {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    "usage: tile diff <baseline> [current]",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            std::process::exit(1);
        },
    };

    // Load baseline
    let baseline = load_results(&baseline_path, "baseline");

    // Load or generate current
    let current = if let Some(ref path) = current_path {
        load_results(path, "current")
    } else {
        eprintln!(
            "  {}",
            paint_stdout("Running bench suite for current...", Style::new().fg(Color::Cyan).bold()),
        );
        let temp_file =
            std::env::temp_dir().join(format!(".tile-diff-tmp-{}.json", std::process::id()));
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("bench")
            .arg("--json")
            .arg(temp_file.to_str().unwrap())
            .spawn()
            .expect("failed to run tile bench");

        let status = child.wait().expect("tile bench did not start");
        if !status.success() {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr("bench suite failed", Style::new().fg(Color::BrightWhite)),
            );
            let _ = std::fs::remove_file(&temp_file);
            std::process::exit(1);
        }

        let content = std::fs::read_to_string(&temp_file).unwrap_or_else(|e| {
            eprintln!("cannot read temp results: {e}");
            std::process::exit(1);
        });
        let _ = std::fs::remove_file(&temp_file);

        // Map of shape -> mt_perf for pct calc
        let json: Value = serde_json::from_str(&content).unwrap_or_else(|e| {
            eprintln!("invalid bench JSON: {e}");
            std::process::exit(1);
        });
        json.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default()
    };

    // Build lookup maps: key -> (ref_perf, mt_perf)
    let baseline_map = build_result_map(&baseline);
    let current_map = build_result_map(&current);

    // Collect all keys (sorted by op then shape).
    let mut all_keys: Vec<&RowKey> = baseline_map.keys().collect();
    for k in current_map.keys() {
        if !baseline_map.contains_key(k) {
            all_keys.push(k);
        }
    }
    // Sort: collect into Vec of owned entries so we can sort by op+shape
    let mut diff_rows: Vec<DiffRow> = Vec::new();

    for key in &all_keys {
        if !matches_filter(filter.as_deref(), &key.op) {
            continue;
        }
        let b = baseline_map.get(*key);
        let c = current_map.get(*key);

        let (kind, delta_pct, baseline_pct, current_pct) = match (b, c) {
            (Some(&(br, bm)), Some(&(cr, cm))) => {
                let bpct = if br > 0.0 { Some(bm / br * 100.0) } else { None };
                let cpct = if cr > 0.0 { Some(cm / cr * 100.0) } else { None };
                let delta = match (bpct, cpct) {
                    (Some(bp), Some(cp)) => Some(cp - bp),
                    _ => None,
                };
                let kind = match delta {
                    Some(d) if d < -threshold => DeltaKind::Regression,
                    Some(d) if d > threshold => DeltaKind::Improvement,
                    _ => DeltaKind::Unchanged,
                };
                (kind, delta, bpct, cpct)
            },
            (Some(&(br, bm)), None) => {
                let bpct = if br > 0.0 { Some(bm / br * 100.0) } else { None };
                (DeltaKind::Removed, None, bpct, None)
            },
            (None, Some(&(cr, cm))) => {
                let cpct = if cr > 0.0 { Some(cm / cr * 100.0) } else { None };
                (DeltaKind::New, None, None, cpct)
            },
            (None, None) => continue,
        };

        if only_regressions && kind != DeltaKind::Regression {
            continue;
        }
        if only_improvements && kind != DeltaKind::Improvement {
            continue;
        }

        diff_rows.push(DiffRow {
            op: key.op.clone(),
            shape: key.shape.clone(),
            baseline_pct,
            current_pct,
            delta_pct,
            kind,
        });
    }

    // Sort
    match sort.as_str() {
        "delta" => diff_rows.sort_by(|a, b| {
            b.delta_pct
                .unwrap_or(0.0)
                .partial_cmp(&a.delta_pct.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        "regression" => diff_rows.sort_by(|a, b| {
            // Regressions first, then improvements, then unchanged
            let rank = |k: DeltaKind| match k {
                DeltaKind::Regression => 0,
                DeltaKind::Removed => 1,
                DeltaKind::New => 2,
                DeltaKind::Improvement => 3,
                DeltaKind::Unchanged => 4,
            };
            let cmp = rank(a.kind).cmp(&rank(b.kind));
            if cmp == std::cmp::Ordering::Equal {
                // Within same kind, sort by delta magnitude
                b.delta_pct
                    .map(|d| d.abs())
                    .unwrap_or(0.0)
                    .partial_cmp(&a.delta_pct.map(|d| d.abs()).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                cmp
            }
        }),
        _ => diff_rows.sort_by(|a, b| {
            let cmp = a.op.cmp(&b.op);
            if cmp == std::cmp::Ordering::Equal { a.shape.cmp(&b.shape) } else { cmp }
        }),
    }

    if diff_rows.is_empty() {
        println!(
            "  {}",
            paint_stdout("No matching results to diff.", Style::new().fg(Color::BrightBlack)),
        );
        return;
    }

    // Print diff table
    eprintln!();
    for row in &diff_rows {
        let (op_col, shape_col) = format_op_shape(&row.op, &row.shape);

        let baseline_str =
            row.baseline_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());
        let current_str = row.current_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());
        let delta_str = match row.kind {
            DeltaKind::New => "new".to_string(),
            DeltaKind::Removed => "removed".to_string(),
            _ => {
                let arrow = match row.kind {
                    DeltaKind::Regression => paint_stderr("▼", Style::new().fg(Color::Red).bold()),
                    DeltaKind::Improvement =>
                        paint_stdout("▲", Style::new().fg(Color::Green).bold()),
                    _ => paint_stdout("—", Style::new().fg(Color::BrightBlack)),
                };
                let delta = row.delta_pct.unwrap_or(0.0);
                format!(
                    "{} {}",
                    arrow,
                    paint_stdout(format!("{:+.0}%", delta), match row.kind {
                        DeltaKind::Regression => Style::new().fg(Color::Red).bold(),
                        DeltaKind::Improvement => Style::new().fg(Color::Green).bold(),
                        _ => Style::new().fg(Color::BrightBlack),
                    },),
                )
            },
        };

        let (baseline_cell, current_cell) = match row.kind {
            DeltaKind::New | DeltaKind::Removed => (
                paint_stdout(&baseline_str, Style::new().fg(Color::BrightBlack)),
                paint_stdout(&current_str, Style::new().fg(Color::BrightBlack)),
            ),
            _ => (
                paint_stdout(&baseline_str, Style::new().fg(Color::BrightWhite)),
                paint_stdout(&current_str, Style::new().fg(Color::BrightWhite)),
            ),
        };

        let sep = paint_stdout("│", Style::new().fg(Color::BrightBlack).dim());

        let kind_label = match row.kind {
            DeltaKind::Regression => paint_stderr("REGRESSION", Style::new().fg(Color::Red).bold()),
            DeltaKind::Improvement => paint_stdout("improvement", Style::new().fg(Color::Green)),
            DeltaKind::New => paint_stdout("new", Style::new().fg(Color::Cyan)),
            DeltaKind::Removed => paint_stderr("removed", Style::new().fg(Color::Red)),
            DeltaKind::Unchanged => String::new(),
        };

        eprintln!(
            "  {} {sep} {} {sep} {} → {} {sep} {}  {}",
            op_col, shape_col, baseline_cell, current_cell, delta_str, kind_label,
        );
    }

    // Summary
    let regressions = diff_rows.iter().filter(|r| r.kind == DeltaKind::Regression).count();
    let improvements = diff_rows.iter().filter(|r| r.kind == DeltaKind::Improvement).count();
    let unchanged = diff_rows.iter().filter(|r| r.kind == DeltaKind::Unchanged).count();
    let new_count = diff_rows.iter().filter(|r| r.kind == DeltaKind::New).count();
    let removed_count = diff_rows.iter().filter(|r| r.kind == DeltaKind::Removed).count();

    let mut parts: Vec<String> = Vec::new();
    if regressions > 0 {
        parts.push(format!(
            "{} regression{} (threshold {}%)",
            paint_stderr(regressions.to_string(), Style::new().fg(Color::Red).bold()),
            if regressions == 1 { "" } else { "s" },
            threshold,
        ));
    }
    if improvements > 0 {
        parts.push(format!(
            "{} improved",
            paint_stdout(improvements.to_string(), Style::new().fg(Color::Green).bold()),
        ));
    }
    if unchanged > 0 {
        parts.push(format!(
            "{} unchanged",
            paint_stdout(unchanged.to_string(), Style::new().fg(Color::BrightBlack)),
        ));
    }
    if new_count > 0 {
        parts.push(format!(
            "{} new",
            paint_stdout(new_count.to_string(), Style::new().fg(Color::Cyan)),
        ));
    }
    if removed_count > 0 {
        parts.push(format!(
            "{} removed",
            paint_stderr(removed_count.to_string(), Style::new().fg(Color::Red)),
        ));
    }

    let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
    eprintln!("\n  {}\n", parts.join(&format!("  {sep}  ")),);

    if regressions > 0 {
        std::process::exit(1);
    }
}

fn load_results(path: &str, label: &str) -> Vec<Value> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("cannot read {label} {path}: {e}"),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        std::process::exit(1);
    });
    let json: Value = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(format!("invalid {label} JSON: {e}"), Style::new().fg(Color::BrightWhite)),
        );
        std::process::exit(1);
    });
    // Support both bench_dump format (results array at top level) and snapshot format
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        results.clone()
    } else if let Some(results) = json.as_array() {
        results.clone()
    } else {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("{label} has no 'results' array"),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        std::process::exit(1);
    }
}

/// Build a map from (op, shape) -> (ref_perf, mt_perf).
fn build_result_map(results: &[Value]) -> HashMap<RowKey, (f64, f64)> {
    let mut map = HashMap::new();
    for item in results {
        let op = item.get("op").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        let shape = item.get("shape").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        let ref_perf = item.get("ref").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mt_perf = item.get("mt").and_then(|v| v.as_f64()).unwrap_or(0.0);
        map.insert(RowKey { op, shape }, (ref_perf, mt_perf));
    }
    map
}

fn format_op_shape(op: &str, shape: &str) -> (String, String) {
    let op_col = paint_stdout(format!("{op:<20}"), Style::new().fg(Color::Cyan).bold());
    let shape_col = paint_stdout(format!("{shape:<26}"), Style::new().fg(Color::BrightWhite));
    (op_col, shape_col)
}
