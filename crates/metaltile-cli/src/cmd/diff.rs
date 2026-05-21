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
    CliError,
    DiffArgs,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &DiffArgs) -> Result<(), CliError> {
    let _span = tracing::info_span!(
        "diff",
        baseline = %args.baseline,
        threshold = args.threshold,
    )
    .entered();
    let baseline_path = &args.baseline;
    let current_path = &args.current;

    let baseline = load_results(baseline_path, "baseline")?;

    let current = if let Some(path) = current_path {
        load_results(path, "current")?
    } else {
        eprintln!(
            "  {}",
            paint_stdout("Running bench suite for current...", Style::new().fg(Color::Cyan).bold()),
        );
        let temp_file =
            std::env::temp_dir().join(format!(".tile-diff-tmp-{}.json", std::process::id()));
        let mut child = Command::new(std::env::current_exe().map_err(CliError::Io)?)
            .arg("bench")
            .arg("--json")
            .arg(temp_file.to_str().ok_or_else(|| CliError::Other("non-UTF8 temp path".into()))?)
            .spawn()
            .map_err(|e| CliError::Subprocess(format!("failed to spawn tile bench: {e}")))?;

        let status = child
            .wait()
            .map_err(|e| CliError::Subprocess(format!("tile bench did not start: {e}")))?;
        if !status.success() {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr("bench suite failed", Style::new().fg(Color::BrightWhite)),
            );
            let _ = std::fs::remove_file(&temp_file);
            return Err(CliError::Other("bench suite failed".into()));
        }

        let content = std::fs::read_to_string(&temp_file).map_err(|e| {
            eprintln!("cannot read temp results: {e}");
            CliError::Io(e)
        })?;
        let _ = std::fs::remove_file(&temp_file);

        let json: Value = serde_json::from_str(&content).map_err(|e| {
            eprintln!("invalid bench JSON: {e}");
            CliError::Json(e)
        })?;
        json.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default()
    };

    let opts = RenderOpts {
        filter: args.filter.as_deref(),
        threshold: args.threshold,
        sort: &args.sort,
        only_regressions: args.only_regressions,
        only_improvements: args.only_improvements,
        heading: Some("tile diff"),
    };
    let outcome = render(&baseline, &current, &opts);

    if outcome.total_rows == 0 {
        eprintln!(
            "  {}",
            paint_stdout("No matching results to diff.", Style::new().fg(Color::BrightBlack)),
        );
        return Ok(());
    }

    if outcome.regressions > 0 {
        return Err(CliError::Other(format!("{} regression(s) detected", outcome.regressions)));
    }
    Ok(())
}

// ── Public render API ───────────────────────────────────────────────────

/// Options for [`render`]. Mirrors the relevant subset of [`DiffArgs`]
/// without the file-loading concerns.
pub struct RenderOpts<'a> {
    pub filter: Option<&'a str>,
    pub threshold: f64,
    pub sort: &'a str,
    pub only_regressions: bool,
    pub only_improvements: bool,
    /// Heading printed above the diff table. Suppress with `None`.
    pub heading: Option<&'a str>,
}

impl Default for RenderOpts<'_> {
    fn default() -> Self {
        Self {
            filter: None,
            threshold: 5.0,
            sort: "name",
            only_regressions: false,
            only_improvements: false,
            heading: Some("tile diff"),
        }
    }
}

/// Summary counts returned from [`render`]. `total_rows` is the number
/// of (op, shape) keys printed after filtering — callers can use it to
/// decide whether to print a "no matching results" message vs. trust
/// the table to speak for itself. `regressions` drives the `tile diff`
/// exit code.
pub struct RenderOutcome {
    pub regressions: usize,
    pub total_rows: usize,
}

/// Compute, sort, and print the diff between two result sets. Pure
/// w.r.t. the filesystem — callers (both `tile diff` and `tile bench`)
/// load JSON however they like and hand the parsed arrays in here.
pub fn render(baseline: &[Value], current: &[Value], opts: &RenderOpts) -> RenderOutcome {
    let baseline_map = build_result_map(baseline);
    let current_map = build_result_map(current);

    let mut all_keys: Vec<&RowKey> = baseline_map.keys().collect();
    for k in current_map.keys() {
        if !baseline_map.contains_key(k) {
            all_keys.push(k);
        }
    }

    let mut diff_rows: Vec<DiffRow> = Vec::new();
    for key in &all_keys {
        if !matches_filter(opts.filter, &key.op) {
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
                    Some(d) if d < -opts.threshold => DeltaKind::Regression,
                    Some(d) if d > opts.threshold => DeltaKind::Improvement,
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

        if opts.only_regressions && kind != DeltaKind::Regression {
            continue;
        }
        if opts.only_improvements && kind != DeltaKind::Improvement {
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

    sort_diff_rows(&mut diff_rows, opts.sort);

    let regressions = diff_rows.iter().filter(|r| r.kind == DeltaKind::Regression).count();
    let improvements = diff_rows.iter().filter(|r| r.kind == DeltaKind::Improvement).count();
    let unchanged = diff_rows.iter().filter(|r| r.kind == DeltaKind::Unchanged).count();
    let new_rows = diff_rows.iter().filter(|r| r.kind == DeltaKind::New).count();
    let removed = diff_rows.iter().filter(|r| r.kind == DeltaKind::Removed).count();
    let total_rows = diff_rows.len();

    if total_rows == 0 {
        return RenderOutcome { regressions, total_rows };
    }

    if let Some(heading) = opts.heading {
        eprintln!("{}", paint_stdout(heading, Style::new().fg(Color::Cyan).bold()));
    }

    eprintln!();
    for row in &diff_rows {
        print_diff_row(row);
    }

    print_summary(regressions, improvements, unchanged, new_rows, removed, opts.threshold);

    RenderOutcome { regressions, total_rows }
}

// ── Data types ───────────────────────────────────────────────────────────

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

// ── Helpers ──────────────────────────────────────────────────────────────

fn sort_diff_rows(diff_rows: &mut [DiffRow], sort: &str) {
    match sort {
        "delta" => diff_rows.sort_by(|a, b| {
            b.delta_pct
                .unwrap_or(0.0)
                .partial_cmp(&a.delta_pct.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        "regression" => diff_rows.sort_by(|a, b| {
            let rank = |k: DeltaKind| match k {
                DeltaKind::Regression => 0,
                DeltaKind::Removed => 1,
                DeltaKind::New => 2,
                DeltaKind::Improvement => 3,
                DeltaKind::Unchanged => 4,
            };
            let cmp = rank(a.kind).cmp(&rank(b.kind));
            if cmp == std::cmp::Ordering::Equal {
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
}

fn print_diff_row(row: &DiffRow) {
    let (op_col, shape_col) = format_op_shape(&row.op, &row.shape);

    let baseline_str = row.baseline_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());
    let current_str = row.current_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());
    let delta_str = match row.kind {
        DeltaKind::New => "new".to_string(),
        DeltaKind::Removed => "removed".to_string(),
        _ => {
            let arrow = match row.kind {
                DeltaKind::Regression => paint_stderr("▼", Style::new().fg(Color::Red).bold()),
                DeltaKind::Improvement => paint_stdout("▲", Style::new().fg(Color::Green).bold()),
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

fn print_summary(
    regressions: usize,
    improvements: usize,
    unchanged: usize,
    new_count: usize,
    removed_count: usize,
    threshold: f64,
) {
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
    eprintln!("\n  {}\n", parts.join(&format!("  {sep}  ")));
}

fn load_results(path: &str, label: &str) -> Result<Vec<Value>, CliError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("cannot read {label} {path}: {e}"),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        CliError::Io(e)
    })?;
    let json: Value = serde_json::from_str(&content).map_err(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(format!("invalid {label} JSON: {e}"), Style::new().fg(Color::BrightWhite)),
        );
        CliError::Json(e)
    })?;
    // Support both bench_dump format (results array at top level) and snapshot format
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        Ok(results.clone())
    } else if let Some(results) = json.as_array() {
        Ok(results.clone())
    } else {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("{label} has no 'results' array"),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        Err(CliError::Other(format!("{label} has no 'results' array")))
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn rows(pairs: &[(&str, &str, f64, f64)]) -> Vec<Value> {
        pairs
            .iter()
            .map(|(op, shape, r, m)| {
                json!({ "op": op, "shape": shape, "metric": "GB/s", "ref": r, "mt": m })
            })
            .collect()
    }

    #[test]
    fn render_classifies_regression_and_improvement() {
        let baseline =
            rows(&[("softmax", "B=1 N=8", 100.0, 100.0), ("rms_norm", "B=1 N=8", 100.0, 50.0)]);
        let current =
            rows(&[("softmax", "B=1 N=8", 100.0, 60.0), ("rms_norm", "B=1 N=8", 100.0, 90.0)]);
        let opts = RenderOpts { heading: None, threshold: 5.0, ..RenderOpts::default() };
        let outcome = render(&baseline, &current, &opts);
        assert_eq!(outcome.regressions, 1);
        assert_eq!(outcome.total_rows, 2);
    }

    #[test]
    fn render_marks_new_and_removed_keys() {
        let baseline = rows(&[("softmax", "B=1 N=8", 100.0, 100.0)]);
        let current = rows(&[("rope", "B=1 N=8", 100.0, 80.0)]);
        let opts = RenderOpts { heading: None, ..RenderOpts::default() };
        let outcome = render(&baseline, &current, &opts);
        // One new + one removed, neither counts as a regression.
        assert_eq!(outcome.regressions, 0);
        assert_eq!(outcome.total_rows, 2);
    }

    #[test]
    fn render_only_regressions_filter_drops_improvements() {
        let baseline =
            rows(&[("softmax", "B=1 N=8", 100.0, 100.0), ("rms_norm", "B=1 N=8", 100.0, 50.0)]);
        let current =
            rows(&[("softmax", "B=1 N=8", 100.0, 50.0), ("rms_norm", "B=1 N=8", 100.0, 90.0)]);
        let opts = RenderOpts { heading: None, only_regressions: true, ..RenderOpts::default() };
        let outcome = render(&baseline, &current, &opts);
        assert_eq!(outcome.regressions, 1);
        assert_eq!(outcome.total_rows, 1);
    }

    #[test]
    fn render_filter_substring_match_is_case_insensitive() {
        let baseline =
            rows(&[("softmax", "B=1 N=8", 100.0, 100.0), ("rope", "B=1 N=8", 100.0, 100.0)]);
        let current =
            rows(&[("softmax", "B=1 N=8", 100.0, 50.0), ("rope", "B=1 N=8", 100.0, 50.0)]);
        let opts = RenderOpts { heading: None, filter: Some("ROPE"), ..RenderOpts::default() };
        let outcome = render(&baseline, &current, &opts);
        assert_eq!(outcome.total_rows, 1);
        assert_eq!(outcome.regressions, 1);
    }

    #[test]
    fn render_zero_baseline_perf_does_not_panic() {
        let baseline = rows(&[("op", "shape", 0.0, 0.0)]);
        let current = rows(&[("op", "shape", 100.0, 50.0)]);
        let opts = RenderOpts { heading: None, ..RenderOpts::default() };
        let outcome = render(&baseline, &current, &opts);
        assert_eq!(outcome.regressions, 0);
        assert_eq!(outcome.total_rows, 1);
    }
}
