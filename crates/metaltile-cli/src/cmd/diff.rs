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
            paint_stderr("Running bench suite for current...", Style::new().fg(Color::Cyan).bold()),
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
        println!(
            "  {}",
            paint_stderr("No matching results to diff.", Style::new().fg(Color::BrightBlack)),
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
        eprintln!("{}", paint_stderr(heading, Style::new().fg(Color::Cyan).bold()));
    }

    println!();
    print_diff_table(&diff_rows);

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

/// Display-ready column data for one diff row, with unstyled text for
/// width measurement and style parameters for final styling (matching
/// bench's approach: pad text then style the whole padded string with
/// `paint_stdout`).
struct DiffRowDisplay {
    op: String,       // unstyled
    shape: String,    // unstyled
    baseline: String, // unstyled, e.g. "104%" or "—"
    current: String,  // unstyled, e.g. "—" or "420%"
    delta: String,    // unstyled, e.g. "▲ +88%" or "removed"
    baseline_style: Style,
    current_style: Style,
    delta_style: Style,
}

fn build_diff_row_display(row: &DiffRow) -> DiffRowDisplay {
    let baseline = row.baseline_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());
    let current = row.current_pct.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());

    // Baseline/current style — dim for new/removed, bright white otherwise
    let (baseline_style, current_style) = match row.kind {
        DeltaKind::New | DeltaKind::Removed =>
            (Style::new().fg(Color::BrightBlack), Style::new().fg(Color::BrightBlack)),
        _ => (Style::new().fg(Color::BrightWhite), Style::new().fg(Color::BrightWhite)),
    };

    // Delta column style:
    //   green bold = improvement (▲ +88%)
    //   red bold   = regression (▼ -35%)
    //   yellow     = borderline / within threshold (▲ +5%, ▼ -3%)
    //   dim        = exactly 0% (— +0%)
    //   red        = "removed"
    //   cyan       = "new"
    let (delta, delta_style) = match row.kind {
        DeltaKind::Removed => ("removed".to_string(), Style::new().fg(Color::Red)),
        DeltaKind::New => ("new".to_string(), Style::new().fg(Color::Cyan)),
        DeltaKind::Unchanged => {
            let pct = row.delta_pct.unwrap_or(0.0);
            if pct == 0.0 {
                (format!("— {:+.0}%", pct), Style::new().fg(Color::BrightBlack))
            } else {
                let arrow = if pct > 0.0 { "▲" } else { "▼" };
                (format!("{arrow} {:+.0}%", pct), Style::new().fg(Color::Yellow))
            }
        },
        DeltaKind::Regression =>
            (format!("▼ {:+.0}%", row.delta_pct.unwrap_or(0.0)), Style::new().fg(Color::Red).bold()),
        DeltaKind::Improvement => (
            format!("▲ {:+.0}%", row.delta_pct.unwrap_or(0.0)),
            Style::new().fg(Color::Green).bold(),
        ),
    };

    DiffRowDisplay {
        op: row.op.clone(),
        shape: row.shape.clone(),
        baseline,
        current,
        delta,
        baseline_style,
        current_style,
        delta_style,
    }
}

/// Print the whole diff table with aligned columns.
fn print_diff_table(rows: &[DiffRow]) {
    let displays: Vec<DiffRowDisplay> = rows.iter().map(build_diff_row_display).collect();

    // Measure column widths from unstyled text.
    let op_w = displays.iter().map(|d| d.op.len()).max().unwrap_or(4).max(4);
    let shape_w = displays.iter().map(|d| d.shape.len()).max().unwrap_or(5).max(5);
    let bl_w = displays.iter().map(|d| d.baseline.len()).max().unwrap_or(3);
    let cur_w = displays.iter().map(|d| d.current.len()).max().unwrap_or(3);
    let delta_w = displays.iter().map(|d| d.delta.len()).max().unwrap_or(2).max(2);

    let sep = paint_stdout("│", Style::new().fg(Color::BrightBlack).dim());

    // Apply style to padded text directly (matching bench's approach).
    let op_style = Style::new().fg(Color::Cyan).bold();
    let shape_style = Style::new().fg(Color::BrightWhite);

    for d in &displays {
        let op_styled = paint_stdout(pad_right(&d.op, op_w), op_style);
        let shape_styled = paint_stdout(pad_right(&d.shape, shape_w), shape_style);
        let bl_styled = paint_stdout(pad_left(&d.baseline, bl_w), d.baseline_style);
        let cur_styled = paint_stdout(pad_left(&d.current, cur_w), d.current_style);
        let delta_styled = paint_stdout(pad_left(&d.delta, delta_w), d.delta_style);

        println!(
            "  {op_styled} {sep} {shape_styled} {sep} {bl_styled} → {cur_styled} {sep} {delta_styled}",
        );
    }
}

/// Pad `s` on the right with spaces to reach `width`.
fn pad_right(s: &str, width: usize) -> String {
    let len = s.len();
    if len >= width { s.to_string() } else { format!("{s}{: <len$}", "", len = width - len) }
}

/// Pad `s` on the left with spaces to reach `width`.
fn pad_left(s: &str, width: usize) -> String {
    let len = s.len();
    if len >= width { s.to_string() } else { format!("{: >len$}{s}", "", len = width - len) }
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
            paint_stderr(regressions.to_string(), Style::new().fg(Color::BrightRed).bold()),
            if regressions == 1 { "" } else { "s" },
            threshold,
        ));
    }
    if improvements > 0 {
        parts.push(format!(
            "{} improved",
            paint_stderr(improvements.to_string(), Style::new().fg(Color::BrightGreen).bold()),
        ));
    }
    if unchanged > 0 {
        parts.push(format!(
            "{} unchanged",
            paint_stderr(unchanged.to_string(), Style::new().fg(Color::BrightBlack)),
        ));
    }
    if new_count > 0 {
        parts.push(format!(
            "{} new",
            paint_stderr(new_count.to_string(), Style::new().fg(Color::Cyan)),
        ));
    }
    if removed_count > 0 {
        parts.push(format!(
            "{} removed",
            paint_stderr(removed_count.to_string(), Style::new().fg(Color::BrightRed)),
        ));
    }

    let sep = paint_stderr("·", Style::new().fg(Color::BrightBlack).dim());
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

    // ── pad_left / pad_right ─────────────────────────────────────────

    #[test]
    fn pad_left_adds_spaces() {
        assert_eq!(pad_left("42", 5), "   42");
    }

    #[test]
    fn pad_left_no_op_when_equal() {
        assert_eq!(pad_left("hello", 5), "hello");
    }

    #[test]
    fn pad_left_no_op_when_longer() {
        assert_eq!(pad_left("hello", 3), "hello");
    }

    #[test]
    fn pad_right_adds_spaces() {
        assert_eq!(pad_right("ab", 4), "ab  ");
    }

    #[test]
    fn pad_right_no_op_when_equal() {
        assert_eq!(pad_right("rust", 4), "rust");
    }

    // ── build_diff_row_display: delta column carries all info ─────

    #[test]
    fn removed_row_shows_label_in_delta() {
        let row = DiffRow {
            op: "affine".into(),
            shape: "bits=4 f32".into(),
            baseline_pct: Some(104.0),
            current_pct: None,
            delta_pct: None,
            kind: DeltaKind::Removed,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "removed");
    }

    #[test]
    fn new_row_shows_label_in_delta() {
        let row = DiffRow {
            op: "new_op".into(),
            shape: "N=64M f32".into(),
            baseline_pct: None,
            current_pct: Some(95.0),
            delta_pct: None,
            kind: DeltaKind::New,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "new");
    }

    #[test]
    fn unchanged_zero_delta_is_dim() {
        let row = DiffRow {
            op: "rms_norm".into(),
            shape: "B=1024 N=4096 f16".into(),
            baseline_pct: Some(101.0),
            current_pct: Some(101.0),
            delta_pct: Some(0.0),
            kind: DeltaKind::Unchanged,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "— +0%");
    }

    #[test]
    fn unchanged_borderline_uses_arrow() {
        let row = DiffRow {
            op: "rms_norm".into(),
            shape: "B=1024 N=4096 f32".into(),
            baseline_pct: Some(110.0),
            current_pct: Some(113.0),
            delta_pct: Some(3.0),
            kind: DeltaKind::Unchanged,
        };
        let d = build_diff_row_display(&row);
        // Arrow appears (not dash) for borderline non-zero changes
        assert_eq!(d.delta, "▲ +3%");
    }

    #[test]
    fn unchanged_borderline_negative_uses_down_arrow() {
        let row = DiffRow {
            op: "rms_norm".into(),
            shape: "B=1024 N=4096 bf16".into(),
            baseline_pct: Some(102.0),
            current_pct: Some(99.0),
            delta_pct: Some(-3.0),
            kind: DeltaKind::Unchanged,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "▼ -3%");
    }

    #[test]
    fn regression_delta_is_unstyled_text() {
        let row = DiffRow {
            op: "sort".into(),
            shape: "B=1024 N=1024 f32".into(),
            baseline_pct: Some(100.0),
            current_pct: Some(65.0),
            delta_pct: Some(-35.0),
            kind: DeltaKind::Regression,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "▼ -35%");
    }

    #[test]
    fn improvement_delta_is_unstyled_text() {
        let row = DiffRow {
            op: "rms_norm".into(),
            shape: "B=1024 N=64 bf16".into(),
            baseline_pct: Some(332.0),
            current_pct: Some(420.0),
            delta_pct: Some(88.0),
            kind: DeltaKind::Improvement,
        };
        let d = build_diff_row_display(&row);
        assert_eq!(d.delta, "▲ +88%");
    }
}
