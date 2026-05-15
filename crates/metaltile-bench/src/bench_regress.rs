//! CI regression checker.
//!
//! Compares two bench_suite JSON result files and exits 1 if any op regresses
//! beyond a threshold.
//!
//! Usage:
//!   cargo run --release -p metaltile-bench --bin bench_regress -- \
//!       --current results/current.json --baseline results/baseline.json \
//!       --threshold 0.05
//!   cargo run --release -p metaltile-bench --bin bench_regress -- \
//!       --current results/current.json --baseline results/baseline.json \
//!       --update-baseline

use std::collections::HashMap;

use metaltile_bench::term::{Color, Style, paint_stderr, paint_stdout};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RunResult {
    op: String,
    shape: String,
    metric: String,
    mt: Option<f64>,
}

fn load(path: &str) -> Vec<RunResult> {
    #[derive(Deserialize)]
    struct Root {
        results: Vec<RunResult>,
    }
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!(
            "{} {} {}",
            paint_stderr("cannot read", Style::new().fg(Color::Red).bold()),
            paint_stderr(path, Style::new().fg(Color::BrightWhite).bold()),
            paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
        );
        std::process::exit(2);
    });
    serde_json::from_str::<Root>(&txt)
        .unwrap_or_else(|e| {
            eprintln!(
                "{} {} {}",
                paint_stderr("bad JSON", Style::new().fg(Color::Red).bold()),
                paint_stderr(path, Style::new().fg(Color::BrightWhite).bold()),
                paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
            );
            std::process::exit(2);
        })
        .results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let current = flag_val(&args, "--current").unwrap_or_else(|| die("--current required"));
    let baseline = flag_val(&args, "--baseline").unwrap_or_else(|| die("--baseline required"));
    let thresh: f64 = flag_val(&args, "--threshold").and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let update_baseline = has_flag(&args, "--update-baseline");

    if update_baseline {
        update_baseline_file(&current, &baseline);
        println!(
            "  {} {}",
            paint_stdout("Updated baseline →", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(&baseline, Style::new().fg(Color::BrightWhite))
        );
        return;
    }

    let cur: HashMap<String, f64> =
        load(&current).into_iter().filter_map(|r| r.mt.map(|v| (result_key(&r), v))).collect();
    let base: HashMap<String, f64> =
        load(&baseline).into_iter().filter_map(|r| r.mt.map(|v| (result_key(&r), v))).collect();

    let mut regressions = 0usize;
    for (key, base_val) in &base {
        if let Some(&cur_val) = cur.get(key) {
            let drop = (base_val - cur_val) / base_val;
            if drop > thresh {
                eprintln!(
                    "  {} {}  {}  {}  {}",
                    paint_stderr("REGRESS", Style::new().fg(Color::Red).bold()),
                    paint_stderr(key, Style::new().fg(Color::BrightWhite).bold()),
                    paint_stderr(
                        format!("baseline={base_val:.1}"),
                        Style::new().fg(Color::BrightBlack).bold()
                    ),
                    paint_stderr(
                        format!("current={cur_val:.1}"),
                        Style::new().fg(Color::BrightBlack).bold()
                    ),
                    paint_stderr(
                        format!("drop={:.1}%", drop * 100.0),
                        Style::new().fg(Color::Red).bold()
                    )
                );
                regressions += 1;
            }
        }
    }

    if regressions == 0 {
        println!(
            "  {} {}",
            paint_stdout("No regressions", Style::new().fg(Color::Green).bold()),
            paint_stdout(
                format!("(threshold={:.1}%)", thresh * 100.0),
                Style::new().fg(Color::BrightBlack).bold()
            )
        );
    } else {
        eprintln!(
            "  {} {}",
            paint_stderr("Regressions:", Style::new().fg(Color::Red).bold()),
            paint_stderr(regressions.to_string(), Style::new().fg(Color::BrightWhite).bold())
        );
        std::process::exit(1);
    }
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool { args.iter().any(|arg| arg == name) }

fn result_key(r: &RunResult) -> String { format!("{}/{}/{}", r.op, r.shape, r.metric) }

fn update_baseline_file(current: &str, baseline: &str) {
    let current_txt = std::fs::read_to_string(current).unwrap_or_else(|e| {
        eprintln!(
            "{} {} {}",
            paint_stderr("cannot read", Style::new().fg(Color::Red).bold()),
            paint_stderr(current, Style::new().fg(Color::BrightWhite).bold()),
            paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
        );
        std::process::exit(2);
    });
    serde_json::from_str::<serde_json::Value>(&current_txt).unwrap_or_else(|e| {
        eprintln!(
            "{} {} {}",
            paint_stderr("bad JSON", Style::new().fg(Color::Red).bold()),
            paint_stderr(current, Style::new().fg(Color::BrightWhite).bold()),
            paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
        );
        std::process::exit(2);
    });

    if let Some(parent) = std::path::Path::new(baseline).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "{} {} {}",
                paint_stderr("cannot create", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    parent.display().to_string(),
                    Style::new().fg(Color::BrightWhite).bold()
                ),
                paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
            );
            std::process::exit(2);
        }
    }
    if let Err(e) = std::fs::write(baseline, current_txt) {
        eprintln!(
            "{} {} {}",
            paint_stderr("cannot write", Style::new().fg(Color::Red).bold()),
            paint_stderr(baseline, Style::new().fg(Color::BrightWhite).bold()),
            paint_stderr(format!(": {e}"), Style::new().fg(Color::BrightWhite))
        );
        std::process::exit(2);
    }
}

fn die(msg: &str) -> String {
    eprintln!(
        "{} {}",
        paint_stderr("error:", Style::new().fg(Color::Red).bold()),
        paint_stderr(msg, Style::new().fg(Color::BrightWhite))
    );
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::{RunResult, has_flag, result_key};

    #[test]
    fn result_key_includes_metric() {
        let rr = RunResult {
            op: "softmax".into(),
            shape: "B=4 N=256".into(),
            metric: "GB/s".into(),
            mt: Some(1.0),
        };
        assert_eq!(result_key(&rr), "softmax/B=4 N=256/GB/s");
    }

    #[test]
    fn flag_detection_is_exact() {
        let args = vec![
            "bench_regress".to_string(),
            "--update-baseline".to_string(),
            "--current".to_string(),
        ];
        assert!(has_flag(&args, "--update-baseline"));
        assert!(!has_flag(&args, "--ci"));
    }
}
