//! MetalTile Benchmark Suite
//!
//! Runs all LLM operations and prints a single comprehensive table showing
//! reference (MLX Metal kernels) vs MetalTile-generated performance.
//!
//! Usage:  cargo run --release -p metaltile-bench --bin bench_suite
//!         cargo run --release -p metaltile-bench --bin bench_suite -- --json results/run.json
//!         cargo run --release -p metaltile-bench --bin bench_suite -- --filter softmax

use metaltile_bench::{
    ops::{
        CorrectnessStatus,
        OpResult,
        SuitePrinter,
        bench_matmul_fp16,
        set_result_reporter,
        validate_results,
    },
    runner::GpuRunner,
    spec::BenchSpec,
    term::{Color, Style, paint_stderr, paint_stdout},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_out = flag_val(&args, "--json");
    let filter = flag_val(&args, "--filter");

    let runner = match GpuRunner::new() {
        Ok(r) => r,
        Err(e) => {
            println!(
                "{} {}",
                paint_stdout("[skip]", Style::new().fg(Color::Yellow).bold()),
                paint_stdout(e, Style::new().fg(Color::BrightWhite))
            );
            return;
        },
    };

    println!(
        "{}",
        paint_stdout(
            "╔═══════════════════════════════════════════════════════════════════════════════╗",
            Style::new().fg(Color::Cyan).bold()
        )
    );
    println!(
        "{}",
        paint_stdout(
            "║  MetalTile Benchmark Suite                                                  ║",
            Style::new().fg(Color::BrightWhite).bold()
        )
    );
    println!(
        "{}",
        paint_stdout(
            "╚═══════════════════════════════════════════════════════════════════════════════╝",
            Style::new().fg(Color::Cyan).bold()
        )
    );
    println!(
        "\n{} {}",
        paint_stdout("Device:", Style::new().fg(Color::BrightBlack).bold()),
        paint_stdout(&runner.device_name, Style::new().fg(Color::BrightWhite).bold())
    );

    // Run all ops, optionally narrowed to a single substring filter.
    let mut all: Vec<OpResult> = Vec::new();
    let mut printer = SuitePrinter::new(true);
    {
        let mut report = |result: &OpResult| {
            if matches_filter(filter.as_deref(), result.op()) {
                printer.print_batch(std::slice::from_ref(result));
            }
        };
        let _reporter = set_result_reporter(&mut report);

        // Matrix multiply
        extend_if_selected(&mut all, &runner, &filter, bench_matmul_fp16);
        // All other ops — inventory-registered via #[bench_kernel]
        {
            let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
            specs.sort_unstable_by_key(|s| (s.op, s.subop));
            for spec in specs {
                if matches_filter(filter.as_deref(), spec.op) {
                    for &dt in spec.dtypes {
                        runner.flush_slc();
                        all.extend(spec.run(&runner, dt));
                    }
                }
            }
        }
    }

    if all.is_empty() {
        if let Some(pattern) = &filter {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    format!("No benchmarks matched --filter {pattern:?}"),
                    Style::new().fg(Color::BrightWhite)
                )
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr("No benchmarks ran", Style::new().fg(Color::BrightWhite))
            );
        }
        return;
    }

    validate_results(&all).unwrap_or_else(|err| panic!("{err}"));
    printer.finish();

    // Summary
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

    let mut summary = vec![
        summary_item(
            "Implemented",
            &format!("{impl_count}/{}", all.len()),
            Style::new().fg(Color::Green).bold(),
        ),
        summary_item("NYI", &nyi_count.to_string(), Style::new().fg(Color::Yellow).bold()),
    ];
    if let Some(p) = avg_pct {
        summary.push(summary_item("Avg MT%", &format!("{p:.0}%"), pct_style(p)));
    }
    if checked_count > 0 {
        summary.push(summary_item(
            "Correct",
            &format!("{equiv_pass}/{checked_count}"),
            if equiv_fail == 0 {
                Style::new().fg(Color::Green).bold()
            } else {
                Style::new().fg(Color::Yellow).bold()
            },
        ));
    }
    if !unchecked.is_empty() {
        summary.push(summary_item(
            "Unchecked",
            &unchecked.len().to_string(),
            Style::new().fg(Color::Yellow).bold(),
        ));
    }
    let summary_sep = format!(" {} ", summary_sep());
    println!("  {}", summary.join(&summary_sep));
    if equiv_fail > 0 {
        println!(
            "  {} {}",
            paint_stdout("Correctness failures:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(equiv_fail.to_string(), Style::new().fg(Color::Red).bold())
        );
    }
    if !unchecked.is_empty() {
        println!(
            "  {} {}",
            paint_stdout("Unchecked MT results:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(unchecked.join(", "), Style::new().fg(Color::Yellow).bold())
        );
    }
    println!();

    if let Some(path) = json_out {
        save_json(&runner.device_name, &all, &path);
    }
}

fn save_json(device: &str, results: &[OpResult], path: &str) {
    use std::io::Write;
    let mut out = String::new();
    out.push_str(&format!("{{\"device\":{:?},\"results\":[\n", device));
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        out.push_str(&format!(
            "  {{\"op\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}{}\n",
            r.op(),
            r.shape(),
            r.metric(),
            json_f(r.ref_perf()),
            json_f(r.mt_perf()),
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
            paint_stdout(path, Style::new().fg(Color::BrightWhite))
        ),
        Err(e) => eprintln!(
            "  {} {}",
            paint_stderr("save failed:", Style::new().fg(Color::Red).bold()),
            paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite))
        ),
    }
}

fn json_f(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "null".into())
}

fn extend_if_selected(
    all: &mut Vec<OpResult>,
    runner: &GpuRunner,
    filter: &Option<String>,
    run: fn(&GpuRunner) -> Vec<OpResult>,
) {
    all.extend(run(runner).into_iter().filter(|r| matches_filter(filter.as_deref(), r.op())));
}

fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn summary_item(label: &str, value: &str, value_style: Style) -> String {
    format!(
        "{} {}",
        paint_stdout(label, Style::new().fg(Color::BrightBlack).bold()),
        paint_stdout(value, value_style)
    )
}

fn summary_sep() -> String { paint_stdout("|", Style::new().fg(Color::BrightBlack).dim()) }

fn pct_style(pct: f64) -> Style {
    if pct >= 90.0 {
        Style::new().fg(Color::Green).bold()
    } else if pct >= 60.0 {
        Style::new().fg(Color::Yellow).bold()
    } else {
        Style::new().fg(Color::Red).bold()
    }
}

#[cfg(test)]
mod tests {
    use super::matches_filter;

    #[test]
    fn filter_matches_case_insensitively() {
        assert!(matches_filter(Some("Soft"), "softmax"));
        assert!(matches_filter(Some("NORM"), "layer_norm"));
        assert!(!matches_filter(Some("gemv"), "matmul"));
        assert!(matches_filter(None, "anything"));
    }
}
