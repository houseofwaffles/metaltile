//! GPU micro-benchmark runner.
//! Usage: cargo run --release -p metaltile-bench --bin gpu_bench
//!
//! Runs individual op benchmarks with verbose output.

use metaltile_bench::{
    ops::{
        OpResult,
        SuitePrinter,
        bench_elementwise_f32,
        bench_rms_norm_f32,
        bench_softmax_f32,
        set_result_reporter,
        validate_results,
    },
    runner::GpuRunner,
    term::{Color, Style, paint_stdout},
};

fn main() {
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
        "{} {}",
        paint_stdout("Device:", Style::new().fg(Color::BrightBlack).bold()),
        paint_stdout(&runner.device_name, Style::new().fg(Color::BrightWhite).bold())
    );
    let mut all = Vec::new();
    let mut printer = SuitePrinter::new(true);
    {
        let mut report = |result: &OpResult| printer.print_batch(std::slice::from_ref(result));
        let _reporter = set_result_reporter(&mut report);
        all.extend(bench_elementwise_f32(&runner));
        all.extend(bench_softmax_f32(&runner));
        all.extend(bench_rms_norm_f32(&runner));
    }
    validate_results(&all).unwrap_or_else(|err| panic!("{err}"));
    printer.finish();
}
