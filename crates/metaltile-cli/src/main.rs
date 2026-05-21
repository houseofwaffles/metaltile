//! MetalTile CLI — `tile` binary.
//!
//! Subcommands:
//!   bench     Benchmark suite: MetalTile vs MLX reference
//!   build     Compile all kernels to MSL and report errors
//!   inspect   Print IR and/or MSL for one kernel
//!   device    Show GPU device info and supported features
//!   snap      Save bench results as a regression baseline
//!   diff      Compare bench results to a saved baseline

mod cmd;
mod error;
pub mod git;
pub mod suite_printer;
pub mod term;
use anstyle::AnsiColor;
use clap::{Parser, builder::Styles};
pub use error::CliError;

const CLAP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Cyan.on_default().bold())
    .usage(AnsiColor::Cyan.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::BrightBlack.on_default())
    .error(AnsiColor::Red.on_default().bold())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

/// MetalTile CLI — benchmark and inspect GPU kernels on Apple Silicon.
#[derive(Parser)]
#[command(name = "tile", version, about, styles = CLAP_STYLES)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Benchmark suite: MetalTile vs MLX reference
    Bench(BenchArgs),
    /// Compile all kernels to MSL and report errors
    Build(BuildArgs),
    /// Print IR and/or MSL for registered kernels
    Inspect(InspectArgs),
    /// Show GPU device info and supported feature flags
    Device(DeviceArgs),
    /// Save bench results as a regression baseline
    Snap(SnapArgs),
    /// Compare bench results against a saved baseline
    Diff(DiffArgs),
}

// ── Bench ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct BenchArgs {
    /// Only run kernels whose name contains this text
    #[arg(long = "filter", short = 'f')]
    filter: Option<String>,
    /// Show occupancy and register profile (-v) and GPU timing stats (-vv).
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    verbose: u8,
    /// Write results as JSON to this file
    #[arg(long = "json", short = 'o')]
    json: Option<String>,
    /// Run even if the working tree has tracked-file modifications.
    /// Without this flag, bench refuses to run on a dirty tree so the
    /// numbers always tie back to a clean commit SHA.
    #[arg(long = "allow-dirty")]
    allow_dirty: bool,
    /// Skip the post-bench diff against the target-branch baseline.
    #[arg(long = "no-diff")]
    no_diff: bool,
    /// Git ref whose `baselines/<chip>.json` to diff against (default:
    /// first of `origin/dev`, `upstream/dev`, `dev` that resolves).
    #[arg(long = "baseline-ref")]
    baseline_ref: Option<String>,
}

// ── Build ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct BuildArgs {
    /// Only build kernels whose name contains this text
    #[arg(long = "filter", short = 'f')]
    filter: Option<String>,
    /// Comma-separated list of dtypes to build (f32,f16,bf16)
    #[arg(long = "dtypes")]
    dtypes: Option<String>,
    /// Print generated MSL for each kernel (-v for verbose)
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    verbose: u8,
    /// Comma-separated: msl,metallib,swift,ir,all
    #[arg(long = "emit")]
    emit: Option<String>,
    /// Output directory (required when --emit is set)
    #[arg(long = "out", short = 'o')]
    out: Option<String>,
    /// xcrun SDK (default: macosx)
    #[arg(long = "sdk", default_value = "macosx")]
    sdk: String,
    /// Run the standard pass pipeline 25× per kernel and print per-pass
    /// median wall_us instead of emitting MSL (after 5 warmup iters).
    /// Inherits `--filter` and `--dtypes`.
    #[arg(long = "time-passes", short = 't')]
    time_passes: bool,
}

// ── Inspect ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct InspectArgs {
    /// Kernel name to inspect (list all if omitted)
    kernel: Option<String>,
    /// Filter kernels by name substring
    #[arg(long = "filter")]
    filter: Option<String>,
    /// Process all kernels
    #[arg(long = "all")]
    all: bool,
    /// Print raw IR before any passes
    #[arg(long = "ir")]
    ir: bool,
    /// Print per-pass op-count reduction table
    #[arg(long = "stats")]
    stats: bool,
    /// Print IR after a specific pass name (or 'all' for every stage)
    #[arg(long = "pass")]
    pass: Option<String>,
    /// Dtype override (f32, f16, bf16, i32, u32)
    #[arg(long = "dtype")]
    dtype: Option<String>,
    /// Write output files to <path> instead of stdout
    #[arg(long = "dir", short = 'o')]
    dir: Option<String>,
}

// ── Device ───────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DeviceArgs {
    /// Output as JSON
    #[arg(long = "json")]
    json: bool,
}

// ── Snap ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct SnapArgs {
    /// Write snapshot to <file> (default: .tile-snapshots/<sha>.json)
    #[arg(long = "out", short = 'o')]
    out: Option<String>,
    /// Promote an existing JSON file instead of re-running bench
    #[arg(long = "from")]
    from: Option<String>,
    /// Attach a note to the snapshot
    #[arg(long = "note")]
    note: Option<String>,
    /// Only include kernels whose name contains this text
    #[arg(long = "filter", short = 'f')]
    filter: Option<String>,
}

// ── Diff ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DiffArgs {
    /// Baseline JSON file
    baseline: String,
    /// Current JSON file (runs bench if omitted)
    current: Option<String>,
    /// Only show kernels whose name contains this text
    #[arg(long = "filter", short = 'f')]
    filter: Option<String>,
    /// Highlight regressions larger than this percentage (default: 5)
    #[arg(long = "threshold", default_value = "5.0")]
    threshold: f64,
    /// Sort by: name, delta, pct (default: name)
    #[arg(long = "sort", default_value = "name")]
    sort: String,
    /// Only show regressions
    #[arg(long = "only-regressions")]
    only_regressions: bool,
    /// Only show improvements
    #[arg(long = "only-improvements")]
    only_improvements: bool,
}

// ── Dispatch ─────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing. METALTILE_DEBUG=1 enables debug-level output for all
    // metaltile crates; METALTILE_DEBUG=trace enables trace level.
    // When the env-var is absent the subscriber is still installed but the filter
    // rejects everything, so library crates pay only the ~1 ns no-subscriber cost.
    let _debug_level = std::env::var("METALTILE_DEBUG").ok();
    let filter = match _debug_level.as_deref() {
        Some("1") | Some("debug") => "metaltile=debug",
        Some("trace") => "metaltile=trace",
        _ => "off",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        // Diagnostics go to stderr so they don't interleave with bench/build
        // output on stdout. `with_target` shows which crate/module emitted
        // each event — useful when tracing spans multiple crates.
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        // Print a line when each span closes so you see elapsed wall time.
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .compact()
        .init();

    let cli = Cli::parse();
    let _span = tracing::info_span!("tile", command = ?cli.command).entered();

    match cli.command {
        Command::Bench(args) => cmd::bench::run(&args)?,
        Command::Build(args) => cmd::build::run(&args)?,
        Command::Inspect(args) => cmd::inspect::run(&args)?,
        Command::Device(args) => cmd::device::run(&args)?,
        Command::Snap(args) => cmd::snap::run(&args)?,
        Command::Diff(args) => cmd::diff::run(&args)?,
    }
    Ok(())
}

/// Filter helper: case-insensitive substring match.
pub(crate) fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}
