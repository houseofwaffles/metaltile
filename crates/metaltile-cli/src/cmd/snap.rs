//! `tile snap` — Save current bench results as a regression baseline.
//!
//! Usage:
//!   tile snap                                                     # run bench then save
//!   tile snap -o .tile-snapshots/$(git rev-parse --short HEAD).json
//!   tile snap --from results/run.json                             # promote existing JSON
//!   tile snap --from results/run.json --note "after fusion fix"

use std::process::Command;

use serde::Serialize;
use serde_json::Value;

use crate::{
    flag_val,
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(Serialize)]
struct Snapshot {
    device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_sha: Option<String>,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    results: Vec<Value>,
}

pub fn run(args: &[String]) {
    let out_path = flag_val(args, "--out").or_else(|| flag_val(args, "-o"));
    let from_path = flag_val(args, "--from");
    let note = flag_val(args, "--note");
    let filter = flag_val(args, "--filter").or_else(|| flag_val(args, "-f"));

    // Default output path
    let out_path = out_path.unwrap_or_else(|| {
        let date = chrono_like_now();
        format!(".tile-snapshots/{}.json", date)
    });

    let results_json: Value = if let Some(ref from) = from_path {
        // Load existing JSON
        let content = std::fs::read_to_string(from).unwrap_or_else(|e| {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    format!("cannot read {from}: {e}"),
                    Style::new().fg(Color::BrightWhite)
                ),
            );
            std::process::exit(1);
        });
        serde_json::from_str(&content).unwrap_or_else(|e| {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(format!("invalid JSON: {e}"), Style::new().fg(Color::BrightWhite)),
            );
            std::process::exit(1);
        })
    } else {
        // Run bench and capture JSON
        eprintln!(
            "  {}",
            paint_stdout("Running bench suite...", Style::new().fg(Color::Cyan).bold()),
        );
        let temp_file =
            std::env::temp_dir().join(format!(".tile-snap-tmp-{}.json", std::process::id()));
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

        serde_json::from_str(&content).unwrap_or_else(|e| {
            eprintln!("invalid bench JSON: {e}");
            std::process::exit(1);
        })
    };

    // Extract results array and device name
    let device =
        results_json.get("device").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    let mut results: Vec<Value> =
        results_json.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default();

    // Apply filter
    if let Some(ref f) = filter {
        let f_lower = f.to_ascii_lowercase();
        results.retain(|r| {
            r.get("op")
                .and_then(|v| v.as_str())
                .map(|op| op.to_ascii_lowercase().contains(&f_lower))
                .unwrap_or(false)
        });
    }

    // Get git SHA
    let git_sha =
        Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    // GPU family heuristic
    let gpu_family = gpu_family_from_name(&device).map(|s| s.to_string());

    // Timestamp
    let timestamp = chrono_like_now();
    let result_count = results.len();
    let note_suffix = note.as_ref().map(|n| format!(", \"{n}\"")).unwrap_or_default();

    let snapshot = Snapshot { device, gpu_family, git_sha, timestamp, note, results };

    // Write snapshot
    let dir = std::path::Path::new(&out_path).parent().unwrap_or(".".as_ref());
    std::fs::create_dir_all(dir).unwrap_or_else(|e| {
        eprintln!("cannot create directory: {e}");
        std::process::exit(1);
    });

    let json = serde_json::to_string_pretty(&snapshot).unwrap();
    std::fs::write(&out_path, &json).unwrap_or_else(|e| {
        eprintln!("cannot write snapshot: {e}");
        std::process::exit(1);
    });

    println!(
        "  {} {}  ({} results{})",
        paint_stdout("Saved →", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(&out_path, Style::new().fg(Color::BrightWhite)),
        result_count,
        note_suffix,
    );
}

/// Basic ISO-like timestamp without a chrono dependency.
fn chrono_like_now() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    // Convert to a rough readable date string
    // epoch days since 1970-01-01
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let mins = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    // Simple date calculation (approximate, good enough for snapshot naming)
    let y = 1970;
    let mut remaining_days = days as i64;
    let mut year = y;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }
    let month_days = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        month += 1;
    }
    let day = remaining_days + 1;

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{s:02}Z")
}

fn is_leap(y: i64) -> bool { (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) }

fn gpu_family_from_name(name: &str) -> Option<&'static str> {
    if name.contains("M4") || name.contains("M3") {
        Some("Apple9")
    } else if name.contains("M2") {
        Some("Apple8")
    } else if name.contains("M1") {
        Some("Apple7")
    } else {
        None
    }
}
