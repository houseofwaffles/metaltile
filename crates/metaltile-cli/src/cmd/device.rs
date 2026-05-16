//! `tile device` — Show GPU device info and supported feature flags.

use crate::{
    flag_present,
    runner::GpuRunner,
    term::{Color, Style, paint_stdout},
};

pub fn run(args: &[String]) {
    let json_out = flag_present(args, "--json");

    let runner = match GpuRunner::new() {
        Ok(r) => r,
        Err(e) => {
            if json_out {
                println!("{{\"error\":{:?}}}", e);
                return;
            }
            eprintln!(
                "{} {}",
                paint_stdout("error:", Style::new().fg(Color::Red).bold()),
                paint_stdout(e, Style::new().fg(Color::BrightWhite)),
            );
            std::process::exit(1);
        },
    };

    let device_name = &runner.device_name;
    let simd = runner.supports_simd_matrix();

    // Heuristic GPU family string based on device name.
    let gpu_family = gpu_family_from_name(device_name);

    // Native bfloat (Metal 3.1 `bfloat` type) and async threadgroup copy both
    // require Apple9 (M3 / A17) or later, independent of SIMD matrix support.
    let apple9_or_later = gpu_family.contains("Apple9");
    let native_bfloat = apple9_or_later;
    let async_copy = apple9_or_later;

    // Threadgroup memory is inferred from GPU family.
    let tpg_mem = tpg_memory_from_family(gpu_family);
    let max_tpg = max_threads_per_threadgroup(gpu_family);

    if json_out {
        println!(
            "{{\"device\":{:?},\"gpu_family\":{:?},\"simdgroup_hw\":{},\"native_bfloat\":{},\"threadgroup_mem_kb\":{},\"max_tpg\":{}}}",
            device_name, gpu_family, simd, native_bfloat, tpg_mem, max_tpg
        );
        return;
    }

    let label_style = Style::new().fg(Color::BrightBlack).bold();

    println!();
    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "Device"), label_style),
        paint_stdout(device_name, Style::new().fg(Color::BrightWhite)),
    );
    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "GPU family"), label_style),
        paint_stdout(gpu_family, Style::new().fg(Color::BrightWhite)),
    );
    println!("  {}", paint_stdout("─".repeat(42), Style::new().fg(Color::BrightBlack).dim(),),);

    // Feature flags
    let check = |label: &str, supported: bool, note: &str| {
        let sym = if supported {
            paint_stdout("✓", Style::new().fg(Color::Green).bold())
        } else {
            paint_stdout("✗", Style::new().fg(Color::Red).bold())
        };
        println!(
            "  {}  {sym}   {}",
            paint_stdout(format!("{label:<16}"), label_style),
            paint_stdout(note, Style::new().fg(Color::BrightBlack).dim()),
        );
    };

    check("native_bfloat", native_bfloat, "Metal 3.1+ bfloat type");
    check("simdgroup_hw", simd, "simdgroup matrix multiply");
    check("async_copy", async_copy, "async threadgroup copy (M3+)");

    println!("  {}", paint_stdout("─".repeat(42), Style::new().fg(Color::BrightBlack).dim(),),);

    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "Threadgroup"), label_style),
        paint_stdout(format!("{tpg_mem} KB"), Style::new().fg(Color::BrightWhite)),
    );
    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "Max TPG"), label_style),
        paint_stdout(format!("{max_tpg}"), Style::new().fg(Color::BrightWhite)),
    );
    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "SLC"), label_style),
        paint_stdout(slc_size_from_name(device_name), Style::new().fg(Color::BrightWhite)),
    );
    println!();
}

/// Heuristic GPU family based on device name substring.
/// M-series checked before A-series since "M1 Pro" etc. contain neither A-chip name.
fn gpu_family_from_name(name: &str) -> &'static str {
    if name.contains("M4") {
        "Apple9 (M4)"
    } else if name.contains("M3") {
        "Apple9 (M3)"
    } else if name.contains("M2") {
        "Apple8 (M2)"
    } else if name.contains("M1") {
        "Apple7 (M1)"
    } else if name.contains("A18") || name.contains("A17") {
        "Apple9 (A17+)"
    } else if name.contains("A16") || name.contains("A15") {
        "Apple8 (A15+)"
    } else if name.contains("A14") {
        "Apple7 (A14)"
    } else {
        "unknown"
    }
}

/// Known SLC sizes per chip tier. Returns "varies" when the tier is not recognised.
fn slc_size_from_name(name: &str) -> &'static str {
    if name.contains("Ultra") {
        "~96 MB"
    } else if name.contains("Max") && name.contains("M4") {
        "~64 MB"
    } else if name.contains("Max") {
        "~48 MB"
    } else {
        "varies"
    }
}

fn tpg_memory_from_family(family: &str) -> u32 {
    match family {
        "Apple9 (M4)" | "Apple9 (M3+)" | "Apple9 (A17+)" => 32,
        "Apple8 (M2)" | "Apple8 (A16)" => 32,
        "Apple7 (M1)" | "Apple7 (A15)" => 32,
        _ => 32,
    }
}

fn max_threads_per_threadgroup(family: &str) -> u32 {
    match family {
        "Apple9 (M4)" | "Apple9 (M3+)" | "Apple9 (A17+)" => 1024,
        _ => 1024,
    }
}
