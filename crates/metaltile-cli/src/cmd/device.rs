//! `tile device` — Show GPU device info and supported feature flags.

use metaltile_core::GpuFamily;
use metaltile_std::runner::GpuRunner;

use crate::{
    DeviceArgs,
    term::{Color, Style, paint_stdout},
};

pub fn run(args: &DeviceArgs) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!("device", json = args.json).entered();
    let json_out = args.json;

    let runner = match GpuRunner::new() {
        Ok(r) => r,
        Err(e) => {
            if json_out {
                println!("{{\"error\":{:?}}}", e);
                return Ok(());
            }
            eprintln!(
                "{} {}",
                paint_stdout("error:", Style::new().fg(Color::Red).bold()),
                paint_stdout(&e, Style::new().fg(Color::BrightWhite)),
            );
            return Err(crate::CliError::GpuInit(e));
        },
    };

    let device_name = &runner.device_name;
    let simd = runner.supports_simd_matrix();

    let gpu_family = GpuFamily::from_device_name(device_name);

    // Native bfloat (Metal 3.1 `bfloat` type) and async threadgroup copy both
    // require Apple9 (M3 / A17) or later, independent of SIMD matrix support.
    let apple9_or_later = gpu_family.is_apple9_or_later();

    // Threadgroup memory and max TPG are constant across Apple7-9.
    let tpg_mem = gpu_family.threadgroup_mem_kb();
    let max_tpg = gpu_family.max_threads_per_threadgroup();

    if json_out {
        println!(
            "{{\"device\":{:?},\"gpu_family\":{:?},\"simdgroup_hw\":{},\"native_bfloat\":{},\"threadgroup_mem_kb\":{},\"max_tpg\":{}}}",
            device_name,
            gpu_family.code().unwrap_or("unknown"),
            simd,
            apple9_or_later,
            tpg_mem,
            max_tpg,
        );
        return Ok(());
    }

    let label_style = Style::new().fg(Color::BrightBlack).bold();

    eprintln!("{}", paint_stdout("tile device", Style::new().fg(Color::Cyan).bold()),);
    eprintln!();
    eprintln!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "Device"), label_style),
        paint_stdout(device_name, Style::new().fg(Color::BrightWhite)),
    );
    println!(
        "  {}  {}",
        paint_stdout(format!("{:<16}", "GPU family"), label_style),
        paint_stdout(gpu_family.display_label(), Style::new().fg(Color::BrightWhite)),
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

    check("native_bfloat", apple9_or_later, "Metal 3.1+ bfloat type");
    check("simdgroup_hw", simd, "simdgroup matrix multiply");
    check("async_copy", apple9_or_later, "async threadgroup copy (M3+)");

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
        paint_stdout(GpuFamily::slc_label(device_name), Style::new().fg(Color::BrightWhite)),
    );
    println!();
    Ok(())
}
