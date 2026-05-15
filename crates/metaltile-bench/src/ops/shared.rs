use std::{cell::RefCell, io::Write, ptr::NonNull};

pub use metaltile::core::dtype::DType;

use crate::{
    runner::{CompiledKernel, GpuBuffer, GpuRunner},
    term::{Color, Style, paint_stdout},
};

// ── Dtype variant helpers ─────────────────────────────────────────────────────

/// All floating-point dtypes to iterate over in multi-variant benches.
pub const FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];
/// Integer dtypes supported by MLX elementwise and copy kernels.
pub const INTEGER_DTYPES: &[DType] = &[DType::I32, DType::U32, DType::I8, DType::U8];

pub fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::Bool => "bool",
        _ => "?",
    }
}

/// MLX template-name suffix used in kernel instantiation strings.
pub fn mlx_tname(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I32 => "int32",
        DType::U32 => "uint32",
        DType::I8 => "int8",
        DType::U8 => "uint8",
        DType::Bool => "bool_",
        _ => "float32",
    }
}

/// Bytes per element.
pub fn elem_bytes(dt: DType) -> usize {
    match dt {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U8 | DType::Bool | DType::I8 => 1,
        _ => 4,
    }
}

/// Absolute-error tolerance for elementwise op correctness checks.
pub fn dtype_tol(dt: DType) -> f32 {
    match dt {
        DType::F32 => 1e-4,
        // f16 ULP at magnitude ~20 (e.g. exp(3)) is ~0.016, so 1.5e-2 covers one ULP.
        DType::F16 => 1.5e-2,
        // bf16 ULP at magnitude ~17 (e.g. pow(3,2.5)) is ~0.125, so 1.3e-1 covers 1 ULP.
        DType::BF16 => 1.3e-1,
        // Integers are exact — zero tolerance.
        _ => 0.0,
    }
}

/// Absolute-error tolerance for reduction ops (accumulated rounding over many elements).
pub fn dtype_tol_reduce(dt: DType) -> f32 {
    match dt {
        DType::F32 => 1e-3,
        // f16 accumulation of ~512 elements summing to ~224 can have 1 ULP ≈ 0.25 error
        // vs an f32-accumulated reference.
        DType::F16 => 0.5,
        // MT accumulates in float32 (accurate), MLX accumulates in bfloat (lossy).
        // For 16 384 elements summing to ~9 000, BF16 accumulated error ≈ sum * 2^-7 ≈ 70.
        DType::BF16 => 128.0,
        _ => 1e-3,
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    // Round-to-nearest-even
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    // Mantissa overflow bumps exponent
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    let rounded = x.wrapping_add(0x7FFF).wrapping_add((x >> 16) & 1);
    (rounded >> 16) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        return f32::from_bits(sign);
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exp5 + 112) << 23) | (mantissa << 13))
}

fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

/// Upload `vals` (f32) to GPU as the given dtype.
pub fn buffer_typed(runner: &GpuRunner, vals: &[f32], dt: DType) -> GpuBuffer {
    match dt {
        DType::F32 => runner.buffer_f32(vals),
        DType::F16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_f16(v)).collect::<Vec<_>>()),
        DType::BF16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>()),
        DType::I32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect::<Vec<_>>()),
        DType::U32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect::<Vec<_>>()),
        DType::I8 => runner.buffer_bytes(&vals.iter().map(|&v| v as i8 as u8).collect::<Vec<_>>()),
        DType::U8 => runner.buffer_bytes(&vals.iter().map(|&v| v as u8).collect::<Vec<_>>()),
        _ => runner.buffer_f32(vals),
    }
}

/// Allocate a zeroed output buffer for `n` elements of dtype `dt`.
pub fn zeros_typed(runner: &GpuRunner, n: usize, dt: DType) -> GpuBuffer {
    runner.buffer_zeros(n * elem_bytes(dt))
}

/// Read `n` typed elements back as f32.
pub fn read_typed(runner: &GpuRunner, buf: &GpuBuffer, n: usize, dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => runner.read_f32_slice(buf, n),
        DType::F16 => runner.read_f16_slice(buf, n),
        DType::BF16 => runner.read_bf16_slice(buf, n),
        DType::I32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::U32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::I8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as i8 as f32).collect()
        },
        DType::U8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as f32).collect()
        },
        _ => runner.read_f32_slice(buf, n),
    }
}

/// Quantize `vals` through `dt` and back to f32 so the cpu_ref uses the same
/// representable values that the GPU will actually receive.
pub fn quantize_roundtrip(vals: &[f32], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => vals.to_vec(),
        DType::F16 => vals.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect(),
        DType::BF16 => vals.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect(),
        DType::I32 => vals.iter().map(|&v| v as i32 as f32).collect(),
        DType::U32 => vals.iter().map(|&v| v as u32 as f32).collect(),
        DType::I8 => vals.iter().map(|&v| v as i8 as f32).collect(),
        DType::U8 => vals.iter().map(|&v| v as u8 as f32).collect(),
        _ => vals.to_vec(),
    }
}

thread_local! {
    static RESULT_REPORTER: RefCell<Option<NonNull<dyn FnMut(&OpResult)>>> = RefCell::new(None);
}

pub const DEFAULT_MIN_COSINE_SIM: f32 = 0.999;

/// Result of a numerical equivalence check between the reference and MT kernels.
#[derive(Debug, Clone)]
pub struct EquivResult {
    /// Number of elements compared.
    pub n_checked: usize,
    /// Maximum absolute element-wise error.
    pub max_abs_err: f32,
    /// Cosine similarity across the compared vectors.
    pub cosine_sim: f32,
    /// True iff all correctness thresholds were satisfied.
    pub passed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EquivTolerance {
    pub max_abs_err: f32,
    pub min_cosine_sim: f32,
}

impl EquivTolerance {
    pub const fn new(max_abs_err: f32, min_cosine_sim: f32) -> Self {
        Self { max_abs_err, min_cosine_sim }
    }
}

/// Compare reference and MT output arrays element-wise.
/// Uses the provided absolute error tolerance plus a cosine-similarity floor.
pub fn check_equiv_with(
    ref_vals: &[f32],
    mt_vals: &[f32],
    tolerance: EquivTolerance,
) -> EquivResult {
    let n = ref_vals.len().min(mt_vals.len());
    let mut max_err = 0.0f32;
    let mut dot = 0.0f64;
    let mut ref_norm_sq = 0.0f64;
    let mut mt_norm_sq = 0.0f64;
    for (&r, &m) in ref_vals[..n].iter().zip(&mt_vals[..n]) {
        let err = (r - m).abs();
        if err > max_err {
            max_err = err;
        }
        let r = r as f64;
        let m = m as f64;
        dot += r * m;
        ref_norm_sq += r * r;
        mt_norm_sq += m * m;
    }

    let cosine_sim = match (ref_norm_sq > 0.0, mt_norm_sq > 0.0) {
        (false, false) => 1.0,
        (false, true) | (true, false) => 0.0,
        (true, true) => {
            let denom = ref_norm_sq.sqrt() * mt_norm_sq.sqrt();
            (dot / denom) as f32
        },
    }
    .clamp(-1.0, 1.0);

    let same_len = ref_vals.len() == mt_vals.len();
    EquivResult {
        n_checked: n,
        max_abs_err: max_err,
        cosine_sim,
        passed: same_len
            && max_err.is_finite()
            && cosine_sim.is_finite()
            && max_err <= tolerance.max_abs_err
            && cosine_sim >= tolerance.min_cosine_sim,
    }
}

/// Compare reference and MT output arrays element-wise.
/// `max_abs_err` is the maximum allowed absolute error; cosine similarity uses
/// the shared default floor to catch gross directional mismatches.
pub fn check_equiv(ref_vals: &[f32], mt_vals: &[f32], max_abs_err: f32) -> EquivResult {
    check_equiv_with(ref_vals, mt_vals, EquivTolerance::new(max_abs_err, DEFAULT_MIN_COSINE_SIM))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrectnessStatus {
    Passed { max_abs_err: f32, cosine_sim: f32 },
    Failed { max_abs_err: f32, cosine_sim: f32 },
    Unchecked,
    Unavailable,
}

#[derive(Debug, Clone, Copy)]
pub struct OpBench {
    op: &'static str,
    metric: &'static str,
}

impl OpBench {
    pub const fn new(op: &'static str, metric: &'static str) -> Self { Self { op, metric } }

    pub fn result(
        &self,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
    ) -> OpResult {
        let shape = shape.into();
        if mt_perf.is_some() && equiv.is_none() {
            panic!("implemented benchmark '{}' [{}] is missing correctness", self.op, shape);
        }
        let result = OpResult { op: self.op, shape, metric: self.metric, ref_perf, mt_perf, equiv };
        report_result(&result);
        result
    }

    pub fn implemented(
        &self,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: f64,
        equiv: EquivResult,
    ) -> OpResult {
        self.result(shape, ref_perf, Some(mt_perf), Some(equiv))
    }

    pub fn nyi(&self, shape: impl Into<String>, ref_perf: Option<f64>) -> OpResult {
        self.result(shape, ref_perf, None, None)
    }
}

pub struct OpResult {
    op: &'static str,
    shape: String,
    /// "GFLOPS" or "GB/s"
    metric: &'static str,
    /// Performance of the MLX Metal reference kernel.
    ref_perf: Option<f64>,
    /// Performance of MetalTile-generated kernel; None = not yet implemented.
    mt_perf: Option<f64>,
    /// Numerical equivalence check result.
    equiv: Option<EquivResult>,
}

impl OpResult {
    pub fn op(&self) -> &'static str { self.op }

    pub fn shape(&self) -> &str { &self.shape }

    pub fn metric(&self) -> &'static str { self.metric }

    pub fn ref_perf(&self) -> Option<f64> { self.ref_perf }

    pub fn mt_perf(&self) -> Option<f64> { self.mt_perf }

    pub fn equiv(&self) -> Option<&EquivResult> { self.equiv.as_ref() }

    pub fn pct(&self) -> Option<f64> {
        match (self.ref_perf, self.mt_perf) {
            (Some(r), Some(m)) if r > 0.0 => Some(m / r * 100.0),
            _ => None,
        }
    }

    pub fn correctness_status(&self) -> CorrectnessStatus {
        match (&self.equiv, self.mt_perf) {
            (Some(e), _) if e.passed =>
                CorrectnessStatus::Passed { max_abs_err: e.max_abs_err, cosine_sim: e.cosine_sim },
            (Some(e), _) =>
                CorrectnessStatus::Failed { max_abs_err: e.max_abs_err, cosine_sim: e.cosine_sim },
            (None, Some(_)) => CorrectnessStatus::Unchecked,
            (None, None) => CorrectnessStatus::Unavailable,
        }
    }

    pub fn correctness_cell(&self) -> String {
        match self.correctness_status() {
            CorrectnessStatus::Passed { max_abs_err, cosine_sim } => {
                format!("✓ cos={cosine_sim:.6} err={max_abs_err:.2e}")
            },
            CorrectnessStatus::Failed { max_abs_err, cosine_sim } => {
                format!("✗ cos={cosine_sim:.6} err={max_abs_err:.2e}")
            },
            CorrectnessStatus::Unchecked => "! missing-check".into(),
            CorrectnessStatus::Unavailable => "—".into(),
        }
    }

    pub fn is_unchecked(&self) -> bool {
        matches!(self.correctness_status(), CorrectnessStatus::Unchecked)
    }
}

pub fn validate_results(results: &[OpResult]) -> Result<(), String> {
    let unchecked: Vec<String> = results
        .iter()
        .filter(|r| r.is_unchecked())
        .map(|r| format!("{} [{}]", r.op(), r.shape()))
        .collect();
    if unchecked.is_empty() {
        Ok(())
    } else {
        Err(format!("implemented benchmarks missing correctness checks: {}", unchecked.join(", ")))
    }
}

pub fn print_suite(results: &[OpResult]) {
    validate_results(results).unwrap_or_else(|err| panic!("{err}"));

    let mut printer = SuitePrinter::new(
        results.iter().any(|r| !matches!(r.correctness_status(), CorrectnessStatus::Unavailable)),
    );
    printer.print_batch(results);
    printer.finish();
}

pub struct ResultReporterGuard {
    previous: Option<NonNull<dyn FnMut(&OpResult)>>,
}

impl Drop for ResultReporterGuard {
    fn drop(&mut self) {
        RESULT_REPORTER.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub fn set_result_reporter(reporter: &mut dyn FnMut(&OpResult)) -> ResultReporterGuard {
    // SAFETY: The guard restores the previous reporter on drop. The caller's &mut
    // borrow ensures the closure outlives the guard (Rust's borrow checker enforces this
    // at the call site). We erase the lifetime here to satisfy the 'static bound of
    // the thread-local, which is safe because the guard guarantees restoration before
    // the reference could become dangling.
    let reporter: NonNull<dyn FnMut(&OpResult)> =
        unsafe { std::mem::transmute(NonNull::from(reporter)) };
    let previous =
        RESULT_REPORTER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), Some(reporter)));
    ResultReporterGuard { previous }
}

pub struct SuitePrinter {
    show_correctness: bool,
    started: bool,
    last_op: Option<&'static str>,
    /// DRAM bandwidth ceiling for the current device (GB/s).  When set,
    /// any GB/s value that exceeds this ceiling is flagged with `~`.
    peak_gbps: Option<f64>,
    any_above_peak: bool,
}

impl SuitePrinter {
    pub fn new(show_correctness: bool) -> Self {
        Self {
            show_correctness,
            started: false,
            last_op: None,
            peak_gbps: None,
            any_above_peak: false,
        }
    }

    /// Set the DRAM bandwidth ceiling so values above it are flagged with `~`.
    pub fn with_peak_gbps(mut self, peak: f64) -> Self {
        self.peak_gbps = Some(peak);
        self
    }

    pub fn print_batch(&mut self, results: &[OpResult]) {
        if results.is_empty() {
            return;
        }
        if !self.started {
            self.print_header();
            self.started = true;
        }
        for result in results {
            if self.last_op.is_some() && self.last_op != Some(result.op()) {
                println!();
            }
            self.last_op = Some(result.op());
            // Track whether any perf value exceeds the DRAM ceiling.
            if let Some(peak) = self.peak_gbps {
                if result.metric() == "GB/s" {
                    let over = result.ref_perf().map_or(false, |v| v > peak)
                        || result.mt_perf().map_or(false, |v| v > peak);
                    if over {
                        self.any_above_peak = true;
                    }
                }
            }
            println!("{}", format_row(result, self.show_correctness, self.peak_gbps));
        }
        self.flush();
    }

    pub fn finish(&mut self) {
        if !self.started {
            return;
        }
        println!("  {}", separator(separator_width(self.show_correctness)));
        if self.any_above_peak {
            println!(
                "  {} GB/s exceeds DRAM peak — cache-resident working set or compute-bound; metric counts application bytes only",
                paint_stdout("~", Style::new().fg(Color::BrightBlack))
            );
        }
        self.flush();
    }

    fn print_header(&self) {
        println!();
        if self.show_correctness {
            println!(
                "  {} {} {}  {}  {}  {}",
                header_cell("Op", 28, false),
                header_cell("Shape", 18, false),
                header_cell("Reference", 13, true),
                header_cell("MetalTile", 13, true),
                header_cell("MT%", 6, true),
                header_cell("Correct", 28, true),
            );
        } else {
            println!(
                "  {} {} {}  {}  {}",
                header_cell("Op", 28, false),
                header_cell("Shape", 18, false),
                header_cell("Reference", 13, true),
                header_cell("MetalTile", 13, true),
                header_cell("MT%", 6, true),
            );
        }
        println!("  {}", separator(separator_width(self.show_correctness)));
    }

    fn flush(&self) { let _ = std::io::stdout().flush(); }
}

fn report_result(result: &OpResult) {
    RESULT_REPORTER.with(|slot| {
        if let Some(mut reporter) = *slot.borrow() {
            // Safety: the pointer is installed by `set_result_reporter` and restored by its
            // guard before the captured closure can go out of scope.
            unsafe {
                reporter.as_mut()(result);
            }
        }
    });
}

fn fmt_perf(v: Option<f64>, metric: &str, fallback: &str, peak_gbps: Option<f64>) -> String {
    match v {
        None => fallback.into(),
        Some(x) => {
            let flag =
                if metric == "GB/s" && peak_gbps.map_or(false, |p| x > p) { "~" } else { "" };
            format!("{flag}{x:.1} {metric}")
        },
    }
}

fn format_row(result: &OpResult, show_correctness: bool, peak_gbps: Option<f64>) -> String {
    let ref_s = fmt_perf(result.ref_perf(), result.metric(), "—", peak_gbps);
    let mt_s = fmt_perf(result.mt_perf(), result.metric(), "NYI", peak_gbps);
    let pct_s = result.pct().map(|p| format!("{:.0}%", p)).unwrap_or_else(|| "—".into());
    let ref_cell = style_reference(&ref_s, result.ref_perf());
    let mt_cell = style_metaltile(&mt_s, result);
    let pct_cell = style_pct(&pct_s, result);
    if show_correctness {
        let eq_s = result.correctness_cell();
        format!(
            "  {} {} {}  {}  {}  {}",
            paint_stdout(&pad_left(result.op(), 28), Style::new().fg(Color::Cyan).bold()),
            paint_stdout(&pad_left(result.shape(), 18), Style::new().fg(Color::BrightWhite)),
            ref_cell,
            mt_cell,
            pct_cell,
            style_correctness(&eq_s, result.correctness_status()),
        )
    } else {
        format!(
            "  {} {} {}  {}  {}",
            paint_stdout(&pad_left(result.op(), 28), Style::new().fg(Color::Cyan).bold()),
            paint_stdout(&pad_left(result.shape(), 18), Style::new().fg(Color::BrightWhite)),
            ref_cell,
            mt_cell,
            pct_cell,
        )
    }
}

fn separator_width(show_correctness: bool) -> usize { if show_correctness { 115 } else { 82 } }

fn header_cell(label: &str, width: usize, right_align: bool) -> String {
    let cell = if right_align { pad_right(label, width) } else { pad_left(label, width) };
    paint_stdout(&cell, Style::new().fg(Color::BrightWhite).bold())
}

fn separator(width: usize) -> String {
    paint_stdout(&"─".repeat(width), Style::new().fg(Color::BrightBlack).dim())
}

fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

fn style_reference(text: &str, value: Option<f64>) -> String {
    let style = if value.is_some() {
        Style::new().fg(Color::BrightWhite)
    } else {
        Style::new().fg(Color::Red).bold()
    };
    paint_stdout(&pad_right(text, 13), style)
}

fn style_metaltile(text: &str, result: &OpResult) -> String {
    let style = match (result.mt_perf(), result.correctness_status()) {
        (Some(_), CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(_), _) => Style::new().fg(Color::BrightWhite).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(&pad_right(text, 13), style)
}

fn style_pct(text: &str, result: &OpResult) -> String {
    let style = match (result.pct(), result.correctness_status()) {
        (_, CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(p), _) if p >= 90.0 => Style::new().fg(Color::Green).bold(),
        (Some(p), _) if p >= 60.0 => Style::new().fg(Color::Yellow).bold(),
        (Some(_), _) => Style::new().fg(Color::Red).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(&pad_right(text, 6), style)
}

fn style_correctness(text: &str, status: CorrectnessStatus) -> String {
    let style = match status {
        CorrectnessStatus::Passed { .. } => Style::new().fg(Color::Green).bold(),
        CorrectnessStatus::Failed { .. } => Style::new().fg(Color::Red).bold(),
        CorrectnessStatus::Unchecked => Style::new().fg(Color::Yellow).bold(),
        CorrectnessStatus::Unavailable => Style::new().fg(Color::BrightBlack).dim(),
    };
    paint_stdout(&pad_right(text, 28), style)
}

pub(crate) fn run_f32_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    runner.read_f32_slice(out, n)
}

pub(crate) fn run_typed_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
    dt: DType,
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    read_typed(runner, out, n, dt)
}

pub(crate) fn run_f16_once_as_f32(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    runner.read_f16_slice(out, n)
}

pub(crate) fn to_gflops(st: &crate::stats::BenchStats, flops: f64) -> Option<f64> {
    st.is_valid().then(|| flops / (st.mean_us * 1e-6) / 1e9)
}

pub(crate) fn to_gbps(st: &crate::stats::BenchStats, bytes: f64) -> Option<f64> {
    st.is_valid().then(|| bytes / (st.mean_us * 1e-6) / 1e9)
}

#[cfg(test)]
mod tests {
    use super::{CorrectnessStatus, EquivResult, OpBench, OpResult, check_equiv, validate_results};

    fn sample_result(mt_perf: Option<f64>, equiv: Option<EquivResult>) -> OpResult {
        OpBench::new("sample", "GB/s").result("shape", Some(1.0), mt_perf, equiv)
    }

    #[test]
    fn correctness_status_distinguishes_unchecked_from_unavailable() {
        let unchecked = OpResult {
            op: "sample",
            shape: "shape".into(),
            metric: "GB/s",
            ref_perf: Some(1.0),
            mt_perf: Some(2.0),
            equiv: None,
        };
        let unavailable = sample_result(None, None);
        assert_eq!(unchecked.correctness_status(), CorrectnessStatus::Unchecked);
        assert_eq!(unchecked.correctness_cell(), "! missing-check");
        assert!(unchecked.is_unchecked());
        assert_eq!(unavailable.correctness_status(), CorrectnessStatus::Unavailable);
        assert_eq!(unavailable.correctness_cell(), "—");
    }

    #[test]
    fn check_equiv_reports_cosine_similarity() {
        let equiv = check_equiv(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.001], 1e-2);
        assert_eq!(equiv.n_checked, 3);
        assert!(equiv.passed);
        assert!(equiv.cosine_sim > 0.999_999);
        assert!(equiv.max_abs_err > 0.0);
    }

    #[test]
    fn correctness_status_formats_checked_results() {
        let passed = sample_result(
            Some(2.0),
            Some(EquivResult { n_checked: 16, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }),
        );
        let failed = sample_result(
            Some(2.0),
            Some(EquivResult { n_checked: 16, max_abs_err: 1.5, cosine_sim: 0.5, passed: false }),
        );

        assert_eq!(passed.correctness_status(), CorrectnessStatus::Passed {
            max_abs_err: 0.0,
            cosine_sim: 1.0
        });
        assert_eq!(passed.correctness_cell(), "✓ cos=1.000000 err=0.00e0");
        assert_eq!(failed.correctness_status(), CorrectnessStatus::Failed {
            max_abs_err: 1.5,
            cosine_sim: 0.5
        });
        assert_eq!(failed.correctness_cell(), "✗ cos=0.500000 err=1.50e0");
    }

    #[test]
    #[should_panic(expected = "missing correctness")]
    fn op_bench_rejects_implemented_row_without_correctness() {
        let _ = OpBench::new("sample", "GB/s").result("shape", Some(1.0), Some(2.0), None);
    }

    #[test]
    fn validation_reports_unchecked_rows() {
        let unchecked = OpResult {
            op: "sample",
            shape: "shape".into(),
            metric: "GB/s",
            ref_perf: Some(1.0),
            mt_perf: Some(2.0),
            equiv: None,
        };
        let err = validate_results(&[unchecked]).expect_err("unchecked rows should fail");
        assert!(err.contains("sample [shape]"));
    }
}
