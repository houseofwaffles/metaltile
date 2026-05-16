use std::{cell::RefCell, io::Write, ptr::NonNull};

use metaltile_codegen::msl::MslGenerator;
pub use metaltile_core::dtype::DType;
use metaltile_core::ir::{Kernel, KernelMode};

use crate::term::{Color, Style, paint_stdout};

// ── Dtype variant helpers ─────────────────────────────────────────────────────

/// All floating-point dtypes to iterate over in multi-variant benches.
pub const FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];
/// Short names for the three floating-point dtypes, matching MLX convention.
pub const FLOAT_DTYPE_STRS: &[&str] = &["f32", "f16", "bf16"];
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

type ResultReporterFn = NonNull<dyn FnMut(&OpResult)>;

thread_local! {
    static RESULT_REPORTER: RefCell<Option<ResultReporterFn>> = RefCell::new(None);
}

pub const DEFAULT_MIN_COSINE_SIM: f32 = 0.999;

/// Result of a numerical equivalence check between the reference and MT kernels.
#[derive(Debug, Clone, Copy)]
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
    pub const fn op(&self) -> &'static str { self.op }

    pub fn result(
        &self,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
    ) -> OpResult {
        self.result_sub(None::<&str>, shape, ref_perf, mt_perf, equiv)
    }

    /// Like `result()` but with a sub-operation label displayed as "op (subop)".
    pub fn result_sub(
        &self,
        subop: Option<impl Into<String>>,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
    ) -> OpResult {
        let shape = shape.into();
        if mt_perf.is_some() && equiv.is_none() {
            panic!("implemented benchmark '{}' [{}] is missing correctness", self.op, shape);
        }
        let result = OpResult {
            op: self.op,
            subop: subop.map(|s| s.into()),
            shape,
            metric: self.metric,
            ref_perf,
            mt_perf,
            equiv,
        };
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
    /// Optional sub-operation displayed as "op (subop)" in the Op column.
    /// Does not affect blank-line grouping — that still uses `op`.
    subop: Option<String>,
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

    /// Rendered op name: "op (subop)" if subop is set, else "op".
    pub fn op_display(&self) -> String {
        match &self.subop {
            Some(s) => format!("{} ({})", self.op, s),
            None => self.op.to_string(),
        }
    }

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
            CorrectnessStatus::Passed { max_abs_err, .. } =>
                if max_abs_err < 1e-5 {
                    "✓".into()
                } else {
                    format!("✓ {max_abs_err:.2e}")
                },
            CorrectnessStatus::Failed { max_abs_err, cosine_sim } =>
                if cosine_sim < 0.999 {
                    format!("✗ {max_abs_err:.2e} cos={cosine_sim:.3}")
                } else {
                    format!("✗ {max_abs_err:.2e}")
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
    previous: Option<ResultReporterFn>,
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
    let previous = RESULT_REPORTER.with(|slot| (*slot.borrow_mut()).replace(reporter));
    ResultReporterGuard { previous }
}

pub struct SuitePrinter {
    show_correctness: bool,
    started: bool,
    last_op: Option<&'static str>,
    last_op_display: Option<String>,
    last_shape_base: Option<String>,
}

impl SuitePrinter {
    pub fn new(show_correctness: bool) -> Self {
        Self {
            show_correctness,
            started: false,
            last_op: None,
            last_op_display: None,
            last_shape_base: None,
        }
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
            // Blank line between top-level op groups.
            let new_group = self.last_op.is_some() && self.last_op != Some(result.op());
            if new_group {
                println!();
                self.last_shape_base = None;
            }
            self.last_op = Some(result.op());

            // Blank repeated op_display within a group; reset shape tracking on change.
            let op_display = result.op_display();
            let show_op = Some(&op_display) != self.last_op_display.as_ref();
            if show_op {
                self.last_shape_base = None;
            }
            self.last_op_display = Some(op_display.clone());
            let shown_op = if show_op { &op_display } else { "" };

            // Dim repeated shape base (e.g. "N=64M") when only the dtype changes.
            let (shape_cell, new_base) =
                build_shape_cell(result.shape(), self.last_shape_base.as_deref());
            self.last_shape_base = new_base;

            println!("{}", format_row(result, self.show_correctness, shown_op, &shape_cell));
        }
        self.flush();
    }

    pub fn finish(&mut self) {
        if !self.started {
            return;
        }
        println!("  {}", separator(separator_width(self.show_correctness)));
        self.flush();
    }

    fn print_header(&self) {
        println!();
        let sep = col_sep();
        if self.show_correctness {
            println!(
                "  {} {sep} {} {sep} {} {sep} {} {sep} {} {sep} {}",
                header_cell("Op", 28, false),
                header_cell("Shape", 26, false),
                header_cell("Reference", 14, true),
                header_cell("MetalTile", 14, true),
                header_cell("MT%", 6, true),
                header_cell("Correct", 14, true),
            );
        } else {
            println!(
                "  {} {sep} {} {sep} {} {sep} {} {sep} {}",
                header_cell("Op", 28, false),
                header_cell("Shape", 26, false),
                header_cell("Reference", 14, true),
                header_cell("MetalTile", 14, true),
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

fn fmt_perf(v: Option<f64>, metric: &str, fallback: &str) -> String {
    match v {
        None => fallback.into(),
        Some(x) => format!("{x:.1} {metric}"),
    }
}

/// Returns (painted_shape_cell, new_last_base).
/// When the shape ends with a known dtype token and the base matches `last_base`,
/// the base is dimmed and only the dtype is bright, reducing visual noise for
/// consecutive rows that differ only in dtype (e.g. "N=64M f32" / f16 / bf16).
fn build_shape_cell(shape: &str, last_base: Option<&str>) -> (String, Option<String>) {
    const DTYPES: &[&str] = &["bf16", "f16", "f32", "i32", "u32", "i8", "u8"];
    const W: usize = 26;
    for &dtype in DTYPES {
        if let Some(base) = shape.strip_suffix(&format!(" {dtype}")) {
            if Some(base) == last_base {
                // Dim the repeated base, highlight just the dtype.
                let prefix = format!("{base} ");
                let pad_w = W.saturating_sub(dtype.len());
                let cell = format!(
                    "{}{}",
                    paint_stdout(pad_left(&prefix, pad_w), Style::new().fg(Color::BrightBlack)),
                    paint_stdout(dtype, Style::new().fg(Color::BrightWhite)),
                );
                return (cell, Some(base.to_string()));
            } else {
                let cell = paint_stdout(pad_left(shape, W), Style::new().fg(Color::BrightWhite));
                return (cell, Some(base.to_string()));
            }
        }
    }
    (paint_stdout(pad_left(shape, W), Style::new().fg(Color::BrightWhite)), None)
}

fn format_row(
    result: &OpResult,
    show_correctness: bool,
    shown_op: &str,
    shape_cell: &str,
) -> String {
    let ref_s = fmt_perf(result.ref_perf(), result.metric(), "—");
    let mt_s = fmt_perf(result.mt_perf(), result.metric(), "NYI");
    let pct_s = result.pct().map(|p| format!("{:.0}%", p)).unwrap_or_else(|| "—".into());
    let ref_cell = style_reference(&ref_s, result.ref_perf());
    let mt_cell = style_metaltile(&mt_s, result);
    let pct_cell = style_pct(&pct_s, result);
    let sep = col_sep();
    let op_col = paint_stdout(pad_left(shown_op, 28), Style::new().fg(Color::Cyan).bold());
    if show_correctness {
        let eq_s = result.correctness_cell();
        format!(
            "  {} {sep} {} {sep} {} {sep} {} {sep} {} {sep} {}",
            op_col,
            shape_cell,
            ref_cell,
            mt_cell,
            pct_cell,
            style_correctness(&eq_s, result.correctness_status()),
        )
    } else {
        format!(
            "  {} {sep} {} {sep} {} {sep} {} {sep} {}",
            op_col, shape_cell, ref_cell, mt_cell, pct_cell,
        )
    }
}

// Visible widths: 2 prefix + 28 op + 3 sep + 26 shape + 3 sep + 14 ref + 3 sep + 14 mt
//                + 3 sep + 6 pct [+ 3 sep + 14 correct]
fn separator_width(show_correctness: bool) -> usize { if show_correctness { 119 } else { 99 } }

fn header_cell(label: &str, width: usize, right_align: bool) -> String {
    let cell = if right_align { pad_right(label, width) } else { pad_left(label, width) };
    paint_stdout(&cell, Style::new().fg(Color::BrightWhite).bold())
}

fn col_sep() -> String { paint_stdout("│", Style::new().fg(Color::BrightBlack).dim()) }

fn separator(width: usize) -> String {
    paint_stdout("─".repeat(width), Style::new().fg(Color::BrightBlack).dim())
}

fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

fn style_reference(text: &str, value: Option<f64>) -> String {
    let style = if value.is_some() {
        Style::new().fg(Color::BrightWhite)
    } else {
        Style::new().fg(Color::Red).bold()
    };
    paint_stdout(pad_right(text, 14), style)
}

fn style_metaltile(text: &str, result: &OpResult) -> String {
    let style = match (result.mt_perf(), result.correctness_status()) {
        (Some(_), CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(_), _) => Style::new().fg(Color::BrightWhite).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(pad_right(text, 14), style)
}

fn style_pct(text: &str, result: &OpResult) -> String {
    let style = match (result.pct(), result.correctness_status()) {
        (_, CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(p), _) if p >= 90.0 => Style::new().fg(Color::Green).bold(),
        (Some(p), _) if p >= 60.0 => Style::new().fg(Color::Yellow).bold(),
        (Some(_), _) => Style::new().fg(Color::Red).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(pad_right(text, 6), style)
}

fn style_correctness(text: &str, status: CorrectnessStatus) -> String {
    let style = match status {
        CorrectnessStatus::Passed { .. } => Style::new().fg(Color::Green).bold(),
        CorrectnessStatus::Failed { .. } => Style::new().fg(Color::Red).bold(),
        CorrectnessStatus::Unchecked => Style::new().fg(Color::Yellow).bold(),
        CorrectnessStatus::Unavailable => Style::new().fg(Color::BrightBlack).dim(),
    };
    paint_stdout(pad_right(text, 14), style)
}

// ── Shared bench abstractions ─────────────────────────────────────────────────

/// Generate MSL for an elementwise kernel IR produced by `make_ir`.
///
/// Uses default `KernelMode::Elementwise`. `label` is used only in the error message.
pub fn generate_elementwise_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    MslGenerator::default().generate(&make_ir()).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Generate MSL for a reduction kernel IR produced by `make_ir`, setting `Reduction` mode.
///
/// `label` is used only in the error message when code generation fails.
pub fn generate_reduction_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    let mut k = make_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Per-dtype context bundled at the top of every bench function.
pub struct DtypeCtx {
    pub dt: DType,
    /// MLX template-name suffix (e.g. `"float32"`).
    pub tn: &'static str,
    /// Short label used in shape strings (e.g. `"f32"`).
    pub label: &'static str,
    /// Bytes per element.
    pub eb: usize,
    /// Absolute-error tolerance for correctness checks.
    pub tol: f32,
}

impl DtypeCtx {
    /// Context for reduction ops — uses `dtype_tol_reduce`.
    pub fn reduce(dt: DType) -> Self {
        Self {
            dt,
            tn: mlx_tname(dt),
            label: dtype_label(dt),
            eb: elem_bytes(dt),
            tol: dtype_tol_reduce(dt),
        }
    }

    /// Context for elementwise ops — uses `dtype_tol`.
    pub fn elementwise(dt: DType) -> Self {
        Self {
            dt,
            tn: mlx_tname(dt),
            label: dtype_label(dt),
            eb: elem_bytes(dt),
            tol: dtype_tol(dt),
        }
    }
}

/// Emit the standard two-test block for a reduction op.
///
/// Generates:
/// - `msl_generates_for_all_dtypes` — calls `$msl_fn(dt)` for each float dtype
/// - `kernels_compile` (macos only) — compiles the generated MSL
///
/// Usage:
/// ```ignore
/// bench_tests!(msl_fn: layer_norm_msl_for, kernel_name: "mt_layer_norm");
/// ```
#[macro_export]
macro_rules! bench_tests {
    (msl_fn: $msl_fn:ident, kernel_name: $name:expr) => {
        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn msl_generates_for_all_dtypes() {
                for &dt in $crate::ops::FLOAT_DTYPES {
                    let msl = $msl_fn(dt);
                    assert!(!msl.trim().is_empty(), "MSL empty for {dt:?}");
                }
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn kernels_compile() {
                // NOTE: GpuRunner is not available in metaltile-std.
                // This test is only meaningful in metaltile-bench or metaltile-cli.
                // The MSL generation test above covers the pure path.
            }
        }
    };
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
            subop: None,
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
        assert_eq!(passed.correctness_cell(), "✓");
        assert_eq!(failed.correctness_status(), CorrectnessStatus::Failed {
            max_abs_err: 1.5,
            cosine_sim: 0.5
        });
        assert_eq!(failed.correctness_cell(), "✗ 1.50e0 cos=0.500");
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
            subop: None,
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
