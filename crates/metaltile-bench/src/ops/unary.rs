//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal
//!
//! MLX kernel: v_{Op}{tname}{tname} (unary.metal)
//!   e.g. v_Expfloat32float32, v_Logfloat16float16, v_Sqrtbfloat16bfloat16
//!   Params: (in: device T*, out: device T*, size: constant uint&) — slots [0, 1, 2]
//!   Grid: [ceil(N/TPG), 1, 1] × [256, 1, 1]  (one thread per element)
//!   Algorithm: out[i] = op(in[i])  (elementwise unary)
//!
//! MetalTile: mt_{op} — same elementwise algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{
        DType,
        FLOAT_DTYPES,
        OpBench,
        OpResult,
        buffer_typed,
        check_equiv,
        dtype_label,
        dtype_tol,
        elem_bytes,
        mlx_tname,
        quantize_roundtrip,
        run_typed_once,
        to_gbps,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/unary.metal");

const BENCH: OpBench = OpBench::new("unary", "GB/s");
pub const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 2_048;
const TPG: usize = 256;

#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}
#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}
#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}
#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}
#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}
#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}
#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}
// ── New ops with DSL builtins ─────────────────────────────────────────────
#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}
#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}
#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}
#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}
#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}
#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}
#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}
#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}
#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}
#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}
#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}
// ── Composable ops ────────────────────────────────────────────────────────
// square: out = x * x
#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}
// sigmoid: out = 1 / (1 + exp(-x))
#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], 1.0f32.cast::<T>() / (1.0f32.cast::<T>() + exp(-x)));
}
// log1p: out = log(1 + x)
#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

struct UnaryEntry {
    name: &'static str,
    msl: String,
    /// MLX reference kernel name, None for ops without a direct reference (activations).
    ref_fn: Option<String>,
    cpu: fn(f32) -> f32,
    check_input: fn(usize) -> f32,
    tolerance: f32,
}

fn signed_check_input(i: usize) -> f32 {
    match i % 8 {
        0 => -3.0,
        1 => -1.5,
        2 => -0.5,
        3 => 0.0,
        4 => 0.25,
        5 => 0.75,
        6 => 1.5,
        _ => 3.0,
    }
}

fn positive_check_input(i: usize) -> f32 { 0.25 + (i % 16) as f32 * 0.25 }

fn cpu_exp(x: f32) -> f32 { x.exp() }
fn cpu_log(x: f32) -> f32 { x.ln() }
fn cpu_sqrt(x: f32) -> f32 { x.sqrt() }
fn cpu_rsqrt(x: f32) -> f32 { x.sqrt().recip() }
fn cpu_abs(x: f32) -> f32 { x.abs() }
fn cpu_silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn cpu_gelu(x: f32) -> f32 {
    let k = 0.797_884_6_f32;
    0.5 * x * (1.0 + (k * (x + 0.044_715 * x * x * x)).tanh())
}
fn cpu_relu(x: f32) -> f32 { x.max(0.0) }
fn cpu_cos(x: f32) -> f32 { x.cos() }
fn cpu_sin(x: f32) -> f32 { x.sin() }
fn cpu_ceil(x: f32) -> f32 { x.ceil() }
fn cpu_floor(x: f32) -> f32 { x.floor() }
fn cpu_exp2(x: f32) -> f32 { x.exp2() }
fn cpu_log2(x: f32) -> f32 { x.log2() }
fn cpu_neg(x: f32) -> f32 { -x }
fn cpu_recip(x: f32) -> f32 { x.recip() }
fn cpu_erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 (max err < 1.5e-7)
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}
fn cpu_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
fn cpu_round(x: f32) -> f32 { x.round() }
fn cpu_square(x: f32) -> f32 { x * x }
fn cpu_sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn cpu_log1p(x: f32) -> f32 { x.ln_1p() }

/// Ops with MLX reference kernels: (op_name, mlx_op_tag, cpu_fn, check_fn)
/// Activations have no MLX ref (tagged None).
type UnarySpec = (
    &'static str,
    Option<&'static str>,
    fn(f32) -> f32,
    fn(usize) -> f32,
    f32, // f32 tolerance (tightened per dtype by dtype_tol)
);

const UNARY_SPECS: &[UnarySpec] = &[
    ("exp", Some("Exp"), cpu_exp, signed_check_input, 1e-4),
    ("log", Some("Log"), cpu_log, positive_check_input, 1e-4),
    ("sqrt", Some("Sqrt"), cpu_sqrt, positive_check_input, 1e-4),
    ("rsqrt", Some("Rsqrt"), cpu_rsqrt, positive_check_input, 1e-4),
    ("abs", Some("Abs"), cpu_abs, signed_check_input, 1e-6),
    ("silu", None, cpu_silu, signed_check_input, 1e-4),
    ("gelu", None, cpu_gelu, signed_check_input, 1e-4),
    ("relu", None, cpu_relu, signed_check_input, 1e-6),
    // ── New ops ──────────────────────────────────────────────────────────
    ("cos", Some("Cos"), cpu_cos, signed_check_input, 1e-4),
    ("sin", Some("Sin"), cpu_sin, signed_check_input, 1e-4),
    ("ceil", Some("Ceil"), cpu_ceil, signed_check_input, 1e-6),
    ("floor", Some("Floor"), cpu_floor, signed_check_input, 1e-6),
    ("erf", Some("Erf"), cpu_erf, signed_check_input, 1e-3),
    ("exp2", None, cpu_exp2, signed_check_input, 1e-4), // MLX has no standalone exp2 kernel
    ("log2", Some("Log2"), cpu_log2, positive_check_input, 1e-4),
    ("sign", Some("Sign"), cpu_sign, signed_check_input, 0.0),
    ("round", Some("Round"), cpu_round, signed_check_input, 0.0),
    ("neg", Some("Negative"), cpu_neg, signed_check_input, 1e-6),
    ("recip", None, cpu_recip, positive_check_input, 1e-4),
    // ── Composable ───────────────────────────────────────────────────────
    ("square", Some("Square"), cpu_square, signed_check_input, 1e-4),
    ("sigmoid", Some("Sigmoid"), cpu_sigmoid, signed_check_input, 1e-4),
    ("log1p", Some("Log1p"), cpu_log1p, positive_check_input, 1e-4),
];

fn make_unary_entries(dt: DType) -> Vec<UnaryEntry> {
    let tn = mlx_tname(dt);
    UNARY_SPECS
        .iter()
        .filter_map(|&(name, mlx_op, cpu, check_input, _base_tol)| {
            // kernel name in DSL: mt_{name}
            let kernel_fn = match name {
                "exp" => mt_exp::kernel_ir_for(dt),
                "log" => mt_log::kernel_ir_for(dt),
                "sqrt" => mt_sqrt::kernel_ir_for(dt),
                "rsqrt" => mt_rsqrt::kernel_ir_for(dt),
                "abs" => mt_abs::kernel_ir_for(dt),
                "silu" => mt_silu::kernel_ir_for(dt),
                "gelu" => mt_gelu::kernel_ir_for(dt),
                "relu" => mt_relu::kernel_ir_for(dt),
                "cos" => mt_cos::kernel_ir_for(dt),
                "sin" => mt_sin::kernel_ir_for(dt),
                "ceil" => mt_ceil::kernel_ir_for(dt),
                "floor" => mt_floor::kernel_ir_for(dt),
                "erf" => mt_erf::kernel_ir_for(dt),
                "exp2" => mt_exp2::kernel_ir_for(dt),
                "log2" => mt_log2::kernel_ir_for(dt),
                "sign" => mt_sign::kernel_ir_for(dt),
                "round" => mt_round::kernel_ir_for(dt),
                "neg" => mt_neg::kernel_ir_for(dt),
                "recip" => mt_recip::kernel_ir_for(dt),
                "square" => mt_square::kernel_ir_for(dt),
                "sigmoid" => mt_sigmoid::kernel_ir_for(dt),
                "log1p" => mt_log1p::kernel_ir_for(dt),
                _ => return None,
            };
            let msl = MslGenerator::default().generate(&kernel_fn).ok()?;
            // exp(3.0) ≈ 20 in bf16 has ULP ≈ 0.125, so errors up to ~0.15 are expected.
            // log1p(-0.5) ≈ -0.69 in bf16 has ULP ≈ 0.007; exp is the main concern.
            let effective_base_tol = match (name, dt) {
                ("exp", DType::BF16) | ("exp2", DType::BF16) | ("sigmoid", DType::BF16) =>
                    _base_tol.max(0.15),
                _ => _base_tol,
            };
            Some(UnaryEntry {
                name,
                msl,
                ref_fn: mlx_op.map(|op| format!("v_{op}{tn}{tn}")),
                cpu,
                check_input,
                tolerance: dtype_tol(dt).max(effective_base_tol),
            })
        })
        .collect()
}

pub fn bench_all_unary(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_unary_for(runner, dt)).collect()
}

fn bench_unary_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let entries = make_unary_entries(dt);
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);

    let mut results = Vec::new();
    let inp_bench = buffer_typed(runner, &vec![0.5f32; N_ELEM], dt);
    let bytes = (N_ELEM * eb * 2) as f64; // 1 read + 1 write
    let tgs = [N_ELEM.div_ceil(TPG), 1, 1];
    let tpg = [TPG, 1, 1];

    for entry in &entries {
        let Some(mk) = runner.compile(&entry.msl, &format!("mt_{}", entry.name)).ok() else {
            continue;
        };

        // --- correctness: compare MT vs CPU reference ---
        let check_in_vals: Vec<f32> = (0..N_CHECK).map(entry.check_input).collect();
        let check_in_q = quantize_roundtrip(&check_in_vals, dt);
        let cpu_ref: Vec<f32> = check_in_q.iter().copied().map(entry.cpu).collect();
        let check_in = buffer_typed(runner, &check_in_vals, dt);
        let check_out = zeros_typed(runner, N_CHECK, dt);
        let mt_check = run_typed_once(
            runner,
            &mk,
            &[&check_in, &check_out],
            &check_out,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(&cpu_ref, &mt_check, entry.tolerance);

        // --- performance: MT ---
        let mt_out = zeros_typed(runner, N_ELEM, dt);
        let mt_perf = to_gbps(&runner.bench(&mk, &[&inp_bench, &mt_out], tgs, tpg, 3, 10), bytes);

        // --- performance: MLX reference (if available) ---
        let ref_perf = entry.ref_fn.as_ref().and_then(|fn_name| {
            let rk = runner.compile(SRC, fn_name).ok()?;
            let ref_out = zeros_typed(runner, N_ELEM, dt);
            let ref_size = runner.buffer_u32(N_ELEM as u32);
            let st = runner.bench(&rk, &[&inp_bench, &ref_out, &ref_size], tgs, tpg, 3, 10);
            to_gbps(&st, bytes)
        });

        let shape = format!("N={N_ELEM} {} {dlabel}", entry.name);
        results.push(match mt_perf {
            Some(p) => BENCH.implemented(shape, ref_perf, p, equiv),
            None => BENCH.nyi(shape, ref_perf),
        });
    }
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let entries = make_unary_entries(dt);
            assert!(!entries.is_empty(), "no unary entries for {dt:?}");
            for entry in &entries {
                assert!(!entry.msl.trim().is_empty(), "MSL empty for {} {dt:?}", entry.name);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernels_compile() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        for &dt in FLOAT_DTYPES {
            let entries = make_unary_entries(dt);
            for entry in &entries {
                runner.compile(&entry.msl, &format!("mt_{}", entry.name)).unwrap();
            }
        }
    }
}
