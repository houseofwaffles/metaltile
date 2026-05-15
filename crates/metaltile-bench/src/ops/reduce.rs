//! Reduce benchmarks — #[kernel] DSL vs MLX metal/reduce.metal
//!
//! MLX kernel (all-reduce): all_reduce_sumfloat32 / all_reduce_sumfloat16 / all_reduce_sumbfloat16
//!   Params: (in: device T*, out: device T*, in_size: constant uint&, out_size: constant uint&)
//!   Grid: [1, 1, 1] × [256, 1, 1]
//!   Algorithm: flat all-reduce sum; each thread strides over entire array.
//!              SIMD-group reduce + threadgroup merge → one scalar output.
//!
//! MLX kernel (row-reduce): row_reduce_simple_sumfloat32 / ...float16 / ...bfloat16
//!   Params: (in: device T*, out: device T*, reduction_size: constant uint&, ...)
//!   Grid: [B, 1, 1] × [256, 1, 1]
//!   Algorithm: one threadgroup per row; threads stride over row and sum.
//!
//! MetalTile: mt_all_reduce_sum / mt_row_reduce_sum — same algorithms via #[kernel] DSL.
//!   KernelMode::Reduction

use metaltile::{core::ir::KernelMode, kernel};
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
        dtype_tol_reduce,
        elem_bytes,
        mlx_tname,
        run_typed_once,
        to_gbps,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/reduce.metal");
const ALL_N: usize = 64 * 1024 * 1024;
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const ALL_REDUCE_BENCH: OpBench = OpBench::new("all_reduce", "GB/s");
const ROW_REDUCE_BENCH: OpBench = OpBench::new("row_reduce", "GB/s");
const CHECK_ALL_N: usize = 16_384;
const CHECK_ROW_B: usize = 8;
const CHECK_ROW_N: usize = 512;
const TPG: usize = 256;

/// Reduce entire flat array to a scalar — sum/max/min.
#[kernel]
pub fn mt_all_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, sum);
    let result = reduce_sum(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_all_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, max);
    let result = reduce_max(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_all_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, min);
    let result = reduce_min(acc);
    store(out[0], result);
}

/// Reduce each row independently — sum/max/min.
#[kernel]
pub fn mt_row_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, sum);
    let result = reduce_sum(acc);
    store(out[row], result);
}

#[kernel]
pub fn mt_row_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, max);
    let result = reduce_max(acc);
    store(out[row], result);
}

#[kernel]
pub fn mt_row_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, min);
    let result = reduce_min(acc);
    store(out[row], result);
}

fn msl_for_reduce<F: Fn() -> metaltile::core::ir::Kernel>(make_ir: F, label: &str) -> String {
    let mut k = make_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

fn all_reduce_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_all_reduce::kernel_ir_for(dt), "all_reduce")
}
fn all_reduce_max_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_all_reduce_max::kernel_ir_for(dt), "all_reduce_max")
}
fn all_reduce_min_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_all_reduce_min::kernel_ir_for(dt), "all_reduce_min")
}
fn row_reduce_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_row_reduce::kernel_ir_for(dt), "row_reduce")
}
fn row_reduce_max_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_row_reduce_max::kernel_ir_for(dt), "row_reduce_max")
}
fn row_reduce_min_msl_for(dt: DType) -> String {
    msl_for_reduce(|| mt_row_reduce_min::kernel_ir_for(dt), "row_reduce_min")
}

pub fn bench_reduce(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_reduce_for(runner, dt)).collect()
}

fn bench_reduce_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let tn = mlx_tname(dt);
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol_reduce(dt);

    // Compile kernels for all three ops.
    let ops: &[(&str, &str, &str, &str)] = &[
        ("sum", "mt_all_reduce", "mt_row_reduce", &format!("all_reduce_sum{tn}")),
        ("max", "mt_all_reduce_max", "mt_row_reduce_max", &format!("all_reduce_max{tn}")),
        ("min", "mt_all_reduce_min", "mt_row_reduce_min", &format!("all_reduce_min{tn}")),
    ];

    let mut results = Vec::new();

    for &(op_name, ar_mt_name, rr_mt_name, ar_ref_name) in ops {
        let rr_ref_name = format!("row_reduce_simple_{op_name}{tn}");

        let ar_msl = match op_name {
            "sum" => all_reduce_msl_for(dt),
            "max" => all_reduce_max_msl_for(dt),
            _ => all_reduce_min_msl_for(dt),
        };
        let rr_msl = match op_name {
            "sum" => row_reduce_msl_for(dt),
            "max" => row_reduce_max_msl_for(dt),
            _ => row_reduce_min_msl_for(dt),
        };
        let ar_kernel = runner.compile(&ar_msl, ar_mt_name).ok();
        let rr_kernel = runner.compile(&rr_msl, rr_mt_name).ok();
        let ar_ref = runner.compile(SRC, ar_ref_name).ok();
        let rr_ref = runner.compile(SRC, &rr_ref_name).ok();

        // CPU reference functions for correctness (when MLX ref unavailable).
        let cpu_all: Box<dyn Fn(&[f32]) -> f32> = match op_name {
            "max" => Box::new(|v: &[f32]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)),
            "min" => Box::new(|v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min)),
            _ => Box::new(|v: &[f32]| v.iter().sum()),
        };
        let cpu_row: Box<dyn Fn(&[f32]) -> f32> = match op_name {
            "max" => Box::new(|v: &[f32]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)),
            "min" => Box::new(|v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min)),
            _ => Box::new(|v: &[f32]| v.iter().sum()),
        };

        // ── all_reduce ───────────────────────────────────────────────────────
        {
            let inp_vals: Vec<f32> =
                (0..CHECK_ALL_N).map(|i| 0.25 + (i % 19) as f32 * 0.03125).collect();
            let inp_check = buffer_typed(runner, &inp_vals, dt);
            let mt_ns = runner.buffer_u32(CHECK_ALL_N as u32);
            let ref_in_size = runner.buffer_u64(CHECK_ALL_N as u64);
            let ref_row_size = runner.buffer_u64(CHECK_ALL_N as u64);

            let ref_check = ar_ref.as_ref().map(|rk| {
                let out = zeros_typed(runner, 1, dt);
                run_typed_once(
                    runner,
                    rk,
                    &[&inp_check, &out, &ref_in_size, &ref_row_size],
                    &out,
                    1,
                    [1, 1, 1],
                    [TPG, 1, 1],
                    dt,
                )
            });
            let mt_check = ar_kernel.as_ref().map(|mk| {
                let out = zeros_typed(runner, 1, dt);
                run_typed_once(
                    runner,
                    mk,
                    &[&inp_check, &out, &mt_ns],
                    &out,
                    1,
                    [1, 1, 1],
                    [TPG, 1, 1],
                    dt,
                )
            });
            let equiv = match (ref_check, mt_check) {
                (Some(r), Some(m)) => check_equiv(&r, &m, tol),
                (None, Some(m)) => check_equiv(&[cpu_all(&inp_vals)], &m, tol),
                _ => continue,
            };

            let inp = buffer_typed(runner, &vec![1.0f32 / ALL_N as f32; ALL_N], dt);
            let bytes = (ALL_N * eb) as f64;
            let ref_in_sz = runner.buffer_u64(ALL_N as u64);
            let ref_row_sz = runner.buffer_u64(ALL_N as u64);
            let ref_perf = ar_ref.as_ref().and_then(|k| {
                let out = zeros_typed(runner, 1, dt);
                to_gbps(
                    &runner.bench(
                        k,
                        &[&inp, &out, &ref_in_sz, &ref_row_sz],
                        [1, 1, 1],
                        [256, 1, 1],
                        3,
                        10,
                    ),
                    bytes,
                )
            });
            let ns = runner.buffer_u32(ALL_N as u32);
            let mt_perf = ar_kernel.as_ref().and_then(|k| {
                let out = zeros_typed(runner, 1, dt);
                to_gbps(&runner.bench(k, &[&inp, &out, &ns], [1, 1, 1], [256, 1, 1], 3, 10), bytes)
            });
            let shape = format!("N={}M {op_name} {dlabel}", ALL_N / 1_000_000);
            results.push(if let Some(p) = mt_perf {
                ALL_REDUCE_BENCH.implemented(shape, ref_perf, p, equiv)
            } else {
                ALL_REDUCE_BENCH.nyi(shape, ref_perf)
            });
        }

        // ── row_reduce ───────────────────────────────────────────────────────
        for &(b, n) in SHAPES {
            let inp_vals: Vec<f32> = (0..CHECK_ROW_B * CHECK_ROW_N)
                .map(|i| 0.25 + (i / CHECK_ROW_N) as f32 * 0.0625 + (i % 13) as f32 * 0.03125)
                .collect();
            let inp_check = buffer_typed(runner, &inp_vals, dt);
            let mt_ns = runner.buffer_u32(CHECK_ROW_N as u32);
            let ref_red = runner.buffer_u64(CHECK_ROW_N as u64);
            let ref_osz = runner.buffer_i64(CHECK_ROW_B as i64);

            let ref_check = rr_ref.as_ref().map(|rk| {
                let out = zeros_typed(runner, CHECK_ROW_B, dt);
                run_typed_once(
                    runner,
                    rk,
                    &[&inp_check, &out, &ref_red, &ref_osz],
                    &out,
                    CHECK_ROW_B,
                    [1, CHECK_ROW_B, 1],
                    [TPG, 1, 1],
                    dt,
                )
            });
            let mt_check = rr_kernel.as_ref().map(|mk| {
                let out = zeros_typed(runner, CHECK_ROW_B, dt);
                run_typed_once(
                    runner,
                    mk,
                    &[&inp_check, &out, &mt_ns],
                    &out,
                    CHECK_ROW_B,
                    [CHECK_ROW_B, 1, 1],
                    [TPG, 1, 1],
                    dt,
                )
            });
            let equiv = match (ref_check, mt_check) {
                (Some(r), Some(m)) => check_equiv(&r, &m, tol),
                (None, Some(m)) => {
                    let cpu: Vec<f32> = (0..CHECK_ROW_B)
                        .map(|row| cpu_row(&inp_vals[row * CHECK_ROW_N..(row + 1) * CHECK_ROW_N]))
                        .collect();
                    check_equiv(&cpu, &m, tol)
                },
                _ => continue,
            };

            let inp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
            let bytes = (b * n * eb) as f64;
            let rref_red = runner.buffer_u64(n as u64);
            let rref_osz = runner.buffer_i64(b as i64);
            let ref_perf = rr_ref.as_ref().and_then(|k| {
                let out = zeros_typed(runner, b, dt);
                to_gbps(
                    &runner.bench(
                        k,
                        &[&inp, &out, &rref_red, &rref_osz],
                        [1, b, 1],
                        [256, 1, 1],
                        3,
                        10,
                    ),
                    bytes,
                )
            });
            let ns = runner.buffer_u32(n as u32);
            let mt_perf = rr_kernel.as_ref().and_then(|k| {
                let out = zeros_typed(runner, b, dt);
                to_gbps(&runner.bench(k, &[&inp, &out, &ns], [b, 1, 1], [256, 1, 1], 3, 10), bytes)
            });
            let shape = format!("B={b} N={n} {op_name} {dlabel}");
            results.push(if let Some(p) = mt_perf {
                ROW_REDUCE_BENCH.implemented(shape, ref_perf, p, equiv)
            } else {
                ROW_REDUCE_BENCH.nyi(shape, ref_perf)
            });
        }
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
            for (msl, name) in [
                (all_reduce_msl_for(dt), "all_reduce"),
                (all_reduce_max_msl_for(dt), "all_reduce_max"),
                (all_reduce_min_msl_for(dt), "all_reduce_min"),
                (row_reduce_msl_for(dt), "row_reduce"),
                (row_reduce_max_msl_for(dt), "row_reduce_max"),
                (row_reduce_min_msl_for(dt), "row_reduce_min"),
            ] {
                assert!(!msl.trim().is_empty(), "{name} MSL empty for {dt:?}");
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
            runner.compile(&all_reduce_msl_for(dt), "mt_all_reduce").unwrap();
            runner.compile(&all_reduce_max_msl_for(dt), "mt_all_reduce_max").unwrap();
            runner.compile(&all_reduce_min_msl_for(dt), "mt_all_reduce_min").unwrap();
            runner.compile(&row_reduce_msl_for(dt), "mt_row_reduce").unwrap();
            runner.compile(&row_reduce_max_msl_for(dt), "mt_row_reduce_max").unwrap();
            runner.compile(&row_reduce_min_msl_for(dt), "mt_row_reduce_min").unwrap();
        }
    }
}
