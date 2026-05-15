//! Scan benchmarks — metal/scan.metal  (MLX, Apache-2.0)
//!
//! Inclusive prefix-sum (cumsum) over rows.
//!
//! MLX reference: `contig_scan_inclusive_sum_float32_float32`
//!   Params: (in, out, axis_size: constant size_t&)
//!   Grid: [1, rows, 1] × [256, 1, 1]
//!
//! MetalTile: `mt_scan_f32` — parallel SIMD two-phase scan, #[kernel] DSL.
//!   Grid: [1, rows, 1] × [256, 1, 1]
//!   N_READS = 1 element per thread per outer iteration.
//!   Phase 1: SIMD exclusive scan within warp; inclusive = exclusive + val.
//!   Phase 2: first warp exclusive-scans the n_simd warp totals.
//!   Phase 3: combine running_prefix + warp_excl + val_incl.
//!   KernelMode::Reduction
//!
//! Note: f32-only (the MLX reference is f32-only).  DSL codegen now supports
//! simd_prefix_exclusive_sum, so the scan kernel is pure DSL.

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{OpBench, OpResult, check_equiv, run_f32_once, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/scan.metal");

const BENCH: OpBench = OpBench::new("scan_f32", "GB/s");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)]; // (rows, cols)
const CHECK_ROWS: usize = 4;
const CHECK_N: usize = 256;
const TPG: usize = 256;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Parallel inclusive prefix-sum over a row using two-phase SIMD scan.
///
/// `sgs[9]`: slots 0..n_simd-1 hold per-warp exclusive prefixes after phase 3;
///            slot n_simd holds the running prefix across outer iterations.
///
/// Grid: [1, rows, 1] × [256, 1, 1]  (`program_id::<1>()` == row index)
#[kernel]
pub fn mt_scan_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let lane = simd_lane;
    let sg = simd_id;
    let ns = n_simd; // = lsize / 32
    let row_off = row * n;
    // 8 warp slots + 1 running-prefix slot (supports lsize=256, n_simd=8)
    threadgroup_alloc("sgs", 9);
    if lid == 0 {
        threadgroup_store("sgs", ns, 0);
    }
    threadgroup_barrier();
    // Read back 0 as float — used as the zero-pad value for OOB threads.
    let zero_f = threadgroup_load("sgs", ns);
    let n_iters = (n + lsize - 1) / lsize;
    for _r in range(0, n_iters, 1) {
        let pos = _r * lsize + lid;
        // OOB threads contribute 0 so they don't inflate the prefix.
        let val = select(pos < n, load(inp[row_off + pos]), zero_f);
        // Phase 1: SIMD exclusive scan; inclusive = exclusive + val.
        let excl = simd_scan_exclusive(val);
        let val_incl = excl + val;
        // Phase 2: lane 31 holds the warp's inclusive sum (== warp total).
        if lane == 31 {
            threadgroup_store("sgs", sg, val_incl);
        }
        threadgroup_barrier();
        // Phase 3: first warp exclusive-scans the n_simd warp totals.
        if sg == 0 {
            let wt = select(lane < ns, threadgroup_load("sgs", lane), zero_f);
            let wt_excl = simd_scan_exclusive(wt);
            if lane < ns {
                threadgroup_store("sgs", lane, wt_excl);
            }
        }
        threadgroup_barrier();
        // Phase 4: combine.
        let cur_prefix = threadgroup_load("sgs", ns);
        let warp_excl = threadgroup_load("sgs", sg);
        let out_val = cur_prefix + warp_excl + val_incl;
        if pos < n {
            store(out[row_off + pos], out_val);
        }
        // Phase 5: last thread updates running prefix for next iteration.
        threadgroup_barrier();
        let last = lsize - 1;
        if lid == last {
            threadgroup_store("sgs", ns, out_val);
        }
        threadgroup_barrier();
    }
}

fn scan_msl() -> String {
    let mut k = mt_scan_f32::kernel_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[scan]: {e}");
        String::new()
    })
}

// ── Bench ─────────────────────────────────────────────────────────────────────

fn cpu_cumsum(inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += inp[r * cols + c];
            out[r * cols + c] = acc;
        }
    }
    out
}

pub fn bench_scan(runner: &GpuRunner) -> Vec<OpResult> {
    let mt_msl = scan_msl();
    let mk = if mt_msl.is_empty() { None } else { runner.compile(&mt_msl, "mt_scan_f32").ok() };

    // MLX reference: contig_scan_inclusive_sum_float32_float32
    // Params: (in, out, axis_size: constant size_t [u64])
    let rk = runner.compile(SRC, "contig_scan_inclusive_sum_float32_float32").ok();

    let mut results = Vec::new();
    for &(rows, n) in SHAPES {
        let inp_vals: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();

        let equiv = mk.as_ref().map(|mk| {
            let ref_vals = cpu_cumsum(&inp_vals[..CHECK_ROWS * CHECK_N], CHECK_ROWS, CHECK_N);
            let inp_b = runner.buffer_f32(&inp_vals[..CHECK_ROWS * CHECK_N]);
            let out_b = runner.buffer_zeros(CHECK_ROWS * CHECK_N * 4);
            let ns = runner.buffer_u32(CHECK_N as u32);
            let mt_vals = run_f32_once(
                runner,
                mk,
                &[&inp_b, &out_b, &ns],
                &out_b,
                CHECK_ROWS * CHECK_N,
                [1, CHECK_ROWS, 1],
                [TPG, 1, 1],
            );
            check_equiv(&ref_vals, &mt_vals, 1e-3)
        });

        let inp_buf = runner.buffer_f32(&inp_vals);
        let bytes = (rows * n * 8) as f64; // read + write
        let ns_u64 = runner.buffer_u64(n as u64); // MLX uses size_t (u64)
        let ns_u32 = runner.buffer_u32(n as u32);

        let ref_out = runner.buffer_zeros(rows * n * 4);
        let ref_perf = rk.as_ref().and_then(|rk| {
            let st =
                runner.bench(rk, &[&inp_buf, &ref_out, &ns_u64], [1, rows, 1], [TPG, 1, 1], 3, 10);
            to_gbps(&st, bytes)
        });

        let mt_out = runner.buffer_zeros(rows * n * 4);
        let mt_perf = mk.as_ref().and_then(|mk| {
            let st =
                runner.bench(mk, &[&inp_buf, &mt_out, &ns_u32], [1, rows, 1], [TPG, 1, 1], 3, 10);
            to_gbps(&st, bytes)
        });

        let shape = format!("B={rows} N={n} f32");
        let result = if let Some(mt_perf) = mt_perf {
            BENCH.implemented(shape, ref_perf, mt_perf, equiv.expect("mk Some → equiv Some"))
        } else {
            BENCH.nyi(shape, ref_perf)
        };
        results.push(result);
    }
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_msl_generates() {
        let msl = scan_msl();
        assert!(!msl.trim().is_empty(), "scan MSL should not be empty");
        assert!(msl.contains("mt_scan_f32"), "kernel name missing");
        assert!(msl.contains("simd_prefix_exclusive_sum"), "SIMD scan missing");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn scan_kernel_compiles() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        let msl = scan_msl();
        runner
            .compile(&msl, "mt_scan_f32")
            .unwrap_or_else(|e| panic!("mt_scan_f32 compile error: {e}\nMSL:\n{msl}"));
    }
}
