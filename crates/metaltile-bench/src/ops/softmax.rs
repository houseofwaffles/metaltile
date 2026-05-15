//! Softmax benchmark — #[kernel] DSL vs MLX metal/softmax.metal
//!
//! MLX kernel: looped_softmax_{tname} (softmax.metal, line ~1201)
//!   Params: (inp: device T*, out: device T*, n: constant uint&) — slots [0, 1, 2]
//!   Grid: [B, 1, 1] × [256, 1, 1]  (one threadgroup per row)
//!   Algorithm: 2-pass online softmax with N_READS=4. Pass 1: single loop
//!              accumulates per-thread (max, sum-of-exps) via Welford merge,
//!              then simd+threadgroup reduce to get global (row_max, row_sum).
//!              Pass 2: write-back exp(xi - row_max) / row_sum.
//!
//! MetalTile: mt_softmax — 2-pass online softmax (N_READS=4, pure DSL) for f32/f16/bf16.
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
        quantize_roundtrip,
        run_typed_once,
        to_gbps,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/softmax.metal");
const BENCH: OpBench = OpBench::new("softmax", "GB/s");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const N_CHECK: usize = 256;
const B_CHECK: usize = 4;

/// Softmax: 2-pass online algorithm matching MLX looped_softmax.
///
/// Pass 1 (single loop): per-thread online (max, sum-of-exps) Welford merge.
///   N_READS=4 for full lsize*4 chunks + N_READS=1 remainder.
///   After loop: reduce_max → row_max, rescale per-thread sum, reduce_sum → row_sum.
/// Pass 2 (write-back): exp(xi - row_max) * inv_sum. N_READS=4 + remainder.
/// Reads inp twice (1 stats pass + write-back). Dispatch: [B, 1, 1] × [256, 1, 1].
#[kernel]
pub fn mt_softmax<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    // Pass 1: combined online (max, sum-of-exps) in a single N_READS=4 loop.
    let n_full = n / (lsize * 4u32);
    let mut local_m = neg_infinity();
    let mut local_s = 0.0f32;
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(inp[base]).cast::<f32>();
        let v1 = load(inp[base + 1u32]).cast::<f32>();
        let v2 = load(inp[base + 2u32]).cast::<f32>();
        let v3 = load(inp[base + 3u32]).cast::<f32>();
        let chunk_max = max(max(v0, v1), max(v2, v3));
        let new_m = max(local_m, chunk_max);
        let scale = exp(local_m - new_m);
        let e0 = exp(v0 - new_m);
        let e1 = exp(v1 - new_m);
        let e2 = exp(v2 - new_m);
        let e3 = exp(v3 - new_m);
        local_s = local_s * scale + e0 + e1 + e2 + e3;
        local_m = new_m;
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let xi = load(inp[_i]).cast::<f32>();
        let new_m = max(local_m, xi);
        local_s = local_s * exp(local_m - new_m) + exp(xi - new_m);
        local_m = new_m;
    }
    // Two-step global reduction: max then rescaled sum.
    let row_max = reduce_max(local_m);
    let rescaled = local_s * exp(local_m - row_max);
    let row_sum = reduce_sum(rescaled);
    let inv_sum = recip(row_sum);
    // Pass 2: write-back N_READS=4 + remainder.
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let f0 = exp(load(inp[base]).cast::<f32>() - row_max) * inv_sum;
        let f1 = exp(load(inp[base + 1u32]).cast::<f32>() - row_max) * inv_sum;
        let f2 = exp(load(inp[base + 2u32]).cast::<f32>() - row_max) * inv_sum;
        let f3 = exp(load(inp[base + 3u32]).cast::<f32>() - row_max) * inv_sum;
        store(out[base], f0.cast::<T>());
        store(out[base + 1u32], f1.cast::<T>());
        store(out[base + 2u32], f2.cast::<T>());
        store(out[base + 3u32], f3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let fi = exp(load(inp[_i]).cast::<f32>() - row_max) * inv_sum;
        store(out[_i], fi.cast::<T>());
    }
}

fn softmax_msl_for(dt: DType) -> Result<String, String> {
    let mut k = mt_softmax::kernel_ir_for(dt);
    k.mode = KernelMode::Reduction;
    MslGenerator::default()
        .generate(&k)
        .map_err(|e| format!("softmax codegen failed: {e}"))
        .and_then(|msl| {
            if msl.trim().is_empty() {
                Err("softmax codegen returned empty MSL".into())
            } else {
                Ok(msl)
            }
        })
}

fn cpu_softmax(inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let slice = &inp[base..base + cols];
        let row_max = slice.iter().copied().fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
        let sum: f32 = slice.iter().map(|&x| (x - row_max).exp()).sum();
        for (col, &x) in slice.iter().enumerate() {
            out[base + col] = (x - row_max).exp() / sum;
        }
    }
    out
}

pub fn bench_softmax_f32(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_softmax_for(runner, dt)).collect()
}

fn bench_softmax_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let tn = mlx_tname(dt);
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol_reduce(dt);

    let msl = match softmax_msl_for(dt) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[softmax {dlabel}]: {e}");
            return vec![];
        },
    };
    let rk = runner.compile(SRC, &format!("looped_softmax_{tn}")).ok();
    let mk = match runner.compile(&msl, "mt_softmax") {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[softmax {dlabel}] compile: {e}");
            return vec![];
        },
    };

    let equiv = {
        let inp_vals: Vec<f32> =
            (0..B_CHECK * N_CHECK).map(|i| (i % 32) as f32 * 0.1 - 1.5).collect();
        let inp_q = quantize_roundtrip(&inp_vals, dt);
        let ref_vals = cpu_softmax(&inp_q, B_CHECK, N_CHECK);
        let inp = buffer_typed(runner, &inp_vals, dt);
        let ns = runner.buffer_u32(N_CHECK as u32);
        let mt_out = zeros_typed(runner, B_CHECK * N_CHECK, dt);
        let mt_vals = run_typed_once(
            runner,
            &mk,
            &[&inp, &mt_out, &ns],
            &mt_out,
            B_CHECK * N_CHECK,
            [B_CHECK, 1, 1],
            [256, 1, 1],
            dt,
        );
        check_equiv(&ref_vals, &mt_vals, tol)
    };

    let mut results = Vec::new();
    for &(b, n) in SHAPES {
        let inp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
        let ns = runner.buffer_u32(n as u32);
        let bytes = (b * n * eb * 2) as f64;
        let ref_perf = rk.as_ref().and_then(|r| {
            let out = zeros_typed(runner, b * n, dt);
            let st = runner.bench(r, &[&inp, &out, &ns], [b, 1, 1], [256, 1, 1], 3, 10);
            to_gbps(&st, bytes)
        });
        let mt_out = zeros_typed(runner, b * n, dt);
        let mt_perf = {
            let st = runner.bench(&mk, &[&inp, &mt_out, &ns], [b, 1, 1], [256, 1, 1], 3, 10);
            to_gbps(&st, bytes)
        };
        let shape = format!("B={b} N={n} {dlabel}");
        results.push(match mt_perf {
            Some(p) => BENCH.implemented(shape, ref_perf, p, equiv.clone()),
            None => BENCH.nyi(shape, ref_perf),
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let msl = softmax_msl_for(dt).expect("softmax MSL gen");
            assert!(!msl.trim().is_empty());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernels_compile() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        for &dt in FLOAT_DTYPES {
            let msl = softmax_msl_for(dt).expect("softmax MSL gen");
            runner.compile(&msl, "mt_softmax").unwrap();
        }
    }
}
