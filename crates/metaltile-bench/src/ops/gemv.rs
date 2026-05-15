//! GEMV benchmark — #[kernel] DSL vs MLX metal/gemv.metal
//!
//! MLX kernel: gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 (gemv.metal)
//!   Params: (mat, vec, bias, out, in_vec_size, out_vec_size, mat_ld,
//!            alpha, beta, batch_ndim, ...) — same layout for f32/f16/bf16
//!   Grid: [M/(BM*TM), 1, 1] × [BM*BN*32, 1, 1] = [M/16, 1, 1] × [128, 1, 1]
//!   Algorithm: y[row] = sum(A[row*K + i] * x[i]) for i in 0..K
//!
//! MetalTile: mt_gemv — per-row reduction via strided_reduce_dot, #[kernel] DSL.
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

static SRC: &str = include_str!("../metal/gemv.metal");

// BM=4, BN=1, SM=1, SN=32, TM=4, TN=4, nc=0, axpby=0
const REF_BM: usize = 4;
const REF_BN: usize = 1;
const REF_TM: usize = 4;

const BENCH: OpBench = OpBench::new("gemv", "GB/s");
const SHAPES: &[(usize, usize)] = &[(4096, 4096)]; // (M, K)
const TPG: usize = 256;

/// GEMV: y[row] = sum(mat[row*k + i] * vec[i]) for i in 0..k
/// One threadgroup per output row; threads cooperate via StrideReduce.
#[kernel]
pub fn mt_gemv<T>(mat: Tensor<T>, vec: Tensor<T>, out: Tensor<T>, #[constexpr] k: u32) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let acc = strided_reduce_dot(mat, vec, rs, rs, re);
    let result = reduce_sum(acc);
    store(out[row], result);
}

fn gemv_msl_for(dt: DType) -> String {
    let mut k = mt_gemv::kernel_ir_for(dt);
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[mt_gemv {dt:?}]: {e}");
        String::new()
    })
}

fn cpu_gemv(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for row in 0..m {
        let base = row * k;
        out[row] = (0..k).map(|col| mat[base + col] * vec[col]).sum();
    }
    out
}

pub fn bench_gemv(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_gemv_for(runner, dt)).collect()
}

fn bench_gemv_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol(dt).max(1e-2);

    let msl = gemv_msl_for(dt);
    let mk = runner.compile(&msl, "mt_gemv").ok();

    let mut results = Vec::new();
    for &(m, k) in SHAPES {
        let equiv = mk.as_ref().map(|mk| {
            let cm = 64usize;
            let ck = 256usize;
            let sm: Vec<f32> = (0..cm * ck).map(|i| (i % 16) as f32 * 0.01).collect();
            let sv: Vec<f32> = (0..ck).map(|i| (i % 8) as f32 * 0.01).collect();
            let sm_q = quantize_roundtrip(&sm, dt);
            let sv_q = quantize_roundtrip(&sv, dt);
            let ref_out = cpu_gemv(&sm_q, &sv_q, cm, ck);
            let mat_b = buffer_typed(runner, &sm, dt);
            let vec_b = buffer_typed(runner, &sv, dt);
            let out_b = zeros_typed(runner, cm, dt);
            let k_b = runner.buffer_u32(ck as u32);
            let mt_vals = run_typed_once(
                runner,
                mk,
                &[&mat_b, &vec_b, &out_b, &k_b],
                &out_b,
                cm,
                [cm, 1, 1],
                [TPG, 1, 1],
                dt,
            );
            check_equiv(&ref_out, &mt_vals, tol)
        });

        let mat_vals: Vec<f32> = (0..m * k).map(|i| (i % 16) as f32 * 0.01).collect();
        let vec_vals: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01).collect();
        let mat_buf = buffer_typed(runner, &mat_vals, dt);
        let vec_buf = buffer_typed(runner, &vec_vals, dt);
        let k_buf = runner.buffer_u32(k as u32);
        let bytes = (m * k * eb + k * eb + m * eb) as f64;

        let ref_name = format!(
            "gemv_{}_bm{REF_BM}_bn{REF_BN}_sm1_sn32_tm{REF_TM}_tn4_nc0_axpby0",
            mlx_tname(dt)
        );
        let rk = runner.compile(SRC, &ref_name).ok();
        let ref_perf = rk.as_ref().and_then(|rk| {
            let out_r = runner.buffer_zeros(m * eb);
            let bias_r = runner.buffer_zeros(m * eb);
            let zero_buf = runner.buffer_zeros(8); // empty batch arrays
            let in_vec_size = runner.buffer_i32(k as i32);
            let out_vec_size = runner.buffer_i32(m as i32);
            let mat_ld = runner.buffer_i32(k as i32);
            let alpha = runner.buffer_f32_scalar(1.0f32);
            let beta = runner.buffer_f32_scalar(0.0f32);
            let batch_ndim = runner.buffer_i32(0i32);
            let bias_stride = runner.buffer_i32(1i32);
            // tgs: [M/(BM*TM), 1, 1], tpg: [BM*BN*32, 1, 1]
            let ref_tgs = [m / (REF_BM * REF_TM), 1, 1];
            let ref_tpg = [REF_BM * REF_BN * 32, 1, 1];
            let st = runner.bench(
                rk,
                &[
                    &mat_buf,
                    &vec_buf,
                    &bias_r,
                    &out_r,
                    &in_vec_size,
                    &out_vec_size,
                    &mat_ld,
                    &alpha,
                    &beta,
                    &batch_ndim,
                    &zero_buf,
                    &zero_buf,
                    &zero_buf,
                    &zero_buf,
                    &bias_stride,
                ],
                ref_tgs,
                ref_tpg,
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let mt_perf = mk.as_ref().and_then(|mk| {
            let out_buf = zeros_typed(runner, m, dt);
            let st = runner.bench(
                mk,
                &[&mat_buf, &vec_buf, &out_buf, &k_buf],
                [m, 1, 1],
                [TPG, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes)
        });

        let shape = format!("M={m} K={k} {dlabel}");
        results.push(match mt_perf {
            Some(p) => BENCH.implemented(shape, ref_perf, p, equiv.expect("mk Some → equiv Some")),
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
            let msl = gemv_msl_for(dt);
            assert!(!msl.trim().is_empty(), "MSL empty for {dt:?}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernels_compile() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        for &dt in FLOAT_DTYPES {
            let msl = gemv_msl_for(dt);
            runner.compile(&msl, "mt_gemv").unwrap();
        }
    }
}
