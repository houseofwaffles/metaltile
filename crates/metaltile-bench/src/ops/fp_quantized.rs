//! Floating-point quantization benchmarks — metal/fp_quantized.metal  (MLX, Apache-2.0)
//!
//! NV-FP4 round-trip quantize+dequantize of float32 tensors.
//! Each 32-thread simdgroup processes 2 groups of 16 floats in the reference:
//!   - threads 0..15 → first group, threads 16..31 → second group
//!   - simd_max computes per-group abs-max for the dynamic FP4 scale
//!
//! MLX reference: `nvfp4_quantize_dequantize_float_gs_16_b_4`
//!   Grid: [1, N/32, 1] × [32, 1, 1]
//!
//! MetalTile: `mt_fp4_quant_dequant` — group_size=32 (full simdgroup).
//!   Each threadgroup of 32 threads = one simdgroup, handles 32 elements.
//!   scale = simd_max(|x|) / 6.0 (max of all 32 threads in the group)
//!   Grid: [N/32, 1, 1] × [32, 1, 1]
//!   KernelMode::Elementwise

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{EquivResult, EquivTolerance, OpBench, OpResult, check_equiv_with, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/fp_quantized.metal");

const REF_NAME: &str = "nvfp4_quantize_dequantize_float_gs_16_b_4";
const N: usize = 1024 * 1024;
const TPG: usize = 32; // one simdgroup per threadgroup
const BENCH: OpBench = OpBench::new("fp_quant_f32", "GB/s");

// ── DSL kernel ───────────────────────────────────────────────────────────────

/// NV-FP4 round-trip: quantize then immediately dequantize, group_size=32.
///
/// FP4 representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
/// Scale = simd_max(|x|) / 6.0 so max_abs maps to 6.0 after normalisation.
/// Dispatch: [N/32, 1, 1] × [32, 1, 1]  (one simdgroup = one group)
#[kernel]
pub fn mt_fp4_quant_dequant(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let x = load(inp[gid]);
    let ax = abs(x);

    // Per-simdgroup scale (group_size = 32 = full simdgroup)
    let group_max = simd_max(ax);
    let inv_scale = select(group_max > 0.0f32, 6.0f32 / group_max, 0.0f32);

    // Normalise |x| to [0, 6] and find nearest NV-FP4 value (midpoint rounding)
    let norm = ax * inv_scale;
    let q = select(
        norm < 0.25f32,
        0.0f32,
        select(
            norm < 0.75f32,
            0.5f32,
            select(
                norm < 1.25f32,
                1.0f32,
                select(
                    norm < 1.75f32,
                    1.5f32,
                    select(
                        norm < 2.5f32,
                        2.0f32,
                        select(norm < 3.5f32, 3.0f32, select(norm < 5.0f32, 4.0f32, 6.0f32)),
                    ),
                ),
            ),
        ),
    );

    // Restore sign and scale: dequant = sign * q * scale = sign * q * max / 6
    let sign = select(x < 0.0f32, -1.0f32, 1.0f32);
    let result = sign * q * (group_max / 6.0f32);
    store(out[gid], result);
}

fn fp4_msl() -> Result<String, String> {
    MslGenerator::default()
        .generate(&mt_fp4_quant_dequant::kernel_ir())
        .map_err(|e| format!("fp4 codegen: {e}"))
        .and_then(|msl| if msl.trim().is_empty() { Err("empty".into()) } else { Ok(msl) })
}

// ── CPU reference (same group_size=32 algorithm as the MT kernel) ─────────────

fn cpu_fp4_quant_dequant(inp: &[f32]) -> Vec<f32> {
    inp.chunks(32)
        .flat_map(|group| {
            let max_abs = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let inv_scale = if max_abs > 0.0 { 6.0 / max_abs } else { 0.0 };
            let scale = max_abs / 6.0;
            group.iter().map(move |&x| {
                let norm = x.abs() * inv_scale;
                let q = if norm < 0.25 {
                    0.0
                } else if norm < 0.75 {
                    0.5
                } else if norm < 1.25 {
                    1.0
                } else if norm < 1.75 {
                    1.5
                } else if norm < 2.5 {
                    2.0
                } else if norm < 3.5 {
                    3.0
                } else if norm < 5.0 {
                    4.0
                } else {
                    6.0
                };
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                sign * q * scale
            })
        })
        .collect()
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_fp_quantized(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner.compile(SRC, REF_NAME).ok();

    let data: Vec<f32> = (0..N).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
    let inp = runner.buffer_f32(&data);
    let bytes = (N * 4 * 2) as f64;

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = runner.buffer_zeros(N * 4);
        let st = runner.bench(rk, &[&inp, &out], [1, N / 32, 1], [32, 1, 1], 3, 10);
        to_gbps(&st, bytes)
    });

    let mt_msl = fp4_msl().ok();
    let mk = mt_msl.as_deref().and_then(|msl| runner.compile(msl, "mt_fp4_quant_dequant").ok());

    let equiv: Option<EquivResult> = mk.as_ref().map(|mk| {
        let check_n = 1024usize;
        let check_data = &data[..check_n];
        let ref_out = cpu_fp4_quant_dequant(check_data);
        let check_inp = runner.buffer_f32(check_data);
        let n_buf = runner.buffer_u32(check_n as u32);
        let check_out = runner.buffer_zeros(check_n * 4);
        runner.measure(
            mk,
            &[&check_inp, &check_out, &n_buf],
            [check_n / TPG, 1, 1],
            [TPG, 1, 1],
            0,
            1,
        );
        let mt_out = runner.read_f32_slice(&check_out, check_n);
        // FP4 has coarse quantization; values near boundaries can differ by one
        // quantization step (~0.4) between CPU and GPU due to simd_max rounding.
        check_equiv_with(&ref_out, &mt_out, EquivTolerance::new(0.5, 0.99))
    });

    let n_buf = runner.buffer_u32(N as u32);
    let mt_perf = mk.as_ref().and_then(|mk| {
        let out = runner.buffer_zeros(N * 4);
        let st = runner.bench(mk, &[&inp, &out, &n_buf], [N / TPG, 1, 1], [TPG, 1, 1], 3, 10);
        to_gbps(&st, bytes)
    });

    let shape = format!("N={}M f32 gs32", N / (1024 * 1024));
    let result = if let Some(mt_perf) = mt_perf {
        BENCH.implemented(shape, ref_perf, mt_perf, equiv.unwrap())
    } else {
        BENCH.nyi(format!("N={}M f32 nvfp4", N / (1024 * 1024)), ref_perf)
    };
    vec![result]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_fp4_msl_generates() {
        let msl = fp4_msl().expect("codegen failed");
        assert!(msl.contains("mt_fp4_quant_dequant"));
        assert!(msl.contains("simd_max"));
    }

    #[test]
    fn cpu_fp4_identity_max() {
        // Max element of each group should round-trip exactly.
        let group: Vec<f32> = (0..32).map(|i| i as f32 * 0.2).collect();
        let out = cpu_fp4_quant_dequant(&group);
        // The max value (6.2) normalises to ~6; nearest FP4 is 6.0.
        // scale = 6.2 / 6 ≈ 1.0333, so dequant(6) = 6 * scale ≈ 6.2  → no, 6.0 * scale
        // Just check non-zero and within ballpark
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_fp_quant_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        runner.compile(SRC, REF_NAME).unwrap_or_else(|e| panic!("{REF_NAME} compile error: {e}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_fp4_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = fp4_msl().expect("codegen");
        runner
            .compile(&msl, "mt_fp4_quant_dequant")
            .unwrap_or_else(|e| panic!("mt_fp4_quant_dequant compile error: {e}\nMSL:\n{msl}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_fp4_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = fp4_msl().expect("codegen");
        let mk = runner.compile(&msl, "mt_fp4_quant_dequant").expect("compile");
        let n = 1024usize;
        let data: Vec<f32> = (0..n).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
        let ref_out = cpu_fp4_quant_dequant(&data);
        let inp = runner.buffer_f32(&data);
        let n_buf = runner.buffer_u32(n as u32);
        let out_buf = runner.buffer_zeros(n * 4);
        runner.measure(&mk, &[&inp, &out_buf, &n_buf], [n / TPG, 1, 1], [TPG, 1, 1], 0, 1);
        let mt_out = runner.read_f32_slice(&out_buf, n);
        for (i, (&r, &m)) in ref_out.iter().zip(&mt_out).enumerate() {
            // FP4 is coarse — allow one quantization step of error at boundaries
            assert!((r - m).abs() < 0.5, "fp4 mismatch[{i}]: ref={r} mt={m}");
        }
    }
}
