//! Quantized benchmarks — metal/quantized.metal  (MLX, Apache-2.0)
//!
//! Affine int4 quantized matrix-vector multiply.
//! group_size=64, bits=4, f32 scales/biases/x/y.
//!
//! MLX reference: `affine_qmv_fast_float16_t_gs_64_b_4_batch_0`
//!   Grid: [1, M/8, 1] × [64, 1, 1]  (f16, num_simdgroups=2, results/sg=4)
//!
//! MetalTile: `mt_qmv_f32` — one threadgroup per output row, 64 threads (one per group).
//!   Dequant: w_f = int4_val * scale + bias  (int4_val ∈ 0..15)
//!   Grid: [M, 1, 1] × [64, 1, 1]
//!   KernelMode::Reduction

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{EquivResult, OpBench, OpResult, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/quantized.metal");

const REF_NAME: &str = "affine_qmv_fast_float16_t_gs_64_b_4_batch_0";
const GROUP_SIZE: usize = 64;
const SHAPES: &[(usize, usize)] = &[(4096, 4096)];
// MLX reference: num_simdgroups=2 × results_per_simdgroup=4 = 8 rows/TG
const ROWS_PER_TG: usize = 8;
const TPG: usize = GROUP_SIZE; // one thread per group

const BENCH: OpBench = OpBench::new("qmv_f32_gs64_b4", "GB/s");

// ── DSL kernel ────────────────────────────────────────────────────────────────

/// Int4 quantized matrix-vector multiply: y[row] = Σ (scale[g] * int4(w) + bias[g]) * x[i]
///
/// Weights packed as uint32: 8 × int4 per u32, row-major.
/// Group size = 64 → 8 u32 packs per group, gs_per_row groups per row.
/// Each of the 64 threads handles one group; `reduce_sum` sums 64 partial results.
///
/// Dispatch: [M, 1, 1] × [64, 1, 1]
#[kernel]
pub fn mt_qmv_f32(
    w: Tensor<u32>,
    scales: Tensor<f32>,
    biases: Tensor<f32>,
    x: Tensor<f32>,
    out: Tensor<f32>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    let row = program_id::<0>();
    let packs_per_row = k / 8u32;
    let w_base = row * packs_per_row;
    let sb_base = row * gs_per_row;

    let mut acc = 0.0f32;

    // Stride over groups: with lsize=64 and gs_per_row=64, each thread handles exactly one.
    for _g in range(tid, gs_per_row, lsize) {
        let s = load(scales[sb_base + _g]);
        let bias = load(biases[sb_base + _g]);
        let g_w_base = w_base + _g * 8u32; // 8 u32 packs per group (GROUP_SIZE=64 / 8)
        let g_x_base = _g * 64u32;

        // Process 8 u32 packs per group (= 64 int4 elements)
        for _p in range(0u32, 8u32, 1u32) {
            let packed = load(w[g_w_base + _p]);
            let xb = g_x_base + _p * 8u32;

            // Unpack 8 int4 values from one u32 (4 bits each, LSB-first)
            for _b in range(0u32, 8u32, 1u32) {
                let shift = _b * 4u32;
                let int4_val = (packed >> shift) & 15u32;
                let xi = load(x[xb + _b]);
                acc = acc + (s * (int4_val * 1.0f32) + bias) * xi;
            }
        }
    }

    let result = reduce_sum(acc);
    store(out[row], result);
}

fn qmv_msl() -> Result<String, String> {
    let mut k = mt_qmv_f32::kernel_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default()
        .generate(&k)
        .map_err(|e| format!("qmv codegen: {e}"))
        .and_then(|msl| if msl.trim().is_empty() { Err("empty".into()) } else { Ok(msl) })
}

// ── CPU reference ──────────────────────────────────────────────────────────────

fn cpu_qmv(w: &[u32], scales: &[f32], biases: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let gs = GROUP_SIZE;
    let gs_per_row = k / gs;
    let packs_per_row = k / 8;
    let mut out = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f32;
        for g in 0..gs_per_row {
            let s = scales[row * gs_per_row + g];
            let b = biases[row * gs_per_row + g];
            for p in 0..8 {
                let packed = w[row * packs_per_row + g * 8 + p];
                for bit in 0..8u32 {
                    let int4_val = ((packed >> (bit * 4)) & 0xF) as f32;
                    let xi = x[g * gs + p * 8 + bit as usize];
                    acc += (s * int4_val + b) * xi;
                }
            }
        }
        out[row] = acc;
    }
    out
}

// ── Bench ──────────────────────────────────────────────────────────────────────

pub fn bench_quantized(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner.compile(SRC, REF_NAME).ok();

    let mt_msl = qmv_msl().ok();
    let mk = mt_msl.as_deref().and_then(|msl| runner.compile(msl, "mt_qmv_f32").ok());

    let mut results = Vec::new();
    for &(m, k) in SHAPES {
        let w_elems = m * k / 8;
        let sb_elems = m * k / GROUP_SIZE;

        // ── Reference (f16) data ──────────────────────────────────────────────
        let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
        let scale_f16: Vec<u8> =
            (0..sb_elems * 2).map(|i| if i % 2 == 0 { 0x66 } else { 0x2E }).collect();
        let bias_f16: Vec<u8> = vec![0u8; sb_elems * 2];
        let x_f16: Vec<u8> = (0..k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();

        let w_buf = runner.buffer_bytes(&w_data);
        let scales_f16_buf = runner.buffer_bytes(&scale_f16);
        let biases_f16_buf = runner.buffer_bytes(&bias_f16);
        let x_f16_buf = runner.buffer_bytes(&x_f16);
        let in_size = runner.buffer_i32(k as i32);
        let out_size = runner.buffer_i32(m as i32);
        let batch_zero = runner.buffer_i32(0i32);
        let zero = runner.buffer_zeros(8);

        // f16 reference byte count
        let bytes_f16 = (m * k / 2 + sb_elems * 2 * 2 + k * 2 + m * 2) as f64;

        let ref_perf = rk.as_ref().and_then(|rk| {
            let y_buf = runner.buffer_zeros(m * 2);
            let st = runner.bench(
                rk,
                &[
                    &w_buf,
                    &scales_f16_buf,
                    &biases_f16_buf,
                    &x_f16_buf,
                    &y_buf,
                    &in_size,
                    &out_size,
                    &batch_zero,
                    &zero,
                    &zero,
                    &batch_zero,
                    &zero,
                    &zero,
                    &zero,
                    &zero,
                ],
                [1, m / ROWS_PER_TG, 1],
                [64, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes_f16)
        });

        // ── MT (f32) data ──────────────────────────────────────────────────────
        // Re-use same packed u32 weights (w_data is already raw bytes of u32s)
        let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
        let biases_f32: Vec<f32> = vec![0.0f32; sb_elems];
        let x_f32: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();

        let equiv: Option<EquivResult> = mk.as_ref().map(|mk| {
            let cm = 4usize;
            let ck = GROUP_SIZE; // K = one group per row (gs_per_row=1)
            let w_check: Vec<u32> = (0..cm * ck / 8)
                .map(|i| {
                    // Pack 8 nibbles: 0,1,2,...,7 repeating
                    let mut v = 0u32;
                    for bit in 0..8u32 {
                        v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                    }
                    v
                })
                .collect();
            let s_check: Vec<f32> = vec![0.1f32; cm]; // gs_per_row=1
            let b_check: Vec<f32> = vec![0.0f32; cm];
            let x_check: Vec<f32> = vec![1.0f32; ck];
            let ref_out = cpu_qmv(&w_check, &s_check, &b_check, &x_check, cm, ck);

            let w_check_bytes: Vec<u8> = w_check.iter().flat_map(|v| v.to_le_bytes()).collect();
            let w_buf_c = runner.buffer_bytes(&w_check_bytes);
            let s_buf_c = runner.buffer_f32(&s_check);
            let b_buf_c = runner.buffer_f32(&b_check);
            let x_buf_c = runner.buffer_f32(&x_check);
            let out_c = runner.buffer_zeros(cm * 4);
            let k_buf = runner.buffer_u32(ck as u32);
            let gpr_buf = runner.buffer_u32(1u32); // gs_per_row = 1
            runner.measure(
                mk,
                &[&w_buf_c, &s_buf_c, &b_buf_c, &x_buf_c, &out_c, &k_buf, &gpr_buf],
                [cm, 1, 1],
                [TPG, 1, 1],
                0,
                1,
            );
            let mt_out = runner.read_f32_slice(&out_c, cm);
            let n_bad =
                ref_out.iter().zip(mt_out.iter()).filter(|(r, m)| (*r - *m).abs() > 1e-3).count();
            EquivResult {
                n_checked: cm,
                max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
                cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
                passed: n_bad == 0,
            }
        });

        let gs_per_row = k / GROUP_SIZE;
        let w_mt_buf = runner.buffer_bytes(&w_data[..w_elems * 4]);
        let s_mt_buf = runner.buffer_f32(&scales_f32);
        let b_mt_buf = runner.buffer_f32(&biases_f32);
        let x_mt_buf = runner.buffer_f32(&x_f32);
        let k_buf = runner.buffer_u32(k as u32);
        let gpr_buf = runner.buffer_u32(gs_per_row as u32);

        // f32 MT byte count
        let bytes_mt = (m * k / 2 + sb_elems * 4 * 2 + k * 4 + m * 4) as f64;

        let mt_perf = mk.as_ref().and_then(|mk| {
            let out_buf = runner.buffer_zeros(m * 4);
            let st = runner.bench(
                mk,
                &[&w_mt_buf, &s_mt_buf, &b_mt_buf, &x_mt_buf, &out_buf, &k_buf, &gpr_buf],
                [m, 1, 1],
                [TPG, 1, 1],
                3,
                10,
            );
            to_gbps(&st, bytes_mt)
        });

        let shape = format!("M={m} K={k} f32 gs{GROUP_SIZE} b4");
        let result = if let Some(mt_perf) = mt_perf {
            BENCH.implemented(shape, ref_perf, mt_perf, equiv.unwrap())
        } else {
            BENCH.nyi(format!("M={m} K={k} f16 gs{GROUP_SIZE} b4"), ref_perf)
        };
        results.push(result);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_qmv_msl_generates() {
        let msl = qmv_msl().expect("codegen failed");
        assert!(msl.contains("mt_qmv_f32"));
    }

    #[test]
    fn cpu_qmv_basic() {
        // Single row, K=8 (one group of 8, one pack), all int4=7, scale=1, bias=0, x=1
        let w = vec![0x7777_7777u32]; // all nibbles = 7
        let scales = vec![1.0f32];
        let biases = vec![0.0f32];
        let x = vec![1.0f32; 8];
        let _out = cpu_qmv(&w, &scales, &biases, &x, 1, 8);
        // 8 elements × (1.0 * 7 + 0) * 1.0 = 56.0
        // But GROUP_SIZE=64, so K must be a multiple of GROUP_SIZE...
        // Use K=GROUP_SIZE=64 instead:
        let w64: Vec<u32> = vec![0x7777_7777u32; 8]; // 8 packs × 8 nibbles = 64 elements
        let scales64 = vec![1.0f32];
        let biases64 = vec![0.0f32];
        let x64 = vec![1.0f32; 64];
        let out64 = cpu_qmv(&w64, &scales64, &biases64, &x64, 1, 64);
        assert!((out64[0] - 448.0f32).abs() < 1e-4, "expected 448.0 got {}", out64[0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_qmv_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        runner.compile(SRC, REF_NAME).unwrap_or_else(|e| panic!("{REF_NAME} compile error: {e}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_qmv_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = qmv_msl().expect("codegen");
        runner
            .compile(&msl, "mt_qmv_f32")
            .unwrap_or_else(|e| panic!("mt_qmv_f32 compile error: {e}\nMSL:\n{msl}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_qmv_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = qmv_msl().expect("codegen");
        let mk = runner.compile(&msl, "mt_qmv_f32").expect("compile");

        // M=4 rows, K=64 (one group per row, gs_per_row=1)
        let m = 4usize;
        let k = GROUP_SIZE;
        let gs_per_row = 1usize;

        // Pack known int4 values: row r, pack p, bit b → nibble = (r + p + b) & 0xF
        let w: Vec<u32> = (0..m * k / 8)
            .map(|i| {
                let mut v = 0u32;
                for bit in 0..8u32 {
                    v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                }
                v
            })
            .collect();
        let scales: Vec<f32> = vec![0.1f32; m * gs_per_row];
        let biases: Vec<f32> = vec![0.0f32; m * gs_per_row];
        let x: Vec<f32> = vec![1.0f32; k];

        let ref_out = cpu_qmv(&w, &scales, &biases, &x, m, k);

        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_buf = runner.buffer_bytes(&w_bytes);
        let s_buf = runner.buffer_f32(&scales);
        let b_buf = runner.buffer_f32(&biases);
        let x_buf = runner.buffer_f32(&x);
        let out_buf = runner.buffer_zeros(m * 4);
        let k_buf = runner.buffer_u32(k as u32);
        let gpr_buf = runner.buffer_u32(gs_per_row as u32);

        runner.measure(
            &mk,
            &[&w_buf, &s_buf, &b_buf, &x_buf, &out_buf, &k_buf, &gpr_buf],
            [m, 1, 1],
            [TPG, 1, 1],
            0,
            1,
        );
        let mt_out = runner.read_f32_slice(&out_buf, m);
        for (i, (&r, &mt)) in ref_out.iter().zip(&mt_out).enumerate() {
            assert!((r - mt).abs() < 1e-2, "row {i}: ref={r} mt={mt}");
        }
    }
}
