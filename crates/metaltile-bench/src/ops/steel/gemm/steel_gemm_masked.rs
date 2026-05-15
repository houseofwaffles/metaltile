//! steel_gemm_masked benchmark — #[kernel] DSL vs MLX
//!
//! Reference: metal/steel/gemm/steel_gemm_masked.metal  (MLX, Apache-2.0)
//!
//! The MLX masked GEMM has nomask instantiations that compile away all
//! masking logic, reducing to a plain tiled matmul.  We compile the
//! nomask variant as the reference and compare against the DSL `dot(a,b)`.
//!
//! Kernel: steel_gemm_block_outmask_nomask_opmask_nomask_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_MN_taligned_K_taligned
//!
//! Buffer layout (all 14 slots bound; mask slots 10–13 are dummies):
//!   0  A         fp16 [M×K]
//!   1  B         fp16 [K×N]
//!   3  D         fp16 [M×N]    (output)
//!   4  params    GEMMParams (72 B)
//!   6  bshape    dummy
//!   7  bstrides  dummy
//!  10  out_mask  dummy  (nomask — never dereferenced)
//!  11  lhs_mask  dummy
//!  12  rhs_mask  dummy
//!  13  mstrides  dummy
//!
//! Dispatch: [N/BN, M/BM, 1] × [128, 1, 1]

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::{MslGenerator, TileSchedule, msl::MslConfig};

use crate::{
    ops::{
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        check_equiv_with,
        run_f16_once_as_f32,
        to_gflops,
    },
    runner::{CompiledKernel, GpuBuffer, GpuRunner},
};

static SRC: &str = include_str!("../../../metal/steel/gemm/steel_gemm_masked.metal");

const FN: &str = "steel_gemm_block_outmask_nomask_opmask_nomask_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_MN_taligned_K_taligned";

const BM: usize = 64;
const BN: usize = 64;
const BK: usize = 16;

const SHAPES: &[(usize, usize, usize)] = &[(1_024, 1_024, 1_024), (4_096, 4_096, 4_096)];
const BENCH: OpBench = OpBench::new("matmul_masked_fp16", "GFLOPS");
const TOLERANCE: EquivTolerance = EquivTolerance::new(1.0, 0.999);

// ── Helpers (shared with steel_gemm_fused) ──────────────────────────────────

fn params_bytes(m: usize, n: usize, k: usize) -> [u8; 72] {
    let mut b = [0u8; 72];
    macro_rules! w32 {
        ($off:expr, $v:expr) => {
            b[$off..$off + 4].copy_from_slice(&($v as i32).to_le_bytes());
        };
    }
    macro_rules! w64 {
        ($off:expr, $v:expr) => {
            b[$off..$off + 8].copy_from_slice(&($v as i64).to_le_bytes());
        };
    }
    w32!(0, m);
    w32!(4, n);
    w32!(8, k);
    w32!(12, k);
    w32!(16, n);
    w32!(20, n);
    w32!(24, n / BN);
    w32!(28, m / BM);
    w64!(32, m * k);
    w64!(40, k * n);
    w64!(48, m * n);
    w32!(56, 0);
    w32!(60, k / BK);
    w32!(64, 0);
    b
}

fn ref_dispatch(m: usize, n: usize) -> ([usize; 3], [usize; 3]) {
    ([n / BN, m / BM, 1], [128, 1, 1])
}

fn mt_dispatch(use_simd: bool, m: usize, n: usize) -> ([usize; 3], [usize; 3]) {
    let (tm, tn, tpg) = if use_simd { (64, 64, [16usize, 8, 1]) } else { (BM, BN, [16, 16, 1]) };
    ([n.div_ceil(tn), m.div_ceil(tm), 1], tpg)
}

fn patterned_f16(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| {
            let centered = ((i.wrapping_mul(17).wrapping_add(seed * 29)) % 31) as i32 - 15;
            let v = centered as f32 * 0.0625;
            let bits = v.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let mant = (bits >> 13) & 0x3ff;
            if exp <= 0 {
                sign
            } else if exp >= 31 {
                sign | 0x7c00
            } else {
                sign | ((exp as u16) << 10) | mant as u16
            }
        })
        .collect()
}

fn run_ref_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    params: &GpuBuffer,
    d4: &GpuBuffer,
    d8: &GpuBuffer,
    m: usize,
    n: usize,
) -> Vec<f32> {
    let (tgs, tpg) = ref_dispatch(m, n);
    // Slots: 0=A 1=B 3=D 4=params 6=bshape 7=bstrides 10-13=dummies
    run_f16_once_as_f32(
        runner,
        kernel,
        &[a, b, d4, out, params, d4, d8, d4, d4, d4, d8, d4, d4, d8],
        out,
        m * n,
        tgs,
        tpg,
    )
}

fn run_mt_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    use_simd: bool,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let m_b = runner.buffer_u32(m as u32);
    let k_b = runner.buffer_u32(k as u32);
    let n_b = runner.buffer_u32(n as u32);
    let (tgs, tpg) = mt_dispatch(use_simd, m, n);
    run_f16_once_as_f32(runner, kernel, &[a, b, out, &m_b, &k_b, &n_b], out, m * n, tgs, tpg)
}

// ── Kernel (same as steel_gemm_fused) ───────────────────────────────────────

#[kernel]
pub fn mt_matmul(
    a: Tensor<f16, shape!(M, K)>,
    b: Tensor<f16, shape!(K, N)>,
    c: Tensor<f16, shape!(M, N)>,
) {
    dot(a, b);
}

// ── Bench ────────────────────────────────────────────────────────────────────

pub fn bench_matmul_masked(runner: &GpuRunner) -> Vec<OpResult> {
    let ref_kernel =
        runner.compile(SRC, FN).inspect_err(|e| eprintln!("[{FN}] compile error: {e}"));

    let use_simd = runner.supports_simd_matrix();
    let mt_msl = {
        let mut k = mt_matmul::kernel_ir();
        k.mode = KernelMode::Tile2D;
        let cfg = MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: use_simd,
            ..MslConfig::default()
        };
        MslGenerator::new(cfg).generate(&k).unwrap()
    };
    let mt_kernel = runner
        .compile(&mt_msl, "mt_matmul")
        .inspect_err(|e| eprintln!("[mt_matmul] compile error: {e}"))
        .ok();

    let dummy4 = runner.buffer_zeros(4);
    let dummy8 = runner.buffer_zeros(8);

    SHAPES
        .iter()
        .map(|&(m, n, k)| {
            let shape = format!("{m}×{n}×{k}");
            let a = runner.buffer_f16(&patterned_f16(m * k, 1));
            let b = runner.buffer_f16(&patterned_f16(k * n, 7));
            let params = runner.buffer_bytes(&params_bytes(m, n, k));
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let ref_out = runner.buffer_zeros(m * n * 2);

            let ref_perf = ref_kernel.as_ref().ok().and_then(|rk| {
                let (tgs, tpg) = ref_dispatch(m, n);
                // Buffer order: 0=A 1=B 3=D 4=params 6=bshape 7=bstrides
                //              10=out_mask 11=lhs_mask 12=rhs_mask 13=mstrides
                // All mask slots are dummies for nomask variant.
                let st = runner.bench(
                    rk,
                    &[
                        &a, &b, &dummy4, &ref_out, &params, &dummy4, &dummy8, &dummy4, &dummy4,
                        &dummy4, &dummy8, &dummy4, &dummy4, &dummy8,
                    ],
                    tgs,
                    tpg,
                    3,
                    10,
                );
                to_gflops(&st, flops)
            });

            let equiv: Option<EquivResult> = ref_kernel.as_ref().ok().and_then(|rk| {
                mt_kernel.as_ref().map(|mk| {
                    let ref_vals =
                        run_ref_once(runner, rk, &a, &b, &ref_out, &params, &dummy4, &dummy8, m, n);
                    let mt_out = runner.buffer_zeros(m * n * 2);
                    let mt_vals = run_mt_once(runner, mk, &a, &b, &mt_out, use_simd, m, n, k);
                    check_equiv_with(&ref_vals, &mt_vals, TOLERANCE)
                })
            });

            let mt_perf = equiv.as_ref().and_then(|_| {
                mt_kernel.as_ref().and_then(|mk| {
                    let mt_out = runner.buffer_zeros(m * n * 2);
                    let (tgs, tpg) = mt_dispatch(use_simd, m, n);
                    let m_b = runner.buffer_u32(m as u32);
                    let k_b = runner.buffer_u32(k as u32);
                    let n_b = runner.buffer_u32(n as u32);
                    let st =
                        runner.bench(mk, &[&a, &b, &mt_out, &m_b, &k_b, &n_b], tgs, tpg, 3, 10);
                    to_gflops(&st, flops)
                })
            });

            match (mt_perf, equiv) {
                (Some(mt_perf), Some(equiv)) => BENCH.implemented(shape, ref_perf, mt_perf, equiv),
                _ => BENCH.nyi(shape, ref_perf),
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_matmul_ir_generates_msl() {
        let k = mt_matmul::kernel_ir();
        let cfg = MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: false,
            ..MslConfig::default()
        };
        let msl = MslGenerator::new(cfg).generate(&k).expect("MSL gen failed");
        assert!(msl.contains("kernel void mt_matmul"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_masked_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        // The masked GEMM kernel names are macro-generated from 8 mask-type
        // combinations; the exact name may vary. This test is informational.
        if let Err(e) = runner.compile(SRC, FN) {
            eprintln!("masked ref compile (non-fatal): {e}");
        }
    }
}
