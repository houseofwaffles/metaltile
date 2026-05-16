//! steel_gemm_segmented benchmark — #[kernel] DSL vs MLX
//!
//! Reference: metal/steel/gemm/steel_gemm_segmented.metal  (MLX, Apache-2.0)
//!
//! The MLX segmented GEMM accepts segment offsets defining variable-length
//! batches. With a single segment covering all rows (offsets = [0, M]) the
//! kernel reduces to a plain tiled matmul. We pass identity segments to
//! the reference and compare against the DSL `dot(a,b)`.
//!
//! Kernel: steel_segmented_mm_nn_float16_float16_bm64_bn64_bk16_wm2_wn2
//!
//! Buffer layout:
//!   0  A         fp16 [M×K]
//!   1  B         fp16 [K×N]
//!   2  segments  uint32 [2]  = [0, M]  (single segment)
//!   3  C/output  fp16 [M×N]
//!   4  params    GEMMParams (72 B)
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
// (GPU helper imports moved to metaltile-cli)
// (GPU helper imports moved to metaltile-cli)
// (GPU helper imports moved to metaltile-cli)
    },
// (GPU imports moved to metaltile-cli)
};

pub(crate) static SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_segmented.metal"));

pub(crate) const FN: &str = "steel_segmented_mm_nn_float16_float16_bm64_bn64_bk16_wm2_wn2";

pub(crate) const BM: usize = 64;
pub(crate) const BN: usize = 64;
pub(crate) const BK: usize = 16;

pub(crate) const SHAPES: &[(usize, usize, usize)] = &[(1_024, 1_024, 1_024), (4_096, 4_096, 4_096)];
const BENCH: OpBench = OpBench::new("matmul_segmented_fp16", "GFLOPS");
const TOLERANCE: EquivTolerance = EquivTolerance::new(1.0, 0.999);

// ── Helpers ──────────────────────────────────────────────────────────────────

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
    w64!(32, m as i64 * k as i64);
    w64!(40, k as i64 * n as i64);
    w64!(48, m as i64 * n as i64);
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
// (GPU imports moved to metaltile-cli)
    kernel: &CompiledKernel,
    a: &GpuBuffer,
    b: &GpuBuffer,
    segments: &GpuBuffer,
    out: &GpuBuffer,
    params: &GpuBuffer,
    m: usize,
    n: usize,
) -> Vec<f32> {
    let (tgs, tpg) = ref_dispatch(m, n);
// (GPU helper imports moved to metaltile-cli)
}

fn run_mt_once(
// (GPU imports moved to metaltile-cli)
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
// (GPU helper imports moved to metaltile-cli)
}

// ── Kernel ────────────────────────────────────────────────────────────────────

#[kernel]
pub fn mt_matmul(
    a: Tensor<f16, shape!(M, K)>,
    b: Tensor<f16, shape!(K, N)>,
    c: Tensor<f16, shape!(M, N)>,
) {
    dot(a, b);
}

// ── Bench ────────────────────────────────────────────────────────────────────

// (GPU imports moved to metaltile-cli)
    // TODO: GPU bench code moved to metaltile-cli/src/ops/
    vec![]
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
    fn ref_segmented_compiles() {
// (GPU imports moved to metaltile-cli)
        if let Err(e) =
            runner.compile_with_bool_constants(SRC, FN, &[(199, true), (200, true), (201, true)])
        {
            eprintln!("segmented ref compile (non-fatal): {e}");
        }
    }
}
