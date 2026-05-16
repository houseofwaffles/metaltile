//! matmul_fp16 benchmark using the MLX Steel GEMM kernel.
//!
//! Reference: metal/steel/gemm/steel_gemm_fused.metal  (MLX, Apache-2.0)
//! Kernel:    steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2
//!
//! The Steel GEMM uses Metal function_constants. We pre-specialise the source
//! by replacing each [[function_constant(N)]] declaration with a hardcoded
//! value and stripping the conditional buffer attributes so all slots are
//! always present. Dead-code branches are never executed because the constants
//! are compile-time booleans.
//!
//! Specialisation: has_batch=false, use_out_source=false, do_axpby=false,
//!                 align_M=true, align_N=true, align_K=true.
//!
//! Buffer layout (post-specialisation):
//!   0  A              (fp16, [M×K])
//!   1  B              (fp16, [K×N])
//!   2  C              (dummy 4 B — was conditional on use_out_source)
//!   3  D / output     (fp16, [M×N])
//!   4  GEMMParams     (struct, 72 B)
//!   5  GEMMAddMMParams (dummy 24 B — was conditional)
//!   6  batch_shape    (dummy 4 B  — was conditional on has_batch)
//!   7  batch_strides  (dummy 8 B  — was conditional)
//!
//! Dispatch: [N/BN, M/BM, 1] threadgroups × [128, 1, 1] threads/tg

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
    include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_fused.metal"));
pub(crate) const FN: &str = "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2";

pub(crate) const BM: usize = 64;
pub(crate) const BN: usize = 64;
pub(crate) const BK: usize = 16;

pub(crate) const SHAPES: &[(usize, usize, usize)] = &[(1_024, 1_024, 1_024), (4_096, 4_096, 4_096)];
const BENCH: OpBench = OpBench::new("matmul", "GFLOPS");
const MATMUL_TOLERANCE: EquivTolerance = EquivTolerance::new(1.0, 0.999);

/// Replace [[function_constant(N)]] declarations with compile-time constants
/// and strip conditional buffer annotations so all slots are always bound.
pub(crate) fn specialise(src: &str) -> String {
    src.replace(
        "constant bool has_batch [[function_constant(10)]];",
        "constant bool has_batch = false;",
    )
    .replace(
        "constant bool use_out_source [[function_constant(100)]];",
        "constant bool use_out_source = false;",
    )
    .replace(
        "constant bool do_axpby [[function_constant(110)]];",
        "constant bool do_axpby = false;",
    )
    .replace("constant bool align_M [[function_constant(200)]];", "constant bool align_M = true;")
    .replace("constant bool align_N [[function_constant(201)]];", "constant bool align_N = true;")
    .replace("constant bool align_K [[function_constant(202)]];", "constant bool align_K = true;")
    .replace("[[buffer(2), function_constant(use_out_source)]]", "[[buffer(2)]]")
    .replace("[[buffer(5), function_constant(use_out_source)]]", "[[buffer(5)]]")
    .replace("[[buffer(6), function_constant(has_batch)]]", "[[buffer(6)]]")
    .replace("[[buffer(7), function_constant(has_batch)]]", "[[buffer(7)]]")
}

/// Serialise GEMMParams as its C struct layout (little-endian, 72 bytes).
///
/// Offsets:
///   0  M              i32
///   4  N              i32
///   8  K              i32
///  12  lda            i32  (= K, stride of A rows)
///  16  ldb            i32  (= N, stride of B rows)
///  20  ldd            i32  (= N, stride of D rows)
///  24  tiles_n        i32  (= N/BN)
///  28  tiles_m        i32  (= M/BM)
///  32  batch_stride_a i64
///  40  batch_stride_b i64
///  48  batch_stride_d i64
///  56  swizzle_log    i32
///  60  gemm_k_iters   i32  (= K/BK)
///  64  batch_ndim     i32
///  68  [4 bytes padding]
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
    w32!(20, n); // lda=K, ldb=N, ldd=N
    w32!(24, n / BN);
    w32!(28, m / BM); // tiles_n, tiles_m
    w64!(32, m * k);
    w64!(40, k * n);
    w64!(48, m * n); // batch strides
    w32!(56, 0);
    w32!(60, k / BK);
    w32!(64, 0); // swizzle_log=0, k_iters, batch_ndim=0
    b
}

fn ref_dispatch(m: usize, n: usize) -> ([usize; 3], [usize; 3]) {
    ([n / BN, m / BM, 1], [128, 1, 1])
}

fn mt_dispatch(use_simd: bool, m: usize, n: usize) -> ([usize; 3], [usize; 3]) {
    // Simdgroup path: 64x64 tile, 128 threads (16x8) matching emit_tiled_simdgroup.
    // Scalar path: BM x BN tile, 256 threads (16x16).
    let (tm, tn, tpg) = if use_simd { (64, 64, [16usize, 8, 1]) } else { (BM, BN, [16, 16, 1]) };
    ([n.div_ceil(tn), m.div_ceil(tm), 1], tpg)
}

fn run_ref_matmul_once(
// (GPU imports moved to metaltile-cli)
    kernel: &CompiledKernel,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    params: &GpuBuffer,
    dummy4: &GpuBuffer,
    dummy8: &GpuBuffer,
    dummy24: &GpuBuffer,
    m: usize,
    n: usize,
) -> Vec<f32> {
    let (tgs, tpg) = ref_dispatch(m, n);
// (GPU helper imports moved to metaltile-cli)
        runner,
        kernel,
        &[a, b, dummy4, out, params, dummy24, dummy4, dummy8],
        out,
        m * n,
        tgs,
        tpg,
    )
}

fn run_mt_matmul_once(
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
    // Constexprs are emitted in first-appearance order: M, K, N.
    let m_b = runner.buffer_u32(m as u32);
    let k_b = runner.buffer_u32(k as u32);
    let n_b = runner.buffer_u32(n as u32);
    let (tgs, tpg) = mt_dispatch(use_simd, m, n);
// (GPU helper imports moved to metaltile-cli)
}

fn patterned_f16(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| {
            let centered = ((i.wrapping_mul(17).wrapping_add(seed * 29)) % 31) as i32 - 15;
            fp32_to_f16(centered as f32 * 0.0625)
        })
        .collect()
}

fn fp32_to_f16(v: f32) -> u16 {
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
}

// (GPU imports moved to metaltile-cli)
    // TODO: GPU bench code moved to metaltile-cli/src/ops/
    vec![]
}

#[kernel]
pub fn mt_matmul(
    a: Tensor<f16, shape!(M, K)>,
    b: Tensor<f16, shape!(K, N)>,
    c: Tensor<f16, shape!(M, N)>,
) {
    dot(a, b);
}

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
        let msl =
            MslGenerator::new(cfg).generate(&k).expect("matmul MSL generation should succeed");
        assert!(
            msl.contains("kernel void mt_matmul"),
            "expected generated kernel entrypoint, got:\n{msl}"
        );
    }
}
