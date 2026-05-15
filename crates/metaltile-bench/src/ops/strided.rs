//! Strided-tensor benchmark — #[kernel] DSL with #[strided] params vs MLX metal/copy.metal
//!
//! MLX kernel: copy_g_nd2 (copy.metal)
//!   Params: (src: device T*, dst: device T*, src_strides: constant int64_t*, index: uint2)
//!   Grid: [N, M, 1] × [1, 1, 1]  (one thread per output element)
//!   Algorithm: dst[row*N + col] = src[row*src_strides[0] + col*src_strides[1]]
//!
//! MetalTile: mt_strided_copy — same algorithm with #[strided] attribute on src.
//!   KernelMode::Elementwise, Grid3D dispatch [N, M, 1] × [1, 1, 1]
//!
//! The test uses a non-contiguous view: a sub-matrix of M×N taken from a M×(N+PAD) buffer,
//! so src_strides = [N+PAD, 1] while the logical shape is M×N.

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
        run_typed_once,
        to_gbps,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/copy.metal");

const BENCH: OpBench = OpBench::new("strided_copy", "GB/s");
// M rows × N cols, padded row stride = N+PAD (non-contiguous source)
const M: usize = 1024;
const N: usize = 4096;
const PAD: usize = 128; // extra elements per row making source non-contiguous
const TPG: usize = 1; // copy_g_nd2 uses one thread per element

/// Strided copy: dst[row, col] = src[row, col] where src has a non-unit row stride.
/// The #[strided] attribute causes codegen to emit {src}_strides[d] for index computation.
#[kernel]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}

fn strided_copy_msl_for(dt: DType) -> String {
    let mut k = mt_strided_copy::kernel_ir_for(dt);
    k.mode = KernelMode::Grid3D;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[strided_copy {dt:?}]: {e}");
        String::new()
    })
}

pub fn bench_strided(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_strided_for(runner, dt)).collect()
}

fn bench_strided_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let dlabel = dtype_label(dt);
    let tn = mlx_tname(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol(dt);

    let msl = strided_copy_msl_for(dt);
    let mk = runner.compile(&msl, "mt_strided_copy").ok();

    // copy_g_nd2 takes src_strides as a constant int64_t* buffer.
    // Buffer layout: [stride_for_dim0, stride_for_dim1] = [N+PAD, 1]
    let ref_name = format!("copy_g_nd2{tn}{tn}");
    let rk = runner.compile(SRC, &ref_name).ok();

    // ── Correctness ──────────────────────────────────────────────────────────
    // Small check: 8 rows × 16 cols, padded stride = 16+4 = 20
    const CM: usize = 8;
    const CN: usize = 16;
    const CP: usize = 4;
    let src_stride = CN + CP;

    // Source buffer: CM × (CN+CP) filled with recognisable values; only CM×CN are read.
    let src_vals: Vec<f32> = (0..CM * src_stride)
        .map(|i| {
            let row = i / src_stride;
            let col = i % src_stride;
            if col < CN { (row * CN + col) as f32 + 1.0 } else { -999.0 }
        })
        .collect();

    // Expected output: row-major CM×CN block.
    let expected: Vec<f32> = (0..CM * CN).map(|i| (i as f32 + 1.0)).collect();

    let src_buf = buffer_typed(runner, &src_vals, dt);
    let strides_buf = runner.buffer_bytes(
        &[src_stride as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let cols_buf = runner.buffer_u32(CN as u32);
    let cols_constexpr = runner.buffer_u32(CN as u32);

    let ref_equiv = rk.as_ref().map(|rk| {
        let out = zeros_typed(runner, CM * CN, dt);
        run_typed_once(
            runner,
            rk,
            &[&src_buf, &out, &strides_buf],
            &out,
            CM * CN,
            [CN, CM, 1],
            [TPG, TPG, 1],
            dt,
        )
    });
    let mt_equiv = mk.as_ref().map(|mk| {
        let out = zeros_typed(runner, CM * CN, dt);
        run_typed_once(
            runner,
            mk,
            &[&src_buf, &out, &cols_constexpr],
            &out,
            CM * CN,
            [M, N, 1],
            [TPG, TPG, 1],
            dt,
        )
    });

    let mt_check_small = mk.as_ref().map(|mk| {
        let out = zeros_typed(runner, CM * CN, dt);
        run_typed_once(
            runner,
            mk,
            &[&src_buf, &out, &cols_buf],
            &out,
            CM * CN,
            [CM, CN, 1],
            [1, 1, 1],
            dt,
        )
    });

    let equiv = match mt_check_small {
        Some(got) => check_equiv(&expected, &got, tol),
        None => {
            return vec![BENCH.nyi(format!("M={M} N={N}+{PAD} {dlabel}"), None)];
        },
    };
    let _ = (ref_equiv, mt_equiv); // suppress unused warnings

    // ── Throughput ───────────────────────────────────────────────────────────
    // Full M×N copy from a M×(N+PAD) source.
    let full_src: Vec<f32> = (0..M * (N + PAD)).map(|i| (i % 256) as f32 * 0.01).collect();
    let full_src_buf = buffer_typed(runner, &full_src, dt);
    let full_strides = runner.buffer_bytes(
        &[(N + PAD) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_cols = runner.buffer_u32(N as u32);
    let bytes = (M * N * eb * 2) as f64; // 1 read + 1 write

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, M * N, dt);
        let st = runner.bench(
            rk,
            &[&full_src_buf, &out, &full_strides],
            [N, M, 1],
            [TPG, TPG, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let mt_perf = mk.as_ref().and_then(|mk| {
        let out = zeros_typed(runner, M * N, dt);
        let st = runner.bench(mk, &[&full_src_buf, &out, &full_cols], [M, N, 1], [1, 1, 1], 3, 10);
        to_gbps(&st, bytes)
    });

    let shape = format!("M={M} N={N}+{PAD} {dlabel}");
    vec![match mt_perf {
        Some(p) => BENCH.implemented(shape, ref_perf, p, equiv),
        None => BENCH.nyi(shape, ref_perf),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let msl = strided_copy_msl_for(dt);
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
            let msl = strided_copy_msl_for(dt);
            runner.compile(&msl, "mt_strided_copy").unwrap();
        }
    }
}
