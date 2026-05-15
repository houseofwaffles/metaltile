//! Arange benchmark — #[kernel] DSL vs MLX metal/arange.metal
//!
//! MLX kernel: arangefloat32 / arangefloat16 / arangebfloat16 (arange.metal)
//!   Params: (start: constant T&, step: constant T&, out: device T*) — slots [0, 1, 2]
//!   Grid: [ceil(N/1024), 1, 1] × [1024, 1, 1]  (TPG=1024)
//!   Algorithm: out[index] = start + index * step  (one thread per element)
//!
//! MetalTile: mt_arange — same one-thread-per-element algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;
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

static SRC: &str = include_str!("../metal/arange.metal");

const BENCH: OpBench = OpBench::new("arange", "GB/s");
const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 4_096;
const TPG: usize = 1_024;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Arange: out[idx] = start + idx * step
///
/// `start` and `step` are passed as single-element typed buffers.
/// Dispatch: [ceil(N/TPG), 1, 1] x [TPG, 1, 1]
#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_arange_f32(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_arange_for(runner, dt)).collect()
}

fn bench_arange_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol(dt);

    let msl = MslGenerator::default().generate(&mt_arange::kernel_ir_for(dt)).unwrap_or_else(|e| {
        eprintln!("[arange {dlabel}]: {e}");
        String::new()
    });
    let mk = runner.compile(&msl, "mt_arange").ok();

    // MLX ref (may not exist for all dtypes — silently skip)
    let ref_name = format!("arange{}", mlx_tname(dt));
    let rk = runner.compile(SRC, &ref_name).ok();

    let start = 0.0f32;
    let step = 1.0f32;

    // Correctness: compare MT against CPU reference using quantize_roundtrip
    let equiv = mk.as_ref().map(|mk| {
        let inp_f32: Vec<f32> = (0..N_CHECK).map(|i| i as f32).collect();
        let cpu_ref: Vec<f32> =
            inp_f32.iter().map(|&i| quantize_roundtrip(&[start + step * i], dt)[0]).collect();
        let s_buf = buffer_typed(runner, &[start], dt);
        let st_buf = buffer_typed(runner, &[step], dt);
        let out_buf = zeros_typed(runner, N_CHECK, dt);
        let n_buf = runner.buffer_u32(N_CHECK as u32);
        let mt_vals = run_typed_once(
            runner,
            mk,
            &[&out_buf, &s_buf, &st_buf, &n_buf],
            &out_buf,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        );
        check_equiv(&cpu_ref, &mt_vals, tol)
    });

    let bytes = (N_ELEM * eb) as f64; // write-only

    // Reference perf (f32-only)
    let ref_perf = rk.as_ref().and_then(|rk| {
        let ref_start = runner.buffer_f32_scalar(start);
        let ref_step = runner.buffer_f32_scalar(step);
        let ref_out = runner.buffer_zeros(N_ELEM * 4);
        let st = runner.bench(
            rk,
            &[&ref_start, &ref_step, &ref_out],
            [N_ELEM.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    // MT perf
    let mt_start = buffer_typed(runner, &[start], dt);
    let mt_step = buffer_typed(runner, &[step], dt);
    let mt_out = zeros_typed(runner, N_ELEM, dt);
    let mt_n = runner.buffer_u32(N_ELEM as u32);
    let mt_perf = mk.as_ref().and_then(|mk| {
        let st = runner.bench(
            mk,
            &[&mt_out, &mt_start, &mt_step, &mt_n],
            [N_ELEM.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let shape = format!("N={N_ELEM} {dlabel}");
    vec![match mt_perf {
        Some(p) => BENCH.implemented(shape, ref_perf, p, equiv.expect("mk Some → equiv Some")),
        None => BENCH.nyi(shape, ref_perf),
    }]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let msl = MslGenerator::default().generate(&mt_arange::kernel_ir_for(dt)).unwrap();
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
            let msl = MslGenerator::default().generate(&mt_arange::kernel_ir_for(dt)).unwrap();
            runner.compile(&msl, "mt_arange").unwrap();
        }
    }
}
