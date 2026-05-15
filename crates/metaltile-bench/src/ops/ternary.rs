//! Ternary select benchmark — #[kernel] DSL vs MLX metal/ternary.metal
//!
//! MLX kernel: v_Selectfloat32 / v_Selectfloat16 / v_Selectbfloat16 (ternary.metal)
//!   Params: (cond: device T*, a: device T*, b: device T*, dst: device T*,
//!            size: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Grid: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: dst[i] = cond[i] != 0 ? a[i] : b[i]  (one thread per element)
//!
//! MetalTile: mt_select — same algorithm via #[kernel] DSL.
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

static SRC: &str = include_str!("../metal/ternary.metal");
const BENCH: OpBench = OpBench::new("select", "GB/s");
pub const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 2_048;
const TPG: usize = 256;

/// Select: out[i] = cond[i] != 0.0 ? on_true[i] : on_false[i]
///
/// Matches MLX `v_Select{tn}` — dispatch one thread per element.
#[kernel]
pub fn mt_select<T>(cond: Tensor<T>, on_true: Tensor<T>, on_false: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let c = load(cond[idx]);
    let t = load(on_true[idx]);
    let f = load(on_false[idx]);
    store(out[idx], select(c, t, f));
}

pub fn bench_select_f32(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_select_for(runner, dt)).collect()
}

fn bench_select_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let tn = mlx_tname(dt);
    let dlabel = dtype_label(dt);
    let eb = elem_bytes(dt);
    let tol = dtype_tol(dt);

    let msl = MslGenerator::default().generate(&mt_select::kernel_ir_for(dt)).unwrap_or_else(|e| {
        eprintln!("[select {dlabel}]: {e}");
        String::new()
    });
    let mk = runner.compile(&msl, "mt_select").ok();
    let rk = runner.compile(SRC, &format!("v_Select{tn}")).ok();

    // Correctness: cond is typed (0.0 or non-zero), true/false are typed data.
    let cond_f32: Vec<f32> =
        (0..N_CHECK).map(|i| if i % 3 == 0 { 0.0f32 } else { 1.0f32 }).collect();
    let true_f32: Vec<f32> = (0..N_CHECK).map(|i| 1.0 + i as f32 * 0.01).collect();
    let false_f32: Vec<f32> = (0..N_CHECK).map(|i| -2.0 - i as f32 * 0.02).collect();
    let true_q = quantize_roundtrip(&true_f32, dt);
    let false_q = quantize_roundtrip(&false_f32, dt);
    let cpu_ref: Vec<f32> =
        (0..N_CHECK).map(|i| if cond_f32[i] != 0.0 { true_q[i] } else { false_q[i] }).collect();

    // ref kernel: v_Select{tn}(bool* cond, T* true, T* false, T* out, uint size)
    let cond_bool: Vec<u8> = cond_f32.iter().map(|&v| if v != 0.0 { 1u8 } else { 0u8 }).collect();
    let ref_cond = runner.buffer_bytes(&cond_bool);
    let ref_true = buffer_typed(runner, &true_f32, dt);
    let ref_false = buffer_typed(runner, &false_f32, dt);
    let ref_out = zeros_typed(runner, N_CHECK, dt);
    let ref_size = runner.buffer_u32(N_CHECK as u32);
    let ref_check = rk.as_ref().map(|rk| {
        run_typed_once(
            runner,
            rk,
            &[&ref_cond, &ref_true, &ref_false, &ref_out, &ref_size],
            &ref_out,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        )
    });

    let mt_cond = buffer_typed(runner, &cond_f32, dt);
    let mt_true = buffer_typed(runner, &true_f32, dt);
    let mt_false = buffer_typed(runner, &false_f32, dt);
    let mt_out = zeros_typed(runner, N_CHECK, dt);
    let mt_check = mk.as_ref().map(|mk| {
        run_typed_once(
            runner,
            mk,
            &[&mt_cond, &mt_true, &mt_false, &mt_out],
            &mt_out,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        )
    });

    let equiv = if let Some(mt_vals) = mt_check {
        if let Some(ref_vals) = ref_check {
            check_equiv(&ref_vals, &mt_vals, tol)
        } else {
            check_equiv(&cpu_ref, &mt_vals, tol)
        }
    } else {
        return vec![];
    };

    // Perf: 3 typed data arrays + 1 bool cond ≈ 3*eb + 1 bytes per element
    // Simplified: use 4*eb for cond+true+false+out (close enough)
    let bytes = (N_ELEM * eb * 4) as f64;
    let cond_bool_perf: Vec<u8> = (0..N_ELEM).map(|i| if i % 2 == 0 { 1u8 } else { 0u8 }).collect();
    let ref_cond_perf = runner.buffer_bytes(&cond_bool_perf);
    let true_perf = buffer_typed(runner, &vec![1.0f32; N_ELEM], dt);
    let false_perf = buffer_typed(runner, &vec![-1.0f32; N_ELEM], dt);
    let ref_size_perf = runner.buffer_u32(N_ELEM as u32);

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, N_ELEM, dt);
        let st = runner.bench(
            rk,
            &[&ref_cond_perf, &true_perf, &false_perf, &out, &ref_size_perf],
            [N_ELEM.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let mt_cond_perf = buffer_typed(
        runner,
        &(0..N_ELEM).map(|i| if i % 2 == 0 { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>(),
        dt,
    );
    let mt_perf = mk.as_ref().and_then(|mk| {
        let out = zeros_typed(runner, N_ELEM, dt);
        let st = runner.bench(
            mk,
            &[&mt_cond_perf, &true_perf, &false_perf, &out],
            [N_ELEM.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let shape = format!("N={N_ELEM} {dlabel}");
    let result = if let Some(mt_perf) = mt_perf {
        BENCH.implemented(shape, ref_perf, mt_perf, equiv)
    } else {
        BENCH.nyi(shape, ref_perf)
    };
    vec![result]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let msl = MslGenerator::default().generate(&mt_select::kernel_ir_for(dt)).unwrap();
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
            let msl = MslGenerator::default().generate(&mt_select::kernel_ir_for(dt)).unwrap();
            runner.compile(&msl, "mt_select").unwrap();
        }
    }
}
