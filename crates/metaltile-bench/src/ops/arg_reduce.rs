//! arg_reduce benchmarks — metal/arg_reduce.metal  (MLX, Apache-2.0)
//!
//! Argmax over a flat 1-D array.
//!
//! MLX reference: `argmax_float32`
//!   Params: (in, out:u32*, shape:i32*, in_strides:i64*, out_strides:i64*,
//!            ndim:u64, axis_stride:i64, axis_size:u64)
//!   For 1-D: ndim=0 (empty shape/strides), axis_stride=1, axis_size=N.
//!   Grid: [TPG, 1, 1] × [TPG, 1, 1]
//!
//! MetalTile: `mt_argmax_f32` — parallel argmax, #[kernel] DSL.
//!   Grid: [1, 1, 1] × [256, 1, 1]
//!   Algorithm: mutable per-thread best (val, idx) + 8-stage threadgroup
//!   binary-tree reduction.  Tie-breaking: strict > keeps smallest index.
//!   KernelMode::Reduction
//!
//! Note: f32-only (the MLX argmax reference is f32-only). The tree reduction
//! is implemented in pure DSL using threadgroup_alloc + barriers.

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{OpBench, OpResult, check_equiv, run_f32_once, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/arg_reduce.metal");

const BENCH: OpBench = OpBench::new("argmax_f32", "GB/s");
const N: usize = 4_096 * 256; // ~1 M elements
const CHECK_N: usize = 4_096;
const TPG: usize = 256;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Parallel argmax: per-thread mutable best + threadgroup binary-tree reduction.
///
/// N_READS=1: each thread processes one element per outer loop iteration.
/// `tg_vals[256]` / `tg_idxs[256]`: threadgroup scratch (float; indices stored
/// as float since IEEE-754 f32 represents integers exactly up to 2^24 ≈ 16 M).
///
/// Tree reduction uses BitAnd/BitOr to compute:
///   better = (ov > tv) | ((ov == tv) & (oi < ti))
/// which prefers larger values and breaks ties by smaller index.
///
/// Grid: [1, 1, 1] × [256, 1, 1]
#[kernel]
pub fn mt_argmax_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let lid = tid;
    // `lid - lid` evaluates to 0 with uint type, so auto deduces uint for best_idx.
    let mut best_val = neg_infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1) / lsize;
    for _r in range(0, n_iters, 1) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]);
            let better = v > best_val;
            if better {
                best_val = v;
                best_idx = pos;
            }
        }
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();
    // Stage 1: stride=128
    {
        let ov128 = threadgroup_load("tg_vals", lid + 128);
        let oi128 = threadgroup_load("tg_idxs", lid + 128);
        let tv128 = threadgroup_load("tg_vals", lid);
        let ti128 = threadgroup_load("tg_idxs", lid);
        let bgt128 = ov128 > tv128;
        let beq128 = ov128 == tv128;
        let blt128 = oi128 < ti128;
        let b23128 = beq128 & blt128;
        let bet128 = bgt128 | b23128;
        let nv128 = select(bet128, ov128, tv128);
        let ni128 = select(bet128, oi128, ti128);
        if lid < 128 {
            threadgroup_store("tg_vals", lid, nv128);
            threadgroup_store("tg_idxs", lid, ni128);
        }
    }
    threadgroup_barrier();
    // Stage 2: stride=64
    {
        let ov64 = threadgroup_load("tg_vals", lid + 64);
        let oi64 = threadgroup_load("tg_idxs", lid + 64);
        let tv64 = threadgroup_load("tg_vals", lid);
        let ti64 = threadgroup_load("tg_idxs", lid);
        let bgt64 = ov64 > tv64;
        let beq64 = ov64 == tv64;
        let blt64 = oi64 < ti64;
        let b2364 = beq64 & blt64;
        let bet64 = bgt64 | b2364;
        let nv64 = select(bet64, ov64, tv64);
        let ni64 = select(bet64, oi64, ti64);
        if lid < 64 {
            threadgroup_store("tg_vals", lid, nv64);
            threadgroup_store("tg_idxs", lid, ni64);
        }
    }
    threadgroup_barrier();
    // Stage 3: stride=32
    {
        let ov32 = threadgroup_load("tg_vals", lid + 32);
        let oi32 = threadgroup_load("tg_idxs", lid + 32);
        let tv32 = threadgroup_load("tg_vals", lid);
        let ti32 = threadgroup_load("tg_idxs", lid);
        let bgt32 = ov32 > tv32;
        let beq32 = ov32 == tv32;
        let blt32 = oi32 < ti32;
        let b2332 = beq32 & blt32;
        let bet32 = bgt32 | b2332;
        let nv32 = select(bet32, ov32, tv32);
        let ni32 = select(bet32, oi32, ti32);
        if lid < 32 {
            threadgroup_store("tg_vals", lid, nv32);
            threadgroup_store("tg_idxs", lid, ni32);
        }
    }
    threadgroup_barrier();
    // Stage 4: stride=16
    {
        let ov16 = threadgroup_load("tg_vals", lid + 16);
        let oi16 = threadgroup_load("tg_idxs", lid + 16);
        let tv16 = threadgroup_load("tg_vals", lid);
        let ti16 = threadgroup_load("tg_idxs", lid);
        let bgt16 = ov16 > tv16;
        let beq16 = ov16 == tv16;
        let blt16 = oi16 < ti16;
        let b2316 = beq16 & blt16;
        let bet16 = bgt16 | b2316;
        let nv16 = select(bet16, ov16, tv16);
        let ni16 = select(bet16, oi16, ti16);
        if lid < 16 {
            threadgroup_store("tg_vals", lid, nv16);
            threadgroup_store("tg_idxs", lid, ni16);
        }
    }
    threadgroup_barrier();
    // Stage 5: stride=8
    {
        let ov8 = threadgroup_load("tg_vals", lid + 8);
        let oi8 = threadgroup_load("tg_idxs", lid + 8);
        let tv8 = threadgroup_load("tg_vals", lid);
        let ti8 = threadgroup_load("tg_idxs", lid);
        let bgt8 = ov8 > tv8;
        let beq8 = ov8 == tv8;
        let blt8 = oi8 < ti8;
        let b238 = beq8 & blt8;
        let bet8 = bgt8 | b238;
        let nv8 = select(bet8, ov8, tv8);
        let ni8 = select(bet8, oi8, ti8);
        if lid < 8 {
            threadgroup_store("tg_vals", lid, nv8);
            threadgroup_store("tg_idxs", lid, ni8);
        }
    }
    threadgroup_barrier();
    // Stage 6: stride=4
    {
        let ov4 = threadgroup_load("tg_vals", lid + 4);
        let oi4 = threadgroup_load("tg_idxs", lid + 4);
        let tv4 = threadgroup_load("tg_vals", lid);
        let ti4 = threadgroup_load("tg_idxs", lid);
        let bgt4 = ov4 > tv4;
        let beq4 = ov4 == tv4;
        let blt4 = oi4 < ti4;
        let b234 = beq4 & blt4;
        let bet4 = bgt4 | b234;
        let nv4 = select(bet4, ov4, tv4);
        let ni4 = select(bet4, oi4, ti4);
        if lid < 4 {
            threadgroup_store("tg_vals", lid, nv4);
            threadgroup_store("tg_idxs", lid, ni4);
        }
    }
    threadgroup_barrier();
    // Stage 7: stride=2
    {
        let ov2 = threadgroup_load("tg_vals", lid + 2);
        let oi2 = threadgroup_load("tg_idxs", lid + 2);
        let tv2 = threadgroup_load("tg_vals", lid);
        let ti2 = threadgroup_load("tg_idxs", lid);
        let bgt2 = ov2 > tv2;
        let beq2 = ov2 == tv2;
        let blt2 = oi2 < ti2;
        let b232 = beq2 & blt2;
        let bet2 = bgt2 | b232;
        let nv2 = select(bet2, ov2, tv2);
        let ni2 = select(bet2, oi2, ti2);
        if lid < 2 {
            threadgroup_store("tg_vals", lid, nv2);
            threadgroup_store("tg_idxs", lid, ni2);
        }
    }
    threadgroup_barrier();
    // Stage 8: stride=1
    {
        let ov1 = threadgroup_load("tg_vals", lid + 1);
        let oi1 = threadgroup_load("tg_idxs", lid + 1);
        let tv1 = threadgroup_load("tg_vals", lid);
        let ti1 = threadgroup_load("tg_idxs", lid);
        let bgt1 = ov1 > tv1;
        let beq1 = ov1 == tv1;
        let blt1 = oi1 < ti1;
        let b231 = beq1 & blt1;
        let bet1 = bgt1 | b231;
        let nv1 = select(bet1, ov1, tv1);
        let ni1 = select(bet1, oi1, ti1);
        if lid < 1 {
            threadgroup_store("tg_vals", lid, nv1);
            threadgroup_store("tg_idxs", lid, ni1);
        }
    }
    threadgroup_barrier();
    if lid == 0 {
        store(out[0], threadgroup_load("tg_idxs", 0));
    }
}

fn argmax_msl() -> String {
    let mut k = mt_argmax_f32::kernel_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[argmax]: {e}");
        String::new()
    })
}

// ── Bench ─────────────────────────────────────────────────────────────────────

fn cpu_argmax(inp: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut idx = 0usize;
    for (i, &v) in inp.iter().enumerate() {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx as f32
}

pub fn bench_arg_reduce(runner: &GpuRunner) -> Vec<OpResult> {
    let mt_msl = argmax_msl();
    let mk = if mt_msl.is_empty() { None } else { runner.compile(&mt_msl, "mt_argmax_f32").ok() };

    // MLX argmax_float32: (in, out, shape, in_strides, out_strides, ndim, axis_stride, axis_size)
    let rk = runner.compile(SRC, "argmax_float32").ok();

    let vals: Vec<f32> = (0..N).map(|i| ((i * 13 + 7) % 1009) as f32 * 0.001).collect();
    let inp_buf = runner.buffer_f32(&vals);
    let bytes = (N * 4) as f64;

    let dummy = runner.buffer_u32(0u32);
    let ndim = runner.buffer_u64(0u64);
    let ax_stride = runner.buffer_i64(1i64);
    let ax_size = runner.buffer_u64(N as u64);
    let ref_out = runner.buffer_zeros(4);

    let ref_perf = rk.as_ref().and_then(|rk| {
        let st = runner.bench(
            rk,
            &[&inp_buf, &ref_out, &dummy, &dummy, &dummy, &ndim, &ax_stride, &ax_size],
            [TPG, 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let equiv = mk.as_ref().map(|mk| {
        let check_vals: Vec<f32> = (0..CHECK_N).map(|i| ((i * 7 + 3) % 97) as f32 * 0.1).collect();
        let expected = cpu_argmax(&check_vals);
        let inp_c = runner.buffer_f32(&check_vals);
        let out_c = runner.buffer_zeros(4);
        let ns_c = runner.buffer_u32(CHECK_N as u32);
        let mt_vals =
            run_f32_once(runner, mk, &[&inp_c, &out_c, &ns_c], &out_c, 1, [1, 1, 1], [TPG, 1, 1]);
        check_equiv(&[expected], &mt_vals, 0.5)
    });

    let ns_u32 = runner.buffer_u32(N as u32);
    let mt_out = runner.buffer_zeros(4);

    let mt_perf = mk.as_ref().and_then(|mk| {
        let st = runner.bench(mk, &[&inp_buf, &mt_out, &ns_u32], [1, 1, 1], [TPG, 1, 1], 3, 10);
        to_gbps(&st, bytes)
    });

    let shape = format!("N={N} f32");
    let result = if let Some(mt_perf) = mt_perf {
        BENCH.implemented(shape, ref_perf, mt_perf, equiv.expect("mk Some → equiv Some"))
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
    fn argmax_msl_generates() {
        let msl = argmax_msl();
        assert!(!msl.trim().is_empty(), "argmax MSL should not be empty");
        assert!(msl.contains("mt_argmax_f32"), "kernel name missing");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn argmax_kernel_compiles() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        let msl = argmax_msl();
        runner
            .compile(&msl, "mt_argmax_f32")
            .unwrap_or_else(|e| panic!("mt_argmax_f32 compile error: {e}\nMSL:\n{msl}"));
    }
}
