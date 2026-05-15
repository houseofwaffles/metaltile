//! Sort benchmarks — metal/sort.metal  (MLX, Apache-2.0)
//!
//! Single-block sort of float32 arrays using bitonic sort.
//!
//! MLX reference: `c_block_sort_float32_float32_bn256_tn4`
//!   BN=256 threads, TN=4 elements/thread → sorts N ≤ 1024 per block
//!   Grid: [B, 1, 1] × [256, 1, 1]
//!
//! MetalTile: `mt_sort_f32` — bitonic sort, 256 threads × 4 elements/thread.
//!   55 stages, each with a threadgroup_barrier.
//!   Grid: [B, 1, 1] × [256, 1, 1]
//!   KernelMode::Reduction

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{EquivResult, OpBench, OpResult, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/sort.metal");

const REF_NAME: &str = "c_block_sort_float32_float32_bn256_tn4";
const N: usize = 1024; // elements per array (must be a power of 2, ≤ 1024)
const B: usize = 1024; // number of independent arrays
const BENCH: OpBench = OpBench::new("sort_f32", "GB/s");

// ── DSL kernel ───────────────────────────────────────────────────────────────

/// Bitonic sort of N float32 elements using 256 threads × 4 elements/thread.
///
/// Each threadgroup sorts one array of N elements (N ≤ 1024 = 256 × 4).
/// Uses 55 stages (log2(1024)×(log2(1024)+1)/2) with threadgroup barriers.
/// Sort direction: ascending.
///
/// Dispatch: [B, 1, 1] × [256, 1, 1]  (Reduction mode; one TG per array)
#[kernel]
pub fn mt_sort_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let block_id = program_id::<0>(); // tgid_x = array index
    let t = tid; // local thread index 0..255

    // Allocate 1024 floats in threadgroup shared memory.
    threadgroup_alloc("shared", 1024);

    // Load 4 elements per thread into shared memory.
    let base = block_id * n;
    threadgroup_store("shared", t * 4u32, load(inp[base + t * 4u32]));
    threadgroup_store("shared", t * 4u32 + 1u32, load(inp[base + t * 4u32 + 1u32]));
    threadgroup_store("shared", t * 4u32 + 2u32, load(inp[base + t * 4u32 + 2u32]));
    threadgroup_store("shared", t * 4u32 + 3u32, load(inp[base + t * 4u32 + 3u32]));
    threadgroup_barrier();

    // Bitonic sort: outer pass k = 2^1..2^10, inner pass j_bits = 0.._k-1.
    // flip_bit = _k - _jb - 1  →  j = 1 << flip_bit  →  partner = gi ^ j.
    // Direction: ascending when (gi >> _k) is even.
    for _k in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k, 1u32) {
            let flip = _k - _jb - 1u32;
            threadgroup_barrier();
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a = threadgroup_load("shared", gi);
                    let b = threadgroup_load("shared", partner);
                    // ascending when bit _k of gi is 0
                    let dir = (gi >> _k) & 1u32;
                    let want_swap = select(dir == 0u32, a > b, a < b);
                    threadgroup_store("shared", gi, select(want_swap, b, a));
                    threadgroup_store("shared", partner, select(want_swap, a, b));
                }
            }
        }
    }

    threadgroup_barrier();

    // Write sorted results back to global memory.
    store(out[base + t * 4u32], threadgroup_load("shared", t * 4u32));
    store(out[base + t * 4u32 + 1u32], threadgroup_load("shared", t * 4u32 + 1u32));
    store(out[base + t * 4u32 + 2u32], threadgroup_load("shared", t * 4u32 + 2u32));
    store(out[base + t * 4u32 + 3u32], threadgroup_load("shared", t * 4u32 + 3u32));
}

fn sort_msl() -> Result<String, String> {
    use metaltile::core::ir::KernelMode;
    let mut k = mt_sort_f32::kernel_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default()
        .generate(&k)
        .map_err(|e| format!("sort codegen: {e}"))
        .and_then(|msl| if msl.trim().is_empty() { Err("empty".into()) } else { Ok(msl) })
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_sort(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner.compile(SRC, REF_NAME).ok();

    let data: Vec<f32> = (0..B * N).map(|i| (B * N - i) as f32).collect();
    let inp = runner.buffer_f32(&data);
    let bytes = (B * N * 4 * 2) as f64;

    let size = runner.buffer_i32(N as i32);
    let stride1 = runner.buffer_i32(1i32);
    #[allow(non_snake_case)]
    let strideN = runner.buffer_i32(N as i32);

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = runner.buffer_zeros(B * N * 4);
        let st = runner.bench(
            rk,
            &[&inp, &out, &size, &stride1, &stride1, &strideN, &strideN],
            [B, 1, 1],
            [256, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let mt_msl = sort_msl().ok();
    let mk = mt_msl.as_deref().and_then(|msl| runner.compile(msl, "mt_sort_f32").ok());

    let equiv: Option<EquivResult> = mk.as_ref().map(|mk| {
        // Sort 4 small arrays of 1024 elements each; compare to CPU reference.
        let check_b = 4usize;
        let check_data: Vec<f32> = (0..check_b * N).map(|i| (check_b * N - i) as f32).collect();
        let ref_out = cpu_sort(&check_data, N);
        let check_inp = runner.buffer_f32(&check_data);
        let n_buf = runner.buffer_u32(N as u32);
        let check_out = runner.buffer_zeros(check_b * N * 4);
        runner.measure(mk, &[&check_inp, &check_out, &n_buf], [check_b, 1, 1], [256, 1, 1], 0, 1);
        let mt_out = runner.read_f32_slice(&check_out, check_b * N);
        let n_bad = ref_out.iter().zip(&mt_out).filter(|(a, b)| a != b).count();
        EquivResult {
            n_checked: check_b * N,
            max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
            passed: n_bad == 0,
        }
    });

    let n_buf = runner.buffer_u32(N as u32);
    let mt_perf = mk.as_ref().and_then(|mk| {
        let out = runner.buffer_zeros(B * N * 4);
        let st = runner.bench(mk, &[&inp, &out, &n_buf], [B, 1, 1], [256, 1, 1], 3, 10);
        to_gbps(&st, bytes)
    });

    let shape = format!("B={B} N={N} f32");
    let result = if let Some(mt_perf) = mt_perf {
        BENCH.implemented(shape, ref_perf, mt_perf, equiv.unwrap())
    } else {
        BENCH.nyi(shape, ref_perf)
    };
    vec![result]
}

/// CPU reference: sort each array of size `n` independently.
fn cpu_sort(data: &[f32], n: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    for chunk in out.chunks_mut(n) {
        chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_sort_msl_generates() {
        let msl = sort_msl().expect("codegen failed");
        assert!(msl.contains("mt_sort_f32"));
        assert!(msl.contains("threadgroup_barrier"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_sort_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        runner.compile(SRC, REF_NAME).unwrap_or_else(|e| panic!("{REF_NAME} compile error: {e}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_sort_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let rk = runner.compile(SRC, REF_NAME).expect("compile");
        let n = 64usize;
        let data: Vec<f32> = (0..n).rev().map(|i| i as f32).collect();
        let inp = runner.buffer_f32(&data);
        let out = runner.buffer_zeros(n * 4);
        let size = runner.buffer_i32(n as i32);
        let s1 = runner.buffer_i32(1i32);
        let sn = runner.buffer_i32(n as i32);
        runner.measure(&rk, &[&inp, &out, &size, &s1, &s1, &sn, &sn], [1, 1, 1], [256, 1, 1], 0, 1);
        let result = runner.read_f32_slice(&out, n);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, i as f32, "sort result wrong at {i}: got {v}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_sort_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = sort_msl().expect("codegen");
        runner
            .compile(&msl, "mt_sort_f32")
            .unwrap_or_else(|e| panic!("mt_sort_f32 compile error: {e}\nMSL:\n{msl}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_sort_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = sort_msl().expect("codegen");
        let mk = runner.compile(&msl, "mt_sort_f32").expect("compile");
        let n = N;
        let data: Vec<f32> = (0..n).rev().map(|i| i as f32).collect();
        let ref_out = cpu_sort(&data, n);
        let inp = runner.buffer_f32(&data);
        let n_buf = runner.buffer_u32(n as u32);
        let out_buf = runner.buffer_zeros(n * 4);
        runner.measure(&mk, &[&inp, &out_buf, &n_buf], [1, 1, 1], [256, 1, 1], 0, 1);
        let mt_out = runner.read_f32_slice(&out_buf, n);
        for (i, (&r, &m)) in ref_out.iter().zip(&mt_out).enumerate() {
            assert_eq!(r, m, "sort mismatch at {i}: ref={r} mt={m}");
        }
    }
}
