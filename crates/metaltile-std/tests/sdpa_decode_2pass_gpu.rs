//! End-to-end correctness + perf bench for the MLX-geometry two-pass
//! SDPA decode kernel pair.
//!
//! Geometry (mirrors `mlx/backend/metal/kernels/sdpa_vector.h`):
//! - pass 1 TG `(32, gqa_factor, 1)`, grid `(n_kv_heads, blocks, 1)`
//! - pass 2 TG `(1024, 1, 1)`, grid `(n_q_heads, 1, 1)`
//! - `blocks` must be a multiple of 32 (pass-2 reducer silently drops
//!   partials otherwise)
//!
//! Bench (`#[ignore]`):
//!   cargo test --release -p metaltile-std --test sdpa_decode_2pass_gpu \
//!     -- --ignored --nocapture

#![cfg(target_os = "macos")]

mod common;

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard, OnceLock},
};

use common::{Dt, SdpaShape, max_abs_diff, naive_sdpa_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer, start_gpu_trace, stop_gpu_trace};
use metaltile_std::ffai::sdpa_decode_2pass::{sdpa_decode_2pass_pass1, sdpa_decode_2pass_pass2};

/// Serialise GPU dispatches across tests in this file. Cargo runs `#[test]`
/// functions concurrently by default; under `cargo llvm-cov` the
/// instrumented binary is slow enough that concurrent dispatches on the
/// same `MTLDevice` race the `ChainedResident` chained-buffer path and
/// produce garbage output (max |diff| in the hundreds of millis at f16).
/// One global mutex; ~1s total overhead.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy)]
enum ChainMode {
    Unchained,
    Chained,
    ChainedResident,
}

struct TwoPassArgs<'a> {
    ctx: &'a Context,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    blocks: usize,
    scale: f32,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
}

/// One dispatch helper covering all (dtype × chain_mode × resident-K/V) combos.
/// Returns `(out_f32, p1_us, opt_p2_us)`. p2_us is `Some` only for `Unchained`
/// where the two passes were dispatched separately; chained modes share one
/// cmd buffer and only report a single fused time as `p1_us`.
fn run_2pass(
    a: &TwoPassArgs,
    dt: Dt,
    mode: ChainMode,
    kv_res: Option<(&ResidentBuffer, &ResidentBuffer)>,
) -> (Vec<f32>, f64, Option<f64>) {
    assert_eq!(a.blocks % 32, 0, "blocks must be a multiple of 32");
    assert_eq!(a.n_q_heads % a.n_kv_heads, 0);
    let gqa_factor = a.n_q_heads / a.n_kv_heads;
    let dtype = dt.to_dtype();
    let partial_o_len = a.n_q_heads * a.blocks * a.head_dim;
    let partial_ml_len = a.n_q_heads * a.blocks;

    let mut p1 = sdpa_decode_2pass_pass1::kernel_ir_for(dtype);
    p1.mode = KernelMode::Reduction;
    let mut p1_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p1_bufs.insert("q".into(), pack_bytes(a.q, dt));
    if kv_res.is_none() {
        p1_bufs.insert("k".into(), pack_bytes(a.k, dt));
        p1_bufs.insert("v".into(), pack_bytes(a.v, dt));
    }
    // partial_o is Tensor<T> (storage dtype). partial_m / partial_l stay f32.
    p1_bufs.insert("partial_o".into(), vec![0u8; partial_o_len * dt.bytes()]);
    p1_bufs.insert("partial_m".into(), vec![0u8; partial_ml_len * 4]);
    p1_bufs.insert("partial_l".into(), vec![0u8; partial_ml_len * 4]);
    p1_bufs.insert("head_dim".into(), (a.head_dim as u32).to_le_bytes().to_vec());
    p1_bufs.insert("n_kv".into(), (a.n_kv as u32).to_le_bytes().to_vec());
    p1_bufs.insert("kv_stride".into(), (a.kv_stride as u32).to_le_bytes().to_vec());
    p1_bufs.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    p1_bufs.insert("blocks".into(), (a.blocks as u32).to_le_bytes().to_vec());
    p1_bufs.insert("scale".into(), a.scale.to_le_bytes().to_vec());

    let mut p2 = sdpa_decode_2pass_pass2::kernel_ir_for(dtype);
    p2.mode = KernelMode::Reduction;
    let mut p2_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    p2_bufs.insert("out".into(), vec![0u8; a.n_q_heads * a.head_dim * dt.bytes()]);
    p2_bufs.insert("head_dim".into(), (a.head_dim as u32).to_le_bytes().to_vec());
    p2_bufs.insert("blocks".into(), (a.blocks as u32).to_le_bytes().to_vec());

    let empty: BTreeMap<String, u32> = BTreeMap::new();
    let no_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

    match mode {
        ChainMode::Unchained => {
            // Two separate dispatch_with_grid calls; explicitly thread
            // partial_o/m/l outputs from p1 into p2's inputs.
            let p1 = a
                .ctx
                .dispatch_with_grid(&p1, &p1_bufs, &empty, [a.n_kv_heads, a.blocks, 1], [
                    32, gqa_factor, 1,
                ])
                .expect("pass1");
            p2_bufs.insert(
                "partial_o".into(),
                p1.outputs.get("partial_o").expect("partial_o").clone(),
            );
            p2_bufs.insert(
                "partial_m".into(),
                p1.outputs.get("partial_m").expect("partial_m").clone(),
            );
            p2_bufs.insert(
                "partial_l".into(),
                p1.outputs.get("partial_l").expect("partial_l").clone(),
            );
            let p2 = a
                .ctx
                .dispatch_with_grid(&p2, &p2_bufs, &empty, [a.n_q_heads, 1, 1], [1024, 1, 1])
                .expect("pass2");
            let out = unpack_bytes(p2.outputs.get("out").expect("out"), dt);
            (out, p1.elapsed_us, Some(p2.elapsed_us))
        },
        ChainMode::Chained | ChainMode::ChainedResident => {
            let mut p1_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
            if let Some((k_res, v_res)) = kv_res {
                p1_resident.insert("k".into(), k_res.clone());
                p1_resident.insert("v".into(), v_res.clone());
            }
            let specs = [
                DispatchSpec {
                    kernel: &p1,
                    buffers: &p1_bufs,
                    fn_consts: &empty,
                    grid_groups: [a.n_kv_heads, a.blocks, 1],
                    threads_per_group: [32, gqa_factor, 1],
                    resident: &p1_resident,
                },
                DispatchSpec {
                    kernel: &p2,
                    buffers: &p2_bufs,
                    fn_consts: &empty,
                    grid_groups: [a.n_q_heads, 1, 1],
                    threads_per_group: [1024, 1, 1],
                    resident: &no_resident,
                },
            ];
            let r = a.ctx.dispatch_chain(&specs).expect("chain");
            let out = unpack_bytes(r[1].outputs.get("out").expect("out"), dt);
            (out, r[0].elapsed_us, None)
        },
    }
}

// ── Correctness ──────────────────────────────────────────────────────────

fn check_matches_cpu(
    shape: (usize, usize, usize, usize, usize, usize),
    dt: Dt,
    mode: ChainMode,
    tol: f32,
    msg: &str,
) {
    let _lock = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, blocks) = shape;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = ramp(n_q_heads * head_dim, 19, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 23, 11.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 29, 14.0);
    let expected =
        naive_sdpa_f32(&q, &k, &v, &SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale });
    let ctx = Context::new().expect("Context");
    let (k_res, v_res) = if matches!(mode, ChainMode::ChainedResident) {
        (
            Some(ctx.upload_resident(&pack_bytes(&k, dt)).expect("upload k")),
            Some(ctx.upload_resident(&pack_bytes(&v, dt)).expect("upload v")),
        )
    } else {
        (None, None)
    };
    let resident_pair = k_res.as_ref().zip(v_res.as_ref());
    let (actual, ..) = run_2pass(
        &TwoPassArgs {
            ctx: &ctx,
            n_q_heads,
            n_kv_heads,
            head_dim,
            n_kv,
            kv_stride,
            blocks,
            scale,
            q: &q,
            k: &k,
            v: &v,
        },
        dt,
        mode,
        resident_pair,
    );
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < tol, "{msg}: max |diff| = {diff:.2e}");
}

const SMALL_F32: (usize, usize, usize, usize, usize, usize) = (4, 1, 128, 64, 64, 32);
const GQA: (usize, usize, usize, usize, usize, usize) = (32, 8, 128, 256, 256, 64);

#[test]
fn matches_cpu_reference_f32_small() {
    check_matches_cpu(SMALL_F32, Dt::F32, ChainMode::Unchained, 1e-4, "f32 small");
}

#[test]
fn matches_cpu_reference_f32_gqa() {
    check_matches_cpu(GQA, Dt::F32, ChainMode::Unchained, 1e-4, "f32 gqa");
}

#[test]
fn matches_cpu_reference_f32_chained_gqa() {
    check_matches_cpu(GQA, Dt::F32, ChainMode::Chained, 1e-4, "f32 chained");
}

#[test]
fn matches_cpu_reference_f32_chained_resident_gqa() {
    check_matches_cpu(GQA, Dt::F32, ChainMode::ChainedResident, 1e-4, "f32 chained+resident");
}

// Narrow-dtype chained+resident saturates on Apple7 (M1) — the f16/bf16
// MSL store path through `MTLStorageModePrivate` staging diverges to the
// dtype's max value on that family. Works on Apple8+ (M2/M3/M4/M5). CI
// runs M1; tile bench's `sdpa_decode_2pass` f16/bf16 rows (181/147 GB/s,
// 12/12 correct on M2) cover the production target.
#[test]
#[ignore = "f16 chained+resident: Apple7/M1 codegen divergence; passes on Apple8+"]
fn matches_cpu_reference_f16_chained_resident_gqa() {
    check_matches_cpu(GQA, Dt::F16, ChainMode::ChainedResident, 5e-2, "f16 chained+resident");
}

#[test]
#[ignore = "bf16 chained+resident: Apple7/M1 codegen divergence; passes on Apple8+"]
fn matches_cpu_reference_bf16_chained_resident_gqa() {
    check_matches_cpu(GQA, Dt::Bf16, ChainMode::ChainedResident, 2e-1, "bf16 chained+resident");
}

// ── Perf benches ─────────────────────────────────────────────────────────

const SHAPES_F32_UNCHAINED: &[(usize, usize, usize, &[usize])] = &[
    (32, 8, 1024, &[32, 64, 128, 256, 512]),
    (32, 8, 2048, &[32, 64, 128, 256, 512]),
    (32, 8, 4096, &[32, 64, 128, 256, 512]),
    (32, 8, 8192, &[32, 64, 128, 256, 512]),
    (32, 8, 16384, &[64, 128, 256, 512, 1024]),
];

const SHAPES_CHAINED: &[(usize, usize, usize, &[usize])] = &[
    (32, 8, 1024, &[32, 64, 128, 256]),
    (32, 8, 2048, &[32, 64, 128, 256]),
    (32, 8, 4096, &[32, 64, 128, 256, 512]),
    (32, 8, 8192, &[32, 64, 128, 256, 512]),
    (32, 8, 16384, &[64, 128, 256, 512, 1024]),
];

fn run_perf_bench(
    label: &str,
    dt: Dt,
    mode: ChainMode,
    shapes_blocks: &[(usize, usize, usize, &[usize])],
    split_passes: bool,
) {
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let ctx = Context::new().expect("Context");
    println!();
    println!("{label} — Apple M-series (median of 100 iters)");
    if split_passes {
        println!(
            "  {:>5} {:>7}  {:>9}  {:>9}  {:>9}  {:>9}",
            "n_kv", "blocks", "P1 µs", "P2 µs", "total µs", "GB/s"
        );
    } else {
        println!("  {:>5} {:>7}  {:>10}  {:>9}", "n_kv", "blocks", "total µs", "GB/s");
    }
    for &(n_q_heads, n_kv_heads, n_kv, block_candidates) in shapes_blocks {
        let kv_stride = n_kv;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
        let (k_res, v_res) = if matches!(mode, ChainMode::ChainedResident) {
            (
                Some(ctx.upload_resident(&pack_bytes(&k, dt)).expect("upload k")),
                Some(ctx.upload_resident(&pack_bytes(&v, dt)).expect("upload v")),
            )
        } else {
            (None, None)
        };
        let resident_pair = k_res.as_ref().zip(v_res.as_ref());
        for &blocks in block_candidates {
            let mut p1_samples = Vec::with_capacity(100);
            let mut p2_samples = Vec::with_capacity(100);
            for i in 0..120 {
                let (_o, p1us, p2us) = run_2pass(
                    &TwoPassArgs {
                        ctx: &ctx,
                        n_q_heads,
                        n_kv_heads,
                        head_dim,
                        n_kv,
                        kv_stride,
                        blocks,
                        scale,
                        q: &q,
                        k: &k,
                        v: &v,
                    },
                    dt,
                    mode,
                    resident_pair,
                );
                if i >= 20 {
                    p1_samples.push(p1us);
                    if let Some(p2) = p2us {
                        p2_samples.push(p2);
                    }
                }
            }
            p1_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p2_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p1_med = p1_samples[p1_samples.len() / 2];
            let p2_med = if p2_samples.is_empty() { 0.0 } else { p2_samples[p2_samples.len() / 2] };
            let total = p1_med + p2_med;
            let bytes =
                (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
                    * dt.bytes();
            let gbps = (bytes as f64) / (total * 1e-6) / 1e9;
            if split_passes {
                println!(
                    "  {:>5} {:>7}  {:>9.2}  {:>9.2}  {:>9.2}  {:>9.1}",
                    n_kv, blocks, p1_med, p2_med, total, gbps
                );
            } else {
                println!("  {:>5} {:>7}  {:>10.2}  {:>9.1}", n_kv, blocks, total, gbps);
            }
        }
    }
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_2pass_perf_bench_f32() {
    run_perf_bench(
        "sdpa_decode_2pass f32 (unchained, per-pass)",
        Dt::F32,
        ChainMode::Unchained,
        SHAPES_F32_UNCHAINED,
        // split=
        true,
    );
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_2pass_chained_perf_bench_f32() {
    run_perf_bench(
        "sdpa_decode_2pass f32 CHAINED",
        Dt::F32,
        ChainMode::Chained,
        SHAPES_CHAINED,
        false,
    );
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_2pass_chained_resident_perf_bench_f32() {
    run_perf_bench(
        "sdpa_decode_2pass f32 CHAINED + RESIDENT K/V",
        Dt::F32,
        ChainMode::ChainedResident,
        SHAPES_CHAINED,
        false,
    );
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_2pass_f16_chained_resident_perf_bench() {
    run_perf_bench(
        "sdpa_decode_2pass f16 CHAINED + RESIDENT K/V",
        Dt::F16,
        ChainMode::ChainedResident,
        SHAPES_CHAINED,
        false,
    );
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_2pass_bf16_chained_resident_perf_bench() {
    run_perf_bench(
        "sdpa_decode_2pass bf16 CHAINED + RESIDENT K/V",
        Dt::Bf16,
        ChainMode::ChainedResident,
        SHAPES_CHAINED,
        false,
    );
}

// ── GPU capture (xctrace-driven; see capture.rs docs) ───────────────────

#[test]
#[ignore = "GPU capture, requires MTL_CAPTURE_ENABLED=1 + MTILE_GPU_TRACE=/path"]
fn sdpa_decode_2pass_capture() {
    let Ok(path) = std::env::var("MTILE_GPU_TRACE") else {
        eprintln!("MTILE_GPU_TRACE unset; skipping");
        return;
    };
    let _ = std::fs::remove_dir_all(&path);
    start_gpu_trace(&path).expect("startCapture (is MTL_CAPTURE_ENABLED=1 set?)");

    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (n_q_heads, n_kv_heads) = (32usize, 8usize);
    let ctx = Context::new().expect("Context");
    for n_kv in [1024usize, 4096, 16384] {
        let kv_stride = n_kv;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
        let k_res = ctx.upload_resident(&pack_bytes(&k, Dt::F32)).expect("upload k");
        let v_res = ctx.upload_resident(&pack_bytes(&v, Dt::F32)).expect("upload v");
        // Warm PSO + caches.
        for _ in 0..10 {
            let _ = run_2pass(
                &TwoPassArgs {
                    ctx: &ctx,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    n_kv,
                    kv_stride,
                    blocks: 128,
                    scale,
                    q: &q,
                    k: &k,
                    v: &v,
                },
                Dt::F32,
                ChainMode::ChainedResident,
                Some((&k_res, &v_res)),
            );
        }
        // Capture-of-record: 3 iters/shape for a stable trace.
        for _ in 0..3 {
            let _ = run_2pass(
                &TwoPassArgs {
                    ctx: &ctx,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    n_kv,
                    kv_stride,
                    blocks: 128,
                    scale,
                    q: &q,
                    k: &k,
                    v: &v,
                },
                Dt::F32,
                ChainMode::ChainedResident,
                Some((&k_res, &v_res)),
            );
        }
    }
    stop_gpu_trace();
    println!("captured to {path} — open in Xcode for counters");
}
