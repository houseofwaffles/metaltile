//! Sliding-window + sink-token perf bench for `ffai::sdpa_decode`.
//!
//! Companion to `sdpa_decode_gpu_correctness.rs`. The correctness file
//! pins that the dense path (`sink_end = 0, window_start = 0`) matches
//! the pre-SWA kernel's CPU naive reference within fp32 tolerance, and
//! that the SWA bound split matches a masked naive reference. This
//! file measures the resulting decode speedup at Qwen3-class GQA
//! shapes and the long-context regimes where sliding window is
//! actually deployed (industry SWA config: `window = 4096` over
//! `n_kv ∈ {8192, 16384, 32768}`).
//!
//! Ignored by default. Run manually:
//!
//!     cargo test --release -p metaltile-std --test sdpa_decode_swa_gpu \
//!         -- --ignored --nocapture
//!
//! Reports median GPU µs and effective GB/s (computed from the
//! *attended* KV bytes, not the full cache — sliding window doesn't
//! touch the masked range, so the bandwidth metric reflects what the
//! kernel actually does). Q / K / V are uploaded once per shape via
//! [`Context::upload_resident`] and bound through the
//! [`DispatchSpec::resident`] map — same pattern as the
//! `sdpa_decode_2pass_gpu.rs` chained+resident bench. Each dispatch
//! reuses the resident handles, so the measured µs reflects pure
//! kernel time, not per-call host→GPU traffic.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, pack_bytes, ramp};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer};
use metaltile_std::ffai::sdpa_decode::sdpa_decode;

// (n_q_heads, n_kv_heads, n_kv, window_size, sink_tokens). Qwen3-class
// GQA shape (32 Q heads / 8 KV heads) at the long-context regimes where
// sliding window is deployed. Window = 4096 mirrors the industry
// SWA configs; the sink-tokens column mirrors the "attention sink"
// findings (4 is the canonical count from Xiao et al. 2023).
const SHAPES_SWA: &[(usize, usize, usize, usize, usize)] =
    &[(32, 8, 8192, 4096, 4), (32, 8, 16384, 4096, 4), (32, 8, 32768, 4096, 4)];

const WARMUP_ITERS: usize = 20;
const MEASURED_ITERS: usize = 100;

struct DispatchCfg {
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    sink_end: u32,
    window_start: u32,
    scale: f32,
}

fn build_scalar_buffers(cfg: &DispatchCfg, dt: Dt) -> BTreeMap<String, Vec<u8>> {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("out".into(), vec![0u8; cfg.n_q_heads * cfg.head_dim * dt.bytes()]);
    b.insert("head_dim".into(), (cfg.head_dim as u32).to_le_bytes().to_vec());
    b.insert("n_kv".into(), (cfg.n_kv as u32).to_le_bytes().to_vec());
    b.insert("kv_stride".into(), (cfg.kv_stride as u32).to_le_bytes().to_vec());
    b.insert("heads_per_group".into(), (cfg.heads_per_group as u32).to_le_bytes().to_vec());
    b.insert("sink_end".into(), cfg.sink_end.to_le_bytes().to_vec());
    b.insert("window_start".into(), cfg.window_start.to_le_bytes().to_vec());
    b.insert("scale".into(), cfg.scale.to_le_bytes().to_vec());
    b
}

fn bench_one(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    cfg: &DispatchCfg,
    qkv: &BTreeMap<String, ResidentBuffer>,
    empty_fn_consts: &BTreeMap<String, u32>,
    dt: Dt,
) -> f64 {
    let buffers = build_scalar_buffers(cfg, dt);
    let spec = DispatchSpec {
        kernel,
        buffers: &buffers,
        fn_consts: empty_fn_consts,
        grid_groups: [cfg.n_q_heads, 1, 1],
        threads_per_group: [1024, 1, 1],
        resident: qkv,
    };
    let specs = std::slice::from_ref(&spec);

    let mut samples = Vec::with_capacity(MEASURED_ITERS);
    for i in 0..(WARMUP_ITERS + MEASURED_ITERS) {
        let r = ctx.dispatch_chain(specs).expect("dispatch_chain should succeed");
        if i >= WARMUP_ITERS {
            samples.push(r[0].elapsed_us);
        }
    }
    let mid = samples.len() / 2;
    *samples.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
}

fn run_dense_vs_swa(label: &str, dt: Dt) {
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let dtype = dt.to_dtype();
    let dt_bytes = dt.bytes();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    let empty_fn_consts: BTreeMap<String, u32> = BTreeMap::new();

    // Cold GPU DVFS gave the first-printed shape a ~2× bandwidth deficit
    // in the resident-buffer regime (was hidden by host→GPU memcpy
    // overhead in the pre-resident bench). Burn one dummy shape worth
    // of dispatches before the measured loop to pin the GPU clock.
    {
        let (n_q_heads, n_kv_heads, n_kv, ..) = SHAPES_SWA[0];
        let kv_stride = n_kv;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
        let q_res = ctx.upload_resident(&pack_bytes(&q, dt)).expect("preheat q");
        let k_res = ctx.upload_resident(&pack_bytes(&k, dt)).expect("preheat k");
        let v_res = ctx.upload_resident(&pack_bytes(&v, dt)).expect("preheat v");
        let mut qkv: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        qkv.insert("q".into(), q_res);
        qkv.insert("k".into(), k_res);
        qkv.insert("v".into(), v_res);
        let cfg = DispatchCfg {
            n_q_heads,
            head_dim,
            n_kv,
            kv_stride,
            heads_per_group: n_q_heads / n_kv_heads,
            sink_end: 0,
            window_start: 0,
            scale,
        };
        let _ = bench_one(&ctx, &kernel, &cfg, &qkv, &empty_fn_consts, dt);
    }

    println!();
    println!("{label} — Apple M-series (median of {MEASURED_ITERS} iters, head_dim=128, gqa=4)");
    println!(
        "  {:>5} {:>6} {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "n_kv", "window", "sinks", "dense µs", "SWA µs", "dense GB/s", "SWA GB/s", "speedup"
    );

    for &(n_q_heads, n_kv_heads, n_kv, window, sinks) in SHAPES_SWA {
        let kv_stride = n_kv;
        let heads_per_group = n_q_heads / n_kv_heads;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

        // Q/K/V are static across all 240 dispatches per shape (120 dense
        // + 120 SWA). Upload once into GPU-resident handles; the per-call
        // dispatch path then skips the per-iter host→GPU alloc + memcpy
        // for these inputs. Mirrors `sdpa_decode_2pass_gpu.rs` chained
        // +resident pattern.
        let q_res = ctx.upload_resident(&pack_bytes(&q, dt)).expect("upload q");
        let k_res = ctx.upload_resident(&pack_bytes(&k, dt)).expect("upload k");
        let v_res = ctx.upload_resident(&pack_bytes(&v, dt)).expect("upload v");
        let mut qkv: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        qkv.insert("q".into(), q_res);
        qkv.insert("k".into(), k_res);
        qkv.insert("v".into(), v_res);

        // Sliding window keeps sinks + the last `window` positions.
        // `window_start = max(sink_end, n_kv - window)` — saturating
        // sub guards short-context regimes where window >= n_kv.
        let sink_end = sinks as u32;
        let window_start = n_kv.saturating_sub(window).max(sinks) as u32;
        let attended = (window_start as usize..n_kv).len() + sinks;

        let dense_cfg = DispatchCfg {
            n_q_heads,
            head_dim,
            n_kv,
            kv_stride,
            heads_per_group,
            sink_end: 0,
            window_start: 0,
            scale,
        };
        let swa_cfg = DispatchCfg {
            n_q_heads,
            head_dim,
            n_kv,
            kv_stride,
            heads_per_group,
            sink_end,
            window_start,
            scale,
        };

        let dense_us = bench_one(&ctx, &kernel, &dense_cfg, &qkv, &empty_fn_consts, dt);
        let swa_us = bench_one(&ctx, &kernel, &swa_cfg, &qkv, &empty_fn_consts, dt);

        // Bandwidth model: Q + K + V + O, with K/V sized by the
        // *attended* position count for the SWA row (the kernel
        // doesn't touch masked positions, so charging full n_kv would
        // inflate the GB/s figure and hide actual register/SLC stall
        // patterns). Dense row uses full n_kv.
        let dense_bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
                * dt_bytes;
        let swa_bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * attended * head_dim + n_q_heads * head_dim)
                * dt_bytes;
        let dense_gbps = (dense_bytes as f64) / (dense_us * 1e-6) / 1e9;
        let swa_gbps = (swa_bytes as f64) / (swa_us * 1e-6) / 1e9;
        let speedup = dense_us / swa_us;

        println!(
            "  {:>5} {:>6} {:>5}  {:>9.2}  {:>9.2}  {:>9.1}  {:>9.1}  {:>6.2}x",
            n_kv, window, sinks, dense_us, swa_us, dense_gbps, swa_gbps, speedup,
        );
    }
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_f32() {
    run_dense_vs_swa("sdpa_decode SWA f32 (dense vs sliding-window+sinks)", Dt::F32);
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_f16() {
    run_dense_vs_swa("sdpa_decode SWA f16 (dense vs sliding-window+sinks)", Dt::F16);
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_bf16() {
    run_dense_vs_swa("sdpa_decode SWA bf16 (dense vs sliding-window+sinks)", Dt::Bf16);
}
