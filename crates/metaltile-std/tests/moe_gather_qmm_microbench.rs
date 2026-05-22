//! Microbench: `mt_moe_gather_qmm_int4` at Qwen3.6-35B-A3B shape.
//!
//! `#[ignore]`-gated — run with:
//!   cargo test -p metaltile-std --test moe_gather_qmm_microbench --release -- --ignored --nocapture

#![cfg(target_os = "macos")]

mod common;

use std::{collections::BTreeMap, time::Instant};

use common::{Dt, gpu_lock, pack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{
    mt_moe_gather_qmm_int4,
    mt_moe_gather_qmm_int4_m8,
    mt_moe_gather_qmm_mma_int4,
    mt_moe_gather_qmm_mma_int4_bm16,
};

fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

#[derive(Copy, Clone)]
enum Variant {
    M1,
    M8,
    Mma,
    MmaBm16,
}

#[allow(clippy::too_many_arguments)]
fn time_gather_qmm(
    ctx: &Context,
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
) -> f64 {
    time_gather_qmm_vd(ctx, t_rows, k_in, m_out, n_experts, group_size, iters, Variant::M1, Dt::F32)
}

#[allow(clippy::too_many_arguments)]
fn time_gather_qmm_v(
    ctx: &Context,
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    variant: Variant,
) -> f64 {
    time_gather_qmm_vd(ctx, t_rows, k_in, m_out, n_experts, group_size, iters, variant, Dt::F32)
}

#[allow(clippy::too_many_arguments)]
fn time_gather_qmm_vd(
    ctx: &Context,
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    variant: Variant,
    dt: Dt,
) -> f64 {
    // Distribute T_rows evenly across N_experts so the kernel does real work.
    let rows_per_expert = t_rows / n_experts;
    let mut expert_offsets: Vec<u32> =
        (0..=n_experts).map(|e| (e * rows_per_expert) as u32).collect();
    expert_offsets[n_experts] = t_rows as u32;

    let total_weights = n_experts * m_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    let groups_total = n_experts * m_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * ((i as f32 * 0.03).sin())).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * ((i as f32 * 0.07).cos())).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * ((i as f32 * 0.013).sin())).collect();

    // Per-row indices (for MMA kernel which uses rhs_indices). Built from the
    // CSR-style expert_offsets the m1/m8 kernels consume.
    let mut indices: Vec<u32> = Vec::with_capacity(t_rows);
    for e in 0..n_experts {
        let s = expert_offsets[e];
        let e_end = expert_offsets[e + 1];
        for _ in s..e_end {
            indices.push(e as u32);
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, dt));
    let w_bytes: Vec<u8> = weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffers.insert("weight_packed".into(), w_bytes.clone());
    buffers.insert("w".into(), w_bytes);
    buffers.insert("scales".into(), pack_bytes(&scales, dt));
    buffers.insert("biases".into(), pack_bytes(&biases, dt));
    buffers.insert(
        "expert_offsets".into(),
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
    );
    buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], dt));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
    buffers.insert("t_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());

    let mut kernel = match variant {
        Variant::M1 => mt_moe_gather_qmm_int4::kernel_ir_for(dt.to_dtype()),
        Variant::M8 => mt_moe_gather_qmm_int4_m8::kernel_ir_for(dt.to_dtype()),
        Variant::Mma => mt_moe_gather_qmm_mma_int4::kernel_ir_for(dt.to_dtype()),
        Variant::MmaBm16 => mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for(dt.to_dtype()),
    };
    kernel.mode = KernelMode::Reduction;
    let (grid, tg) = match variant {
        Variant::M1 => ([m_out, t_rows, 1], [32usize, 1, 1]),
        Variant::M8 => ([m_out / 8, t_rows, 1], [32, 1, 1]),
        Variant::Mma => ([m_out / 32, t_rows.div_ceil(32), 1], [128, 1, 1]),
        Variant::MmaBm16 => ([m_out / 32, t_rows.div_ceil(16), 1], [64, 1, 1]),
    };

    let _ =
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, tg).expect("dispatch");

    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, tg)
            .expect("dispatch");
    }
    start.elapsed().as_micros() as f64 / iters as f64
}

#[ignore]
#[test]
fn bench_qwen36_a3b_moe_layer_shape() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");

    // Qwen3.6-35B-A3B per-layer MoE FFN projection shapes.
    // Per-projection params: K_in × M_out × n_experts:
    //   gate_proj:  hidden=2048 → moe_intermediate=256 (per expert)
    //   up_proj:    hidden=2048 → moe_intermediate=256
    //   down_proj:  moe_intermediate=256 → hidden=2048
    //
    // At T=1024 (a moderate-prefill window) × topk=8 = 8192 routed tokens.
    let t_rows = 1024; // matches MLX bench script (T*topk after permute)
    let n_experts = 128;
    let group_size = 64;
    let iters = 3;

    for (k_in, m_out, label) in [(2048, 256, "gate/up"), (256, 2048, "down")] {
        let us_m1 =
            time_gather_qmm_v(&ctx, t_rows, k_in, m_out, n_experts, group_size, iters, Variant::M1);
        let us_m8 =
            time_gather_qmm_v(&ctx, t_rows, k_in, m_out, n_experts, group_size, iters, Variant::M8);
        let us_mma = time_gather_qmm_v(
            &ctx,
            t_rows,
            k_in,
            m_out,
            n_experts,
            group_size,
            iters,
            Variant::Mma,
        );
        let flops = (t_rows * m_out * k_in * 2) as f64;
        let gf_m1 = flops / us_m1 / 1e3;
        let gf_m8 = flops / us_m8 / 1e3;
        let gf_mma = flops / us_mma / 1e3;
        eprintln!(
            "Qwen3.6-A3B {label:>8} K={k_in} M={m_out} T={t_rows}: \
             m1={us_m1:>7.0}us ({gf_m1:>5.0} GF) \
             m8={us_m8:>7.0}us ({gf_m8:>5.0} GF) \
             mma={us_mma:>7.0}us ({gf_mma:>5.0} GF) \
             mma-vs-m8={:.2}×",
            us_m8 / us_mma
        );
    }
}

#[ignore]
#[test]
fn bench_qwen36_a3b_moe_layer_shape_f16() {
    // Same shapes as the F32 bench but at fp16 — matches MLX's gather_qmm
    // bench dtype. Apple GPUs run f16 MMA at 2× the f32 rate, so the gap
    // to MLX should close substantially here.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let t_rows = 1024;
    let n_experts = 128;
    let group_size = 64;
    let iters = 3;

    for (k_in, m_out, label) in [(2048, 256, "gate/up"), (256, 2048, "down")] {
        let us_m8 = time_gather_qmm_vd(
            &ctx,
            t_rows,
            k_in,
            m_out,
            n_experts,
            group_size,
            iters,
            Variant::M8,
            Dt::F16,
        );
        let us_mma = time_gather_qmm_vd(
            &ctx,
            t_rows,
            k_in,
            m_out,
            n_experts,
            group_size,
            iters,
            Variant::Mma,
            Dt::F16,
        );
        let us_bm16 = time_gather_qmm_vd(
            &ctx,
            t_rows,
            k_in,
            m_out,
            n_experts,
            group_size,
            iters,
            Variant::MmaBm16,
            Dt::F16,
        );
        let flops = (t_rows * m_out * k_in * 2) as f64;
        let gf_m8 = flops / us_m8 / 1e3;
        let gf_mma = flops / us_mma / 1e3;
        let gf_bm16 = flops / us_bm16 / 1e3;
        eprintln!(
            "Qwen3.6-A3B-F16 {label:>8} K={k_in} M={m_out} T={t_rows}: \
             m8={us_m8:>7.0}us ({gf_m8:>5.0} GF) \
             mma={us_mma:>7.0}us ({gf_mma:>5.0} GF) \
             bm16={us_bm16:>7.0}us ({gf_bm16:>5.0} GF) \
             bm16-vs-mma={:.2}× bm16-vs-m8={:.2}×",
            us_mma / us_bm16,
            us_m8 / us_bm16
        );
    }
}

#[ignore]
#[test]
fn bench_qwen36_a3b_short_prefill() {
    // Decode-shape: T=1 step × topk=8 routed tokens.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let t_rows = 8;
    let n_experts = 128;
    let group_size = 64;
    let iters = 50;
    for (k_in, m_out, label) in [(2048, 256, "gate/up"), (256, 2048, "down")] {
        let us = time_gather_qmm(&ctx, t_rows, k_in, m_out, n_experts, group_size, iters);
        eprintln!("Decode-shape {label} T={t_rows} K={k_in} M={m_out}: {us:.1}us");
    }
}
