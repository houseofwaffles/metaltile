#![allow(
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::doc_lazy_continuation
)]

//! GPU correctness for `ffai::moe::mt_moe_gather_qmm_int4`.
//!
//! Matches MLX's `gatherQuantizedMM` (called by `SwitchLinear` →
//! `SwitchGLU` in mlx-swift-lm) at the cell level: per-row expert routing
//! + int4-quantized per-expert weight matmul. Verifies against a
//! full-precision CPU oracle that does the same routing + matmul in f32.
//!
//! Tolerance is wide (1e-1 abs) because the int4 quantization itself is a
//! lossy approximation of the f32 reference; the kernel just has to match
//! its own dequant + matmul (cosine ≥ 0.99 vs the dequant-and-matmul CPU
//! oracle).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{
    mt_moe_gather_qmm_int4,
    mt_moe_gather_qmm_int4_m8,
    mt_moe_gather_qmm_mma_int4,
    mt_moe_gather_qmm_mma_int4_bm16,
};

/// Pack a row of int4 weights into uint32s (8 per uint, LSB-first per nibble).
fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    assert!(weights.len() % 8 == 0);
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

/// CPU oracle: per-row, look up expert via expert_offsets, dequantize
/// weight row, dot against input row.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_qmm_int4(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    expert_offsets: &[u32],
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t_rows * m_out];
    let weight_stride_m = k_in / 8;
    let groups_per_row = k_in / group_size;

    for row in 0..t_rows {
        // Resolve expert: first e where row < expert_offsets[e+1].
        let mut expert = 0;
        for e in 0..n_experts {
            if (row as u32) < expert_offsets[e + 1] {
                expert = e;
                break;
            }
        }

        for m in 0..m_out {
            let weight_row_base = expert * m_out * weight_stride_m + m * weight_stride_m;
            let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
            let x_row_base = row * k_in;

            let mut acc = 0.0_f32;
            for pack_idx in 0..(k_in / 8) {
                let packed = weight_packed[weight_row_base + pack_idx];
                let k_first = pack_idx * 8;
                let g = k_first / group_size;
                let scale = scales[scale_row_base + g];
                let bias = biases[scale_row_base + g];
                for nib in 0..8 {
                    let q = (packed >> (nib * 4)) & 0xf;
                    let w = q as f32 * scale + bias;
                    let xv = x[x_row_base + k_first + nib];
                    acc += w * xv;
                }
            }
            out[row * m_out + m] = acc;
        }
    }
    out
}

/// Make a small but realistic test case: 3 experts, hidden=32, m_out=8,
/// group_size=32 (one group per row), 6 rows distributed across experts.
#[test]
fn moe_gather_qmm_int4_matches_cpu_oracle_f32() {
    let _g = gpu_lock();
    let n_experts = 3usize;
    let k_in = 32usize;
    let m_out = 8usize;
    let group_size = 32usize;
    let t_rows = 6usize;

    // Expert offsets: rows [0..2) → expert 0, [2..5) → expert 1, [5..6) → expert 2.
    let expert_offsets: Vec<u32> = vec![0, 2, 5, 6];

    // Random-ish quantized weights, scales, biases.
    let mut weight_unpacked = vec![0u32; n_experts * m_out * k_in];
    for (i, w) in weight_unpacked.iter_mut().enumerate() {
        *w = (((i as u32) * 7 + 3) & 0xf) as u32;
    }
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    let scales: Vec<f32> =
        (0..n_experts * m_out * (k_in / group_size)).map(|i| 0.01 + 0.001 * (i as f32)).collect();
    let biases: Vec<f32> =
        (0..n_experts * m_out * (k_in / group_size)).map(|i| -0.05 + 0.002 * (i as f32)).collect();

    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.1 * ((i as f32 * 0.17).sin())).collect();

    let y_cpu = cpu_gather_qmm_int4(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    // Run on GPU.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("weight_packed".into(), {
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect()
    });
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert("expert_offsets".into(), {
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    });
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("mt_moe_gather_qmm_int4 dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);

    let max_diff = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-4, "max |y_cpu - y_gpu| = {max_diff:.2e}");
}

/// Realistic Qwen3.6-35B-A3B layer shape: K_in=2048, M_out=256,
/// N_experts=128, group_size=64. 4 routed tokens across 3 experts (token
/// count tiny for test speed; kernel handles arbitrary T via grid_y).
#[test]
fn moe_gather_qmm_int4_qwen36_shape_f32() {
    let _g = gpu_lock();
    let n_experts = 128usize;
    let k_in = 2048usize;
    let m_out = 256usize;
    let group_size = 64usize;
    let t_rows = 4usize;

    // Most experts have zero rows; a handful own all the work.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    // Rows 0..2 → expert 7, rows 2..3 → expert 42, row 3 → expert 100.
    for e in 0..=n_experts {
        let off = if e <= 7 {
            0
        } else if e <= 42 {
            2
        } else if e <= 100 {
            3
        } else {
            t_rows as u32
        };
        expert_offsets[e] = off;
    }

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

    let y_cpu = cpu_gather_qmm_int4(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("weight_packed".into(), {
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect()
    });
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert("expert_offsets".into(), {
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    });
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("mt_moe_gather_qmm_int4 qwen36-shape dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);

    // Tighter cosine check at production shape — large reductions can
    // accumulate fp noise.
    let mut dot = 0.0_f64;
    let mut nc = 0.0_f64;
    let mut ng = 0.0_f64;
    for (a, b) in y_cpu.iter().zip(&y_gpu) {
        dot += (*a as f64) * (*b as f64);
        nc += (*a as f64) * (*a as f64);
        ng += (*b as f64) * (*b as f64);
    }
    let cos = dot / (nc.sqrt() * ng.sqrt() + 1e-12);
    assert!(cos >= 0.999, "cosine vs CPU oracle = {cos:.6} (want ≥ 0.999)");
}

/// m8 variant matches m1 (the original scalar) at Qwen3.6 shape.
/// Both implement the same dot product so output should agree to fp noise.
#[test]
fn moe_gather_qmm_int4_m8_matches_m1_qwen36_shape_f32() {
    let _g = gpu_lock();
    let n_experts = 128usize;
    let k_in = 2048usize;
    let m_out = 256usize;
    let group_size = 64usize;
    let t_rows = 4usize;

    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for e in 0..=n_experts {
        let off = if e <= 7 {
            0
        } else if e <= 42 {
            2
        } else if e <= 100 {
            3
        } else {
            t_rows as u32
        };
        expert_offsets[e] = off;
    }

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

    fn run(
        kernel_ir: impl FnOnce(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
        grid_x: usize,
        x: &[f32],
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        expert_offsets: &[u32],
        t_rows: usize,
        k_in: usize,
        m_out: usize,
        n_experts: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = kernel_ir(Dt::F32.to_dtype());
        kernel.mode = KernelMode::Reduction;
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, t_rows, 1], [
                32, 1, 1,
            ])
            .expect("dispatch");
        unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32)
    }

    let y_m1 = run(
        mt_moe_gather_qmm_int4::kernel_ir_for,
        m_out,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );
    let y_m8 = run(
        mt_moe_gather_qmm_int4_m8::kernel_ir_for,
        m_out / 8,
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let max_diff = y_m1.iter().zip(&y_m8).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_diff < 5e-4, "m8 vs m1 max diff = {max_diff:.2e}");
}

/// MMA variant — single expert (gather logic becomes trivial). Pins
/// the inner MMA / dequant path independently of expert resolution.
#[test]
fn moe_gather_qmm_mma_int4_single_expert() {
    let _g = gpu_lock();
    let n_experts = 1usize;
    let k_in = 32usize;
    let n_out = 32usize;
    let group_size = 32usize;
    let t_rows = 32usize;

    let indices: Vec<u32> = vec![0; t_rows];
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let expert_offsets: Vec<u32> = vec![0, t_rows as u32];
    let y_m1 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let y_mma = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_mma_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out / 32, t_rows.div_ceil(32), 1],
                [128, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (a, b) in y_m1.iter().zip(&y_mma) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("y_m1[0..8] = {:?}", &y_m1[..8]);
    eprintln!("y_mma[0..8] = {:?}", &y_mma[..8]);
    assert!(cos >= 0.999, "single-expert MMA vs m1 cosine = {cos:.6} (want ≥ 0.999)");
}

/// MMA variant correctness: matches scalar m1 at a clean shape.
/// Tile contract: T must be multiple of 32, N multiple of 32, K multiple of 32.
#[test]
fn moe_gather_qmm_mma_int4_matches_m1_clean_tile() {
    let _g = gpu_lock();
    let n_experts = 4usize;
    let k_in = 64usize; // mult of 32
    let n_out = 64usize; // mult of 32 (=2 n-tiles of BN=32)
    let group_size = 32usize;
    let t_rows = 64usize; // mult of 32 (=2 m-tiles of BM=32)

    // Per-row expert indices, sorted: rows 0..16 → e0, 16..32 → e1, 32..48 → e2, 48..64 → e3.
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    // Reference: m1 (already validated against CPU oracle). Build the
    // expert_offsets array m1 needs from the per-row indices.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // Run m1 (uses expert_offsets).
    let y_m1 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // Run MMA (uses per-row indices).
    let y_mma = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_mma_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        // Grid: [N/32, ceil(T/32), 1], TG: [128, 1, 1] (4 SGs)
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out / 32, t_rows.div_ceil(32), 1],
                [128, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // MMA tiled vs scalar reduction — fp accumulation order differs. Use cosine.
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (a, b) in y_m1.iter().zip(&y_mma) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    assert!(cos >= 0.999, "MMA vs m1 cosine = {cos:.6} (want ≥ 0.999)");
}

/// BM=16 MMA variant matches scalar m1 at a clean shape.
/// Tile contract: T multiple of 16, N multiple of 32, K multiple of 32.
#[test]
fn moe_gather_qmm_mma_int4_bm16_matches_m1_clean_tile() {
    let _g = gpu_lock();
    let n_experts = 4usize;
    let k_in = 64usize;
    let n_out = 64usize; // BN=32 → 2 n-tiles
    let group_size = 32usize;
    let t_rows = 64usize; // BM=16 → 4 m-tiles

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    let y_m1 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let y_bm16 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        // Grid: [N/32, ceil(T/16), 1], TG: [64, 1, 1] (2 SGs)
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out / 32, t_rows.div_ceil(16), 1],
                [64, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (a, b) in y_m1.iter().zip(&y_bm16) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("y_m1[0..8] = {:?}", &y_m1[..8]);
    eprintln!("y_bm16[0..8] = {:?}", &y_bm16[..8]);
    assert!(cos >= 0.999, "BM=16 MMA vs m1 cosine = {cos:.6} (want ≥ 0.999)");
}
