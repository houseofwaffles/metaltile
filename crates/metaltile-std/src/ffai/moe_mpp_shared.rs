//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Shared test/bench helpers for the MPP MoE grouped BGEMM family.
//!
//! Every MPP MoE kernel (`moe_mpp{,_int8,_bm8,_bm8_int8,_bm64,_bm64_int8}`)
//! shares one ABI — `x, w, scales, biases, indices, out` plus the four
//! `{m_total, n_out, k_in, group_size}` constexprs — and the same math:
//! per-row expert routing via `indices[t]`, dequant-then-grouped-matmul.
//! Only the tile geometry (BM/BN/BK, SG count) and the weight bit-width
//! differ. These helpers centralise the int4 dequant oracle, the
//! per-variant `TestSetup`, and the per-variant `BenchSetup` so each
//! kernel file stays a thin shape-binding wrapper.

use metaltile::{core::ir::Kernel, test::*};

use crate::{
    bench_types::DType,
    utils::{pack_f32, unpack_f32},
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// Pack a row of int4 codes into u32s (8 nibbles per u32, LSB-first).
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

/// Per-row-`indices` int4 dequant-then-matmul reference. `weight_packed`
/// stacks `[n_experts, n_out, k_in/8]` int4 codes; `scales`/`biases` stack
/// `[n_experts, n_out, k_in/group_size]`. Row `t`'s expert is `indices[t]`.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_qmm_int4_indexed(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    indices: &[u32],
    m_total: usize,
    k_in: usize,
    n_out: usize,
    group_size: usize,
) -> Vec<f32> {
    let weight_stride_m = k_in / 8;
    let groups_per_row = k_in / group_size;
    let mut out = vec![0.0f32; m_total * n_out];
    for row in 0..m_total {
        let expert = indices[row] as usize;
        for n in 0..n_out {
            let weight_row_base = expert * n_out * weight_stride_m + n * weight_stride_m;
            let scale_row_base = expert * n_out * groups_per_row + n * groups_per_row;
            let x_row_base = row * k_in;
            let mut acc = 0.0f32;
            for pack_idx in 0..(k_in / 8) {
                let packed = weight_packed[weight_row_base + pack_idx];
                let k_first = pack_idx * 8;
                let g = k_first / group_size;
                let scale = scales[scale_row_base + g];
                let bias = biases[scale_row_base + g];
                for nib in 0..8 {
                    let q = ((packed >> (nib * 4)) & 0xf) as f32;
                    acc += (q * scale + bias) * x[x_row_base + k_first + nib];
                }
            }
            out[row * n_out + n] = acc;
        }
    }
    out
}

/// Test shape for an int4 MMA/MPP variant. Dims chosen per the variant's
/// tile contract (all clean multiples; no edge padding).
pub struct MmaTestShape {
    pub n_experts: usize,
    pub m_total: usize,
    pub n_out: usize,
    pub k_in: usize,
    pub group_size: usize,
}

/// Build a `TestSetup` for an int4 indexed-MMA/MPP kernel. `bn`/`bm` give the
/// tile dims (grid `[n_out/bn, ceil(m_total/bm), 1]`), `tpg` the threadgroup
/// width (lanes). Reduction mode for all.
pub fn int4_indexed_setup(
    kernel: Kernel,
    shape: MmaTestShape,
    bn: u32,
    bm: u32,
    tpg: u32,
    dt: DType,
) -> TestSetup {
    let MmaTestShape { n_experts, m_total, n_out, k_in, group_size } = shape;

    // Per-row expert indices, sorted (post-permute layout).
    let indices: Vec<u32> = (0..m_total).map(|r| (r / (m_total / n_experts)) as u32).collect();

    let mut weight_unpacked = vec![0u32; n_experts * n_out * k_in];
    for (i, w) in weight_unpacked.iter_mut().enumerate() {
        *w = ((i as u32) * 7 + 3) & 0xf;
    }
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    let n_groups = k_in / group_size;
    let scales_f: Vec<f32> = (0..n_experts * n_out * n_groups)
        .map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin())
        .collect();
    let biases_f: Vec<f32> = (0..n_experts * n_out * n_groups)
        .map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos())
        .collect();
    let x_f: Vec<f32> = (0..m_total * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
    let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
    let x = unpack_f32(&pack_f32(&x_f, dt), dt);
    let expected = cpu_gather_qmm_int4_indexed(
        &x,
        &weight_packed,
        &s,
        &b,
        &indices,
        m_total,
        k_in,
        n_out,
        group_size,
    );

    TestSetup::new(kernel)
        .mode(KernelMode::Reduction)
        .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
        .input(TestBuffer::from_vec("w", u32_bytes(&weight_packed), DType::U32))
        .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
        .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
        .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
        .input(TestBuffer::zeros("out", m_total * n_out, dt))
        .constexpr("m_total", m_total as u32)
        .constexpr("n_out", n_out as u32)
        .constexpr("k_in", k_in as u32)
        .constexpr("group_size", group_size as u32)
        .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
        .grid_3d(n_out as u32 / bn, (m_total as u32).div_ceil(bm), 1, [tpg, 1, 1])
}

/// Bench shape for an int4/int8 MMA/MPP variant. `bits` selects the packed
/// word count (`k_in*bits/32` u32 per row).
pub struct MmaBenchShape {
    pub bits: u32,
    pub bn: u32,
    pub bm: u32,
    pub tpg: u32,
    pub m_total: usize,
    pub n_out: usize,
    pub k_in: usize,
    pub n_experts: usize,
    pub group_size: usize,
}

/// Build a `BenchSetup` for an indexed-MMA/MPP kernel. Production-ish shape;
/// `bytes_moved` counts the full expert weight slab (dominant) + scales/biases
/// + x + out.
pub fn int4_mma_bench(kernel: Kernel, shape: MmaBenchShape, dt: DType) -> BenchSetup {
    let MmaBenchShape { bits, bn, bm, tpg, m_total, n_out, k_in, n_experts, group_size } = shape;
    let groups_per_row = k_in / group_size;
    let words_per_row = k_in * bits as usize / 32;
    let sz = dt.size_bytes();
    let bytes = n_experts * n_out * words_per_row * 4
        + 2 * n_experts * n_out * groups_per_row * sz
        + m_total * k_in * sz
        + m_total * n_out * sz;
    BenchSetup::new(kernel)
        .mode(KernelMode::Reduction)
        .buffer(BenchBuffer::random("x", m_total * k_in, dt))
        .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
        .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
        .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
        .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
        .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
        .constexpr("m_total", m_total as u32)
        .constexpr("n_out", n_out as u32)
        .constexpr("k_in", k_in as u32)
        .constexpr("group_size", group_size as u32)
        .with_shape_label(format!(
            "M{m_total} N{n_out} K{k_in} E{n_experts} {}",
            crate::bench_types::dtype_label(dt)
        ))
        .grid_3d(n_out as u32 / bn, (m_total as u32).div_ceil(bm), 1, [tpg, 1, 1])
        .bytes_moved(bytes as u64)
}
