//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Per-expert indexed int4 dequantizing GEMV.
//!
//! Variant of `dequant_gemv_int4` where the weight / scale / bias tensors
//! are **stacked across experts** and the kernel reads which expert to
//! index from a GPU-resident `expert_index: Tensor<u32>` buffer at
//! runtime. The standard `dequant_gemv_int4_<T>` kernel uses the same
//! row-of-output / threadgroup model and an identical pack-strided inner
//! loop; the only difference is two additional per-row offsets
//! (`expert * out_dim * n_packs_per_row` for the weight, and
//! `expert * out_dim * n_groups` for the scale/bias) computed from the
//! expert index loaded once per threadgroup at lane 0.
//!
//! ## Why a separate kernel
//!
//! The standalone `dequant_gemv_int4` operates on a single expert's
//! weights at a known CPU-determined offset (the caller picks the
//! `Tensor` view into the slot's weight slab). Moving the per-slot
//! decision from CPU to GPU lets FFAI eliminate the
//! `cmd.commit + waitUntilCompleted` it does today to ship the gate
//! logits to the host for routing — instead, a GPU router writes the
//! top-K expert indices into a buffer + this kernel reads them.
//!
//! ## Memory layout
//!
//! For `n_experts` experts each with `[out_dim, in_dim/8]` u32-packed
//! int4 weight (and `[out_dim, in_dim/group_size]` scales + biases):
//!
//!   weights_stacked  [n_experts, out_dim, in_dim / 8]   uint32
//!   scales_stacked   [n_experts, out_dim, in_dim / G]   T
//!   biases_stacked   [n_experts, out_dim, in_dim / G]   T
//!   input            [in_dim]                           T
//!   expert_index     [1]                                u32
//!   output           [out_dim]                          T
//!
//! Loading the expert id once per row (instead of per-pack) keeps the
//! inner loop identical to `dequant_gemv_int4` — just offset the row
//! base pointers by `expert * <row count> * <stride>`.
//!
//! ## Dispatch
//!
//! One threadgroup per output row (same as `dequant_gemv_int4`):
//!   grid       = MTLSize(out_dim, 1, 1)
//!   threadgroup = MTLSize(32, 1, 1)  // pinned in Reduction mode
//!
//! ## Correctness invariant
//!
//! At greedy decode this kernel is bit-identical to dispatching the
//! standard `dequant_gemv_int4` with the matching expert's `Tensor`
//! views — same threadgroup geometry, same reduction order, same
//! reduce_sum tree. FFAI's `MoELayer` end-to-end equivalence tests
//! cover both paths once the GPU router is wired.

use metaltile::kernel;

#[kernel]
pub fn dequant_gemv_int4_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<T>,
    biases_stacked: Tensor<T>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    // Per-row offsets — int4 packs 8 values per u32.
    let vals_per_pack = 8u32;
    let mask = 0xFu32;
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    // INVARIANT: expert_index[0] is in [0, n_experts). Caller (FFAI's
    // MoE router kernel) writes one expert id per top-K slot; we read
    // the slot's id once into a register and stride by the
    // per-expert weight/scale span.
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_groups;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_group_off = scale_expert_off + row * n_groups;
    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[row_group_off + g]).cast::<f32>();
            let bias = load(biases_stacked[row_group_off + g]).cast::<f32>();
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                acc = acc + (q.cast::<f32>() * scale + bias) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// New-syntax correctness test for `dequant_gemv_int4_expert_indexed` — the
/// per-expert-indexed int4 dequant GEMV. Reduction-mode (one threadgroup per
/// output row, `reduce_sum` across the threadgroup).
///
/// Oracle: stack `n_experts` int4-quantized `[out_dim, in_dim]` weight slabs
/// (pack-strided, 8 nibbles per u32), pick a non-zero expert index, then replay
/// `output[row] = Σ_i (q[expert,row,i]·scale_g + bias_g)·input[i]` in f32 for the
/// selected expert. Verifies the expert-stride offset math is wired correctly.
/// Inputs are dtype-rounded so the GPU sees exactly what the oracle does.
///
/// Grid: `grid_3d(out_dim, 1, 1, [tpg, 1, 1])` — one TG per output row, tpg = 64.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::dequant_gemv_int4_expert_indexed;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// 64 lanes per output row (≥ 32, multiple of 32 — Reduction contract).
    const TPG: u32 = 64;

    /// Synthesize int4 (8-nibbles-per-u32 pack-strided) weights for a stack of
    /// `n_experts` `[out_dim, in_dim]` slabs.
    fn synth_stacked_w(n_experts: usize, out_dim: usize, in_dim: usize) -> Vec<u32> {
        let pf = 8usize; // 8 int4 codes per u32
        let mask = 0xFu32;
        let packs_per_row = in_dim / pf;
        let mut packed = vec![0u32; n_experts * out_dim * packs_per_row];
        for e in 0..n_experts {
            for row in 0..out_dim {
                let row_base = (e * out_dim + row) * packs_per_row;
                for d in 0..in_dim {
                    let code = ((e * out_dim * in_dim + row * in_dim + d) as u32)
                        .wrapping_mul(2_654_435_761)
                        & mask;
                    packed[row_base + d / pf] |= code << ((d % pf) as u32 * 4);
                }
            }
        }
        packed
    }

    /// Dequant-then-dot reference for the selected expert. `weight` stacks
    /// `[n_experts, out_dim, in_dim]` int4 codes; `scales`/`biases` stack
    /// `[n_experts, out_dim, in_dim/group_size]`.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        input: &[f32],
        expert: usize,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let pf = 8usize;
        let mask = 0xFu32;
        let packs_per_row = in_dim / pf;
        let n_groups = in_dim / group_size;
        let w_expert_off = expert * out_dim * packs_per_row;
        let sb_expert_off = expert * out_dim * n_groups;
        let mut out = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut acc = 0.0f32;
            for (i, &x_i) in input.iter().enumerate().take(in_dim) {
                let g = i / group_size;
                let word = weight[w_expert_off + row * packs_per_row + i / pf];
                let q = ((word >> ((i % pf) as u32 * 4)) & mask) as f32;
                acc += (q * scales[sb_expert_off + row * n_groups + g]
                    + biases[sb_expert_off + row * n_groups + g])
                    * x_i;
            }
            out[row] = acc;
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4_expert_indexed(dt: DType) -> TestSetup {
        let (n_experts, out_dim, in_dim, group_size) = (4usize, 4usize, 256usize, 64usize);
        let n_groups = in_dim / group_size;
        let expert = 2usize; // exercise a non-zero expert stride
        let w = synth_stacked_w(n_experts, out_dim, in_dim);
        let scales_f: Vec<f32> =
            (0..n_experts * out_dim * n_groups).map(|i| 0.004 + (i % 7) as f32 * 0.0008).collect();
        let biases_f: Vec<f32> =
            (0..n_experts * out_dim * n_groups).map(|i| ((i % 5) as f32 - 2.0) * 0.0009).collect();
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = oracle(&w, &s, &b, &x, expert, out_dim, in_dim, group_size);
        TestSetup::new(dequant_gemv_int4_expert_indexed::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weights_stacked", u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales_stacked", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases_stacked", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("expert_index", u32_bytes(&[expert as u32]), DType::U32))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d(out_dim as u32, 1, 1, [TPG, 1, 1])
    }
}

/// New-syntax benchmark for `dequant_gemv_int4_expert_indexed`. Production-ish
/// shape (out_dim/in_dim 4096, group_size 64, 8 experts). One TG per output row.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::dequant_gemv_int4_expert_indexed;

    #[bench(name = "ffai/dequant_gemv_expert_indexed/int4", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int4_expert_indexed(dt: DType) -> BenchSetup {
        let (n_experts, out_dim, in_dim, group_size) = (8usize, 4096usize, 4096usize, 64usize);
        let n_groups = in_dim / group_size;
        let packs_per_row = in_dim / 8;
        let sz = dt.size_bytes();
        // Active stream: one expert's weight slab + its scales/biases + input + output.
        let bytes =
            out_dim * packs_per_row * 4 + 2 * out_dim * n_groups * sz + in_dim * sz + out_dim * sz;
        BenchSetup::new(dequant_gemv_int4_expert_indexed::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random(
                "weights_stacked",
                n_experts * out_dim * packs_per_row,
                DType::U32,
            ))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("expert_index", 1, DType::U32))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
