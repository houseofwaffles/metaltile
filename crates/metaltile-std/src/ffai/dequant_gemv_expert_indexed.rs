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

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="dequant_gemv_expert_indexed",
    subop="int4",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
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
