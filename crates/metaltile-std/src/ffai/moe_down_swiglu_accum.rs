//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused MoE decode kernel: SwiGLU + 8-way indexed down-projection + scalar-FMA chain.
//!
//! Collapses the three back-to-back dispatches FFAI's MoE GPU-router
//! decode path runs per layer (under `FFAI_MOE_GPU_ROUTER`, default-on
//! as of ITER 80) into ONE kernel launch:
//!
//!   1. `mt_swiglu` (many=8): `inner[k][d] = silu(gate[k][d]) * up[k][d]`
//!   2. `ffai_dequant_gemv_int4_expert_indexed` (many=8): per slot k,
//!      `down_out[k] = W_down[expert[k]] · inner[k]`  (out_dim = hidden)
//!   3. `mt_scalar_fma_chain8`:
//!      `acc[i] = Σ_{k=0..8} scalar[k] * down_out[k][i]`
//!
//! The three stages have a strict data dependency chain so they MUST
//! run on separate encoders today. Fusing them eliminates two encoder
//! begin/end pairs + the global memory round-trip on `inner` (8 × 768
//! floats per layer × 40 layers = ~960 KB DRAM/token at Qwen3.6-A3B)
//! AND removes the materialisation of `down_out` (8 × 2048 floats per
//! layer × 40 layers = ~2.5 MB DRAM/token).
//!
//! ## Geometry
//!
//! - One threadgroup per output row of the down projection.
//!   grid       = MTLSize(out_dim, 1, 1)        // out_dim = hidden
//!   threadgroup = MTLSize(lsize, 1, 1)         // caller picks (typ. 128)
//!
//! - Each TG iterates the 8 slots sequentially:
//!   (a) cooperatively populate `tg_inner[d] = silu(gate[k][d]) * up[k][d]`
//!   for d in `[0, in_dim)`, every thread writes a strided slice.
//!   (b) threadgroup_barrier (RAW: qmm reads `tg_inner` filled by all
//!   lanes in (a)).
//!   (c) dequant-gemv inner loop against `W_down[expert[k]][row, :]`,
//!   each thread accumulating into a per-thread `acc` that runs
//!   across ALL slots (with slot scalar baked in at accumulation).
//!   (d) threadgroup_barrier (WAR: next slot's swiglu overwrites
//!   `tg_inner` and must not race with this slot's qmm reads).
//!
//!   After 8 slots, `reduce_sum(acc)` and lane 0 stores to `out[row]`.
//!
//! ## Threadgroup memory
//!
//! `tg_inner: [IN_DIM_MAX] f32`, staged inner activations for the
//! currently-active slot. Reused across all 8 slots (no per-slot copy).
//!
//! IN_DIM_MAX is pinned at 768 (Qwen3.6-A3B `moeIntermediate`). At f32,
//! that's 3 KiB TG memory, comfortably below the 32 KiB Apple9 cap, and
//! leaves headroom for concurrent TGs on the same simdgroup-block.
//! Caller MUST validate `in_dim <= 768`; the kernel reads only the first
//! `in_dim` entries so smaller intermediates work, but larger ones
//! would scribble past the alloc.
//!
//! ## ABI
//!
//! ```text
//!   gate_0..gate_7    [in_dim]                          T
//!   up_0..up_7        [in_dim]                          T
//!   expert_indices    [8]                               u32
//!   slot_weights      [8]                               T   (routing scalars)
//!   weights_stacked   [n_experts, out_dim, in_dim / 8]  u32 (int4-packed)
//!   scales_stacked    [n_experts, out_dim, in_dim / G]  T
//!   biases_stacked    [n_experts, out_dim, in_dim / G]  T
//!   out               [out_dim]                         T
//! ```
//!
//! `expert_indices` and `slot_weights` are the contiguous outputs of
//! `mt_moe_router_topk` (k=8, packed). The 16 gate/up tensors mirror
//! FFAI's per-slot scratch caches: one `Tensor.empty([moeIntermediate])`
//! per slot, instance-cached per `MoELayer` per the ITER 32-36
//! scratch-caching rule (see CLAUDE.md / MEMORY.md).
//!
//! ## Correctness invariant
//!
//! At greedy decode this kernel is mathematically equivalent (modulo
//! floating-point reorder of the per-thread reduction) to:
//!
//!   for k in 0..8: tmp_k = mt_swiglu(gate_k, up_k)
//!   for k in 0..8: down_k = ffai_dequant_gemv_int4_expert_indexed(
//!                              W, S, B, tmp_k, expert_indices[k:k+1])
//!   out = mt_scalar_fma_chain8(slot_weights[0:1], down_0, ...,
//!                              slot_weights[7:8], down_7)
//!
//! Tolerance budget: 1e-3 (f32), 5e-2 (bf16 / f16). The reduce_sum at
//! the tail fuses 8 slots' partial-sums in ONE simd reduction (vs 8
//! separate reductions in the unfused chain), so the rounding tree is
//! shallower; we err on the side of higher precision, not lower.
//!
//! ## Why this fusion is safe
//!
//! Each threadgroup is independent (one output row each). The
//! `tg_inner` scratch is private to its TG. Cross-slot ordering inside
//! a TG is enforced by two barriers per slot (RAW on fill→read, WAR on
//! read→next-fill). Per-thread `acc` is private; the final
//! `reduce_sum` (Reduction mode) handles the cross-thread fold.
//!
//! ## Why NOT register-resident `inner`
//!
//! Tempting alternative: each thread holds its `inner_k` slice in
//! registers and reads it during qmm without TG mem. Doesn't work,
//! the qmm pack-stride pattern means thread `t` needs `inner_k[d]`
//! values at offsets `pack_idx*8 + i` for i in 0..8, which are
//! neighbours, not strided slots. Threads can't share registers.
//! TG memory is the right abstraction here. 3 KiB is cheap.
//!
//! ## Source-level dedup via `define_kernel!` macro
//!
//! The 8 slot bodies are byte-for-byte identical modulo the 4 per-slot
//! identifiers (`gate_k`, `up_k`, `exp_k`, `sw_k`). To avoid 8 hand-
//! copies of a ~50-line block, we wrap the ENTIRE `#[kernel] fn`
//! declaration in a `macro_rules!` that takes 8 slot tuples and
//! expands the per-slot body via `$(...)*` repetition. Macro
//! expansion happens at Rust compile time BEFORE the `#[kernel]`
//! proc-macro parses the body, so the emitted IR + MSL are byte-
//! identical to the 8 hand-unrolled blocks.
//!
//! NB: the proc-macro's `body_parser` explicitly rejects
//! `macro_rules!` invocations INSIDE a kernel body (see
//! `metaltile-macros/src/body_parser.rs:210`) — they'd silently
//! produce no IR. The "wrap the whole fn" pattern is the supported
//! workaround called out in that same error message.

use metaltile::kernel;

/// Build the fused MoE down+SwiGLU+chain8 kernel from 8 slot tuples.
///
/// Each tuple = `($gate, $up, $exp, $sw, $we, $se, $rpo, $rgo, $trailing)`:
/// - `$gate`, `$up`: kernel param idents for the slot's gate/up
///   activations
/// - `$exp`, `$sw`: local idents that hold the slot's expert index
///   and routing scalar (declared in the kernel prologue)
/// - `$we`, `$se`, `$rpo`, `$rgo`: per-slot unique idents for the
///   weight-expert / scale-expert / row-pack-offset / row-group-offset
///   locals. Passing them in (instead of declaring `let we = ...`
///   inside the macro body) gives each slot a distinct C-level name,
///   so the emitted MSL is byte-identical to the hand-unrolled version
///   (verified via `tile build --emit` diff across f32/f16/bf16).
/// - `$trailing`: either `{ threadgroup_barrier(); }` (slots 0..6,
///   WAR barrier before next slot overwrites `tg_inner`) or `{}`
///   (slot 7, no further `tg_inner` access after this slot).
///
/// All other identifiers (`tg_inner`, `acc`, `row`, `n_packs_per_row`,
/// `n_groups`, `packs_per_group`, `vals_per_pack`, `mask`, `p_iters`,
/// `in_iters`, `weights_stacked`, `scales_stacked`, `biases_stacked`,
/// `out_dim`, `in_dim`, `lsize`, `tid`) are shared kernel scope and
/// captured by name from the surrounding body.
macro_rules! define_moe_down_swiglu_accum_chain8 {
    (
        $(
            (
                $gate:ident, $up:ident, $exp:ident, $sw:ident,
                $we:ident, $se:ident, $rpo:ident, $rgo:ident,
                $trailing:tt
            )
        ),* $(,)?
    ) => {
        #[kernel(
            bench(
                op="moe_down_swiglu_accum",
                subop="int4_chain8",
                class=GenericEmpty,
                tol=0.0,
                kernel_mode=Reduction,
            )
        )]
        #[allow(clippy::too_many_arguments)]
        pub fn ffai_moe_down_swiglu_accum_int4_chain8<T>(
            gate_0: Tensor<T>,
            up_0: Tensor<T>,
            gate_1: Tensor<T>,
            up_1: Tensor<T>,
            gate_2: Tensor<T>,
            up_2: Tensor<T>,
            gate_3: Tensor<T>,
            up_3: Tensor<T>,
            gate_4: Tensor<T>,
            up_4: Tensor<T>,
            gate_5: Tensor<T>,
            up_5: Tensor<T>,
            gate_6: Tensor<T>,
            up_6: Tensor<T>,
            gate_7: Tensor<T>,
            up_7: Tensor<T>,
            expert_indices: Tensor<u32>,
            slot_weights: Tensor<T>,
            weights_stacked: Tensor<u32>,
            scales_stacked: Tensor<T>,
            biases_stacked: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] group_size: u32,
        ) {
            // Threadgroup scratch for the active slot's inner activations.
            // 768 = Qwen3.6-A3B moeIntermediate. Bump the literal (and
            // re-validate Apple9 TG-mem ceiling) if a future model needs a
            // larger intermediate. f32 stage so the qmm consumer reads at
            // accumulation precision.
            threadgroup_alloc("tg_inner", 768, "f32");

            // Int4 dequant constants, match `dequant_gemv_int4_expert_indexed`.
            let vals_per_pack = 8u32;
            let mask = 0xFu32;
            let row = program_id::<0>();
            let n_packs_per_row = in_dim / vals_per_pack;
            let n_groups = in_dim / group_size;
            let packs_per_group = group_size / vals_per_pack;

            // Pre-load all 8 expert indices and slot weights into registers.
            // Cheap: 8 u32 loads + 8 T loads; reused across the slot loop.
            let exp_0 = load(expert_indices[0u32]);
            let exp_1 = load(expert_indices[1u32]);
            let exp_2 = load(expert_indices[2u32]);
            let exp_3 = load(expert_indices[3u32]);
            let exp_4 = load(expert_indices[4u32]);
            let exp_5 = load(expert_indices[5u32]);
            let exp_6 = load(expert_indices[6u32]);
            let exp_7 = load(expert_indices[7u32]);
            let sw_0 = load(slot_weights[0u32]).cast::<f32>();
            let sw_1 = load(slot_weights[1u32]).cast::<f32>();
            let sw_2 = load(slot_weights[2u32]).cast::<f32>();
            let sw_3 = load(slot_weights[3u32]).cast::<f32>();
            let sw_4 = load(slot_weights[4u32]).cast::<f32>();
            let sw_5 = load(slot_weights[5u32]).cast::<f32>();
            let sw_6 = load(slot_weights[6u32]).cast::<f32>();
            let sw_7 = load(slot_weights[7u32]).cast::<f32>();

            // Running per-thread accumulator across all 8 slots. Each slot's
            // contribution = slot_weight * Σ_packs (q*s+b) * tg_inner[d].
            // Final reduce_sum fuses the 8 slots' partials in one fold.
            let mut acc = 0.0f32;

            // Iteration counts, same shape as the indexed-expert dequant-gemv.
            let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
            let in_iters = (in_dim + lsize - 1u32) / lsize;

            // Expand one slot body per tuple. Each body:
            //   (a) Cooperatively fill tg_inner with silu($gate) * $up.
            //   (b) RAW barrier.
            //   (c) Dequant-gemv inner loop, accumulating into `acc`
            //       with $sw baked in.
            //   (d) $trailing: WAR barrier for slots 0..6, empty for
            //       slot 7 (no further tg_inner access).
            //
            // The per-slot `$we / $se / $rpo / $rgo` idents are supplied
            // by the caller so each slot gets a distinct C-level local
            // (matches the hand-unroll exactly — keeps emit byte-equal).
            $(
                for s_iter in range(0u32, in_iters, 1u32) {
                    let d = s_iter * lsize + tid;
                    if d < in_dim {
                        let g = load($gate[d]).cast::<f32>();
                        let u = load($up[d]).cast::<f32>();
                        // Inline silu in f32, same form as
                        // gated_rmsnorm.rs and swiglu.rs. Avoids
                        // T→f32→T round-trip and keeps the gate
                        // precise before the multiply.
                        let s = g / (1.0f32 + exp(0.0f32 - g));
                        threadgroup_store("tg_inner", d, s * u);
                    }
                }
                threadgroup_barrier();
                let $we = $exp * out_dim * n_packs_per_row;
                let $se = $exp * n_groups * out_dim;
                let $rpo = $we + row * n_packs_per_row;
                let $rgo = $se + row * n_groups;
                for p_iter in range(0u32, p_iters, 1u32) {
                    let pack_idx = p_iter * lsize + tid;
                    if pack_idx < n_packs_per_row {
                        let g = pack_idx / packs_per_group;
                        let scale = load(scales_stacked[$rgo + g]).cast::<f32>();
                        let bias = load(biases_stacked[$rgo + g]).cast::<f32>();
                        let packed = load(weights_stacked[$rpo + pack_idx]);
                        let p_off = pack_idx * vals_per_pack;
                        for i in range(0u32, vals_per_pack, 1u32) {
                            let q = (packed >> (i * 4u32)) & mask;
                            let dq = q.cast::<f32>() * scale + bias;
                            let inner_v = threadgroup_load("tg_inner", p_off + i);
                            acc = acc + $sw * dq * inner_v;
                        }
                    }
                }
                $trailing
            )*

            // ── Cross-thread fold + store ────────────────────────────
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}

define_moe_down_swiglu_accum_chain8!(
    (gate_0, up_0, exp_0, sw_0, we_0, se_0, rpo_0, rgo_0, {
        threadgroup_barrier();
    }),
    (gate_1, up_1, exp_1, sw_1, we_1, se_1, rpo_1, rgo_1, {
        threadgroup_barrier();
    }),
    (gate_2, up_2, exp_2, sw_2, we_2, se_2, rpo_2, rgo_2, {
        threadgroup_barrier();
    }),
    (gate_3, up_3, exp_3, sw_3, we_3, se_3, rpo_3, rgo_3, {
        threadgroup_barrier();
    }),
    (gate_4, up_4, exp_4, sw_4, we_4, se_4, rpo_4, rgo_4, {
        threadgroup_barrier();
    }),
    (gate_5, up_5, exp_5, sw_5, we_5, se_5, rpo_5, rgo_5, {
        threadgroup_barrier();
    }),
    (gate_6, up_6, exp_6, sw_6, we_6, se_6, rpo_6, rgo_6, {
        threadgroup_barrier();
    }),
    (gate_7, up_7, exp_7, sw_7, we_7, se_7, rpo_7, rgo_7, {}),
);

/// New-syntax benchmark for the fused MoE decode kernel
/// (`ffai_moe_down_swiglu_accum_int4_chain8`). Bench-only: the 8-way
/// SwiGLU + indexed int4 down-projection + scalar-FMA-chain fusion has no
/// clean single-stage oracle — its end-to-end correctness is validated in
/// FFAI integration tests and `tests/moe_down_swiglu_accum_gpu_correctness.rs`
/// against the unfused 3-stage chain.
///
/// Geometry: one threadgroup per down-projection output row.
/// Grid (Reduction): `grid_3d(out_dim, 1, 1, [lsize,1,1])` (lsize = 128).
/// ABI: 8 × (gate, up) activation tensors, `expert_indices[8]`,
/// `slot_weights[8]`, stacked int4 weights/scales/biases, `output[out_dim]`
/// + `{in_dim, out_dim, group_size}`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_down_swiglu_accum_int4_chain8;

    /// Lanes per threadgroup — the caller-picked `lsize` (typically 128).
    const LSIZE: u32 = 128;
    /// Top-k slot count this kernel fuses (8-way chain).
    const N_SLOTS: usize = 8;

    #[bench(name = "ffai/moe_down_swiglu_accum/int4_chain8", dtypes = [f32, f16, bf16])]
    fn bench_moe_down_swiglu_accum_int4_chain8(dt: DType) -> BenchSetup {
        // Qwen3.6-A3B-ish: hidden=2048 (out_dim), moeIntermediate=768 (in_dim).
        let n_experts = 128usize;
        let in_dim = 768usize;
        let out_dim = 2048usize;
        let group_size = 64usize;
        let n_groups = in_dim / group_size;
        let packs_per_row = in_dim / 8;
        let sz = dt.size_bytes();
        // Active stream per token: 8 × (gate + up) inner reads, the touched
        // experts' weight slab (approximate with full slab), scales/biases,
        // and the single output row. Weight slab dominates.
        let bytes = (2 * N_SLOTS * in_dim) * sz
            + n_experts * out_dim * packs_per_row * 4
            + 2 * n_experts * out_dim * n_groups * sz
            + out_dim * sz;

        let mut bs = BenchSetup::new(ffai_moe_down_swiglu_accum_int4_chain8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction);
        // 8 gate/up activation pairs.
        for k in 0..N_SLOTS {
            bs = bs
                .buffer(BenchBuffer::random(&format!("gate_{k}"), in_dim, dt))
                .buffer(BenchBuffer::random(&format!("up_{k}"), in_dim, dt));
        }
        bs.buffer(BenchBuffer::zeros("expert_indices", N_SLOTS, DType::U32))
            .buffer(BenchBuffer::random("slot_weights", N_SLOTS, dt))
            .buffer(BenchBuffer::random(
                "weights_stacked",
                n_experts * out_dim * packs_per_row,
                DType::U32,
            ))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "in{in_dim} out{out_dim} E{n_experts} k{N_SLOTS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(out_dim as u32, 1, 1, [LSIZE, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
