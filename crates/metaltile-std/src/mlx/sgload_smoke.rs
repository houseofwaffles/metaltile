//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Smoke kernel for `simdgroup_load` HW intrinsic — first kernel to
//! actually use the `Op::SimdgroupLoad` DSL primitive end-to-end, so
//! future kernels (qmm B-load fast path etc.) have a working call
//! site to reference.
//!
//! What it does:
//!   1. 32-lane simdgroup stages a flat-64 input into TG memory
//!      (`tg_tile`, row-major 8×8, row-stride = 8).
//!   2. `simdgroup_load(frag, "tg_tile", 0, 8)` issues the HW-fused
//!      coalesced load — one MSL `simdgroup_load(...)` instruction
//!      lands the 8×8 tile into the simdgroup-matrix fragment.
//!   3. Per-lane fragment scatter writes the frag back to `dst` in
//!      the A/C lane convention, so for `f32` / `f16` the values
//!      round-trip byte-exactly.
//!
//! No math, no MMA — this is a plumbing test. If the round-trip
//! preserves values bit-for-bit, the parser → IR → codegen chain for
//! `Op::SimdgroupLoad` is correctly hooked up and the produced MSL
//! issues a real `simdgroup_load(...)` call against threadgroup
//! memory.
//!
//! Lane → element mapping is the **A/C convention** used everywhere
//! else in the codebase (see `mt_mma_probe_a_identity_b_identity`):
//!
//! ```text
//!   qid = lane / 4
//!   fm  = (qid & 4) + ((lane / 2) % 4)         ∈ 0..8
//!   fn0 = (qid & 2) * 2 + (lane % 2) * 2       ∈ 0..8 (even)
//!   fn1 = fn0 + 1                              ∈ 0..8 (odd)
//!   frag.elem[0] at (fm, fn0) ↔ tg_tile[fm*8 + fn0]
//!   frag.elem[1] at (fm, fn1) ↔ tg_tile[fm*8 + fn1]
//! ```
//!
//! Dispatch: grid `[1, 1, 1]`, tpg `[32, 1, 1]` (one simdgroup).
//!
//! Sample MSL the codegen produces (look for these in
//! `cargo run -p metaltile-cli -- inspect mt_sgload_smoke`):
//!
//! ```text
//!   threadgroup T tg_tile[64];
//!   ...
//!   simdgroup_matrix<T, 8, 8> frag;
//!   simdgroup_load(frag, &tg_tile[0u], 8, ulong2(0, 0), false);
//!   ...
//! ```

use metaltile::kernel;

/// Round-trip an 8×8 tile through TG memory + a simdgroup-matrix
/// fragment via the `simdgroup_load` HW intrinsic. f32 / f16 should
/// produce byte-exact equality between `src` and `dst`.
///
/// Inputs:
///   - `src`: `[64]` flat row-major 8×8 source values
/// Outputs:
///   - `dst`: `[64]` flat row-major 8×8 destination, written from
///     the fragment in A/C lane convention
#[kernel(
    bench(
        op="sgload",
        subop="smoke",
        class=GenericEmpty,
        tol=0.0,
        dtypes=&[DType::F32, DType::F16],
        kernel_mode=Reduction,
    )
)]
pub fn mt_sgload_smoke<T>(src: Tensor<T>, mut dst: Tensor<T>) {
    let lane = simd_lane;
    // A/C lane → frag-element mapping. Matches the probe kernel +
    // every MMA call site in `quantized.rs`. `fn0` / `fn1` are the
    // two consecutive columns this lane owns inside its `fm` row.
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // ── 1. Stage `src` into TG memory cooperatively ────────────────
    // 32 lanes × 2 elems = 64. Lane writes its two destination
    // positions (fm, fn0) and (fm, fn1) — both ∈ [0, 64) — directly,
    // so the staging layout matches the A/C convention the
    // `simdgroup_load` consumer expects.
    threadgroup_alloc("tg_tile", 64u32, T);
    let i0 = fm * 8u32 + fn0;
    let i1 = fm * 8u32 + fn1;
    threadgroup_store("tg_tile", i0, load(src[i0]));
    threadgroup_store("tg_tile", i1, load(src[i1]));
    // Single-simdgroup, so a compiler-only reorder fence (zero
    // runtime cost) is enough to publish the stores before the
    // HW-fused load.
    simdgroup_barrier_mem_none();
    // ── 2. HW-fused fragment load ──────────────────────────────────
    // One MSL `simdgroup_load(...)` instruction — the whole point of
    // this kernel. `offset = 0` (top-left of the tile), `stride = 8`
    // (row stride in elements), `transpose = false`.
    let frag = simdgroup_alloc::<T, 8, 8>();
    let off = 0u32;
    simdgroup_load(frag, "tg_tile", off, 8u32);
    simdgroup_barrier_mem_none();
    // ── 3. Scatter the fragment back out ───────────────────────────
    // A/C convention: lane (fm, fn0) holds tile[fm, fn0]. So writing
    // `dst[fm*8 + fn0] = frag.elem[0]` round-trips the input value
    // when no transpose was applied.
    let v0 = simdgroup_elem_load(frag, 0);
    let v1 = simdgroup_elem_load(frag, 1);
    store(dst[fm * 8u32 + fn0], v0);
    store(dst[fm * 8u32 + fn1], v1);
}
