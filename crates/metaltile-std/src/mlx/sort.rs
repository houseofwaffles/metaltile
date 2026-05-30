//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Sort benchmark — #[kernel] DSL vs MLX metal/sort.metal
//!
//! Two stages cover arrays larger than one threadgroup, mirroring MLX's
//! `block_sort` + `mb_block_merge` multi-block sort:
//!
//!   1. **`mt_sort<T>`** — single-block bitonic sort. Each threadgroup
//!      sorts its own `n = 1024`-element block in shared memory. For an
//!      array of `n_blocks * 1024` elements this leaves `n_blocks`
//!      independently-sorted runs of length 1024.
//!
//!   2. **`mt_merge<T>`** — one bottom-up merge pass. Given sorted runs
//!      of length `run`, it merges each adjacent pair into a sorted run
//!      of length `2 * run`. Running it for `log2(n_blocks)` passes
//!      (`run` = 1024, 2048, 4096, …) collapses every per-block run into
//!      one fully-sorted array. Caller ping-pongs two buffers between
//!      passes.
//!
//! The merge is a per-output-element **merge-path / co-rank** merge:
//! every thread owns one output slot, binary-searches the diagonal to
//! find how many elements of run A precede it, then picks the smaller
//! of the two candidate elements (A wins ties → the whole sort is
//! stable). This is the textbook parallel-stable-merge; unlike a
//! bitonic merge it needs no power-of-two run length, so a final
//! partial run (when the total length is not `n_blocks * 1024` or
//! `n_blocks` is not a power of two) is handled by clamping run
//! boundaries to `n` and treating out-of-range reads as `+∞` sentinels.
//!
//! ## DISPATCH INVARIANTS (mt_sort)
//! - **TPG: 256 threads** (each thread processes 4 elements).
//! - **n = TPG * 4 = 1024** (elements per block — hardcoded in the kernel).
//! - **Grid: 1 threadgroup per block** (1D, program_id<0> = block index).
//!
//! ## DISPATCH INVARIANTS (mt_merge)
//! - **Grid3D / Elementwise**, one thread per *output* element over the
//!   whole array of `n` elements: `grid_x = ceil(n / tpg)`, any `tpg`.
//! - `program_id<0>()` is the global output index `gi`; threads with
//!   `gi >= n` (the ragged tail of the last threadgroup) early-out.
//! - `run` = current sorted-run length (1024 on the first merge pass,
//!   doubling each pass). A merged run is `2 * run` long; the last
//!   merged run is clamped to `n`.
//! - `log_steps` = binary-search iteration count. Must satisfy
//!   `2^log_steps >= 2 * run` so the co-rank search fully converges.
//!   `run` and `log_steps` are `#[constexpr]` so the search loop unrolls.
//! - Input must already hold sorted runs of length `run`; output is a
//!   *separate* buffer (no in-place — caller ping-pongs).

use metaltile::kernel;

#[kernel]
pub fn mt_sort<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let block_id = program_id::<0>();
    let t = tid;
    threadgroup_alloc("shared", 1024, T);
    let base = block_id * n;
    threadgroup_store("shared", t * 4u32, load(inp[base + t * 4u32]));
    threadgroup_store("shared", t * 4u32 + 1u32, load(inp[base + t * 4u32 + 1u32]));
    threadgroup_store("shared", t * 4u32 + 2u32, load(inp[base + t * 4u32 + 2u32]));
    threadgroup_store("shared", t * 4u32 + 3u32, load(inp[base + t * 4u32 + 3u32]));
    threadgroup_barrier();
    for _k in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k, 1u32) {
            let flip = _k - _jb - 1u32;
            if flip >= 7u32 {
                threadgroup_barrier();
            }
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a = threadgroup_load("shared", gi);
                    let b = threadgroup_load("shared", partner);
                    let dir = (gi >> _k) & 1u32;
                    let want_swap = select(dir == 0u32, a > b, a < b);
                    threadgroup_store("shared", gi, select(want_swap, b, a));
                    threadgroup_store("shared", partner, select(want_swap, a, b));
                }
            }
        }
    }
    threadgroup_barrier();
    store(out[base + t * 4u32], threadgroup_load("shared", t * 4u32));
    store(out[base + t * 4u32 + 1u32], threadgroup_load("shared", t * 4u32 + 1u32));
    store(out[base + t * 4u32 + 2u32], threadgroup_load("shared", t * 4u32 + 2u32));
    store(out[base + t * 4u32 + 3u32], threadgroup_load("shared", t * 4u32 + 3u32));
}

// ── Multi-block merge pass ───────────────────────────────────────────────
//
// One bottom-up merge-sort pass. The input `inp` holds sorted runs of
// length `run`; this kernel merges each adjacent pair of runs into one
// sorted run of length `2 * run`, written to `out`.
//
// One thread per output element. For its global output index `gi`:
//
//   pair    = gi / (2 * run)     — which run-pair this element belongs to
//   o       = gi - pair * 2*run  — offset within the merged run
//   aStart  = pair * 2*run                       (clamped ≤ n)
//   aEnd    = aStart + run                       (clamped ≤ n)
//   bStart  = aEnd                               (clamped ≤ n)
//   bEnd    = bStart + run                       (clamped ≤ n)
//   aLen    = aEnd - aStart,  bLen = bEnd - bStart
//
// Run A = inp[aStart .. aEnd), run B = inp[bStart .. bEnd) — both
// already sorted ascending. We want the element that lands at output
// offset `o` of the merged run.
//
// Co-rank: let `i` = number of A-elements that precede output `o`
// (i.e. appear at merged offsets `0 .. o`). Then `j = o - i` is the
// number of B-elements preceding it. The element at `o` is `min(A[i],
// B[j])`, with A winning ties so the merge is stable.
//
// `i` is the largest index in `[lo, hi]` for which "taking the i-th A
// element before the j-th B element" is still consistent with sorted
// order — i.e. `A[i-1] <= B[o-i]`. Bounds:
//   lo = max(0, o - bLen)   — can't take fewer A's than this or B runs out
//   hi = min(o, aLen)       — can't take more A's than exist (or than `o`)
// Binary search converges in `ceil(log2(hi-lo+1)) <= log_steps` steps.
// Out-of-range reads use a `+∞` sentinel so the partial-run / boundary
// cases need no special-casing: an exhausted run always compares
// "greater", so the other run is drained first.
//
// `BenchDispatch::Generic` — correctness is pinned by
// `tests/sort_gpu_correctness.rs`; there is no single-dispatch MLX
// merge-pass to bench against (MLX fuses partition + merge differently).

#[kernel]
pub fn mt_merge<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] run: u32,
    #[constexpr] log_steps: u32,
) {
    let gi = program_id::<0>();
    if gi < n {
        let merged = run + run;
        let pair = gi / merged;
        let o = gi - pair * merged;
        // Run boundaries, each clamped into [0, n] so a final partial
        // pair (n not a multiple of `merged`, or an odd run count) is
        // handled without a separate code path.
        let a_start_raw = pair * merged;
        let a_start = select(a_start_raw < n, a_start_raw, n);
        let a_end_raw = a_start + run;
        let a_end = select(a_end_raw < n, a_end_raw, n);
        let b_start = a_end;
        let b_end_raw = b_start + run;
        let b_end = select(b_end_raw < n, b_end_raw, n);
        let a_len = a_end - a_start;
        let b_len = b_end - b_start;
        // Binary-search bounds for the co-rank `i` (count of A-elements
        // preceding output offset `o`).
        //   lo = max(0, o - b_len)
        //   hi = min(o, a_len)
        let lo0 = select(o > b_len, o - b_len, 0u32);
        let hi0 = select(o < a_len, o, a_len);
        let mut lo = lo0;
        let mut hi = hi0;
        // Branchless binary search. `log_steps` is a constexpr so this
        // loop fully unrolls; each step is a no-op once `lo == hi`.
        for _s in range(0u32, log_steps, 1u32) {
            // mid = ceil((lo + hi) / 2), biased high so the search can
            // settle on `hi`. Only meaningful while lo < hi.
            let active = lo < hi;
            let mid = (lo + hi + 1u32) / 2u32;
            // Probe A[mid-1] vs B[o-mid]. While `active`, mid is in
            // (lo, hi] with 1 <= mid <= hi <= a_len, so `mid-1` indexes
            // a valid A element and `o-mid` is in [0, b_len]. Once the
            // search has converged (`active` false) `mid` collapses to
            // `lo`, which can be 0 — so `mid - 1` would underflow. The
            // probe values are unused in that case, but `select`
            // evaluates both arms, so the *read address* must still be
            // in bounds: clamp both indices and gate the probe on
            // `active`. `b_idx == b_len` (B exhausted) maps to +inf.
            let a_idx = select(mid > 0u32, mid - 1u32, 0u32);
            let b_idx = o - mid;
            let a_in_range = active & (a_start + a_idx < n);
            let a_load = load(inp[a_start + a_idx]).cast::<f32>();
            let a_probe = select(a_in_range, a_load, infinity());
            let b_in_range = active & (b_idx < b_len);
            let b_load = load(inp[b_start + b_idx]).cast::<f32>();
            let b_probe = select(b_in_range, b_load, infinity());
            // Taking the mid-th A element keeps sorted order when
            // A[mid-1] <= B[o-mid]. If so, raise `lo` to `mid`;
            // otherwise lower `hi` to `mid-1`. Both updates are gated
            // on `active` so they are no-ops once `lo == hi`. Nested
            // `select` keeps the whole step branchless.
            let take_more_a = a_probe <= b_probe;
            lo = select(active, select(take_more_a, mid, lo), lo);
            hi = select(active, select(take_more_a, hi, mid - 1u32), hi);
        }
        // i = co-rank, j = its B counterpart.
        let i = lo;
        let j = o - i;
        // Pick the smaller candidate; out-of-range slots are +inf so an
        // exhausted run never wins. `i + j == o < a_len + b_len`, so at
        // least one of the two is a real element. Indices are clamped
        // strictly inside `inp` before the load — the clamped value is
        // discarded by `select` whenever the slot is out of range, so
        // clamping never changes the result, it only keeps the read
        // address in bounds.
        let a_real = i < a_len;
        let b_real = j < b_len;
        let a_safe = select(a_start + i < n, a_start + i, 0u32);
        let b_safe = select(b_start + j < n, b_start + j, 0u32);
        let a_val = select(a_real, load(inp[a_safe]).cast::<f32>(), infinity());
        let b_val = select(b_real, load(inp[b_safe]).cast::<f32>(), infinity());
        // A wins ties (a_val <= b_val) → stable merge.
        let pick_a = a_val <= b_val;
        let chosen = select(pick_a, a_val, b_val);
        store(out[gi], chosen.cast::<T>());
    }
}

// ── Per-row (segmented) sort ─────────────────────────────────────────────
//
// `mt_sort_segmented<T>` sorts each row of a `[batch, n]` matrix
// independently. One threadgroup per row; each threadgroup uses a
// single-block bitonic sort identical to `mt_sort`, covering rows of up
// to `n = TPG * 4 = 1024` elements.
//
// For the typical top-k logits-processing shape (vocab chunks ≤ 1024),
// one dispatch suffices. Rows larger than 1024 are a follow-up (they
// require the per-row multi-block + merge path).
//
// ## DISPATCH INVARIANTS
//
// - **TPG: 256 threads** (each thread processes 4 elements → 1024/row).
// - **n ≤ 1024**: the bitonic sort covers exactly `n` elements per row;
//   for `n < 1024` each thread bounds-guards its loads and treats
//   out-of-range slots with a `+∞` sentinel (they sink to the tail).
// - **Grid: `[batch, 1, 1]`** — one threadgroup per row.
// - Output is a sorted copy (not in-place). Caller manages buffers.
//
// ## Stability
//
// The bitonic sort network is NOT stable by construction — elements with
// equal values may appear in any relative order. For top-k masking this
// is acceptable (we only need the threshold value, not tie-breaking).
// Stability is documented as a non-guarantee.

#[kernel]
pub fn mt_sort_segmented<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    // `tgid_x` = row index; `tid` = thread-local ID within the TG.
    let row = tgid_x;
    let t = tid;
    // 1024-slot shared memory for the bitonic network.
    threadgroup_alloc("shared", 1024, T);
    let row_base = row * n;
    // Load 4 elements per thread. Out-of-range slots get `+∞` so they
    // sink to the tail and the in-range result is correct for any n ≤ 1024.
    let i0 = t * 4u32;
    let i1 = i0 + 1u32;
    let i2 = i0 + 2u32;
    let i3 = i0 + 3u32;
    let inf_f = infinity();
    let v0 = select(i0 < n, load(inp[row_base + i0]).cast::<f32>(), inf_f);
    let v1 = select(i1 < n, load(inp[row_base + i1]).cast::<f32>(), inf_f);
    let v2 = select(i2 < n, load(inp[row_base + i2]).cast::<f32>(), inf_f);
    let v3 = select(i3 < n, load(inp[row_base + i3]).cast::<f32>(), inf_f);
    threadgroup_store("shared", i0, v0.cast::<T>());
    threadgroup_store("shared", i1, v1.cast::<T>());
    threadgroup_store("shared", i2, v2.cast::<T>());
    threadgroup_store("shared", i3, v3.cast::<T>());
    threadgroup_barrier();
    // Bitonic sort network — identical structure to `mt_sort`.
    // Outer loop `_k` grows the sorted sub-sequence length (2^_k).
    // Inner loop `_jb` walks the merge stages from `_k-1` down to 0.
    for _k in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k, 1u32) {
            let flip = _k - _jb - 1u32;
            // `flip >= 7` means the partner may be in a different bank
            // group — barrier is needed to keep the sort coherent.
            if flip >= 7u32 {
                threadgroup_barrier();
            }
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a = threadgroup_load("shared", gi);
                    let b = threadgroup_load("shared", partner);
                    let dir = (gi >> _k) & 1u32;
                    let a_f = a.cast::<f32>();
                    let b_f = b.cast::<f32>();
                    let want_swap = select(dir == 0u32, a_f > b_f, a_f < b_f);
                    threadgroup_store("shared", gi, select(want_swap, b, a));
                    threadgroup_store("shared", partner, select(want_swap, a, b));
                }
            }
        }
    }
    threadgroup_barrier();
    // Write sorted result back, skipping out-of-range sentinel slots.
    if i0 < n {
        store(out[row_base + i0], threadgroup_load("shared", i0));
    }
    if i1 < n {
        store(out[row_base + i1], threadgroup_load("shared", i1));
    }
    if i2 < n {
        store(out[row_base + i2], threadgroup_load("shared", i2));
    }
    if i3 < n {
        store(out[row_base + i3], threadgroup_load("shared", i3));
    }
}

/// New-syntax correctness for the sort family. `mt_sort` (single-block bitonic,
/// Reduction, one TG per 1024-block), `mt_merge` (one bottom-up merge pass,
/// Grid3D — input holds sorted runs of `run`, output sorts each `2*run` block),
/// and `mt_sort_segmented` (Reduction, per-row sort, n ≤ 1024). All exact on the
/// multiset; oracles sort the relevant chunk.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_merge, mt_sort, mt_sort_segmented};
    use crate::utils::{pack_f32, unpack_f32};

    fn sorted_chunks(v: &[f32], chunk: usize) -> Vec<f32> {
        let mut out = v.to_vec();
        for c in out.chunks_mut(chunk) {
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_merge(dt: DType) -> TestSetup {
        let (run, n) = (64usize, 256usize); // 4 runs → 2 merged 128-blocks
        let raw: Vec<f32> =
            (0..n).map(|i| (((i * 2_654_435_761) % 9973) as f32) * 0.01 - 50.0).collect();
        // Input must already hold sorted runs of length `run`.
        let inp = sorted_chunks(&raw, run);
        let inp_dt = unpack_f32(&pack_f32(&inp, dt), dt);
        // A merge pass turns each pair of `run`-runs into one sorted `2*run` run.
        let expected = sorted_chunks(&inp_dt, 2 * run);
        TestSetup::new(mt_merge::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n", n as u32)
            .constexpr("run", run as u32)
            .constexpr("log_steps", 8u32) // 2^8 >= 2*run
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_sort_segmented(dt: DType) -> TestSetup {
        let (batch, n) = (3usize, 512usize); // n ≤ 1024
        let raw: Vec<f32> = (0..batch * n)
            .map(|i| (((i * 2_654_435_761 + 7) % 9973) as f32) * 0.01 - 50.0)
            .collect();
        let raw_dt = unpack_f32(&pack_f32(&raw, dt), dt);
        let expected = sorted_chunks(&raw_dt, n); // each row sorted
        TestSetup::new(mt_sort_segmented::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&raw, dt), dt))
            .input(TestBuffer::zeros("out", batch * n, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(batch as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_sort(dt: DType) -> TestSetup {
        let (n_blocks, n) = (3usize, 1024usize); // n hardcoded to TPG*4 in the kernel
        let mut inp = Vec::with_capacity(n_blocks * n);
        let mut expected = Vec::with_capacity(n_blocks * n);
        for b in 0..n_blocks {
            // A scrambled-but-distinct block; sort is exact on the multiset.
            let block: Vec<f32> = (0..n)
                .map(|i| (((i * 2_654_435_761 + b * 40_503) % 9973) as f32) * 0.01 - 50.0)
                .collect();
            let bd = unpack_f32(&pack_f32(&block, dt), dt);
            let mut sorted = bd.clone();
            sorted.sort_by(|a, c| a.partial_cmp(c).unwrap());
            expected.extend_from_slice(&sorted);
            inp.extend_from_slice(&block);
        }
        TestSetup::new(mt_sort::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n_blocks * n, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_blocks as u32, 1, 1, [256, 1, 1])
    }
}

/// New-syntax benchmark for `mt_sort` (vs MLX `metal/sort.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_merge, mt_sort, mt_sort_segmented};

    #[bench(name = "mlx/sort", dtypes = [f32, f16, bf16])]
    fn bench_sort(dt: DType) -> BenchSetup {
        let (n_blocks, n) = (16384usize, 1024usize);
        BenchSetup::new(mt_sort::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n_blocks * n, dt))
            .buffer(BenchBuffer::zeros("out", n_blocks * n, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(n_blocks as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * n_blocks * n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/sort/merge", dtypes = [f32, f16, bf16])]
    fn bench_merge(dt: DType) -> BenchSetup {
        let (run, n) = (1024usize, 16 * 1024 * 1024usize);
        BenchSetup::new(mt_merge::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n", n as u32)
            .constexpr("run", run as u32)
            .constexpr("log_steps", 12u32) // 2^12 >= 2*run
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }

    #[bench(name = "mlx/sort/segmented", dtypes = [f32, f16, bf16])]
    fn bench_segmented(dt: DType) -> BenchSetup {
        let (batch, n) = (16384usize, 1024usize);
        BenchSetup::new(mt_sort_segmented::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", batch * n, dt))
            .buffer(BenchBuffer::zeros("out", batch * n, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(batch as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * batch * n * dt.size_bytes()) as u64)
    }
}
