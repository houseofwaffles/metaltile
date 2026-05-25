//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched raw KV-cache append — writes T tokens' K (or V) rows into the
//! per-head cache in ONE dispatch.
//!
//! Identical data move to `kv_cache::kv_cache_update` but the scalar
//! `position` constexpr is replaced by a per-row `positions: Tensor<u32>`
//! of length T. For Qwen / Llama prefill the caller used to T-loop over
//! `kv_cache_update` once per token (T dispatches × N attention layers,
//! ×2 for the K+V pair); this kernel collapses that to ONE dispatch per
//! (layer, K-or-V buffer). Same dispatch-saving pattern as
//! `rope_llama_many` — see that file's docstring for the broader
//! prefill-time motivation.
//!
//! Layout:
//!
//!   src       [T, n_kv_heads, head_dim]              T (dtype)
//!   positions [T]                                    u32
//!   out       [n_kv_heads, max_seq, head_dim]        T (dtype)
//!
//! Note the layout flip: `src` is row-major over tokens (caller writes
//! T fresh rows back-to-back), `out` is the cache laid out by head
//! across `max_seq` slots (so different rows write into different
//! `[*, position, *]` slices, decided per-row by `positions[r]`).
//!
//! Grid: one thread per source element, total `T * n_kv_heads * head_dim`
//! threads. The kernel uses a flat `program_id::<0>()` and recovers the
//! `(r, h, d)` triple from it — same shape as the single-token primitive
//! (which uses `(h, d)` from a flat program_id), just with one extra
//! outer axis. `kernel_mode=Grid3D` is preserved so the launcher's
//! dispatch_threadgroups path stays identical.
//!
//! `n_kv_heads_x_head_dim` is a separate constexpr (not n_kv_heads +
//! head_dim individually) so the per-thread decomposition only needs one
//! divisor on the hot path. The DSL constant-folds, but keeping the
//! multiply out of the kernel body is the same micro-tweak the existing
//! `dequant_gather` / `gather` kernels use for their `dim` parameter.
//!
//! Pure data move — no arithmetic, no rounding. `tol=0.0` matches
//! `kv_cache_update`.
//!
//! Codegen-only. Correctness validated against `kv_cache_update` looped
//! per-row in `tests/kv_cache_update_many_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="kv_cache",
    subop="update_many",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
pub fn kv_cache_update_many<T>(
    src: Tensor<T>,
    positions: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] n_kv_heads_x_head_dim: u32,
) {
    let idx = program_id::<0>();
    // Per-source-element decomposition:
    //   idx = r * (n_kv_heads * head_dim) + h * head_dim + d
    // `n_kv_heads_x_head_dim` is the row-stride of `src`; `head_dim` is
    // the inner-row width. Two divs, no mults on the index path itself.
    let r = idx / n_kv_heads_x_head_dim;
    let in_row = idx - r * n_kv_heads_x_head_dim;
    let h = in_row / head_dim;
    let d = in_row - h * head_dim;
    // Per-row position lookup — same shape as `rope_llama_many`.
    let position = load(positions[r]);
    // Cache layout: [n_kv_heads, max_seq, head_dim]. Each row r writes
    // into out[h, positions[r], d] for its own h and d.
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}
