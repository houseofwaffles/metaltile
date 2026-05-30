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

use metaltile::kernel;

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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::kv_cache_update_many;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_kv_cache_update_many(dt: DType) -> TestSetup {
        let (n_tokens, n_kv_heads, head_dim, max_seq) = (8usize, 8usize, 16usize, 32usize);
        let sentinel = 999.0f32;
        let src: Vec<f32> =
            (0..n_tokens * n_kv_heads * head_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();
        // Distinct positions (no overlap → matches production prefill).
        let positions: Vec<u32> = (0..n_tokens as u32).map(|r| r * 3 + 1).collect();
        let cache = vec![sentinel; n_kv_heads * max_seq * head_dim];
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let mut expected = unpack_f32(&pack_f32(&cache, dt), dt);
        let row_stride = n_kv_heads * head_dim;
        for (r, &pos_u32) in positions.iter().enumerate() {
            let pos = pos_u32 as usize;
            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    let s = r * row_stride + h * head_dim + d;
                    let dst = h * max_seq * head_dim + pos * head_dim + d;
                    expected[dst] = src_dt[s];
                }
            }
        }
        let total = n_tokens * n_kv_heads * head_dim;
        TestSetup::new(kv_cache_update_many::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("positions", u32_bytes(&positions), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&cache, dt), dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("n_kv_heads_x_head_dim", (n_kv_heads * head_dim) as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(total, 256)
    }
}

/// New-syntax benchmark for `kv_cache_update_many` — a Qwen-class prefill
/// batch appended in one dispatch (Grid3D, one thread per source element).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::kv_cache_update_many;

    fn u32_bytes(v: impl Iterator<Item = u32>) -> Vec<u8> {
        v.flat_map(|x| x.to_le_bytes()).collect()
    }

    #[bench(name = "ffai/kv_cache/update_many", dtypes = [f32, f16, bf16])]
    fn bench_kv_cache_update_many(dt: DType) -> BenchSetup {
        let (n_tokens, n_kv_heads, head_dim, max_seq) = (512usize, 8usize, 128usize, 4096usize);
        let total = n_tokens * n_kv_heads * head_dim;
        BenchSetup::new(kv_cache_update_many::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", total, dt))
            .buffer(BenchBuffer::from_vec(
                "positions",
                u32_bytes((0..n_tokens).map(|r| r as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", n_kv_heads * max_seq * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("n_kv_heads_x_head_dim", (n_kv_heads * head_dim) as u32)
            .grid_1d(total, 256)
            .bytes_moved((2 * total * dt.size_bytes()) as u64)
    }
}
