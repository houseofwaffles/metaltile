//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA bulk dequant — unpack codebook-quantized values into rotated
//! codec space, ready to be consumed by the AURA flash-SDPA path or
//! materialised as a fp16/bf16 tensor for downstream SDPA.
//!
//! Port of `turbo_dequant_rotated` from
//! `ekryski/mlx@alpha`/`mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Input:
//! - `packed [B*H, T, packed_width]` u32  — bit-packed codebook indices.
//!   `packed_width = ceil(dim * bits / 32)`.
//! - `norms  [B*H, T]`               T    — per-token norm correction; cast-
//!   at-load to f32 internally.
//! - `codebook [2**bits]`            T    — Lloyd-Max centroids; cast-at-load
//!   to f32 internally.
//!
//! Output:
//! - `out  [B*H, T, dim]`            T    — fp16 / bf16 / fp32 in rotated
//!   codec space; caller applies the inverse rotation (e.g. via
//!   flash-SDPA p2-with-fused-rot).
//!
//! ## Bit-extract paths
//!
//! - `bits ∈ {2, 4, 8}`: 32 / bits divides cleanly → each packed word
//!   holds exactly `32 / bits` quantized dims with no cross-word spill.
//!   Inner loop emits `DIMS_PER_WORD` outputs per thread with a single
//!   load.
//! - `bits ∈ {3, 5, 6}`: odd-width packs straddle word boundaries.  Each
//!   per-dim emit re-fetches `packed[word_idx]` (and `packed[word_idx+1]`
//!   if spilling) to grab the bits whose absolute offset is `d * bits`.
//!   Same logic as `dequant_gemv_int{3,5,6}` in the affine-quant path.
//!
//! ## Macro structure
//!
//! Outer `aura_dequant_rotated_clean!` (for bits ∈ {2,4,8}) and
//! `aura_dequant_rotated_odd!` (for bits=3) emit the entire
//! `#[kernel] pub fn …` at module scope.  Required because
//! the `#[kernel]` proc-macro doesn't expand inner `macro_rules!`
//! invocations (see CLAUDE.md note about PR #19's macro regression).

use metaltile::kernel;

// ── Clean nibble/byte path: bits ∈ {2, 4, 8} ─────────────────────────────
//
// Each thread owns one packed word w covering DIMS_PER_WORD = 32/bits
// dim slots starting at `d_base = w * DIMS_PER_WORD`.  One u32 load
// amortises across all dims in the pack.
#[rustfmt::skip]
macro_rules! aura_dequant_rotated_clean {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            packed: Tensor<u32>,
            norms: Tensor<T>,
            codebook: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
        ) {
            // Dispatch grid is exactly (packed_width, tokens, B*H); Metal's
            // `dispatchThreads` doesn't pad, so the MLX-source's
            // `if (w >= packed_width) return;` guards are unnecessary
            // belt-and-suspenders.  Omitted here — the DSL has no early
            // `return`, and bounded `for k < dims_per_word` plus
            // `if d < dim` keeps any spurious thread from writing out of
            // bounds.
            let w = program_id::<0>();
            let t = program_id::<1>();
            let bh = program_id::<2>();

            let mask = (1u32 << $bits) - 1u32;
            let dims_per_word = 32u32 / $bits;

            let base = (bh * tokens + t) * packed_width;
            let word = load(packed[base + w]);
            let norm_val = load(norms[bh * tokens + t]).cast::<f32>();

            let d_base = w * dims_per_word;
            let out_row_base = (bh * tokens + t) * dim + d_base;
            for k in range(0u32, dims_per_word, 1u32) {
                let d = d_base + k;
                if d < dim {
                    let val = (word >> (k * $bits)) & mask;
                    let centroid = load(codebook[val]).cast::<f32>();
                    let result = centroid * norm_val;
                    store(out[out_row_base + k], result.cast::<T>());
                }
            }
        }
    };
}

// ── Odd-width spill path: bits ∈ {3, 5, 6} ───────────────────────────────
//
// Words straddle dim boundaries: thread `w` may need to read packed[w]
// AND packed[w+1] for any dim whose bit-range crosses word index 32.
// Same bit-stream formula as `dequant_gather_int{3,5,6}`.
//
// `ceil(32 / bits)` outputs per thread; `d_base = w * DIMS_PER_WORD`
// for the iteration but the bit-offset arithmetic is keyed on the
// absolute dim index `d`, so cross-word spills resolve correctly.
#[rustfmt::skip]
macro_rules! aura_dequant_rotated_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            packed: Tensor<u32>,
            norms: Tensor<T>,
            codebook: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
        ) {
            // Dispatch grid is exactly (packed_width, tokens, B*H); Metal's
            // `dispatchThreads` doesn't pad, so the MLX-source's
            // `if (w >= packed_width) return;` guards are unnecessary
            // belt-and-suspenders.  Omitted here — the DSL has no early
            // `return`, and bounded `for k < dims_per_word` plus
            // `if d < dim` keeps any spurious thread from writing out of
            // bounds.
            let w = program_id::<0>();
            let t = program_id::<1>();
            let bh = program_id::<2>();

            let mask = (1u32 << $bits) - 1u32;
            let dims_per_word = (32u32 + $bits - 1u32) / $bits;

            let base = (bh * tokens + t) * packed_width;
            let norm_val = load(norms[bh * tokens + t]).cast::<f32>();

            let d_base = w * dims_per_word;
            for k in range(0u32, dims_per_word, 1u32) {
                let d = d_base + k;
                if d < dim {
                    let bit_offset = d * $bits;
                    let word_idx = bit_offset / 32u32;
                    let bit_in_w = bit_offset & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(packed[base + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[base + w1_idx]);

                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let val = (lo | hi) & mask;

                    let centroid = load(codebook[val]).cast::<f32>();
                    let result = centroid * norm_val;
                    store(out[(bh * tokens + t) * dim + d], result.cast::<T>());
                }
            }
        }
    };
}

// Bit-width × dim instantiations.  AURA today supports kb ∈ {2,3,4,6,8}
// per the session plan (kb=5 isn't shipped); add new variants here when
// the planning doc adds another kb level.
aura_dequant_rotated_clean!(aura_dequant_rotated_int2, 2u32, "dequant_rotated_int2");
aura_dequant_rotated_clean!(aura_dequant_rotated_int4, 4u32, "dequant_rotated_int4");
aura_dequant_rotated_clean!(aura_dequant_rotated_int8, 8u32, "dequant_rotated_int8");
aura_dequant_rotated_odd!(aura_dequant_rotated_int3, 3u32, "dequant_rotated_int3");
aura_dequant_rotated_odd!(aura_dequant_rotated_int6, 6u32, "dequant_rotated_int6");

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_dequant_rotated_int4;
    use crate::utils::{pack_f32, unpack_f32};

    /// Bit-pack a flat `[bh, t, dim]` int4 index array into
    /// `[bh, t, packed_width]` u32 words — what `aura_encode` produces.
    fn pack_int4_indices(indices: &[u32], bh: usize, tokens: usize, dim: usize) -> Vec<u32> {
        let bits = 4;
        let packed_width = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; bh * tokens * packed_width];
        for b in 0..bh {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(b * tokens + t) * dim + d];
                    let bit_offset = d * bits;
                    let word_idx = bit_offset / 32;
                    let shift = bit_offset & 31;
                    packed[(b * tokens + t) * packed_width + word_idx] |= (idx & 0xf) << shift;
                }
            }
        }
        packed
    }

    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1)]
    fn test_aura_dequant_rotated_int4(dt: DType) -> TestSetup {
        // bits=4, dim=128, packed_width=16, 2 heads × 3 tokens.
        let (dim, bh, tokens) = (128usize, 2usize, 3usize);
        let packed_width = (dim * 4).div_ceil(32);
        // 16-level symmetric codebook in [-1, 1]. Now T-typed (#212): the
        // buffer follows the activation dtype so the same codebook feeds the
        // encoder and decoder with no per-call cast.
        let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let indices: Vec<u32> = (0..bh * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
        let packed = pack_int4_indices(&indices, bh, tokens, dim);
        let norms: Vec<f32> = (0..bh * tokens).map(|i| 0.5 + 0.1 * i as f32).collect();

        // `norms` and `codebook` are now `Tensor<T>` — round them through the
        // GPU dtype so the oracle sees the same cast-at-load values the kernel
        // does. Output is also rounded through `dt` to match the store-cast.
        let codebook_r = unpack_f32(&pack_f32(&codebook, dt), dt);
        let norms_r = unpack_f32(&pack_f32(&norms, dt), dt);
        let expected: Vec<f32> = (0..bh * tokens * dim)
            .map(|i| codebook_r[indices[i] as usize] * norms_r[i / dim])
            .collect();

        TestSetup::new(aura_dequant_rotated_int4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack_f32(&norms, dt), dt))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook, dt), dt))
            .input(TestBuffer::zeros("out", bh * tokens * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            // tpg=1 → total threads == group counts == (packed_width, tokens, B*H).
            .grid_3d(packed_width as u32, tokens as u32, bh as u32, [1, 1, 1])
    }
}

/// New-syntax benchmarks for the AURA bulk-dequant family (int2/3/4/6/8) —
/// MLX-less Grid3D kernels. Shape: head_dim 128, 64 tokens, 8 KV heads.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        aura_dequant_rotated_int2,
        aura_dequant_rotated_int3,
        aura_dequant_rotated_int4,
        aura_dequant_rotated_int6,
        aura_dequant_rotated_int8,
    };

    fn setup(
        s: BenchSetup,
        dim: usize,
        bits: usize,
        bh: usize,
        tokens: usize,
        dt: DType,
    ) -> BenchSetup {
        let packed_width = (dim * bits).div_ceil(32);
        let levels = 1usize << bits;
        s.mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("packed", bh * tokens * packed_width, DType::U32))
            .buffer(BenchBuffer::random("norms", bh * tokens, dt))
            .buffer(BenchBuffer::random("codebook", levels, dt))
            .buffer(BenchBuffer::zeros("out", bh * tokens * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .bytes_moved((bh * tokens * dim * dt.size_bytes()) as u64)
            .grid_3d(packed_width as u32, tokens as u32, bh as u32, [1, 1, 1])
    }

    #[bench(name = "ffai/aura_dequant_rotated_int2", dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_dequant_rotated_int2::kernel_ir_for(dt)), 128, 2, 8, 64, dt)
    }

    #[bench(name = "ffai/aura_dequant_rotated_int3", dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_dequant_rotated_int3::kernel_ir_for(dt)), 128, 3, 8, 64, dt)
    }

    #[bench(name = "ffai/aura_dequant_rotated_int4", dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_dequant_rotated_int4::kernel_ir_for(dt)), 128, 4, 8, 64, dt)
    }

    #[bench(name = "ffai/aura_dequant_rotated_int6", dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_dequant_rotated_int6::kernel_ir_for(dt)), 128, 6, 8, 64, dt)
    }

    #[bench(name = "ffai/aura_dequant_rotated_int8", dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_dequant_rotated_int8::kernel_ir_for(dt)), 128, 8, 8, 64, dt)
    }
}
