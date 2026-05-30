//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MLX-format dequantizing gather kernels (quantized embedding tables).
//! For each output element `(token, d)`: look up the packed weight,
//! extract the right value, dequantize via `q * scale + bias`.
//!
//! Layouts (per dtype, with H = `hidden`, G = `group_size`):
//!
//!   weight   [vocab, H * bits / 32]   uint32
//!   scales   [vocab, H / G]           T
//!   biases   [vocab, H / G]           T
//!   indices  [n_tokens]               u32
//!   out      [n_tokens, H]            T
//!
//! One thread per output element.  All bit widths share one formula:
//! element `d` occupies bits `[d*bits, (d+1)*bits)` in the row's bit stream,
//! spanning at most two adjacent u32 words.
//!
//! ```text
//!   bit_off  = d * bits
//!   word_idx = bit_off / 32
//!   bit_in_w = bit_off & 31
//!   lo_bits  = min(bits, 32 - bit_in_w)        ← bits from word 0
//!   spill    = bits - lo_bits                   ← bits from word 1
//!   lo       = (w0 >> bit_in_w) & ((1 << lo_bits) - 1)
//!   hi       = (w1 & ((1 << spill) - 1)) << lo_bits
//!   q        = lo | hi
//! ```
//!
//! When `spill == 0`, `w1` loads from `word_idx` (same as w0) so the address
//! is always in-bounds; the `(1 << 0) - 1 == 0` mask zeroes `hi` regardless.
//!
//! ## Macro structure
//!
//! `dequant_gather_kernel!` emits the entire `#[kernel] pub fn …`
//! at module scope.  The compiler expands the outer macro before the
//! `#[kernel]` proc-macro runs, so the body parser sees concrete tokens
//! with `$bits` already substituted.  Embedding the body inside an *inner*
//! `macro_rules!` call (the previous shape of this file) silently produced
//! empty kernels — the proc-macro doesn't expand inner declarative macros.

use metaltile::kernel;

macro_rules! dequant_gather_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            indices: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] hidden: u32,
            #[constexpr] group_size: u32,
        ) {
            let idx = program_id::<0>();
            let token = idx / hidden;
            let d = idx - token * hidden;
            let token_id = load(indices[token]);

            let groups_per_row = hidden / group_size;
            let g = d / group_size;
            let u32_per_row = hidden * $bits / 32u32;
            let row_off = token_id * u32_per_row;

            let bit_off = d * $bits;
            let word_idx = bit_off / 32u32;
            let bit_in_w = bit_off & 31u32;

            let bits_in_w0 = 32u32 - bit_in_w;
            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
            let spill = $bits - lo_bits;

            let w0 = load(weight[row_off + word_idx]);
            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
            let w1 = load(weight[row_off + w1_idx]);

            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
            let q = lo | hi;

            let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
            let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
            let w_real = q.cast::<f32>() * scale + bias;
            store(out[idx], w_real.cast::<T>());
        }
    };
}

dequant_gather_kernel!(dequant_gather_int2, 2u32, "int2");
dequant_gather_kernel!(dequant_gather_int3, 3u32, "int3");
dequant_gather_kernel!(dequant_gather_int4, 4u32, "int4");
dequant_gather_kernel!(dequant_gather_int5, 5u32, "int5");
dequant_gather_kernel!(dequant_gather_int6, 6u32, "int6");
dequant_gather_kernel!(dequant_gather_int8, 8u32, "int8");

/// New-syntax correctness tests for the `dequant_gather_int{2,3,4,5,6,8}`
/// quantized-embedding-gather family. Grid3D, one thread per output element
/// (`n_tokens * hidden` threads via `grid_1d` ceil-div).
///
/// Oracle: synthesize a quantized vocab table (bit-stream-packed int-`bits`
/// codes `[vocab, hidden]`, per-group scale/bias), pick a non-monotonic gather
/// order that repeats a row, then replay `out[token, d] = q·scale_g + bias_g`
/// in f32. Pure dequant (no matmul) → tight tolerances. Inputs are dtype-rounded
/// so the GPU sees exactly what the oracle does.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Synthesize bit-stream-packed int-`bits` weights for a `[vocab, hidden]`
    /// table — same two-word `lo | hi` layout the kernel decodes (covers both
    /// pack-aligned pow2 widths and the odd word-spilling widths).
    fn synth_bitstream_w(vocab: usize, hidden: usize, bits: u32) -> Vec<u32> {
        let mask = (1u32 << bits) - 1;
        let u32_per_row = hidden * bits as usize / 32;
        let mut packed = vec![0u32; vocab * u32_per_row];
        for r in 0..vocab {
            let row_base = r * u32_per_row;
            for d in 0..hidden {
                let code =
                    ((r * hidden + d) as u32).wrapping_mul(2_654_435_761).wrapping_add(d as u32)
                        & mask;
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                if bits_in_w0 >= bits {
                    packed[row_base + word] |= code << in_w;
                } else {
                    packed[row_base + word] |= code << in_w;
                    packed[row_base + word + 1] |= code >> bits_in_w0;
                }
            }
        }
        packed
    }

    /// CPU oracle: per `(token, d)`, gather row `indices[token]`, unpack the
    /// `bits`-wide code at bit offset `d*bits`, dequantize via `q*scale+bias`.
    #[allow(clippy::too_many_arguments)]
    fn dequant_gather_oracle(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        indices: &[u32],
        hidden: usize,
        group_size: usize,
        bits: u32,
    ) -> Vec<f32> {
        let n_tokens = indices.len();
        let groups_per_row = hidden / group_size;
        let u32_per_row = hidden * bits as usize / 32;
        let mask: u64 = (1u64 << bits) - 1;
        let mut out = vec![0.0f32; n_tokens * hidden];
        for (token, &tid) in indices.iter().enumerate() {
            let token_id = tid as usize;
            let row_w = &weight[token_id * u32_per_row..(token_id + 1) * u32_per_row];
            for d in 0..hidden {
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                let q = if bits_in_w0 >= bits {
                    ((row_w[word] as u64) >> in_w) & mask
                } else {
                    let lo_bits = bits_in_w0;
                    let spill = bits - lo_bits;
                    let lo = ((row_w[word] as u64) >> in_w) & ((1u64 << lo_bits) - 1);
                    let hi = ((row_w[word + 1] as u64) & ((1u64 << spill) - 1)) << lo_bits;
                    lo | hi
                };
                let g = d / group_size;
                out[token * hidden + d] = (q as f32) * scales[token_id * groups_per_row + g]
                    + biases[token_id * groups_per_row + g];
            }
        }
        out
    }

    fn gather_setup(
        kernel: Kernel,
        bits: u32,
        hidden: usize,
        group_size: usize,
        dt: DType,
    ) -> TestSetup {
        let vocab = 8usize;
        let n_groups = hidden / group_size;
        let w = synth_bitstream_w(vocab, hidden, bits);
        let scales_f: Vec<f32> =
            (0..vocab * n_groups).map(|i| 0.02 + (i % 7) as f32 * 0.01).collect();
        let biases_f: Vec<f32> =
            (0..vocab * n_groups).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        // Non-monotonic gather that repeats row 4 — surfaces token→row bugs.
        let indices: Vec<u32> = vec![3, 0, 7, 1, 4, 4];
        let n_tokens = indices.len();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let expected = dequant_gather_oracle(&w, &s, &b, &indices, hidden, group_size, bits);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("weight", u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", n_tokens * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_tokens * hidden, 256)
    }

    // Pack-strided pow2 widths (codes never span a u32 boundary).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int2(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int2::kernel_ir_for(dt), 2, 256, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int4(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int4::kernel_ir_for(dt), 4, 256, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int8(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int8::kernel_ir_for(dt), 8, 256, 64, dt)
    }

    // Odd widths (word-spilling `lo | hi` decode). hidden*bits a multiple of 32:
    //   int3: 64*3 = 192; int5: 64*5 = 320; int6: 64*6 = 384. group_size 32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int3(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int3::kernel_ir_for(dt), 3, 64, 32, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int5(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int5::kernel_ir_for(dt), 5, 64, 32, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1])]
    fn test_dequant_gather_int6(dt: DType) -> TestSetup {
        gather_setup(dequant_gather_int6::kernel_ir_for(dt), 6, 64, 32, dt)
    }
}

/// New-syntax benchmarks for the dequant-gather family. Representative decode
/// shape (gather `n_tokens` rows of a `vocab × hidden` quantized table).
/// Grid3D, one thread per output element. bytes_moved counts the dequantized
/// output stream (the gather reads a small packed-weight subset per token).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    fn gb(kernel: Kernel, bits: u32, hidden: usize, group_size: usize, dt: DType) -> BenchSetup {
        let vocab = 4096usize;
        let n_tokens = 32usize;
        let n_groups = hidden / group_size;
        let u32_per_row = hidden * bits as usize / 32;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("weight", vocab * u32_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", vocab * n_groups, dt))
            .buffer(BenchBuffer::random("biases", vocab * n_groups, dt))
            .buffer(BenchBuffer::random("indices", n_tokens, DType::U32))
            .buffer(BenchBuffer::zeros("out", n_tokens * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("group_size", group_size as u32)
            .grid_1d(n_tokens * hidden, 256)
            .bytes_moved((n_tokens * hidden * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/dequant_gather/int2", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int2(dt: DType) -> BenchSetup {
        gb(dequant_gather_int2::kernel_ir_for(dt), 2, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gather/int3", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int3(dt: DType) -> BenchSetup {
        gb(dequant_gather_int3::kernel_ir_for(dt), 3, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gather/int4", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int4(dt: DType) -> BenchSetup {
        gb(dequant_gather_int4::kernel_ir_for(dt), 4, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gather/int5", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int5(dt: DType) -> BenchSetup {
        gb(dequant_gather_int5::kernel_ir_for(dt), 5, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gather/int6", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int6(dt: DType) -> BenchSetup {
        gb(dequant_gather_int6::kernel_ir_for(dt), 6, 4096, 64, dt)
    }
    #[bench(name = "ffai/dequant_gather/int8", dtypes = [f32, f16, bf16])]
    fn bench_dequant_gather_int8(dt: DType) -> BenchSetup {
        gb(dequant_gather_int8::kernel_ir_for(dt), 8, 4096, 64, dt)
    }
}
