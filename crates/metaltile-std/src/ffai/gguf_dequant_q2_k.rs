//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GGUF Q2_K block dequant — k-quant 2-bit-per-weight with two-level scales.
//!
//! Reference: `dequantize_row_q2_K` in
//! [llama.cpp ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
//!
//! ## On-disk block layout (decomposed CPU-side at load time)
//!
//! ```text
//!   struct block_q2_K {
//!     uint8_t  scales[16];   // 16 bytes — low 4 bits = scale, high 4 bits = min
//!     uint8_t  qs[64];       // 64 bytes — 2-bit-packed quants, 4 vals per byte
//!     uint16_t d;            //  2 bytes — fp16 super-scale for scales
//!     uint16_t dmin;         //  2 bytes — fp16 super-scale for mins
//!   };                       // 84 bytes per 256 values (BPW = 2.625)
//! ```
//!
//! Per output value `i ∈ [0, 256)`:
//!
//! ```text
//!   sub        = i / 16           // 0..15, picks the (4-bit scale, 4-bit min) pair
//!   in_sub     = i & 15            // 0..15 inside the sub-block
//!   scale_byte = scales[sub]
//!   scale_4bit = scale_byte & 0xf
//!   min_4bit   = (scale_byte >> 4) & 0xf
//!   qs_byte    = qs[i / 4]
//!   shift      = (i & 3) * 2
//!   q_2bit     = (qs_byte >> shift) & 0x3
//!   out[i]     = d * scale_4bit * q_2bit - dmin * min_4bit
//! ```
//!
//! ## GPU-resident split (the loader produces these from the packed block)
//!
//! 1. `qs_packed [n_blocks * 16]`   — `u32`, the 64 packed-quant bytes
//!    per block re-laid as 16 u32 words. `qs_packed[block*16 + j]`
//!    carries 16 two-bit quants in the lower / upper bytes of each
//!    u32. Output index `i ∈ [0, 256)` → `u32 j = i / 16`, then a
//!    `(i % 16) * 2`-bit shift on the byte that holds it.
//! 2. `scales    [n_blocks * 16]`   — `u8`, the raw scale/min byte
//!    pairs (low nibble = scale, high nibble = min) — kept packed
//!    because both nibbles are used per dequant.
//! 3. `d_f32     [n_blocks]`        — `f32`, host-converted from fp16.
//! 4. `dmin_f32  [n_blocks]`        — `f32`, host-converted from fp16.
//!
//! ## Dispatch
//!
//! 1D grid: one thread per *output value*. ~6 reads (1 qs_packed + 1
//! scales + 1 each of d_f32 / dmin_f32, scales cache-multicast across
//! 16 lanes that share a sub-block) and ~4 arithmetic ops per output —
//! cleanly bandwidth-bound on Apple9.

use metaltile::kernel;

#[kernel(
    bench(
        op="gguf_dequant",
        subop="q2_k",
        class=GenericEmpty,
        tol=1e-3,
    )
)]
pub fn ffai_gguf_dequant_q2_k<T>(
    qs_packed: Tensor<u32>,
    scales: Tensor<u8>,
    d_f32: Tensor<f32>,
    dmin_f32: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 256u32;
        let in_block = i - block * 256u32;
        let sub = in_block / 16u32; // 0..15
        // qs is 64 bytes per block, re-laid as 16 u32 words → byte `i % 64`
        // lives in `qs_packed[block*16 + (i%64)/4]`, in the
        // `(i%4)`-th byte (LSB-first, little-endian).
        let q_byte_idx = in_block / 4u32; // 0..63 (the byte within block)
        let word_idx = q_byte_idx / 4u32; // 0..15 (the u32 within block)
        let byte_in_word = q_byte_idx & 3u32; // which of the 4 bytes
        let word = load(qs_packed[block * 16u32 + word_idx]);
        let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
        let shift = (in_block & 3u32) * 2u32;
        let q_2bit = (qs_byte >> shift) & 0x3u32;

        let scale_byte = load(scales[block * 16u32 + sub]).cast::<u32>();
        let scale_4bit = scale_byte & 0xfu32;
        let min_4bit = (scale_byte >> 4u32) & 0xfu32;

        let d = load(d_f32[block]);
        let dmin = load(dmin_f32[block]);

        let scaled =
            d * (scale_4bit.cast::<i32>().cast::<f32>()) * (q_2bit.cast::<i32>().cast::<f32>());
        let offset = dmin * (min_4bit.cast::<i32>().cast::<f32>());
        // Implicit narrowing — see playbook §"DSL implicit Store
        // coercion" (no `.cast::<T>()` at the Store site).
        store(out[i], scaled - offset);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gguf_dequant_q2_k;
    use crate::utils::pack_f32;

    /// Quantize a slice into the Q2_K GPU-resident split.
    ///
    /// This is a *non-trained* quantizer (per-sub-block min/max with
    /// uniform 2-bit bucketing) sufficient to drive the kernel's
    /// correctness test — `quantize_row_q2_K` in ggml-quants.c uses a
    /// search-and-fit procedure tuned for perplexity that we don't
    /// need to replicate for a dequant test.
    fn quantize_q2_k(values: &[f32]) -> (Vec<u32>, Vec<u8>, Vec<f32>, Vec<f32>) {
        assert_eq!(values.len() % 256, 0, "Q2_K needs multiple-of-256 values");
        let n_blocks = values.len() / 256;
        let mut qs_packed = Vec::with_capacity(n_blocks * 16);
        let mut scales = Vec::with_capacity(n_blocks * 16);
        let mut d_f32 = Vec::with_capacity(n_blocks);
        let mut dmin_f32 = Vec::with_capacity(n_blocks);

        for block in values.chunks_exact(256) {
            // Per-sub-block (16 sub-blocks of 16 values each) compute
            // mn, scale; quantize to 2 bits.
            let mut sub_scales = [0u8; 16];
            let mut sub_mins = [0u8; 16];
            let mut qs_bytes = [0u8; 64];
            for s in 0..16 {
                let sub = &block[s * 16..(s + 1) * 16];
                let mn = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((mx - mn) / 3.0).max(1e-30);
                let mn_q = (-mn / scale).round().clamp(0.0, 15.0) as u8;
                let scale_q = (scale * 15.0).round().clamp(0.0, 15.0) as u8;
                // Reconstruct the encoded `d` / `dmin` super-scales so
                // dequant exactly recovers the per-sub-block scale.
                sub_scales[s] = scale_q;
                sub_mins[s] = mn_q;
                // Pack two 2-bit quants per nibble: out = round((x + min) / scale).
                let recon_scale = (scale_q as f32) / 15.0;
                let recon_min = -(mn_q as f32) * recon_scale;
                for (i, &v) in sub.iter().enumerate() {
                    let q =
                        ((v - recon_min) / recon_scale.max(1e-30)).round().clamp(0.0, 3.0) as u8;
                    let target = s * 16 + i;
                    let byte = target / 4;
                    let shift = (target & 3) * 2;
                    qs_bytes[byte] |= q << shift;
                }
            }
            // Combine the per-sub-block 4-bit scale + 4-bit min into
            // the on-disk scales[16] layout.
            for s in 0..16 {
                scales.push((sub_mins[s] << 4) | sub_scales[s]);
            }
            // Repack `qs_bytes` (64 bytes) into 16 u32 LE words.
            for w in 0..16 {
                let bs = w * 4;
                qs_packed.push(
                    (qs_bytes[bs] as u32)
                        | ((qs_bytes[bs + 1] as u32) << 8)
                        | ((qs_bytes[bs + 2] as u32) << 16)
                        | ((qs_bytes[bs + 3] as u32) << 24),
                );
            }
            // The two super-scales encode `d = 1/15` and `dmin = 1`
            // after the reconstruction above. Materialize as fp32 the
            // same way the GGUF loader would after fp16-converting.
            d_f32.push(half::f16::from_f32(1.0 / 15.0).to_f32());
            dmin_f32.push(half::f16::from_f32(1.0).to_f32());
        }
        (qs_packed, scales, d_f32, dmin_f32)
    }

    /// CPU reference. Mirrors the GPU kernel exactly so any kernel-side
    /// divergence shows up as a tolerance miss.
    fn cpu_dequant(qs_packed: &[u32], scales: &[u8], d_f32: &[f32], dmin_f32: &[f32]) -> Vec<f32> {
        let n_blocks = d_f32.len();
        let mut out = Vec::with_capacity(n_blocks * 256);
        for b in 0..n_blocks {
            let d = d_f32[b];
            let dmin = dmin_f32[b];
            for i in 0..256_usize {
                let sub = i / 16;
                let q_byte_idx = i / 4;
                let word = qs_packed[b * 16 + q_byte_idx / 4];
                let byte_in_word = q_byte_idx & 3;
                let qs_byte = (word >> (byte_in_word * 8)) & 0xff;
                let shift = (i & 3) * 2;
                let q_2bit = (qs_byte >> shift) & 0x3;
                let scale_byte = scales[b * 16 + sub] as u32;
                let scale_4bit = scale_byte & 0xf;
                let min_4bit = (scale_byte >> 4) & 0xf;
                out.push(d * (scale_4bit as f32) * (q_2bit as f32) - dmin * (min_4bit as f32));
            }
        }
        out
    }

    fn setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 256;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007 - 0.5).sin() * 1.5).collect();
        let (qs_packed, scales, d_f32, dmin_f32) = quantize_q2_k(&values);
        let dequantized = cpu_dequant(&qs_packed, &scales, &d_f32, &dmin_f32);
        // Pack u32 vec as little-endian bytes for the test framework.
        let qs_bytes: Vec<u8> = qs_packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        TestSetup::new(ffai_gguf_dequant_q2_k::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_packed", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", scales, DType::U8))
            .input(TestBuffer::from_vec("d_f32", pack_f32(&d_f32, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("dmin_f32", pack_f32(&dmin_f32, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_single_block(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_many_blocks(dt: DType) -> TestSetup { setup(8, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_dequant_q2_k;

    #[bench(name = "ffai/gguf_dequant_q2_k", dtypes = [f32, f16, bf16])]
    fn bench_q2_k(dt: DType) -> BenchSetup {
        // Representative MoE-expert down-proj slab — 4096 × 4096.
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(ffai_gguf_dequant_q2_k::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_packed", n_blocks * 16, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks * 16, DType::U8))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("dmin_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            // qs_packed 64 B + scales 16 B + 2*4 B per block + output T
            .bytes_moved(((n_blocks * (64 + 16 + 8)) + n * dt.size_bytes()) as u64)
    }
}
