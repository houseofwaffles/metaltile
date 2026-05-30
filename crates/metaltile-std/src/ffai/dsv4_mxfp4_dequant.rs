//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MXFP4 (OCP FP4 e2m1) block dequant.
//!
//! DSv4 MoE expert weights ship as native MXFP4 in the safetensors
//! checkpoint (separate code path from the GGUF IQ2_XXS recompression).
//! MXFP4 is the OCP Microscaling spec format:
//!
//! ```text
//!   struct block_mxfp4 {
//!     uint8_t e;        // E8M0 unsigned exponent (block super-scale)
//!     uint8_t qs[16];   // 32 packed 4-bit values, low nibble first
//!   };                  // 17 bytes per 32 values (BPW = 4.25)
//! ```
//!
//! Each 4-bit code maps to one of 16 signed magnitudes (1 sign + 2
//! exp + 1 mantissa per OCP MX v1.0):
//!
//! ```text
//!   code  | magnitude
//!   ------+----------
//!   0000  | +0
//!   0001  | +0.5
//!   0010  | +1.0
//!   0011  | +1.5
//!   0100  | +2.0
//!   0101  | +3.0
//!   0110  | +4.0
//!   0111  | +6.0
//!   1000  | -0
//!   1001  | -0.5
//!   1010  | -1.0
//!   1011  | -1.5
//!   1100  | -2.0
//!   1101  | -3.0
//!   1110  | -4.0
//!   1111  | -6.0
//! ```
//!
//! No NaN in the value table — only the E8M0 scale carries a NaN
//! sentinel (`0xFF`). The decoder downstream must skip a NaN-scaled
//! block; this kernel does NOT special-case that.
//!
//! Dequant: `out = scale_f32 * mxfp4_lut[code]`.
//!
//! ## GPU-resident split (loader produces these from raw blocks)
//!
//! 1. `qs_packed [n_blocks * 4]` — `u32`, 16 packed-byte payload re-laid
//!    as 4 LE u32 words per block (8 nibbles per word).
//! 2. `scales    [n_blocks]`     — `f32`, host-converted from E8M0
//!    (`scale = float_from_bits(uint32_t(e) << 23)` — Apple bit-cast
//!    is unavailable in the DSL, so the loader does this once at
//!    parse time).
//! 3. `lut       [16]`           — `f32`, the OCP-spec value table
//!    uploaded once at runtime init. Shared across all dequant calls.
//!
//! ## Apple GPU shape notes
//!
//! - 16-entry LUT in a `Tensor<f32>` (read via `load(lut[code])`).
//!   Apple's L1 multicast collapses the gather across lanes that
//!   land on the same code.
//! - 1 thread per output value. Adjacent threads in a simdgroup
//!   read consecutive nibbles within the same u32 word → coalesced
//!   loads.
//! - This kernel is the dequant primitive; a fused-with-routing
//!   MoE-GEMV variant lands in `dsv4_mxfp4_moe_gemv.rs` as a
//!   follow-up (dequant→fp16→GEMM materialization is the perf wall
//!   for MoE matmul, so the fused path skips the intermediate).

use metaltile::kernel;

// Bare `#[kernel]` — mixed-dtype param set (concrete u32/f32 + generic T).
#[kernel]
pub fn ffai_dsv4_mxfp4_dequant<T>(
    qs_packed: Tensor<u32>,
    scales: Tensor<f32>,
    lut: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 32u32;
        let in_block = i - block * 32u32; // 0..31
        // 32 nibbles per block, packed 8-per-u32 → word index ∈ [0..4), low-nibble-first.
        let word_idx = in_block / 8u32;
        let nibble_in_word = in_block & 7u32;
        let word = load(qs_packed[block * 4u32 + word_idx]);
        let code = (word >> (nibble_in_word * 4u32)) & 0xfu32;
        let mag = load(lut[code]);
        let s = load(scales[block]);
        // Implicit narrowing per playbook §"DSL implicit Store coercion".
        store(out[i], s * mag);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_mxfp4_dequant;
    use crate::utils::pack_f32;

    /// The canonical OCP-spec MXFP4 value table. Order matches the
    /// 4-bit code's binary interpretation (low 4 bits of nibble).
    const MXFP4_LUT: [f32; 16] =
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

    /// Reference quantizer: assign each input value to the nearest
    /// MXFP4 magnitude, picking a per-block E8M0 scale that maximises
    /// the representable range without saturating. Mirrors the
    /// reference OCP encoder closely enough for round-trip dequant
    /// invariance.
    fn quantize_mxfp4(values: &[f32]) -> (Vec<u32>, Vec<f32>) {
        assert_eq!(values.len() % 32, 0, "MXFP4 needs multiple-of-32 values");
        let n_blocks = values.len() / 32;
        let mut qs_packed = Vec::with_capacity(n_blocks * 4);
        let mut scales = Vec::with_capacity(n_blocks);
        // Max representable magnitude in MXFP4 is 6.0; scale picks the
        // exponent that maps `max_abs(block)` into that.
        for block in values.chunks_exact(32) {
            let amax = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            // E8M0 = unsigned 8-bit raw exponent, value 0xFF reserved
            // as NaN sentinel. We compute the matching exponent in
            // fp32 terms: `scale = 2^(e - 127)`, so `e = floor(log2(amax / 6)) + 127`.
            let target = if amax > 0.0 { amax / 6.0 } else { 1.0 };
            let raw_exp = (target.log2().floor() as i32 + 127).clamp(0, 254);
            let scale = (raw_exp - 127) as f32;
            let scale_f32 = (2.0_f32).powf(scale);
            scales.push(scale_f32);
            let inv = if scale_f32 > 0.0 { 1.0 / scale_f32 } else { 0.0 };
            let mut nibbles = [0u8; 32];
            for (i, &v) in block.iter().enumerate() {
                let normalised = v * inv;
                // Nearest-magnitude search over the 16-entry table.
                let mut best = 0u8;
                let mut best_err = f32::INFINITY;
                for (code, &mag) in MXFP4_LUT.iter().enumerate() {
                    let err = (normalised - mag).abs();
                    if err < best_err {
                        best_err = err;
                        best = code as u8;
                    }
                }
                nibbles[i] = best;
            }
            // Repack 32 nibbles into 4 LE u32 words.
            for w in 0..4 {
                let mut word = 0u32;
                for n in 0..8 {
                    word |= (nibbles[w * 8 + n] as u32) << (n * 4);
                }
                qs_packed.push(word);
            }
        }
        (qs_packed, scales)
    }

    fn cpu_dequant(qs_packed: &[u32], scales: &[f32]) -> Vec<f32> {
        let n_blocks = scales.len();
        let mut out = Vec::with_capacity(n_blocks * 32);
        for b in 0..n_blocks {
            let s = scales[b];
            for i in 0..32_usize {
                let word = qs_packed[b * 4 + i / 8];
                let nibble = (word >> ((i & 7) * 4)) & 0xf;
                out.push(s * MXFP4_LUT[nibble as usize]);
            }
        }
        out
    }

    fn setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 32;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0073 - 0.42).sin() * 2.7).collect();
        let (qs_packed, scales) = quantize_mxfp4(&values);
        let dequantized = cpu_dequant(&qs_packed, &scales);
        // Pack u32 vec as little-endian bytes for the test framework.
        let qs_bytes: Vec<u8> = qs_packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        TestSetup::new(ffai_dsv4_mxfp4_dequant::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_packed", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("lut", pack_f32(&MXFP4_LUT, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mxfp4_single_block(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mxfp4_many_blocks(dt: DType) -> TestSetup { setup(64, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_mxfp4_dequant;

    #[bench(name = "ffai/dsv4_mxfp4_dequant", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4(dt: DType) -> BenchSetup {
        // Representative MoE expert slab — 4096 × 2048 (DSv4 Flash
        // expert intermediate=2048, hidden=4096).
        let n = 4096 * 2048usize;
        let n_blocks = n / 32;
        BenchSetup::new(ffai_dsv4_mxfp4_dequant::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_packed", n_blocks * 4, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("lut", 16, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            // qs_packed 16 B + scale 4 B per block + lut 64 B once + output T.
            .bytes_moved(((n_blocks * 20 + 64) + n * dt.size_bytes()) as u64)
    }
}
