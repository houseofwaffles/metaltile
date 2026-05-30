//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-FP8 e4m3 dequant — DSv4 attention + router weight format.
//!
//! DSv4 attention + router weights ship as 1-byte-per-weight FP8
//! e4m3 storage with **per-(128×128)-block fp32 scales**.
//!
//! ```text
//!   weight    [M, N]                                 u8    — fp8 e4m3 bytes
//!   scales    [M/128, N/128]                         f32   — per-(128×128)-block scale
//!
//!   out[i, j] = fp8_lut[weight[i, j]] * scales[i/128, j/128]
//! ```
//!
//! FP8 e4m3: 1 sign + 4 exp + 3 mantissa, bias=7, max=±448, finite
//! everywhere except `0x7F` / `0xFF` (NaN). Bit-format conversion to
//! fp32 is ~6-8 ALU ops with denormal handling; on Apple GPUs (no
//! native FP8 type) a **256-entry LUT** (512 bytes total, host-
//! precomputed) is the proven faster path — confirmed by the
//! turboquant_plus M5 Max / M2 Pro analysis showing constant-cache
//! lookups cost 14% / 25% of decode time vs the arithmetic path.
//!
//! ## GPU-resident split (loader produces these from raw weights)
//!
//! 1. `weight_bytes [M * N]`                u8   — raw fp8 e4m3 bytes
//! 2. `scales       [M/128 * N/128]`        f32  — block scales
//! 3. `fp8_lut      [256]`                  f32  — `e4m3_to_fp32(byte)`
//!    table uploaded once at runtime init
//!
//! Output dtype follows T (fp32 / fp16 / bf16). The dequant stays in
//! f32 (scale × LUT product) before the implicit narrowing at store.
//!
//! ## Apple GPU shape
//!
//! 1 thread per output value, `m_dim` constexpr lets the kernel
//! address the (i/128, j/128) scale tile without per-thread shape
//! math. The LUT fits in constant cache; the scale gather is once
//! per 128×128 = 16384 outputs, so cache-multicast is effectively
//! free.
//!
//! Bench-stage matters: for matrix-vector decode (FFAI hot path),
//! the dequant-then-matmul fallback (which this kernel is the
//! dequant half of) ships fp16 weights to a downstream gemv. The
//! fused `ffai_dsv4_fp8_block_gemv` lands as a follow-up — Apple
//! has no native FP8 multiply, but a dequant-on-the-fly variant
//! that interleaves the LUT lookup with the gemv accumulator can
//! avoid materialising the full fp16 matrix.

use metaltile::kernel;

// Bare `#[kernel]` — mixed-dtype param set (concrete u8/f32 + generic T).
#[kernel]
pub fn ffai_dsv4_fp8_block_dequant<T>(
    weight_bytes: Tensor<u8>,
    scales: Tensor<f32>,
    fp8_lut: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] m_dim: u32,
    #[constexpr] n_dim: u32,
) {
    let i = tid;
    let total = m_dim * n_dim;
    if i < total {
        let row = i / n_dim;
        let col = i - row * n_dim; // i % n_dim
        // 128×128 block-scale grid. `cols_per_block = n_dim / 128`
        // is the number of column-blocks per row-block; reconstruct
        // by `i / 16384`-style integer division on the 1D flat
        // `[M/128, N/128]` scale buffer.
        let block_row = row / 128u32;
        let block_col = col / 128u32;
        let n_block_cols = n_dim / 128u32;
        let s = load(scales[block_row * n_block_cols + block_col]);

        let byte = load(weight_bytes[i]).cast::<u32>();
        let mag = load(fp8_lut[byte]);

        // Implicit narrowing per playbook §"DSL implicit Store coercion".
        store(out[i], s * mag);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_fp8_block_dequant;
    use crate::utils::pack_f32;

    /// Build the FP8 e4m3 byte → fp32 LUT. e4m3 layout:
    /// `[s (1)] [e (4)] [m (3)]` with `bias = 7`; subnormals + NaN
    /// per the IEEE-754-ish FP8 spec used by DeepSeek-V3.
    fn build_fp8_lut() -> [f32; 256] {
        let mut lut = [0.0f32; 256];
        for byte in 0..256_u32 {
            let sign = (byte >> 7) & 1;
            let exp = (byte >> 3) & 0xf;
            let mantissa = byte & 0x7;
            let sign_mult = if sign == 0 { 1.0 } else { -1.0 };
            let value: f32 = if exp == 0xf && mantissa == 0x7 {
                // 0x7F / 0xFF = NaN. Encode as f32 NaN so the
                // downstream NaN filter (if any) sees it.
                f32::NAN
            } else if exp == 0 {
                // Subnormal: mantissa / 2^9 (no implicit leading 1).
                sign_mult * (mantissa as f32) * (1.0 / 64.0) * (1.0 / 8.0)
            } else {
                // Normal: (1 + m/8) * 2^(e - 7).
                let m_frac = 1.0 + (mantissa as f32) / 8.0;
                let exp_scale = (2.0_f32).powf((exp as f32) - 7.0);
                sign_mult * m_frac * exp_scale
            };
            lut[byte as usize] = value;
        }
        lut
    }

    /// Reference quantizer: per (128×128) block compute the matching
    /// scale (`max_abs / 448`), encode each value to the nearest
    /// FP8 e4m3 byte via the LUT. Round-trip dequant should be
    /// quant-error-only, not pattern noise.
    fn quantize_block_fp8(values: &[f32], m: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(m % 128, 0, "block-FP8 needs M divisible by 128");
        assert_eq!(n % 128, 0, "block-FP8 needs N divisible by 128");
        let lut = build_fp8_lut();
        let block_rows = m / 128;
        let block_cols = n / 128;
        let mut bytes = vec![0u8; m * n];
        let mut scales = vec![0f32; block_rows * block_cols];
        for br in 0..block_rows {
            for bc in 0..block_cols {
                // amax over the 128×128 block.
                let mut amax = 0.0f32;
                for di in 0..128 {
                    for dj in 0..128 {
                        let v = values[(br * 128 + di) * n + bc * 128 + dj].abs();
                        if v > amax {
                            amax = v;
                        }
                    }
                }
                let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
                scales[br * block_cols + bc] = scale;
                let inv = 1.0 / scale;
                for di in 0..128 {
                    for dj in 0..128 {
                        let normalised = values[(br * 128 + di) * n + bc * 128 + dj] * inv;
                        // Nearest byte search. 256 bytes — brute force
                        // is fine for the test fixture (production
                        // quantization is offline anyway).
                        let mut best = 0u8;
                        let mut best_err = f32::INFINITY;
                        for byte in 0..256_u32 {
                            let mag = lut[byte as usize];
                            if !mag.is_finite() {
                                continue;
                            }
                            let err = (normalised - mag).abs();
                            if err < best_err {
                                best_err = err;
                                best = byte as u8;
                            }
                        }
                        bytes[(br * 128 + di) * n + bc * 128 + dj] = best;
                    }
                }
            }
        }
        (bytes, scales)
    }

    fn cpu_dequant(bytes: &[u8], scales: &[f32], m: usize, n: usize) -> Vec<f32> {
        let lut = build_fp8_lut();
        let block_cols = n / 128;
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let br = i / 128;
                let bc = j / 128;
                let scale = scales[br * block_cols + bc];
                let mag = lut[bytes[i * n + j] as usize];
                out[i * n + j] = scale * mag;
            }
        }
        out
    }

    fn setup(m: usize, n: usize, dt: DType) -> TestSetup {
        let values: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.00031 - 1.7).sin() * 5.0).collect();
        let (bytes, scales) = quantize_block_fp8(&values, m, n);
        let lut = build_fp8_lut();
        let dequantized = cpu_dequant(&bytes, &scales, m, n);
        TestSetup::new(ffai_dsv4_fp8_block_dequant::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("weight_bytes", bytes, DType::U8))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("fp8_lut", pack_f32(&lut, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("m_dim", m as u32)
            .constexpr("n_dim", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(m * n, 256)
    }

    /// Single-block (128×128) round-trip — the smallest valid shape.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_fp8_single_block(dt: DType) -> TestSetup { setup(128, 128, dt) }

    /// 4-block grid (256×256) — exercises the per-block-row/col
    /// stride math.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_fp8_2x2_blocks(dt: DType) -> TestSetup { setup(256, 256, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_fp8_block_dequant;

    #[bench(name = "ffai/dsv4_fp8_block_dequant", dtypes = [f32, f16, bf16])]
    fn bench_fp8(dt: DType) -> BenchSetup {
        // DSv4-Flash attention shape: wq_b is 1024 → 64*512 = 32768
        // (fused QK_nope + QK_rope across heads). Use 4096 × 4096 as
        // a representative slab — same shape as wq_a.
        let (m, n) = (4096usize, 4096usize);
        let block_rows = m / 128;
        let block_cols = n / 128;
        BenchSetup::new(ffai_dsv4_fp8_block_dequant::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("weight_bytes", m * n, DType::U8))
            .buffer(BenchBuffer::random("scales", block_rows * block_cols, DType::F32))
            .buffer(BenchBuffer::random("fp8_lut", 256, DType::F32))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m_dim", m as u32)
            .constexpr("n_dim", n as u32)
            .grid_1d(m * n, 256)
            // weights 1 B/elt + scale 4 B/(128*128 elts) + LUT 1 KB + output T
            .bytes_moved((m * n + block_rows * block_cols * 4 + 1024 + m * n * dt.size_bytes()) as u64)
    }
}
