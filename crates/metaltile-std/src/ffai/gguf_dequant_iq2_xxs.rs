//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GGUF IQ2_XXS block dequant — i-quant 2-bit-per-weight with codebook lookup.
//!
//! Reference: `dequantize_row_iq2_xxs` in
//! [llama.cpp ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
//!
//! **WIP scaffold.** The kernel ABI + dispatch geometry + reference
//! quantizer are laid down so this file integrates with the rest of
//! the GGUF dequant pipeline today; the in-kernel codebook lookup is
//! stubbed pending a verbatim port of llama.cpp's `iq2xxs_grid` table
//! (256 × 8-byte signed-octet entries) — see TODO blocks below.
//!
//! ## On-disk block layout
//!
//! ```text
//!   struct block_iq2_xxs {
//!     uint16_t d;          // fp16 super-scale (2 bytes)
//!     uint16_t qs[32];     // 64 bytes — 8 groups of 32 outputs each
//!   };                      // 66 bytes per 256 values (BPW = 2.0625)
//! ```
//!
//! Per group of 32 outputs (8 groups per block):
//!
//! ```text
//!   // qs[4*g .. 4*g + 4]  = 2 uint16 = 1 uint32 of grid-index payload
//!   // qs[4*g + 4 .. 4*g + 8] = 2 uint16 = 1 uint32 of sign+scale payload
//!   aux32_idx   = u32 of qs[4*g + 0..4*g + 2]
//!   aux32_sgn   = u32 of qs[4*g + 2..4*g + 4]
//!   scale_4bit  = aux32_sgn >> 28                  // upper nibble of sign u32
//!   db          = d * (0.5 + scale_4bit) * 0.25     // per-group composite scale
//!   for j in 0..4:
//!     grid_idx  = (aux32_idx >> (8*j)) & 0xff       // byte j → 256-entry LUT key
//!     grid_row  = iq2xxs_grid[grid_idx]            // 8 signed octets (i8x8)
//!     sign_bits = (aux32_sgn >> (7*j)) & 0x7f       // 7-bit sign vector
//!     // Bit-popcount-parity expansion: the high bit is implicit XOR of
//!     // the low 7 bits. Reconstruct as ±1 per octet.
//!     for k in 0..8:
//!       sign  = sign_bits has bit k set ? -1 : +1
//!       out[32*g + 8*j + k] = db * sign * grid_row[k]
//! ```
//!
//! ## GPU-resident split (the loader produces these from the packed block)
//!
//! 1. `qs_u32   [n_blocks * 16]`  — `u32`, the 64 bytes of `qs[32]`
//!    re-laid as 16 little-endian u32 words. Each 32-element group
//!    consumes 2 u32 (indices + sign+scale).
//! 2. `d_f32    [n_blocks]`        — `f32`, host-converted from fp16.
//! 3. `grid     [256 * 8]`         — `i8`, the canonical `iq2xxs_grid`
//!    table from ggml-quants.c, repacked one signed octet per byte
//!    (8 bytes per row, 256 rows). Shared across all dequant calls —
//!    upload once at runtime init.
//!
//! ## Dispatch
//!
//! 1D grid over output values, but inner work is per-group-of-32 so
//! threads cooperate via the same (aux32_idx, aux32_sgn) pair.
//! Simplest shape: 1 thread per output, gather the relevant grid row
//! per output. Wasteful (8× redundant grid loads vs the per-group
//! pattern) but correctness-first; a per-group tile follow-up cuts
//! that.
//!
//! ## ABI
//!
//! ```text
//!   qs_u32   [n_blocks * 16]   u32   — packed grid-index + sign payloads
//!   d_f32    [n_blocks]        f32   — per-block super-scale
//!   grid     [2048]            i8    — iq2xxs_grid as 256 × 8 signed octets
//!   out      [n_values]        T     — dequantized output
//!   n_values u32 (constexpr)         — total output count = n_blocks * 256
//! ```

use metaltile::kernel;

#[kernel(
    bench(
        op="gguf_dequant",
        subop="iq2_xxs",
        class=GenericEmpty,
        tol=1e-3,
    )
)]
pub fn ffai_gguf_dequant_iq2_xxs<T>(
    qs_u32: Tensor<u32>,
    d_f32: Tensor<f32>,
    grid: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 256u32;
        let in_block = i - block * 256u32;
        let group = in_block / 32u32; // 0..7 — which of the 8 groups
        let in_group = in_block & 31u32; // 0..31 — position inside group
        let octet_within_index = in_group / 8u32; // 0..3 — which of 4 grid indices
        let lane_in_octet = in_group & 7u32; // 0..7 — which octet byte

        // Two u32s per group: indices (qs_u32[block*16 + group*2]) and
        // sign+scale payload (qs_u32[block*16 + group*2 + 1]).
        let aux_idx = load(qs_u32[block * 16u32 + group * 2u32]);
        let aux_sgn = load(qs_u32[block * 16u32 + group * 2u32 + 1u32]);

        let scale_4bit = aux_sgn >> 28u32;
        // db = d * (0.5 + scale_4bit) * 0.25 — composite per-group scale.
        let scale_factor = (scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25;
        let db = load(d_f32[block]) * scale_factor;

        // Grid lookup: byte `octet_within_index` of `aux_idx` is the
        // 256-entry LUT key.
        let grid_key = (aux_idx >> (octet_within_index * 8u32)) & 0xffu32;
        let grid_row_base = grid_key * 8u32;
        let octet_u = load(grid[grid_row_base + lane_in_octet]).cast::<u32>();
        let octet_signed = select(octet_u >= 128u32, octet_u - 256u32, octet_u);
        let octet = octet_signed.cast::<i32>().cast::<f32>();

        // Sign reconstruction: 7-bit sign field at bit (7 * octet_within_index)
        // in aux_sgn; the high bit is implicit-parity (XOR of low 7).
        let sign_field = (aux_sgn >> (octet_within_index * 7u32)) & 0x7fu32;
        let low_sign_bit = (sign_field >> lane_in_octet) & 1u32;
        // Implicit-parity bit for the 8th octet of each group of 8 —
        // XOR-parity of the low 7 sign bits. Inlined here because the
        // DSL only cross-calls between `#[kernel]`-registered fns.
        let parity = (((sign_field)
            ^ (sign_field >> 1u32)
            ^ (sign_field >> 2u32)
            ^ (sign_field >> 3u32)
            ^ (sign_field >> 4u32)
            ^ (sign_field >> 5u32)
            ^ (sign_field >> 6u32))
            & 1u32);
        let sign_bit = select(lane_in_octet == 7u32, parity, low_sign_bit);
        let sign = select(sign_bit == 1u32, -1.0f32, 1.0f32);

        // Implicit narrowing — see playbook §"DSL implicit Store
        // coercion" (no `.cast::<T>()` at the Store site).
        store(out[i], db * sign * octet);
    }
}

pub mod kernel_tests {
    //! **WIP**: end-to-end correctness ignored pending the verbatim
    //! `iq2xxs_grid` table port from llama.cpp. The kernel itself
    //! compiles + dispatches cleanly; the codegen-shape invariants
    //! below catch ABI / IR regressions even without the grid table.

    use metaltile::test::*;

    use super::ffai_gguf_dequant_iq2_xxs;
    use crate::utils::pack_f32;

    /// MSL emission smoke test — confirms the kernel codegens for all
    /// float output dtypes without a real input set. Catches IR /
    /// macro-expansion regressions; correctness is gated separately
    /// once `iq2xxs_grid` lands.
    #[test]
    fn codegen_iq2_xxs_smoke() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let ir = ffai_gguf_dequant_iq2_xxs::kernel_ir_for(dt);
            assert!(!ir.body.ops.is_empty(), "kernel body emitted no ops for {dt:?}");
            assert!(ir.params.iter().any(|p| p.name == "qs_u32"), "missing qs_u32 param");
            assert!(ir.params.iter().any(|p| p.name == "grid"), "missing grid param");
        }
    }

    /// TODO end-to-end correctness vs llama.cpp reference. Blocked on
    /// porting the 2048-byte `iq2xxs_grid` constant from
    /// `ggml-quants.c` (256 × 8 signed-octet table, source under MIT
    /// — needs verbatim copy + endian-pack into `Tensor<u8>` layout).
    /// Once the table lands, this fixture quantizes a known vector
    /// with the reference quantizer and asserts the kernel
    /// reproduces the dequant within tolerance.
    #[allow(dead_code)]
    fn _placeholder_setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 256;
        TestSetup::new(ffai_gguf_dequant_iq2_xxs::kernel_ir_for(dt))
            .input(TestBuffer::zeros("qs_u32", n_blocks * 16, DType::U32))
            .input(TestBuffer::from_vec(
                "d_f32",
                pack_f32(&vec![1.0; n_blocks], DType::F32),
                DType::F32,
            ))
            .input(TestBuffer::zeros("grid", 2048, DType::U8))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::zeros("out", n, dt))
            .grid_1d(n, 256)
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_dequant_iq2_xxs;

    #[bench(name = "ffai/gguf_dequant_iq2_xxs", dtypes = [f32, f16, bf16])]
    fn bench_iq2_xxs(dt: DType) -> BenchSetup {
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(ffai_gguf_dequant_iq2_xxs::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_u32", n_blocks * 16, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            .bytes_moved(((n_blocks * 64 + n_blocks * 4 + 2048) + n * dt.size_bytes()) as u64)
    }
}
