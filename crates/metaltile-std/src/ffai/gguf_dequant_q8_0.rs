//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GGUF Q8_0 block dequant — `out[i] = qs[i] * d[i/32]`.
//!
//! Reference: `dequantize_row_q8_0` in
//! [llama.cpp ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
//!
//! ## On-disk block layout (decomposed CPU-side at load time)
//!
//! ```text
//!   struct block_q8_0 {
//!     uint16_t d;         // fp16 super-scale (2 bytes)
//!     int8_t   qs[32];    // 32 quantized int8 values (32 bytes)
//!   };                    // 34 bytes per block
//! ```
//!
//! GGUF blocks are tightly packed: block N starts at byte `34*N`. To
//! avoid in-kernel bit-cast / fp16-bit-reconstruction (which the
//! metaltile DSL doesn't expose today), the GGUF loader splits each
//! block into two GPU-resident tensors at parse time:
//!
//! 1. `qs_signed [n_blocks * 32]` — `u8` view of the original int8
//!    quants (signed reconstruction happens via the
//!    `select(q >= 128, q - 256, q)` trick inside the kernel; no
//!    arithmetic-shift / bit-cast intrinsic needed).
//! 2. `scales    [n_blocks]`      — `f32`, the fp16 super-scale
//!    converted to f32 by the host loader.
//!
//! That single conversion pass is `O(n_blocks)` and runs once at load.
//! The hot-path kernel below does ~0 setup work per output value.
//!
//! ## Dispatch
//!
//! 1D grid: one thread per *output value*. Thread `tid` computes
//! `block = tid / 32`, `lane = tid % 32`. Reads
//! `q = qs_signed[tid]` and `d = scales[block]`. Adjacent lanes share
//! the same scale cache line — Apple's L1 multicast hides the gather.
//!
//! ## ABI
//!
//! ```text
//!   qs_signed [n_values]   u8    — the 32 int8 quants per block, packed
//!                                  back-to-back (sign-reconstructed at use).
//!   scales    [n_blocks]   f32   — host-extracted block super-scales.
//!   out       [n_values]   T     — dequantized output.
//!   n_values  u32 (constexpr)    — total output count = n_blocks * 32.
//! ```

use metaltile::kernel;

#[kernel(
    bench(
        op="gguf_dequant",
        subop="q8_0",
        class=GenericEmpty,
        tol=1e-3,
    )
)]
pub fn ffai_gguf_dequant_q8_0<T>(
    qs_signed: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    // Bounds guard — caller may dispatch a power-of-two grid above
    // `n_values` for alignment reasons.
    if i < n_values {
        let block = i / 32u32;
        let q_u = load(qs_signed[i]).cast::<u32>();
        // Sign-extend the u8 to a signed f32 without a bit_cast: the
        // u8 value `q_u >= 128` represents a negative int8, value
        // `q_u - 256`. The `select` collapses to a `csel` in MSL.
        let q_signed = select(q_u >= 128u32, q_u - 256u32, q_u);
        let q = q_signed.cast::<i32>().cast::<f32>();
        let d = load(scales[block]);
        // Implicit narrowing — playbook §"DSL implicit Store coercion
        // beats explicit Op::Cast": no `.cast::<T>()` at the store
        // site avoids the spurious f32→f32 MSL cast (measured 8.3e-3
        // numerical drift) and keeps the Store eligible for any future
        // vectorize-window grouping.
        store(out[i], q * d);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gguf_dequant_q8_0;
    use crate::utils::pack_f32;

    /// Reference quantizer: mirrors `quantize_row_q8_0` in
    /// ggml-quants.c. Returns the two GPU-resident tensors the kernel
    /// expects: `(qs_signed, scales)`.
    fn quantize_q8_0(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(values.len() % 32, 0, "Q8_0 needs multiple-of-32 values");
        let n_blocks = values.len() / 32;
        let mut qs = Vec::with_capacity(n_blocks * 32);
        let mut scales = Vec::with_capacity(n_blocks);
        for block in values.chunks_exact(32) {
            let amax = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            // Match the precision the on-disk fp16 super-scale gives
            // us: round-trip through fp16 so the test's expected
            // values match what the loader hands the kernel.
            let d_f16 = half::f16::from_f32(d).to_f32();
            scales.push(d_f16);
            let inv_d = if d_f16 > 0.0 { 1.0 / d_f16 } else { 0.0 };
            for &v in block {
                let q = (v * inv_d).round().clamp(-128.0, 127.0) as i8;
                qs.push(q as u8);
            }
        }
        (qs, scales)
    }

    fn cpu_dequant(qs: &[u8], scales: &[f32]) -> Vec<f32> {
        let n_blocks = scales.len();
        let mut out = Vec::with_capacity(n_blocks * 32);
        for b in 0..n_blocks {
            let d = scales[b];
            for i in 0..32 {
                let q = qs[b * 32 + i] as i8;
                out.push(q as f32 * d);
            }
        }
        out
    }

    fn setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 32;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 0.4).sin() * 3.5).collect();
        let (qs, scales) = quantize_q8_0(&values);
        let dequantized = cpu_dequant(&qs, &scales);
        TestSetup::new(ffai_gguf_dequant_q8_0::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_signed", qs, DType::U8))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q8_0_single_block(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q8_0_many_blocks(dt: DType) -> TestSetup { setup(64, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_dequant_q8_0;

    #[bench(name = "ffai/gguf_dequant_q8_0", dtypes = [f32, f16, bf16])]
    fn bench_q8_0(dt: DType) -> BenchSetup {
        // 4096 × 4096 attn projection slab.
        let n = 4096 * 4096usize;
        let n_blocks = n / 32;
        BenchSetup::new(ffai_gguf_dequant_q8_0::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_signed", n, DType::U8))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            .bytes_moved(((n + n_blocks * 4) + n * dt.size_bytes()) as u64)
    }
}
