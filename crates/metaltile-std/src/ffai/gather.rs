//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Embedding-table gather. For each output element `(token, d)`: copy
//! `table[indices[token], d]`. One thread per output element.
//!
//! Bare-tensor (non-quantized) variant for embedding lookups.
//! Quantized embeddings live in `dequant_gather.rs`.
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;

#[kernel(
    bench(
        op="gather",
        subop="gather",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Grid3D,
    )
)]
pub fn ffai_gather<T>(
    table: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] dim: u32,
) {
    let idx = program_id::<0>();
    let token = idx / dim;
    let d = idx - token * dim;
    let token_id = load(indices[token]);
    let src = token_id * dim + d;
    store(out[idx], load(table[src]));
}

/// New-syntax correctness for `ffai_gather` (Grid3D, one thread per output
/// element). Pure copy — expected output is the gathered rows
/// `out[token, d] = table[indices[token], d]`, exact (tol 0).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gather;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_ffai_gather(dt: DType) -> TestSetup {
        let (vocab, dim, n_tokens) = (17usize, 8usize, 6usize);
        // table[r, d] = r * 1000 + d — a wrong token/dim decomposition
        // cross-contaminates immediately.
        let table: Vec<f32> =
            (0..vocab * dim).map(|i| ((i / dim) * 1000 + (i % dim)) as f32).collect();
        let table_dt = unpack_f32(&pack_f32(&table, dt), dt);
        let indices: Vec<u32> = vec![3, 0, 11, 7, 11, 16];
        let n_elems = n_tokens * dim;
        let expected: Vec<f32> = (0..n_elems)
            .map(|idx| {
                let token = idx / dim;
                let d = idx - token * dim;
                table_dt[indices[token] as usize * dim + d]
            })
            .collect();
        TestSetup::new(ffai_gather::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("table", pack_f32(&table, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", n_elems, dt))
            .constexpr("dim", dim as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_elems, 256)
    }
}

/// New-syntax benchmark for `ffai_gather` (embedding-table lookup, Qwen-class
/// dim, one thread per output element).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gather;

    fn u32_bytes(v: impl Iterator<Item = u32>) -> Vec<u8> {
        v.flat_map(|x| x.to_le_bytes()).collect()
    }

    #[bench(name = "ffai/gather/gather", dtypes = [f32, f16, bf16])]
    fn bench_gather(dt: DType) -> BenchSetup {
        let (vocab, dim, n_tokens) = (8192usize, 4096usize, 1024usize);
        let n_elems = n_tokens * dim;
        BenchSetup::new(ffai_gather::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("table", vocab * dim, dt))
            .buffer(BenchBuffer::from_vec(
                "indices",
                u32_bytes((0..n_tokens).map(|t| (t % vocab) as u32)),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", n_elems, dt).output())
            .constexpr("dim", dim as u32)
            .grid_1d(n_elems, 256)
            // One row read + one row written per output element.
            .bytes_moved((2 * n_elems * dt.size_bytes()) as u64)
    }
}
