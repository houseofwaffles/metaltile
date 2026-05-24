//! Embedding-table gather. For each output element `(token, d)`: copy
//! `table[indices[token], d]`. One thread per output element.
//!
//! Bare-tensor (non-quantized) variant for embedding lookups.
//! Quantized embeddings live in `dequant_gather.rs`.
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gather",
    subop="gather",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
#[kernel]
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
