//! Embedding-table gather (FFAI-specific). For each output element
//! `(token, d)`: copy `table[indices[token], d]`. One thread per output
//! element.
//!
//! Upstream has no equivalent yet — this is FFAI's bare-tensor lookup
//! used for non-quantized embeddings. Quantized embeddings live in
//! `dequant_gather.rs`.

use metaltile::kernel;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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

inventory::submit! {
    BenchSpec {
        op: "gather",
        subop: "ffai_gather",
        kernel_name: "ffai_gather",
        kernel_ir: ffai_gather::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: None,
    }
}
