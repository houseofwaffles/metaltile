//! Multi-row GEMM — `out[r, :] = weight · input[r, :]` for a block of
//! `n_rows` rows in one dispatch. Generic over T.
//!
//! Used by Nemotron-Labs-Diffusion's block-diffusion / self-speculation
//! `forwardTokens`: a 32-token block runs 7 projections per layer
//! (q/k/v/o/gate/up/down). Done as N single-row `gemv`s the weight is
//! re-streamed once per row — N× the weight bandwidth. This kernel
//! tiles the output into 32×32 blocks and stages a `[32, 16]` weight
//! tile + `[32, 16]` input tile in threadgroup memory, so the weight
//! is read once and reused across all 32 rows of the tile.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel (threadgroup memory + barriers). No
//! `simd_*`-strided walk, so it is not exposed to the n_simd==0 freeze
//! — but it still has a fixed geometry the wrapper must honour:
//!
//! - **TPG = 1024 threads** (BM·BN = 32·32). The 1024 threads
//!   cooperatively load the two tiles (512 weight + 512 input
//!   elements) and then each computes one output element.
//! - **Grid: (out_dim/32) × (n_rows rounded up to /32) threadgroups**,
//!   2-D — `tgid_x` = output-column tile, `tgid_y` = row tile.
//! - **`in_dim % 16 == 0`** — the K loop strides by the 16-wide tile
//!   with no remainder handling.
//! - `weight` is `[out_dim, in_dim]`, `input` is `[n_rows, in_dim]`,
//!   `out` is `[n_rows, out_dim]`, all row-major.
//!
//! Output / row-count edges (`out_dim`, `n_rows` not multiples of 32)
//! are handled in-kernel: out-of-range loads clamp to index 0 and
//! contribute 0, out-of-range stores are skipped.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gemm",
    subop="gemm",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_gemm<T>(
    weight: Tensor<T>,
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] n_rows: u32,
) {
    // 32×32 output tile, 16-wide K tile. 1024 threads, one per output.
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32; // output row within the tile (0..31)
    let lo = tid % 32u32; // output col within the tile (0..31)

    // Weight tile [BN=32][BK=16] + input tile [BM=32][BK=16].
    threadgroup_alloc("gemm_w", 512);
    threadgroup_alloc("gemm_x", 512);

    let mut acc = 0.0f32;

    for k0 in range(0u32, in_dim, 16u32) {
        // Cooperative load: threads 0..511 fill the weight tile, threads
        // 512..1023 fill the input tile — one element each.
        if tid < 512u32 {
            let s = tid;
            let w_col = tgid_x * 32u32 + s / 16u32;
            let w_valid = w_col < out_dim;
            let w_col_safe = select(w_valid, w_col, 0u32);
            let w_raw = load(weight[w_col_safe * in_dim + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gemm_w", s, select(w_valid, w_raw, 0.0f32));
        }
        if tid >= 512u32 {
            let s = tid - 512u32;
            let x_row = tgid_y * 32u32 + s / 16u32;
            let x_valid = x_row < n_rows;
            let x_row_safe = select(x_valid, x_row, 0u32);
            let x_raw = load(input[x_row_safe * in_dim + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gemm_x", s, select(x_valid, x_raw, 0.0f32));
        }
        threadgroup_barrier();

        // Each thread accumulates its output element from the tiles.
        for k in range(0u32, 16u32, 1u32) {
            let w = threadgroup_load("gemm_w", lo * 16u32 + k);
            let x = threadgroup_load("gemm_x", lr * 16u32 + k);
            acc = acc + w * x;
        }
        threadgroup_barrier();
    }

    let r = tgid_y * 32u32 + lr;
    let o = tgid_x * 32u32 + lo;
    if r < n_rows {
        if o < out_dim {
            store(out[r * out_dim + o], acc.cast::<T>());
        }
    }
}
