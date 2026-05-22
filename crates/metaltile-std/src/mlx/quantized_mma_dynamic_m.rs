//! Dynamic-M qmm path — host-side driver for batched-T quantized matmul.
//!
//! Closes the bandwidth-bound prefill gap in FFAI. The existing
//! `mt_qmm_mma`, `mt_qmm_mma_m16`, `mt_qmm_bm4`, `mt_qmm_bm2`, `mt_qmm`
//! kernels each handle a *fixed* M class — `mt_qmm_mma` requires `M % 32 == 0`,
//! `mt_qmm_mma_m16` is hard-wired to `M = 16`, etc. The model-level
//! prefill entry point (`Qwen35Model.forwardMany`) has a logical
//! token count `T` that is arbitrary (T=1 decode, T=37 ragged
//! chunk, T=4096 production prefill cell). Without a path that
//! accepts any `T` per dispatch, the model is forced into a
//! per-token loop and reads the full int4 weight tile once per
//! token — bandwidth-bound, 70× slower than MLX at T=32K.
//!
//! This module provides the host-side dispatcher that pads `T` to
//! the next multiple of 32 and routes to `mt_qmm_mma` for the full
//! BM=BN=BK=32 simdgroup-matrix tile. The kernel itself is unchanged
//! (one dispatch reads each int4 weight tile once and produces
//! `m_padded × N` outputs). The caller discards the trailing
//! `m_padded - T` rows of the output. Padding the X buffer with
//! zeros makes the masked rows valid (zero contribution to the
//! valid outputs) and the trailing rows numerically defined.
//!
//! Routing: `mt_qmm_mma` over `mt_qmm_mma_mpp`. The MPP variant only
//! ships fp32 / fp16 (see `quantized_mpp.rs` — the InlineMSL is
//! per-dtype templated and asserts `F32 | F16`). Production prefill
//! for Qwen3.6-A3B runs bf16, so we use the DSL-generic `mt_qmm_mma`
//! that supports `F32 | F16 | BF16` via `#[kernel]` generics. The
//! `dispatch_padded_grid` helper is dtype-agnostic.
//!
//! ## Composition with FFAI's `Ops.dequantGemm`
//!
//! The Swift-side `Ops.dequantGemm(x, w, scales, biases, ...)` calls
//! `mt_qmm_for(dtype, m)` today — which only handles fixed-class M.
//! After this lands, `Ops.dequantGemm` can call into the dynamic-M
//! path by:
//!   1. Padding `x` to `m_padded` rows (`(T + 31) / 32 * 32`).
//!   2. Calling `dispatch_padded_grid` with the padded shape.
//!   3. Slicing the first `T` rows of the output.
//!
//! No changes to the kernel binaries or the per-dtype emit are
//! needed — `mt_qmm_mma` is already in the kernel pack at every
//! shipped dtype.

use metaltile_core::{dtype::DType, ir::Kernel};

use crate::mlx::quantized::{mt_qmm_mma, patch_qmm_mma_dtype_aware_skew};

/// Tile geometry mirrors `mt_qmm_mma`. Exposed for callers sizing
/// the dispatch grid + the M-padding step.
pub const BM_TILE: u32 = 32;
pub const BN_TILE: u32 = 32;
pub const BK_TILE: u32 = 32;
/// Threads per group — 4 SG × 32 lanes — matches `mt_qmm_mma`.
pub const TPG: u32 = 128;

/// Round `t` up to the next multiple of [`BM_TILE`] (32). The
/// padded value is the `m` we hand to the kernel; the caller
/// discards the trailing `m_padded - t` output rows.
///
/// ```ignore
/// assert_eq!(pad_t_to_bm(1), 32);
/// assert_eq!(pad_t_to_bm(32), 32);
/// assert_eq!(pad_t_to_bm(33), 64);
/// assert_eq!(pad_t_to_bm(4096), 4096);
/// ```
pub const fn pad_t_to_bm(t: usize) -> usize {
    let bm = BM_TILE as usize;
    t.div_ceil(bm) * bm
}

/// Pad an X buffer `[t, k]` to `[m_padded, k]` by appending zero
/// rows. `x` is a row-major fp byte stream (`f32 = 4B`, `f16 = 2B`,
/// `bf16 = 2B`). The trailing rows are zero-filled so their
/// contribution to any output column is exactly zero — the kernel's
/// `Σ q · x_row + bias · Σ x_row` term collapses to `bias · 0 + 0`
/// on padded rows. (We discard those output rows anyway, but zero
/// padding is the defensible value.)
pub fn pad_x_rows_bytes(x_bytes: &[u8], t: usize, k: usize, bytes_per_elem: usize) -> Vec<u8> {
    let m_padded = pad_t_to_bm(t);
    let row_bytes = k * bytes_per_elem;
    assert_eq!(x_bytes.len(), t * row_bytes, "x_bytes must be t * k * bytes_per_elem");
    let mut out = Vec::with_capacity(m_padded * row_bytes);
    out.extend_from_slice(x_bytes);
    out.resize(m_padded * row_bytes, 0);
    out
}

/// Build the kernel IR for the dynamic-M path. Returns
/// `mt_qmm_mma::kernel_ir_for(dtype)` with the dtype-aware TG skew
/// patch applied (matches the path in `mt_qmm_for` for `M % 32 == 0`).
/// The caller dispatches with grid `[N / 32, m_padded / 32, 1]`
/// and `tpg = 128`.
pub fn kernel_ir_for(dtype: DType) -> Kernel {
    let mut k = mt_qmm_mma::kernel_ir_for(dtype);
    patch_qmm_mma_dtype_aware_skew(&mut k, dtype);
    k.mode = metaltile_core::ir::KernelMode::Reduction;
    k
}

/// Dispatch grid for the dynamic-M path given a *logical* token
/// count `t` and a row width `n`. Returns the threadgroup grid
/// `[N / 32, m_padded / 32, 1]`. Caller still owns `tpg = [128, 1, 1]`.
pub fn dispatch_grid(t: usize, n: usize) -> [usize; 3] {
    assert!(n.is_multiple_of(BN_TILE as usize), "n must be multiple of {} (BN tile)", BN_TILE);
    let m_padded = pad_t_to_bm(t);
    [n / BN_TILE as usize, m_padded / BM_TILE as usize, 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_t_to_bm_rounds_up_to_multiple_of_32() {
        assert_eq!(pad_t_to_bm(0), 0);
        assert_eq!(pad_t_to_bm(1), 32);
        assert_eq!(pad_t_to_bm(31), 32);
        assert_eq!(pad_t_to_bm(32), 32);
        assert_eq!(pad_t_to_bm(33), 64);
        assert_eq!(pad_t_to_bm(37), 64);
        assert_eq!(pad_t_to_bm(64), 64);
        assert_eq!(pad_t_to_bm(4096), 4096);
        assert_eq!(pad_t_to_bm(4097), 4128);
    }

    #[test]
    fn dispatch_grid_pads_m_axis() {
        // T=1 decode → 1 TG in M, N/32 TGs in N.
        assert_eq!(dispatch_grid(1, 128), [4, 1, 1]);
        // T=37 ragged → ceil(37/32) = 2 TGs in M.
        assert_eq!(dispatch_grid(37, 128), [4, 2, 1]);
        // T=4096 production → 128 TGs in M.
        assert_eq!(dispatch_grid(4096, 2048), [64, 128, 1]);
    }

    #[test]
    fn pad_x_rows_zero_fills_trailing() {
        // T=2, K=4, 2 bytes/elem (f16/bf16) → 16 bytes input.
        let x = vec![0x01u8; 16];
        let padded = pad_x_rows_bytes(&x, 2, 4, 2);
        // m_padded = 32, k=4, 2B → 256 bytes total.
        assert_eq!(padded.len(), 32 * 4 * 2);
        // First 16 bytes preserved.
        assert!(padded[..16].iter().all(|&b| b == 0x01));
        // Rest zero.
        assert!(padded[16..].iter().all(|&b| b == 0));
    }

    #[test]
    fn kernel_ir_for_returns_mt_qmm_mma_per_dtype() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_mma", "dynamic-M routes to mt_qmm_mma for dtype {:?}", dt);
            assert_eq!(k.mode, metaltile_core::ir::KernelMode::Reduction);
        }
    }
}
