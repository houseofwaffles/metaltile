//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! RMS normalization benchmark — #[kernel] DSL vs MLX metal/rms_norm.metal
//!
//! The kernel is generic over `N = tpg * 4` — each thread owns 4
//! consecutive elements, the partial sum-of-squares reduces across
//! the threadgroup. The bench wires `n=4096, tpg=1024` for the
//! hidden-axis case. For per-head normalisation (Qwen3-style q_norm
//! / k_norm pre-RoPE), the same kernel is dispatched as one
//! threadgroup per `(batch*token*n_heads)` row at `tpg = head_dim/4`
//! with the per-head_dim weight broadcast across all rows. The
//! per-head contract is pinned by
//! `tests/rms_norm_per_head_gpu.rs`.
//!
//! Models with head_dim < 128 (older 7B-class, head_dim=64) dispatch
//! [`mt_rms_norm_small`] instead, which uses a 2-elements-per-thread
//! layout so head_dim=64 still hits the tpg=32 minimum.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! output (best case) or pins the GPU in an infinite loop (worst
//! case — see FFAI post-mortem 2026-05-19). Consumers MUST encode
//! these as preconditions in their wrappers.
//!
//! - **`N = TPG * 4`.** Each thread owns exactly 4 consecutive
//!   elements of the row, loaded unconditionally at offsets
//!   `tid*4 + {0..3}`. The wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full Apple simdgroup).
//!   The cross-simdgroup combine reads `n_simd = TPG / 32` slots
//!   from threadgroup memory; with `TPG < 32` the combine reads
//!   zero everywhere and `tg_ssq` silently collapses to 0.
//! - **`TPG ≤ 1024`** (Apple's max-threads-per-threadgroup cap on
//!   M-series). Combined with `N = TPG*4`, this means `N ≤ 4096`;
//!   larger rows need the multi-row dispatch variant + chunking.
//! - **Combined**: `n` must be a multiple of 128 and `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row.** Multi-row dispatch uses
//!   `grid = (nRows * TPG, 1, 1)`, `tg = (TPG, 1, 1)`; Metal slices
//!   that into `nRows` threadgroups of `TPG` threads each.

use metaltile::kernel;

/// Cross-kernel callee: threadgroup-wide RMS inverse.
///
/// Given each thread's pre-computed `partial_ssq` (sum of squares for its
/// slice of the row), reduces across the threadgroup and returns:
///
/// ```text
///   rsqrt(reduce_sum(partial_ssq) / n + eps)
/// ```
///
/// This kernel exists **only** as a cross-kernel callee. Kernels that fuse
/// RMSNorm with a second operation (residual add, RoPE, quantized GEMV) call
/// it via the DSL cross-kernel syntax so that the reduction + rsqrt body is
/// expressed once and inlined by `KernelInlinePass` rather than copy-pasted.
///
/// ## Calling convention
///
/// ```rust
/// // In the caller kernel body (after computing per-thread partial_ssq):
/// let inv_rms = mt_rms_inv_scalar(partial_ssq, eps_buf, n);
/// ```
///
/// - `partial_ssq` → `KernelCallArg::Value`: the callee's param-load is
///   replaced by the caller's pre-computed scalar. No memory round-trip.
/// - `eps_buf`, `n` → `KernelCallArg::Tensor`: the callee's loads are kept
///   but renamed to the caller's buffer/constexpr names, so the inlined code
///   reads the correct per-kernel eps and row length.
/// - The output param `out` receives no arg; its store is skipped and the
///   stored `inv_rms` value is returned as the call result.
///
/// ## Standalone vs inlined semantics
///
/// `mt_rms_inv_scalar` is a **valid standalone kernel**: `partial_ssq` is a
/// real 1-element `Tensor<f32>` and `load(partial_ssq[0u32])` is a legal
/// memory access. It can be dispatched directly (e.g. in tests) by passing a
/// 1-element buffer containing the pre-summed partial sum.
///
/// When called via the cross-kernel DSL (`let inv = mt_rms_inv_scalar(g, ...)`)
/// the caller passes `g` as a `KernelCallArg::Value` — a pre-computed scalar
/// already in registers. `KernelInlinePass` detects the `Value` arg, skips the
/// load, and substitutes `g` directly, eliminating the memory round-trip.
/// This is load-forwarding: the callee is correct both ways.
#[kernel]
pub fn mt_rms_inv_scalar(
    partial_ssq: Tensor<f32>,
    eps_buf: Tensor<f32>,
    mut out: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let v = load(partial_ssq[0u32]); // replaced by Value arg at inline time
    let tg_ssq = reduce_sum(v);
    let eps = load(eps_buf[0u32]);
    store(out[0u32], rsqrt(tg_ssq / n + eps));
}

#[kernel]
pub fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // Each thread owns exactly 4 consecutive elements (N = TPG * 4).
    // The wrapper enforces this — but as belt-and-braces (the original
    // 2026-05-19 freeze came from a wrong-TPG dispatch in a sibling
    // kernel), clamp the load base for OOB threads and mask their SSQ
    // contribution + skip their stores. Threads with `col >= n` re-read
    // row[0..3] (benign, since `partial_ssq` for them is forced to 0),
    // participate in `reduce_sum` (required — Apple simdgroup
    // primitives need all lanes active), and skip their stores so
    // they don't trample a neighbouring row.
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col; // only used inside the in_bounds-guarded store block.
    // Read x once, cache in registers, reuse for both ssq and output — 3 reads total.
    let x0 = load(x[safe_base]).cast::<f32>();
    let x1 = load(x[safe_base + 1u32]).cast::<f32>();
    let x2 = load(x[safe_base + 2u32]).cast::<f32>();
    let x3 = load(x[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    // Mask OOB lanes to 0 contribution so `mean(x²) = tg_ssq / n` stays
    // correct: in-bounds lanes contribute their real x² values, the
    // sum/n divisor is unchanged. Only valid when the wrapper has
    // ensured the in-bounds lanes cover the full row exactly once;
    // duplicate / missing coverage is a wrapper bug we can't repair here.
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
        store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
        store(out[base + 2u32], (x2 * rms * load(w[col + 2u32]).cast::<f32>()).cast::<T>());
        store(out[base + 3u32], (x3 * rms * load(w[col + 3u32]).cast::<f32>()).cast::<T>());
    }
}

/// Small-head RMSNorm — 2 consecutive elements per thread, so
/// `N = tpg * 2`. Covers per-head dispatch at head_dim ∈ {64, 128,
/// 192, 256} (head_dim=64 → tpg=32 hits the single-simdgroup
/// minimum that the 4-element variant misses). At head_dim ≥ 128
/// the 4-element [`mt_rms_norm`] has better ILP per lane and is
/// preferred; this variant exists to cover the small-head_dim
/// regime (older 7B-class architectures) without a dispatch-time
/// fallback.
///
/// Algorithm-identical to `mt_rms_norm`: f32 accumulator for the
/// sum-of-squares, threadgroup-wide `reduce_sum`, `rsqrt(ssq/n + eps)`
/// scaling, per-element output store rounded through `T`.
#[kernel]
pub fn mt_rms_norm_small<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // 2 elements per thread → tpg = n / 2. The minimum supported is
    // tpg = 32 (one full simdgroup) → n ≥ 64.
    let base = rs + tid * 2u32;
    let col = tid * 2u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1;
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
}

/// Wide-row RMSNorm — handles rows wider than the 4096-element cap of
/// [`mt_rms_norm`]. Where `mt_rms_norm` fixes `N = TPG * 4` (so a
/// 1024-thread group tops out at 4096), this kernel has each thread
/// *stride* over the row in steps of one full threadgroup, so any `n`
/// is covered regardless of the threadgroup size. Needed for
/// large-hidden models (e.g. Gemma 4 31B, hidden 5376).
///
/// Two passes over device memory: pass 1 accumulates the strided
/// sum-of-squares and reduces it threadgroup-wide; pass 2 re-reads `x`
/// and writes the scaled output. The per-thread element count is
/// `ceil(n / TPG)` and varies with `n`, so the `x` values cannot be
/// held in registers across the reduction the way `mt_rms_norm` does
/// — hence the re-read. RMSNorm is memory-bound; the extra `x` read is
/// the price of unbounded `n`.
///
/// ## DISPATCH INVARIANTS
///
/// - **TPG a multiple of 32** (one full Apple simdgroup) so the
///   `reduce_sum` cross-simdgroup combine is well-defined. The wrapper
///   uses TPG = 1024. The stride is derived as `n_simd * 32`, so the
///   kernel is correct for any such TPG.
/// - **Grid: 1 threadgroup per row.** Multi-row dispatch uses
///   `grid = (nRows * TPG, 1, 1)`, `tg = (TPG, 1, 1)`.
/// - **`n` may be any positive value.** The strided loops bound on
///   `n`, so no `N = TPG * k` relationship is required; threads whose
///   stride walks past `n` simply stop. Unlike `mt_rms_norm` there is
///   no 128-alignment or `n ≤ 4096` requirement.
#[kernel]
pub fn mt_rms_norm_wide<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // One full threadgroup of threads; every thread strides by this.
    let tpg = n_simd * 32u32;
    // Pass 1: strided sum-of-squares. A thread with `tid >= n` runs
    // zero iterations and contributes 0 — still required to reach
    // `reduce_sum` (Apple simdgroup reductions need all lanes active).
    let mut acc = 0.0f32;
    for i in range(tid, n, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        acc = acc + xi * xi;
    }
    let tg_ssq = reduce_sum(acc);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    // Pass 2: strided scaled store. `x` is re-read from device memory
    // (see the doc note above).
    for i in range(tid, n, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        let wi = load(w[i]).cast::<f32>();
        store(out[rs + i], (xi * rms * wi).cast::<T>());
    }
}

/// Fused gated-mixer-norm: `out = rms_norm(y, w) · silu(z)`. Per-row
/// across `[Hv, Dv]` — one row per threadgroup. Used by the FFAI
/// Qwen3.5 / Qwen3.6 GDN mixer's phase-2 step (`y` is the recurrence
/// output in fp32; `z` is the gate from `in_proj_z` in the model
/// dtype; `w` is `mixer.norm.weight`). Folding RMSNorm + weight +
/// `silu(z)` into one dispatch kills the host round-trip the legacy
/// path needed to compute this on the CPU between phases — 30 host
/// commit+waits per Qwen3.6-A3B decode token recovered.
///
/// Math (one row):
///   rms = rsqrt(mean(y²) + eps)
///   y_normed[i] = y[i] * rms * w[i]
///   silu(z)[i]  = z[i] / (1 + exp(-z[i]))
///   out[i] = y_normed[i] * silu(z)[i]
///
/// Same `N = TPG * 4` invariant as `mt_rms_norm` — Dv is multiple of
/// 4 on every shipped Qwen3 hybrid (128 / 256 / 512). One thread owns
/// 4 consecutive `Dv`-axis elements; the OOB clamp + mask copies the
/// `mt_rms_norm` template so a wrong-TPG dispatch fails loudly rather
/// than silently miscomputing.
#[kernel]
pub fn mt_gated_mixer_norm<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    // y is already fp32, but mirror the mt_rms_norm load pattern
    // (`.cast::<f32>()` after each load) — the vectorize pass on this
    // codegen reads the cast as the consumer hook for the float4
    // load+extract emit. Removing the cast leaves the vectorize pass
    // half-finished (load merges into a float4, scalar y_n references
    // never get rewritten into VectorExtract — see emit + bug-report
    // in metaltile codegen `vectorize.rs`).
    let y0 = load(y[safe_base]).cast::<f32>();
    let y1 = load(y[safe_base + 1u32]).cast::<f32>();
    let y2 = load(y[safe_base + 2u32]).cast::<f32>();
    let y3 = load(y[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        let w0 = load(w[col]).cast::<f32>();
        let w1 = load(w[col + 1u32]).cast::<f32>();
        let w2 = load(w[col + 2u32]).cast::<f32>();
        let w3 = load(w[col + 3u32]).cast::<f32>();
        let z0 = load(z[base]).cast::<f32>();
        let z1 = load(z[base + 1u32]).cast::<f32>();
        let z2 = load(z[base + 2u32]).cast::<f32>();
        let z3 = load(z[base + 3u32]).cast::<f32>();
        // silu(z) = z / (1 + exp(-z)). Inlined per the `mt_sigmoid`
        // precedent — Activation::Sigmoid folds into FusedElementwise
        // and the per-kernel feature analyzer would miss it, so the
        // emitted MSL stays self-contained without an `mt_sigmoid`
        // helper. Same as `mt_gated_delta_prep_step`'s `beta` path.
        let silu0 = z0 / (1.0f32 + exp(0.0f32 - z0));
        let silu1 = z1 / (1.0f32 + exp(0.0f32 - z1));
        let silu2 = z2 / (1.0f32 + exp(0.0f32 - z2));
        let silu3 = z3 / (1.0f32 + exp(0.0f32 - z3));
        store(out[base], ((y0 * rms * w0) * silu0).cast::<T>());
        store(out[base + 1u32], ((y1 * rms * w1) * silu1).cast::<T>());
        store(out[base + 2u32], ((y2 * rms * w2) * silu2).cast::<T>());
        store(out[base + 3u32], ((y3 * rms * w3) * silu3).cast::<T>());
    }
}

#[cfg(test)]
mod wide_tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::mt_rms_norm_wide;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = mt_rms_norm_wide::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("mt_rms_norm_wide codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void mt_rms_norm_wide"),
                "MSL for {dt:?} should declare mt_rms_norm_wide:\n{src}",
            );
        }
    }
}

/// New-syntax correctness for `mt_rms_norm` (Reduction mode, one threadgroup
/// per row, `tpg = n/4` — `n` a multiple of 128). Per-row oracle on
/// dtype-rounded inputs: `out_i = x_i / sqrt(mean(x²) + eps) * w_i`. The
/// rms_norm_small/wide/gated variants stay for the complex-mlx PR.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_rms_norm, mt_rms_norm_wide};
    use crate::utils::{pack_f32, unpack_f32};

    // Per-row oracle: out_i = x_i / sqrt(mean(x²) + eps) * w_i, on
    // dtype-rounded inputs.
    fn expected_rms(
        x: &[f32],
        w_dt: &[f32],
        rows: usize,
        n: usize,
        eps: f32,
        dt: DType,
    ) -> Vec<f32> {
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let xr = unpack_f32(&pack_f32(&x[r * n..(r + 1) * n], dt), dt);
            let ms: f32 = xr.iter().map(|&v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            expected.extend(xr.iter().zip(w_dt).map(|(&xi, &wi)| xi * inv * wi));
        }
        expected
    }

    fn make_inputs(rows: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
        let mut x = Vec::with_capacity(rows * n);
        for r in 0..rows {
            x.extend((0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.03));
        }
        (w, x)
    }

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        let (w, x) = make_inputs(rows, n);
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let expected = expected_rms(&x, &w_dt, rows, n, eps, dt);
        TestSetup::new(mt_rms_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_mt_rms_norm(dt: DType) -> TestSetup { setup(4, 512, dt) }

    // Non-128-aligned wide-row coverage (SmolVLM2 d=960; 960 = 7·128 +
    // 64 is NOT a multiple of 128, so `mt_rms_norm` can't dispatch it).
    // `mt_rms_norm_wide` strides over the row with a fixed TPG=1024 and
    // imposes no `N = TPG·k` / 128-alignment constraint, so it covers
    // arbitrary widths. One threadgroup per row, TPG=1024.
    fn setup_wide(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        const TPG: u32 = 1024;
        let (w, x) = make_inputs(rows, n);
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let expected = expected_rms(&x, &w_dt, rows, n, eps, dt);
        TestSetup::new(mt_rms_norm_wide::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [TPG, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-4, 2e-2, 1e-1])]
    fn test_mt_rms_norm_wide_d960(dt: DType) -> TestSetup { setup_wide(3, 960, dt) }
}

/// New-syntax benchmark for `mt_rms_norm` (vs MLX `metal/rms_norm.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_gated_mixer_norm, mt_rms_norm, mt_rms_norm_small, mt_rms_norm_wide};

    #[bench(name = "mlx/rms_norm", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(mt_rms_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
    }

    // rms_norm_small: 2 elements per thread → tpg = n/2. Per-head shape
    // (head_dim=64, 1024 rows) matching the legacy bench(b=1024, n=64).
    #[bench(name = "mlx/rms_norm/rms_norm_small", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_small(dt: DType) -> BenchSetup {
        let (rows, n) = (1024usize, 64usize);
        BenchSetup::new(mt_rms_norm_small::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 2) as u32, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
    }

    // rms_norm_wide: strided over the row, one threadgroup (tpg=1024) per
    // row. Handles rows wider than the 4096 cap of mt_rms_norm — use a
    // large-hidden shape (n=5376, Gemma-class) to exercise the strided path.
    #[bench(name = "mlx/rms_norm/rms_norm_wide", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_wide(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 5376usize);
        const TPG: u32 = 1024;
        BenchSetup::new(mt_rms_norm_wide::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [TPG, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
    }

    // gated_mixer_norm: out = rms_norm(y, w) · silu(z). N = tpg*4. `y` is
    // fp32 (recurrence output); z/w/out in model dtype. GDN-mixer shape.
    #[bench(name = "mlx/rms_norm/gated_mixer_norm", dtypes = [f32, f16, bf16])]
    fn bench_gated_mixer_norm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 512usize);
        BenchSetup::new(mt_gated_mixer_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("y", rows * n, DType::F32))
            .buffer(BenchBuffer::random("z", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            .bytes_moved(((rows * n) * (DType::F32.size_bytes() + 2 * dt.size_bytes())) as u64)
    }
}
