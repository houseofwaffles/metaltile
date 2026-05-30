//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_steel_gemm_splitk_nax` — split-K GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the two-kernel split-K GEMM.
//! Requires Metal 4 / macOS 26+ and Apple10+ hardware; runtime-gated
//! via `Context::chip_family()`.
//!
//! Split-K partitions the K dimension across the grid z-axis so a
//! skinny-M / skinny-N matmul with a very large K still saturates the
//! GPU. It is a **two-kernel** dispatch:
//!
//!   1. `mt_steel_gemm_splitk_nax` — each K-split computes a partial
//!      `[M, N]` product over its slice of K via cooperative `matmul2d`
//!      and writes it (fp32) to a `[n_splits, M, N]` partials buffer.
//!   2. `mt_steel_gemm_splitk_accum_nax` — reduces the `n_splits`
//!      partial `[M, N]` matrices into the final `[M, N]` output.
//!
//! Both kernels are expressed in the `#[kernel]` DSL — no `Op::InlineMsl`.
//! The split-K kernel is exactly `mt_steel_gemm_fused_nax` with a 3-D
//! grid: `tgid_z` selects the K-split and the K-loop walks only this
//! split's `[k_start, k_end)` range. The accumulator is fp32 so the
//! cross-split sum keeps full precision for f16/bf16 inputs — the
//! partials tensor is f32 regardless of the operand dtype.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for bf16 (Apple `matmul2d` mishandles
//! `bfloat` cooperative tensors), else `T`. Accumulation is fp32; the
//! partials slab is fp32 regardless of operand dtype.
//!
//! ## Geometry (mirrors `mt_steel_gemm_fused_nax`)
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile per TG.
//! - **Grid: `[n/32, m/32, n_splits]`** — `tgid_z` = K-split index.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile + one 16×16×32 MMA
//!   per K-block.
//! - **TG row stride = BK + 4 (skew) = 36** — bank-conflict avoidance.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`**; callers pad.
//! - **`k_per_split % 32 == 0`, `n_splits * k_per_split >= k`** — the
//!   K-loop is clamped to `k` so the last split may legally over-run.
//! - **`partials` is fp32, length `n_splits * m * n`**, `[split, M, N]`.
//! - **`KernelMode::Reduction`** so `tgid_*` lower to threadgroup indices.
//!
//! ## Accum kernel
//!
//! - **One thread per `[M, N]` output element** — grid `[m*n, 1, 1]`.
//! - **`partials` length `n_splits * m * n` (fp32)**, `out` length `m*n`.
//! - Accumulates via a `stack_alloc` fp32 register; final cast to `T`.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_splitk_nax_gpu_correctness.rs`.

use metaltile::kernel;

/// NAX split-K partial GEMM. Each `tgid_z` computes the partial
/// `[M, N]` product over `[k_start, k_end)` and writes (fp32) to
/// `partials[split, :, :]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_steel_gemm_splitk_nax<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    mut partials: Tensor<f32>,
    #[constexpr] m: u32,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] k_per_split: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // 2×2 warp grid.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    // This split's K-range. The last split may legally have
    // `k_start + k_per_split > k` — clamp via `min`.
    let split = tgid_z;
    let k_start = split * k_per_split;
    let k_end_raw = k_start + k_per_split;
    let k_end = min(k_end_raw, k);
    // Per-split partials base: this slab is `[split, M, N]`.
    let part_base = split * m * n;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("Ws", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG × 16 × 16
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    // Per-lane stage coordinates: 128 lanes × 8 elems = 1024 = 32 × 32.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    let b_n = w_n_base + x_m_row;
    // K-loop only over this split's range.
    for kb in range(k_start, k_end, 32u32) {
        let a_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let av = load(a[a_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, av);
        }
        let b_k_base = kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let bv = load(b[(b_k_base + _i) * n + b_n]).cast::<f32>();
            threadgroup_store("Ws", x_ws_base + _i, bv);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    // Write per-SG 16×16 fp32 result → this split's partials slab.
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(partials[part_base + (out_m_base + o_row) * n + (out_n_base + col)], v);
    }
}

/// NAX split-K accumulator. One thread per `[M, N]` output element;
/// sums `n_splits` partial slabs into the final `out` tensor.
#[kernel]
pub fn mt_steel_gemm_splitk_accum_nax<T>(
    partials: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] m: u32,
    #[constexpr] n: u32,
    #[constexpr] n_splits: u32,
) {
    let idx = tgid_x;
    let total = m * n;
    // Seed accumulator with split-0; loop adds the rest. fp32 throughout
    // so cross-split sums keep precision for f16/bf16 operands.
    let mut acc = load(partials[idx]);
    for s in range(1u32, n_splits, 1u32) {
        acc = acc + load(partials[s * total + idx]);
    }
    store(out[idx], acc.cast::<T>());
}

/// New-syntax benches for the two-kernel NAX split-K steel GEMM.
///
/// Pass 1 (`splitk_nax`) — `m = n = k = 4096` (multiples of 32),
/// `N_SPLITS = 4`, `k_per_split = k / N_SPLITS = 1024` (multiple of 32).
/// NAX geometry fixed: `BM = BN = BK = 32`, `tpg = 128`.
/// `KernelMode::Reduction`: grid is threadgroup counts
/// `(n/32, m/32, n_splits)` — `tgid_z` selects the K-split. Constexprs
/// are `m`, `k`, `n`, `k_per_split` (the kernel param order — note `k`
/// precedes `n`). The `partials` slab is fp32, `[split, M, N]`.
///
/// Pass 2 (`splitk_accum_nax`) — also `Reduction` (the kernel reads
/// `tgid_x` as the flat element index): grid `(m*n, 1, 1)`, one
/// threadgroup per `[M, N]` element. Constexprs `m`, `n`, `n_splits`.
///
/// `bytes_moved` counts the dominant streams. Bench-only — correctness
/// stays on `steel_gemm_splitk_nax_gpu_correctness.rs`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_steel_gemm_splitk_accum_nax, mt_steel_gemm_splitk_nax};

    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;
    const N_SPLITS: u32 = 4;
    /// Per-split K extent (`n_splits * k_per_split >= k`, multiple of 32).
    const K_PER_SPLIT: u32 = K / N_SPLITS;
    /// Fixed NAX tile dim and threads-per-group.
    const TILE: u32 = 32;
    const TPG: u32 = 128;

    // ── Pass 1 — NAX split-K partial GEMM (Reduction, 3-D grid) ────────────
    #[bench(name = "mlx/steel_gemm/splitk_nax", dtypes = [f32, f16, bf16])]
    fn bench_splitk_nax(dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let f32_sz = DType::F32.size_bytes();
        let bytes = (m * k + k * n) * sz + N_SPLITS as usize * m * n * f32_sz;
        BenchSetup::new(mt_steel_gemm_splitk_nax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            .buffer(BenchBuffer::zeros("partials", N_SPLITS as usize * m * n, DType::F32).output())
            // Kernel param order: m, k, n, k_per_split.
            .constexpr("m", M)
            .constexpr("k", K)
            .constexpr("n", N)
            .constexpr("k_per_split", K_PER_SPLIT)
            .with_shape_label(format!(
                "m{M} n{N} k{K} split{N_SPLITS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(N / TILE, M / TILE, N_SPLITS, [TPG, 1, 1])
            .bytes_moved(bytes as u64)
    }

    // ── Pass 2 — NAX partial-sum reduction (one TG per output element) ─────
    #[bench(name = "mlx/steel_gemm/splitk_accum_nax", dtypes = [f32, f16, bf16])]
    fn bench_splitk_accum_nax(dt: DType) -> BenchSetup {
        let (m, n) = (M as usize, N as usize);
        let sz = dt.size_bytes();
        let f32_sz = DType::F32.size_bytes();
        let bytes = N_SPLITS as usize * m * n * f32_sz + m * n * sz;
        BenchSetup::new(mt_steel_gemm_splitk_accum_nax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("partials", N_SPLITS as usize * m * n, DType::F32))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("m", M)
            .constexpr("n", N)
            .constexpr("n_splits", N_SPLITS)
            .with_shape_label(format!(
                "m{M} n{N} split{N_SPLITS} {}",
                crate::bench_types::dtype_label(dt)
            ))
            // One threadgroup per [M, N] element — grid (m*n, 1, 1).
            .grid_3d((m * n) as u32, 1, 1, [1, 1, 1])
            .bytes_moved(bytes as u64)
    }
}

/// New-syntax correctness tests for the two-kernel NAX split-K steel
/// GEMM — ports the oracle from
/// `tests/steel_gemm_splitk_nax_gpu_correctness.rs`. Each pass is pinned
/// independently (single dispatch per `#[test_kernel]`):
///   - **pass 1** (`mt_steel_gemm_splitk_nax`) — each `tgid_z` K-split
///     writes its partial `[M, N]` product (fp32) to `partials[split]`.
///     Oracle = per-split partial matmul over `[k_start, k_end)`.
///   - **pass 2** (`mt_steel_gemm_splitk_accum_nax`) — sums `n_splits`
///     partial slabs into `[M, N]`. Oracle = straight fp32 sum of seeded
///     partials, cast to `T`.
///
/// Pass-1 shape `M = N = 64`, `K = 128`, `N_SPLITS = 2`, `k_per_split =
/// 64` (multiples of 32 — NAX tile contract). `KernelMode::Reduction`,
/// grid threadgroup counts `(N/32, M/32, n_splits)` — `tgid_z` selects
/// the split — `tpg = [128, 1, 1]`; constexprs `m, k, n, k_per_split`
/// (note `k` precedes `n`). Pass 2 reads `tgid_x` as the flat element
/// index: grid `(M*N, 1, 1)`, `tpg = [1, 1, 1]`. NOTE: `matmul2d` only
/// runs correctly on Metal 4 / Apple10+ (pass-1).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_steel_gemm_splitk_accum_nax, mt_steel_gemm_splitk_nax};
    use crate::utils::{pack_f32, unpack_f32};

    /// Per-split partial fp32 reference.
    fn naive_splitk_partials(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        n_splits: usize,
        k_per_split: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_splits * m * n];
        for s in 0..n_splits {
            let k_start = s * k_per_split;
            let k_end = (k_start + k_per_split).min(k);
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f32;
                    for ki in k_start..k_end {
                        acc += a[mi * k + ki] * b[ki * n + ni];
                    }
                    out[s * m * n + mi * n + ni] = acc;
                }
            }
        }
        out
    }

    // ── Pass 1 — NAX split-K partial GEMM ──────────────────────────────────
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_splitk_nax_pass1(dt: DType) -> TestSetup {
        const TILE: u32 = 32;
        let (m, n, k) = (64usize, 64usize, 128usize);
        let (n_splits, k_per_split) = (2usize, 64usize);
        let a: Vec<f32> = (0..m * k).map(|i| 0.01 + (i as f32 % 17.0) * 0.013).collect();
        let b: Vec<f32> = (0..k * n).map(|i| -0.02 + (i as f32 % 13.0) * 0.011).collect();
        let a = unpack_f32(&pack_f32(&a, dt), dt);
        let b = unpack_f32(&pack_f32(&b, dt), dt);
        let expected = naive_splitk_partials(&a, &b, m, k, n, n_splits, k_per_split);
        TestSetup::new(mt_steel_gemm_splitk_nax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("partials", n_splits * m * n, DType::F32))
            // Kernel param order: m, k, n, k_per_split.
            .constexpr("m", m as u32)
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("k_per_split", k_per_split as u32)
            .expect(TestBuffer::from_vec(
                "partials",
                pack_f32(&expected, DType::F32),
                DType::F32,
            ))
            // 3-D grid: (n/32, m/32, n_splits). tgid_z = K-split.
            .grid_3d(n as u32 / TILE, m as u32 / TILE, n_splits as u32, [128, 1, 1])
    }

    // ── Pass 2 — NAX partial-sum reduction (one TG per output element) ─────
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 5e-2])]
    fn test_splitk_accum_nax(dt: DType) -> TestSetup {
        let (m, n, n_splits) = (32usize, 32usize, 3usize);
        let total = m * n;
        let partials: Vec<f32> =
            (0..n_splits * total).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let mut expected = vec![0.0f32; total];
        for (idx, e) in expected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for s in 0..n_splits {
                acc += partials[s * total + idx];
            }
            *e = acc;
        }
        let expected = unpack_f32(&pack_f32(&expected, dt), dt);
        TestSetup::new(mt_steel_gemm_splitk_accum_nax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("partials", pack_f32(&partials, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", total, dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("n_splits", n_splits as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            // One threadgroup per [M, N] element — grid (m*n, 1, 1).
            .grid_3d(total as u32, 1, 1, [1, 1, 1])
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn splitk_kernel_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_steel_gemm_splitk_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_splitk_nax");
            assert_eq!(k.params.len(), 3);
            assert_eq!(k.params[2].name, "partials");
            assert!(k.params[2].is_output);
            assert_eq!(k.params[2].dtype, DType::F32);
            assert_eq!(k.constexprs.len(), 4);

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileStoreC { .. })));
        }
    }

    #[test]
    fn accum_kernel_constructs() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_steel_gemm_splitk_accum_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_splitk_accum_nax");
            assert_eq!(k.params.len(), 2);
            assert_eq!(k.params[0].name, "partials");
            assert_eq!(k.params[0].dtype, DType::F32);
            assert_eq!(k.params[1].name, "out");
            assert!(k.params[1].is_output);
            assert_eq!(k.constexprs.len(), 3);

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        }
    }

    /// bf16 must stage through `half` for matmul2d compatibility.
    #[test]
    fn splitk_bf16_stages_through_half() {
        let k = mt_steel_gemm_splitk_nax::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half for matmul2d");
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_steel_gemm_splitk_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_splitk_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_splitk_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }

    #[test]
    fn codegen_emits_accum_kernel_decl() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = mt_steel_gemm_splitk_accum_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_splitk_accum_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_splitk_accum_nax_{suffix}")));
            assert!(!msl.contains("InlineMsl"));
        }
    }
}
