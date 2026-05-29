//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_steel_gemm_gather_nax` — gather GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the `nn` steel gather-GEMM
//! `C = A_gathered · B_gathered`:
//!
//!   - `lhs_indices[out_row]` — one `u32` per output row; redirects each
//!     output row to a (non-contiguous) `A` source row.
//!   - `rhs_indices[n_block]` — one `u32` per `BN`-wide N-block; selects
//!     which `[K, N]` `B` matrix this output block multiplies against.
//!     Selected matrix base = `rhs_indices[n_tile] * k * n`.
//!
//! This is the MLX `gather_mm` op — the dense-matmul half of a MoE FFN.
//! Requires Metal 4 / macOS 26+ and Apple10+ hardware; runtime-gated via
//! `Context::chip_family()`.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. It is exactly `mt_steel_gemm_fused_nax`
//! with two extra integer loads before the address arithmetic — the
//! gather index of an output row is a per-row scalar, the B-matrix index
//! a per-N-block scalar. No new codegen primitive is required.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for bf16 (Apple `matmul2d` mishandles
//! `bfloat` cooperative tensors), else `T`. Accumulation is fp32.
//!
//! ## Geometry (mirrors `mt_steel_gemm_fused_nax`)
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM=WN=2). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile per TG.
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile and runs one
//!   `16×16×32` `matmul2d` per K-block.
//! - **TG row stride = BK + 4 (skew) = 36** — scatter bank conflicts on
//!   the column reads inside `matmul2d`'s frag load.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`** — all loads
//!   unconditional; callers must pad.
//! - **`lhs_indices` length `m`** (one gathered `A`-row per output row),
//!   `u32`, each `< n_a_rows`. **`rhs_indices` length `n/32`** (one
//!   selected `B`-matrix per N-block), `u32`, each `< n_b_mats`. No
//!   bounds-check — callers keep indices in range.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_gather_nax_gpu_correctness.rs`.

use metaltile::kernel;

/// NAX gather GEMM `C[m,n] = Σ_k A[lhs[m], k] · B[rhs[n/32], k, n]`.
/// Params: `a [n_a_rows, k]`, `b [n_b_mats, k, n]`, `lhs_indices [m]`,
/// `rhs_indices [n/32]`, `out [m, n]`.
#[kernel(
    bench(
        op="steel_gemm",
        subop="gather_nax",
        class=GenericEmpty,
        tol=5e-2,
        kernel_mode=Reduction,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn mt_steel_gemm_gather_nax<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    lhs_indices: Tensor<u32>,
    rhs_indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // 2×2 warp grid: sm / sn pick this SG's 16×16 sub-tile.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
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
    // ── Gather: which A-row does this output-row pull from?
    let a_src_row = load(lhs_indices[x_m_base + x_m_row]);
    // ── Gather: which [K, N] B-matrix does this N-block multiply against?
    let b_mat = load(rhs_indices[tgid_x]);
    let b_base = b_mat * k * n;
    // N column this lane gathers from device B (transposed Ws read).
    let b_n = w_n_base + x_m_row;
    for kb in range(0u32, k, 32u32) {
        // Stage gathered A[a_src_row, kb + x_k_base..+8] → Xs.
        let a_row_dev_base = a_src_row * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let av = load(a[a_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, av);
        }
        // Stage gathered B^T[w_n_base + x_m_row, kb + x_k_base..+8] → Ws.
        // Device read: b[b_base + (kb + x_k_base + i) * n + b_n].
        let b_k_base = kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let bv = load(b[b_base + (b_k_base + _i) * n + b_n]).cast::<f32>();
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
    // Coop-write OutScratch → out. Destination row is contiguous (not
    // gathered) — the gather only redirects the *A* source rows.
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// New-syntax bench for the NAX gather steel GEMM (MoE `gather_mm`).
///
/// Canonical 4096³ problem (multiples of 32). NAX geometry fixed at
/// `BM = BN = BK = 32`, `tpg = 128`. `KernelMode::Reduction`: grid is
/// threadgroup counts `(n/32, m/32, 1)`. `lhs_indices` (length `m`) and
/// `rhs_indices` (length `n/32`) route the gather; both are seeded zero
/// (gather row / matrix 0 — in-bounds and deterministic). `b` is a
/// single `[K, N]` matrix here (one expert). `bytes_moved` counts the
/// three matmul streams plus the index reads. Bench-only — correctness
/// stays on `steel_gemm_gather_nax_gpu_correctness.rs`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_steel_gemm_gather_nax;

    const M: u32 = 4096;
    const N: u32 = 4096;
    const K: u32 = 4096;
    const TILE: u32 = 32;
    const TPG: u32 = 128;

    #[bench(name = "mlx/steel_gemm/gather_nax", dtypes = [f32, f16, bf16])]
    fn bench_gather_nax(dt: DType) -> BenchSetup {
        let (m, n, k) = (M as usize, N as usize, K as usize);
        let sz = dt.size_bytes();
        let n_blocks = n / TILE as usize;
        let bytes = (m * k + k * n + m * n) * sz + (m + n_blocks) * DType::U32.size_bytes();
        BenchSetup::new(mt_steel_gemm_gather_nax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("a", m * k, dt))
            .buffer(BenchBuffer::random("b", k * n, dt))
            .buffer(BenchBuffer::zeros("lhs_indices", m, DType::U32))
            .buffer(BenchBuffer::zeros("rhs_indices", n_blocks, DType::U32))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("k", K)
            .constexpr("n", N)
            .with_shape_label(format!("m{M} n{N} k{K} {}", crate::bench_types::dtype_label(dt)))
            .grid_3d(N / TILE, M / TILE, 1, [TPG, 1, 1])
            .bytes_moved(bytes as u64)
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_steel_gemm_gather_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_gather_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "lhs_indices");
            assert_eq!(k.params[3].name, "rhs_indices");
            assert_eq!(k.params[4].name, "out");
            assert!(k.params[4].is_output);
            assert_eq!(k.constexprs.len(), 2);

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileStoreC { .. })));
        }
    }

    /// bf16 must stage through `half` for matmul2d compatibility.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_steel_gemm_gather_nax::kernel_ir_for(DType::BF16);
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

    /// Codegen sanity — MPP header + descriptor + the gather index loads.
    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_steel_gemm_gather_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_gather_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_gather_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            assert!(msl.contains("lhs_indices"));
            assert!(msl.contains("rhs_indices"));
        }
    }
}
