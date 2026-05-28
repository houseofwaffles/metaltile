//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Kernel feature analysis.
//!
//! Scans the IR to determine which Metal features and helper functions
//! are needed for the generated MSL.

use metaltile_core::{
    dtype::DType,
    ir::{ActKind, Block, Kernel, Op, UnaryOpKind},
};

use super::MslGenerator;

// ---------------------------------------------------------------------------
// KernelFeatures
// ---------------------------------------------------------------------------

pub(super) struct KernelFeatures {
    pub has_tile: bool,
    pub is_matmul: bool,
    /// `true` iff the emitted MSL references `simd_lane` as a C identifier.
    /// Drives the `uint simd_lane [[thread_index_in_simdgroup]]`
    /// kernel-attr emission.  Set by the per-op `needs_simd_lane`
    /// OpFlag (which now lives only on `Op::SimdLaneId` after the
    /// #209/4 tightening) plus the multi-op cases that the OpFlag
    /// can't express statically: the two-level `Op::Reduce` slow
    /// path, the matmul tiled emit (`feat.is_matmul`), and
    /// `Op::CoopTile*` / `mpp::` `Op::InlineMsl` (cooperative MMA
    /// requires the same parameter for binding-table purposes).
    /// Pre-#209/4 the OpFlag was set by every simdgroup-related op
    /// regardless of whether the emit referenced the identifier; that
    /// over-broad shape produced ~200 `-Wunused-parameter` warnings
    /// against `simd_lane` until #207 routed around it.
    pub needs_simd_lane: bool,
    /// Symmetric to `needs_simd_lane` for `simd_group`.  See
    /// `needs_simd_lane`'s doc for the full rationale.  Same trigger
    /// set: `Op::SimdGroupId`, `Op::Reduce` slow path, matmul,
    /// `Op::CoopTile*` / MPP inline.
    pub needs_simd_group: bool,
    pub needs_simdgroup_matrix: bool,
    pub needs_bf16_struct: bool,
    pub needs_silu: bool,
    pub needs_gelu: bool,
    pub needs_relu: bool,
    pub needs_sigmoid: bool,
    pub needs_erf: bool,
    pub needs_erfinv: bool,
    pub needs_expm1: bool,
    pub needs_simd_product: bool,
    /// MetalPerformancePrimitives (`mpp::tensor_ops::matmul2d` / NAX) needed.
    /// Detected by scanning `Op::InlineMsl::source` for `"mpp::"` — kernels
    /// using NAX-class cooperative-tensor MMA must include the framework
    /// header. Requires macOS 26+ / Metal 4 toolchain.
    pub needs_mpp: bool,
}

impl MslGenerator {
    pub(super) fn analyze(&self, kernel: &Kernel) -> KernelFeatures {
        let mut feat = KernelFeatures {
            has_tile: false,
            is_matmul: false,
            needs_simd_lane: false,
            needs_simd_group: false,
            needs_simdgroup_matrix: false,
            needs_bf16_struct: false,
            needs_silu: false,
            needs_gelu: false,
            needs_relu: false,
            needs_sigmoid: false,
            needs_erf: false,
            needs_erfinv: false,
            needs_expm1: false,
            needs_simd_product: false,
            needs_mpp: false,
        };
        for p in &kernel.params {
            if p.dtype == DType::BF16 {
                feat.needs_bf16_struct = true;
            }
        }
        self.analyze_block(&kernel.body, &mut feat);
        for block in kernel.blocks.values() {
            self.analyze_block(block, &mut feat);
        }
        let tensor_2d = kernel.params.iter().filter(|p| p.shape.rank() == 2).count();
        feat.is_matmul = feat.has_tile && tensor_2d >= 2;

        // Matmul tiled emit (matmul.rs:173-174,311) references both
        // `simd_group` and `simd_lane` as identifiers — gate their
        // kernel-attr emission on `is_matmul`.  Pre-#209/4 the
        // signature gating lived in `msl::kernel_needs_simd_*_attr`
        // helpers that hand-rolled the same predicate; folding it
        // into `KernelFeatures` deletes those helpers.
        if feat.is_matmul {
            feat.needs_simd_lane = true;
            feat.needs_simd_group = true;
        }

        // `Op::Reduce` with `axis == 0` in `Reduction`/`Tile2D` mode
        // routes through the two-level threadgroup-reduction path
        // when the dispatched TPG is > simd_size — see `emit_reduce`
        // in `reduce.rs`.  That path emits `simd_lane == 0` and
        // `simd_group == 0` checks; the fast path (TPG ≤ simd_size)
        // emits a bare `simd_*(value)` call with no identifier ref.
        //
        // The OpFlag layer can't express the conditional ("only when
        // the slow path fires") because it depends on `expected_tpg`
        // at codegen time, not on the op itself.  So we special-case
        // it here at the feature-analysis layer, where the `config`
        // is in scope.
        if super::kernel_reduce_uses_n_simd(kernel, &self.config) {
            feat.needs_simd_lane = true;
            feat.needs_simd_group = true;
        }

        feat
    }

    pub(super) fn analyze_block(&self, block: &Block, feat: &mut KernelFeatures) {
        for op in &block.ops {
            self.analyze_op(op, feat);
        }
    }

    /// Per-op feature detection. Recurses into `FusedElementwise` so that
    /// helpers an inner op needs (e.g. `mt_silu` for an `Activation`
    /// folded into a fused chain) are still emitted — the fusion pass
    /// hides the standalone `Op::Activation` inside the chain.
    fn analyze_op(&self, op: &Op, feat: &mut KernelFeatures) {
        // --- Feature flags derived from OpFlags ---
        if op.needs_simd_lane() {
            feat.needs_simd_lane = true;
        }
        if op.needs_simd_group() {
            feat.needs_simd_group = true;
        }
        if op.needs_simdgroup_matrix() {
            feat.needs_simdgroup_matrix = true;
        }
        if op.needs_simd_product() {
            feat.needs_simd_product = true;
        }

        // --- Op-specific (data-dependent) checks ---
        match op {
            Op::Dot { .. } => feat.has_tile = true,
            Op::Reduce { op: reduce_kind, .. } | Op::Scan { op: reduce_kind, .. } => {
                if matches!(reduce_kind, metaltile_core::ir::ReduceKind::Product) {
                    feat.needs_simd_product = true;
                }
            },
            Op::StrideReduce { op: reduce_kind, .. } => {
                if matches!(reduce_kind, metaltile_core::ir::ReduceKind::Product) {
                    feat.needs_simd_product = true;
                }
            },
            Op::Load { src, indices, .. } if indices.is_empty() => {
                if src == "simd_lane" {
                    feat.needs_simd_lane = true;
                }
                // `simd_id` is a DSL synonym for `simd_group` (the
                // kernel-attr `[[simdgroup_index_in_threadgroup]]`).
                // Reading either name means the kernel signature
                // needs the attr.  `n_simd` is a *different* preamble
                // identifier (`uint n_simd = lsize / 32u;`) derived
                // from `lsize` only — it does NOT require the
                // `simd_group` kernel attr.  Pre-#209/4 these were
                // conflated, producing `-Wunused-parameter` warnings
                // on every kernel that referenced `n_simd` without
                // also using `simd_group`.
                if src == "simd_group" || src == "simd_id" {
                    feat.needs_simd_group = true;
                }
            },
            Op::Zeros { dtype, .. } | Op::Splat { dtype, .. } if *dtype == DType::BF16 => {
                feat.needs_bf16_struct = true;
            },
            Op::Cast { dtype, .. } if *dtype == DType::BF16 => {
                feat.needs_bf16_struct = true;
            },
            Op::Activation { kind, .. } => match kind {
                ActKind::Silu => feat.needs_silu = true,
                ActKind::Gelu => feat.needs_gelu = true,
                ActKind::Relu => feat.needs_relu = true,
                ActKind::Sigmoid => feat.needs_sigmoid = true,
                ActKind::Tanh => {},
            },
            Op::UnaryOp { op: UnaryOpKind::Erf, .. } => feat.needs_erf = true,
            Op::UnaryOp { op: UnaryOpKind::ErfInv, .. } => feat.needs_erfinv = true,
            Op::UnaryOp { op: UnaryOpKind::Expm1, .. } => feat.needs_expm1 = true,
            Op::FusedElementwise { ops } =>
                for inner in ops {
                    self.analyze_op(inner, feat);
                },
            Op::CoopTileSetup { .. }
            | Op::CoopTileZero { .. }
            | Op::CoopTileLoadA { .. }
            | Op::CoopTileLoadB { .. }
            | Op::CoopTileRun { .. }
            | Op::CoopTileStoreC { .. } => {
                feat.needs_mpp = true;
                // CoopTile / MPP cooperative-matmul intrinsics use
                // their own simdgroup binding internally and never
                // reference `simd_lane` / `simd_group` as C
                // identifiers in the emitted MSL.  Pre-#209/4 this
                // arm set both attrs anyway, producing
                // `-Wunused-parameter` warnings on every MPP kernel.
                // If a CoopTile kernel ever does spell those out (via
                // a Load with src="simd_lane" / "simd_group"), the
                // direct-identifier arm below catches it.
            },
            // Detect MPP tensor-ops usage in raw inline MSL — escape-hatch
            // for kernels that call `mpp::tensor_ops::matmul2d` / NAX.
            // Forces the codegen preamble to include the framework header.
            Op::InlineMsl { source, .. } if source.contains("mpp::") => {
                feat.needs_mpp = true;
            },
            _ => {},
        }
    }
}
