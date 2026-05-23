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
    pub needs_simd_lane: bool,
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
                // `n_simd` is the number of simdgroups per threadgroup —
                // it is derived from `lsize / 32` and emitted in the Reduction
                // mode preamble alongside `simd_group`. Any kernel that reads
                // `n_simd` also needs that preamble block.
                if src == "simd_group" || src == "simd_id" || src == "n_simd" {
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
                feat.needs_simd_lane = true;
                feat.needs_simd_group = true;
            },
            // Detect MPP tensor-ops usage in raw inline MSL — escape-hatch
            // for kernels that call `mpp::tensor_ops::matmul2d` / NAX.
            // Forces the codegen preamble to include the framework header.
            // MPP MMA is simdgroup-cooperative — pulls in the same simd
            // built-ins as the simdgroup_matrix path.
            // CoopTile* ops use cooperative matmul — force the MPP framework header.
            Op::InlineMsl { source, .. } if source.contains("mpp::") => {
                feat.needs_mpp = true;
                feat.needs_simd_lane = true;
                feat.needs_simd_group = true;
            },
            _ => {},
        }
    }
}
