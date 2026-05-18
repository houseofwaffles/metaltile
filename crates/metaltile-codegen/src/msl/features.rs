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
            match op {
                Op::Dot { .. } => feat.has_tile = true,
                Op::Reduce { op: reduce_kind, .. } | Op::Scan { op: reduce_kind, .. } => {
                    feat.needs_simd_lane = true;
                    feat.needs_simd_group = true;
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
                // simdgroup matrix ops need simd built-ins and the simdgroup_matrix header
                Op::SimdgroupAlloc { .. }
                | Op::SimdgroupElemLoad { .. }
                | Op::SimdgroupElemStore { .. }
                | Op::SimdgroupMatMul { .. } => {
                    feat.needs_simd_lane = true;
                    feat.needs_simd_group = true;
                    feat.needs_simdgroup_matrix = true;
                },
                Op::SimdLaneId => feat.needs_simd_lane = true,
                Op::SimdGroupId => feat.needs_simd_group = true,
                Op::SimdReduce { .. } => {
                    feat.needs_simd_lane = true;
                    feat.needs_simd_group = true;
                },
                Op::SimdScan { .. } => {
                    feat.needs_simd_lane = true;
                    feat.needs_simd_group = true;
                },
                _ => {},
            }
        }
    }
}
