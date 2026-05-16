//! Shared kernel utilities used by `build`, `inspect`, and `test` subcommands.

use metaltile_core::ir::KernelMode;
use metaltile_std::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Infer the `KernelMode` from a spec's dispatch variant.
pub fn first_mode(spec: &BenchSpec) -> KernelMode {
    match &spec.dispatch {
        BenchDispatch::Generic =>
            spec.shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise),
        BenchDispatch::Sort { .. }
        | BenchDispatch::Scan { .. }
        | BenchDispatch::ArgReduce { .. }
        | BenchDispatch::QuantizedMatVec { .. }
        | BenchDispatch::Attention { .. } => KernelMode::Reduction,
        BenchDispatch::Random { .. } | BenchDispatch::FpQuantized { .. } => KernelMode::Elementwise,
        BenchDispatch::Rope { .. } | BenchDispatch::StridedCopy { .. } => KernelMode::Grid3D,
    }
}

pub fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        _ => "?",
    }
}
