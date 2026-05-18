//! Shared kernel utilities used by `build`, `inspect`, and `test` subcommands.

use metaltile_core::ir::KernelMode;
use metaltile_std::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Infer the `KernelMode` from a spec's dispatch variant.
///
/// For `Generic` dispatch with empty `shapes`, falls back to
/// `Elementwise` — codegen-only kernels with a non-Elementwise mode
/// should set `spec.kernel_mode = Some(...)` and use [`effective_mode`]
/// instead of calling this directly.
pub fn first_mode(spec: &BenchSpec) -> KernelMode {
    match &spec.dispatch {
        BenchDispatch::Generic => {
            spec.shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise)
        },
        BenchDispatch::Sort { .. }
        | BenchDispatch::Scan { .. }
        | BenchDispatch::ArgReduce { .. }
        | BenchDispatch::QuantizedMatVec { .. }
        | BenchDispatch::Attention { .. }
        | BenchDispatch::AffineQuantize { .. }
        | BenchDispatch::SdpaVector { .. } => KernelMode::Reduction,
        BenchDispatch::Random { .. }
        | BenchDispatch::FpQuantized { .. }
        | BenchDispatch::AffineDequantize { .. } => KernelMode::Elementwise,
        BenchDispatch::Rope { .. } | BenchDispatch::StridedCopy { .. } => KernelMode::Grid3D,
    }
}

/// The mode to actually use for codegen / display: prefer the spec's
/// explicit `kernel_mode` override, otherwise fall back to
/// [`first_mode`].
///
/// Codegen-only kernels (e.g. the FFAI ports in `ffai/`) set
/// `kernel_mode: Some(Reduction|Grid3D)` so the MSL header declares
/// the `tid`/`lsize`/`tgid_*` aliases their bodies depend on even
/// though dispatch is `Generic` with empty `shapes`. Subcommands that
/// emit MSL or display a mode label should call this helper, not
/// `first_mode` directly.
pub fn effective_mode(spec: &BenchSpec) -> KernelMode {
    spec.kernel_mode.unwrap_or_else(|| first_mode(spec))
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
