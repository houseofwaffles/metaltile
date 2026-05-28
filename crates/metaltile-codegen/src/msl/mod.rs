//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MSL (Metal Shading Language) code generator.
//!
//! Walks the MetalTile IR and emits valid MSL source text.
//! Handles constexpr params, tiled matmul, vectorized loads, and thread indexing.

pub(crate) mod config;
mod emit_block;
pub(crate) mod features;
mod fused;
mod helpers;
pub(crate) mod matmul;
pub(crate) mod preamble;
pub(crate) mod reduce;

use std::{collections::BTreeMap, fmt::Write};

pub use config::MslConfig;
use config::TileSchedule;
use features::KernelFeatures;
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode, Op, ParamKind, ValueId},
};

use crate::{error::Result, passes, passes::type_check::infer_types};

#[macro_export]
macro_rules! wl {
    ($out:expr) => {{ let _ = writeln!($out); }};
    ($out:expr, $($arg:tt)*) => {{ let _ = writeln!($out, $($arg)*); }};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return `true` if any op in the kernel (body + all child blocks) is
/// `Op::ProgramId { axis }` for the given axis value.
///
/// Also catches the direct-identifier form (`tgid_y` written verbatim in DSL
/// source) which the macro lowers to `Op::Load { src: "tgid_y", ... }` rather
/// than `Op::ProgramId`. Both forms reach the same `tgid_y` reference in the
/// emitted MSL, so the preamble must declare the alias in either case.
fn kernel_uses_program_id_axis(kernel: &Kernel, axis: u32) -> bool {
    // Match either Op::ProgramId { axis } or a direct Op::Load { src: "tgid_y" }
    // (DSL kernels that use `tgid_y` directly instead of `program_id::<1>()`).
    let tgid_name = match axis {
        0 => "tgid_x",
        1 => "tgid_y",
        2 => "tgid_z",
        _ => "",
    };
    let check = |ops: &[Op]| {
        ops.iter().any(|op| {
            op.program_id_axis() == Some(axis)
                || (op.load_src() == Some(tgid_name) && op.load_indices().is_empty())
        })
    };
    check(&kernel.body.ops) || kernel.blocks.values().any(|b| check(&b.ops))
}

/// Return `true` if the kernel directly reads the named scalar identifier
/// — i.e. it has an `Op::Load { src: name, indices: [] }` somewhere. This
/// is how the DSL parser lowers a bare identifier read (`let x = tid;` →
/// `Op::Load { src: "tid", indices: [] }`).
fn kernel_uses_identifier(kernel: &Kernel, name: &str) -> bool {
    let check = |ops: &[Op]| {
        ops.iter().any(|op| op.load_src() == Some(name) && op.load_indices().is_empty())
    };
    check(&kernel.body.ops) || kernel.blocks.values().any(|b| check(&b.ops))
}

/// Return `true` if `name` is referenced anywhere in the kernel IR as a
/// load source, store destination, or other named target.  Used to
/// decide whether a kernel-signature parameter would be flagged
/// `-Wunused-parameter` at the MSL `metal -W` stage — DSL authors
/// commonly declare constexpr / scalar params for documentation or
/// runtime-assert hooks even when the kernel body doesn't consult
/// them, and the body parser keeps those decls on the kernel
/// signature for ABI consistency.
fn kernel_references_param(kernel: &Kernel, name: &str) -> bool {
    let op_refs = |op: &Op| -> bool {
        if op.load_src() == Some(name) || op.store_dst() == Some(name) {
            return true;
        }
        // Threadgroup / stack / local ops use a `name` field instead of
        // `src` / `dst` (e.g. `threadgroup_store("xs", …)`).
        if match op {
            Op::ThreadgroupAlloc { name: n, .. }
            | Op::ThreadgroupStore { name: n, .. }
            | Op::ThreadgroupLoad { name: n, .. }
            | Op::StackAlloc { name: n, .. }
            | Op::StackStore { name: n, .. }
            | Op::StackLoad { name: n, .. }
            | Op::DeclareLocal { name: n, .. }
            | Op::SetLocal { name: n, .. } => n == name,
            _ => false,
        } {
            return true;
        }
        // FusedElementwise chains can hide inner Loads — recurse into
        // the sub-op vector.
        if let Op::FusedElementwise { ops } = op {
            for inner in ops {
                if inner.load_src() == Some(name) {
                    return true;
                }
            }
        }
        false
    };
    for block in kernel.iter_blocks() {
        if block.ops.iter().any(op_refs) {
            return true;
        }
    }
    false
}

/// Return `true` if the kernel contains any `Op::StrideReduce`.  In
/// `Reduction` / `Tile2D` modes the emitter lowers `StrideReduce` to a
/// vectorized loop that references `tid` and `lsize` (see the `has_tid`
/// branch of the `Op::StrideReduce` arm in `emit_block.rs`).  Outside
/// those modes the lowering is a plain serial loop and references
/// neither — but `Reduction` is the only mode whose preamble is gated
/// by these predicates, so we can use the unqualified "is any
/// StrideReduce present" check and let the mode guard at the call site
/// do the rest.
fn kernel_has_stride_reduce(kernel: &Kernel) -> bool {
    let check = |ops: &[Op]| ops.iter().any(|op| matches!(op, Op::StrideReduce { .. }));
    check(&kernel.body.ops) || kernel.blocks.values().any(|b| check(&b.ops))
}

/// Return `true` if the kernel contains any `Op::Reduce` with kind
/// `Mean`.  `lsize` is referenced in the reduce emit ONLY for the Mean
/// kind: the final step divides the simdgroup/threadgroup total by
/// `float(lsize)` in both the fast and slow paths (see `reduce.rs`).
/// Other reduce kinds (Sum/Max/Min/Product) lower to a bare
/// `simd_*(value)` / `__mt_simd_product(value)` call with no `lsize`
/// reference.
fn kernel_has_reduce_mean(kernel: &Kernel) -> bool {
    let check = |ops: &[Op]| {
        ops.iter()
            .any(|op| matches!(op, Op::Reduce { op: metaltile_core::ir::ReduceKind::Mean, .. }))
    };
    check(&kernel.body.ops) || kernel.blocks.values().any(|b| check(&b.ops))
}

/// Return `true` if any `Op::Reduce` in the kernel will go through the
/// two-level threadgroup reduction path that references `n_simd`.  That
/// path fires for `axis == 0` in `Reduction` / `Tile2D` mode when
/// `expected_tpg` is `None` or greater than the simd width — see
/// `MslGenerator::emit_reduce` in `reduce.rs`.
pub(super) fn kernel_reduce_uses_n_simd(kernel: &Kernel, config: &MslConfig) -> bool {
    let tg_modes = matches!(
        kernel.mode,
        metaltile_core::ir::KernelMode::Reduction | metaltile_core::ir::KernelMode::Tile2D
    );
    if !tg_modes {
        return false;
    }
    // Slow path fires unless we statically know TPG fits in one simdgroup.
    let single_simdgroup = matches!(config.expected_tpg, Some(t) if t <= config.simd_size);
    if single_simdgroup {
        return false;
    }
    let has_axis0_reduce =
        |ops: &[Op]| ops.iter().any(|op| matches!(op, Op::Reduce { axis: 0, .. }));
    has_axis0_reduce(&kernel.body.ops) || kernel.blocks.values().any(|b| has_axis0_reduce(&b.ops))
}

/// Precomputed flags for Reduction-mode signature + preamble emission.
/// Each field is `true` when the corresponding MSL identifier
/// (`tid`/`tgid_x`/`lsize`/`n_simd`) has a real consumer in the emitted
/// MSL — both the kernel-parameter attribute AND the preamble alias
/// gate on the same value so the signature and body stay in sync.
/// Pre-fix, the signature unconditionally declared `_tid3`/`_tgid3`/
/// `_lsize3` and the preamble unconditionally emitted their aliases —
/// any kernel that didn't use them produced `-Wunused-parameter` (on
/// the attr) AND `-Wunused-variable` (on the alias).  Gating both off
/// the same predicate eliminates both.
struct ReductionPreambleGates {
    needs_tid: bool,
    needs_tgid_x: bool,
    needs_tgid_y: bool,
    needs_tgid_z: bool,
    needs_lsize: bool,
    needs_n_simd: bool,
}

impl ReductionPreambleGates {
    fn compute(kernel: &Kernel, config: &MslConfig) -> Self {
        let has_stride_reduce = kernel_has_stride_reduce(kernel);
        let has_reduce_mean = kernel_has_reduce_mean(kernel);
        let needs_n_simd =
            kernel_reduce_uses_n_simd(kernel, config) || kernel_uses_identifier(kernel, "n_simd");
        let needs_lsize = has_stride_reduce
            || has_reduce_mean
            || needs_n_simd
            || kernel_uses_identifier(kernel, "lsize");
        let needs_tid = has_stride_reduce || kernel_uses_identifier(kernel, "tid");
        Self {
            needs_tid,
            needs_tgid_x: kernel_uses_program_id_axis(kernel, 0),
            needs_tgid_y: kernel_uses_program_id_axis(kernel, 1),
            needs_tgid_z: kernel_uses_program_id_axis(kernel, 2),
            needs_lsize,
            needs_n_simd,
        }
    }

    /// `_tid3` kernel-attr parameter (`uint3 _tid3
    /// [[thread_position_in_threadgroup]]`) is consumed only by
    /// `tid = _tid3.x`.
    fn needs_tid3_attr(&self) -> bool { self.needs_tid }

    /// `_tgid3` is consumed by `tgid_x` / `tgid_y` / `tgid_z` aliases.
    fn needs_tgid3_attr(&self) -> bool {
        self.needs_tgid_x || self.needs_tgid_y || self.needs_tgid_z
    }

    /// `_lsize3` is consumed by `lsize = _lsize3.x` (and `n_simd` via
    /// `lsize / 32u`).
    fn needs_lsize3_attr(&self) -> bool { self.needs_lsize }
}

// Signature gating for `uint simd_lane [[thread_index_in_simdgroup]]`
// and `uint simd_group [[simdgroup_index_in_threadgroup]]` lives on
// `KernelFeatures::needs_simd_lane` / `needs_simd_group` (see
// `features.rs`).  Pre-#209/4 hand-rolled `kernel_needs_simd_*_attr`
// predicates duplicated the work; the OpFlag layer now reflects actual
// MSL identifier consumption directly.

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct MslGenerator {
    config: MslConfig,
}

impl MslGenerator {
    pub fn new(config: MslConfig) -> Self { MslGenerator { config } }

    /// Like [`generate`] but also returns per-pass statistics.
    #[tracing::instrument(skip(self, kernel), fields(kernel = %kernel.name))]
    pub fn generate_with_stats(&self, kernel: &Kernel) -> Result<(String, Vec<passes::PassStats>)> {
        let mut k = kernel.clone();
        let stats = passes::run_passes_with_stats(&mut k, &passes::standard_pipeline())?;
        let msl = self.emit_msl(&k)?;
        Ok((msl, stats))
    }

    #[tracing::instrument(skip(self, kernel), fields(kernel = %kernel.name))]
    pub fn generate(&self, kernel: &Kernel) -> Result<String> {
        // Run the optimization pipeline on a clone before emitting.
        let mut k = kernel.clone();
        passes::run_passes_with_stats(&mut k, &passes::standard_pipeline())?;
        // Per-kernel opt-in overrides the default-off
        // `bfloat_reinterpret_cast` config. See the field doc on
        // `Kernel` for why this is opt-in (truncation vs rounding
        // trade-off — safe for SDPA-prefill MMA, unsafe for tight-
        // tolerance kernels like rms_norm).
        if k.bfloat_reinterpret_cast && !self.config.bfloat_reinterpret_cast {
            let mut opt_in = self.clone();
            opt_in.config.bfloat_reinterpret_cast = true;
            return opt_in.emit_msl(&k);
        }
        self.emit_msl(&k)
    }

    fn emit_msl(&self, k: &Kernel) -> Result<String> {
        tracing::debug!(kernel = %k.name, "starting MSL emit");
        let type_env = infer_types(k)?;
        let feat = self.analyze(k);
        let mut out = String::new();
        wl!(out, "// Generated by MetalTile");
        wl!(out, "#include <metal_stdlib>");
        wl!(out, "using namespace metal;");
        if feat.needs_bf16_struct && !self.config.native_bfloat {
            self.emit_bf16_preamble(&mut out);
        }
        if self.config.use_simd_matrix || feat.needs_simdgroup_matrix {
            wl!(out);
            wl!(out, "#include <metal_simdgroup_matrix>");
        }
        // MetalPerformancePrimitives (NAX / `mpp::tensor_ops::matmul2d`) — only
        // available on macOS 26+ / Metal 4. Gated on a kernel-level feature flag
        // detected from `Op::InlineMsl` sources that mention `mpp::`. Emits a
        // version guard so older targets fall through cleanly at compile time.
        if feat.needs_mpp {
            wl!(out);
            wl!(out, "#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400");
            wl!(out, "#include <metal_simdgroup>");
            wl!(out, "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>");
            wl!(out, "#endif");
        }
        self.emit_activation_helpers(&feat, &mut out);
        wl!(out);
        self.emit_kernel(k, &feat, &type_env, &mut out)?;
        tracing::debug!(kernel = %k.name, bytes = out.len(), "MSL emit complete");
        Ok(out)
    }

    // ---- kernel signature ------------------------------------------------

    fn emit_kernel(
        &self,
        kernel: &Kernel,
        feat: &KernelFeatures,
        type_env: &crate::passes::type_check::TypeEnv,
        out: &mut String,
    ) -> Result<()> {
        let mut buf_idx = 0u32;

        write!(out, "kernel void {}(", kernel.name).unwrap();

        // `[[maybe_unused]]` attribute decoration: DSL authors commonly
        // declare constexpr / scalar / tensor params for documentation
        // or runtime-assert hooks even when the kernel body never
        // reads them.  Without this attribute the Metal compiler
        // flags the param under `-Wunused-parameter`.  Detect at
        // signature time by scanning the post-pipeline IR for any
        // reference to the param's name (Load/Store/Threadgroup/Stack/
        // Local ops — see `kernel_references_param`).
        let attr_for = |name: &str| -> &'static str {
            if kernel_references_param(kernel, name) { "" } else { "[[maybe_unused]] " }
        };

        // Tensor/Strided/Scalar params.
        for p in &kernel.params {
            // Output params are always referenced by the Store at the
            // end of the kernel body — never mark `[[maybe_unused]]`.
            let attr = if p.is_output { "" } else { attr_for(&p.name) };
            match p.kind {
                ParamKind::Tensor => {
                    let q = if p.is_output { "device " } else { "const device " };
                    write!(
                        out,
                        "\n    {attr}{q}{} *{} [[buffer({buf_idx})]],",
                        self.msl_type_name(p.dtype),
                        p.name
                    )
                    .unwrap();
                    buf_idx += 1;
                },
                ParamKind::Strided => {
                    let q = if p.is_output { "device " } else { "const device " };
                    write!(
                        out,
                        "\n    {attr}{q}{} *{} [[buffer({buf_idx})]],",
                        self.msl_type_name(p.dtype),
                        p.name
                    )
                    .unwrap();
                    buf_idx += 1;
                    // The `_shape` / `_strides` companion buffers are
                    // referenced via `Op::Load { src: "{name}_shape" }`
                    // etc. at index-emission time.  Each gets its own
                    // referenced-by-name check.
                    let shape_attr = attr_for(&format!("{}_shape", p.name));
                    let strides_attr = attr_for(&format!("{}_strides", p.name));
                    write!(
                        out,
                        "\n    {shape_attr}constant uint *{}_shape [[buffer({buf_idx})]],",
                        p.name
                    )
                    .unwrap();
                    buf_idx += 1;
                    write!(
                        out,
                        "\n    {strides_attr}constant uint *{}_strides [[buffer({buf_idx})]],",
                        p.name
                    )
                    .unwrap();
                    buf_idx += 1;
                },
                ParamKind::Scalar => {
                    write!(
                        out,
                        "\n    {attr}constant {} &{} [[buffer({buf_idx})]],",
                        self.msl_type_name(p.dtype),
                        p.name
                    )
                    .unwrap();
                    buf_idx += 1;
                },
            }
        }

        // Constexpr params.
        for ce in &kernel.constexprs {
            let msl_type = match ce.dtype {
                DType::F32 => "float",
                DType::F16 => "half",
                DType::BF16 => "float",
                DType::I32 => "int",
                DType::I64 => "long",
                DType::U64 => "ulong",
                _ => "uint",
            };
            let attr = attr_for(ce.name.name());
            write!(
                out,
                "\n    {attr}constant {msl_type} &{} [[buffer({buf_idx})]],",
                ce.name.name()
            )
            .unwrap();
            buf_idx += 1;
        }

        // Thread / threadgroup position attributes.  For Reduction
        // mode we gate each kernel-attr decl on the same predicate as
        // its preamble alias — see `ReductionPreambleGates`.  Pre-fix,
        // both the signature and preamble were unconditional, so a
        // kernel that didn't use `tid`/`tgid_*`/`lsize` produced
        // `-Wunused-parameter` warnings against the attr decls.
        let reduction_gates = if kernel.mode == KernelMode::Reduction {
            Some(ReductionPreambleGates::compute(kernel, &self.config))
        } else {
            None
        };
        match kernel.mode {
            KernelMode::Elementwise => {
                write!(out, "\n    uint tid [[thread_position_in_grid]]").unwrap();
            },
            KernelMode::Reduction => {
                let gates = reduction_gates.as_ref().expect("set above for Reduction mode");
                // Each attr is emitted only when its derived alias has
                // a real consumer.  At least one is virtually always
                // needed (a kernel with no thread/grid awareness is
                // degenerate); but a kernel that uses only `tgid_y`
                // legitimately drops `_tid3` and `_lsize3`.  Track
                // whether we've written any so we know where to put
                // the comma separators.
                let mut wrote_any = false;
                let mut emit_attr = |line: &str| {
                    let prefix = if wrote_any { ",\n    " } else { "\n    " };
                    write!(out, "{prefix}{line}").unwrap();
                    wrote_any = true;
                };
                if gates.needs_tid3_attr() {
                    emit_attr("uint3 _tid3  [[thread_position_in_threadgroup]]");
                }
                if gates.needs_tgid3_attr() {
                    emit_attr("uint3 _tgid3 [[threadgroup_position_in_grid]]");
                }
                if gates.needs_lsize3_attr() {
                    emit_attr("uint3 _lsize3 [[threads_per_threadgroup]]");
                }
                // If none of the above fired the kernel has no
                // thread/threadgroup attrs.  Emit a degenerate `tid`
                // attr to keep the signature non-trivial — MSL accepts
                // a kernel with zero builtin attrs but the parameter
                // list then needs to end without a leading comma for
                // the simd_lane/simd_group conditional path below.
                // `[[maybe_unused]]` keeps `metal -W` quiet about it.
                if !wrote_any {
                    write!(out, "\n    [[maybe_unused]] uint _tid3 [[thread_position_in_grid]]")
                        .unwrap();
                }
            },
            KernelMode::Grid3D => {
                write!(out, "\n    uint3 gid [[thread_position_in_grid]]").unwrap();
            },
            KernelMode::Tile2D => {
                write!(out, "\n    uint2 tid  [[thread_position_in_threadgroup]],").unwrap();
                write!(out, "\n    uint2 tgid [[threadgroup_position_in_grid]]").unwrap();
            },
            KernelMode::SimdGroup2D => {
                // `lid` is referenced only when the DSL reads
                // `tid_x` / `tid_y` / `tid_z` as a direct identifier
                // (emit_block.rs maps those loads to `lid.x` / `.y` /
                // `.z` in SimdGroup2D mode).  Matmul kernels in
                // SimdGroup2D mode go through `emit_tiled` and never
                // touch `lid` — they'd emit `-Wunused-parameter` for
                // an unconditional `lid` attr.
                let needs_lid = kernel_uses_identifier(kernel, "tid_x")
                    || kernel_uses_identifier(kernel, "tid_y")
                    || kernel_uses_identifier(kernel, "tid_z");
                if needs_lid {
                    write!(out, "\n    uint3 lid [[thread_position_in_threadgroup]],").unwrap();
                }
                write!(out, "\n    uint3 tid [[threadgroup_position_in_grid]]").unwrap();
                write!(out, ",\n    uint simd_lane [[thread_index_in_simdgroup]]").unwrap();
                write!(out, ",\n    uint simd_group [[simdgroup_index_in_threadgroup]]").unwrap();
            },
        }
        // NOTE: SimdGroup2D mode emits simd_lane and simd_group inline above.
        // Other modes add them conditionally here.  `feat.needs_simd_*`
        // now reflects actual MSL identifier consumption directly (post-#209/4):
        // the `needs_simd_lane`/`needs_simd_group` OpFlags live only on
        // `Op::SimdLaneId`/`Op::SimdGroupId`, plus the per-feature multi-op
        // cases (matmul / Reduce slow path) handled in `features.rs`.
        if !matches!(kernel.mode, KernelMode::SimdGroup2D) && feat.needs_simd_lane {
            write!(out, ",\n    uint simd_lane [[thread_index_in_simdgroup]]").unwrap();
        }
        if !matches!(kernel.mode, KernelMode::SimdGroup2D) && feat.needs_simd_group {
            write!(out, ",\n    uint simd_group [[simdgroup_index_in_threadgroup]]").unwrap();
        }

        wl!(out, "\n) {{");

        // MPP (`mpp::tensor_ops::matmul2d`) is macOS-26+/Metal-4 only. The
        // framework `#include` is already `#if __METAL_VERSION__ >= 400`
        // guarded; the kernel *body* references `mpp::` symbols too, so it
        // must sit behind the same guard or it fails to compile on older
        // toolchains. The `#else` branch is an empty no-op stub — the
        // metallib still links; the kernel is simply inert pre-Metal-4
        // (NAX hardware requires macOS 26+ at runtime anyway).
        if feat.needs_mpp {
            wl!(out, "#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400");
        }

        // Inject scalar aliases for Reduction mode.  Each alias is gated
        // on actual consumption — these decls show up in `swift build`'s
        // shader-compile pass as `-Wunused-variable` warnings when they
        // aren't referenced, and a single kernel-suite emit at the >5000
        // warning level scales linearly with kernel count.
        //
        // Consumer summary (one MSL emit site per row):
        //   tid     ← Op::ProgramId{axis:0} in Elementwise (n/a in Reduction);
        //             Op::StrideReduce vectorized-loop path (`{has_tid}` branch
        //             in emit_block.rs); direct identifier `Op::Load { src:"tid" }`.
        //   tgid_x  ← Op::ProgramId{axis:0} in Reduction; direct `tgid_x` load.
        //   lsize   ← Op::StrideReduce; Op::Reduce (Mean path emits
        //             `…/float(lsize)` in both fast and slow); n_simd's own
        //             definition (`n_simd = lsize / 32u`); direct `lsize` load.
        //   n_simd  ← Op::Reduce slow path (tpg > simd_size); direct `n_simd` load.
        if let Some(gates) = reduction_gates.as_ref() {
            if gates.needs_tid {
                wl!(out, "    uint tid    = _tid3.x;");
            }
            if gates.needs_tgid_x {
                wl!(out, "    uint tgid_x = _tgid3.x;");
            }
            if gates.needs_tgid_y {
                wl!(out, "    uint tgid_y = _tgid3.y;");
            }
            if gates.needs_tgid_z {
                wl!(out, "    uint tgid_z = _tgid3.z;");
            }
            if gates.needs_lsize {
                wl!(out, "    uint lsize  = _lsize3.x;");
            }
            if gates.needs_n_simd {
                wl!(out, "    uint n_simd = lsize / 32u;");
            }
        }

        // Two-pass: collect threadgroup hoists first, then emit body.
        let mut body_buf = String::new();
        let mut hoists: Vec<String> = Vec::new();
        let extra_names: BTreeMap<ValueId, String> = BTreeMap::new();

        if feat.is_matmul {
            self.emit_tiled(&mut body_buf, "    ", kernel, None)?;
        } else {
            self.emit_block(
                &kernel.body,
                &kernel.blocks,
                &mut body_buf,
                1,
                kernel,
                type_env,
                &extra_names,
                &mut hoists,
            )?;
        }

        for h in &hoists {
            wl!(out, "    {h}");
        }
        if !hoists.is_empty() {
            wl!(out);
        }
        out.push_str(&body_buf);

        if feat.needs_mpp {
            wl!(out, "#else");
            wl!(out, "    // Pre-Metal-4 stub: mpp::tensor_ops is unavailable.");
            wl!(out, "#endif");
        }

        wl!(out, "}}");
        Ok(())
    }
}

/// Create a generator configured for the given kernel mode and the
/// kernel's expected dispatch threadgroup size.
///
/// Tile2D and SimdGroup2D modes enable simdgroup matrix intrinsics; all
/// other modes use the default config. `expected_tpg` is forwarded to
/// `MslConfig::expected_tpg` so codegen paths that depend on `lsize`
/// (notably the Reduction-mode `Op::Reduce` emit — single `simd_*(value)`
/// at `tpg ≤ simd_size`, two-level threadgroup reduction otherwise) pick
/// the right specialization. Pass `None` when the dispatch TPG isn't
/// known at codegen time (caller falls back to the conservative slow path).
///
/// **Why `tile build` callers should pass this:** the bench harness sets
/// `expected_tpg` from `ShapeSpec.tpg`. If `tile build --emit all`
/// dropped the TPG and produced different MSL than `tile bench` measured,
/// the bench numbers would describe a kernel nobody actually runs in
/// production. `cmd/build.rs` reads `spec.shapes[0].tpg` so the emitted
/// `.metal` files match exactly what bench measured.
pub fn generator_for_mode(mode: KernelMode, expected_tpg: Option<u32>) -> MslGenerator {
    let base = if matches!(mode, KernelMode::Tile2D | KernelMode::SimdGroup2D) {
        MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: true,
            ..MslConfig::default()
        }
    } else {
        MslConfig::default()
    };
    MslGenerator::new(MslConfig { expected_tpg, ..base })
}

impl Default for MslGenerator {
    fn default() -> Self { MslGenerator::new(MslConfig::default()) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ValueId},
        shape::Shape,
    };

    use super::*;

    /// Test-only helper: append a sink `Store` so DCE keeps the chain of
    /// ops that produces `vid` alive.  Without this, kernels constructed
    /// in unit tests for lowering-pattern verification get cleaned out
    /// entirely by `DeadValueElimPass` and the assertions can't find the
    /// MSL substrings they're checking for.  Real DSL kernels always
    /// terminate in a Store to an output param; this helper mirrors that
    /// shape minimally.
    ///
    /// Allocates `out` as a F32 output param and uses a fresh high
    /// ValueId for the index Const so it can't collide with the IDs the
    /// caller has already assigned.
    fn sink(k: &mut Kernel, vid: ValueId) {
        const SINK_IDX_VID: u32 = 0x3fff_fffe;
        k.params.push(Param {
            name: "_sink".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: Default::default(),
        });
        let idx_vid = ValueId::new(SINK_IDX_VID);
        k.body.push_op(Op::Const { value: 0 }, idx_vid);
        k.body.push_op_no_result(Op::Store {
            dst: "_sink".into(),
            indices: vec![IndexExpr::Value(idx_vid)],
            value: vid,
            mask: None,
        });
    }

    fn make_vadd() -> Kernel {
        let mut k = Kernel::new("vector_add");
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "b".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "c".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: Default::default(),
        });
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.name_value(ValueId::new(0), "idx");
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(1),
        );
        k.body.name_value(ValueId::new(1), "x");
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(2),
        );
        k.body.name_value(ValueId::new(2), "y");
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        k.body.name_value(ValueId::new(3), "sum");
        k.body.push_op_no_result(Op::Store {
            mask: None,
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
        });
        k
    }

    #[test]
    fn vadd_msl_structure() {
        let k = make_vadd();
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("kernel void vector_add"), "missing kernel declaration");
        assert!(msl.contains("const device float *a [[buffer(0)]]"), "missing param a");
        assert!(msl.contains("const device float *b [[buffer(1)]]"), "missing param b");
        assert!(msl.contains("device float *c [[buffer(2)]]"), "missing output param c");
        assert!(msl.contains("uint tid [[thread_position_in_grid]]"), "missing scalar tid");
        assert!(!msl.contains("tid.x"), "scalar kernel must not use tid.x");
    }

    #[test]
    fn vadd_msl_load_store() {
        let k = make_vadd();
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("v_idx"), "program_id should map to tid");
        assert!(msl.contains("a[v_idx]"), "load from a at idx");
        assert!(msl.contains("b[v_idx]"), "load from b at idx");
        assert!(msl.contains("v_x + v_y"), "add x and y");
        assert!(msl.contains("c[v_idx] = v_sum"), "store sum to c");
    }

    #[test]
    fn const_op_emits_uint_for_nonneg() {
        let mut k = Kernel::new("const_test");
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(0));
        sink(&mut k, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("uint v0 = 42u"), "non-negative Const should emit as uint");
    }

    #[test]
    fn const_op_emits_int_for_negative() {
        let mut k = Kernel::new("const_neg_test");
        k.body.push_op(Op::Const { value: -7 }, ValueId::new(0));
        sink(&mut k, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("int v0 = -7"), "negative Const should emit as int");
    }

    #[test]
    fn cast_op_emits_static_cast() {
        let mut k = Kernel::new("cast_test");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F16 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("static_cast<half>"), "cast to f16 should use static_cast<half>");
    }

    #[test]
    fn native_bfloat_omits_compat_preamble() {
        let mut k = Kernel::new("native_bfloat_param");
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::BF16,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        let msl = MslGenerator::new(MslConfig { native_bfloat: true, ..MslConfig::default() })
            .generate(&k)
            .unwrap();
        assert!(
            !msl.contains("struct bfloat16_t"),
            "native bfloat mode must not emit the compatibility preamble"
        );
        assert!(
            msl.contains("const device bfloat *a [[buffer(0)]]"),
            "native bfloat mode should keep native buffer types"
        );
    }

    #[test]
    fn bf16_cast_uses_compat_ctor_when_native_bfloat_disabled() {
        let mut k = Kernel::new("compat_bf16_cast");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::BF16 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::new(MslConfig {
            native_bfloat: false,
            bfloat_reinterpret_cast: false,
            ..MslConfig::default()
        })
        .generate(&k)
        .unwrap();
        assert!(msl.contains("struct bfloat16_t"), "compat mode should emit the bfloat16_t helper");
        assert!(
            msl.contains("bfloat16_t v1 = bfloat16_t(v0);"),
            "compat mode should cast to the compatibility struct"
        );
    }

    #[test]
    fn bf16_cast_uses_native_static_cast_when_enabled() {
        let mut k = Kernel::new("native_bf16_cast");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::BF16 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::new(MslConfig {
            native_bfloat: true,
            bfloat_reinterpret_cast: false,
            ..MslConfig::default()
        })
        .generate(&k)
        .unwrap();
        assert!(
            !msl.contains("struct bfloat16_t"),
            "native mode should not emit the compatibility helper"
        );
        assert!(
            msl.contains("bfloat v1 = bfloat(v0);"),
            "native mode should cast directly to bfloat via constructor"
        );
    }

    #[test]
    fn bf16_cast_uses_reinterpret_when_flag_enabled_and_src_is_f32() {
        // The reinterpret peephole only fires for f32→bf16; integer sources
        // fall back to the rounding constructor (see
        // `bf16_cast_from_int_uses_rounding_constructor`).
        let mut k = Kernel::new("reinterpret_bf16_cast");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Cast { value: ValueId::new(1), dtype: DType::BF16 }, ValueId::new(2));
        sink(&mut k, ValueId::new(2));
        let msl = MslGenerator::new(MslConfig {
            native_bfloat: true,
            bfloat_reinterpret_cast: true,
            ..MslConfig::default()
        })
        .generate(&k)
        .unwrap();
        assert!(
            msl.contains("as_type<bfloat2>(") && msl.contains(")[1];"),
            "reinterpret mode bypasses the slow IEEE bfloat() builtin on f32→bf16:\n{msl}"
        );
        assert!(
            !msl.contains("v2 = bfloat("),
            "should not emit the rounding constructor when reinterpret applies:\n{msl}"
        );
    }

    #[test]
    fn bf16_cast_defaults_to_rounding_constructor() {
        // Default config must NOT enable the reinterpret peephole — it
        // truncates lower 16 bits of fp32 instead of round-to-nearest-even,
        // drifting up to 1 ULP per cast. Tight-tolerance kernels (e.g.
        // rms_norm) fail Tile Bench quality with reinterpret on. Opt in
        // per kernel only where the drift is provably tolerable.
        // Regression: PR #47 CI rms_norm bf16 ✗ at B=1024 N=4096.
        let mut k = Kernel::new("default_bf16_cast");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Cast { value: ValueId::new(1), dtype: DType::BF16 }, ValueId::new(2));
        sink(&mut k, ValueId::new(2));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("as_type<bfloat2>("),
            "default config must not emit truncating reinterpret on f32→bf16:\n{msl}"
        );
        assert!(
            msl.contains("v2 = bfloat("),
            "default config must emit rounding constructor:\n{msl}"
        );
    }

    #[test]
    fn bf16_cast_from_int_uses_rounding_constructor() {
        // Int→bf16 reinterpret would read upper-half int bits as bf16
        // (e.g. `as_type<bfloat2>(123)[1]` = 0 because the upper 16 bits
        // of int 123 are zero). The peephole must not fire here — fall
        // back to `bfloat(value)` which performs the actual integer →
        // float → bf16 rounding chain. Regression: caught by Tile Bench's
        // `arange` kernel emitting all-zero output at bf16.
        let mut k = Kernel::new("int_to_bf16");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::BF16 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::new(MslConfig {
            native_bfloat: true,
            bfloat_reinterpret_cast: true,
            ..MslConfig::default()
        })
        .generate(&k)
        .unwrap();
        assert!(
            msl.contains("bfloat v1 = bfloat(v0);"),
            "int→bf16 must use rounding ctor, not reinterpret:\n{msl}"
        );
        assert!(!msl.contains("as_type<bfloat2>(v0)"), "reinterpret must not fire on int source");
    }

    #[test]
    fn unary_op_emit() {
        use metaltile_core::ir::UnaryOpKind;
        let mut k = Kernel::new("unary_test");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body
            .push_op(Op::UnaryOp { op: UnaryOpKind::Exp, value: ValueId::new(0) }, ValueId::new(1));
        k.body.name_value(ValueId::new(1), "r");
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("exp(v0)"), "exp unary op");
        assert!(msl.contains("v_r"), "named result");
    }

    #[test]
    fn activation_silu_emits_helper() {
        use metaltile_core::ir::ActKind;
        let mut k = Kernel::new("silu_test");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(
            Op::Activation { kind: ActKind::Silu, value: ValueId::new(0) },
            ValueId::new(1),
        );
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("mt_silu"), "silu helper function name");
        assert!(msl.contains("inline T mt_silu"), "silu helper definition");
    }

    #[test]
    fn fused_activation_emits_helper() {
        // `silu(a) * b`: the fusion pass folds the Activation into an
        // `Op::FusedElementwise`, hiding the standalone `Op::Activation`.
        // Feature analysis must recurse into the fused chain so the
        // `mt_silu` helper preamble is still emitted — otherwise the MSL
        // calls an undeclared identifier and fails to compile.
        use metaltile_core::{
            ir::{ActKind, BinOpKind, IndexExpr, Param},
            shape::Shape,
        };
        let mut k = Kernel::new("fused_silu_mul");
        for (name, is_output) in [("a", false), ("b", false), ("c", true)] {
            k.params.push(Param {
                name: name.into(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output,
                kind: Default::default(),
            });
        }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::Activation { kind: ActKind::Silu, value: ValueId::new(1) },
            ValueId::new(3),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(3), rhs: ValueId::new(2) },
            ValueId::new(4),
        );
        k.body.push_op_no_result(Op::Store {
            mask: None,
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(4),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("inline T mt_silu"),
            "fused silu must still emit the helper definition:\n{msl}"
        );
    }

    #[test]
    fn select_emit() {
        // Use a runtime-derived condition (ProgramId) so the
        // ConstFoldPass can't pick a branch at compile time and remove
        // the Op::Select entirely.  Same for the arms — load from two
        // input params so neither side folds away.
        let mut k = Kernel::new("select_test");
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "b".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0)); // cond
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::Select {
                cond: ValueId::new(0),
                on_true: ValueId::new(1),
                on_false: ValueId::new(2),
            },
            ValueId::new(3),
        );
        sink(&mut k, ValueId::new(3));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("bool("), "bool cast on condition: {msl}");
        assert!(msl.contains("? v1 : v2"), "ternary select: {msl}");
    }

    #[test]
    fn multi_dim_load_stride() {
        use metaltile_core::{constexpr::ConstExpr, shape::Dim};
        let mut k = Kernel::new("matload_test");
        let m_ce = ConstExpr::new("M");
        let n_ce = ConstExpr::new("N");
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::new([Dim::ConstExpr(m_ce.clone()), Dim::ConstExpr(n_ce.clone())]),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "out".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: Default::default(),
        });
        k.constexprs.push(metaltile_core::ir::ConstExprDecl {
            name: m_ce,
            dtype: DType::U32,
            value: None,
        });
        k.constexprs.push(metaltile_core::ir::ConstExprDecl {
            name: n_ce,
            dtype: DType::U32,
            value: None,
        });
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.name_value(ValueId::new(0), "row");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(1));
        k.body.name_value(ValueId::new(1), "col");
        k.body.push_op(
            Op::Load {
                mask: None,
                other: None,
                src: "a".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0)), IndexExpr::Value(ValueId::new(1))],
            },
            ValueId::new(2),
        );
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("(v_row) * N + (v_col)"), "2D load must use row-major stride");
    }

    #[test]
    fn fusion_cast_exp_binop_into_single_expression() {
        use metaltile_core::ir::{BinOpKind, UnaryOpKind};

        use crate::passes::{Pass, fusion::FusionPass};

        let mut k = Kernel::new("fused_test");
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body
            .push_op(Op::UnaryOp { op: UnaryOpKind::Exp, value: ValueId::new(1) }, ValueId::new(2));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(3));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) },
            ValueId::new(4),
        );
        sink(&mut k, ValueId::new(4));
        FusionPass.run(&mut k).unwrap();
        let has_fused = k.body.ops.iter().any(|op| matches!(op, Op::FusedElementwise { .. }));
        assert!(has_fused, "fusion pass should create a FusedElementwise op");
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("exp") && msl.contains(" * "),
            "fused expression should contain exp and mul"
        );
        let lines: Vec<&str> = msl.lines().collect();
        let fused_line = lines.iter().find(|l| l.contains("v4")).unwrap();
        assert!(
            fused_line.contains("exp") && fused_line.contains("static_cast"),
            "single fused line should contain both exp and cast: {}",
            fused_line
        );
        assert!(!msl.contains("v1 = "), "no separate declaration for v1");
        assert!(!msl.contains("v2 = "), "no separate declaration for v2");
    }

    #[test]
    fn reduction_preamble_omits_tgid_y_when_unused() {
        let mut k = Kernel::new("reduction_no_y");
        k.mode = KernelMode::Reduction;
        // axis-0 only → tgid_y should NOT be emitted
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(!msl.contains("tgid_y"), "tgid_y must be omitted when program_id axis 1 is unused");
    }

    #[test]
    fn reduction_preamble_emits_tgid_y_when_used() {
        let mut k = Kernel::new("reduction_with_y");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::ProgramId { axis: 1 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("tgid_y"), "tgid_y must be emitted when program_id axis 1 is used");
    }

    #[test]
    fn reduction_program_id_axis_2_lowers_to_tgid_z() {
        // A reduction kernel using the z grid axis (e.g. batched_qkv_qgemv,
        // where program_id::<2>() selects the Q/K/V matrix) must lower
        // axis 2 to `tgid_z` and declare the alias — not fold it to 0.
        let mut k = Kernel::new("reduction_with_z");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::ProgramId { axis: 2 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint tgid_z = _tgid3.z;"),
            "preamble must declare tgid_z when program_id axis 2 is used: {msl}"
        );
        assert!(msl.contains("= tgid_z;"), "axis 2 must lower to tgid_z, not a constant: {msl}");
    }

    // ── Preamble-emission gates ─────────────────────────────────────────
    //
    // Each preamble decl (`tid` / `tgid_x` / `lsize` / `n_simd`) gets a
    // matching "used → emit" / "unused → omit" pair below. Pre-fix the
    // unconditional emits produced thousands of `-Wunused-variable`
    // warnings against the full kernel suite; these pin the gates the
    // fix introduced so the warnings can't silently come back.

    /// `tgid_x` must be omitted when the kernel doesn't reference
    /// `program_id::<0>()` and doesn't read the identifier directly —
    /// some kernels only iterate `tgid_y`/`tgid_z` and don't touch x.
    #[test]
    fn reduction_preamble_omits_tgid_x_when_unused() {
        let mut k = Kernel::new("reduction_no_x");
        k.mode = KernelMode::Reduction;
        // axis 1 only — no Op::ProgramId axis 0, no direct `tgid_x` load.
        k.body.push_op(Op::ProgramId { axis: 1 }, ValueId::new(0));
        sink(&mut k, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("uint tgid_x = _tgid3.x;"),
            "tgid_x decl must be omitted when program_id axis 0 is unused: {msl}"
        );
    }

    /// `tid` is consumed by `Op::StrideReduce` (in Reduction mode it
    /// lowers to a vectorized loop indexed by `tid * 4u`). When no
    /// `StrideReduce` exists *and* nothing reads `tid` directly, the
    /// preamble must drop the decl.
    #[test]
    fn reduction_preamble_omits_tid_when_unused() {
        let mut k = Kernel::new("reduction_no_tid");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        sink(&mut k, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("uint tid    = _tid3.x;"),
            "tid decl must be omitted when nothing references it: {msl}"
        );
    }

    /// Symmetric: a direct DSL identifier read (`Op::Load { src: "tid"
    /// }`) keeps the `tid` decl alive.
    #[test]
    fn reduction_preamble_emits_tid_when_used_as_identifier() {
        let mut k = Kernel::new("reduction_with_tid_load");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load { src: "tid".into(), indices: Vec::new(), mask: None, other: None },
            ValueId::new(1),
        );
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint tid    = _tid3.x;"),
            "preamble must declare tid when read via direct identifier: {msl}"
        );
    }

    /// `lsize` is consumed by `Op::StrideReduce`, by `Op::Reduce` (the
    /// `Mean` path divides the simdgroup total by `float(lsize)`), and
    /// by the `n_simd = lsize / 32u` derivation. When none of those
    /// fire, drop the decl.
    #[test]
    fn reduction_preamble_omits_lsize_when_unused() {
        let mut k = Kernel::new("reduction_no_lsize");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        sink(&mut k, ValueId::new(0));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("uint lsize  = _lsize3.x;"),
            "lsize decl must be omitted when no StrideReduce/Reduce/n_simd needs it: {msl}"
        );
    }

    /// `lsize` survives only when something actually references it.
    /// `Op::Reduce` with `Sum` lowers to a bare `simd_sum(value)` — no
    /// `lsize`.  Pre-fix, every Op::Reduce dragged `lsize` in
    /// unconditionally; this case ensures we don't over-emit on the
    /// cheap path (`mt_rms_norm_small` etc.).
    #[test]
    fn reduction_preamble_omits_lsize_for_simd_only_reduce_sum() {
        use metaltile_core::ir::ReduceKind;
        let mut k = Kernel::new("reduction_reduce_sum");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(
            Op::Reduce { value: ValueId::new(1), axis: 0, op: ReduceKind::Sum },
            ValueId::new(2),
        );
        sink(&mut k, ValueId::new(2));
        // Single-simdgroup config → fast path emits only `simd_sum`; no
        // lsize divisor exists in the lowering.
        let msl = MslGenerator::new(MslConfig { expected_tpg: Some(32), ..MslConfig::default() })
            .generate(&k)
            .unwrap();
        assert!(
            !msl.contains("uint lsize"),
            "Sum reduce on the fast path must not pull lsize in: {msl}"
        );
        // And it must not get pulled in by the slow-path n_simd
        // requirement either, since we're explicitly single-simdgroup.
        assert!(!msl.contains("uint n_simd"), "fast path must not declare n_simd: {msl}");
    }

    /// `Op::Reduce` with `Mean` divides by `float(lsize)` in both fast
    /// and slow paths — `lsize` must be in the preamble.
    #[test]
    fn reduction_preamble_emits_lsize_when_reduce_mean_present() {
        use metaltile_core::ir::ReduceKind;
        let mut k = Kernel::new("reduction_reduce_mean");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(
            Op::Reduce { value: ValueId::new(1), axis: 0, op: ReduceKind::Mean },
            ValueId::new(2),
        );
        sink(&mut k, ValueId::new(2));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint lsize  = _lsize3.x;"),
            "Op::Reduce Mean must keep lsize in the preamble: {msl}"
        );
    }

    /// `n_simd` is consumed by the slow-path two-level reduction (TPG >
    /// simd width) and by direct identifier reads. A bare `Op::Reduce`
    /// at default config (no `expected_tpg`) routes through the slow
    /// path → `n_simd` must be emitted. A kernel without `Op::Reduce`
    /// and without an `n_simd` identifier load must omit it.
    #[test]
    fn reduction_preamble_omits_n_simd_when_unused() {
        let mut k = Kernel::new("reduction_no_n_simd");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // `Op::SimdGroupId` sets `feat.needs_simd_group` but does NOT
        // need n_simd — pre-fix this case triggered an unused n_simd
        // decl.
        k.body.push_op(Op::SimdGroupId, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("uint n_simd"),
            "n_simd decl must be omitted when only simd_group is read: {msl}"
        );
    }

    /// Symmetric: an `Op::Reduce` on the slow path keeps `n_simd`.
    #[test]
    fn reduction_preamble_emits_n_simd_when_reduce_slow_path() {
        use metaltile_core::ir::ReduceKind;
        let mut k = Kernel::new("reduction_reduce_slow");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(
            Op::Reduce { value: ValueId::new(1), axis: 0, op: ReduceKind::Sum },
            ValueId::new(2),
        );
        sink(&mut k, ValueId::new(2));
        // `MslGenerator::default()` has `expected_tpg = None`, so the
        // slow path fires; the reduce emit will reference `n_simd`.
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint n_simd = lsize / 32u;"),
            "slow-path Op::Reduce must keep n_simd in the preamble: {msl}"
        );
    }

    /// Direct `n_simd` identifier read keeps the decl regardless of
    /// reduce presence.
    #[test]
    fn reduction_preamble_emits_n_simd_when_used_as_identifier() {
        let mut k = Kernel::new("reduction_n_simd_load");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load { src: "n_simd".into(), indices: Vec::new(), mask: None, other: None },
            ValueId::new(1),
        );
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint n_simd = lsize / 32u;"),
            "direct n_simd identifier load must keep the decl: {msl}"
        );
    }

    /// Regression: `ssm_step` and similar kernels reference `tgid_y` via the
    /// direct-identifier form, which the body parser lowers to
    /// `Op::Load { src: "tgid_y", .. }` rather than `Op::ProgramId`. The
    /// preamble must still declare the alias.
    #[test]
    fn reduction_preamble_emits_tgid_y_when_used_as_identifier() {
        let mut k = Kernel::new("reduction_with_y_load");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
            ValueId::new(1),
        );
        // Sink the load result so DCE doesn't eliminate the scalar
        // Load (which would also drop the preamble decl).  The whole
        // point of this regression is the parser's direct-identifier
        // lowering path; with a real consumer present, the decl must
        // be emitted.
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("uint tgid_y = _tgid3.y;"),
            "preamble must declare tgid_y when used via direct identifier: {msl}"
        );
    }

    /// `Op::SimdShuffleXor { value, mask }` must emit `simd_shuffle_xor(v, mask)`
    /// — the Metal 2.1+ butterfly shuffle used by AURA's FWHT inner loop and
    /// Steel attention row reductions. `mask` is a compile-time u32 literal.
    #[test]
    fn simd_shuffle_xor_emits_metal_builtin() {
        let mut k = Kernel::new("simd_xor_smoke");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::SimdShuffleXor { value: ValueId::new(0), mask: 1 }, ValueId::new(1));
        sink(&mut k, ValueId::new(1));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("simd_shuffle_xor("),
            "kernel must emit a simd_shuffle_xor call: {msl}"
        );
    }

    /// `Op::SimdBroadcast { value, lane }` must emit `simd_broadcast(v, lane)`
    /// — the Metal 2.1+ cross-lane broadcast used by AURA's codebook hoist.
    #[test]
    fn simd_broadcast_emits_metal_builtin() {
        let mut k = Kernel::new("simd_bcast_smoke");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(1));
        k.body.push_op(
            Op::SimdBroadcast { value: ValueId::new(0), lane: ValueId::new(1) },
            ValueId::new(2),
        );
        sink(&mut k, ValueId::new(2));
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(msl.contains("simd_broadcast("), "kernel must emit a simd_broadcast call: {msl}");
    }

    /// `Op::ThreadgroupAlloc { dtype: U32, .. }` must emit
    /// `threadgroup uint <name>[<size>];`.  AURA encode's pack stage needs
    /// a `uint` threadgroup buffer so subsequent `atomic_fetch_or_explicit`
    /// can reinterpret it as `threadgroup atomic_uint*`.
    #[test]
    fn threadgroup_alloc_emits_u32_buffer() {
        let mut k = Kernel::new("tg_u32_alloc");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op_no_result(Op::ThreadgroupAlloc {
            dtype: DType::U32,
            size: 128,
            name: "shared_packed".to_string(),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("threadgroup uint shared_packed[128];"),
            "expected `threadgroup uint shared_packed[128];` for U32 alloc: {msl}"
        );
    }

    /// `Op::Atomic { scope: AtomicScope::Threadgroup, .. }` must emit
    /// the cast form `atomic_fetch_or_explicit((threadgroup atomic_uint*)&<dst>[<idx>], …)`.
    /// `Device` scope keeps the existing `dst + idx` form.
    #[test]
    fn atomic_threadgroup_emits_cast_form() {
        use metaltile_core::ir::{AtomicKind, AtomicScope};

        let mut k = Kernel::new("atomic_tg_smoke");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op_no_result(Op::ThreadgroupAlloc {
            dtype: DType::U32,
            size: 128,
            name: "shared_packed".to_string(),
        });
        k.body.push_op_no_result(Op::Atomic {
            op: AtomicKind::Or,
            scope: AtomicScope::Threadgroup,
            dst: "shared_packed".to_string(),
            index: ValueId::new(0),
            value: ValueId::new(1),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("(threadgroup atomic_uint*)&shared_packed["),
            "threadgroup-scope atomic must reinterpret-cast the threadgroup slot: {msl}"
        );
        assert!(
            msl.contains("atomic_fetch_or_explicit"),
            "threadgroup atomic_or must still emit the OR intrinsic: {msl}"
        );
    }

    /// `Op::StackAlloc { dtype: F32, size: 4, name: "o" }` must emit a
    /// per-thread `float o[4];` (no `threadgroup` qualifier).  Subsequent
    /// `Op::StackLoad` / `Op::StackStore` must emit the same indexed
    /// access shape as the threadgroup variants — only the alloc qualifier
    /// distinguishes them.  AURA flash kernels need this for the per-lane
    /// `q_vals[DIMS_PER_LANE]` / `o[DIMS_PER_LANE]` arrays.
    #[test]
    fn stack_array_emits_per_thread_buffer() {
        let mut k = Kernel::new("stack_smoke");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(1));
        k.body.push_op_no_result(Op::StackAlloc {
            dtype: DType::F32,
            size: 4,
            name: "o".to_string(),
        });
        k.body.push_op(
            Op::StackLoad { name: "o".to_string(), index: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::StackStore {
            name: "o".to_string(),
            index: ValueId::new(1),
            value: ValueId::new(2),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("float o[4];"),
            "stack array must emit unqualified `float o[4];` (no threadgroup): {msl}"
        );
        assert!(
            !msl.contains("threadgroup float o["),
            "stack array must NOT carry the threadgroup qualifier: {msl}"
        );
        assert!(
            msl.contains("o[v_v1]") || msl.contains("o["),
            "stack store/load must use plain `name[idx]` shape: {msl}"
        );
    }

    /// Sanity: device-scope atomics keep the unchanged `<dst> + <idx>` form.
    #[test]
    fn atomic_device_emits_buffer_offset_form() {
        use metaltile_core::ir::{AtomicKind, AtomicScope};

        let mut k = Kernel::new("atomic_dev_smoke");
        k.mode = KernelMode::Elementwise;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Atomic {
            op: AtomicKind::Add,
            scope: AtomicScope::Device,
            dst: "counter".to_string(),
            index: ValueId::new(0),
            value: ValueId::new(1),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("atomic_fetch_add_explicit(counter +"),
            "device-scope atomic must keep buffer-offset form: {msl}"
        );
        assert!(
            !msl.contains("(threadgroup"),
            "device-scope atomic must NOT emit a threadgroup cast: {msl}"
        );
    }

    /// `needs_mpp` triggers the MetalPerformancePrimitives include when an
    /// `Op::InlineMsl` body references `mpp::`. Detection lives in
    /// `KernelFeatures` (see `msl/features.rs`); the preamble emits the
    /// header gated on Metal 4 so older toolchains still link.
    #[test]
    fn mpp_preamble_emits_when_inline_msl_contains_mpp() {
        let mut k = Kernel::new("mpp_smoke");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op_no_result(Op::InlineMsl {
            source: "mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;"
                .to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            msl.contains("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>"),
            "MPP include must appear when InlineMsl body references `mpp::`: {msl}"
        );
        assert!(
            msl.contains("__METAL_VERSION__ >= 400"),
            "MPP include must be gated on Metal 4: {msl}"
        );
    }

    /// `needs_mpp` stays off when no `Op::InlineMsl` body references `mpp::`.
    /// Forcing the include unconditionally would break older toolchains that
    /// don't ship the MPP header.
    #[test]
    fn mpp_preamble_omitted_when_no_mpp_marker() {
        let mut k = Kernel::new("no_mpp");
        k.mode = KernelMode::Reduction;
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // InlineMsl body present but no `mpp::` token anywhere.
        k.body.push_op_no_result(Op::InlineMsl {
            source: "float x = 1.0;".to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
        let msl = MslGenerator::default().generate(&k).unwrap();
        assert!(
            !msl.contains("MetalPerformancePrimitives"),
            "MPP include must NOT appear when InlineMsl has no `mpp::` marker: {msl}"
        );
    }
}
