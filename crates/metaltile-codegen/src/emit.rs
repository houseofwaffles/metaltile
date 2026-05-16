//! Emit-side helpers: write per-kernel MSL source files, a manifest
//! JSON, Swift dispatch wrappers, and (optionally) shell out to
//! `xcrun metal` to compile a `kernels.metallib`.
//!
//! Used by the `tile build --emit` flow in `metaltile-cli`. Kept in
//! `metaltile-codegen` so other tooling (custom build scripts, IDE
//! integrations, future SwiftPM build plugins) can also consume the
//! emit pipeline without depending on the CLI binary.
//!
//! Naming convention: kernels are written under their per-dtype
//! monomorphized name (e.g. `mt_add_f32`, `mt_add_f16`, `mt_add_bf16`).
//! The caller sets `kernel.name` before passing it in — see the CLI's
//! `cmd::build` for the canonical iteration over `BenchSpec`s.

use std::{
    io,
    path::{Path, PathBuf},
    process::Command,
};

use metaltile_core::{
    dtype::DType,
    ir::{ConstExprDecl, Kernel, KernelMode, Param, ParamKind},
};
use serde::Serialize;

use crate::msl::MslGenerator;

// ─── Manifest schema ─────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Manifest {
    /// Schema version. Bump on breaking changes.
    pub version: u32,
    /// `tile` version that produced this manifest.
    pub metaltile_version: String,
    pub kernels: Vec<KernelManifest>,
}

#[derive(Serialize)]
pub struct KernelManifest {
    /// Public kernel name (matches the MSL function symbol).
    pub name: String,
    /// Path to the MSL source file relative to the manifest.
    pub source: String,
    /// Thread-indexing mode — informs default grid/threadgroup sizing.
    pub kernel_mode: String,
    /// Buffer-bound parameters in slot order.
    pub params: Vec<ParamManifest>,
    /// Constexpr scalars bound via `setBytes` after the param buffers.
    pub constexprs: Vec<ConstExprManifest>,
}

#[derive(Serialize)]
pub struct ParamManifest {
    pub name: String,
    /// `"Tensor"`, `"Strided"`, or `"Scalar"`.
    pub kind: String,
    pub dtype: String,
    pub is_output: bool,
}

#[derive(Serialize)]
pub struct ConstExprManifest {
    pub name: String,
    pub dtype: String,
}

// ─── Errors ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum EmitError {
    Io(io::Error),
    Codegen(String),
    Json(serde_json::Error),
    MetalToolchain(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Io(e) => write!(f, "I/O error: {e}"),
            EmitError::Codegen(s) => write!(f, "codegen error: {s}"),
            EmitError::Json(e) => write!(f, "JSON serialization error: {e}"),
            EmitError::MetalToolchain(s) => write!(f, "metal toolchain error: {s}"),
        }
    }
}

impl std::error::Error for EmitError {}

impl From<io::Error> for EmitError {
    fn from(e: io::Error) -> Self { EmitError::Io(e) }
}
impl From<serde_json::Error> for EmitError {
    fn from(e: serde_json::Error) -> Self { EmitError::Json(e) }
}

type Result<T> = std::result::Result<T, EmitError>;

// ─── MSL ─────────────────────────────────────────────────────────────

/// Render `kernel` to MSL and write `<dir>/<kernel.name>.metal`. Returns
/// the written path. Caller chooses the `MslGenerator` so e.g. Tile2D
/// kernels can opt into `use_simd_matrix` without coupling the emit
/// helpers to a single config.
pub fn write_msl(kernel: &Kernel, dir: &Path, generator: &MslGenerator) -> Result<PathBuf> {
    let msl = generator
        .generate(kernel)
        .map_err(|e| EmitError::Codegen(format!("{e:?}")))?;
    let path = dir.join(format!("{}.metal", kernel.name));
    std::fs::write(&path, msl)?;
    Ok(path)
}

// ─── Manifest JSON ───────────────────────────────────────────────────

/// Serialize `kernels` to a manifest and write it to `path`.
pub fn write_manifest(kernels: &[Kernel], path: &Path) -> Result<()> {
    let manifest = build_manifest(kernels);
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn build_manifest(kernels: &[Kernel]) -> Manifest {
    Manifest {
        version: 1,
        metaltile_version: env!("CARGO_PKG_VERSION").to_string(),
        kernels: kernels.iter().map(kernel_to_manifest).collect(),
    }
}

fn kernel_to_manifest(k: &Kernel) -> KernelManifest {
    KernelManifest {
        name: k.name.clone(),
        source: format!("kernels/{}.metal", k.name),
        kernel_mode: kernel_mode_str(k.mode).to_string(),
        params: k.params.iter().map(param_to_manifest).collect(),
        constexprs: k.constexprs.iter().map(constexpr_to_manifest).collect(),
    }
}

fn param_to_manifest(p: &Param) -> ParamManifest {
    ParamManifest {
        name: p.name.clone(),
        kind: param_kind_str(&p.kind).to_string(),
        dtype: dtype_suffix(p.dtype).to_string(),
        is_output: p.is_output,
    }
}

fn constexpr_to_manifest(c: &ConstExprDecl) -> ConstExprManifest {
    ConstExprManifest {
        name: c.name.name().to_string(),
        dtype: dtype_suffix(c.dtype).to_string(),
    }
}

// ─── Swift dispatch wrappers ─────────────────────────────────────────

/// Render `MetalTileKernels.swift` — one static function per kernel,
/// looking up the PSO from `PSOCache.shared` and encoding the dispatch
/// onto the supplied command buffer. The PSOCache + metallib loading is
/// hand-written on the Swift side (lives in `MetalTileSwift`).
pub fn render_swift_wrappers(kernels: &[Kernel]) -> String {
    let mut out = String::new();
    out.push_str(
        "// AUTOGENERATED by `tile build --emit swift`. DO NOT EDIT.\n\
         //\n\
         // Each function dispatches a single Metal kernel from kernels.metallib.\n\
         // Looks up the pre-compiled PSO from PSOCache.shared, encodes the\n\
         // dispatch on the supplied command buffer, ends the encoder.\n\n\
         import Metal\n\n\
         public enum MetalTileKernels {\n",
    );
    for k in kernels {
        emit_swift_wrapper(&mut out, k);
    }
    out.push_str("}\n");
    out
}

pub fn write_swift_wrappers(kernels: &[Kernel], path: &Path) -> Result<()> {
    std::fs::write(path, render_swift_wrappers(kernels))?;
    Ok(())
}

fn emit_swift_wrapper(out: &mut String, k: &Kernel) {
    use std::fmt::Write as _;
    let fn_name = swift_safe_name(&k.name);

    writeln!(out, "    /// Dispatches `{}` from kernels.metallib.", k.name).ok();
    writeln!(out, "    public static func {fn_name}(").ok();

    // Buffer params (Tensor / Strided / Scalar all bind as buffers in Phase 0).
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(out, "        {label}: MTLBuffer, {label}Offset: Int = 0,").ok();
    }
    // Constexpr scalars (bound via setBytes after the param buffers).
    for c in &k.constexprs {
        let label = swift_safe_name(c.name.name());
        let swift_ty = swift_scalar_type(dtype_suffix(c.dtype));
        writeln!(out, "        {label}: {swift_ty},").ok();
    }
    writeln!(out, "        gridSize: MTLSize,").ok();
    writeln!(out, "        threadgroupSize: MTLSize,").ok();
    writeln!(out, "        on commandBuffer: MTLCommandBuffer").ok();
    writeln!(out, "    ) {{").ok();
    writeln!(
        out,
        "        let pso = PSOCache.shared.pipelineState(for: \"{}\")",
        k.name
    )
    .ok();
    writeln!(
        out,
        "        guard let enc = commandBuffer.makeComputeCommandEncoder() else {{ return }}"
    )
    .ok();
    writeln!(out, "        enc.setComputePipelineState(pso)").ok();

    let mut slot = 0usize;
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(
            out,
            "        enc.setBuffer({label}, offset: {label}Offset, index: {slot})"
        )
        .ok();
        slot += 1;
    }
    for c in &k.constexprs {
        let label = swift_safe_name(c.name.name());
        let len = swift_scalar_size(dtype_suffix(c.dtype));
        writeln!(out, "        var {label}_v = {label}").ok();
        writeln!(
            out,
            "        enc.setBytes(&{label}_v, length: {len}, index: {slot})"
        )
        .ok();
        slot += 1;
    }
    // dispatchThreads (in threads, not threadgroups) so out-of-bound
    // threads aren't created and the kernel doesn't need bounds checks.
    // Requires Metal 2.0 non-uniform threadgroup support (M-series ✓).
    writeln!(
        out,
        "        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)"
    )
    .ok();
    writeln!(out, "        enc.endEncoding()").ok();
    writeln!(out, "    }}\n").ok();
}

// ─── metallib compilation (xcrun metal + metallib) ───────────────────

/// Compile every `.metal` in `metal_files` and link them into a single
/// `metallib` written to `output`. Uses `xcrun -sdk <sdk> metal` for
/// per-file `.air` and `xcrun -sdk <sdk> metallib` for the final link.
///
/// `sdk` is the SDK name (e.g. `"macosx"`, `"iphoneos"`); `air_dir` is
/// a scratch directory the caller controls (so it can land under
/// cargo's `target/` and not litter `/tmp/`).
pub fn compile_metallib(
    metal_files: &[PathBuf],
    output: &Path,
    sdk: &str,
    air_dir: &Path,
) -> Result<()> {
    if metal_files.is_empty() {
        return Err(EmitError::MetalToolchain("no .metal files to compile".into()));
    }
    std::fs::create_dir_all(air_dir)?;

    let mut air_files: Vec<PathBuf> = Vec::with_capacity(metal_files.len());
    for metal in metal_files {
        let stem = metal
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| EmitError::MetalToolchain(format!("bad filename: {}", metal.display())))?;
        let air = air_dir.join(format!("{stem}.air"));
        let status = Command::new("xcrun")
            .args(["-sdk", sdk, "metal", "-c"])
            .arg(metal)
            .arg("-o")
            .arg(&air)
            .status()
            .map_err(|e| {
                EmitError::MetalToolchain(format!("invoke xcrun metal for {}: {e}", metal.display()))
            })?;
        if !status.success() {
            return Err(EmitError::MetalToolchain(format!(
                "xcrun metal failed for {}",
                metal.display()
            )));
        }
        air_files.push(air);
    }

    let status = Command::new("xcrun")
        .args(["-sdk", sdk, "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(output)
        .status()
        .map_err(|e| EmitError::MetalToolchain(format!("invoke xcrun metallib: {e}")))?;
    if !status.success() {
        return Err(EmitError::MetalToolchain("xcrun metallib failed".into()));
    }
    Ok(())
}

// ─── String helpers ──────────────────────────────────────────────────

pub fn dtype_suffix(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I64 => "i64",
        DType::U64 => "u64",
        DType::I4 => "i4",
        DType::Bool => "bool",
    }
}

fn param_kind_str(k: &ParamKind) -> &'static str {
    match k {
        ParamKind::Tensor => "Tensor",
        ParamKind::Strided => "Strided",
        ParamKind::Scalar => "Scalar",
    }
}

fn kernel_mode_str(m: KernelMode) -> &'static str {
    match m {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Grid3D => "Grid3D",
        KernelMode::Tile2D => "Tile2D",
    }
}

fn swift_safe_name(s: &str) -> String { s.replace('-', "_") }

fn swift_scalar_type(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "Float",
        "f16" => "Float16",
        "bf16" => "Float", // no native Swift bfloat16; pass widened
        "i32" => "Int32",
        "u32" => "UInt32",
        "i64" => "Int64",
        "u64" => "UInt64",
        "i8" => "Int8",
        "u8" => "UInt8",
        "bool" => "Bool",
        _ => "UInt32",
    }
}

fn swift_scalar_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" | "u32" => 4,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "i8" | "u8" | "bool" => 1,
        "i64" | "u64" => 8,
        _ => 4,
    }
}
