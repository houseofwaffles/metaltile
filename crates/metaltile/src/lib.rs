//! MetalTile facade crate.
//!
//! `metaltile` re-exports the DSL macros, compile-time placeholder types, IR/codegen crates,
//! and runtime entry points used to define and launch `#[kernel]` functions.
//!
//! # Quick start
//!
//! ```no_run
//! use metaltile::prelude::*;
//!
//! fn encode_f32s(values: &[f32]) -> Vec<u8> {
//!     let mut bytes = Vec::with_capacity(values.len() * core::mem::size_of::<f32>());
//!     for value in values {
//!         bytes.extend_from_slice(&value.to_ne_bytes());
//!     }
//!     bytes
//! }
//!
//! fn decode_f32s(bytes: &[u8]) -> Vec<f32> {
//!     bytes
//!         .chunks_exact(core::mem::size_of::<f32>())
//!         .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
//!         .collect()
//! }
//!
//! #[kernel]
//! fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
//!     let idx = program_id::<0>();
//!     store(c[idx], load(a[idx]) + load(b[idx]));
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = Context::new()?;
//! if !ctx.has_gpu() {
//!     return Ok(());
//! }
//!
//! let n = 256usize;
//! let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
//! let b = vec![1.0f32; n];
//! let result = vector_add::launch(&ctx)
//!     .input("a", encode_f32s(&a))
//!     .input("b", encode_f32s(&b))
//!     .input("c", vec![0; a.len() * core::mem::size_of::<f32>()])
//!     .dispatch()?;
//!
//! let c = decode_f32s(result.outputs.get("c").expect("output buffer"));
//! assert_eq!(c[0], 1.0);
//! assert_eq!(c[255], 256.0);
//! # Ok(())
//! # }
//! ```
//!
//! # Writing kernels
//!
//! - Import `metaltile::prelude::*` anywhere you define `#[kernel]` functions.
//! - Output tensors are identified by parameter name today. The facade treats `c`, `out`, and
//!   `output` as writable outputs.
//! - `launch(&ctx).input(name, bytes)` binds raw `Vec<u8>` buffers by parameter name, and
//!   [`DispatchResult`] returns output bytes under the same output name.
//! - `Tensor<T, Shape>` and `#[constexpr]` annotate IR/codegen metadata, but the current launch
//!   builder is still byte-buffer oriented.
//! - `#[constexpr]` parameters become extra constant-buffer bindings in generated MSL. The facade
//!   launch builder does not bind them automatically yet, so dispatch examples currently avoid
//!   constexpr-dependent kernels.
//! - The current elementwise launch path sizes its grid from the output buffer and uses 256-thread
//!   groups without inserting a tail guard, so examples should use element counts that match that
//!   dispatch shape.
//! - `KernelMode` starts as [`core::ir::KernelMode::Elementwise`] and may be adjusted by later
//!   lowering/codegen passes based on the IR shape and ops you emit.
//! - Use `<kernel>::build_kernel_ir()` or `metaltile::codegen::msl::MslGenerator` to inspect the
//!   generated IR and MSL before dispatching.

pub mod prelude;

/// Crate version from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the crate version from `Cargo.toml`.
pub const fn version() -> &'static str { VERSION }

// Re-exports
/// Codegen entry points, including the MSL generator.
pub use metaltile_codegen as codegen;
/// Error returned by `metaltile::codegen` helpers.
pub use metaltile_codegen::error::Error as CodegenError;
/// Core IR, dtype, shape, and constexpr definitions.
pub use metaltile_core as core;
/// CPU interpreter for running IR without Metal.
pub use metaltile_interp as interp;
/// Proc macros and helper macros used by kernel definitions.
pub use metaltile_macros::{constexpr, kernel, shape, tile};
/// Runtime context, dispatch result, and top-level runtime error.
pub use metaltile_runtime::{Context, DispatchResult, MetalTileError};
/// Placeholder tensor marker used in `#[kernel]` signatures.
pub use prelude::Tensor;
