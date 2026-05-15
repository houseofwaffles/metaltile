//! Re-exports and placeholder DSL items for `#[kernel]` functions.
//!
//! Import this module with `use metaltile::prelude::*;` in the same Rust module as your kernels.
//! It provides:
//!
//! - facade macros: `#[kernel]`, `#[constexpr]`, `shape!`, and `tile!`
//! - IR-facing helper types: [`ConstExpr`], [`DType`], [`Dim`], [`Shape`], [`KernelMode`], and
//!   [`Context`]
//! - placeholder syntax items such as [`Tensor`], [`program_id`], [`load`], and [`store`]
//!
//! The exported functions exist so Rust can parse kernel bodies before the proc macro runs. The
//! `#[kernel]` macro rewrites the function body, so calling these helpers outside a kernel will
//! panic.
//!
//! Output tensors are identified by parameter name today. Use `c`, `out`, or `output` when you
//! want the generated launch path to treat a tensor parameter as writable output.

use std::{marker::PhantomData, ops::Index};

/// Compile-time symbolic values used in shape annotations and generated IR.
pub use metaltile_core::constexpr::ConstExpr;
/// Scalar and tensor element types supported by the IR and MSL codegen.
pub use metaltile_core::dtype::DType;
/// Kernel execution mode metadata for IR/codegen inspection.
pub use metaltile_core::ir::KernelMode;
/// Shape-building helpers used in tensor annotations.
pub use metaltile_core::shape::{Dim, Shape, tile};
/// Facade macros used in kernel signatures and bodies.
pub use metaltile_macros::{constexpr, kernel, shape, tile};
/// Runtime context used by generated `launch` helpers.
pub use metaltile_runtime::Context;

/// Placeholder tensor type used in `#[kernel]` signatures.
///
/// `Tensor<T, S>` is a zero-sized marker that carries element type `T` and optional shape metadata
/// `S` for proc-macro parsing. The generated launch surface still binds raw byte buffers by
/// parameter name; this type does not own storage or runtime shape information yet.
pub struct Tensor<T, S = ()> {
    _p: PhantomData<(T, S)>,
}

/// `a[idx]` syntax inside a kernel body.
///
/// The body parser recognizes tensor indexing syntactically and lowers it into IR load/store index
/// expressions. This implementation only exists so the Rust parser accepts the syntax.
impl<T, S> Index<u32> for Tensor<T, S> {
    type Output = u32;
    fn index(&self, _idx: u32) -> &u32 { panic!("Tensor indexing only valid inside #[kernel]") }
}

// ---- DSL function stubs (panic if called outside #[kernel]) ----

/// Return the current program/thread id for the given axis.
pub fn program_id<const AXIS: u32>() -> u32 { panic!("program_id only valid inside #[kernel]") }

/// Load a value from a tensor index expression.
pub fn load<T>(_src: u32) -> T { panic!("load only valid inside #[kernel]") }

/// Store a value into a tensor index expression.
pub fn store<T>(_dst: u32, _value: T) { panic!("store only valid inside #[kernel]") }

/// Dot product placeholder used by tiled kernels.
pub fn dot<T>(_a: T, _b: T) -> T { panic!("dot only valid inside #[kernel]") }

// Elementwise math — the body parser recognizes these by name
macro_rules! unary {
    ($name:ident) => {
        pub fn $name<T>(_x: T) -> T {
            panic!(concat!(stringify!($name), " only valid inside #[kernel]"))
        }
    };
}
unary!(exp);
unary!(log);
unary!(sqrt);
unary!(rsqrt);
unary!(abs);
unary!(silu);
unary!(gelu);
unary!(relu);
unary!(tanh);
unary!(sigmoid);
unary!(sin);
unary!(cos);
unary!(ceil);
unary!(floor);
unary!(recip);
