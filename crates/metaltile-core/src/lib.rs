//! MetalTile core: IR types, shape algebra, and DType system.
//!
//! This crate defines the foundational types that all other crates share:
//! - [`DType`]: numeric types (f16, f32, i32, etc.)
//! - [`Shape`]: compile-time dimension tracking via type-level markers
//! - [`ConstExpr`]: constexpr values resolved at kernel compile time
//! - Kernel IR nodes: the SSA-form intermediate representation

pub mod constexpr;
pub mod dtype;
pub mod error;
pub mod ir;
pub mod shape;
pub mod utils;

pub use constexpr::ConstExpr;
pub use dtype::DType;
pub use error::{Error, Result};
pub use ir::{
    ActKind,
    Block,
    BlockId,
    Kernel,
    KernelMode,
    Op,
    Param,
    TypedSlot,
    UnaryOpKind,
    ValueId,
    VarId,
};
pub use shape::{Dim, DimExpr, Shape, tile};
