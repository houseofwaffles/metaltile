//! MetalTile runtime: GPU dispatch, buffer management, and autotuning.
//!
//! This crate handles the runtime execution of compiled MetalTile kernels:
//! - Metal device and command queue management
//! - Pipeline state compilation and caching
//! - Buffer allocation and transfer
//! - Autotuner with persistent disk cache

pub mod autotune;
pub mod buffer;
pub mod context;
pub mod error;

pub use context::{Context, DispatchResult};
pub use error::MetalTileError;
