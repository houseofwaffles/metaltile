//! MetalTile codegen: lowers the algorithm IR to Metal Shading Language (MSL).
//!
//! This crate performs:
//! - Schedule application (thread-to-tile mapping, vectorization)
//! - MSL text generation
//! - Optimization passes (fusion, working-set analysis, pipelining)
//!
//! The output is a valid MSL source string that can be compiled by the
//! Metal runtime.

pub mod error;
pub mod msl;
pub mod passes;

pub use error::{Error, Result};
pub use msl::MslGenerator;
pub use passes::tile_lowering::TileSchedule;
