//! MetalTile CPU reference interpreter.
//!
//! Executes kernel IR on the CPU for correctness testing and CI.
//! Every kernel can be run here, compared against reference implementations,
//! and validated without requiring an Apple Silicon Mac.
//!
//! The interpreter operates on `TensorData` — flat buffers of raw bytes
//! with shape and dtype metadata attached.

pub mod interpreter;
pub mod tensor;

pub use interpreter::Interpreter;
pub use tensor::TensorData;
