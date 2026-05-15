//! Runtime errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalTileError {
    #[error("metal error: {0}")]
    Metal(String),

    #[error("no Metal device found")]
    NoDevice,

    #[error("kernel compilation failed: {0}")]
    Compilation(String),

    #[error("buffer allocation failed: {0}")]
    Buffer(String),

    #[error("dispatch failed: {0}")]
    Dispatch(String),

    #[error("autotune error: {0}")]
    Autotune(String),

    #[error("core error: {0}")]
    Core(#[from] metaltile_core::error::Error),

    #[error("codegen error: {0}")]
    Codegen(#[from] metaltile_codegen::error::Error),

    #[error("not implemented on this platform")]
    UnsupportedPlatform,
}
