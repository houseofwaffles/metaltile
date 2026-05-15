//! Codegen errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported operation in MSL codegen: {0}")]
    UnsupportedOp(String),

    #[error("MSL generation error: {0}")]
    Generation(String),

    #[error("core error: {0}")]
    Core(#[from] metaltile_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
