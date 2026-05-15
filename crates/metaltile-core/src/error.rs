//! Error types for metaltile-core.

use thiserror::Error;

/// Core error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Type mismatch between expected and actual shapes.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// A constexpr variable was not resolved.
    #[error("unresolved constexpr: {0}")]
    UnresolvedConstExpr(String),

    /// A dimension was expected to be known but wasn't.
    #[error("dimension is not statically known: {0}")]
    UnknownDimension(String),

    /// Invalid rank for an operation.
    #[error("invalid rank: expected {expected}, got {actual}")]
    InvalidRank { expected: usize, actual: usize },

    /// General IR validation error.
    #[error("IR validation error: {0}")]
    Validation(String),

    /// An operation references an unknown value.
    #[error("unknown value reference: {0}")]
    UnknownValue(String),

    /// Internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, Error>;
