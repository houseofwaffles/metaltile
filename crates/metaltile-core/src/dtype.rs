//! Numeric types supported in MetalTile kernels.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Supported data types for tensor elements and tile values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754 binary16)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit signed integer
    I32,
    /// 8-bit signed integer
    I8,
    /// 4-bit signed integer (packed, used for quantized weights)
    I4,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 64-bit signed integer
    I64,
    /// Boolean
    Bool,
}

impl DType {
    /// Size in bytes of a single element.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::I4 => 1, // packed, but addressable as 1 byte
            DType::U64 | DType::I64 => 8,
            DType::Bool => 1,
        }
    }

    /// Whether this is a floating-point type.
    pub const fn is_float(self) -> bool { matches!(self, DType::F32 | DType::F16 | DType::BF16) }

    /// Whether this is an integer type.
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            DType::I32 | DType::I8 | DType::I4 | DType::U8 | DType::U32 | DType::U64 | DType::I64
        )
    }

    /// Metal Shading Language name for this type.
    pub fn msl_name(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::BF16 => "bfloat", // custom type in MSL
            DType::I32 => "int",
            DType::I8 => "char",
            DType::I4 => "char", // packed char
            DType::U8 => "uchar",
            DType::U32 => "uint",
            DType::U64 => "ulong",
            DType::I64 => "long",
            DType::Bool => "bool",
        }
    }

    /// Rust equivalent type for CPU interpretation.
    pub fn rust_name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "half::f16",
            DType::BF16 => "half::bf16",
            DType::I32 => "i32",
            DType::I8 => "i8",
            DType::I4 => "i8", // stored as i8
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I8 => write!(f, "i8"),
            DType::I4 => write!(f, "i4"),
            DType::U8 => write!(f, "u8"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::I64 => write!(f, "i64"),
            DType::Bool => write!(f, "bool"),
        }
    }
}
