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

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[DType] = &[
        DType::F32, DType::F16, DType::BF16,
        DType::I32, DType::I8, DType::I4,
        DType::U8, DType::U32, DType::U64, DType::I64,
        DType::Bool,
    ];

    #[test]
    fn size_bytes_per_variant() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::U32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::I4.size_bytes(), 1);
        assert_eq!(DType::Bool.size_bytes(), 1);
        assert_eq!(DType::U64.size_bytes(), 8);
        assert_eq!(DType::I64.size_bytes(), 8);
    }

    #[test]
    fn is_float_classification() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        for &dt in [DType::I32, DType::I8, DType::I4, DType::U8, DType::U32,
                    DType::U64, DType::I64, DType::Bool].iter() {
            assert!(!dt.is_float(), "{dt:?} should not be float");
        }
    }

    #[test]
    fn is_integer_classification() {
        for &dt in [DType::I32, DType::I8, DType::I4, DType::U8, DType::U32,
                    DType::U64, DType::I64].iter() {
            assert!(dt.is_integer(), "{dt:?} should be integer");
        }
        // Floats and Bool are not classified as integer.
        for &dt in [DType::F32, DType::F16, DType::BF16, DType::Bool].iter() {
            assert!(!dt.is_integer(), "{dt:?} should not be integer");
        }
    }

    #[test]
    fn float_and_integer_are_disjoint() {
        for &dt in ALL {
            assert!(!(dt.is_float() && dt.is_integer()), "{dt:?} can't be both");
        }
    }

    #[test]
    fn msl_names() {
        assert_eq!(DType::F32.msl_name(), "float");
        assert_eq!(DType::F16.msl_name(), "half");
        assert_eq!(DType::BF16.msl_name(), "bfloat");
        assert_eq!(DType::I32.msl_name(), "int");
        assert_eq!(DType::I8.msl_name(), "char");
        assert_eq!(DType::I4.msl_name(), "char");
        assert_eq!(DType::U8.msl_name(), "uchar");
        assert_eq!(DType::U32.msl_name(), "uint");
        assert_eq!(DType::U64.msl_name(), "ulong");
        assert_eq!(DType::I64.msl_name(), "long");
        assert_eq!(DType::Bool.msl_name(), "bool");
    }

    #[test]
    fn rust_names() {
        assert_eq!(DType::F32.rust_name(), "f32");
        assert_eq!(DType::F16.rust_name(), "half::f16");
        assert_eq!(DType::BF16.rust_name(), "half::bf16");
        assert_eq!(DType::I32.rust_name(), "i32");
        assert_eq!(DType::I8.rust_name(), "i8");
        assert_eq!(DType::I4.rust_name(), "i8");
        assert_eq!(DType::U8.rust_name(), "u8");
        assert_eq!(DType::U32.rust_name(), "u32");
        assert_eq!(DType::U64.rust_name(), "u64");
        assert_eq!(DType::I64.rust_name(), "i64");
        assert_eq!(DType::Bool.rust_name(), "bool");
    }

    #[test]
    fn display_matches_rust_name_for_simple_dtypes() {
        // Display uses the short Rust-style name (without crate prefix).
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::I4), "i4");
        assert_eq!(format!("{}", DType::Bool), "bool");
    }

    #[test]
    fn copy_and_eq_round_trip_through_match() {
        // Smoke check: Copy + PartialEq + Hash trait derives wire up.
        use std::hash::{Hash, Hasher};
        for &dt in ALL {
            let copy = dt;
            assert_eq!(dt, copy);
            let mut h = std::collections::hash_map::DefaultHasher::new();
            dt.hash(&mut h);
            let _ = h.finish();
        }
    }
}
