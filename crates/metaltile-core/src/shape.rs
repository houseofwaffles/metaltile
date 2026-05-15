//! Shape algebra: type-level dimension tracking for compile-time shape checking.
//!
//! Dimensions can be:
//! - [`Dim::Known(n)`]: a concrete size known at compile time
//! - [`Dim::ConstExpr`]: a constexpr variable resolved at kernel launch
//! - [`Dim::Any`]: a fully dynamic dimension
//!
//! Shapes are sequences of dimensions with algebraic operations.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::constexpr::ConstExpr;

/// A single dimension — either known, constexpr, or dynamic.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Statically known size (from const generic params, e.g. TILE_M = 32).
    Known(usize),
    /// A constexpr variable (e.g. `N` in a kernel signature).
    ConstExpr(ConstExpr),
    /// Fully dynamic — determined at runtime from the input tensor.
    Any,
}

impl Dim {
    /// The value if known, otherwise panics.
    pub fn as_known(&self) -> usize {
        match self {
            Dim::Known(n) => *n,
            _ => panic!("expected known dimension, got {self:?}"),
        }
    }

    /// Whether this dimension is statically known.
    pub const fn is_known(&self) -> bool { matches!(self, Dim::Known(_)) }

    /// Whether this dimension is constexpr.
    pub const fn is_constexpr(&self) -> bool { matches!(self, Dim::ConstExpr(_)) }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self { Dim::Known(n) }
}

impl From<ConstExpr> for Dim {
    fn from(ce: ConstExpr) -> Self { Dim::ConstExpr(ce) }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Known(n) => write!(f, "{n}"),
            Dim::ConstExpr(ce) => write!(f, "{ce}"),
            Dim::Any => write!(f, "?"),
        }
    }
}

/// A dimension expression: a linear combination of dimensions for indexing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimExpr {
    /// Multiply a constexpr by a coefficient.
    Scale(i64, ConstExpr),
    /// Constant offset.
    Const(i64),
    /// A named constexpr as-is.
    Var(ConstExpr),
    /// Sum of two expressions.
    Add(Box<DimExpr>, Box<DimExpr>),
    /// A fixed range slice: [start..end).
    Range(ConstExpr, i64, i64),
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Scale(c, v) => write!(f, "{c}*{v}"),
            DimExpr::Const(c) => write!(f, "{c}"),
            DimExpr::Var(v) => write!(f, "{v}"),
            DimExpr::Add(a, b) => write!(f, "({a} + {b})"),
            DimExpr::Range(v, start, end) => write!(f, "{v}[{start}..{end})"),
        }
    }
}

/// The shape of a tensor or tile value: a list of dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: smallvec::SmallVec<[Dim; 4]>,
}

impl Shape {
    /// Create a new shape from a list of dimensions.
    pub fn new(dims: impl IntoIterator<Item = Dim>) -> Self {
        Shape { dims: dims.into_iter().collect() }
    }

    /// A scalar (zero-dimensional value).
    pub fn scalar() -> Self { Shape { dims: smallvec::SmallVec::new() } }

    /// The number of dimensions (rank).
    pub fn rank(&self) -> usize { self.dims.len() }

    /// Access a dimension by index.
    pub fn dim(&self, index: usize) -> Option<&Dim> { self.dims.get(index) }

    /// Total number of elements (product of known dimensions).
    /// Returns None if any dimension is not statically known.
    pub fn num_elements(&self) -> Option<usize> {
        self.dims.iter().try_fold(1usize, |acc, d| match d {
            Dim::Known(n) => Some(acc * n),
            _ => None,
        })
    }

    /// Iterate over dimensions.
    pub fn iter(&self) -> impl Iterator<Item = &Dim> { self.dims.iter() }

    /// Check if this shape equals another shape exactly.
    pub fn matches(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        self.dims.iter().zip(other.dims.iter()).all(|(a, b)| {
            match (a, b) {
                // Known must match exactly
                (Dim::Known(an), Dim::Known(bn)) => an == bn,
                // ConstExpr must match by name
                (Dim::ConstExpr(a), Dim::ConstExpr(b)) => a == b,
                // Any matches anything
                (Dim::Any, _) | (_, Dim::Any) => true,
                _ => false,
            }
        })
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return write!(f, "scalar");
        }
        write!(f, "(")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, ")")
    }
}

/// Convenience: create a 2D tile shape from two dimensions.
pub fn tile(rows: impl Into<Dim>, cols: impl Into<Dim>) -> Shape {
    Shape::new([rows.into(), cols.into()])
}
