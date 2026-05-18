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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Dim ──────────────────────────────────────────────────────────────

    #[test]
    fn dim_classification_and_accessors() {
        let known = Dim::Known(32);
        let ce = Dim::ConstExpr(ConstExpr::new("M"));
        let any = Dim::Any;
        assert!(known.is_known());
        assert!(!known.is_constexpr());
        assert!(!ce.is_known());
        assert!(ce.is_constexpr());
        assert!(!any.is_known());
        assert!(!any.is_constexpr());
        assert_eq!(known.as_known(), 32);
    }

    #[test]
    #[should_panic(expected = "expected known dimension")]
    fn dim_as_known_panics_on_dynamic() {
        Dim::Any.as_known();
    }

    #[test]
    fn dim_from_conversions() {
        let d: Dim = 64usize.into();
        assert_eq!(d, Dim::Known(64));
        let d: Dim = ConstExpr::new("K").into();
        assert!(matches!(d, Dim::ConstExpr(_)));
    }

    #[test]
    fn dim_display_renders_each_variant() {
        assert_eq!(format!("{}", Dim::Known(8)), "8");
        assert_eq!(format!("{}", Dim::ConstExpr(ConstExpr::new("M"))), "M");
        assert_eq!(format!("{}", Dim::Any), "?");
    }

    // ── DimExpr ──────────────────────────────────────────────────────────

    #[test]
    fn dim_expr_display_covers_each_variant() {
        let m = ConstExpr::new("M");
        assert_eq!(format!("{}", DimExpr::Scale(4, m.clone())), "4*M");
        assert_eq!(format!("{}", DimExpr::Const(7)), "7");
        assert_eq!(format!("{}", DimExpr::Var(m.clone())), "M");
        assert_eq!(
            format!("{}", DimExpr::Add(Box::new(DimExpr::Const(1)), Box::new(DimExpr::Var(m.clone())))),
            "(1 + M)",
        );
        assert_eq!(format!("{}", DimExpr::Range(m, 0, 16)), "M[0..16)");
    }

    // ── Shape ────────────────────────────────────────────────────────────

    #[test]
    fn scalar_has_rank_zero() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert!(s.dim(0).is_none());
        assert_eq!(s.num_elements(), Some(1));
        assert_eq!(format!("{s}"), "scalar");
    }

    #[test]
    fn rank_and_dim_access() {
        let s = Shape::new([Dim::Known(4), Dim::Known(8), Dim::Any]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dim(0), Some(&Dim::Known(4)));
        assert_eq!(s.dim(1), Some(&Dim::Known(8)));
        assert_eq!(s.dim(2), Some(&Dim::Any));
        assert!(s.dim(3).is_none());
    }

    #[test]
    fn num_elements_is_some_only_when_all_known() {
        assert_eq!(Shape::new([Dim::Known(2), Dim::Known(3)]).num_elements(), Some(6));
        assert_eq!(Shape::new([Dim::Known(4), Dim::Any]).num_elements(), None);
        assert_eq!(
            Shape::new([Dim::Known(2), Dim::ConstExpr(ConstExpr::new("N"))]).num_elements(),
            None,
        );
    }

    #[test]
    fn iter_yields_each_dim() {
        let s = Shape::new([Dim::Known(2), Dim::Known(3)]);
        let collected: Vec<&Dim> = s.iter().collect();
        assert_eq!(collected, vec![&Dim::Known(2), &Dim::Known(3)]);
    }

    #[test]
    fn matches_strict_for_known_dims() {
        let a = Shape::new([Dim::Known(4), Dim::Known(8)]);
        let b = Shape::new([Dim::Known(4), Dim::Known(8)]);
        let c = Shape::new([Dim::Known(4), Dim::Known(9)]);
        assert!(a.matches(&b));
        assert!(!a.matches(&c));
    }

    #[test]
    fn matches_strict_for_constexpr_by_name() {
        let m = ConstExpr::new("M");
        let n = ConstExpr::new("N");
        let a = Shape::new([Dim::ConstExpr(m.clone())]);
        let b = Shape::new([Dim::ConstExpr(m)]);
        let c = Shape::new([Dim::ConstExpr(n)]);
        assert!(a.matches(&b));
        assert!(!a.matches(&c));
    }

    #[test]
    fn matches_any_dim_wildcards_either_side() {
        let known = Shape::new([Dim::Known(7)]);
        let any = Shape::new([Dim::Any]);
        assert!(known.matches(&any));
        assert!(any.matches(&known));
    }

    #[test]
    fn matches_false_on_rank_mismatch() {
        let r1 = Shape::new([Dim::Known(2)]);
        let r2 = Shape::new([Dim::Known(2), Dim::Known(3)]);
        assert!(!r1.matches(&r2));
    }

    #[test]
    fn matches_false_on_mixed_known_vs_constexpr() {
        let known = Shape::new([Dim::Known(8)]);
        let ce = Shape::new([Dim::ConstExpr(ConstExpr::new("N"))]);
        // Known vs ConstExpr falls through the default `_ => false` arm.
        assert!(!known.matches(&ce));
        assert!(!ce.matches(&known));
    }

    #[test]
    fn display_renders_multi_dim_shape() {
        let s = Shape::new([Dim::Known(4), Dim::ConstExpr(ConstExpr::new("N")), Dim::Any]);
        assert_eq!(format!("{s}"), "(4, N, ?)");
    }

    // ── tile() helper ────────────────────────────────────────────────────

    #[test]
    fn tile_builds_two_dim_shape() {
        let s = tile(32usize, 64usize);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.dim(0), Some(&Dim::Known(32)));
        assert_eq!(s.dim(1), Some(&Dim::Known(64)));
        assert_eq!(s.num_elements(), Some(32 * 64));
    }

    #[test]
    fn tile_accepts_constexpr_rows() {
        let s = tile(ConstExpr::new("M"), 16usize);
        assert!(matches!(s.dim(0), Some(Dim::ConstExpr(_))));
        assert_eq!(s.dim(1), Some(&Dim::Known(16)));
    }
}
