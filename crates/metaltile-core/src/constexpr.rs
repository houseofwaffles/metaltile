//! ConstExpr: compile-time constant expressions resolved at kernel specialization time.
//!
//! These represent variables like `M`, `N`, `K` in kernel signatures that are
//! marked `#[constexpr]`. They are not known when the DSL is written but are
//! resolved when the kernel is launched with specific dimensions.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A named constexpr variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConstExpr {
    /// The name of the constexpr (e.g. "M", "BLOCK_SIZE").
    name: String,
}

impl ConstExpr {
    /// Create a new constexpr with the given name.
    pub fn new(name: impl Into<String>) -> Self { ConstExpr { name: name.into() } }

    /// The name of this constexpr.
    pub fn name(&self) -> &str { &self.name }
}

impl fmt::Display for ConstExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.name) }
}

impl From<&str> for ConstExpr {
    fn from(s: &str) -> Self { ConstExpr::new(s) }
}

impl From<String> for ConstExpr {
    fn from(s: String) -> Self { ConstExpr::new(s) }
}

/// A collection of resolved constexpr values for a specific kernel launch.
#[derive(Debug, Clone, Default)]
pub struct ConstExprValues {
    values: std::collections::BTreeMap<String, usize>,
}

impl ConstExprValues {
    /// Create empty.
    pub fn new() -> Self { ConstExprValues::default() }

    /// Insert a resolved value.
    pub fn insert(&mut self, name: impl Into<String>, value: usize) {
        self.values.insert(name.into(), value);
    }

    /// Get a resolved value. Panics if not found.
    ///
    /// If `name` is a numeric literal (e.g. "256" from `arange::<256>()`),
    /// it is parsed directly without requiring a mapping entry.
    pub fn get(&self, name: &str) -> usize {
        if let Ok(n) = name.parse::<usize>() {
            return n;
        }
        *self.values.get(name).unwrap_or_else(|| panic!("constexpr '{name}' not resolved"))
    }

    /// Try to get a resolved value.
    pub fn try_get(&self, name: &str) -> Option<usize> { self.values.get(name).copied() }

    /// Resolve a [`ConstExpr`] to its concrete value.
    pub fn resolve(&self, ce: &ConstExpr) -> usize { self.get(ce.name()) }

    /// Iterator over all resolved values.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &usize)> { self.values.iter() }

    /// Number of resolved values.
    pub fn len(&self) -> usize { self.values.len() }

    /// Whether there are no resolved values.
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctor_and_accessors() {
        let ce = ConstExpr::new("BLOCK_SIZE");
        assert_eq!(ce.name(), "BLOCK_SIZE");
        assert_eq!(format!("{ce}"), "BLOCK_SIZE");
    }

    #[test]
    fn from_conversions() {
        let a: ConstExpr = "M".into();
        let b: ConstExpr = "M".to_string().into();
        assert_eq!(a, b);
        assert_eq!(a.name(), "M");
    }

    #[test]
    fn equality_and_ordering() {
        let a = ConstExpr::new("A");
        let b = ConstExpr::new("B");
        assert!(a < b);
        assert_ne!(a, b);
        // Hashing — relies on derive(Hash) producing a consistent value
        // for equal instances; covered by use as a BTreeMap key in
        // ConstExprValues below.
    }

    #[test]
    fn values_new_default_and_empty() {
        let v = ConstExprValues::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        let v2: ConstExprValues = Default::default();
        assert!(v2.is_empty());
    }

    #[test]
    fn values_insert_get_try_get() {
        let mut v = ConstExprValues::new();
        v.insert("M", 64);
        v.insert("N".to_string(), 128);
        assert_eq!(v.len(), 2);
        assert!(!v.is_empty());
        assert_eq!(v.get("M"), 64);
        assert_eq!(v.try_get("M"), Some(64));
        assert_eq!(v.try_get("Z"), None);
    }

    #[test]
    fn values_resolve_via_constexpr() {
        let mut v = ConstExprValues::new();
        v.insert("K", 256);
        let ce = ConstExpr::new("K");
        assert_eq!(v.resolve(&ce), 256);
    }

    #[test]
    fn values_get_parses_numeric_literals() {
        // `arange::<256>()` lowers to a constexpr named "256" — get() must
        // parse it directly without a mapping entry.
        let v = ConstExprValues::new();
        assert_eq!(v.get("256"), 256);
        assert_eq!(v.get("0"), 0);
    }

    #[test]
    #[should_panic(expected = "constexpr 'MISSING' not resolved")]
    fn values_get_panics_on_unresolved_non_numeric() {
        ConstExprValues::new().get("MISSING");
    }

    #[test]
    fn values_iter_yields_inserted_pairs() {
        let mut v = ConstExprValues::new();
        v.insert("A", 1);
        v.insert("B", 2);
        let collected: Vec<(&String, &usize)> = v.iter().collect();
        assert_eq!(collected.len(), 2);
        // BTreeMap → sorted by key, so A then B
        assert_eq!(collected[0].0, "A");
        assert_eq!(*collected[0].1, 1);
        assert_eq!(collected[1].0, "B");
        assert_eq!(*collected[1].1, 2);
    }
}
