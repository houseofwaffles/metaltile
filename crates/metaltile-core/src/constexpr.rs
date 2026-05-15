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
