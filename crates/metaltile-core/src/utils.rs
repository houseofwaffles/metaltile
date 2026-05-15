//! Utility types for metaltile-core.

/// A counter for generating unique IDs.
#[derive(Debug, Clone, Default)]
pub struct IdCounter {
    next: u32,
}

impl IdCounter {
    pub fn new() -> Self { IdCounter { next: 0 } }

    pub fn next(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
}
