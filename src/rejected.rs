//! [`Rejected`]: what a refused write says.

use std::fmt;

/// A refused write: which source or view refused, and why. What the
/// predicate forms of the constraints (`reject`, `foreign_key`,
/// `unique`, `unique_key`) refuse with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    pub at: String,
    pub reason: String,
}

impl Rejected {
    pub fn new(at: impl Into<String>, reason: impl Into<String>) -> Self {
        Rejected {
            at: at.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.at, self.reason)
    }
}

impl std::error::Error for Rejected {}
