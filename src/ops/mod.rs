//! Operators, one module per family:
//!
//! - [`Sum`], [`Distinct`], [`IndexBy`] and [`Check`], which need only
//!   their own keyspace.
//!
//! - [`linear`]: per-row operators over keyed rows, fused like the rest.
//! - [`Arrange`]: rows grouped by key, as a state other operators read;
//!   [`CountRows`]: just which rows are present.
//! - [`join`]: the inner join, reactive on both sides.
//! - [`semijoin`]: rows whose key the other side does (or does not) have.
//! - [`aggregate`]: per-key reductions in data position (min, max, top-k,
//!   aggregate, and any other). Totals and counts are [`Sum`].
//! - [`foreign_key`]: references to keys the other side lacks, and
//!   [`Reject`] to refuse a tx that has any.
//! - [`DistinctRows`]: each keyed row once, its counts read by key.
//! - [`Materialize`]: a register's rows, kept as a register.
//! - [`Minus`] and [`Flatten`], small adapters.
//!
//! An operator that needs a parent's whole output reads the parent's
//! [state](crate::state) instead of keeping a copy: a table, a register an
//! operator keeps, or an [`Arrange`] node, which several operators can
//! share. Build them with
//! [`OpDag::binary_with`](crate::graph::OpDag::binary_with) and its
//! siblings, which hand them their parents' state handles.

pub mod aggregate;
mod arranged;
mod check;
mod concat;
pub mod distinct;
pub mod foreign_key;
mod index_by;
pub mod join;
pub mod linear;
pub mod materialize;
pub mod semijoin;
mod sum;

pub use aggregate::{Aggregate, ReduceMany, Take};
pub use arranged::{Arrange, CountRows, Key};
pub(crate) use arranged::{Groups, arrange, dirty_keys_into, multiply, read_group};
pub use check::{Check, Reject};
pub use concat::Minus;
pub use distinct::{Distinct, DistinctCounts, DistinctRows, Distinctly};
pub use foreign_key::ForeignKey;
pub use index_by::IndexBy;
pub use join::Join;
pub use linear::{Explode, Filter, FilterMap, FlatMap, Flatten, Map, Negate};
pub use materialize::Materialize;
pub use semijoin::Semijoin;
pub use sum::Sum;

#[cfg(test)]
mod node_tests;
#[cfg(test)]
mod tests;
