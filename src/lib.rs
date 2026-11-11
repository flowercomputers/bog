//! An embedded, incrementally maintained database.
//!
//! Applications write base tables; a dataflow graph keeps derived state
//! (sums, joins, indexes, aggregates, constraint checks) up to date with
//! every commit, and stores it beside the tables.
//!
//! The [`schema`] module is the front door: declare sources and views as
//! the fields of one struct, define the views as dataflow over typed
//! streams, and write and read through the struct's generated faces.
//!
//! Bottom up:
//!
//! - [`store`]: the storage engine. A write-ahead log, immutable layers
//!   in memory folded in the background, and checkpoint tables on disk
//!   read through a block cache, so data can outgrow memory.
//!   [`Snapshot`]s read a consistent version in either direction, and
//!   [`subscribe`] follows a keyspace's committed changes. Keys and
//!   values are encoded with [`codec`]'s traits; a keyspace's values are
//!   whole ([`Mergeless`]) or, as a [`Tally`], amounts added without
//!   reading.
//! - [`dataflow`]: the runtime. The buffers that flow along a graph's
//!   edges, what an operator is, the graph ([`OpDag`]), and the
//!   [state] handles through which a node's children read its whole
//!   output rather than copy it.
//! - [`ops`]: the operators. [`Sum`], [`Distinct`], [`IndexBy`],
//!   [`Check`], the linear stages fused by [`FusedLinears`], joins,
//!   semijoins, reductions and foreign keys.
//! - [`source`] and [`Db`]: stored base relations (tables, sets, cells,
//!   multisets, event logs with minted [`Id`]s and retention) and the
//!   transaction loop: [`Db::write`] runs a transaction, then the graph
//!   over what it wrote, then commits. When the graph changes, opening
//!   rebuilds derived state from the tables.
//! - [`stream`]: the typed stream kinds views are declared over.
//! - [`schema`]: sources and views as one struct, with the write and
//!   read sides derived. [`constraints`] and [`search`] are domains
//!   built on it.

#![allow(clippy::type_complexity)]

pub mod store;
pub use store::{
    DurabilityPolicy, MemoryStats, Snapshot, Store, StoreBuilder, StoreOptions,
    merge::{Change, Mergeless, Tally},
    txn::{Keyspace, KeyspaceId, WriteTx},
};
// the storage engine's modules, at the paths they had as the crate grew
pub use store::txn as tx;
pub use store::{codec, cursor, index, merge, subscribe};

pub mod dataflow;
pub use dataflow::{
    Bag, Binary, Buffer, Delta, Emit, FusedLinears, Ident, Nary, Node, NodeRef, OpDag, Register,
    Then, Unary, graph, state, vec_scratch,
};

pub mod ops;
pub use ops::{Check, Distinct, Explode, Filter, FilterMap, FlatMap, IndexBy, Map, Negate, Sum};

mod database;
pub use database::{Db, DbTx};

pub mod source;

pub mod stream;

pub mod schema;

pub mod constraints;

pub mod search;

mod rejected;
pub use rejected::Rejected;

pub mod error;
pub use error::{Error, Result};

pub mod id;
pub use id::Id;
