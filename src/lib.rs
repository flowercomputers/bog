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
//!   [`store::subscribe`] follows a keyspace's committed changes. Keys and
//!   values are encoded with [`store::codec`]'s traits; a keyspace's values are
//!   whole ([`Mergeless`]) or, as a [`Tally`], amounts added without
//!   reading.
//! - [`dataflow`]: the runtime. The buffers that flow along a graph's
//!   edges, what an operator is, the graph ([`raw::OpDag`]), and the
//!   [state](dataflow::state) handles through which a node's children read its whole
//!   output rather than copy it.
//! - [`ops`]: the operators. [`Sum`](ops::Sum), [`Distinct`](ops::Distinct), [`IndexBy`](ops::IndexBy),
//!   [`Check`](ops::Check), the linear stages fused by
//!   [`FusedLinears`](dataflow::FusedLinears), joins,
//!   semijoins, reductions and foreign keys.
//! - [`source`] and [`raw::Db`]: stored base relations (tables, sets, cells,
//!   multisets, event logs with minted [`Id`]s and retention) and the
//!   transaction loop: [`raw::Db::write`] runs a transaction, then the graph
//!   over what it wrote, then commits. When the graph changes, opening
//!   rebuilds derived state from the tables.
//! - [`stream`]: the typed stream kinds views are declared over.
//! - [`schema`]: sources and views as one struct, with the write and
//!   read sides derived. [`constraints`] and [`search`] are domains
//!   built on it, and [`tick`] batches writes into one commit per tick.

//!
//! The crate root exports the schema surface. The dataflow runtime, the
//! operators and the raw transaction driver are reachable under [`raw`]
//! (and their own modules).

#![allow(clippy::type_complexity)]

pub mod store;
pub use store::{
    DurabilityPolicy, MemoryStats, Snapshot, Store, StoreBuilder, StoreOptions,
    codec::OrderedCodec,
    merge::{Merge, Mergeless, Tally},
};

pub mod dataflow;

pub mod ops;

/// The raw API: build graphs by hand from the dataflow runtime and the
/// operators, write base keyspaces through the [`source`] handles, and
/// drive it all with [`raw::Db`].
pub mod raw {
    pub use crate::database::{Db, DbTx, ReplayGiven};
    pub use crate::dataflow::*;
    pub use crate::ops::*;
    pub use crate::source::*;
    pub use crate::store::{
        merge::Change,
        subscribe::{Closed, Delta as Committed, Subscription},
        txn::{Keyspace, KeyspaceId, WriteTx},
    };
}

pub mod source;

pub mod stream;

pub mod schema;
pub use schema::*;

pub mod constraints;

pub mod search;

pub mod tick;

mod database;

pub mod error;
pub use error::{Error, Result};

mod id;

mod rejected;
