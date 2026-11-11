//! Stored sources: typed handles on the base keyspaces an application
//! writes, each a thin layer over a keyspace of a [`Db`](crate::Db).
//!
//! - [`Table`]: one value per key, last writer wins.
//! - [`Set`]: distinct membership, a table of `()`.
//! - [`Cell`]: one value, a table keyed by `()`.
//! - [`Multiset`]: rows with signed multiplicities, kept as counts.
//! - [`Events`]: an append-only log keyed by minted [`Id`]s, with
//!   retention.
//!
//! A handle is a keyspace id and nothing else: `const`, `Copy`, and
//! stateless. Its methods take the [`DbTx`] to write through, and
//! `source` adds it to a graph. The store keeps every one of them, so
//! reopening brings them back, and a changed graph rebuilds from them
//! with nothing for the application to replay.

use std::ops::Bound;

use crate::{
    Bag, Change, Delta, Mergeless, Register, Tally,
    codec::{Codec, OrderedCodec},
    database::DbTx,
    error::Result,
    graph::{NodeRef, OpDag},
    id::Id,
    state::Stored,
    tx::{KeyspaceId, WriteTx},
};

/// What a table can hold as a key.
pub trait Key: OrderedCodec + Clone + Send + Sync + 'static {}
impl<T: OrderedCodec + Clone + Send + Sync + 'static> Key for T {}

/// What a table can hold as a value.
pub trait Value: Codec + Clone + PartialEq + Send + Sync + 'static {}
impl<T: Codec + Clone + PartialEq + Send + Sync + 'static> Value for T {}

/// The graph node of a register-shaped source.
pub type TableNode<K, V> = NodeRef<Register<K, Mergeless<V>>, Stored<K, Mergeless<V>>>;

macro_rules! handle {
    ([$($gen:tt)*] $name:ident [$($args:tt)*], $key:ty, $merge:ty) => {
        impl<$($gen)*> $name<$($args)*> {
            /// The handle on keyspace `id`.
            pub const fn new(id: u64) -> Self {
                $name(KeyspaceId::new(id))
            }

            /// Its keyspace, to read from a snapshot or subscribe to.
            pub const fn id(&self) -> KeyspaceId<$key, $merge> {
                self.0
            }
        }
        impl<$($gen)*> Clone for $name<$($args)*> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($gen)*> Copy for $name<$($args)*> {}
    };
}

mod bag;
pub use bag::*;

mod cell;
pub use cell::*;

mod events;
pub use events::*;

mod set;
pub use set::*;

mod table;
pub use table::*;
