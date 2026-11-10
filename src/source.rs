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
    db::DbTx,
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

/// One value per key: upsert and delete, last writer wins.
pub struct Table<K, V: Value>(KeyspaceId<K, Mergeless<V>>);
handle!([K, V: Value] Table[K, V], K, Mergeless<V>);

impl<K: Key, V: Value> Table<K, V> {
    /// Add it to `graph` as a source: each run's changes, and the table
    /// as state.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<K, V> {
        graph.table(self.0)
    }

    /// The value of `key`, with this transaction's writes.
    pub fn get(&self, tx: &mut DbTx<'_>, key: &K) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(key)?.map(|v| v.into_owned()))
    }

    /// Set `key` to `value`, replacing what was there.
    pub fn upsert(&self, tx: &mut DbTx<'_>, key: K, value: V) -> Result<()> {
        tx.table(self.0).set(key, Some(value))
    }

    /// Delete `key`. Returns whether it was there.
    pub fn remove(&self, tx: &mut DbTx<'_>, key: &K) -> Result<bool> {
        let mut ks = tx.table(self.0);
        let Some(old) = ks.get(key)?.map(|v| v.into_owned()) else {
            return Ok(false);
        };
        ks.absorb(key.clone(), Change::Delete(old))?;
        Ok(true)
    }

    /// Change `key`'s value in place. Returns whether it was there.
    pub fn update(&self, tx: &mut DbTx<'_>, key: &K, f: impl FnOnce(&mut V)) -> Result<bool> {
        let mut ks = tx.table(self.0);
        let Some(old) = ks.get(key)?.map(|v| v.into_owned()) else {
            return Ok(false);
        };
        let mut new = old.clone();
        f(&mut new);
        if new != old {
            ks.absorb(key.clone(), Change::Replace(old, new))?;
        }
        Ok(true)
    }

    /// [`upsert`](Self::upsert) every row.
    pub fn extend(&self, tx: &mut DbTx<'_>, rows: impl IntoIterator<Item = (K, V)>) -> Result<()> {
        let mut ks = tx.table(self.0);
        for (k, v) in rows {
            ks.set(k, Some(v))?;
        }
        Ok(())
    }
}

impl<V: Value> Table<Id<V>, V> {
    /// Add `value` under a freshly minted id.
    pub fn insert(&self, tx: &mut DbTx<'_>, value: V) -> Result<Id<V>> {
        let id = tx.mint(self.0.raw());
        tx.table(self.0).absorb(id, Change::Insert(value))?;
        Ok(id)
    }
}

/// Distinct membership: a [`Table`] of `()`.
pub struct Set<T>(KeyspaceId<T, Mergeless<()>>);
handle!([T] Set[T], T, Mergeless<()>);

impl<T: Key> Set<T> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<T, ()> {
        graph.table(self.0)
    }

    pub fn contains(&self, tx: &mut DbTx<'_>, value: &T) -> Result<bool> {
        Ok(tx.table(self.0).get(value)?.is_some())
    }

    /// Add `value`. Returns whether it was new.
    pub fn insert(&self, tx: &mut DbTx<'_>, value: T) -> Result<bool> {
        let mut ks = tx.table(self.0);
        if ks.get(&value)?.is_some() {
            return Ok(false);
        }
        ks.absorb(value, Change::Insert(()))?;
        Ok(true)
    }

    /// Remove `value`. Returns whether it was there.
    pub fn remove(&self, tx: &mut DbTx<'_>, value: &T) -> Result<bool> {
        let mut ks = tx.table(self.0);
        if ks.get(value)?.is_none() {
            return Ok(false);
        }
        ks.absorb(value.clone(), Change::Delete(()))?;
        Ok(true)
    }

    /// [`insert`](Self::insert) every value.
    pub fn extend(&self, tx: &mut DbTx<'_>, values: impl IntoIterator<Item = T>) -> Result<()> {
        for v in values {
            self.insert(tx, v)?;
        }
        Ok(())
    }
}

/// One value: a [`Table`] keyed by `()`.
pub struct Cell<V: Value>(KeyspaceId<(), Mergeless<V>>);
handle!([V: Value] Cell[V], (), Mergeless<V>);

impl<V: Value> Cell<V> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<(), V> {
        graph.table(self.0)
    }

    pub fn get(&self, tx: &mut DbTx<'_>) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(&())?.map(|v| v.into_owned()))
    }

    pub fn set(&self, tx: &mut DbTx<'_>, value: V) -> Result<()> {
        tx.table(self.0).set((), Some(value))
    }

    pub fn clear(&self, tx: &mut DbTx<'_>) -> Result<()> {
        tx.table(self.0).set((), None)
    }
}

/// Rows with signed multiplicities, stored as each row's count; a row
/// whose count returns to zero is gone. Its graph source gives the
/// changes in count, as a [`Bag`].
pub struct Multiset<T>(KeyspaceId<T, Tally>);
handle!([T] Multiset[T], T, Tally);

impl<T: Key> Multiset<T> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> NodeRef<Bag<T>> {
        graph.multiset(self.0)
    }

    /// How many of `row` there are.
    pub fn get(&self, tx: &mut DbTx<'_>, row: &T) -> Result<Delta> {
        Ok(tx.table(self.0).get(row)?.map_or(0, |n| *n))
    }

    /// Add `times` of `row`; negative to take away. Reads nothing: the
    /// count is the sum of what was added, and a row whose count is zero
    /// is not there.
    pub fn add(&self, tx: &mut DbTx<'_>, row: T, times: Delta) -> Result<()> {
        tx.table(self.0).add(row, times)
    }

    /// Take away `times` of `row`.
    pub fn remove(&self, tx: &mut DbTx<'_>, row: T, times: Delta) -> Result<()> {
        self.add(tx, row, -times)
    }

    /// Take away every `row`.
    pub fn clear(&self, tx: &mut DbTx<'_>, row: &T) -> Result<()> {
        let mut ks = tx.table(self.0);
        match ks.get(row)?.map(|n| *n) {
            Some(n) => ks.add(row.clone(), n.checked_neg().expect("negate overflowed")),
            None => Ok(()),
        }
    }
}

/// An append-only log: each row gets a minted [`Id`], increasing across
/// commits and within one. What is kept is set with
/// [`Db::retain`](crate::Db::retain), and evicted rows leave as deletes
/// in the transaction that pushed them out, so the graph sees them go.
pub struct Events<V: Value>(KeyspaceId<Id<V>, Mergeless<V>>);
handle!([V: Value] Events[V], Id<V>, Mergeless<V>);

impl<V: Value> Events<V> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<Id<V>, V> {
        graph.table(self.0)
    }

    /// Append `value`, returning its id.
    pub fn push(&self, tx: &mut DbTx<'_>, value: V) -> Result<Id<V>> {
        let id = tx.mint(self.0.raw());
        tx.table(self.0).absorb(id, Change::Insert(value))?;
        Ok(id)
    }

    pub fn get(&self, tx: &mut DbTx<'_>, id: Id<V>) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(&id)?.map(|v| v.into_owned()))
    }
}

/// What an [`Events`] log keeps. Evaluated against the log itself, never
/// a wall clock, so replaying it gives the same state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retain {
    Forever,
    /// At most this many rows; the oldest go first.
    Count(usize),
    /// The rows pushed in the last this many epochs, counting back from
    /// the newest row's.
    Epochs(u64),
}

/// A retention policy applied to one log at every commit.
pub(crate) trait Retention: Send {
    /// The keyspace it keeps.
    fn keyspace(&self) -> u64;
    /// Evict what the policy no longer keeps, given how many rows this
    /// transaction pushed.
    fn apply(&mut self, tx: &mut WriteTx<'_>, pushed: u32) -> Result<()>;
    /// The transaction did not commit: forget what was learned from it.
    fn reset(&mut self);
}

pub(crate) struct EventsRetention<V: Value> {
    pub id: KeyspaceId<Id<V>, Mergeless<V>>,
    pub retain: Retain,
    /// Rows in the log as of the last commit; counted on first use, then
    /// kept up with each commit's pushes and evictions.
    pub live: Option<usize>,
    /// Whether the log already satisfies the policy, so a commit that
    /// pushes nothing has nothing to evict.
    pub settled: bool,
}

impl<V: Value> Retention for EventsRetention<V> {
    fn keyspace(&self) -> u64 {
        self.id.raw()
    }

    fn apply(&mut self, tx: &mut WriteTx<'_>, pushed: u32) -> Result<()> {
        if pushed == 0 && self.settled {
            return Ok(());
        }
        let mut ks = tx.keyspace(self.id);
        let evict: Vec<Id<V>> = match self.retain {
            Retain::Forever => return Ok(()),
            Retain::Count(keep) => {
                let live = match self.live {
                    Some(live) => live + pushed as usize,
                    None => {
                        let mut n = 0;
                        for row in ks.range(..) {
                            row?;
                            n += 1;
                        }
                        n
                    }
                };
                let excess = live.saturating_sub(keep);
                self.live = Some(live - excess);
                ks.range(..)
                    .take(excess)
                    .map(|row| row.map(|(k, _)| *k))
                    .collect::<Result<_>>()?
            }
            Retain::Epochs(keep) => {
                let Some(newest) = ks.range_rev(..).next() else {
                    return Ok(());
                };
                let newest = *newest?.0;
                let first = (newest.epoch() + 1).saturating_sub(keep);
                ks.range((Bound::Unbounded, Bound::Excluded(Id::minted(first, 0))))
                    .map(|row| row.map(|(k, _)| *k))
                    .collect::<Result<_>>()?
            }
        };
        for id in evict {
            ks.set(id, None)?;
        }
        self.settled = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.live = None;
        self.settled = false;
    }
}
