//! The read facades: [`Read`] for anything register-shaped (tables,
//! event logs, map views), [`GroupRead`] for grouped views, and the
//! readers behind them, which read a keyspace from a snapshot or from
//! inside a transaction.
//!
//! A facade is bound to one [`At`]: a snapshot, the state a transaction
//! started from, or the transaction itself. Scans of a snapshot are
//! lazy; scans inside a transaction are collected first, since the
//! transaction cannot be held open across an iterator.

use std::{
    borrow::Cow,
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use crate::{
    dataflow::Delta,
    dataflow::state::{Arrangement, Edge},
    error::Result,
    ops::DistinctCounts,
    store::codec::Codec,
    store::cursor::Cursor,
    store::merge::Merge,
    store::merge::{Mergeless, Tally},
    store::txn::{KeyspaceId, WriteTx},
    store::{Snapshot, Store},
};

use super::{
    At, Data,
    subscribe::{self, Change, Subscription},
};

/// Reads a register-shaped keyspace as `(K, V)` rows.
pub(crate) trait KeyedReader<K, V>: Send + Sync {
    fn get(&self, snap: &Snapshot<'_>, key: &K) -> Result<Option<V>>;
    fn get_tx(&self, tx: &mut WriteTx<'_>, key: &K) -> Result<Option<V>>;
    fn scan<'s>(
        &self,
        snap: &'s Snapshot<'s>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
    ) -> Box<dyn Iterator<Item = Result<(K, V)>> + 's>;
    fn scan_tx(
        &self,
        tx: &mut WriteTx<'_>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
        out: &mut Vec<(K, V)>,
    ) -> Result<()>;
    fn seek<'s>(&self, snap: &'s Snapshot<'s>) -> Box<dyn Seeker<K, V> + 's>;
    fn subscribe(&self, store: &Store, after: u64) -> Subscription<(K, Change<V>)>;
}

/// A forward-only probe of a keyspace.
pub(crate) trait Seeker<K, V> {
    fn get(&mut self, key: &K) -> Result<Option<V>>;
}

/// Reads keyspace `id`, each value through `pick`: a table or view's
/// own keyspace as it is, or an operator's keyspace whose values hold
/// more than the view shows, such as a sum's count beside its total.
pub(crate) struct RegisterReader<K, M: Merge, V> {
    id: KeyspaceId<K, M>,
    pick: Arc<dyn Fn(&M::Value) -> V + Send + Sync>,
}

impl<K, M: Merge, V> RegisterReader<K, M, V> {
    pub(crate) fn new(
        id: KeyspaceId<K, M>,
        pick: Arc<dyn Fn(&M::Value) -> V + Send + Sync>,
    ) -> Self {
        RegisterReader { id, pick }
    }
}

impl<K, V: Clone + Send + Sync + 'static> RegisterReader<K, Mergeless<V>, V> {
    pub(crate) fn plain(id: KeyspaceId<K, Mergeless<V>>) -> Self {
        RegisterReader::new(id, Arc::new(|v: &V| v.clone()))
    }
}

impl<K: Data> RegisterReader<K, Tally, Delta> {
    pub(crate) fn tally(id: KeyspaceId<K, Tally>) -> Self {
        RegisterReader::new(id, Arc::new(|n: &Delta| *n))
    }
}

struct CursorSeek<'s, K, M: Merge, V> {
    cursor: Cursor<'s, K, M>,
    pick: Arc<dyn Fn(&M::Value) -> V + Send + Sync>,
}

impl<K, M, V> Seeker<K, V> for CursorSeek<'_, K, M, V>
where
    K: Data,
    M: Merge,
    M::Value: Codec,
{
    fn get(&mut self, key: &K) -> Result<Option<V>> {
        Ok(self.cursor.seek_get(key)?.map(|v| (self.pick)(&v)))
    }
}

impl<K, M, V> KeyedReader<K, V> for RegisterReader<K, M, V>
where
    K: Data,
    M: Merge + Send + Sync,
    M::Value: Codec + Send + Sync,
    M::Diff: Send + Sync,
    V: Data,
{
    fn get(&self, snap: &Snapshot<'_>, key: &K) -> Result<Option<V>> {
        Ok(snap.get(self.id, key)?.map(|v| (self.pick)(&v)))
    }

    fn get_tx(&self, tx: &mut WriteTx<'_>, key: &K) -> Result<Option<V>> {
        Ok(tx.keyspace(self.id).get(key)?.map(|v| (self.pick)(&v)))
    }

    fn scan<'s>(
        &self,
        snap: &'s Snapshot<'s>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
    ) -> Box<dyn Iterator<Item = Result<(K, V)>> + 's> {
        let pick = self.pick.clone();
        let rows = if rev {
            snap.range_rev(self.id, (lo, hi))
        } else {
            snap.range(self.id, (lo, hi))
        };
        Box::new(rows.map(move |r| r.map(|(k, v)| (k.into_owned(), pick(&v)))))
    }

    fn scan_tx(
        &self,
        tx: &mut WriteTx<'_>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
        out: &mut Vec<(K, V)>,
    ) -> Result<()> {
        let mut ks = tx.keyspace(self.id);
        let rows = if rev {
            ks.range_rev((lo, hi))
        } else {
            ks.range((lo, hi))
        };
        for r in rows {
            let (k, v) = r?;
            out.push((k.into_owned(), (self.pick)(&v)));
        }
        Ok(())
    }

    fn seek<'s>(&self, snap: &'s Snapshot<'s>) -> Box<dyn Seeker<K, V> + 's> {
        Box::new(CursorSeek {
            cursor: snap.range(self.id, ..),
            pick: self.pick.clone(),
        })
    }

    fn subscribe(&self, store: &Store, after: u64) -> Subscription<(K, Change<V>)> {
        Subscription::new(subscribe::Registered {
            sub: store.subscribe_from(self.id, after),
            pick: self.pick.clone(),
        })
    }
}

/// Reads a register through another, each value mapped by `f` and
/// dropped when `f` says so: a filtered or mapped view of a register
/// that stores nothing.
pub(crate) struct FilterMappedReader<K, V, W> {
    pub parent: Arc<dyn KeyedReader<K, V>>,
    pub f: crate::stream::dynstate::ValueMap<K, V, W>,
}

struct FilterMappedSeek<'s, K, V, W> {
    inner: Box<dyn Seeker<K, V> + 's>,
    f: crate::stream::dynstate::ValueMap<K, V, W>,
}

impl<K, V, W> Seeker<K, W> for FilterMappedSeek<'_, K, V, W> {
    fn get(&mut self, key: &K) -> Result<Option<W>> {
        Ok(self.inner.get(key)?.and_then(|v| (self.f)(key, &v)))
    }
}

impl<K: Data, V: Data, W: Data> KeyedReader<K, W> for FilterMappedReader<K, V, W> {
    fn get(&self, snap: &Snapshot<'_>, key: &K) -> Result<Option<W>> {
        Ok(self.parent.get(snap, key)?.and_then(|v| (self.f)(key, &v)))
    }

    fn get_tx(&self, tx: &mut WriteTx<'_>, key: &K) -> Result<Option<W>> {
        Ok(self.parent.get_tx(tx, key)?.and_then(|v| (self.f)(key, &v)))
    }

    fn scan<'s>(
        &self,
        snap: &'s Snapshot<'s>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
    ) -> Box<dyn Iterator<Item = Result<(K, W)>> + 's> {
        let f = self.f.clone();
        Box::new(
            self.parent
                .scan(snap, lo, hi, rev)
                .filter_map(move |r| match r {
                    Ok((k, v)) => f(&k, &v).map(|w| Ok((k, w))),
                    Err(e) => Some(Err(e)),
                }),
        )
    }

    fn scan_tx(
        &self,
        tx: &mut WriteTx<'_>,
        lo: Bound<K>,
        hi: Bound<K>,
        rev: bool,
        out: &mut Vec<(K, W)>,
    ) -> Result<()> {
        let mut rows = Vec::new();
        self.parent.scan_tx(tx, lo, hi, rev, &mut rows)?;
        out.extend(
            rows.into_iter()
                .filter_map(|(k, v)| (self.f)(&k, &v).map(|w| (k, w))),
        );
        Ok(())
    }

    fn seek<'s>(&self, snap: &'s Snapshot<'s>) -> Box<dyn Seeker<K, W> + 's> {
        Box::new(FilterMappedSeek {
            inner: self.parent.seek(snap),
            f: self.f.clone(),
        })
    }

    fn subscribe(&self, store: &Store, after: u64) -> Subscription<(K, Change<W>)> {
        let f = self.f.clone();
        self.parent
            .subscribe(store, after)
            .filter_map(move |(k, c)| {
                let change = match c {
                    Change::Insert(v) => Change::Insert(f(&k, &v)?),
                    Change::Delete(o) => Change::Delete(f(&k, &o)?),
                    Change::Replace(o, n) => match (f(&k, &o), f(&k, &n)) {
                        (Some(o), Some(n)) if o == n => return None,
                        (Some(o), Some(n)) => Change::Replace(o, n),
                        (Some(o), None) => Change::Delete(o),
                        (None, Some(n)) => Change::Insert(n),
                        (None, None) => return None,
                    },
                };
                Some((k, change))
            })
    }
}

impl<'a> At<'a> {
    /// Read committed state with `snap`, or the transaction's with `tx`,
    /// after running the graph over its writes.
    pub(crate) fn with<R>(
        self,
        snap: impl FnOnce(&Snapshot<'_>) -> Result<R>,
        tx: impl FnOnce(&mut WriteTx<'_>) -> Result<R>,
    ) -> Result<R> {
        match self {
            At::Snapshot(s) => snap(s),
            At::Before(cell) => snap(cell.borrow().snapshot()),
            At::During(cell) => {
                let mut t = cell.borrow_mut();
                t.run()?;
                tx(t.raw())
            }
        }
    }

    pub(crate) fn store_and_epoch(self) -> (&'a Store, u64) {
        match self {
            At::Snapshot(s) => (s.store(), s.epoch()),
            At::Before(cell) | At::During(cell) => {
                let t = cell.borrow();
                let s = t.snapshot();
                (s.store(), s.epoch())
            }
        }
    }
}

/// A boxed scan, lazy over a snapshot and collected otherwise.
type Scan<'a, T> = Box<dyn Iterator<Item = Result<T>> + 'a>;

fn collected<'a, T: 'a>(rows: Result<Vec<T>>) -> Scan<'a, T> {
    match rows {
        Ok(rows) => Box::new(rows.into_iter().map(Ok)),
        Err(e) => Box::new(std::iter::once(Err(e))),
    }
}

fn clone_bounds<K: Clone>(bounds: impl RangeBounds<K>) -> (Bound<K>, Bound<K>) {
    (bounds.start_bound().cloned(), bounds.end_bound().cloned())
}

/// The read facade of a register-shaped keyspace: one value per key.
/// Tables, event logs, map views and ordered views read through it.
pub struct Read<'a, K, V> {
    at: At<'a>,
    reader: &'a dyn KeyedReader<K, V>,
}

impl<'a, K, V> Clone for Read<'a, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, K, V> Copy for Read<'a, K, V> {}

impl<'a, K: Data, V: Data> Read<'a, K, V> {
    pub(crate) fn new(reader: &'a dyn KeyedReader<K, V>, at: At<'a>) -> Self {
        Read { at, reader }
    }

    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let reader = self.reader;
        self.at
            .with(|s| reader.get(s, key), |tx| reader.get_tx(tx, key))
    }

    pub fn contains(&self, key: &K) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    fn scan(&self, lo: Bound<K>, hi: Bound<K>, rev: bool) -> Scan<'a, (K, V)> {
        let reader = self.reader;
        match self.at {
            At::Snapshot(s) => reader.scan(s, lo, hi, rev),
            at => {
                let (lo2, hi2) = (lo.clone(), hi.clone());
                collected(at.with(
                    |s| reader.scan(s, lo2, hi2, rev).collect(),
                    |tx| {
                        let mut out = Vec::new();
                        reader.scan_tx(tx, lo, hi, rev, &mut out)?;
                        Ok(out)
                    },
                ))
            }
        }
    }

    /// Rows in key order.
    pub fn iter(&self) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        self.scan(Bound::Unbounded, Bound::Unbounded, false)
    }

    /// Rows in reverse key order.
    pub fn iter_rev(&self) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        self.scan(Bound::Unbounded, Bound::Unbounded, true)
    }

    /// Rows whose keys fall in `bounds`, in key order.
    pub fn range(&self, bounds: impl RangeBounds<K>) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan(lo, hi, false)
    }

    /// Rows whose keys fall in `bounds`, greatest key first.
    pub fn range_rev(
        &self,
        bounds: impl RangeBounds<K>,
    ) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan(lo, hi, true)
    }

    /// Rows with keys at or after `from`, in key order.
    pub fn since(&self, from: K) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        self.range(from..)
    }

    /// A forward-only reader: probe keys in ascending order and each
    /// probe moves on from the last, so a sorted batch of lookups costs
    /// one pass. Over a snapshot; inside a transaction it looks each key
    /// up.
    pub fn seek(&self) -> Seek<'a, K, V> {
        Seek(match self.at {
            At::Snapshot(s) => SeekBy::Cursor(self.reader.seek(s)),
            _ => SeekBy::Each(*self),
        })
    }

    /// The rows present for `keys`, which must ascend: one pass.
    pub fn get_sorted<'k>(
        &self,
        keys: impl IntoIterator<Item = K> + 'k,
    ) -> impl Iterator<Item = Result<(K, V)>> + 'k
    where
        'a: 'k,
    {
        let mut seek = self.seek();
        keys.into_iter().filter_map(move |k| match seek.get(&k) {
            Ok(Some(v)) => Some(Ok((k, v))),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
    }

    /// Every commit after the state this facade reads, as its changes
    /// to this keyspace. Subscribe close to taking the snapshot: a
    /// commit between the two is seen only if not yet published.
    pub fn subscribe(&self) -> Subscription<(K, Change<V>)> {
        let (store, epoch) = self.at.store_and_epoch();
        self.reader.subscribe(store, epoch)
    }

    /// Row count: a full scan. Keep a `count()` view for a live count.
    pub fn len(&self) -> Result<usize> {
        let mut n = 0;
        for row in self.iter() {
            row?;
            n += 1;
        }
        Ok(n)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.iter().next().transpose()?.is_none())
    }
}

/// See [`Read::seek`].
pub struct Seek<'a, K, V>(SeekBy<'a, K, V>);

enum SeekBy<'a, K, V> {
    Cursor(Box<dyn Seeker<K, V> + 'a>),
    Each(Read<'a, K, V>),
}

impl<K: Data, V: Data> Seek<'_, K, V> {
    /// The key's value. Keys must not descend.
    pub fn get(&mut self, key: &K) -> Result<Option<V>> {
        match &mut self.0 {
            SeekBy::Cursor(c) => c.get(key),
            SeekBy::Each(r) => r.get(key),
        }
    }

    pub fn contains(&mut self, key: &K) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }
}

/// Bounds on an arrangement's stored rows.
type RowBounds<K, V> = (Bound<(K, Edge<V>)>, Bound<(K, Edge<V>)>);

/// Reads an arrangement, or a [`DistinctRows`] node's counts, as groups.
pub(crate) struct GroupReader<K, V> {
    source: GroupSource<K, V>,
}

enum GroupSource<K, V> {
    Arranged {
        id: Arrangement<K, V>,
        totals: Option<KeyspaceId<K, Tally>>,
    },
    Distinct {
        id: DistinctCounts<K, V>,
    },
}

fn row<K: Clone, V: Clone>(k: Cow<'_, (K, Edge<V>)>, d: Delta) -> (K, V, Delta) {
    let (k, v) = k.into_owned();
    let Edge::Val(v) = v else {
        unreachable!("arrangements hold values")
    };
    (k, v, d)
}

/// A distinct count as a row's multiplicity: once if positive.
fn present(count: Delta) -> Option<Delta> {
    (count > 0).then_some(1)
}

impl<K: Data, V: Data> GroupReader<K, V> {
    pub(crate) fn arranged(id: Arrangement<K, V>, totals: Option<KeyspaceId<K, Tally>>) -> Self {
        GroupReader {
            source: GroupSource::Arranged { id, totals },
        }
    }

    pub(crate) fn distinct(id: DistinctCounts<K, V>) -> Self {
        GroupReader {
            source: GroupSource::Distinct { id },
        }
    }

    /// Key bounds as row bounds: a key's rows lie between its `Lo` and
    /// `Hi` edges, so an included key takes all of them and an excluded
    /// one none.
    fn bounds(lo: Bound<K>, hi: Bound<K>) -> RowBounds<K, V> {
        let lo = match lo {
            Bound::Included(k) => Bound::Included((k, Edge::Lo)),
            Bound::Excluded(k) => Bound::Excluded((k, Edge::Hi)),
            Bound::Unbounded => Bound::Unbounded,
        };
        let hi = match hi {
            Bound::Included(k) => Bound::Included((k, Edge::Hi)),
            Bound::Excluded(k) => Bound::Excluded((k, Edge::Lo)),
            Bound::Unbounded => Bound::Unbounded,
        };
        (lo, hi)
    }

    /// Bounds on `(key, value)` rows as stored row bounds.
    fn row_bounds(lo: Bound<(K, V)>, hi: Bound<(K, V)>) -> RowBounds<K, V> {
        let edge = |b: Bound<(K, V)>| match b {
            Bound::Included((k, v)) => Bound::Included((k, Edge::Val(v))),
            Bound::Excluded((k, v)) => Bound::Excluded((k, Edge::Val(v))),
            Bound::Unbounded => Bound::Unbounded,
        };
        (edge(lo), edge(hi))
    }

    fn scan<'s>(
        &self,
        snap: &'s Snapshot<'s>,
        (lo, hi): RowBounds<K, V>,
        rev: bool,
    ) -> Box<dyn Iterator<Item = Result<(K, V, Delta)>> + 's> {
        match self.source {
            GroupSource::Arranged { id, .. } => {
                let rows = if rev {
                    snap.range_rev(id, (lo, hi))
                } else {
                    snap.range(id, (lo, hi))
                };
                Box::new(rows.map(|r| r.map(|(k, d)| row(k, *d))))
            }
            GroupSource::Distinct { id } => {
                let rows = if rev {
                    snap.range_rev(id, (lo, hi))
                } else {
                    snap.range(id, (lo, hi))
                };
                Box::new(rows.filter_map(|r| match r {
                    Ok((k, c)) => present(*c).map(|d| Ok(row(k, d))),
                    Err(e) => Some(Err(e)),
                }))
            }
        }
    }

    fn scan_tx(
        &self,
        tx: &mut WriteTx<'_>,
        (lo, hi): RowBounds<K, V>,
        rev: bool,
        out: &mut Vec<(K, V, Delta)>,
    ) -> Result<()> {
        match self.source {
            GroupSource::Arranged { id, .. } => {
                let mut ks = tx.keyspace(id);
                let rows = if rev {
                    ks.range_rev((lo, hi))
                } else {
                    ks.range((lo, hi))
                };
                for r in rows {
                    let (k, d) = r?;
                    out.push(row(k, *d));
                }
            }
            GroupSource::Distinct { id } => {
                let mut ks = tx.keyspace(id);
                let rows = if rev {
                    ks.range_rev((lo, hi))
                } else {
                    ks.range((lo, hi))
                };
                for r in rows {
                    let (k, c) = r?;
                    if let Some(d) = present(*c) {
                        out.push(row(k, d));
                    }
                }
            }
        }
        Ok(())
    }

    fn contains(&self, snap: &Snapshot<'_>, key: &K) -> Result<bool> {
        if let GroupSource::Arranged {
            totals: Some(totals),
            ..
        } = self.source
        {
            return Ok(snap.get(totals, key)?.is_some_and(|t| *t > 0));
        }
        let k = || Bound::Included(key.clone());
        Ok(self
            .scan(snap, Self::bounds(k(), k()), false)
            .next()
            .transpose()?
            .is_some())
    }

    fn contains_tx(&self, tx: &mut WriteTx<'_>, key: &K) -> Result<bool> {
        if let GroupSource::Arranged {
            totals: Some(totals),
            ..
        } = self.source
        {
            return Ok(tx.keyspace(totals).get(key)?.is_some_and(|t| *t > 0));
        }
        let mut out = Vec::new();
        let k = || Bound::Included(key.clone());
        self.scan_tx(tx, Self::bounds(k(), k()), false, &mut out)?;
        Ok(!out.is_empty())
    }

    fn subscribe(&self, store: &Store, after: u64) -> Subscription<(K, V, Delta)> {
        match self.source {
            GroupSource::Arranged { id, .. } => Subscription::new(subscribe::Arranged {
                sub: store.subscribe_from(id, after),
            }),
            GroupSource::Distinct { id } => Subscription::new(subscribe::Distinct {
                sub: store.subscribe_from(id, after),
            }),
        }
    }
}

/// The read facade of a grouped view: any number of values per key,
/// each with its multiplicity.
pub struct GroupRead<'a, K, V> {
    at: At<'a>,
    reader: &'a GroupReader<K, V>,
}

impl<'a, K, V> Clone for GroupRead<'a, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, K, V> Copy for GroupRead<'a, K, V> {}

impl<'a, K: Data, V: Data> GroupRead<'a, K, V> {
    pub(crate) fn new(reader: &'a GroupReader<K, V>, at: At<'a>) -> Self {
        GroupRead { at, reader }
    }

    fn scan(&self, lo: Bound<K>, hi: Bound<K>, rev: bool) -> Scan<'a, (K, V, Delta)> {
        self.scan_rows(GroupReader::bounds(lo, hi), rev)
    }

    fn scan_rows(&self, bounds: RowBounds<K, V>, rev: bool) -> Scan<'a, (K, V, Delta)> {
        let reader = self.reader;
        match self.at {
            At::Snapshot(s) => reader.scan(s, bounds, rev),
            at => {
                let bounds2 = bounds.clone();
                collected(at.with(
                    |s| reader.scan(s, bounds2, rev).collect(),
                    |tx| {
                        let mut out = Vec::new();
                        reader.scan_tx(tx, bounds, rev, &mut out)?;
                        Ok(out)
                    },
                ))
            }
        }
    }

    /// `key`'s rows.
    fn group(&self, key: &K, rev: bool) -> impl Iterator<Item = Result<(V, Delta)>> + 'a {
        let k = || Bound::Included(key.clone());
        self.scan(k(), k(), rev).map(|r| r.map(|(_, v, d)| (v, d)))
    }

    /// `key`'s values, ascending, each with its multiplicity.
    pub fn values(&self, key: &K) -> impl Iterator<Item = Result<(V, Delta)>> + 'a {
        self.group(key, false)
    }

    /// `key`'s values, descending.
    pub fn values_rev(&self, key: &K) -> impl Iterator<Item = Result<(V, Delta)>> + 'a {
        self.group(key, true)
    }

    /// The rows of every key in `bounds`, in key then value order: one
    /// scan, however many groups it spans.
    pub fn range(
        &self,
        bounds: impl RangeBounds<K>,
    ) -> impl Iterator<Item = Result<(K, V, Delta)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan(lo, hi, false)
    }

    /// The rows of every key in `bounds`, greatest key (and value) first.
    pub fn range_rev(
        &self,
        bounds: impl RangeBounds<K>,
    ) -> impl Iterator<Item = Result<(K, V, Delta)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan(lo, hi, true)
    }

    /// The rows in `bounds` of `(key, value)`, in key then value order:
    /// where a listing resumes from a row, mid-group.
    pub fn rows(
        &self,
        bounds: impl RangeBounds<(K, V)>,
    ) -> impl Iterator<Item = Result<(K, V, Delta)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan_rows(GroupReader::row_bounds(lo, hi), false)
    }

    /// [`rows`](Self::rows), greatest key (and value) first.
    pub fn rows_rev(
        &self,
        bounds: impl RangeBounds<(K, V)>,
    ) -> impl Iterator<Item = Result<(K, V, Delta)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.scan_rows(GroupReader::row_bounds(lo, hi), true)
    }

    /// `key`'s `n` greatest values, greatest first.
    pub fn top(&self, key: &K, n: usize) -> Result<Vec<(V, Delta)>> {
        self.values_rev(key).take(n).collect()
    }

    /// `key`'s whole group.
    pub fn get(&self, key: &K) -> Result<Vec<(V, Delta)>> {
        self.values(key).collect()
    }

    pub fn contains(&self, key: &K) -> Result<bool> {
        let reader = self.reader;
        self.at.with(
            |s| reader.contains(s, key),
            |tx| reader.contains_tx(tx, key),
        )
    }

    /// Every row, in key then value order.
    pub fn iter(&self) -> impl Iterator<Item = Result<(K, V, Delta)>> + 'a {
        self.scan(Bound::Unbounded, Bound::Unbounded, false)
    }

    /// Every commit after the state this facade reads, as each row's
    /// change in multiplicity. See [`Read::subscribe`].
    pub fn subscribe(&self) -> Subscription<(K, V, Delta)> {
        let (store, epoch) = self.at.store_and_epoch();
        self.reader.subscribe(store, epoch)
    }
}

/// Reads the value order of an ordered view: `(v, k)` pairs.
pub(crate) struct OrderReader<K, V> {
    pub(crate) id: KeyspaceId<(V, Edge<K>), ()>,
}

impl<K: Data, V: Data> OrderReader<K, V> {
    fn bounds(lo: Bound<V>, hi: Bound<V>) -> (Bound<(V, Edge<K>)>, Bound<(V, Edge<K>)>) {
        let lo = match lo {
            Bound::Included(v) => Bound::Included((v, Edge::Lo)),
            Bound::Excluded(v) => Bound::Excluded((v, Edge::Hi)),
            Bound::Unbounded => Bound::Unbounded,
        };
        let hi = match hi {
            Bound::Included(v) => Bound::Included((v, Edge::Hi)),
            Bound::Excluded(v) => Bound::Excluded((v, Edge::Lo)),
            Bound::Unbounded => Bound::Unbounded,
        };
        (lo, hi)
    }

    fn pair((v, k): (V, Edge<K>)) -> (K, V) {
        let Edge::Val(k) = k else {
            unreachable!("the order index holds keys")
        };
        (k, v)
    }

    fn scan<'s>(
        &self,
        snap: &'s Snapshot<'s>,
        lo: Bound<V>,
        hi: Bound<V>,
        rev: bool,
    ) -> Box<dyn Iterator<Item = Result<(K, V)>> + 's> {
        let (lo, hi) = Self::bounds(lo, hi);
        let rows = if rev {
            snap.range_rev(self.id, (lo, hi))
        } else {
            snap.range(self.id, (lo, hi))
        };
        Box::new(rows.map(|r| r.map(|(k, _)| Self::pair(k.into_owned()))))
    }

    fn scan_tx(
        &self,
        tx: &mut WriteTx<'_>,
        lo: Bound<V>,
        hi: Bound<V>,
        rev: bool,
        out: &mut Vec<(K, V)>,
    ) -> Result<()> {
        let (lo, hi) = Self::bounds(lo, hi);
        let mut ks = tx.keyspace(self.id);
        let rows = if rev {
            ks.range_rev((lo, hi))
        } else {
            ks.range((lo, hi))
        };
        for r in rows {
            out.push(Self::pair(r?.0.into_owned()));
        }
        Ok(())
    }
}

/// The read facade of an ordered view: the register (every method of
/// [`Read`], by deref) plus the rows in value order.
pub struct OrderedRead<'a, K, V> {
    read: Read<'a, K, V>,
    order: &'a OrderReader<K, V>,
}

impl<'a, K, V> std::ops::Deref for OrderedRead<'a, K, V> {
    type Target = Read<'a, K, V>;
    fn deref(&self) -> &Read<'a, K, V> {
        &self.read
    }
}

impl<'a, K: Data, V: Data> OrderedRead<'a, K, V> {
    pub(crate) fn new(read: Read<'a, K, V>, order: &'a OrderReader<K, V>) -> Self {
        OrderedRead { read, order }
    }

    fn by(&self, lo: Bound<V>, hi: Bound<V>, rev: bool) -> Scan<'a, (K, V)> {
        let order = self.order;
        match self.read.at {
            At::Snapshot(s) => order.scan(s, lo, hi, rev),
            at => {
                let (lo2, hi2) = (lo.clone(), hi.clone());
                collected(at.with(
                    |s| order.scan(s, lo2, hi2, rev).collect(),
                    |tx| {
                        let mut out = Vec::new();
                        order.scan_tx(tx, lo, hi, rev, &mut out)?;
                        Ok(out)
                    },
                ))
            }
        }
    }

    /// Rows by ascending value, then key.
    pub fn asc(&self) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        self.by(Bound::Unbounded, Bound::Unbounded, false)
    }

    /// Rows by descending value.
    pub fn desc(&self) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        self.by(Bound::Unbounded, Bound::Unbounded, true)
    }

    /// The `n` rows with the greatest values, greatest first.
    pub fn top(&self, n: usize) -> Result<Vec<(K, V)>> {
        self.desc().take(n).collect()
    }

    /// The `n` rows with the least values, least first.
    pub fn bottom(&self, n: usize) -> Result<Vec<(K, V)>> {
        self.asc().take(n).collect()
    }

    /// Rows whose value falls in `bounds`, ascending.
    pub fn by_value(
        &self,
        bounds: impl RangeBounds<V>,
    ) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.by(lo, hi, false)
    }

    /// Rows whose value falls in `bounds`, descending.
    pub fn by_value_rev(
        &self,
        bounds: impl RangeBounds<V>,
    ) -> impl Iterator<Item = Result<(K, V)>> + 'a {
        let (lo, hi) = clone_bounds(bounds);
        self.by(lo, hi, true)
    }
}

/// The read facade of a single value: a cell or a value view.
pub struct ValueRead<'a, V> {
    read: Read<'a, (), V>,
}

impl<'a, V: Data> ValueRead<'a, V> {
    pub(crate) fn new(read: Read<'a, (), V>) -> Self {
        ValueRead { read }
    }

    pub fn get(&self) -> Result<Option<V>> {
        self.read.get(&())
    }

    /// See [`Read::subscribe`].
    pub fn subscribe(&self) -> Subscription<Change<V>> {
        self.read.subscribe().map(|((), c)| c)
    }
}

/// The read facade of a set.
pub struct SetRead<'a, T> {
    read: Read<'a, T, ()>,
}

impl<'a, T: Data> SetRead<'a, T> {
    pub(crate) fn new(read: Read<'a, T, ()>) -> Self {
        SetRead { read }
    }

    pub fn contains(&self, value: &T) -> Result<bool> {
        self.read.contains(value)
    }

    /// The members in order.
    pub fn iter(&self) -> impl Iterator<Item = Result<T>> + 'a {
        self.read.iter().map(|r| r.map(|(t, ())| t))
    }

    pub fn iter_rev(&self) -> impl Iterator<Item = Result<T>> + 'a {
        self.read.iter_rev().map(|r| r.map(|(t, ())| t))
    }

    /// Members in `bounds`, in order.
    pub fn range(&self, bounds: impl RangeBounds<T>) -> impl Iterator<Item = Result<T>> + 'a {
        self.read.range(bounds).map(|r| r.map(|(t, ())| t))
    }

    /// Members in `bounds`, greatest first.
    pub fn range_rev(&self, bounds: impl RangeBounds<T>) -> impl Iterator<Item = Result<T>> + 'a {
        self.read.range_rev(bounds).map(|r| r.map(|(t, ())| t))
    }

    /// See [`Read::seek`].
    pub fn seek(&self) -> Seek<'a, T, ()> {
        self.read.seek()
    }

    /// Which of `values`, which must ascend, are members: one pass.
    pub fn contains_sorted<'k>(
        &self,
        values: impl IntoIterator<Item = T> + 'k,
    ) -> impl Iterator<Item = Result<T>> + 'k
    where
        'a: 'k,
    {
        self.read.get_sorted(values).map(|r| r.map(|(t, ())| t))
    }

    /// See [`Read::subscribe`].
    pub fn subscribe(&self) -> Subscription<(T, Change<()>)> {
        self.read.subscribe()
    }

    pub fn len(&self) -> Result<usize> {
        self.read.len()
    }

    pub fn is_empty(&self) -> Result<bool> {
        self.read.is_empty()
    }
}

/// The read facade of a multiset: each row's count.
pub struct MultisetRead<'a, T> {
    read: Read<'a, T, Delta>,
}

impl<'a, T: Data> MultisetRead<'a, T> {
    pub(crate) fn new(read: Read<'a, T, Delta>) -> Self {
        MultisetRead { read }
    }

    /// How many of `row` there are.
    pub fn get(&self, row: &T) -> Result<Delta> {
        Ok(self.read.get(row)?.unwrap_or(0))
    }

    pub fn contains(&self, row: &T) -> Result<bool> {
        Ok(self.get(row)? > 0)
    }

    /// Every row present, with its count, in order.
    pub fn iter(&self) -> impl Iterator<Item = Result<(T, Delta)>> + 'a {
        self.read.iter()
    }

    /// Rows in `bounds`, with their counts, in order.
    pub fn range(
        &self,
        bounds: impl RangeBounds<T>,
    ) -> impl Iterator<Item = Result<(T, Delta)>> + 'a {
        self.read.range(bounds)
    }

    /// Rows in `bounds`, with their counts, greatest first.
    pub fn range_rev(
        &self,
        bounds: impl RangeBounds<T>,
    ) -> impl Iterator<Item = Result<(T, Delta)>> + 'a {
        self.read.range_rev(bounds)
    }

    /// Every commit after the state this facade reads, as each row's
    /// change in count. See [`Read::subscribe`].
    pub fn subscribe(&self) -> Subscription<(T, Delta)> {
        self.read.subscribe().map(|(t, c)| match c {
            Change::Insert(n) => (t, n),
            _ => unreachable!("a multiset's changes are amounts"),
        })
    }
}
