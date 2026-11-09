use std::{
    any::Any,
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use crate::{
    Change, Register,
    codec::{Codec, OrderedCodec, ResidentSize, heap_size},
    cursor::Cursor,
    error::{Error, Result},
    merge::Merge,
    store::{ENTRY_OVERHEAD, Resident, Snapshot, WriterGuard, resident_entry_bytes},
};

/// A write tx outgrew its memory limit and was killed. Its writes were
/// released, and committing it fails with this error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfMemory {
    /// Bytes the tx's writes held when it was killed.
    pub used: usize,
    pub limit: usize,
}

impl fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "write transaction killed: its writes reached {} bytes, over its {} byte limit",
            self.used, self.limit
        )
    }
}

impl std::error::Error for OutOfMemory {}

/// A write tx's health: the estimated resident bytes of its overlay, and
/// the first failure, if any. Behind cells so reads, which take `&self`,
/// can fail the tx.
pub(crate) struct TxState {
    used: Cell<usize>,
    limit: usize,
    failure: RefCell<Option<Error>>,
}

impl TxState {
    /// The tx's failure, if it has one.
    pub(crate) fn check(&self) -> Result<()> {
        match &*self.failure.borrow() {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Fail the tx with `e`, unless it already failed.
    pub(crate) fn fail(&self, e: &Error) {
        let mut failure = self.failure.borrow_mut();
        if failure.is_none() {
            *failure = Some(e.clone());
        }
    }

    /// Run a read, failing the tx if it fails.
    fn read<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.check()?;
        f().inspect_err(|e| self.fail(e))
    }

    /// Apply an entry's growth; returns true when that kills the tx.
    fn grow(&self, delta: isize) -> bool {
        let used = self.used.get().saturating_add_signed(delta);
        if used > self.limit {
            self.fail(&Error::OutOfMemory(OutOfMemory {
                used,
                limit: self.limit,
            }));
            self.used.set(0);
            return true;
        }
        self.used.set(used);
        false
    }
}

/// Estimated resident bytes of one overlay entry.
fn change_bytes<K: ResidentSize, M: Merge>(key: &K, change: &Change<M>) -> usize
where
    M::Value: ResidentSize,
{
    let values = match change {
        Change::Insert(v) | Change::Delete(v) => heap_size(v),
        Change::Replace(old, new) => heap_size(old) + heap_size(new),
        // never stored in an overlay: resolve materializes it first
        Change::Modify(_) => 0,
    };
    key.resident_size() + size_of::<Change<M>>() + values + ENTRY_OVERHEAD
}

/// Typed, copyable name of a keyspace. The key and merge types travel
/// with the id, so opening it as the wrong type is a compile error and
/// two keyspaces cannot share an id with different types.
pub struct KeyspaceId<K, M: Merge>(u64, PhantomData<fn() -> (K, M)>);

impl<K, M: Merge> KeyspaceId<K, M> {
    pub const fn new(raw: u64) -> Self {
        KeyspaceId(raw, PhantomData)
    }
    pub fn raw(&self) -> u64 {
        self.0
    }
}
impl<K, M: Merge> Clone for KeyspaceId<K, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, M: Merge> Copy for KeyspaceId<K, M> {}

pub trait ErasedUnderlay: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    fn absorb_erased(&mut self, other: Box<dyn ErasedUnderlay>);
    fn is_empty(&self) -> bool;
    /// Turn this tx's changes into the committed form, a
    /// [`Resident`] of new values and tombstones, encoding each entry
    /// into `wal` on the way when there is one. Also returns the tree's
    /// estimated resident bytes.
    fn seal(self: Box<Self>, wal: Option<&mut Vec<u8>>) -> (Box<dyn Any + Send + Sync>, usize);
}

/// A write transaction: this tx's materialized changes layered over a
/// snapshot of the store. Reads see the overlay first and fall through to
/// the snapshot, so the tx reads its own writes. Open one with
/// [`Store::begin`](crate::store::Store::begin); the snapshot must be the
/// latest committed state or `Replace(old, new)` would carry a stale `old`.
///
/// The tx holds the store's writer lock until it is finished or dropped.
/// Dropping it without [`Store::finish`](crate::store::Store::finish)
/// aborts: the overlay is discarded and nothing is logged.
///
/// A tx fails on the first read error, or when a write takes its writes
/// over the memory limit, which also releases every write so far. A failed
/// tx stays failed: every later read, write and the commit return the
/// error that failed it.
pub struct WriteTx<'tx> {
    /// The epoch this tx will commit as: one past the snapshot's.
    epoch: u64,
    overlay: HashMap<u64, Box<dyn ErasedUnderlay>>,
    state: TxState,
    snapshot: Snapshot<'tx>,
    writer: WriterGuard<'tx>,
    /// Part of a rebuild of derived state at open.
    rebuilding: bool,
}

impl<'tx> WriteTx<'tx> {
    pub(crate) fn new(
        snapshot: Snapshot<'tx>,
        writer: WriterGuard<'tx>,
        memory_limit: usize,
    ) -> Self {
        WriteTx {
            epoch: snapshot.epoch() + 1,
            overlay: HashMap::new(),
            state: TxState {
                used: Cell::new(0),
                limit: memory_limit,
                failure: RefCell::new(None),
            },
            snapshot,
            writer,
            rebuilding: false,
        }
    }

    /// Whether this tx is one chunk of a rebuild of derived state, replaying
    /// base rows already accepted when they were first written. Checks
    /// that only make sense on the final state, or in the order rows were
    /// first written, should be skipped in one.
    pub fn rebuilding(&self) -> bool {
        self.rebuilding
    }

    pub(crate) fn set_rebuilding(&mut self) {
        self.rebuilding = true;
    }

    /// Estimated bytes held by this tx's writes.
    pub fn memory_used(&self) -> usize {
        self.state.used.get()
    }

    /// Why this tx failed, if it did.
    pub fn failure(&self) -> Option<Error> {
        self.state.check().err()
    }

    /// Reject this tx with an application error. The tx fails as it would
    /// on a read error: the graph stops, and finishing it returns
    /// [`Error::Rejected`] carrying `err`. A tx that already failed keeps its
    /// first error.
    pub fn reject(&self, err: impl std::error::Error + Send + Sync + 'static) {
        self.state.fail(&Error::Rejected(std::sync::Arc::new(err)));
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        u64,
        Result<HashMap<u64, Box<dyn ErasedUnderlay>>>,
        Snapshot<'tx>,
        WriterGuard<'tx>,
    ) {
        let WriteTx {
            epoch,
            overlay,
            state,
            snapshot,
            writer,
            rebuilding: _,
        } = self;
        let overlay = state.check().map(|()| overlay);
        (epoch, overlay, snapshot, writer)
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// This transaction's materialized changes to one keyspace, as stored in
/// the overlay. Every entry is an Insert/Delete/Replace; `resolve` on the
/// [`Keyspace`] view turns incoming Modifies into those.
pub struct KeyspaceHandle<K, M: Merge> {
    inner: Register<K, M>,
}

/// A transaction's view of one keyspace: its own writes plus the
/// snapshot underneath. Borrowed from the tx for a few calls at a time.
pub struct Keyspace<'a, 'tx, K, M: Merge> {
    id: KeyspaceId<K, M>,
    writes: &'a mut Register<K, M>,
    state: &'a TxState,
    snap: &'a Snapshot<'tx>,
}

impl<K: Ord, M: Merge> Default for KeyspaceHandle<K, M> {
    fn default() -> Self {
        Self {
            inner: Register::new(),
        }
    }
}

impl<K, M> ErasedUnderlay for KeyspaceHandle<K, M>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn absorb_erased(&mut self, other: Box<dyn ErasedUnderlay>) {
        let other = other
            .into_any()
            .downcast::<KeyspaceHandle<K, M>>()
            .expect("absorb: keyspace id collision — two KS types share an id");
        for (k, c) in other.inner {
            self.inner.absorb_one(k, c);
        }
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn seal(self: Box<Self>, mut wal: Option<&mut Vec<u8>>) -> (Box<dyn Any + Send + Sync>, usize) {
        let mut bytes = 0;
        // the register is sorted, so collect bulk-builds the tree
        let tree: Resident<K, M::Value> = self
            .inner
            .into_iter()
            .map(|(k, change)| {
                let v = match change {
                    Change::Insert(v) | Change::Replace(_, v) => Some(v),
                    // a surviving Delete means the key exists below this tx
                    Change::Delete(_) => None,
                    Change::Modify(_) => unreachable!("resolve before absorbing"),
                };
                if let Some(out) = wal.as_deref_mut() {
                    k.encode_ordered(out);
                    v.encode(out);
                }
                bytes += resident_entry_bytes(&k, &v);
                (k, v)
            })
            .collect();
        (Box::new(tree), bytes)
    }
}

impl<K: Ord + 'static, M: Merge> KeyspaceHandle<K, M> {
    #[cfg(test)]
    pub(crate) fn inner_is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<'a, 'tx, K, M> Keyspace<'a, 'tx, K, M>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Sync,
{
    #[cfg(test)]
    pub(crate) fn inner_is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    /// Record a materialized change. Fails if the tx has failed, including
    /// by this write taking it over its memory limit.
    pub fn absorb(&mut self, key: K, change: Change<M>) -> Result<()> {
        debug_assert!(
            !matches!(change, Change::Modify(_)),
            "resolve before absorbing"
        );
        self.state.check()?;
        let delta = self
            .writes
            .absorb_one_measured(key, change, change_bytes::<K, M>);
        if self.state.grow(delta) {
            // release this keyspace's writes now; the tx drops the rest
            // the next time it opens a keyspace, or when it ends
            *self.writes = Register::new();
            return self.state.check();
        }
        Ok(())
    }

    /// Current value of `key` as seen by this transaction: the overlay
    /// wins (a Delete there shadows the snapshot), otherwise the snapshot.
    /// Overlay hits borrow; snapshot hits are owned.
    pub fn get(&self, key: &K) -> Result<Option<Cow<'_, M::Value>>> {
        self.state.read(|| match self.writes.get(key) {
            Some(c) => Ok(c.new_value().map(Cow::Borrowed)),
            None => self.snap.get(self.id, key),
        })
    }

    /// Materialize `change` against the current value of `key`. Returns the
    /// Insert/Delete/Replace to record, or None if nothing changes.
    pub fn resolve(&self, key: &K, change: Change<M>) -> Result<Option<Change<M>>>
    where
        M::Value: PartialEq,
    {
        Ok(materialize(self.get(key)?.as_deref(), change))
    }

    /// [`resolve`](Self::resolve) for a whole batch. `changes` is sorted, so
    /// one cursor walks the overlay, the layers and the tables forward
    /// together, and each table block is read at most once. Resolved
    /// changes land in `out` in key order.
    pub fn resolve_sorted(
        &self,
        changes: Register<K, M>,
        out: &mut Vec<(K, Change<M>)>,
    ) -> Result<()>
    where
        K: Clone,
        M::Value: PartialEq,
    {
        let mut cursor = self.cursor(Bound::Unbounded, Bound::Unbounded);
        for (key, change) in changes {
            let old = cursor.seek_get(&key)?;
            if let Some(resolved) = materialize(old.as_deref(), change) {
                out.push((key, resolved));
            }
        }
        Ok(())
    }

    /// The live entries with keys in `range`, in order, as this tx sees
    /// them: its own writes over the snapshot.
    pub fn range(&self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor(range.start_bound(), range.end_bound().cloned())
    }

    fn cursor(&self, lower: Bound<&K>, upper: Bound<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.snap.cursor(
            self.id,
            Some(self.writes.map()),
            lower,
            upper,
            Some(self.state),
        )
    }
}

/// Compose `change` onto `old`, the key's current value, yielding the
/// fully materialized change or None when the value does not move.
fn materialize<M: Merge>(old: Option<&M::Value>, change: Change<M>) -> Option<Change<M>>
where
    M::Value: PartialEq,
{
    use Change::*;
    let new = match change {
        Modify(d) => {
            let mut v = old.cloned().unwrap_or_else(M::base);
            M::merge(&mut v, std::iter::once(Cow::Owned(d)));
            Some(v)
        }
        Insert(v) | Replace(_, v) => Some(v),
        Delete(_) => None,
    };
    let new = new.filter(|v| !M::consider_delete(v));
    match (old, new) {
        (None, None) => None,
        (None, Some(n)) => Some(Insert(n)),
        (Some(o), None) => Some(Delete(o.clone())),
        (Some(o), Some(n)) if *o == n => None,
        (Some(o), Some(n)) => Some(Replace(o.clone(), n)),
    }
}

impl<'tx> WriteTx<'tx> {
    /// This transaction's view of `id`, opened on first use. The overlay
    /// and the snapshot are separate fields, so the view can hold the
    /// overlay mutably and the snapshot immutably at once.
    pub fn keyspace<K, M>(&mut self, id: KeyspaceId<K, M>) -> Keyspace<'_, 'tx, K, M>
    where
        K: OrderedCodec + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        if self.state.check().is_err() {
            // a failed tx will not commit, so its writes are dead weight
            self.overlay.clear();
        }
        let store = self.snapshot.store();
        let handle: &mut KeyspaceHandle<K, M> = self
            .overlay
            .entry(id.raw())
            .or_insert_with(|| {
                store.assert_declared::<K, M>(id.raw());
                Box::new(KeyspaceHandle::<K, M>::default())
            })
            .as_any_mut()
            .downcast_mut()
            .expect("keyspace id reused with a different type");
        Keyspace {
            id,
            writes: &mut handle.inner,
            state: &self.state,
            snap: &self.snapshot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{DurabilityPolicy, StoreBuilder};

    const KS: KeyspaceId<u8, i64> = KeyspaceId::new(1);

    #[test]
    fn reads_fall_through_to_snapshot_and_overlay_shadows() {
        let store = StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(KS)
            .build()
            .unwrap();
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Insert(10)).unwrap();
        ks.absorb(2, Change::Insert(20)).unwrap();
        store.finish(tx).unwrap();
        let mut tx = store.begin().unwrap();
        tx.keyspace(KS).absorb(2, Change::Delete(20)).unwrap();
        store.finish(tx).unwrap();

        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&10));
        assert_eq!(ks.get(&2).unwrap().as_deref(), None, "committed tombstone");
        assert_eq!(ks.get(&3).unwrap().as_deref(), None, "never written");

        // a Modify resolves against the snapshot value
        let r = ks.resolve(&1, Change::Modify(5)).unwrap().unwrap();
        assert!(matches!(r, Change::Replace(10, 15)));
        ks.absorb(1, r).unwrap();
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&15), "overlay wins");

        // a Delete in the overlay shadows the snapshot
        let r = ks.resolve(&1, Change::Delete(0)).unwrap().unwrap();
        assert!(matches!(r, Change::Delete(15)));
        ks.absorb(1, r).unwrap();
        assert_eq!(ks.get(&1).unwrap().as_deref(), None);

        // resolve_sorted mixes overlay hits and snapshot misses
        let mut changes = Register::new();
        changes.absorb_one(1, Change::Modify(1));
        changes.absorb_one(3, Change::Modify(7));
        let mut out = Vec::new();
        ks.resolve_sorted(changes, &mut out).unwrap();
        // key 1 is deleted in the overlay, so the snapshot's 10 is invisible
        assert!(matches!(out[0], (1, Change::Insert(1))));
        assert!(matches!(out[1], (3, Change::Insert(7))));
    }
}
