use std::{
    any::Any,
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap, btree_map::Entry},
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

/// Estimated bytes a run's memory of a key's prior value holds.
fn prior_bytes<M: Merge>(prior: &Prior<M::Value>) -> usize
where
    M::Value: ResidentSize,
{
    match prior {
        Prior::Was(Some(v)) => heap_size(v),
        _ => 0,
    }
}

/// Estimated bytes a change's values hold beyond the change itself.
fn value_bytes<M: Merge>(change: &Change<M>) -> usize
where
    M::Value: ResidentSize,
{
    match change {
        Change::Insert(v) | Change::Delete(v) => heap_size(v),
        Change::Replace(old, new) => heap_size(old) + heap_size(new),
        // never stored in an overlay: resolve materializes it first
        Change::Modify(_) => 0,
    }
}

/// Estimated resident bytes of one overlay entry, besides its values.
fn slot_bytes<K: ResidentSize, M: Merge>(key: &K) -> usize {
    key.resident_size() + size_of::<Slot<M>>() + ENTRY_OVERHEAD
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
    /// Forget what this run's writes remember of before it, returning
    /// the bytes that frees.
    fn end_run(&mut self) -> usize;
    /// Turn this tx's changes into the committed form, a
    /// `Resident` of new values and tombstones, encoding each entry
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
    /// Which graph run this is; see [`WriteTx::end_run`].
    run: u64,
}

/// One key's entry in a tx's overlay.
pub(crate) struct Slot<M: Merge> {
    /// The change since the tx began: `None` once writes cancelled it
    /// out, kept so the run that did so still remembers what it was.
    pub change: Option<Change<M>>,
    /// The graph run that last wrote the key.
    pub run: u64,
    /// The key's value when that run began, while the run lasts.
    pub prior: Prior<M::Value>,
}

/// What a key held when the run that last wrote it began.
pub(crate) enum Prior<V> {
    /// Not written this run, or the run has ended: `change` is current.
    Stale,
    /// No overlay entry: whatever the snapshot holds.
    Absent,
    /// The overlay's value then, `None` for a tombstone.
    Was(Option<V>),
}

/// A tx's changes to one keyspace, by key.
pub(crate) type Overlay<K, M> = BTreeMap<K, Slot<M>>;

impl<M: Merge> Slot<M> {
    /// The value the key has now, if the overlay decides it: `None` for
    /// a tombstone, and `Err(())` when cancelled writes leave the snapshot
    /// to decide.
    #[inline]
    pub fn value(&self) -> Option<Option<&M::Value>> {
        self.change.as_ref().map(Change::new_value)
    }

    /// The value as of the start of `run`, if the overlay decides it.
    pub fn value_as_of(&self, run: u64) -> Option<Option<&M::Value>> {
        if self.run == run {
            match &self.prior {
                Prior::Stale => unreachable!("a run's writes keep their prior"),
                Prior::Absent => None,
                Prior::Was(v) => Some(v.as_ref()),
            }
        } else {
            self.value()
        }
    }
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
            run: 0,
        }
    }

    /// End a graph run: what was written so far becomes the "old" state
    /// of the next run, and nothing counts as changed in it yet. See
    /// [`Keyspace::get_old`] and [`Keyspace::run_changes`].
    pub(crate) fn end_run(&mut self) {
        let mut freed = 0;
        for handle in self.overlay.values_mut() {
            freed += handle.end_run();
        }
        self.state.grow(-(freed as isize));
        self.run += 1;
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
            run: _,
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
    inner: Overlay<K, M>,
    /// Keys written in the current run, each once.
    touched: Vec<K>,
}

/// A transaction's view of one keyspace: its own writes plus the
/// snapshot underneath. Borrowed from the tx for a few calls at a time.
pub struct Keyspace<'a, 'tx, K, M: Merge> {
    id: KeyspaceId<K, M>,
    writes: &'a mut Overlay<K, M>,
    touched: &'a mut Vec<K>,
    run: u64,
    state: &'a TxState,
    snap: &'a Snapshot<'tx>,
}

impl<K: Ord, M: Merge> Default for KeyspaceHandle<K, M> {
    fn default() -> Self {
        Self {
            inner: Overlay::new(),
            touched: Vec::new(),
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
        for (k, slot) in other.inner {
            if let Some(c) = slot.change {
                self.inner.insert(
                    k,
                    Slot {
                        change: Some(c),
                        run: 0,
                        prior: Prior::Stale,
                    },
                );
            }
        }
    }
    fn is_empty(&self) -> bool {
        self.inner.values().all(|s| s.change.is_none())
    }
    fn end_run(&mut self) -> usize {
        let mut freed = 0;
        for k in self.touched.drain(..) {
            let slot = self.inner.get_mut(&k).expect("touched keys have slots");
            freed += prior_bytes::<M>(&slot.prior) + k.resident_size() + ENTRY_OVERHEAD;
            slot.prior = Prior::Stale;
            // writes that cancelled out were kept only for the run's sake
            if slot.change.is_none() {
                self.inner.remove(&k);
                freed += slot_bytes::<K, M>(&k);
            }
        }
        freed
    }
    fn seal(self: Box<Self>, mut wal: Option<&mut Vec<u8>>) -> (Box<dyn Any + Send + Sync>, usize) {
        let mut bytes = 0;
        // the register is sorted, so collect bulk-builds the tree
        let tree: Resident<K, M::Value> = self
            .inner
            .into_iter()
            .filter_map(|(k, slot)| Some((k, slot.change?)))
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

impl<'a, 'tx, K, M> Keyspace<'a, 'tx, K, M>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Sync,
{
    #[cfg(test)]
    pub(crate) fn inner_is_empty(&self) -> bool {
        self.writes.values().all(|s| s.change.is_none())
    }

    /// Record a materialized change. Fails if the tx has failed, including
    /// by this write taking it over its memory limit.
    pub fn absorb(&mut self, key: K, change: Change<M>) -> Result<()>
    where
        K: Clone,
    {
        debug_assert!(
            !matches!(change, Change::Modify(_)),
            "resolve before absorbing"
        );
        self.state.check()?;
        let delta = match self.writes.entry(key) {
            Entry::Vacant(v) => {
                // the entry, its change, and the run's memory of the key
                let bytes = slot_bytes::<K, M>(v.key())
                    + value_bytes::<M>(&change)
                    + v.key().resident_size()
                    + ENTRY_OVERHEAD;
                self.touched.push(v.key().clone());
                v.insert(Slot {
                    change: Some(change),
                    run: self.run,
                    prior: Prior::Absent,
                });
                bytes as isize
            }
            Entry::Occupied(mut o) => {
                let key_bytes = o.key().resident_size();
                let mut delta = 0isize;
                if o.get().run != self.run {
                    // the first write of this run remembers the value
                    // the key had when the run began
                    self.touched.push(o.key().clone());
                    let slot = o.get_mut();
                    slot.run = self.run;
                    slot.prior = match slot.value() {
                        None => Prior::Absent,
                        Some(v) => Prior::Was(v.cloned()),
                    };
                    delta += (prior_bytes::<M>(&slot.prior) + key_bytes + ENTRY_OVERHEAD) as isize;
                }
                let slot = o.get_mut();
                let before = slot.change.as_ref().map_or(0, value_bytes::<M>);
                slot.change = match slot.change.take() {
                    Some(prev) => prev.compose(change),
                    None => Some(change),
                };
                let after = slot.change.as_ref().map_or(0, value_bytes::<M>);
                delta + after as isize - before as isize
            }
        };
        if self.state.grow(delta) {
            // release this keyspace's writes now; the tx drops the rest
            // the next time it opens a keyspace, or when it ends
            *self.writes = Overlay::new();
            self.touched.clear();
            return self.state.check();
        }
        Ok(())
    }

    /// Set `key` to `value`, or delete it with `None`: reads the old
    /// value to record the change.
    pub fn set(&mut self, key: K, value: Option<M::Value>) -> Result<()>
    where
        K: Clone,
        M::Value: PartialEq,
    {
        let old = self.get(&key)?.map(Cow::into_owned);
        let change = match (old, value) {
            (None, None) => return Ok(()),
            (Some(o), Some(n)) if o == n => return Ok(()),
            (None, Some(n)) => Change::Insert(n),
            (Some(o), None) => Change::Delete(o),
            (Some(o), Some(n)) => Change::Replace(o, n),
        };
        self.absorb(key, change)
    }

    /// Current value of `key` as seen by this transaction: the overlay
    /// wins (a Delete there shadows the snapshot), otherwise the snapshot.
    /// Overlay hits borrow; snapshot hits are owned.
    pub fn get(&self, key: &K) -> Result<Option<Cow<'_, M::Value>>> {
        self.state
            .read(|| match self.writes.get(key).and_then(Slot::value) {
                Some(v) => Ok(v.map(Cow::Borrowed)),
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
        changes: impl IntoIterator<Item = (K, Change<M>)>,
        out: &mut Vec<(K, Change<M>)>,
    ) -> Result<()>
    where
        K: Clone,
        M::Value: PartialEq,
    {
        self.state.check()?;
        let mut lookup = self.snap.lookup(self.id);
        for (key, change) in changes {
            let old = match self.writes.get(&key).and_then(Slot::value) {
                Some(v) => v.map(Cow::Borrowed),
                None => lookup.get(&key).inspect_err(|e| self.state.fail(e))?,
            };
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
        self.cursor(
            range.start_bound().cloned(),
            range.end_bound().cloned(),
            false,
        )
    }

    /// [`range`](Self::range), in reverse key order.
    pub fn range_rev(&self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor(
            range.start_bound().cloned(),
            range.end_bound().cloned(),
            true,
        )
    }

    /// The value of `key` as of the start of this graph run: what
    /// [`get`](Self::get) returned before anything this run wrote.
    pub fn get_old(&self, key: &K) -> Result<Option<Cow<'_, M::Value>>> {
        match self.writes.get(key).and_then(|s| s.value_as_of(self.run)) {
            Some(v) => Ok(v.map(Cow::Borrowed)),
            None => self.state.read(|| self.snap.get(self.id, key)),
        }
    }

    /// [`range`](Self::range) as of the start of this graph run.
    pub fn range_old(&self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor_old(range, false)
    }

    /// [`range_old`](Self::range_old), in reverse key order.
    pub fn range_old_rev(&self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor_old(range, true)
    }

    fn cursor_old(&self, range: impl RangeBounds<K>, rev: bool) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.snap.cursor_as_of(
            self.id,
            &*self.writes,
            self.run,
            range.start_bound().cloned(),
            range.end_bound().cloned(),
            rev,
            Some(self.state),
        )
    }

    /// Whether anything was written to this keyspace in this graph run.
    pub fn changed_this_run(&self) -> bool {
        !self.touched.is_empty()
    }

    /// Every key whose value this graph run changed, from its value at
    /// the start of the run to its value now, in key order. Needs no
    /// reads: the overlay remembers values from before the tx.
    pub fn run_changes(&self, out: &mut Register<K, M>)
    where
        K: Clone,
        M::Value: PartialEq,
    {
        for k in self.touched.iter() {
            let slot = &self.writes[k];
            let old = match &slot.prior {
                Prior::Was(v) => v.as_ref(),
                // nothing before this run: as of before the tx
                Prior::Absent => slot.change.as_ref().and_then(Change::old_value),
                Prior::Stale => unreachable!("touched keys keep their prior"),
            };
            let new = slot.change.as_ref().and_then(Change::new_value);
            let change = match (old, new) {
                (None, None) => continue,
                (Some(o), Some(n)) if o == n => continue,
                (None, Some(n)) => Change::Insert(n.clone()),
                (Some(o), None) => Change::Delete(o.clone()),
                (Some(o), Some(n)) => Change::Replace(o.clone(), n.clone()),
            };
            out.absorb_one(k.clone(), change);
        }
    }

    fn cursor(&self, lower: Bound<K>, upper: Bound<K>, rev: bool) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.snap.cursor(
            self.id,
            Some(&*self.writes),
            lower,
            upper,
            rev,
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
            touched: &mut handle.touched,
            run: self.run,
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

    fn changes(ks: &Keyspace<'_, '_, u8, i64>) -> Vec<(u8, Option<i64>, Option<i64>)> {
        let mut out = Register::new();
        ks.run_changes(&mut out);
        out.iter()
            .map(|(k, c)| (*k, c.old_value().copied(), c.new_value().copied()))
            .collect()
    }

    #[test]
    fn runs_see_old_and_new_state_and_their_own_changes() {
        let store = StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(KS)
            .build()
            .unwrap();
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        for k in 1..=4 {
            ks.absorb(k, Change::Insert(k as i64 * 10)).unwrap();
        }
        store.finish(tx).unwrap();

        let mut tx = store.begin().unwrap();
        // run 0: change 1, delete 2, insert 5, insert then cancel 6
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Replace(10, 11)).unwrap();
        ks.absorb(2, Change::Delete(20)).unwrap();
        ks.absorb(5, Change::Insert(50)).unwrap();
        ks.absorb(6, Change::Insert(60)).unwrap();
        ks.absorb(6, Change::Delete(60)).unwrap();
        assert_eq!(
            changes(&ks),
            vec![
                (1, Some(10), Some(11)),
                (2, Some(20), None),
                (5, None, Some(50))
            ]
        );
        assert_eq!(ks.get_old(&1).unwrap().as_deref(), Some(&10));
        assert_eq!(ks.get_old(&2).unwrap().as_deref(), Some(&20));
        assert_eq!(ks.get_old(&5).unwrap().as_deref(), None);
        assert_eq!(ks.get_old(&3).unwrap().as_deref(), Some(&30), "untouched");
        let old: Vec<(u8, i64)> = ks
            .range_old(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(old, vec![(1, 10), (2, 20), (3, 30), (4, 40)]);

        // run 1: the old state is what run 0 left
        tx.end_run();
        let mut ks = tx.keyspace(KS);
        assert!(!ks.changed_this_run());
        assert_eq!(ks.get_old(&1).unwrap().as_deref(), Some(&11));
        ks.absorb(1, Change::Replace(11, 10)).unwrap(); // back to before the tx
        ks.absorb(5, Change::Delete(50)).unwrap(); // cancels run 0's insert
        ks.absorb(2, Change::Insert(21)).unwrap();
        ks.absorb(3, Change::Delete(30)).unwrap();
        assert_eq!(
            changes(&ks),
            vec![
                (1, Some(11), Some(10)),
                (2, None, Some(21)),
                (3, Some(30), None),
                (5, Some(50), None)
            ]
        );
        assert_eq!(ks.get_old(&5).unwrap().as_deref(), Some(&50));
        assert_eq!(ks.get_old(&2).unwrap().as_deref(), None);
        let old: Vec<(u8, i64)> = ks
            .range_old(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(old, vec![(1, 11), (3, 30), (4, 40), (5, 50)]);
        let new: Vec<(u8, i64)> = ks
            .range(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(new, vec![(1, 10), (2, 21), (4, 40)]);
        let from3: Vec<u8> = ks.range_old(3..).map(|r| *r.unwrap().0).collect();
        assert_eq!(from3, vec![3, 4, 5]);
        store.finish(tx).unwrap();
        assert_eq!(store.snapshot().get(KS, &1).unwrap().as_deref(), Some(&10));
    }
}
