use std::{
    any::Any, borrow::Cow, collections::HashMap, fmt, marker::PhantomData, sync::MutexGuard,
};

use crate::{
    Change, Register,
    codec::{Codec, OrderedCodec, ResidentSize, heap_size},
    merge::Merge,
    store::{ENTRY_OVERHEAD, Resident, Snapshot, StoreWriter, resident_entry_bytes},
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

/// A write tx's memory accounting: the estimated resident bytes of every
/// change in its overlay.
struct Budget {
    used: usize,
    limit: usize,
    killed: Option<OutOfMemory>,
}

impl Budget {
    /// Apply an entry's growth; returns true when that kills the tx.
    fn grow(&mut self, delta: isize) -> bool {
        self.used = self.used.saturating_add_signed(delta);
        if self.used > self.limit && self.killed.is_none() {
            self.killed = Some(OutOfMemory {
                used: self.used,
                limit: self.limit,
            });
            self.used = 0;
            return true;
        }
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
/// Its writes count against a memory limit. A write that takes them over
/// it kills the tx: every write so far is released, later writes fail with
/// [`OutOfMemory`], and so does committing. Reads still work but no longer
/// see the tx's own writes.
pub struct WriteTx<'tx> {
    /// The epoch this tx will commit as: one past the snapshot's.
    epoch: u64,
    overlay: HashMap<u64, Box<dyn ErasedUnderlay>>,
    budget: Budget,
    snapshot: Snapshot<'tx>,
    writer: MutexGuard<'tx, StoreWriter>,
}

impl<'tx> WriteTx<'tx> {
    pub(crate) fn new(
        snapshot: Snapshot<'tx>,
        writer: MutexGuard<'tx, StoreWriter>,
        memory_limit: usize,
    ) -> Self {
        WriteTx {
            epoch: snapshot.epoch() + 1,
            overlay: HashMap::new(),
            budget: Budget {
                used: 0,
                limit: memory_limit,
                killed: None,
            },
            snapshot,
            writer,
        }
    }

    /// Estimated bytes held by this tx's writes.
    pub fn memory_used(&self) -> usize {
        self.budget.used
    }

    /// Why this tx was killed, if it was.
    pub fn killed(&self) -> Option<OutOfMemory> {
        self.budget.killed
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        u64,
        Result<HashMap<u64, Box<dyn ErasedUnderlay>>, OutOfMemory>,
        Snapshot<'tx>,
        MutexGuard<'tx, StoreWriter>,
    ) {
        let WriteTx {
            epoch,
            overlay,
            budget,
            snapshot,
            writer,
        } = self;
        let overlay = match budget.killed {
            Some(oom) => Err(oom),
            None => Ok(overlay),
        };
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
    budget: &'a mut Budget,
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

    /// Record a materialized change. Fails if the tx has been killed for
    /// running out of memory, including by this write.
    pub fn absorb(&mut self, key: K, change: Change<M>) -> Result<(), OutOfMemory> {
        debug_assert!(
            !matches!(change, Change::Modify(_)),
            "resolve before absorbing"
        );
        if let Some(oom) = self.budget.killed {
            return Err(oom);
        }
        let delta = self
            .writes
            .absorb_one_measured(key, change, change_bytes::<K, M>);
        if self.budget.grow(delta) {
            // release this keyspace's writes now; the tx drops the rest
            // the next time it opens a keyspace, or when it ends
            *self.writes = Register::new();
            return Err(self.budget.killed.expect("just killed"));
        }
        Ok(())
    }

    /// Current value of `key` as seen by this transaction: the overlay
    /// wins (a Delete there shadows the snapshot), otherwise the snapshot.
    /// Overlay hits borrow; snapshot hits are owned.
    pub fn get(&self, key: &K) -> Option<Cow<'_, M::Value>> {
        match self.writes.get(key) {
            Some(c) => c.new_value().map(Cow::Borrowed),
            None => self.snap.get(self.id, key),
        }
    }

    /// Materialize `change` against the current value of `key`. Returns the
    /// Insert/Delete/Replace to record, or None if nothing changes.
    pub fn resolve(&self, key: &K, change: Change<M>) -> Option<Change<M>>
    where
        M::Value: PartialEq,
    {
        materialize(self.get(key).as_deref(), change)
    }

    /// [`resolve`](Self::resolve) for a whole batch: `changes` is sorted by
    /// key, so the overlay is read in one forward pass rather than a probe
    /// per key. Keys the overlay does not mention are point-probed in the
    /// snapshot. Resolved changes land in `out` in key order.
    // todo: once the snapshot exposes a sorted cursor, make this a three-way
    // merge so disk runs fault each block once.
    pub fn resolve_sorted(&self, changes: Register<K, M>, out: &mut Vec<(K, Change<M>)>)
    where
        M::Value: PartialEq,
    {
        let mut current = self.writes.iter().peekable();
        for (key, change) in changes {
            // advance the cursor to the first entry at or past `key`
            while current.peek().is_some_and(|(k, _)| **k < key) {
                current.next();
            }
            let old = match current.peek().filter(|(k, _)| **k == key) {
                Some((_, c)) => c.new_value().map(Cow::Borrowed),
                None => self.snap.get(self.id, &key),
            };
            if let Some(resolved) = materialize(old.as_deref(), change) {
                out.push((key, resolved));
            }
        }
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
        if self.budget.killed.is_some() {
            // a killed tx holds no writes
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
            budget: &mut self.budget,
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
        let mut tx = store.begin();
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Insert(10)).unwrap();
        ks.absorb(2, Change::Insert(20)).unwrap();
        store.finish(tx).unwrap();
        let mut tx = store.begin();
        tx.keyspace(KS).absorb(2, Change::Delete(20)).unwrap();
        store.finish(tx).unwrap();

        let mut tx = store.begin();
        let mut ks = tx.keyspace(KS);
        assert_eq!(ks.get(&1).as_deref(), Some(&10));
        assert_eq!(ks.get(&2).as_deref(), None, "committed tombstone");
        assert_eq!(ks.get(&3).as_deref(), None, "never written");

        // a Modify resolves against the snapshot value
        let r = ks.resolve(&1, Change::Modify(5)).unwrap();
        assert!(matches!(r, Change::Replace(10, 15)));
        ks.absorb(1, r).unwrap();
        assert_eq!(ks.get(&1).as_deref(), Some(&15), "overlay wins");

        // a Delete in the overlay shadows the snapshot
        let r = ks.resolve(&1, Change::Delete(0)).unwrap();
        assert!(matches!(r, Change::Delete(15)));
        ks.absorb(1, r).unwrap();
        assert_eq!(ks.get(&1).as_deref(), None);

        // resolve_sorted mixes overlay hits and snapshot misses
        let mut changes = Register::new();
        changes.absorb_one(1, Change::Modify(1));
        changes.absorb_one(3, Change::Modify(7));
        let mut out = Vec::new();
        ks.resolve_sorted(changes, &mut out);
        // key 1 is deleted in the overlay, so the snapshot's 10 is invisible
        assert!(matches!(out[0], (1, Change::Insert(1))));
        assert!(matches!(out[1], (3, Change::Insert(7))));
    }
}
