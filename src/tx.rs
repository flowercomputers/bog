use std::{
    any::Any,
    borrow::Cow,
    collections::HashMap,
    marker::PhantomData,
    sync::MutexGuard,
};

use crate::{
    Change, Register,
    codec::{Codec, OrderedCodec},
    merge::Merge,
    store::{Resident, Snapshot, Writer},
};

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
    /// into `wal` on the way when there is one.
    fn seal(self: Box<Self>, wal: Option<&mut Vec<u8>>) -> Box<dyn Any + Send + Sync>;
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
pub struct WriteTx<'tx> {
    /// The epoch this tx will commit as: one past the snapshot's.
    epoch: u64,
    overlay: HashMap<u64, Box<dyn ErasedUnderlay>>,
    snapshot: Snapshot<'tx>,
    writer: MutexGuard<'tx, Writer>,
}

impl<'tx> WriteTx<'tx> {
    pub(crate) fn new(snapshot: Snapshot<'tx>, writer: MutexGuard<'tx, Writer>) -> Self {
        WriteTx {
            epoch: snapshot.epoch() + 1,
            overlay: HashMap::new(),
            snapshot,
            writer,
        }
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        u64,
        HashMap<u64, Box<dyn ErasedUnderlay>>,
        Snapshot<'tx>,
        MutexGuard<'tx, Writer>,
    ) {
        let WriteTx {
            epoch,
            overlay,
            snapshot,
            writer,
        } = self;
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
    fn seal(self: Box<Self>, mut wal: Option<&mut Vec<u8>>) -> Box<dyn Any + Send + Sync> {
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
                (k, v)
            })
            .collect();
        Box::new(tree)
    }
}

impl<K: Ord + 'static, M: Merge> KeyspaceHandle<K, M> {
    #[cfg(test)]
    pub(crate) fn inner_is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<'a, 'tx, K: Ord + 'static, M: Merge> Keyspace<'a, 'tx, K, M> {
    #[cfg(test)]
    pub(crate) fn inner_is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    pub fn absorb(&mut self, key: K, change: Change<M>) {
        debug_assert!(
            !matches!(change, Change::Modify(_)),
            "resolve before absorbing"
        );
        self.writes.absorb_one(key, change);
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
        ks.absorb(1, Change::Insert(10));
        ks.absorb(2, Change::Insert(20));
        store.finish(tx).unwrap();
        let mut tx = store.begin();
        tx.keyspace(KS).absorb(2, Change::Delete(20));
        store.finish(tx).unwrap();

        let mut tx = store.begin();
        let mut ks = tx.keyspace(KS);
        assert_eq!(ks.get(&1).as_deref(), Some(&10));
        assert_eq!(ks.get(&2).as_deref(), None, "committed tombstone");
        assert_eq!(ks.get(&3).as_deref(), None, "never written");

        // a Modify resolves against the snapshot value
        let r = ks.resolve(&1, Change::Modify(5)).unwrap();
        assert!(matches!(r, Change::Replace(10, 15)));
        ks.absorb(1, r);
        assert_eq!(ks.get(&1).as_deref(), Some(&15), "overlay wins");

        // a Delete in the overlay shadows the snapshot
        let r = ks.resolve(&1, Change::Delete(0)).unwrap();
        assert!(matches!(r, Change::Delete(15)));
        ks.absorb(1, r);
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
