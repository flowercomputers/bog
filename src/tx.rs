use std::{
    any::Any,
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    ops::RangeBounds,
};

use crate::{
    Change, Register,
    codec::{Codec, OrderedCodec, ResidentSize, heap_size},
    cursor::Cursor,
    error::{Error, Result},
    merge::Merge,
    store::{ENTRY_OVERHEAD, Lookup, Resident, Snapshot, WriterGuard, resident_entry_bytes},
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
fn entry_bytes<K: ResidentSize, M: Merge>(key: &K, change: &Change<M>) -> usize
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
    /// Settle the run's writes, returning how the bytes held changed.
    fn end_run(&mut self) -> isize;
    /// Settle every run, before `is_empty` or `seal`.
    fn settle(&mut self);
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

/// One graph run's writes to one keyspace: each key once, in order, with
/// its change over the run, old value to new. Appends in key order stay
/// sorted; others go to a tail, merged in by the next read. The newest
/// run holding a key decides its value.
pub(crate) struct OverlayRun<K, M: Merge> {
    pub run: u64,
    pub entries: Vec<(K, Change<M>)>,
    /// `entries[..sorted]` is sorted with each key once.
    sorted: usize,
}

/// `next` applied after `prev`, both changes to one key; `None` when they
/// cancel out. In an additive keyspace a change is `Insert(amount)`, and
/// two of them add up.
fn compose<M: Merge>(prev: Change<M>, next: Change<M>) -> Option<Change<M>> {
    if !M::ADDITIVE {
        return prev.compose(next);
    }
    match (prev, next) {
        (Change::Insert(mut sum), Change::Insert(amount)) => {
            M::add(&mut sum, &amount);
            (!M::consider_delete(&sum)).then_some(Change::Insert(sum))
        }
        _ => unreachable!("an additive keyspace holds amounts"),
    }
}

impl<K: Ord, M: Merge> OverlayRun<K, M> {
    fn new(run: u64) -> Self {
        OverlayRun {
            run,
            entries: Vec::new(),
            sorted: 0,
        }
    }

    /// The change at `key`, if this run wrote it. The run must be settled.
    fn get(&self, key: &K) -> Option<&Change<M>> {
        debug_assert_eq!(self.sorted, self.entries.len(), "settle before reading");
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// Merge the tail into the sorted entries, composing a key's changes
    /// in the order they came and dropping those that cancel out.
    fn settle(&mut self) {
        if self.sorted == self.entries.len() {
            return;
        }
        let mut tail: Vec<(K, Change<M>)> = self.entries.drain(self.sorted..).collect();
        tail.sort_by(|a, b| a.0.cmp(&b.0));
        let head = std::mem::take(&mut self.entries);
        let mut merged = Vec::with_capacity(head.len() + tail.len());
        let (mut head, mut tail) = (head.into_iter().peekable(), tail.into_iter().peekable());
        loop {
            let from_tail = match (head.peek(), tail.peek()) {
                (None, None) => break,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (Some((h, _)), Some((t, _))) => t <= h,
            };
            let (k, mut change) = if from_tail {
                tail.next().unwrap()
            } else {
                head.next().unwrap()
            };
            // older changes to the key come first; later ones compose on
            let mut alive = Some(change);
            while let Some((_, next)) = tail.next_if(|(t, _)| *t == k) {
                alive = match alive {
                    Some(prev) => compose(prev, next),
                    None => Some(next),
                };
            }
            if from_tail {
                // a sorted entry for the key is older than the tail's
                if let Some((_, older)) = head.next_if(|(h, _)| *h == k) {
                    alive = match alive {
                        Some(newer) => compose(older, newer),
                        None => Some(older),
                    };
                }
            }
            if let Some(c) = alive.take() {
                change = c;
                merged.push((k, change));
            }
        }
        self.entries = merged;
        self.sorted = self.entries.len();
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
        let mut delta = 0;
        for handle in self.overlay.values_mut() {
            delta += handle.end_run();
        }
        self.state.grow(delta);
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
            mut overlay,
            state,
            snapshot,
            writer,
            rebuilding: _,
            run: _,
        } = self;
        for handle in overlay.values_mut() {
            handle.settle();
        }
        let overlay = state.check().map(|()| overlay);
        (epoch, overlay, snapshot, writer)
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// This transaction's changes to one keyspace: one `OverlayRun` per
/// graph run that wrote it, oldest first.
pub struct KeyspaceHandle<K, M: Merge> {
    runs: Vec<OverlayRun<K, M>>,
}

/// A transaction's view of one keyspace: its own writes plus the
/// snapshot underneath. Borrowed from the tx for a few calls at a time.
/// Reads take `&mut self` because they settle writes made out of order.
pub struct Keyspace<'a, 'tx, K, M: Merge> {
    id: KeyspaceId<K, M>,
    runs: &'a mut Vec<OverlayRun<K, M>>,
    run: u64,
    state: &'a TxState,
    snap: &'a Snapshot<'tx>,
}

impl<K: Ord, M: Merge> Default for KeyspaceHandle<K, M> {
    fn default() -> Self {
        Self { runs: Vec::new() }
    }
}

/// The net changes the tx behind `handle` makes to its keyspace, in key
/// order. The handle must be settled, as it is once the tx is finishing.
pub(crate) fn net_changes<K, M>(handle: &dyn ErasedUnderlay) -> Vec<(K, Change<M>)>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Sync,
{
    let handle = handle
        .as_any()
        .downcast_ref::<KeyspaceHandle<K, M>>()
        .expect("keyspace id reused with a different type");
    let mut out = Vec::new();
    composed(
        handle
            .runs
            .iter()
            .map(|r| r.entries.iter().map(|(k, c)| (k, c.clone()))),
        |k, change| {
            if let Some(change) = change {
                out.push((k.clone(), change));
            }
        },
    );
    out
}

/// Walk `runs`' changes, oldest run first, composing each key's changes
/// across runs: `f` gets every key in order with what the tx does to it,
/// `None` where its changes cancelled out. The runs must be settled.
fn composed<Q: Ord, M: Merge, I: Iterator<Item = (Q, Change<M>)>>(
    runs: impl Iterator<Item = I>,
    mut f: impl FnMut(Q, Option<Change<M>>),
) {
    let mut its: Vec<_> = runs.map(Iterator::peekable).collect();
    loop {
        // the oldest run holding the least key
        let heads: Vec<Option<&Q>> = its.iter_mut().map(|it| it.peek().map(|(k, _)| k)).collect();
        let mut win: Option<usize> = None;
        for (i, head) in heads.iter().enumerate() {
            if let Some(k) = head
                && win.is_none_or(|w| *k < heads[w].expect("a winner has a head"))
            {
                win = Some(i);
            }
        }
        drop(heads);
        let Some(w) = win else { break };
        let (key, first) = its[w].next().expect("the winner has a head");
        let mut change = Some(first);
        for it in &mut its[w + 1..] {
            if let Some((_, c)) = it.next_if(|(k, _)| *k == key) {
                change = match change {
                    Some(prev) => compose(prev, c),
                    None => Some(c),
                };
            }
        }
        f(key, change);
    }
}

/// Whether the tx does nothing to any key of `runs`, which must be
/// settled.
fn runs_empty<K: Ord, M: Merge>(runs: &[OverlayRun<K, M>]) -> bool {
    if runs.iter().all(|r| r.entries.is_empty()) {
        return true;
    }
    if runs.len() == 1 {
        return false;
    }
    let mut live = false;
    composed(
        runs.iter()
            .map(|r| r.entries.iter().map(|(k, c)| (k, c.clone()))),
        |_, c| live |= c.is_some(),
    );
    !live
}

/// Estimated resident bytes of the entries of `runs`.
fn runs_bytes<K: ResidentSize, M: Merge>(runs: &[OverlayRun<K, M>]) -> usize
where
    M::Value: ResidentSize,
{
    runs.iter()
        .flat_map(|r| r.entries.iter())
        .map(|(k, c)| entry_bytes::<K, M>(k, c))
        .sum()
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
        self.runs.extend(other.runs);
    }
    fn is_empty(&self) -> bool {
        runs_empty(&self.runs)
    }
    fn settle(&mut self) {
        for r in &mut self.runs {
            r.settle();
        }
    }
    fn end_run(&mut self) -> isize {
        let Some(last) = self.runs.last_mut() else {
            return 0;
        };
        let before = runs_bytes(std::slice::from_ref(last));
        last.settle();
        runs_bytes(std::slice::from_ref(last)) as isize - before as isize
    }
    fn seal(
        mut self: Box<Self>,
        mut wal: Option<&mut Vec<u8>>,
    ) -> (Box<dyn Any + Send + Sync>, usize) {
        for r in &mut self.runs {
            r.settle();
        }
        // composed across runs and in key order, so collect bulk-builds
        // the tree
        let mut bytes = 0;
        let mut tree = Resident::<K, M::Value>::new();
        composed(
            self.runs.into_iter().map(|r| r.entries.into_iter()),
            |k, change| {
                let v = match change {
                    Some(Change::Insert(v) | Change::Replace(_, v)) => Some(v),
                    // a surviving Delete means the key exists below this tx
                    Some(Change::Delete(_)) => None,
                    Some(Change::Modify(_)) => unreachable!("resolve before absorbing"),
                    // written and unwritten within the tx
                    None => return,
                };
                if let Some(out) = wal.as_deref_mut() {
                    k.encode_ordered(out);
                    v.encode(out);
                }
                bytes += resident_entry_bytes(&k, &v);
                tree.insert(k, v);
            },
        );
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
    pub(crate) fn inner_is_empty(&mut self) -> bool {
        for r in self.runs.iter_mut() {
            r.settle();
        }
        runs_empty(self.runs)
    }

    /// The run being written, settled.
    fn current(&mut self) -> &mut OverlayRun<K, M> {
        if self.runs.last().is_none_or(|r| r.run != self.run) {
            self.runs.push(OverlayRun::new(self.run));
        }
        self.runs.last_mut().expect("just ensured")
    }

    /// Settle the current run, so reads see every write in order.
    fn settle(&mut self) {
        if let Some(last) = self.runs.last_mut()
            && last.run == self.run
            && last.sorted != last.entries.len()
        {
            let before = runs_bytes(std::slice::from_ref(last));
            last.settle();
            let after = runs_bytes(std::slice::from_ref(last));
            self.state.grow(after as isize - before as isize);
        }
    }

    /// The runs before this one: the state as of its start.
    fn older(&self) -> &[OverlayRun<K, M>] {
        let n = self.runs.partition_point(|r| r.run < self.run);
        &self.runs[..n]
    }

    /// The newest change to `key` among `runs`, which must be settled.
    fn find<'r>(runs: &'r [OverlayRun<K, M>], key: &K) -> Option<&'r Change<M>> {
        runs.iter().rev().find_map(|r| r.get(key))
    }

    /// Record a materialized change. Fails if the tx has failed, including
    /// by this write taking it over its memory limit.
    pub fn absorb(&mut self, key: K, change: Change<M>) -> Result<()> {
        debug_assert!(
            !matches!(change, Change::Modify(_)),
            "resolve before absorbing"
        );
        debug_assert!(
            !M::ADDITIVE || matches!(change, Change::Insert(_)),
            "an additive keyspace takes amounts: use add"
        );
        self.state.check()?;
        let run = self.current();
        let in_order = run.sorted == run.entries.len();
        let place = match run.entries.last() {
            Some((last, _)) if in_order && *last == key => 0,
            Some((last, _)) if in_order && *last < key => 1,
            _ => 2,
        };
        let mut delta = 0isize;
        match place {
            // a repeat of the last key composes in place
            0 => {
                let (k, prev) = run.entries.pop().expect("a last entry");
                run.sorted -= 1;
                delta -= entry_bytes::<K, M>(&k, &prev) as isize;
                if let Some(c) = compose(prev, change) {
                    delta += entry_bytes::<K, M>(&k, &c) as isize;
                    run.entries.push((k, c));
                    run.sorted += 1;
                }
            }
            // in key order: still sorted
            1 => {
                delta += entry_bytes::<K, M>(&key, &change) as isize;
                run.entries.push((key, change));
                run.sorted += 1;
            }
            // out of order, or the first: the tail, settled by the next read
            _ => {
                delta += entry_bytes::<K, M>(&key, &change) as isize;
                let first = run.entries.is_empty();
                run.entries.push((key, change));
                if first {
                    run.sorted = 1;
                }
            }
        }
        if self.state.grow(delta) {
            // release this keyspace's writes now; the tx drops the rest
            // the next time it opens a keyspace, or when it ends
            self.runs.clear();
            return self.state.check();
        }
        Ok(())
    }

    /// Add `amount` to `key` in an [additive](Merge::ADDITIVE) keyspace.
    /// Reads nothing.
    pub fn add(&mut self, key: K, amount: M::Value) -> Result<()> {
        assert!(M::ADDITIVE, "add needs an additive keyspace");
        if M::consider_delete(&amount) {
            return self.state.check();
        }
        self.absorb(key, Change::Insert(amount))
    }

    /// The sum of what `runs`, oldest first and settled, and the snapshot
    /// hold for `key` in an additive keyspace.
    fn sum(&self, runs: &[OverlayRun<K, M>], key: &K) -> Result<Option<M::Value>> {
        let (id, snap) = (self.id, self.snap);
        let mut sum = self.state.read(|| snap.get(id, key))?.map(Cow::into_owned);
        for amount in runs.iter().filter_map(|r| r.get(key)) {
            let amount = amount.new_value().expect("an amount");
            match &mut sum {
                Some(s) => M::add(s, amount),
                None => sum = Some(amount.clone()),
            }
        }
        Ok(sum.filter(|s| !M::consider_delete(s)))
    }

    /// Set `key` to `value`, or delete it with `None`: reads the old
    /// value to record the change.
    pub fn set(&mut self, key: K, value: Option<M::Value>) -> Result<()>
    where
        M::Value: PartialEq,
    {
        assert!(!M::ADDITIVE, "an additive keyspace takes amounts: use add");
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
    pub fn get(&mut self, key: &K) -> Result<Option<Cow<'_, M::Value>>> {
        self.settle();
        if M::ADDITIVE {
            return Ok(self.sum(&self.runs[..], key)?.map(Cow::Owned));
        }
        let (id, snap, state) = (self.id, self.snap, self.state);
        match Self::find(self.runs, key) {
            Some(c) => Ok(c.new_value().map(Cow::Borrowed)),
            None => state.read(|| snap.get(id, key)),
        }
    }

    /// Materialize `change` against the current value of `key`. Returns the
    /// Insert/Delete/Replace to record, or None if nothing changes.
    pub fn resolve(&mut self, key: &K, change: Change<M>) -> Result<Option<Change<M>>>
    where
        M::Value: PartialEq,
    {
        assert!(!M::ADDITIVE, "an additive keyspace takes amounts: use add");
        Ok(materialize(self.get(key)?.as_deref(), change))
    }

    /// A reader of values for keys asked in ascending order: the batch
    /// form of [`get`](Self::get), one pass over the snapshot's tables.
    pub fn lookup(&mut self) -> TxLookup<'_, K, M> {
        self.settle();
        TxLookup {
            runs: &self.runs[..],
            snap: self.snap.lookup(self.id),
            state: self.state,
        }
    }

    /// [`resolve`](Self::resolve) for a whole batch. `changes` is sorted, so
    /// one pass walks the overlay, the layers and the tables forward
    /// together, and each table block is read at most once. Resolved
    /// changes land in `out` in key order.
    pub fn resolve_sorted(
        &mut self,
        changes: impl IntoIterator<Item = (K, Change<M>)>,
        out: &mut Vec<(K, Change<M>)>,
    ) -> Result<()>
    where
        K: Clone,
        M::Value: PartialEq,
    {
        assert!(!M::ADDITIVE, "an additive keyspace takes amounts: use add");
        self.state.check()?;
        self.settle();
        let mut lookup = self.snap.lookup(self.id);
        for (key, change) in changes {
            let old = match Self::find(self.runs, &key) {
                Some(c) => c.new_value().map(Cow::Borrowed),
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
    pub fn range(&mut self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.settle();
        self.cursor(range, false, false)
    }

    /// [`range`](Self::range), in reverse key order.
    pub fn range_rev(&mut self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.settle();
        self.cursor(range, true, false)
    }

    /// The value of `key` as of the start of this graph run: what
    /// [`get`](Self::get) returned before anything this run wrote.
    pub fn get_old(&mut self, key: &K) -> Result<Option<Cow<'_, M::Value>>> {
        if M::ADDITIVE {
            return Ok(self.sum(self.older(), key)?.map(Cow::Owned));
        }
        let (id, snap, state) = (self.id, self.snap, self.state);
        match Self::find(self.older(), key) {
            Some(c) => Ok(c.new_value().map(Cow::Borrowed)),
            None => state.read(|| snap.get(id, key)),
        }
    }

    /// [`range`](Self::range) as of the start of this graph run.
    pub fn range_old(&mut self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor(range, false, true)
    }

    /// [`range_old`](Self::range_old), in reverse key order.
    pub fn range_old_rev(&mut self, range: impl RangeBounds<K>) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        self.cursor(range, true, true)
    }

    /// Whether anything was written to this keyspace in this graph run.
    pub fn changed_this_run(&self) -> bool {
        self.runs
            .last()
            .is_some_and(|r| r.run == self.run && !r.entries.is_empty())
    }

    /// Every key whose value this graph run changed, from its value at
    /// the start of the run to its value now, in key order. Needs no
    /// reads: the run's changes carry both.
    pub fn run_changes(&mut self, out: &mut Register<K, M>)
    where
        K: Clone,
        M::Value: PartialEq,
    {
        self.settle();
        let Some(last) = self.runs.last() else { return };
        if last.run != self.run {
            return;
        }
        for (k, c) in &last.entries {
            if let (Some(o), Some(n)) = (c.old_value(), c.new_value())
                && o == n
            {
                continue;
            }
            out.absorb_one(k.clone(), c.clone());
        }
    }

    fn cursor(&self, range: impl RangeBounds<K>, rev: bool, old: bool) -> Cursor<'_, K, M>
    where
        K: Clone,
    {
        let runs = if old { self.older() } else { &self.runs[..] };
        self.snap.cursor(
            self.id,
            runs,
            range.start_bound().cloned(),
            range.end_bound().cloned(),
            rev,
            Some(self.state),
        )
    }
}

/// Point reads of a tx's keyspace in ascending key order; see
/// [`Keyspace::lookup`].
pub struct TxLookup<'a, K, M: Merge> {
    runs: &'a [OverlayRun<K, M>],
    snap: Lookup<'a, K, M::Value>,
    state: &'a TxState,
}

impl<'a, K, M> TxLookup<'a, K, M>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Sync,
{
    /// The value of `key`, which must not be less than the last key asked.
    pub fn get(&mut self, key: &K) -> Result<Option<Cow<'a, M::Value>>> {
        if !M::ADDITIVE {
            return match Keyspace::<K, M>::find(self.runs, key) {
                Some(c) => Ok(c.new_value().map(Cow::Borrowed)),
                None => self.snap.get(key).inspect_err(|e| self.state.fail(e)),
            };
        }
        let mut sum = self
            .snap
            .get(key)
            .inspect_err(|e| self.state.fail(e))?
            .map(Cow::into_owned);
        for amount in self.runs.iter().filter_map(|r| r.get(key)) {
            let amount = amount.new_value().expect("an amount");
            match &mut sum {
                Some(s) => M::add(s, amount),
                None => sum = Some(amount.clone()),
            }
        }
        Ok(sum.filter(|s| !M::consider_delete(s)).map(Cow::Owned))
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
            runs: &mut handle.runs,
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

    fn changes(ks: &mut Keyspace<'_, '_, u8, i64>) -> Vec<(u8, Option<i64>, Option<i64>)> {
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
            changes(&mut ks),
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
            changes(&mut ks),
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
