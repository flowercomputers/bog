//! What flows along the graph's edges: a [`Bag`] of rows at multiplicities,
//! or a [`Register`] of keyed changes.

use std::{
    any::Any,
    collections::{BTreeMap, btree_map::Entry},
};

use crate::store::merge::{Change, Merge};

pub trait Buffer: Any + Send {
    fn empty() -> Self
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn clear(&mut self);
    /// Fold `other` into self, leaving `other` empty. Bags append;
    /// registers compose per key.
    fn absorb(&mut self, other: &mut Self)
    where
        Self: Sized;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Put the buffer in the form readers expect; see [`Bag::consolidate`].
    /// The graph does this to a node's output after every run.
    fn consolidate(&mut self) {}
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The bytes of allocation the buffer holds, whether or not they are
    /// in use; with `release`, an empty buffer gives them back first.
    /// The graph asks after every run, and releases once the whole graph
    /// holds more than its limit; see [`OpDag::retain`](super::OpDag::retain).
    fn scratch(&mut self, _release: bool) -> usize {
        0
    }
}

/// [`Buffer::scratch`] of a `Vec` an operator or buffer reuses across
/// runs: its capacity in bytes, after dropping it if `release`.
pub fn vec_scratch<T>(v: &mut Vec<T>, release: bool) -> usize {
    if release {
        *v = Vec::new();
    }
    v.capacity() * size_of::<T>()
}

/// A keyed, consolidated batch: at most one `Change` per key. Composing
/// into it keeps it consolidated, so there is no separate consolidate step.
#[repr(transparent)]
pub struct Register<K, M: Merge> {
    inner: BTreeMap<K, Change<M>>,
}

impl<K: Ord, M: Merge> Default for Register<K, M> {
    fn default() -> Self {
        Register {
            inner: BTreeMap::new(),
        }
    }
}

impl<K: Ord, M: Merge> Register<K, M> {
    pub fn new() -> Self {
        Self::default()
    }
    /// Compose `change` onto whatever is already at `key`. Cancelling pairs
    /// (Insert then Delete) and no-op modifies drop the key entirely.
    pub fn absorb_one(&mut self, key: K, change: Change<M>) {
        self.absorb_one_measured(key, change, |_, _| 0);
    }

    /// [`absorb_one`](Self::absorb_one), returning how much the key's entry
    /// grew (or, negative, shrank) as measured by `size`. A key that is
    /// added or removed counts its whole entry.
    pub fn absorb_one_measured(
        &mut self,
        key: K,
        change: Change<M>,
        size: impl Fn(&K, &Change<M>) -> usize,
    ) -> isize {
        if let Change::Modify(d) = &change
            && M::noop(d)
        {
            return 0;
        }
        match self.inner.entry(key) {
            Entry::Vacant(vac) => {
                let after = size(vac.key(), &change);
                vac.insert(change);
                after as isize
            }
            Entry::Occupied(occ) => {
                let before = size(occ.key(), occ.get()) as isize;
                let (key, prev) = occ.remove_entry();
                let Some(c) = prev.compose(change) else {
                    return -before;
                };
                if let Change::Modify(d) = &c
                    && M::noop(d)
                {
                    return -before;
                }
                let after = size(&key, &c) as isize;
                self.inner.insert(key, c);
                after - before
            }
        }
    }
    pub fn get(&self, key: &K) -> Option<&Change<M>> {
        self.inner.get(key)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &Change<M>)> {
        self.inner.iter()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (K, Change<M>)> {
        std::mem::take(&mut self.inner).into_iter()
    }
}

impl<K: Ord, M: Merge> IntoIterator for Register<K, M> {
    type Item = (K, Change<M>);
    type IntoIter = std::collections::btree_map::IntoIter<K, Change<M>>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

pub type Delta = i64;

/// Rows at a multiplicity: a multiset. Pushing appends, so building one
/// costs nothing per row; [`consolidate`](Self::consolidate) sorts it and
/// merges equal rows once, which the graph does to every node's output
/// after it runs, and which reading rows requires.
pub struct Bag<T> {
    rows: Vec<(T, Delta)>,
    /// `rows[..sorted]` is sorted and consolidated.
    sorted: usize,
}

impl<T> Default for Bag<T> {
    fn default() -> Self {
        Bag {
            rows: Vec::new(),
            sorted: 0,
        }
    }
}

impl<T: Ord> Bag<T> {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn push(&mut self, data: T, delta: Delta) {
        if delta != 0 {
            self.rows.push((data, delta));
        }
    }

    /// Sort the rows and merge equal ones, dropping those whose
    /// multiplicity reaches zero. Linear for rows already in order.
    pub fn consolidate(&mut self) {
        if self.sorted == self.rows.len() {
            return;
        }
        self.rows.sort_by(|a, b| a.0.cmp(&b.0));
        if self.rows.windows(2).all(|w| w[0].0 < w[1].0) {
            self.sorted = self.rows.len();
            return;
        }
        let mut out = 0;
        for i in 0..self.rows.len() {
            if out > 0 && self.rows[out - 1].0 == self.rows[i].0 {
                let d = self.rows[i].1;
                self.rows[out - 1].1 = self.rows[out - 1]
                    .1
                    .checked_add(d)
                    .expect("multiplicity overflowed");
                if self.rows[out - 1].1 == 0 {
                    out -= 1;
                }
            } else {
                self.rows.swap(out, i);
                out += 1;
            }
        }
        self.rows.truncate(out);
        self.sorted = self.rows.len();
    }

    fn consolidated(&self) -> bool {
        self.sorted == self.rows.len()
    }

    /// The rows, in order, each with its multiplicity. The bag must be
    /// [consolidated](Self::consolidate).
    pub fn diffs(&self) -> impl Iterator<Item = (&T, Delta)> {
        debug_assert!(self.consolidated(), "consolidate before reading a bag");
        self.rows.iter().map(|(t, d)| (t, *d))
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Take the rows, in order, consolidating first; the bag keeps its
    /// allocation.
    pub fn drain(&mut self) -> impl Iterator<Item = (T, Delta)> + '_ {
        self.consolidate();
        self.sorted = 0;
        self.rows.drain(..)
    }
}

impl<T: Ord> IntoIterator for Bag<T> {
    type Item = (T, Delta);
    type IntoIter = std::vec::IntoIter<(T, Delta)>;
    /// The rows in order, consolidating first.
    fn into_iter(mut self) -> Self::IntoIter {
        self.consolidate();
        self.rows.into_iter()
    }
}

impl<T: Ord + Send + 'static> Buffer for Bag<T> {
    fn empty() -> Self {
        Self::default()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn clear(&mut self) {
        self.rows.clear();
        self.sorted = 0;
    }
    #[inline]
    fn absorb(&mut self, other: &mut Self) {
        self.rows.append(&mut other.rows);
        other.sorted = 0;
    }
    fn consolidate(&mut self) {
        Bag::consolidate(self)
    }
    fn scratch(&mut self, release: bool) -> usize {
        let release = release && self.rows.is_empty();
        vec_scratch(&mut self.rows, release)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<K: Ord + Send + 'static, M: Merge + Send> Buffer for Register<K, M>
where
    M::Value: Send,
    M::Diff: Send,
{
    fn empty() -> Self {
        Self::default()
    }
    fn len(&self) -> usize {
        BTreeMap::len(&self.inner)
    }
    fn clear(&mut self) {
        BTreeMap::clear(&mut self.inner)
    }
    #[inline]
    fn absorb(&mut self, other: &mut Self) {
        for (k, next) in other.drain() {
            self.absorb_one(k, next);
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// todo: NodeTx that can be unboxed once, holding its exact set of keyspaces as tuples?
