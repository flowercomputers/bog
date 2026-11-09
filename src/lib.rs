use std::collections::{BTreeMap, btree_map::Entry};
use std::convert::Infallible;
use std::{any::Any, borrow::Cow, marker::PhantomData};

use crate::codec::{Codec, OrderedCodec};
use crate::graph::Fingerprint;
use crate::tx::{KeyspaceId, WriteTx};
pub mod graph;
pub mod merge;
use crate::merge::*;
mod codec;
mod cursor;
pub mod error;

mod manifest;
mod store;
mod table;
mod tx;

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
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
        if let Change::Modify(d) = &change {
            if M::noop(d) {
                return 0;
            }
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
                if let Change::Modify(d) = &c {
                    if M::noop(d) {
                        return -before;
                    }
                }
                let after = size(&key, &c) as isize;
                self.inner.insert(key, c);
                after - before
            }
        }
    }
    pub(crate) fn map(&self) -> &BTreeMap<K, Change<M>> {
        &self.inner
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

/// Bag-specific helpers: every entry is a `Modify(delta)`.
impl<T: Ord> Register<T, Delta> {
    pub fn push(&mut self, data: T, delta: Delta) {
        self.absorb_one(data, Change::Modify(delta));
    }
    pub fn diffs(&self) -> impl Iterator<Item = (&T, Delta)> {
        self.inner.iter().map(|(k, c)| match c {
            Change::Modify(d) => (k, *d),
            _ => unreachable!("bags only hold Modify"),
        })
    }
}

// an opaque type without merges (emitted by tables, etc)
#[repr(transparent)]
pub struct Mergeless<T>(T);
impl<T: 'static + Send + Clone> Merge for Mergeless<T> {
    type Diff = Infallible;
    type Value = T;
    fn base() -> Self::Value {
        unreachable!("")
    }
    fn merge<'a>(val: &mut Self::Value, diffs: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        unreachable!("")
    }
    fn combine<'a>(old: &mut Self::Diff, new: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        unreachable!("")
    }
    fn scale(diff: &mut Self::Diff, times: Delta) {
        unreachable!("")
    }
}
pub type Delta = i64;
pub type Bag<T> = Register<T, Delta>;

// todo: make Underlay carry Change instead of ValOp?
// when flushed to the storage, we convert it INTO valops
pub enum Change<M: Merge> {
    Insert(M::Value),
    Delete(M::Value),            // old value attached
    Replace(M::Value, M::Value), // old, new
    Modify(M::Diff),             // a Merge diff, no values needed
}
impl<M: Merge> Clone for Change<M> {
    fn clone(&self) -> Self {
        use Change::*;
        match self {
            Insert(v) => Insert(v.clone()),
            Delete(v) => Delete(v.clone()),
            Replace(o, n) => Replace(o.clone(), n.clone()),
            Modify(d) => Modify(d.clone()),
        }
    }
}

impl<M: Merge> Change<M> {
    /// The value at this key after the change, if any.
    pub fn new_value(&self) -> Option<&M::Value> {
        match self {
            Change::Insert(v) | Change::Replace(_, v) => Some(v),
            Change::Delete(_) => None,
            Change::Modify(_) => unreachable!("Modify has no materialized value"),
        }
    }
    pub fn compose(self, next: Self) -> Option<Self> {
        use Change::*;
        Some(match (self, next) {
            (Insert(_), Delete(_)) => return None,
            (Insert(_), Insert(b)) | (Insert(_), Replace(_, b)) => Insert(b),
            (Insert(mut a), Modify(d)) => {
                M::merge(&mut a, std::iter::once(Cow::Owned(d)));
                Insert(a)
            }
            (Delete(o), Insert(b)) | (Delete(o), Replace(_, b)) => Replace(o, b),
            (Delete(o), Delete(_)) => Delete(o),
            (Delete(o), Modify(d)) => {
                let mut v = M::base();
                M::merge(&mut v, std::iter::once(Cow::Owned(d)));
                Replace(o, v)
            }
            (Replace(o, _), Insert(b)) | (Replace(o, _), Replace(_, b)) => Replace(o, b),
            (Replace(o, _), Delete(_)) => Delete(o),
            (Replace(o, mut a), Modify(d)) => {
                M::merge(&mut a, std::iter::once(Cow::Owned(d)));
                Replace(o, a)
            }
            (Modify(mut a), Modify(b)) => {
                M::combine(&mut a, std::iter::once(Cow::Owned(b)));
                Modify(a)
            }
            (Modify(_), _) => unreachable!("change register into delta register"),
        })
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

pub trait Node: Send {
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any);
    fn fingerprint(&self, fp: &mut Fingerprint);
}

/// Commutative, variable-arity
pub trait Nary: Send {
    // TODO: In/Out are redundant with Takes/Gives
    type Takes: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Self::Takes], out: &mut Self::Gives);
    /// Add whatever the operator's type does not already say, such as
    /// the keyspaces it writes, to the graph's fingerprint.
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
}

pub trait Unary: Send {
    type Takes: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives);
    /// See [`Nary::fingerprint`].
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
}

impl<N: Nary> Unary for N {
    type Takes = N::Takes;
    type Gives = N::Gives;
    #[inline]
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives) {
        N::eval(self, tx, &[batch], out);
    }
    fn fingerprint(&self, fp: &mut Fingerprint) {
        N::fingerprint(self, fp)
    }
}

pub trait Binary: Send {
    type Left: Buffer;
    type Right: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, l: &Self::Left, r: &Self::Right, out: &mut Self::Gives);
    /// See [`Nary::fingerprint`].
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
}

pub struct Sum<K, V: Merge> {
    ks: KeyspaceId<K, (i64, V)>,
    scratch: Register<K, (i64, V)>,
    resolved: Vec<(K, Change<(i64, V)>)>,
}

impl<K: Ord, V: Merge> Sum<K, V> {
    pub fn new(ks: KeyspaceId<K, (i64, V)>) -> Self {
        Sum {
            ks,
            scratch: Register::new(),
            resolved: Vec::new(),
        }
    }
}

impl<K, V> Nary for Sum<K, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    V: Ord + Merge<Diff = V, Value = V> + Codec + Clone + PartialEq + Send + Sync,
{
    type Takes = Bag<(K, V)>;
    type Gives = Register<K, (i64, V)>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.ks.raw());
    }

    fn eval(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(K, V)>],
        out: &mut Register<K, (i64, V)>,
    ) {
        // 1. every row becomes a Modify; equal keys combine on the way in
        self.scratch.clear();
        for b in batches {
            for ((k, v), d) in b.diffs() {
                let mut diff = (1, v.clone());
                <(i64, V)>::scale(&mut diff, d);
                self.scratch.absorb_one(k.clone(), Change::Modify(diff));
            }
        }

        // 2. one sorted sweep resolves every Modify against the keyspace
        let mut ks = tx.keyspace(self.ks);
        if ks
            .resolve_sorted(std::mem::take(&mut self.scratch), &mut self.resolved)
            .is_err()
        {
            // the tx failed, and with it this epoch; the graph stops here
            self.resolved.clear();
            return;
        }
        for (k, resolved) in self.resolved.drain(..) {
            // a count of zero means the key is gone
            let resolved = match resolved {
                Change::Insert(n) if n.0 == 0 => continue,
                Change::Replace(o, n) if n.0 == 0 => Change::Delete(o),
                r => r,
            };
            if ks.absorb(k.clone(), resolved.clone()).is_err() {
                // the tx failed; the graph stops at this node and the
                // commit will be refused (the drain empties `resolved`)
                return;
            }
            out.absorb_one(k, resolved);
        }
    }
}

/// Counts each row's multiplicity in `counts` and emits a row when its
/// presence changes: `+1` when its count becomes positive, `-1` when it
/// stops being positive. Downstream sees each row at most once.
pub struct Distinct<T> {
    counts: KeyspaceId<T, Delta>,
    scratch: Bag<T>,
    resolved: Vec<(T, Change<Delta>)>,
}

impl<T: Ord> Distinct<T> {
    pub fn new(counts: KeyspaceId<T, Delta>) -> Self {
        Distinct {
            counts,
            scratch: Bag::new(),
            resolved: Vec::new(),
        }
    }
}

impl<T> Nary for Distinct<T>
where
    T: OrderedCodec + Clone + Send + Sync + 'static,
{
    type Takes = Bag<T>;
    type Gives = Bag<T>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.counts.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<T>], out: &mut Bag<T>) {
        self.scratch.clear();
        for b in batches {
            for (t, d) in b.diffs() {
                self.scratch.push(t.clone(), d);
            }
        }
        let mut ks = tx.keyspace(self.counts);
        if ks
            .resolve_sorted(std::mem::take(&mut self.scratch), &mut self.resolved)
            .is_err()
        {
            self.resolved.clear();
            return;
        }
        for (t, change) in self.resolved.drain(..) {
            let (old, new) = match &change {
                Change::Insert(n) => (0, *n),
                Change::Delete(o) => (*o, 0),
                Change::Replace(o, n) => (*o, *n),
                Change::Modify(_) => unreachable!("resolved"),
            };
            // a count of zero means the row is gone
            let change = match change {
                Change::Insert(0) => continue,
                Change::Replace(o, 0) => Change::Delete(o),
                c => c,
            };
            if ks.absorb(t.clone(), change).is_err() {
                return;
            }
            match (old > 0, new > 0) {
                (false, true) => out.push(t, 1),
                (true, false) => out.push(t, -1),
                _ => {}
            }
        }
    }
}

/// Rejects the tx when a value its input view is left holding fails
/// `check`. Only the values an epoch ends with are checked, so a tx that
/// passes through a bad state and recovers within itself is accepted.
pub struct Check<K, M, F>(F, PhantomData<fn() -> (K, M)>);

impl<K, M, F> Check<K, M, F> {
    pub fn new(check: F) -> Self {
        Check(check, PhantomData)
    }
}

impl<K, M, F, E> Nary for Check<K, M, F>
where
    K: Ord + Send + 'static,
    M: Merge + Send,
    M::Value: Send,
    M::Diff: Send,
    F: Fn(&K, &M::Value) -> Result<(), E> + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    type Takes = Register<K, M>;
    type Gives = Bag<()>;

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Register<K, M>], _out: &mut Bag<()>) {
        for b in batches {
            for (k, change) in b.iter() {
                if let Some(v) = change.new_value()
                    && let Err(e) = (self.0)(k, v)
                {
                    tx.reject(e);
                    return;
                }
            }
        }
    }
}

/// Maintains `index`, a keyspace holding `key(k, v)` for every live
/// `(k, v)` of its input view for which `key` returns one, and emits the
/// index's changes. `key` must not map two live entries to one index key.
///
/// An index ordered differently from its source is how a view is read in
/// another order: a leaderboard is balances keyed by `(-balance, user)`.
pub struct IndexBy<K, M, K2, F> {
    index: KeyspaceId<K2, ()>,
    key: F,
    _m: PhantomData<fn() -> (K, M)>,
}

impl<K, M: Merge, K2, F> IndexBy<K, M, K2, F> {
    pub fn new(index: KeyspaceId<K2, ()>, key: F) -> Self {
        IndexBy {
            index,
            key,
            _m: PhantomData,
        }
    }
}

impl<K, M, K2, F> Nary for IndexBy<K, M, K2, F>
where
    K: Ord + Send + 'static,
    M: Merge + Send,
    M::Value: Send,
    M::Diff: Send,
    K2: OrderedCodec + Clone + Send + Sync + 'static,
    F: Fn(&K, &M::Value) -> Option<K2> + Send,
{
    type Takes = Register<K, M>;
    type Gives = Bag<K2>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.index.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Register<K, M>], out: &mut Bag<K2>) {
        let mut ix = tx.keyspace(self.index);
        for b in batches {
            for (k, change) in b.iter() {
                let (old, new) = match change {
                    Change::Insert(v) => (None, Some(v)),
                    Change::Delete(o) => (Some(o), None),
                    Change::Replace(o, n) => (Some(o), Some(n)),
                    Change::Modify(_) => unreachable!("views hold materialized changes"),
                };
                let old = old.and_then(|v| (self.key)(k, v));
                let new = new.and_then(|v| (self.key)(k, v));
                if old == new {
                    continue;
                }
                if let Some(o) = old {
                    if ix.absorb(o.clone(), Change::Delete(())).is_err() {
                        return;
                    }
                    out.push(o, -1);
                }
                if let Some(n) = new {
                    if ix.absorb(n.clone(), Change::Insert(())).is_err() {
                        return;
                    }
                    out.push(n, 1);
                }
            }
        }
    }
}

//// LINEARS

pub struct FusedLinears<In, X>(X, PhantomData<fn() -> In>);
impl<In, Out, X> Nary for FusedLinears<In, X>
where
    In: Clone + 'static + Send + Ord,
    Out: Clone + 'static + Send + Ord,
    X: Emit<In, Emits = Out>,
{
    type Takes = Bag<In>;
    type Gives = Bag<Out>;

    fn eval(&mut self, _tx: &mut WriteTx, batches: &[&Bag<In>], out: &mut Bag<Out>) {
        // implicitly concatenates batches
        for batch in batches {
            for (data, diff) in batch.diffs() {
                self.0.emit(Cow::Borrowed(data), diff, &mut |data, diff| {
                    out.push(data.into_owned(), diff)
                });
            }
        }
    }
}

pub trait Emit<In: Clone>: 'static + Send {
    type Emits: Clone + 'static;
    fn emit<S: FnMut(Cow<Self::Emits>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S);
}

pub struct Then<A, B>(A, B);
impl<In, Mid, Out, A, B> Emit<In> for Then<A, B>
where
    In: Clone + 'static,
    Mid: Clone + 'static,
    Out: Clone + 'static,
    A: Emit<In, Emits = Mid>,
    B: Emit<Mid, Emits = Out>,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        self.0
            .emit(data, diff, &mut |mid, d| self.1.emit(mid, d, &mut *sink));
    }
}

pub struct Filter<F>(F);
impl<In, F> Emit<In> for Filter<F>
where
    In: Clone + 'static,
    F: Fn(&In) -> bool + 'static + Send,
{
    type Emits = In;
    #[inline]
    fn emit<S: FnMut(Cow<In>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        if (self.0)(&data) {
            sink(data, diff);
        }
    }
}

pub struct Map<F>(F);
impl<In, Out, F> Emit<In> for Map<F>
where
    In: Clone + 'static,
    Out: Clone + 'static,
    F: Fn(&In) -> Out + 'static + Send,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        sink(Cow::Owned((self.0)(&data)), diff);
    }
}

pub struct FilterMap<F>(F);
impl<In, Out, F> Emit<In> for FilterMap<F>
where
    In: Clone + 'static,
    Out: Clone + 'static,
    F: Fn(&In) -> Option<Out> + 'static + Send,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        if let Some(r) = (self.0)(&data) {
            sink(Cow::Owned(r), diff);
        }
    }
}

pub struct FlatMap<F>(F);
impl<In, Out, I, F> Emit<In> for FlatMap<F>
where
    In: Clone + 'static,
    Out: Clone + 'static,
    I: IntoIterator<Item = Out> + 'static,
    F: Fn(&In) -> I + 'static + Send,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        for out in (self.0)(&data) {
            sink(Cow::Owned(out), diff);
        }
    }
}

pub struct Explode<F>(F);
impl<In, Out, I, F> Emit<In> for Explode<F>
where
    In: Clone + 'static,
    Out: Clone + 'static,
    I: IntoIterator<Item = (Out, i64)> + 'static,
    F: Fn(&In) -> I + 'static + Send,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        for (out, d) in (self.0)(&data) {
            let d = d
                .checked_mul(diff)
                .expect("explode multiplication overflowed");
            if d != 0 {
                sink(Cow::Owned(out), d);
            }
        }
    }
}

pub struct Negate;
impl<In> Emit<In> for Negate
where
    In: Clone + 'static,
{
    type Emits = In;
    #[inline]
    fn emit<S: FnMut(Cow<In>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        sink(data, diff.checked_neg().expect("negate overflowed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpDag;
    use crate::store::{DurabilityPolicy, StoreBuilder};

    #[test]
    fn sum_inserts_then_replaces_then_deletes() {
        let mut g = OpDag::new();
        let src = g.source::<Bag<(u8, i64)>>();
        const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);
        let sum = g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
        let _ = sum;
        let store = StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(TOTALS)
            .build()
            .unwrap();
        let mut tx = store.begin().unwrap();

        let mut b = Bag::new();
        b.push((1, 10), 1);
        b.push((1, 5), 1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&(2, 15)));

        let mut b = Bag::new();
        b.push((1, 10), -1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&(1, 5)));

        let mut b = Bag::new();
        b.push((1, 5), -1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).unwrap().as_deref(), None);
        // insert then delete within one tx cancels out entirely
        assert!(ks.inner_is_empty());
    }

    mod rebuild {
        use super::super::*;
        use crate::error::Error;
        use crate::graph::{NodeRef, OpDag};
        use crate::store::{DurabilityPolicy, Store, StoreBuilder};
        use std::{collections::BTreeMap, ops::Bound, path::Path};

        const BASE: KeyspaceId<u8, i64> = KeyspaceId::new(1);
        const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);

        /// Base rows `k -> v`, summed into TOTALS by `k % modulus`.
        struct App {
            g: OpDag,
            src: NodeRef<Bag<(u8, i64)>>,
            modulus: u8,
        }

        impl App {
            fn new(modulus: u8) -> Self {
                let mut g = OpDag::new();
                let src = g.source::<Bag<(u8, i64)>>();
                let m = modulus;
                let class = g.nary(
                    &[src],
                    FusedLinears(Map(move |&(k, v): &(u8, i64)| (k % m, v)), PhantomData),
                );
                g.nary(&[class], Sum::<u8, i64>::new(TOTALS));
                App { g, src, modulus }
            }

            fn fingerprint(&self) -> u64 {
                let mut fp = crate::graph::Fingerprint::new();
                fp.write_u64(self.g.fingerprint());
                fp.write_u64(self.modulus as u64);
                fp.finish()
            }

            fn builder(&self, dir: &Path) -> StoreBuilder {
                StoreBuilder::new(DurabilityPolicy::RELAXED)
                    .dir(dir)
                    .keyspace(BASE)
                    .derived(TOTALS)
                    .graph(self.fingerprint())
            }

            /// Set base rows, feeding their changes to the graph.
            fn write(&mut self, store: &Store, rows: &[(u8, Option<i64>)]) {
                let mut tx = store.begin().unwrap();
                let mut feed = Bag::new();
                {
                    let mut ks = tx.keyspace(BASE);
                    for &(k, v) in rows {
                        let old = ks.get(&k).unwrap().map(|v| *v);
                        if let Some(o) = old {
                            feed.push((k, o), -1);
                        }
                        if let Some(n) = v {
                            feed.push((k, n), 1);
                        }
                        let change = match (old, v) {
                            (None, None) => continue,
                            (None, Some(n)) => Change::Insert(n),
                            (Some(o), None) => Change::Delete(o),
                            (Some(o), Some(n)) => Change::Replace(o, n),
                        };
                        ks.absorb(k, change).unwrap();
                    }
                }
                self.g.give(self.src, &mut feed);
                self.g.run(&mut tx);
                store.finish(tx).unwrap();
            }

            /// Feed base rows to the graph `chunk` at a time.
            fn replay(
                &mut self,
                chunk: usize,
                fail_after: Option<usize>,
            ) -> impl FnMut(&mut WriteTx<'_>) -> crate::error::Result<bool> + '_ {
                let mut after: Option<u8> = None;
                let mut calls = 0;
                move |tx| {
                    calls += 1;
                    let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                    let rows: Vec<(u8, i64)> = tx
                        .keyspace(BASE)
                        .range((lower, Bound::Unbounded))
                        .take(chunk)
                        .map(|r| r.map(|(k, v)| (*k, *v)))
                        .collect::<crate::error::Result<_>>()?;
                    let mut feed = Bag::new();
                    for &(k, v) in &rows {
                        feed.push((k, v), 1);
                    }
                    after = rows.last().map(|r| r.0).or(after);
                    self.g.give(self.src, &mut feed);
                    self.g.run(tx);
                    if fail_after == Some(calls) {
                        return Err(Error::InvalidOption("replay failed on purpose"));
                    }
                    Ok(rows.len() == chunk)
                }
            }
        }

        fn totals(store: &Store) -> Vec<(u8, (i64, i64))> {
            store
                .snapshot()
                .range(TOTALS, ..)
                .map(|r| r.map(|(k, v)| (*k, *v)))
                .collect::<crate::error::Result<_>>()
                .unwrap()
        }

        fn expected(model: &BTreeMap<u8, i64>, modulus: u8) -> Vec<(u8, (i64, i64))> {
            let mut t: BTreeMap<u8, (i64, i64)> = BTreeMap::new();
            for (&k, &v) in model {
                let e = t.entry(k % modulus).or_default();
                e.0 += 1;
                e.1 += v;
            }
            t.into_iter().collect()
        }

        fn tmpdir(name: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!("fold3-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            dir
        }

        /// Base rows spread over checkpoints, the WAL, and deletes.
        fn populate(app: &mut App, store: &Store, model: &mut BTreeMap<u8, i64>) {
            for round in 0..4i64 {
                let rows: Vec<(u8, Option<i64>)> = (0..20u8)
                    .map(|k| {
                        let v = (k as i64 * 7 + round) % 5 != 0;
                        (k, v.then_some(k as i64 * 10 + round))
                    })
                    .collect();
                for &(k, v) in &rows {
                    match v {
                        Some(v) => model.insert(k, v),
                        None => model.remove(&k),
                    };
                }
                app.write(store, &rows);
                if round % 2 == 0 {
                    store.checkpoint().unwrap();
                }
            }
        }

        #[test]
        fn a_changed_graph_truncates_and_replays_derived_state() {
            let dir = tmpdir("rebuild");
            let mut model = BTreeMap::new();
            let mut v1 = App::new(2);
            let store = v1.builder(&dir).build().unwrap();
            populate(&mut v1, &store, &mut model);
            assert_eq!(totals(&store), expected(&model, 2));
            drop(store);

            // the same graph reopens without a rebuild
            let store = v1.builder(&dir).build().unwrap();
            assert_eq!(totals(&store), expected(&model, 2));
            drop(store);

            // a different graph cannot open without a replay
            let mut v2 = App::new(3);
            assert!(matches!(
                v2.builder(&dir).build(),
                Err(Error::RebuildRequired)
            ));

            // a failed replay leaves the rebuild to the next open
            let b = v2.builder(&dir);
            assert!(b.build_with(v2.replay(4, Some(2))).is_err());
            let mut v2 = App::new(3);
            assert!(matches!(
                v2.builder(&dir).build(),
                Err(Error::RebuildRequired)
            ));

            let b = v2.builder(&dir);
            let store = b.build_with(v2.replay(3, None)).unwrap();
            assert_eq!(totals(&store), expected(&model, 3));

            // the new graph keeps running incrementally over rebuilt state
            v2.write(&store, &[(5, None), (40, Some(1))]);
            model.remove(&5);
            model.insert(40, 1);
            assert_eq!(totals(&store), expected(&model, 3));
            drop(store);

            let store = v2.builder(&dir).build().unwrap();
            assert_eq!(totals(&store), expected(&model, 3));

            // compaction leaves truncated data behind, and once every table
            // is newer than the floor, the floor goes too
            store.checkpoint().unwrap();
            store.compact().unwrap();
            assert_eq!(totals(&store), expected(&model, 3));
            drop(store);
            let m = std::fs::read_to_string(dir.join("MANIFEST")).unwrap();
            assert!(!m.contains("floor\t"), "floor outlived its data:\n{m}");
            let store = v2.builder(&dir).build().unwrap();
            assert_eq!(totals(&store), expected(&model, 3));
        }

        #[test]
        fn an_undeclared_derived_keyspace_is_dropped() {
            let dir = tmpdir("drop-derived");
            let mut model = BTreeMap::new();
            let mut app = App::new(2);
            let store = app.builder(&dir).build().unwrap();
            populate(&mut app, &store, &mut model);
            drop(store);

            let base_only = || {
                StoreBuilder::new(DurabilityPolicy::RELAXED)
                    .dir(&dir)
                    .keyspace(BASE)
            };
            // no derived keyspace remains, so there is nothing to replay
            let store = base_only().build().unwrap();
            assert_eq!(store.snapshot().range(BASE, ..).count(), model.len());
            drop(store);
            let m = std::fs::read_to_string(dir.join("MANIFEST")).unwrap();
            assert!(!m.contains("derived"), "{m}");
            assert!(base_only().build().is_ok());

            // bringing it back rebuilds it from scratch
            let mut app = App::new(2);
            let b = app.builder(&dir);
            let store = b.build_with(app.replay(64, None)).unwrap();
            assert_eq!(totals(&store), expected(&model, 2));
        }

        #[test]
        fn graph_fingerprints_see_shape_and_keyspaces() {
            assert_eq!(App::new(2).g.fingerprint(), App::new(2).g.fingerprint());
            let mut g = OpDag::new();
            let src = g.source::<Bag<(u8, i64)>>();
            g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
            let mut h = OpDag::new();
            let src = h.source::<Bag<(u8, i64)>>();
            h.nary(&[src], Sum::<u8, i64>::new(KeyspaceId::new(8)));
            assert_ne!(g.fingerprint(), h.fingerprint());
            assert_ne!(g.fingerprint(), App::new(2).g.fingerprint());
        }
    }

    #[test]
    fn a_killed_tx_stops_the_graph_and_commits_nothing() {
        let mut g = OpDag::new();
        let src = g.source::<Bag<(u8, i64)>>();
        const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);
        let _sum = g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
        let store = StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(TOTALS)
            .tx_memory_limit(2048)
            .build()
            .unwrap();

        let mut b = Bag::new();
        for k in 0..=255u8 {
            b.push((k, 1), 1);
        }
        g.give(src, &mut b);
        let mut tx = store.begin().unwrap();
        g.run(&mut tx);
        assert!(tx.failure().is_some());
        assert!(store.finish(tx).is_err());
        assert_eq!(store.snapshot().get(TOTALS, &0).unwrap(), None);

        // the graph is clean for the next epoch
        let mut b = Bag::new();
        b.push((3, 5), 1);
        g.give(src, &mut b);
        let mut tx = store.begin().unwrap();
        g.run(&mut tx);
        assert_eq!(store.finish(tx).unwrap(), Some(1));
        assert_eq!(
            store.snapshot().get(TOTALS, &3).unwrap().as_deref(),
            Some(&(1, 5))
        );
    }
}
