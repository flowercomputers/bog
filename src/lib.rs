use std::collections::{BTreeMap, btree_map::Entry};
use std::convert::Infallible;
use std::{any::Any, borrow::Cow, marker::PhantomData};

use crate::tx::{KeyspaceId, WriteTx};
pub mod graph;
pub mod merge;
use crate::merge::*;
mod store;
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
        if let Change::Modify(d) = &change {
            if M::noop(d) {
                return;
            }
        }
        match self.inner.entry(key) {
            Entry::Vacant(vac) => {
                vac.insert(change);
            }
            Entry::Occupied(occ) => {
                let (key, prev) = occ.remove_entry();
                if let Some(c) = prev.compose(change) {
                    if let Change::Modify(d) = &c {
                        if M::noop(d) {
                            return;
                        }
                    }
                    self.inner.insert(key, c);
                }
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
}

/// Commutative, variable-arity
pub trait Nary: Send {
    // TODO: In/Out are redundant with Takes/Gives
    type Takes: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Self::Takes], out: &mut Self::Gives);
}

pub trait Unary: Send {
    type Takes: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives);
}

impl<N: Nary> Unary for N {
    type Takes = N::Takes;
    type Gives = N::Gives;
    #[inline]
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives) {
        N::eval(self, tx, &[batch], out);
    }
}

pub trait Binary: Send {
    type Left: Buffer;
    type Right: Buffer;
    type Gives: Buffer;
    fn eval(&mut self, tx: &mut WriteTx, l: &Self::Left, r: &Self::Right, out: &mut Self::Gives);
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
    K: Ord + Clone + Send + 'static,
    V: Ord + Merge<Diff = V, Value = V> + Clone + PartialEq + Send,
{
    type Takes = Bag<(K, V)>;
    type Gives = Register<K, (i64, V)>;

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
        ks.resolve_sorted(std::mem::take(&mut self.scratch), &mut self.resolved);
        for (k, resolved) in self.resolved.drain(..) {
            // a count of zero means the key is gone
            let resolved = match resolved {
                Change::Insert(n) if n.0 == 0 => continue,
                Change::Replace(o, n) if n.0 == 0 => Change::Delete(o),
                r => r,
            };
            ks.absorb(k.clone(), resolved.clone());
            out.absorb_one(k, resolved);
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
        let store = StoreBuilder::new(DurabilityPolicy::NONE).build().unwrap();
        let mut tx = store.begin();

        let mut b = Bag::new();
        b.push((1, 10), 1);
        b.push((1, 5), 1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).as_deref(), Some(&(2, 15)));

        let mut b = Bag::new();
        b.push((1, 10), -1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).as_deref(), Some(&(1, 5)));

        let mut b = Bag::new();
        b.push((1, 5), -1);
        g.give(src, &mut b);
        g.run(&mut tx);
        let ks = tx.keyspace(TOTALS);
        assert_eq!(ks.get(&1).as_deref(), None);
        // insert then delete within one tx cancels out entirely
        assert!(ks.inner_is_empty());
    }
}
