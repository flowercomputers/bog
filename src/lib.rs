use std::collections::BTreeMap;
use std::convert::Infallible;
use std::{any::Any, borrow::Cow, marker::PhantomData};

use crate::tx::{KeyspaceId, WriteTx};
mod consolidate;
pub mod graph;
pub mod merge;
use crate::merge::*;
mod tx;

type Delta = i64;
pub struct Diff<T>(T, Delta);

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
}

pub type Bag<T> = Vec<Diff<T>>;

/// TODO: move to register2 and bag2 everywhere
#[repr(transparent)]
struct Register2<K, M: Merge> {
    inner: BTreeMap<K, Change<M>>,
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
pub type Delta2 = i64;
pub type Bag2<T> = Register2<T, Delta2>;

// todo: make Underlay carry Change instead of ValOp?
// when flushed to the storage, we convert it INTO valops
pub enum Change<M: Merge> {
    Insert(M::Value),
    Delete(M::Value),            // old value attached
    Replace(M::Value, M::Value), // old, new
    Modify(M::Diff),             // a Merge diff, no values needed
}

pub type Register<K, M> = BTreeMap<K, Change<M>>;

impl<M: Merge> Change<M> {
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

impl<T: Send + 'static> Buffer for Bag<T> {
    fn empty() -> Self {
        Vec::new()
    }
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn clear(&mut self) {
        Vec::clear(self)
    }
    fn absorb(&mut self, other: &mut Self) {
        self.append(other)
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
        BTreeMap::new()
    }
    fn len(&self) -> usize {
        BTreeMap::len(self)
    }
    fn clear(&mut self) {
        BTreeMap::clear(self)
    }
    fn absorb(&mut self, other: &mut Self) {
        for (k, next) in std::mem::take(other) {
            match self.remove(&k) {
                None => {
                    self.insert(k, next);
                }
                Some(prev) => {
                    if let Some(c) = prev.compose(next) {
                        self.insert(k, c);
                    }
                }
            }
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
    ks: KeyspaceId,
    scratch: Register<K, (i64, V)>,
}

impl<K, V> Nary for Sum<K, V>
where
    K: Ord + Clone + Send + 'static,
    V: Merge<Diff = V, Value = V> + Clone + PartialEq + Send,
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
        // self.scratch.clear();
        // for b in batches {
        //     for Diff((k, v), d) in b.iter() {
        //         let mut diff = (1, v.clone());
        //         <(i64, V)>::scale(&mut diff, *d);
        //         self.scratch.absorb_one(k.clone(), Change::Modify(diff));
        //     }
        // }

        // // 2. resolve each Modify against the keyspace, write and emit the result
        // let ks = tx.keyspace2::<K, (i64, V)>(self.ks);
        // for (k, change) in std::mem::take(&mut self.scratch) {
        //     if let Some(resolved) = ks.resolve(&k, change) {
        //         ks.absorb(k.clone(), resolved.clone());
        //         out.insert(k, resolved);
        //     }
        // }
    }
}

//// LINEARS

pub struct FusedLinears<In, X>(X, PhantomData<fn() -> In>);
impl<In, Out, X> Nary for FusedLinears<In, X>
where
    In: Clone + 'static + Send,
    Out: Clone + 'static + Send,
    X: Emit<In, Emits = Out>,
{
    type Takes = Bag<In>;
    type Gives = Bag<Out>;

    fn eval(&mut self, _tx: &mut WriteTx, batches: &[&Bag<In>], out: &mut Bag<Out>) {
        // implicitly concatenates batches
        for batch in batches {
            for entry in *batch {
                self.0
                    .emit(Cow::Borrowed(&entry.0), entry.1, &mut |data, diff| {
                        out.push(Diff(data.into_owned(), diff))
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
