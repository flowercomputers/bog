use std::{any::Any, borrow::Cow, marker::PhantomData};

use crate::tx::WriteTx;
mod consolidate;
pub mod graph;
pub mod merge;
use crate::merge::*;
mod tx;

type Delta = i64;
pub struct Diff<T>(T, Delta);

pub type Bag<T> = Vec<Diff<T>>;

pub enum Change<V, D> {
    Insert(V),
    Delete(V),     // old value attached
    Replace(V, V), // old, new
    Modify(D),     // a Merge diff, no values needed
}
pub type Reg<K, V, D> = Vec<(K, Change<V, D>)>; // sorted by K, one entry per key after consolidation

type Batch<T> = Vec<Diff<T>>;

// todo: NodeTx that can be unboxed once, holding its exact set of keyspaces as tuples?

pub trait Node: Send {
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any);
}

/// Commutative, variable-arity
pub trait Nary: Send {
    type In: 'static;
    type Out: 'static;
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Batch<Self::In>], out: &mut Batch<Self::Out>);
}

pub trait Unary: Send {
    type In: 'static;
    type Out: 'static;
    fn eval(&mut self, tx: &mut WriteTx, batch: &Batch<Self::In>, out: &mut Batch<Self::Out>);
}

impl<N: Nary> Unary for N {
    type In = N::In;
    type Out = N::Out;
    #[inline]
    fn eval(&mut self, tx: &mut WriteTx, batch: &Batch<Self::In>, out: &mut Batch<Self::Out>) {
        N::eval(self, tx, &[batch], out);
    }
}

pub trait Binary: Send {
    type Left: 'static;
    type Right: 'static;
    type Out: 'static;
    fn eval(
        &mut self,
        tx: &mut WriteTx,
        l: &Batch<Self::Left>,
        r: &Batch<Self::Right>,
        out: &mut Batch<Self::Out>,
    );
}

struct Sum<K, V> {
    scratch: Vec<(K, (i64, V))>,
    _p: PhantomData<fn() -> (K, V)>,
}
impl<K, V> Nary for Sum<K, V>
where
    K: Ord + Clone + 'static + Send,
    V: Merge<Diff = V, Value = V> + Clone + PartialEq + Send,
{
    type In = (K, V);
    type Out = (K, V);
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Batch<(K, V)>], out: &mut Batch<(K, V)>) {
        let ks = tx.keyspace(0);

        self.scratch.clear();
        for b in batches {
            for Diff((k, v), d) in b.iter() {
                let mut diff = (1, v.clone());
                <(i64, V)>::scale(&mut diff, *d);
                self.scratch.push((k.clone(), diff));
            }
        }
        // sort by key, fold runs with <(i64, V)>::combine
        for (k, d) in self.scratch.drain(..) {
            let old = ks.get(tx, &k).cloned().unwrap_or_else(<(i64, V)>::base);
            let mut new = old.clone();
            <(i64, V)>::merge(&mut new, std::iter::once(Cow::Borrowed(&d)));
            if new.0 == 0 {
                ks.del(tx, k.clone())
            } else {
                ks.modify(tx, k.clone(), d)
            }
            // emit (k, old.1) at -1 if old.0 != 0, (k, new.1) at +1 if new.0 != 0,
            // skipping both when the total did not change
        }
    }
}

//// LINEARS

pub struct FusedLinears<In, X>(X, PhantomData<fn() -> In>);
impl<In, Out, X> Nary for FusedLinears<In, X>
where
    In: Clone + 'static,
    Out: Clone + 'static,
    X: Emit<In, Emits = Out>,
{
    type In = In;
    type Out = Out;
    fn eval(&mut self, _tx: &mut WriteTx, batches: &[&Batch<In>], out: &mut Batch<Out>) {
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
