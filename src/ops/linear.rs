//! Linear operators over rows: map, filter and their kin, keying, and
//! per-value variants over keyed rows `(K, V)`. They are [`Emit`]
//! stages, fused into one node by [`FusedLinears`](crate::FusedLinears).
//! [`Flatten`] turns a register's changes into rows.

use std::{borrow::Cow, marker::PhantomData};

use crate::{Bag, Delta, Emit, Register, Unary, merge::Merge, tx::WriteTx};

/// Keys each row by `f`: `t` becomes `(f(&t), t)`.
#[derive(Clone)]
pub struct KeyBy<F>(pub F);
impl<T, K, F> Emit<T> for KeyBy<F>
where
    T: Clone + 'static,
    K: Clone + 'static,
    F: Fn(&T) -> K + Send + 'static,
{
    type Emits = (K, T);
    #[inline]
    fn emit<S: FnMut(Cow<(K, T)>, Delta)>(&self, data: Cow<T>, diff: Delta, sink: &mut S) {
        let k = (self.0)(&data);
        sink(Cow::Owned((k, data.into_owned())), diff);
    }
}

/// Maps each value, keeping its key.
#[derive(Clone)]
pub struct MapVal<F>(pub F);
impl<K, V, W, F> Emit<(K, V)> for MapVal<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    W: Clone + 'static,
    F: Fn(&V) -> W + Send + 'static,
{
    type Emits = (K, W);
    #[inline]
    fn emit<S: FnMut(Cow<(K, W)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        let w = (self.0)(&data.1);
        sink(Cow::Owned((key(data), w)), diff);
    }
}

/// Maps each value with its key, keeping the key.
#[derive(Clone)]
pub struct MapWith<F>(pub F);
impl<K, V, W, F> Emit<(K, V)> for MapWith<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    W: Clone + 'static,
    F: Fn(&K, &V) -> W + Send + 'static,
{
    type Emits = (K, W);
    #[inline]
    fn emit<S: FnMut(Cow<(K, W)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        let w = (self.0)(&data.0, &data.1);
        sink(Cow::Owned((key(data), w)), diff);
    }
}

/// Keeps the rows `f` accepts, given key and value apart.
#[derive(Clone)]
pub struct FilterKV<F>(pub F);
impl<K, V, F> Emit<(K, V)> for FilterKV<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    F: Fn(&K, &V) -> bool + Send + 'static,
{
    type Emits = (K, V);
    #[inline]
    fn emit<S: FnMut(Cow<(K, V)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        if (self.0)(&data.0, &data.1) {
            sink(data, diff);
        }
    }
}

/// Maps each value to any number of values, each keeping the key. With
/// an `Option`, a filter-map over values.
#[derive(Clone)]
pub struct FlatMapVal<F>(pub F);
impl<K, V, W, I, F> Emit<(K, V)> for FlatMapVal<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    W: Clone + 'static,
    I: IntoIterator<Item = W>,
    F: Fn(&V) -> I + Send + 'static,
{
    type Emits = (K, W);
    #[inline]
    fn emit<S: FnMut(Cow<(K, W)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        for w in (self.0)(&data.1) {
            sink(Cow::Owned((data.0.clone(), w)), diff);
        }
    }
}

/// Maps each value to weighted values, each keeping the key, at the
/// row's multiplicity times its weight.
#[derive(Clone)]
pub struct ExplodeVal<F>(pub F);
impl<K, V, W, I, F> Emit<(K, V)> for ExplodeVal<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    W: Clone + 'static,
    I: IntoIterator<Item = (W, Delta)>,
    F: Fn(&V) -> I + Send + 'static,
{
    type Emits = (K, W);
    #[inline]
    fn emit<S: FnMut(Cow<(K, W)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        for (w, weight) in (self.0)(&data.1) {
            let d = diff
                .checked_mul(weight)
                .expect("explode multiplication overflowed");
            if d != 0 {
                sink(Cow::Owned((data.0.clone(), w)), d);
            }
        }
    }
}

/// Re-keys each row by its value: `(k, v)` becomes `(f(&v), v)`.
#[derive(Clone)]
pub struct ReKey<F>(pub F);
impl<K, V, K2, F> Emit<(K, V)> for ReKey<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    K2: Clone + 'static,
    F: Fn(&V) -> K2 + Send + 'static,
{
    type Emits = (K2, V);
    #[inline]
    fn emit<S: FnMut(Cow<(K2, V)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        let k2 = (self.0)(&data.1);
        sink(Cow::Owned((k2, value(data))), diff);
    }
}

/// Re-keys each row by its key and value: `(k, v)` becomes
/// `(f(&k, &v), v)`.
#[derive(Clone)]
pub struct ReKeyWith<F>(pub F);
impl<K, V, K2, F> Emit<(K, V)> for ReKeyWith<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    K2: Clone + 'static,
    F: Fn(&K, &V) -> K2 + Send + 'static,
{
    type Emits = (K2, V);
    #[inline]
    fn emit<S: FnMut(Cow<(K2, V)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        let k2 = (self.0)(&data.0, &data.1);
        sink(Cow::Owned((k2, value(data))), diff);
    }
}

/// Puts every row under the one key `()`, for reductions over the whole
/// collection.
#[derive(Clone, Copy)]
pub struct Unkey;
impl<K, V> Emit<(K, V)> for Unkey
where
    K: Clone + 'static,
    V: Clone + 'static,
{
    type Emits = ((), V);
    #[inline]
    fn emit<S: FnMut(Cow<((), V)>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        sink(Cow::Owned(((), value(data))), diff);
    }
}

/// Orders rows by value: `(k, v)` becomes `((v, k), ())`.
#[derive(Clone, Copy)]
pub struct Order;
impl<K, V> Emit<(K, V)> for Order
where
    K: Clone + 'static,
    V: Clone + 'static,
{
    type Emits = ((V, K), ());
    #[inline]
    fn emit<S: FnMut(Cow<((V, K), ())>, Delta)>(
        &self,
        data: Cow<(K, V)>,
        diff: Delta,
        sink: &mut S,
    ) {
        let (k, v) = data.into_owned();
        sink(Cow::Owned(((v, k), ())), diff);
    }
}

/// Prefixes each key with `f` of the row: `(k, v)` becomes
/// `((f(&k, &v), k), v)`, to read rows in another order.
#[derive(Clone)]
pub struct Reindex<F>(pub F);
impl<K, V, X, F> Emit<(K, V)> for Reindex<F>
where
    K: Clone + 'static,
    V: Clone + 'static,
    X: Clone + 'static,
    F: Fn(&K, &V) -> X + Send + 'static,
{
    type Emits = ((X, K), V);
    #[inline]
    fn emit<S: FnMut(Cow<((X, K), V)>, Delta)>(
        &self,
        data: Cow<(K, V)>,
        diff: Delta,
        sink: &mut S,
    ) {
        let x = (self.0)(&data.0, &data.1);
        let (k, v) = data.into_owned();
        sink(Cow::Owned(((x, k), v)), diff);
    }
}

/// The key of each row.
#[derive(Clone, Copy)]
pub struct Keys;
impl<K, V> Emit<(K, V)> for Keys
where
    K: Clone + 'static,
    V: Clone + 'static,
{
    type Emits = K;
    #[inline]
    fn emit<S: FnMut(Cow<K>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        sink(Cow::Owned(key(data)), diff);
    }
}

/// The value of each row.
#[derive(Clone, Copy)]
pub struct Values;
impl<K, V> Emit<(K, V)> for Values
where
    K: Clone + 'static,
    V: Clone + 'static,
{
    type Emits = V;
    #[inline]
    fn emit<S: FnMut(Cow<V>, Delta)>(&self, data: Cow<(K, V)>, diff: Delta, sink: &mut S) {
        sink(Cow::Owned(value(data)), diff);
    }
}

/// A row's key, moved out when the row is owned.
fn key<K: Clone, V: Clone>(row: Cow<(K, V)>) -> K {
    match row {
        Cow::Owned((k, _)) => k,
        Cow::Borrowed((k, _)) => k.clone(),
    }
}

/// A row's value, moved out when the row is owned.
fn value<K: Clone, V: Clone>(row: Cow<(K, V)>) -> V {
    match row {
        Cow::Owned((_, v)) => v,
        Cow::Borrowed((_, v)) => v.clone(),
    }
}

#[derive(Clone)]
pub struct Filter<F>(pub F);
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

#[derive(Clone)]
pub struct Map<F>(pub F);
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

#[derive(Clone)]
pub struct FilterMap<F>(pub F);
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

#[derive(Clone)]
pub struct FlatMap<F>(pub F);
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

#[derive(Clone)]
pub struct Explode<F>(pub F);
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

#[derive(Clone, Copy)]
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

/// A register's changes as rows: an insert is the new row at `+1`, a
/// delete the old row at `-1`, a replace both. Turns a register, such as
/// a table or [`Sum`](crate::Sum)'s output, into rows for the linear
/// operators and [`Arrange`](super::Arrange).
pub struct Flatten<K, M>(PhantomData<fn() -> (K, M)>);

impl<K, M> Default for Flatten<K, M> {
    fn default() -> Self {
        Flatten(PhantomData)
    }
}

impl<K, M> Unary for Flatten<K, M>
where
    K: Ord + Clone + Send + 'static,
    M: Merge + Send,
    M::Value: Ord + Send + 'static,
    M::Diff: Send,
{
    type Takes = Register<K, M>;
    type Gives = Bag<(K, M::Value)>;
    type State = ();
    fn state(&self) {}
    fn eval(&mut self, _: &mut WriteTx, batch: &Register<K, M>, out: &mut Self::Gives) {
        for (k, change) in batch.iter() {
            if let Some(o) = change.old_value() {
                out.push((k.clone(), o.clone()), -1);
            }
            if let Some(n) = change.new_value() {
                out.push((k.clone(), n.clone()), 1);
            }
        }
    }
}
