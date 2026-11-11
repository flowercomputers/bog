//! Linear operators over rows: map, filter and their kin, keying, and
//! per-value variants over keyed rows `(K, V)`. They are [`Emit`]
//! stages, fused into one node by [`FusedLinears`](crate::dataflow::FusedLinears).
//! [`Flatten`] turns a register's changes into rows.

use std::{borrow::Cow, marker::PhantomData};

use crate::stream::prelude::*;
use crate::{
    dataflow::{Bag, Delta, Emit, Register, Unary},
    store::merge::Merge,
    store::txn::WriteTx,
};

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
/// a table or [`Sum`]'s output, into rows for the linear
/// operators and [`Arrange`].
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

// The stream methods: how a schema declares this operator.

impl<T: Data, C: Chain<Out = T>> Stream<T, C> {
    pub fn map<U: Data, F>(&self, f: F) -> Stream<U, Ext<C, Map<F>>>
    where
        F: Fn(&T) -> U + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Map(f)))
    }

    pub fn filter<P>(&self, p: P) -> Stream<T, Ext<C, Filter<P>>>
    where
        P: Fn(&T) -> bool + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Filter(p)))
    }

    pub fn flat_map<I, F>(&self, f: F) -> Stream<I::Item, Ext<C, FlatMap<F>>>
    where
        I: IntoIterator + 'static,
        I::Item: Data,
        F: Fn(&T) -> I + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(FlatMap(f)))
    }

    pub fn filter_map<U: Data, F>(&self, f: F) -> Stream<U, Ext<C, FilterMap<F>>>
    where
        F: Fn(&T) -> Option<U> + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(FilterMap(f)))
    }

    /// Any number of weighted rows per row, each at the row's
    /// multiplicity times its weight.
    pub fn explode<I, U: Data, F>(&self, f: F) -> Stream<U, Ext<C, Explode<F>>>
    where
        I: IntoIterator<Item = (U, Delta)> + 'static,
        F: Fn(&T) -> I + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Explode(f)))
    }

    /// Every row at the opposite multiplicity.
    pub fn negate(&self) -> Stream<T, Ext<C, Negate>> {
        Stream::from_flow(self.flow.extend(Negate))
    }

    /// Group the rows by `f`: `t` becomes `(f(&t), t)`.
    pub fn key_by<K: Data, F>(&self, f: F) -> Grouped<K, T, Ext<C, KeyBy<F>>>
    where
        F: Fn(&T) -> K + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(KeyBy(f)))
    }
}

impl<K, V, M, C> Rows<K, V, M, C>
where
    K: Data,
    V: Data,
    M: Kind,
    C: Chain<Out = (K, V)>,
{
    /// Map each value, keeping its key. Over a register an operator
    /// keeps, the result is still one, read through `f`: a view defined
    /// from it stores nothing.
    pub fn map<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, MapVal<F>>>
    where
        F: Fn(&V) -> W + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(MapVal(f), Arc::new(move |_: &K, v: &V| Some(g(v))))
    }

    /// Map each value with its key, keeping the key. See [`map`](Self::map).
    pub fn map_with<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, MapWith<F>>>
    where
        F: Fn(&K, &V) -> W + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(MapWith(f), Arc::new(move |k: &K, v: &V| Some(g(k, v))))
    }

    /// Keep the rows `p` accepts. See [`map`](Self::map).
    pub fn filter<P>(&self, p: P) -> Rows<K, V, M, Ext<C, FilterKV<P>>>
    where
        P: Fn(&K, &V) -> bool + Clone + Send + Sync + 'static,
    {
        let q = p.clone();
        self.through(
            FilterKV(p),
            Arc::new(move |k: &K, v: &V| q(k, v).then(|| v.clone())),
        )
    }

    /// Map each value, dropping those `f` maps to `None`. See
    /// [`map`](Self::map).
    pub fn filter_map<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, FlatMapVal<F>>>
    where
        F: Fn(&V) -> Option<W> + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(FlatMapVal(f), Arc::new(move |_: &K, v: &V| g(v)))
    }

    /// Extend the chain by `stage`, which maps each value as `f` says,
    /// keeping a register origin as a mapped register.
    fn through<W: Data, Y>(&self, stage: Y, f: ValueMap<K, V, W>) -> Rows<K, W, M, Ext<C, Y>>
    where
        Y: Emit<(K, V), Emits = (K, W)> + Clone + Send + 'static,
    {
        Rows {
            flow: self.flow.extend(stage),
            origin: self.origin.as_ref().and_then(|o| o.filter_map(f)),
            _m: PhantomData,
        }
    }

    /// Regroup by `f` of the value. Two rows may now share a key, so
    /// the result is grouped.
    pub fn key_by<K2: Data, F>(&self, f: F) -> Grouped<K2, V, Ext<C, ReKey<F>>>
    where
        F: Fn(&V) -> K2 + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ReKey(f)))
    }

    /// Regroup by `f` of the key and value.
    pub fn key_by_with<K2: Data, F>(&self, f: F) -> Grouped<K2, V, Ext<C, ReKeyWith<F>>>
    where
        F: Fn(&K, &V) -> K2 + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ReKeyWith(f)))
    }

    /// Just the keys, as a bag: one per row.
    pub fn keys(&self) -> Stream<K, Ext<C, Keys>> {
        Stream::from_flow(self.flow.extend(Keys))
    }

    /// Just the values, as a bag: one per row.
    pub fn values(&self) -> Stream<V, Ext<C, Values>> {
        Stream::from_flow(self.flow.extend(Values))
    }

    /// Leave the keyed world: the rows as a bag of pairs.
    pub fn entries(&self) -> Stream<(K, V), C> {
        Stream::from_flow(self.flow.clone())
    }

    /// The same rows, viewed as grouped: a register is a group of at
    /// most one.
    pub fn grouped(&self) -> Grouped<K, V, C> {
        Rows {
            flow: self.flow.clone(),
            origin: self.origin.clone(),
            _m: PhantomData,
        }
    }
}

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Any number of values per value, each keeping the key.
    pub fn flat_map<I, F>(&self, f: F) -> Grouped<K, I::Item, Ext<C, FlatMapVal<F>>>
    where
        I: IntoIterator + 'static,
        I::Item: Data,
        F: Fn(&V) -> I + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(FlatMapVal(f)))
    }

    /// Weighted values per value, each keeping the key; see
    /// [`Stream::explode`].
    pub fn explode<I, W: Data, F>(&self, f: F) -> Grouped<K, W, Ext<C, ExplodeVal<F>>>
    where
        I: IntoIterator<Item = (W, Delta)> + 'static,
        F: Fn(&V) -> I + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ExplodeVal(f)))
    }
}

impl<K, V, C> Keyed<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Re-key by a derived prefix, keeping the key as the last
    /// component: `(k, v)` becomes `((f(&k, &v), k), v)`. Injective by
    /// construction, so still a register: a secondary index, whose rows
    /// sort by the prefix first.
    pub fn index_by<X: Data, F>(&self, f: F) -> Keyed<(X, K), V, Ext<C, Reindex<F>>>
    where
        F: Fn(&K, &V) -> X + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(Reindex(f)))
    }

    /// Every row under the one key `()`, for reductions over the whole
    /// register.
    pub(crate) fn unkeyed(&self) -> Grouped<(), V, Ext<C, Unkey>> {
        Rows::from_flow(self.flow.extend(Unkey))
    }
}

// ---------------------------------------------------------------------------
// Stateful combinators
// ---------------------------------------------------------------------------
