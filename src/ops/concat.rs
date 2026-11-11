//! Combining bags: [`Minus`], one bag less another. (Concatenation itself
//! is free: an operator takes any number of inputs.)

use std::marker::PhantomData;

use crate::stream::prelude::*;
use crate::{
    dataflow::{Bag, Binary},
    store::txn::WriteTx,
};

/// Bag difference: the left input plus the negation of the right, so
/// matching rows cancel. Multiplicities can go negative; for set
/// difference by key, see [`Semijoin::without`](super::Semijoin::without).
pub struct Minus<T>(PhantomData<fn() -> T>);

impl<T> Default for Minus<T> {
    fn default() -> Self {
        Minus(PhantomData)
    }
}

impl<T: Ord + Clone + Send + 'static> Binary for Minus<T> {
    type Left = Bag<T>;
    type Right = Bag<T>;
    type Gives = Bag<T>;
    type State = ();
    fn state(&self) {}
    fn eval(&mut self, _: &mut WriteTx, l: &Bag<T>, r: &Bag<T>, out: &mut Bag<T>) {
        for (t, d) in l.diffs() {
            out.push(t.clone(), d);
        }
        for (t, d) in r.diffs() {
            out.push(t.clone(), d.checked_neg().expect("negate overflowed"));
        }
    }
}

// The stream methods: how a schema declares this operator.

impl<T: Data, C: Chain<Out = T>> Stream<T, C> {
    /// Bag union with another stream. Costs nothing by itself: what
    /// consumes the union takes both as inputs.
    pub fn concat(&self, other: &impl StreamFeed<T>) -> Stream<T> {
        let mut inputs = self.flow.inputs();
        inputs.extend(other.inputs());
        let name = self.flow.name("concat", Some(&other.label()));
        Stream::rooted(Root::union(inputs), self.flow.build.clone(), name.into())
    }
}

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Union with another grouped (or keyed) stream on the same key.
    /// Costs nothing by itself: what consumes the union takes both as
    /// inputs.
    pub fn concat(&self, other: &impl GroupedFeed<K, V>) -> Grouped<K, V> {
        let mut inputs = self.flow.inputs();
        inputs.extend(other.inputs());
        let name = self.flow.name("concat", Some(&other.label()));
        Rows::rooted(
            Root::union(inputs),
            self.flow.build.clone(),
            name.into(),
            None,
        )
    }
}
