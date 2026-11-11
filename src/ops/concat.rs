//! Combining bags: [`Minus`], one bag less another. (Concatenation itself
//! is free: an operator takes any number of inputs.)

use std::marker::PhantomData;

use crate::{Bag, Binary, tx::WriteTx};

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
