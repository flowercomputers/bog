//! Constraints as operators: [`Check`] over a table's changes, [`Reject`]
//! over rows. A violation fails the transaction.

use std::{error::Error as StdError, marker::PhantomData};

use crate::{Bag, Nary, Register, merge::*, tx::WriteTx};

/// Rejects the tx when a value its input view is left holding fails
/// `check`. Only the values an epoch ends with are checked, so a tx that
/// passes through a bad state and recovers within itself is accepted.
///
/// A rebuild of derived state is not checked: it replays rows that were
/// accepted when written, in chunks whose intermediate states they may
/// never have passed through.
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
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Register<K, M>], _out: &mut Bag<()>) {
        if tx.rebuilding() {
            return;
        }
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

/// Rejects the tx when a row reaches it at a positive multiplicity, with
/// the error `error` makes of the row. Put it after an operator whose
/// output is violations, such as [`ForeignKey`](super::ForeignKey).
///
/// A rebuild of derived state is not rejected: its chunks can hold
/// states the rows never passed through, such as a child before its
/// parent.
pub struct Reject<T, F>(F, PhantomData<fn() -> T>);

impl<T, F> Reject<T, F> {
    pub fn new(error: F) -> Self {
        Reject(error, PhantomData)
    }
}

impl<T, F, E> Nary for Reject<T, F>
where
    T: Ord + Send + 'static,
    F: Fn(&T) -> E + Send,
    E: StdError + Send + Sync + 'static,
{
    type Takes = Bag<T>;
    type Gives = Bag<()>;
    type State = ();
    fn state(&self) {}
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<T>], _: &mut Bag<()>) {
        if tx.rebuilding() {
            return;
        }
        for b in batches {
            if let Some((t, _)) = b.diffs().find(|(_, d)| *d > 0) {
                tx.reject((self.0)(t));
                return;
            }
        }
    }
}
