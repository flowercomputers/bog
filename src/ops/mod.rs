//! Operators beyond the basics in the crate root, one module per family,
//! ported from fold2:
//!
//! - [`linear`]: per-row operators over keyed rows, fused like the rest.
//! - [`join`]: the inner join, reactive on both sides.
//! - [`semijoin`]: rows whose key the other side does (or does not) have.
//! - [`reduce`]: per-key reductions in data position (count, min, max,
//!   top-k, aggregate, and any other).
//! - [`foreign_key`]: references to keys the other side lacks, and
//!   [`Reject`] to refuse a tx that has any.
//! - [`Minus`] and [`Flatten`], small adapters.
//!
//! Stateful operators keep their state in keyspaces the application
//! declares, usually as derived keyspaces, so a graph change rebuilds
//! them. Many of them keep an [`Arrangement`]: rows grouped by key, at
//! their multiplicity.

use std::{error::Error as StdError, marker::PhantomData};

use crate::{
    Bag, Binary, Change, Delta, Merge, Nary, Register, Unary,
    codec::OrderedCodec,
    error::Result,
    tx::{KeyspaceId, WriteTx},
};

pub mod foreign_key;
pub mod join;
pub mod linear;
pub mod reduce;
pub mod semijoin;

pub use foreign_key::ForeignKey;
pub use join::Join;
pub use reduce::{Reduce, Take};
pub use semijoin::Semijoin;

/// Rows `(K, V)` at their multiplicity, grouped by `K`: the state of a
/// join side or a reduce. Stored as `(k, Some(v)) -> count`, so that
/// `(k, None)` sorts before every row of `k` and a key's group is one
/// range scan; `None` is never stored. A count that reaches zero removes
/// the row.
pub type Arrangement<K, V> = KeyspaceId<(K, Option<V>), Delta>;

/// Scratch for resolving changes to an [`Arrangement`].
pub(crate) type Resolved<K, V> = Vec<((K, Option<V>), Change<Delta>)>;

/// What a key type must be to key an arrangement or counts.
pub trait Key: OrderedCodec + Clone + Send + Sync + 'static {}
impl<T: OrderedCodec + Clone + Send + Sync + 'static> Key for T {}

/// Read `key`'s rows from `ks` into `out`, ascending by value, as `take`
/// says, as the tx sees them.
pub(crate) fn read_group<K: Key, V: Key>(
    tx: &mut WriteTx,
    ks: Arrangement<K, V>,
    key: &K,
    take: Take,
    out: &mut Vec<(V, Delta)>,
) -> Result<()> {
    out.clear();
    let view = tx.keyspace(ks);
    for row in view.range((key.clone(), None)..) {
        let (k, d) = row?;
        let (kk, v) = &*k;
        if kk != key {
            break;
        }
        out.push((v.clone().expect("arrangements store values"), *d));
        if let Take::First(n) = take
            && out.len() == n
        {
            break;
        }
    }
    // todo: a reverse cursor would read only the last n
    if let Take::Last(n) = take {
        let extra = out.len().saturating_sub(n);
        out.drain(..extra);
    }
    Ok(())
}

/// Add `changes` to the counts in `ks`, removing counts that reach zero.
/// `seen` gets every key whose count moved, with its count before and
/// after.
pub(crate) fn update_counts<T: Key>(
    tx: &mut WriteTx,
    ks: KeyspaceId<T, Delta>,
    changes: Bag<T>,
    resolved: &mut Vec<(T, Change<Delta>)>,
    mut seen: impl FnMut(&T, Delta, Delta),
) -> Result<()> {
    resolved.clear();
    if changes.is_empty() {
        return Ok(());
    }
    let mut view = tx.keyspace(ks);
    view.resolve_sorted(changes, resolved)?;
    for (t, change) in resolved.drain(..) {
        let (old, new) = match &change {
            Change::Insert(n) => (0, *n),
            Change::Delete(o) => (*o, 0),
            Change::Replace(o, n) => (*o, *n),
            Change::Modify(_) => unreachable!("resolved"),
        };
        seen(&t, old, new);
        let change = match change {
            Change::Insert(0) => continue,
            Change::Replace(o, 0) => Change::Delete(o),
            c => c,
        };
        view.absorb(t, change)?;
    }
    Ok(())
}

/// The count at `key` in `ks`, as the tx sees it.
pub(crate) fn count_of<T: Key>(
    tx: &mut WriteTx,
    ks: KeyspaceId<T, Delta>,
    key: &T,
) -> Result<Delta> {
    Ok(tx.keyspace(ks).get(key)?.map_or(0, |c| *c))
}

/// `a * b`, which must not overflow.
pub(crate) fn multiply(a: Delta, b: Delta) -> Delta {
    a.checked_mul(b).expect("multiplicity overflowed")
}

/// `rows`, which are sorted, split into runs of equal keys.
pub(crate) fn by_key<'r, K, V>(
    rows: &'r Bag<(K, V)>,
) -> impl Iterator<Item = (&'r K, Vec<(&'r V, Delta)>)> + 'r
where
    K: Ord,
    V: Ord,
{
    let mut rows = rows.diffs().peekable();
    std::iter::from_fn(move || {
        let ((k, v), d) = rows.next()?;
        let mut group = vec![(v, d)];
        while let Some(((_, v), d)) = rows.next_if(|((kk, _), _)| kk == k) {
            group.push((v, d));
        }
        Some((k, group))
    })
}

/// Bag difference: the left input plus the negation of the right, so
/// matching rows cancel. Multiplicities can go negative; for set
/// difference by key, see [`Semijoin::without`].
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
    fn eval(&mut self, _: &mut WriteTx, l: &Bag<T>, r: &Bag<T>, out: &mut Bag<T>) {
        for (t, d) in l.diffs() {
            out.push(t.clone(), d);
        }
        for (t, d) in r.diffs() {
            out.push(t.clone(), d.checked_neg().expect("negate overflowed"));
        }
    }
}

/// A register's changes as rows: an insert is the new row at `+1`, a
/// delete the old row at `-1`, a replace both. Turns a view, such as
/// [`Sum`](crate::Sum)'s output, into rows every other operator takes.
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
    fn eval(&mut self, _: &mut WriteTx, batch: &Register<K, M>, out: &mut Self::Gives) {
        for (k, change) in batch.iter() {
            let (old, new) = match change {
                Change::Insert(v) => (None, Some(v)),
                Change::Delete(o) => (Some(o), None),
                Change::Replace(o, n) => (Some(o), Some(n)),
                Change::Modify(_) => unreachable!("views hold materialized changes"),
            };
            if let Some(o) = old {
                out.push((k.clone(), o.clone()), -1);
            }
            if let Some(n) = new {
                out.push((k.clone(), n.clone()), 1);
            }
        }
    }
}

/// Rejects the tx when a row reaches it at a positive multiplicity, with
/// the error `error` makes of the row. Put it after an operator whose
/// output is violations, such as [`ForeignKey`].
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

#[cfg(test)]
mod tests;
