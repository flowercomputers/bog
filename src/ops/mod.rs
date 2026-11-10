//! Operators beyond the basics in the crate root, one module per family,
//! ported from fold2 and reworked to read their parents' state:
//!
//! - [`linear`]: per-row operators over keyed rows, fused like the rest.
//! - [`Arrange`]: rows grouped by key, as a state other operators read.
//! - [`join`]: the inner join, reactive on both sides.
//! - [`semijoin`]: rows whose key the other side does (or does not) have.
//! - [`reduce`]: per-key reductions in data position (min, max, top-k,
//!   aggregate, and any other). Totals and counts are [`Sum`](crate::Sum).
//! - [`foreign_key`]: references to keys the other side lacks, and
//!   [`Reject`] to refuse a tx that has any.
//! - [`Minus`] and [`Flatten`], small adapters.
//!
//! An operator that needs a parent's whole output reads the parent's
//! [state](crate::state) instead of keeping a copy: a table, a register an
//! operator keeps, or an [`Arrange`] node, which several operators can
//! share. Build them with
//! [`OpDag::binary_with`](crate::graph::OpDag::binary_with) and its
//! siblings, which hand them their parents' state handles.

use std::{error::Error as StdError, marker::PhantomData};

use crate::{
    Bag, Binary, Buffer, Delta, Merge, Nary, Register, Tally, Unary,
    codec::OrderedCodec,
    error::Result,
    state::{Arranged, Arrangement, Declare, Edge, State, When},
    tx::{KeyspaceId, WriteTx},
};

pub mod foreign_key;
pub mod join;
pub mod linear;
pub mod reduce;
pub mod semijoin;

pub use foreign_key::ForeignKey;
pub use join::Join;
pub use reduce::{Aggregate, ReduceMany, Take};
pub use semijoin::Semijoin;

/// What a key type must be to key an arrangement or counts.
pub trait Key: OrderedCodec + Clone + Send + Sync + 'static {}
impl<T: OrderedCodec + Clone + Send + Sync + 'static> Key for T {}

/// Read `key`'s rows from state `s` into `out`, ascending by value, as
/// `take` says.
pub(crate) fn read_group<S: State>(
    s: &S,
    tx: &mut WriteTx,
    key: &S::Key,
    when: When,
    take: Take,
    out: &mut Vec<(S::Value, Delta)>,
) -> Result<()> {
    out.clear();
    match take {
        Take::Whole | Take::First(_) => s.group(tx, key, when, &mut |v, d| {
            out.push((v.clone(), d));
            !matches!(take, Take::First(n) if out.len() >= n)
        }),
        Take::Last(n) => {
            s.group_rev(tx, key, when, &mut |v, d| {
                out.push((v.clone(), d));
                out.len() < n
            })?;
            out.reverse();
            Ok(())
        }
    }
}

/// Append the distinct keys of `delta`'s rows, in order, to `keys`.
pub(crate) fn dirty_keys_into<S: State>(delta: &S::Delta, keys: &mut Vec<S::Key>) {
    let start = keys.len();
    S::rows(delta, &mut |k, _, _| {
        if keys[start..].last() != Some(k) {
            keys.push(k.clone());
        }
    });
}

/// A delta's rows grouped by key, in key order: an operator's scratch,
/// refilled every run so a steady state allocates nothing.
pub(crate) struct Groups<K, V> {
    keys: Vec<K>,
    rows: Vec<(V, Delta)>,
    /// Where each key's rows end.
    ends: Vec<usize>,
}

impl<K, V> Default for Groups<K, V> {
    fn default() -> Self {
        Groups {
            keys: Vec::new(),
            rows: Vec::new(),
            ends: Vec::new(),
        }
    }
}

impl<K: Clone + PartialEq, V: Clone> Groups<K, V> {
    /// Replace the contents with `delta`'s rows.
    pub fn fill<S: State<Key = K, Value = V>>(&mut self, delta: &S::Delta) {
        let (keys, rows, ends) = (&mut self.keys, &mut self.rows, &mut self.ends);
        keys.clear();
        rows.clear();
        ends.clear();
        S::rows(delta, &mut |k, v, d| {
            if keys.last() != Some(k) {
                if !keys.is_empty() {
                    ends.push(rows.len());
                }
                keys.push(k.clone());
            }
            rows.push((v.clone(), d));
        });
        if !keys.is_empty() {
            ends.push(rows.len());
        }
    }

    /// The distinct keys, in order.
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Each key with its rows.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &[(V, Delta)])> {
        let mut start = 0;
        self.keys.iter().zip(&self.ends).map(move |(k, &end)| {
            let group = &self.rows[start..end];
            start = end;
            (k, group)
        })
    }
}

/// Add `rows`, sorted, to the arrangement `id`, reading nothing.
pub(crate) fn arrange<K: Key, V: Key>(
    tx: &mut WriteTx,
    id: Arrangement<K, V>,
    rows: impl IntoIterator<Item = ((K, V), Delta)>,
) -> Result<()> {
    let mut view = tx.keyspace(id);
    for ((k, v), d) in rows {
        view.add((k, Edge::Val(v)), d)?;
    }
    Ok(())
}

/// Sorted rows `(key, _)` reduced to each key's total multiplicity.
pub(crate) fn key_totals<'a, K, V>(
    rows: impl Iterator<Item = (&'a (K, V), Delta)> + 'a,
) -> impl Iterator<Item = (K, Delta)> + 'a
where
    K: Clone + PartialEq + 'a,
    V: 'a,
{
    let mut rows = rows.peekable();
    std::iter::from_fn(move || {
        let ((k, _), mut total) = rows.next()?;
        while let Some((_, d)) = rows.next_if(|((k2, _), _)| k2 == k) {
            total = total.checked_add(d).expect("multiplicity overflowed");
        }
        Some((k.clone(), total))
    })
}

/// `a * b`, which must not overflow.
pub(crate) fn multiply(a: Delta, b: Delta) -> Delta {
    a.checked_mul(b).expect("multiplicity overflowed")
}

/// Keeps rows `(K, V)` grouped by key, at their multiplicity, in
/// `arrangement`, and each key's total multiplicity in `totals`, and
/// passes the rows on. Nothing is read: both are tallies. Its [`Arranged`] state is what joins, semijoins,
/// foreign keys and reductions over rows read, and several of them can
/// share one. The totals make "is this key present" one lookup, and are
/// a count per key.
pub struct Arrange<K, V> {
    id: Arrangement<K, V>,
    totals: KeyspaceId<K, Tally>,
}

impl<K: Ord, V: Ord> Arrange<K, V> {
    pub fn new(arrangement: Arrangement<K, V>, totals: KeyspaceId<K, Tally>) -> Self {
        Arrange {
            id: arrangement,
            totals,
        }
    }
}

impl<K: Key, V: Key + Ord> Nary for Arrange<K, V> {
    type Takes = Bag<(K, V)>;
    type Gives = Bag<(K, V)>;
    type State = Arranged<K, V>;

    fn state(&self) -> Arranged<K, V> {
        Arranged::new(self.id, Some(self.totals))
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.id), Declare::derived(self.totals)]
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<(K, V)>], out: &mut Bag<(K, V)>) {
        // the output is the rows to arrange, so it is built first and the
        // arrangement's rows and totals are read off it
        for b in batches {
            for ((k, v), d) in b.diffs() {
                out.push((k.clone(), v.clone()), d);
            }
        }
        // linear when there is one input, which arrives sorted
        out.consolidate();
        let rows = out.diffs().map(|(row, d)| (row.clone(), d));
        let done = arrange(tx, self.id, rows).and_then(|()| {
            let mut totals = tx.keyspace(self.totals);
            for (k, d) in key_totals(out.diffs()) {
                totals.add(k, d)?;
            }
            Ok(())
        });
        if done.is_err() {
            // the tx failed; the graph stops here
            out.clear();
        }
    }
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

/// A register's changes as rows: an insert is the new row at `+1`, a
/// delete the old row at `-1`, a replace both. Turns a register, such as
/// a table or [`Sum`](crate::Sum)'s output, into rows for the linear
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

#[cfg(test)]
mod tests;
