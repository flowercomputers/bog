//! Rows grouped by key: [`Arrange`] keeps them, [`CountRows`] counts them,
//! and the helpers here read and write groups for the operators over them.

use crate::{
    Bag, Buffer, Delta, Nary,
    codec::OrderedCodec,
    dataflow::state::{Arranged, Arrangement, Counted, Declare, Edge, State, When},
    error::Result,
    merge::Tally,
    tx::{KeyspaceId, WriteTx},
};

use super::Take;

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

impl<K, V> Groups<K, V> {
    /// See [`Nary::scratch`](crate::Nary::scratch).
    pub(crate) fn scratch(&mut self, release: bool) -> usize {
        crate::vec_scratch(&mut self.keys, release)
            + crate::vec_scratch(&mut self.rows, release)
            + crate::vec_scratch(&mut self.ends, release)
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

/// Keeps each row's count in `counts`, a tally, reading nothing, and
/// passes the rows on. Its [`Counted`] state says which rows are
/// present: what a foreign key reads of the keys its children
/// reference, at the cost of one count per distinct key.
pub struct CountRows<T> {
    counts: KeyspaceId<T, Tally>,
}

impl<T> CountRows<T> {
    pub fn new(counts: KeyspaceId<T, Tally>) -> Self {
        CountRows { counts }
    }
}

impl<T: Key> Nary for CountRows<T> {
    type Takes = Bag<T>;
    type Gives = Bag<T>;
    type State = Counted<T>;

    fn state(&self) -> Counted<T> {
        Counted::new(self.counts)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.counts)]
    }

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        fp.write_u64(self.counts.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<T>], out: &mut Bag<T>) {
        let mut counts = tx.keyspace(self.counts);
        for b in batches {
            for (t, d) in b.diffs() {
                if counts.add(t.clone(), d).is_err() {
                    out.clear();
                    return;
                }
                out.push(t.clone(), d);
            }
        }
    }
}
