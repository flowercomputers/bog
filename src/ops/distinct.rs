//! [`DistinctRows`]: each `(key, value)` row once, with its counts kept
//! by key, so what reads the rows by key reads the counts directly.

use crate::{
    dataflow::Fingerprint,
    dataflow::state::{Counted, Declare, Edge, State, When},
    dataflow::{Bag, Buffer, Delta, Nary},
    error::Result,
    store::codec::OrderedCodec,
    store::merge::Change,
    store::merge::{Mergeless, Tally},
    store::txn::{KeyspaceId, WriteTx},
};

use super::Key;
use crate::stream::prelude::*;

/// Where [`DistinctRows`] keeps each row's count: `(k, Val(v)) -> count`,
/// so a key's rows are the range `(k, Lo)..=(k, Hi)`. A count is a whole
/// value, not a tally, so a subscriber sees it cross zero.
pub type DistinctCounts<K, V> = KeyspaceId<(K, Edge<V>), Mergeless<Delta>>;

/// Counts each `(key, value)` row's multiplicity in `counts` and emits
/// the row when its presence changes, like [`Distinct`],
/// but keyed: its [`Distinctly`] state is each key's present values,
/// which joins, reductions and grouped views read in place.
pub struct DistinctRows<K, V> {
    counts: DistinctCounts<K, V>,
    scratch: Bag<(K, V)>,
}

impl<K: Ord, V: Ord> DistinctRows<K, V> {
    pub fn new(counts: DistinctCounts<K, V>) -> Self {
        DistinctRows {
            counts,
            scratch: Bag::new(),
        }
    }
}

impl<K: Key, V: Key> DistinctRows<K, V> {
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(K, V)>],
        out: &mut Bag<(K, V)>,
    ) -> Result<()> {
        let single;
        let rows: &Bag<(K, V)> = if let [only] = batches {
            only
        } else {
            self.scratch.clear();
            for b in batches {
                for (t, d) in b.diffs() {
                    self.scratch.push(t.clone(), d);
                }
            }
            self.scratch.consolidate();
            single = &self.scratch;
            single
        };
        // each row's count as it was, read in order, then rewritten
        let mut ks = tx.keyspace(self.counts);
        let mut lookup = ks.lookup();
        let mut olds = Vec::with_capacity(rows.len());
        for ((k, v), _) in rows.diffs() {
            olds.push(
                lookup
                    .get(&(k.clone(), Edge::Val(v.clone())))?
                    .map_or(0, |c| *c),
            );
        }
        drop(lookup);
        for (((k, v), d), old) in rows.diffs().zip(olds) {
            let new = old.checked_add(d).expect("multiplicity overflowed");
            let key = (k.clone(), Edge::Val(v.clone()));
            let change = match (old, new) {
                (_, 0) => Change::Delete(old),
                (0, _) => Change::Insert(new),
                _ => Change::Replace(old, new),
            };
            ks.absorb(key, change)?;
            match (old > 0, new > 0) {
                (false, true) => out.push((k.clone(), v.clone()), 1),
                (true, false) => out.push((k.clone(), v.clone()), -1),
                _ => {}
            }
        }
        Ok(())
    }
}

impl<K: Key, V: Key> Nary for DistinctRows<K, V> {
    type Takes = Bag<(K, V)>;
    type Gives = Bag<(K, V)>;
    type State = Distinctly<K, V>;

    fn state(&self) -> Distinctly<K, V> {
        Distinctly { id: self.counts }
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.counts)]
    }

    fn fingerprint(&self, fp: &mut crate::dataflow::graph::Fingerprint) {
        fp.write_u64(self.counts.raw());
    }

    fn scratch(&mut self, release: bool) -> usize {
        Buffer::scratch(&mut self.scratch, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<(K, V)>], out: &mut Bag<(K, V)>) {
        if self.run(tx, batches, out).is_err() {
            out.clear();
        }
    }
}

/// The state of a [`DistinctRows`] node: each key's values whose count
/// is positive, each once.
pub struct Distinctly<K, V> {
    id: DistinctCounts<K, V>,
}

impl<K, V> Clone for Distinctly<K, V> {
    fn clone(&self) -> Self {
        Distinctly { id: self.id }
    }
}

impl<K, V> Distinctly<K, V> {
    pub fn id(&self) -> DistinctCounts<K, V> {
        self.id
    }
}

impl<K: Key, V: Key> Distinctly<K, V> {
    fn scan(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        rev: bool,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        let mut ks = tx.keyspace(self.id);
        let span = (key.clone(), Edge::Lo)..=(key.clone(), Edge::Hi);
        let rows = match (when, rev) {
            (When::New, false) => ks.range(span),
            (When::New, true) => ks.range_rev(span),
            (When::Old, false) => ks.range_old(span),
            (When::Old, true) => ks.range_old_rev(span),
        };
        for row in rows {
            let (k, count) = row?;
            if *count <= 0 {
                continue;
            }
            let Edge::Val(v) = &k.1 else {
                unreachable!("counts are kept by value")
            };
            if !f(v, 1) {
                break;
            }
        }
        Ok(())
    }
}

impl<K: Key, V: Key> State for Distinctly<K, V> {
    type Key = K;
    type Value = V;
    type Delta = Bag<(K, V)>;

    fn rows(delta: &Bag<(K, V)>, f: &mut dyn FnMut(&K, &V, Delta)) {
        for ((k, v), d) in delta.diffs() {
            f(k, v, d);
        }
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        self.scan(tx, key, when, false, f)
    }

    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        self.scan(tx, key, when, true, f)
    }
}

/// Counts each row's multiplicity in `counts` and emits a row when its
/// presence changes: `+1` when its count becomes positive, `-1` when it
/// stops being positive. Downstream sees each row at most once.
pub struct Distinct<T> {
    counts: KeyspaceId<T, Tally>,
    scratch: Bag<T>,
}

impl<T: Ord> Distinct<T> {
    pub fn new(counts: KeyspaceId<T, Tally>) -> Self {
        Distinct {
            counts,
            scratch: Bag::new(),
        }
    }
}

impl<T> Nary for Distinct<T>
where
    T: OrderedCodec + Clone + Send + Sync + 'static,
{
    type Takes = Bag<T>;
    type Gives = Bag<T>;
    type State = Counted<T>;

    fn state(&self) -> Counted<T> {
        Counted::new(self.counts)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.counts)]
    }

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.counts.raw());
    }

    fn scratch(&mut self, release: bool) -> usize {
        Buffer::scratch(&mut self.scratch, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<T>], out: &mut Bag<T>) {
        // a single input is sorted already; several merge through the scratch
        let single;
        let rows: &Bag<T> = if let [only] = batches {
            only
        } else {
            self.scratch.clear();
            for b in batches {
                for (t, d) in b.diffs() {
                    self.scratch.push(t.clone(), d);
                }
            }
            self.scratch.consolidate();
            single = &self.scratch;
            single
        };
        // each row's count as it was, read in order, then the change added
        // blind: the count decides nothing but whether the row crossed zero
        let mut ks = tx.keyspace(self.counts);
        let mut lookup = ks.lookup();
        let mut counts = Vec::with_capacity(rows.len());
        for (t, _) in rows.diffs() {
            match lookup.get(t) {
                Ok(c) => counts.push(c.map_or(0, |c| *c)),
                Err(_) => return,
            }
        }
        drop(lookup);
        for ((t, d), old) in rows.diffs().zip(counts) {
            let new = old.checked_add(d).expect("multiplicity overflowed");
            if ks.add(t.clone(), d).is_err() {
                return;
            }
            match (old > 0, new > 0) {
                (false, true) => out.push(t.clone(), 1),
                (true, false) => out.push(t.clone(), -1),
                _ => {}
            }
        }
    }
}

// The stream methods: how a schema declares this operator.

impl<T: Data, C: Chain<Out = T>> Stream<T, C> {
    /// Each row once, however many times it occurs: set semantics.
    pub fn distinct(&self) -> Stream<T> {
        let inputs = self.flow.inputs();
        let name = self.flow.name("distinct", None);
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<T, Tally> = b.id(&name);
        let node = b.graph.nary(&inputs, Distinct::new(id));
        drop(b);
        Stream::rooted(Root::at(node.rows()), self.flow.build.clone(), name.into())
    }
}

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Each `(key, value)` once, however many times it occurs. The rows
    /// are kept by key, so a grouped view defined from them, or a join
    /// or reduction over them, reads them in place.
    pub fn distinct(&self) -> Grouped<K, V> {
        let inputs = self.flow.inputs();
        let name = self.flow.name("distinct", None);
        let mut b = self.flow.build.borrow_mut();
        let id: DistinctCounts<K, V> = b.id(&name);
        let node = b.graph.nary(&inputs, DistinctRows::new(id));
        let origin = KeyedOrigin::distinct(b.graph.state(node));
        drop(b);
        Rows::rooted(
            Root::at(node.rows()),
            self.flow.build.clone(),
            name.into(),
            Some(origin),
        )
    }
}
