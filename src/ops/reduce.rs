//! Per-key reductions that put aggregates in data position:
//! [`Aggregate`], one value per key, kept as a register, and
//! [`ReduceMany`], any number of values per key, kept as an arrangement.
//! The built-ins are [`min`], [`max`], [`aggregate`] and [`top_k`].
//!
//! Totals and counts need no group; see [`Sum`](crate::Sum).

use crate::{
    Bag, Change, Delta, Mergeless, Nary, Register,
    codec::Codec,
    error::Result,
    state::{Arranged, Arrangement, Declare, State, Stored, When},
    tx::{KeyspaceId, WriteTx},
};

use super::{Key, arrange, dirty_keys_into, read_group};

/// How much of a dirty key's group a reduction reads before calling its
/// logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// The whole group, ascending.
    Whole,
    /// Only the `n` least values: MIN.
    First(usize),
    /// Only the `n` greatest values, still ascending: MAX, top-k. Reads
    /// the whole group for now, since cursors only move forward.
    Last(usize),
}

/// What a reduction's logic is: given a key and its values, ascending,
/// each with its nonzero multiplicity, as much of the group as [`Take`]
/// says.
pub trait Logic<K, V, O>: FnMut(&K, &[(V, Delta)]) -> O + Send {}
impl<K, V, O, F: FnMut(&K, &[(V, Delta)]) -> O + Send> Logic<K, V, O> for F {}

/// Keeps `logic` of each key's group, one value per key, in `output`, a
/// register keyed like its parent, and emits its changes.
///
/// It reads its group from its parent's state, which must be grouped by
/// the key: an [`Arrange`](super::Arrange) node, or a register. `logic`
/// is not called for a key whose group emptied; that key's value is
/// deleted. It must be deterministic.
///
/// A dirty key costs reading its group, as far as `Take` says, and one
/// lookup of its old value.
pub struct Aggregate<S: State, O: Clone + Send + 'static, F> {
    parent: S,
    output: KeyspaceId<S::Key, Mergeless<O>>,
    take: Take,
    logic: F,
    /// Scratch, reused across runs.
    dirty: Vec<S::Key>,
    vals: Vec<(S::Value, Delta)>,
}

impl<S: State, O: Clone + Send + 'static, F> Aggregate<S, O, F> {
    pub fn new(parent: S, output: KeyspaceId<S::Key, Mergeless<O>>, logic: F) -> Self {
        Aggregate {
            parent,
            output,
            take: Take::Whole,
            logic,
            dirty: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Hand the logic only part of each group; it must be correct on that
    /// part alone.
    pub fn reading(mut self, take: Take) -> Self {
        self.take = take;
        self
    }
}

impl<S, O, F> Aggregate<S, O, F>
where
    S: State,
    S::Key: Key,
    O: Codec + Clone + PartialEq + Send + Sync + 'static,
    F: Logic<S::Key, S::Value, O>,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&S::Delta],
        out: &mut Register<S::Key, Mergeless<O>>,
    ) -> Result<()> {
        self.dirty.clear();
        for b in batches {
            dirty_keys_into::<S>(b, &mut self.dirty);
        }
        if batches.len() > 1 {
            self.dirty.sort();
            self.dirty.dedup();
        }
        for k in &self.dirty {
            read_group(&self.parent, tx, k, When::New, self.take, &mut self.vals)?;
            let new = (!self.vals.is_empty()).then(|| (self.logic)(k, &self.vals));
            let mut ks = tx.keyspace(self.output);
            let old = ks.get(k)?.map(|o| o.into_owned());
            let change = match (old, new) {
                (None, None) => continue,
                (Some(o), Some(n)) if o == n => continue,
                (None, Some(n)) => Change::Insert(n),
                (Some(o), None) => Change::Delete(o),
                (Some(o), Some(n)) => Change::Replace(o, n),
            };
            ks.absorb(k.clone(), change.clone())?;
            out.absorb_one(k.clone(), change);
        }
        Ok(())
    }
}

impl<S, O, F> Nary for Aggregate<S, O, F>
where
    S: State,
    S::Key: Key,
    O: Codec + Clone + PartialEq + Send + Sync + 'static,
    F: Logic<S::Key, S::Value, O>,
{
    type Takes = S::Delta;
    type Gives = Register<S::Key, Mergeless<O>>;
    type State = Stored<S::Key, Mergeless<O>>;

    fn state(&self) -> Self::State {
        Stored::new(self.output)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.output)]
    }

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        take_fingerprint(self.take, fp);
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&S::Delta], out: &mut Self::Gives) {
        let _ = self.run(tx, batches, out);
    }
}

fn take_fingerprint(take: Take, fp: &mut crate::graph::Fingerprint) {
    match take {
        Take::Whole => fp.write_u64(0),
        Take::First(n) => fp.write_u64(1 << 32 | n as u64),
        Take::Last(n) => fp.write_u64(2 << 32 | n as u64),
    }
}

/// Keeps `logic` of each key's group, any number of rows per key, in
/// `output`, an arrangement, and emits the difference from what it held:
/// so downstream always holds `logic` of the current groups. See
/// [`Aggregate`] for one value per key, which is cheaper.
pub struct ReduceMany<S: State, O, F> {
    parent: S,
    output: Arrangement<S::Key, O>,
    take: Take,
    logic: F,
    /// Scratch, reused across runs.
    dirty: Vec<S::Key>,
    vals: Vec<(S::Value, Delta)>,
    fresh: Vec<(O, Delta)>,
    old: Vec<(O, Delta)>,
}

impl<S: State, O: Ord, F> ReduceMany<S, O, F> {
    pub fn new(parent: S, output: Arrangement<S::Key, O>, logic: F) -> Self {
        ReduceMany {
            parent,
            output,
            take: Take::Whole,
            logic,
            dirty: Vec::new(),
            vals: Vec::new(),
            fresh: Vec::new(),
            old: Vec::new(),
        }
    }

    /// See [`Aggregate::reading`].
    pub fn reading(mut self, take: Take) -> Self {
        self.take = take;
        self
    }
}

impl<S, O, F> ReduceMany<S, O, F>
where
    S: State,
    S::Key: Key,
    O: Key + Ord,
    F: FnMut(&S::Key, &[(S::Value, Delta)], &mut Vec<(O, Delta)>) + Send,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&S::Delta],
        out: &mut Bag<(S::Key, O)>,
    ) -> Result<()> {
        self.dirty.clear();
        for b in batches {
            dirty_keys_into::<S>(b, &mut self.dirty);
        }
        if batches.len() > 1 {
            self.dirty.sort();
            self.dirty.dedup();
        }
        // the corrections are the output: fresh results in, old ones out
        let output = Arranged::new(self.output, None);
        for k in &self.dirty {
            read_group(&self.parent, tx, k, When::New, self.take, &mut self.vals)?;
            if !self.vals.is_empty() {
                self.fresh.clear();
                (self.logic)(k, &self.vals, &mut self.fresh);
                for (o, d) in self.fresh.drain(..) {
                    out.push((k.clone(), o), d);
                }
            }
            read_group(&output, tx, k, When::New, Take::Whole, &mut self.old)?;
            for (o, d) in self.old.drain(..) {
                out.push((k.clone(), o), -d);
            }
        }
        out.consolidate();
        arrange(
            tx,
            self.output,
            out.diffs().map(|(row, d)| (row.clone(), d)),
        )
    }
}

impl<S, O, F> Nary for ReduceMany<S, O, F>
where
    S: State,
    S::Key: Key,
    O: Key + Ord,
    F: FnMut(&S::Key, &[(S::Value, Delta)], &mut Vec<(O, Delta)>) + Send,
{
    type Takes = S::Delta;
    type Gives = Bag<(S::Key, O)>;
    type State = Arranged<S::Key, O>;

    fn state(&self) -> Self::State {
        Arranged::new(self.output, None)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.output)]
    }

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        take_fingerprint(self.take, fp);
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&S::Delta], out: &mut Self::Gives) {
        let _ = self.run(tx, batches, out);
    }
}

/// The logic of the built-in one-value reductions: a plain `fn`, so their
/// types stay nameable.
pub type AggregateFn<K, V, O> = fn(&K, &[(V, Delta)]) -> O;

/// The least value of each key. Retracting it resurfaces the runner-up,
/// which the parent's state holds.
pub fn min<S>(
    parent: S,
    output: KeyspaceId<S::Key, Mergeless<S::Value>>,
) -> Aggregate<S, S::Value, AggregateFn<S::Key, S::Value, S::Value>>
where
    S: State,
{
    fn logic<K, V: Clone>(_: &K, vals: &[(V, Delta)]) -> V {
        vals[0].0.clone()
    }
    Aggregate::new(
        parent,
        output,
        logic::<S::Key, S::Value> as AggregateFn<_, _, _>,
    )
    .reading(Take::First(1))
}

/// The greatest value of each key. See [`min`].
pub fn max<S>(
    parent: S,
    output: KeyspaceId<S::Key, Mergeless<S::Value>>,
) -> Aggregate<S, S::Value, AggregateFn<S::Key, S::Value, S::Value>>
where
    S: State,
{
    fn logic<K, V: Clone>(_: &K, vals: &[(V, Delta)]) -> V {
        vals[vals.len() - 1].0.clone()
    }
    Aggregate::new(
        parent,
        output,
        logic::<S::Key, S::Value> as AggregateFn<_, _, _>,
    )
    .reading(Take::Last(1))
}

/// `f` of each key's whole group.
pub fn aggregate<S, O: Clone + Send + 'static, F>(
    parent: S,
    output: KeyspaceId<S::Key, Mergeless<O>>,
    f: F,
) -> Aggregate<S, O, F>
where
    S: State,
    F: Logic<S::Key, S::Value, O>,
{
    Aggregate::new(parent, output, f)
}

/// The logic of [`top_k`].
pub type TopK<K, V> = Box<dyn FnMut(&K, &[(V, Delta)], &mut Vec<(V, Delta)>) + Send>;

/// The `k` greatest distinct values of each key, each once. Retracting
/// one resurfaces the next.
pub fn top_k<S>(
    parent: S,
    output: Arrangement<S::Key, S::Value>,
    k: usize,
) -> ReduceMany<S, S::Value, TopK<S::Key, S::Value>>
where
    S: State,
    S::Value: Ord,
{
    let logic: TopK<S::Key, S::Value> = Box::new(move |_, vals, out| {
        for (v, _) in vals.iter().rev().take(k) {
            out.push((v.clone(), 1));
        }
    });
    ReduceMany::new(parent, output, logic).reading(Take::Last(k))
}
