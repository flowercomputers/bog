//! [`Reduce`]: per-key reductions that put aggregates in data position,
//! and the built-in ones: [`count`], [`min`], [`max`], [`top_k`] and
//! [`aggregate`].
//!
//! Totals that only add up need no group; see [`Sum`](crate::Sum).

use crate::{Bag, Delta, Nary, error::Result, graph::Fingerprint, tx::WriteTx};

use super::{Arrangement, Key, Resolved, read_group, update_counts};

/// How much of a dirty key's group a reduce reads before calling its
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

/// Recomputes a key's output from its group of values whenever the group
/// changes, and emits the difference from what it emitted before, so
/// downstream always holds exactly `logic` of the current groups.
///
/// `logic` gets the key and its live values, ascending by value, each
/// with its nonzero multiplicity, as much of the group as [`Take`] says,
/// and pushes the output rows for the key into its buffer, which is
/// cleared beforehand. It is not called for a key whose group emptied;
/// that key's old output is retracted. It must be deterministic.
///
/// The input is arranged in `state`, and what was emitted in `output`,
/// to retract exactly. A dirty key costs reading its group, as far as
/// `Take` says, and its old output.
pub struct Reduce<K, V, O, F> {
    state: Arrangement<K, V>,
    output: Arrangement<K, O>,
    take: Take,
    logic: F,
    dirty: Vec<K>,
    vals: Vec<(V, Delta)>,
    fresh: Vec<(O, Delta)>,
    old: Vec<(O, Delta)>,
    resolved: Resolved<K, V>,
    resolved_out: Resolved<K, O>,
}

impl<K, V, O, F> Reduce<K, V, O, F> {
    pub fn new(state: Arrangement<K, V>, output: Arrangement<K, O>, logic: F) -> Self {
        Reduce {
            state,
            output,
            take: Take::Whole,
            logic,
            dirty: Vec::new(),
            vals: Vec::new(),
            fresh: Vec::new(),
            old: Vec::new(),
            resolved: Vec::new(),
            resolved_out: Vec::new(),
        }
    }

    /// Hand the logic only part of each group; it must be correct on that
    /// part alone.
    pub fn reading(mut self, take: Take) -> Self {
        self.take = take;
        self
    }
}

impl<K, V, O, F> Reduce<K, V, O, F>
where
    K: Key,
    V: Key + Ord,
    O: Key + Ord,
    F: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(K, V)>],
        out: &mut Bag<(K, O)>,
    ) -> Result<()> {
        let mut arranged = Bag::new();
        for b in batches {
            for ((k, v), d) in b.diffs() {
                arranged.push((k.clone(), Some(v.clone())), d);
            }
        }
        // cancelling rows leave no key dirty
        self.dirty.clear();
        for ((k, _), _) in arranged.iter() {
            if self.dirty.last() != Some(k) {
                self.dirty.push(k.clone());
            }
        }
        update_counts(tx, self.state, arranged, &mut self.resolved, |_, _, _| {})?;

        let mut corrections = Bag::new();
        for key in &self.dirty {
            read_group(tx, self.state, key, self.take, &mut self.vals)?;
            if !self.vals.is_empty() {
                self.fresh.clear();
                (self.logic)(key, &self.vals, &mut self.fresh);
                for (o, d) in self.fresh.drain(..) {
                    corrections.push((key.clone(), Some(o)), d);
                }
            }
            read_group(tx, self.output, key, super::Take::Whole, &mut self.old)?;
            for (o, d) in self.old.drain(..) {
                corrections.push((key.clone(), Some(o)), -d);
            }
        }
        for ((k, o), d) in corrections.diffs() {
            out.push((k.clone(), o.clone().expect("outputs are values")), d);
        }
        update_counts(
            tx,
            self.output,
            corrections,
            &mut self.resolved_out,
            |_, _, _| {},
        )
    }
}

impl<K, V, O, F> Nary for Reduce<K, V, O, F>
where
    K: Key,
    V: Key + Ord,
    O: Key + Ord,
    F: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send,
{
    type Takes = Bag<(K, V)>;
    type Gives = Bag<(K, O)>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.state.raw());
        fp.write_u64(self.output.raw());
        match self.take {
            Take::Whole => fp.write_u64(0),
            Take::First(n) => fp.write_u64(1 << 32 | n as u64),
            Take::Last(n) => fp.write_u64(2 << 32 | n as u64),
        }
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<(K, V)>], out: &mut Bag<(K, O)>) {
        let _ = self.run(tx, batches, out);
    }
}

/// What a reduce's logic is; see [`Reduce`].
pub trait ReduceLogic<K, V, O>: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send {}
impl<K, V, O, F: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send> ReduceLogic<K, V, O> for F {}

/// The logic of the built-in reductions: a plain `fn`, so their types
/// stay nameable.
pub type ReduceFn<K, V, O> = fn(&K, &[(V, Delta)], &mut Vec<(O, Delta)>);

/// Logic that captures configuration, such as [`top_k`]'s `k`.
pub type BoxedReduceFn<K, V, O> = Box<dyn FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send>;

/// The multiplicity of each key, `(k, total)`: with ±1 rows, how many
/// there are. Takes unit values, so its state is one row per key; map
/// values away first with [`MapVal`](super::linear::MapVal).
pub fn count<K: Key>(
    state: Arrangement<K, ()>,
    output: Arrangement<K, Delta>,
) -> Reduce<K, (), Delta, ReduceFn<K, (), Delta>> {
    fn logic<K>(_: &K, vals: &[((), Delta)], out: &mut Vec<(Delta, Delta)>) {
        out.push((vals.iter().map(|(_, d)| d).sum(), 1));
    }
    Reduce::new(state, output, logic::<K> as ReduceFn<K, (), Delta>)
}

/// The least value of each key. Retracting it resurfaces the runner-up,
/// which the arrangement remembers.
pub fn min<K: Key, V: Key + Ord>(
    state: Arrangement<K, V>,
    output: Arrangement<K, V>,
) -> Reduce<K, V, V, ReduceFn<K, V, V>> {
    fn logic<K, V: Clone>(_: &K, vals: &[(V, Delta)], out: &mut Vec<(V, Delta)>) {
        out.push((vals[0].0.clone(), 1));
    }
    Reduce::new(state, output, logic::<K, V> as ReduceFn<K, V, V>).reading(Take::First(1))
}

/// The greatest value of each key. See [`min`].
pub fn max<K: Key, V: Key + Ord>(
    state: Arrangement<K, V>,
    output: Arrangement<K, V>,
) -> Reduce<K, V, V, ReduceFn<K, V, V>> {
    fn logic<K, V: Clone>(_: &K, vals: &[(V, Delta)], out: &mut Vec<(V, Delta)>) {
        out.push((vals[vals.len() - 1].0.clone(), 1));
    }
    Reduce::new(state, output, logic::<K, V> as ReduceFn<K, V, V>).reading(Take::Last(1))
}

/// The `k` greatest distinct values of each key, each once. Retracting
/// one resurfaces the next.
pub fn top_k<K: Key, V: Key + Ord>(
    state: Arrangement<K, V>,
    output: Arrangement<K, V>,
    k: usize,
) -> Reduce<K, V, V, BoxedReduceFn<K, V, V>> {
    let logic: BoxedReduceFn<K, V, V> = Box::new(move |_, vals, out| {
        for (v, _) in vals.iter().rev().take(k) {
            out.push((v.clone(), 1));
        }
    });
    Reduce::new(state, output, logic).reading(Take::Last(k))
}

/// Exactly one output per key, `f` of its whole group.
pub fn aggregate<K, V, O, F>(
    state: Arrangement<K, V>,
    output: Arrangement<K, O>,
    mut f: F,
) -> Reduce<K, V, O, impl ReduceLogic<K, V, O>>
where
    K: Key,
    V: Key + Ord,
    O: Key + Ord,
    F: FnMut(&K, &[(V, Delta)]) -> O + Send,
{
    Reduce::new(
        state,
        output,
        move |k: &K, vals: &[(V, Delta)], out: &mut Vec<(O, Delta)>| out.push((f(k, vals), 1)),
    )
}
