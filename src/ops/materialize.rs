//! [`Materialize`]: a register's rows, kept as a register.

use crate::{
    dataflow::state::{Declare, Stored},
    dataflow::{Bag, Buffer, Delta, Nary, Register},
    error::Result,
    store::codec::Codec,
    store::merge::Change,
    store::merge::Mergeless,
    store::txn::{KeyspaceId, WriteTx},
};

use super::Key;

/// Keeps rows `(K, V)` that arrive one per key in `output`, a register,
/// and emits its changes. The input must be a register's changes as
/// rows: for a key, at most one row leaving at `-1` (its old value) and
/// one arriving at `+1` (its new value) per run. Nothing is read: the
/// rows say what the key held before.
///
/// This is how a stream whose kind is a register, but which no operator
/// stores, such as a semijoin of a table or a re-keyed table, is kept to
/// read.
///
/// # Panics
///
/// If a run brings a key several rows in the same direction, or a row
/// at another multiplicity: the input was not a register's changes.
pub struct Materialize<K, V: Clone + Send + 'static> {
    output: KeyspaceId<K, Mergeless<V>>,
}

impl<K, V: Clone + Send + 'static> Materialize<K, V> {
    pub fn new(output: KeyspaceId<K, Mergeless<V>>) -> Self {
        Materialize { output }
    }
}

impl<K, V> Materialize<K, V>
where
    K: Key,
    V: Codec + Clone + Ord + Send + Sync + 'static,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(K, V)>],
        out: &mut Register<K, Mergeless<V>>,
    ) -> Result<()> {
        let mut ks = tx.keyspace(self.output);
        let mut rows = batches.iter().flat_map(|b| b.diffs()).peekable();
        while let Some(((k, v), d)) = rows.next() {
            let (mut old, mut new) = (None, None);
            let mut take = |v: &V, d: Delta| match d {
                1 if new.is_none() => new = Some(v.clone()),
                -1 if old.is_none() => old = Some(v.clone()),
                _ => panic!("rows reaching Materialize are not a register's changes"),
            };
            take(v, d);
            while let Some(((_, v), d)) = rows.next_if(|((k2, _), _)| k2 == k) {
                take(v, d);
            }
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

impl<K, V> Nary for Materialize<K, V>
where
    K: Key,
    V: Codec + Clone + Ord + Send + Sync + 'static,
{
    type Takes = Bag<(K, V)>;
    type Gives = Register<K, Mergeless<V>>;
    type State = Stored<K, Mergeless<V>>;

    fn state(&self) -> Self::State {
        Stored::new(self.output)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.output)]
    }

    fn fingerprint(&self, fp: &mut crate::dataflow::graph::Fingerprint) {
        fp.write_u64(self.output.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Self::Takes], out: &mut Self::Gives) {
        if self.run(tx, batches, out).is_err() {
            Buffer::clear(out);
        }
    }
}
