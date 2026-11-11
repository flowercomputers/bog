//! [`Unique`]: at most one row per key, as an operator.

use super::*;
use crate::stream::prelude::*;

/// Keeps each key's one row in `output`, refusing the transaction when a
/// key is left with any other number of rows.
pub(crate) struct Unique<S: State> {
    pub(super) name: Arc<str>,
    pub(super) parent: S,
    pub(super) output: KeyspaceId<S::Key, Mergeless<S::Value>>,
    pub(super) dirty: Vec<S::Key>,
    pub(super) vals: Vec<(S::Value, Delta)>,
}

impl<S: State> Unique<S> {
    pub(crate) fn new(
        name: Arc<str>,
        parent: S,
        output: KeyspaceId<S::Key, Mergeless<S::Value>>,
    ) -> Self {
        Unique {
            name,
            parent,
            output,
            dirty: Vec::new(),
            vals: Vec::new(),
        }
    }
}

impl<S> Unique<S>
where
    S: State,
    S::Key: Data + Debug,
    S::Value: Data,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&S::Delta],
        out: &mut Register<S::Key, Mergeless<S::Value>>,
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
            read_group(&self.parent, tx, k, When::New, Take::Whole, &mut self.vals)?;
            let new = match &self.vals[..] {
                [] => None,
                [(v, 1)] => Some(v.clone()),
                // a rebuild replays rows accepted when they were written
                _ if tx.rebuilding() => None,
                vals => {
                    let total: Delta = vals.iter().map(|(_, d)| d).sum();
                    tx.reject(Rejected::new(
                        &*self.name,
                        format!(
                            "key {k:?} has {} rows (total multiplicity {total})",
                            vals.len()
                        ),
                    ));
                    return tx.failure().map_or(Ok(()), Err);
                }
            };
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

impl<S> Nary for Unique<S>
where
    S: State,
    S::Key: Data + Debug,
    S::Value: Data,
{
    type Takes = S::Delta;
    type Gives = Register<S::Key, Mergeless<S::Value>>;
    type State = Stored<S::Key, Mergeless<S::Value>>;

    fn state(&self) -> Self::State {
        Stored::new(self.output)
    }

    fn keyspaces(&self) -> Vec<DeclareKeyspace> {
        vec![DeclareKeyspace::derived(self.output)]
    }

    fn fingerprint(&self, fp: &mut crate::dataflow::graph::Fingerprint) {
        fp.write_u64(self.output.raw());
    }

    fn scratch(&mut self, release: bool) -> usize {
        crate::dataflow::vec_scratch(&mut self.dirty, release)
            + crate::dataflow::vec_scratch(&mut self.vals, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&S::Delta], out: &mut Self::Gives) {
        let _ = self.run(tx, batches, out);
    }
}

// The stream methods: how a schema declares this operator.

impl<K, V, C> Grouped<K, V, C>
where
    K: Data + std::fmt::Debug,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Treat this grouped stream as a register, enforcing that it is
    /// one row per key. A transaction that leaves a key with more than
    /// one row, or one row at a multiplicity other than one, is refused
    /// with a [`Rejected`] naming the operator. Prefer
    /// [`min`](Self::min) or [`max`](Self::max) when a key may
    /// legitimately have several rows.
    pub fn unique(&self) -> Keyed<K, V> {
        let name = self.flow.name("unique", None);
        let (input, state) = arranged(self, &format!("{name}.state"));
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<K, Mergeless<V>> = b.id(&name);
        let node = b.graph.nary(
            &[input],
            crate::constraints::Unique::new(name.as_str().into(), state, id),
        );
        drop(b);
        Rows::at_stored(id, node.rows(), self.flow.build.clone(), name.into())
    }
}
