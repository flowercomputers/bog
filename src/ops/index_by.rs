//! [`IndexBy`]: a secondary index of a table.

use std::marker::PhantomData;

use crate::{
    Bag, Change, Nary, Register,
    codec::OrderedCodec,
    dataflow::{
        Fingerprint,
        state::{Declare, Stored},
    },
    merge::*,
    tx::{KeyspaceId, WriteTx},
};

/// Maintains `index`, a keyspace holding `key(k, v)` for every live
/// `(k, v)` of its input view for which `key` returns one, and emits the
/// index's changes. `key` must not map two live entries to one index key.
///
/// An index ordered differently from its source is how a view is read in
/// another order: a leaderboard is balances keyed by `(-balance, user)`.
pub struct IndexBy<K, M, K2, F> {
    index: KeyspaceId<K2, ()>,
    key: F,
    _m: PhantomData<fn() -> (K, M)>,
}

impl<K, M: Merge, K2, F> IndexBy<K, M, K2, F> {
    pub fn new(index: KeyspaceId<K2, ()>, key: F) -> Self {
        IndexBy {
            index,
            key,
            _m: PhantomData,
        }
    }
}

impl<K, M, K2, F> Nary for IndexBy<K, M, K2, F>
where
    K: Ord + Send + 'static,
    M: Merge + Send,
    M::Value: Send,
    M::Diff: Send,
    K2: OrderedCodec + Clone + Send + Sync + 'static,
    F: Fn(&K, &M::Value) -> Option<K2> + Send,
{
    type Takes = Register<K, M>;
    type Gives = Bag<K2>;
    type State = Stored<K2, ()>;

    fn state(&self) -> Self::State {
        Stored::new(self.index)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.index)]
    }

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.index.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Register<K, M>], out: &mut Bag<K2>) {
        let mut ix = tx.keyspace(self.index);
        for b in batches {
            for (k, change) in b.iter() {
                let (old, new) = match change {
                    Change::Insert(v) => (None, Some(v)),
                    Change::Delete(o) => (Some(o), None),
                    Change::Replace(o, n) => (Some(o), Some(n)),
                    Change::Modify(_) => unreachable!("views hold materialized changes"),
                };
                let old = old.and_then(|v| (self.key)(k, v));
                let new = new.and_then(|v| (self.key)(k, v));
                if old == new {
                    continue;
                }
                if let Some(o) = old {
                    if ix.absorb(o.clone(), Change::Delete(())).is_err() {
                        return;
                    }
                    out.push(o, -1);
                }
                if let Some(n) = new {
                    if ix.absorb(n.clone(), Change::Insert(())).is_err() {
                        return;
                    }
                    out.push(n, 1);
                }
            }
        }
    }
}

// Linear operators
