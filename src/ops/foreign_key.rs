//! [`ForeignKey`]: references to keys the parent side lacks.

use std::{collections::BTreeSet, marker::PhantomData};

use crate::{
    Bag, Binary, Change, Delta,
    error::Result,
    graph::Fingerprint,
    tx::{KeyspaceId, WriteTx},
};

use super::{Key, count_of, update_counts};

/// Emits, at `+1`, every key that after a run is referenced by a child
/// row `(k, a)` on the left but has no parent row `(k, b)` on the right:
/// the run's foreign-key violations. Feed it to a
/// [`Reject`](super::Reject) to refuse transactions that have any.
///
/// It keeps only two counts per parent key, child references and parent
/// rows, never the rows themselves, so its state is bounded by the keys
/// referenced however many children there are. Its output is the
/// violations of each run, not a maintained relation: a rejected tx
/// leaves nothing to retract.
pub struct ForeignKey<K, A, B> {
    refs: KeyspaceId<K, Delta>,
    parents: KeyspaceId<K, Delta>,
    resolved: Vec<(K, Change<Delta>)>,
    _p: PhantomData<fn() -> (A, B)>,
}

impl<K, A, B> ForeignKey<K, A, B> {
    pub fn new(refs: KeyspaceId<K, Delta>, parents: KeyspaceId<K, Delta>) -> Self {
        ForeignKey {
            refs,
            parents,
            resolved: Vec::new(),
            _p: PhantomData,
        }
    }
}

impl<K, A, B> ForeignKey<K, A, B>
where
    K: Key,
    A: Ord + Send + 'static,
    B: Ord + Send + 'static,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &Bag<(K, A)>,
        r: &Bag<(K, B)>,
        out: &mut Bag<K>,
    ) -> Result<()> {
        let mut drefs = Bag::new();
        for ((k, _), d) in l.diffs() {
            drefs.push(k.clone(), d);
        }
        let mut dparents = Bag::new();
        for ((k, _), d) in r.diffs() {
            dparents.push(k.clone(), d);
        }
        let touched: BTreeSet<K> = drefs
            .iter()
            .chain(dparents.iter())
            .map(|(k, _)| k.clone())
            .collect();
        update_counts(tx, self.refs, drefs, &mut self.resolved, |_, _, _| {})?;
        update_counts(tx, self.parents, dparents, &mut self.resolved, |_, _, _| {})?;
        for k in touched {
            if count_of(tx, self.refs, &k)? > 0 && count_of(tx, self.parents, &k)? <= 0 {
                out.push(k, 1);
            }
        }
        Ok(())
    }
}

impl<K, A, B> Binary for ForeignKey<K, A, B>
where
    K: Key,
    A: Ord + Send + 'static,
    B: Ord + Send + 'static,
{
    type Left = Bag<(K, A)>;
    type Right = Bag<(K, B)>;
    type Gives = Bag<K>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.refs.raw());
        fp.write_u64(self.parents.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &Bag<(K, A)>, r: &Bag<(K, B)>, out: &mut Bag<K>) {
        let _ = self.run(tx, l, r, out);
    }
}
