//! [`Sum`]: each key's count and total, kept by adding.

use std::borrow::Cow;

use crate::{
    Bag, Buffer, Change, Delta, Nary, Register,
    codec::{Codec, OrderedCodec},
    dataflow::{
        Fingerprint,
        state::{Declare, Stored},
    },
    merge::*,
    tx::{KeyspaceId, WriteTx},
    vec_scratch,
};

pub struct Sum<K, V: Merge> {
    ks: KeyspaceId<K, (i64, V)>,
    scratch: Register<K, (i64, V)>,
    resolved: Vec<(K, Change<(i64, V)>)>,
}

impl<K: Ord, V: Merge> Sum<K, V> {
    pub fn new(ks: KeyspaceId<K, (i64, V)>) -> Self {
        Sum {
            ks,
            scratch: Register::new(),
            resolved: Vec::new(),
        }
    }
}

impl<K, V> Nary for Sum<K, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    V: Ord + Merge<Diff = V, Value = V> + Codec + Clone + PartialEq + Send + Sync,
{
    type Takes = Bag<(K, V)>;
    type Gives = Register<K, (i64, V)>;
    type State = Stored<K, (i64, V)>;

    fn state(&self) -> Self::State {
        Stored::new(self.ks)
    }

    fn keyspaces(&self) -> Vec<Declare> {
        vec![Declare::derived(self.ks)]
    }

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.ks.raw());
    }

    fn scratch(&mut self, release: bool) -> usize {
        vec_scratch(&mut self.resolved, release)
    }

    fn eval(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(K, V)>],
        out: &mut Register<K, (i64, V)>,
    ) {
        // 1. every row becomes a Modify, equal keys combined: streamed
        // off a single input, which is sorted; merged through the scratch
        // register when there are several
        let mut ks = tx.keyspace(self.ks);
        let diff = |v: &V, d: Delta| {
            let mut diff = (1, v.clone());
            <(i64, V)>::scale(&mut diff, d);
            diff
        };
        // 2. one sorted sweep resolves every Modify against the keyspace
        let resolved = if let [only] = batches {
            let mut rows = only.diffs().peekable();
            let changes = std::iter::from_fn(|| {
                let ((k, v), d) = rows.next()?;
                let mut acc = diff(v, d);
                while let Some(((_, v), d)) = rows.next_if(|((k2, _), _)| k2 == k) {
                    <(i64, V)>::combine(&mut acc, std::iter::once(Cow::Owned(diff(v, d))));
                }
                Some((k.clone(), Change::Modify(acc)))
            });
            ks.resolve_sorted(changes, &mut self.resolved)
        } else {
            self.scratch.clear();
            for b in batches {
                for ((k, v), d) in b.diffs() {
                    self.scratch
                        .absorb_one(k.clone(), Change::Modify(diff(v, d)));
                }
            }
            ks.resolve_sorted(std::mem::take(&mut self.scratch), &mut self.resolved)
        };
        if resolved.is_err() {
            // the tx failed, and with it this epoch; the graph stops here
            self.resolved.clear();
            return;
        }
        for (k, resolved) in self.resolved.drain(..) {
            // a count of zero means the key is gone
            let resolved = match resolved {
                Change::Insert(n) if n.0 == 0 => continue,
                Change::Replace(o, n) if n.0 == 0 => Change::Delete(o),
                r => r,
            };
            if ks.absorb(k, resolved).is_err() {
                // the tx failed; the graph stops at this node and the
                // commit will be refused (the drain empties `resolved`)
                return;
            }
        }
        // this node is the keyspace's only writer, so what changed in it
        // this run is exactly the delta
        ks.run_changes(out);
    }
}
