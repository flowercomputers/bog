//! [`Join`]: the inner join, reactive on both sides.

use crate::{Bag, Binary, Delta, graph::Fingerprint, tx::WriteTx};

use super::{Arrangement, Key, Resolved, Take, by_key, multiply, read_group, update_counts};

/// Pairs each left row `(k, a)` with every right row `(k, b)` and emits
/// `logic(&k, &a, &b)` at the product of their multiplicities: SQL's
/// INNER JOIN. Retracting a row on either side retracts its pairs.
///
/// Each side is arranged in its own keyspace. A run emits the bilinear
/// delta `ΔA ⋈ B + A' ⋈ ΔB`, where `A'` already includes `ΔA`, so rows
/// arriving on both sides at once pair exactly once. Reading a dirty key
/// costs its group on the other side.
pub struct Join<K, A, B, F> {
    left: Arrangement<K, A>,
    right: Arrangement<K, B>,
    logic: F,
    group_a: Vec<(A, Delta)>,
    group_b: Vec<(B, Delta)>,
    resolved: Resolved<K, A>,
    resolved_b: Resolved<K, B>,
}

impl<K, A, B, F> Join<K, A, B, F> {
    pub fn new(left: Arrangement<K, A>, right: Arrangement<K, B>, logic: F) -> Self {
        Join {
            left,
            right,
            logic,
            group_a: Vec::new(),
            group_b: Vec::new(),
            resolved: Vec::new(),
            resolved_b: Vec::new(),
        }
    }
}

impl<K, A, B, O, F> Join<K, A, B, F>
where
    K: Key,
    A: Key + Ord,
    B: Key + Ord,
    O: Ord + Send + 'static,
    F: FnMut(&K, &A, &B) -> O + Send,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &Bag<(K, A)>,
        r: &Bag<(K, B)>,
        out: &mut Bag<O>,
    ) -> crate::error::Result<()> {
        // ΔA ⋈ B, against the right side as it was
        for (k, rows) in by_key(l) {
            read_group(tx, self.right, k, Take::Whole, &mut self.group_b)?;
            for (a, da) in &rows {
                for (b, db) in &self.group_b {
                    out.push((self.logic)(k, a, b), multiply(*da, *db));
                }
            }
        }
        let mut arranged = Bag::new();
        for ((k, a), d) in l.diffs() {
            arranged.push((k.clone(), Some(a.clone())), d);
        }
        update_counts(tx, self.left, arranged, &mut self.resolved, |_, _, _| {})?;

        // A' ⋈ ΔB, against the left side including ΔA
        for (k, rows) in by_key(r) {
            read_group(tx, self.left, k, Take::Whole, &mut self.group_a)?;
            for (b, db) in &rows {
                for (a, da) in &self.group_a {
                    out.push((self.logic)(k, a, b), multiply(*da, *db));
                }
            }
        }
        let mut arranged = Bag::new();
        for ((k, b), d) in r.diffs() {
            arranged.push((k.clone(), Some(b.clone())), d);
        }
        update_counts(tx, self.right, arranged, &mut self.resolved_b, |_, _, _| {})
    }
}

impl<K, A, B, O, F> Binary for Join<K, A, B, F>
where
    K: Key,
    A: Key + Ord,
    B: Key + Ord,
    O: Ord + Send + 'static,
    F: FnMut(&K, &A, &B) -> O + Send,
{
    type Left = Bag<(K, A)>;
    type Right = Bag<(K, B)>;
    type Gives = Bag<O>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.left.raw());
        fp.write_u64(self.right.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &Bag<(K, A)>, r: &Bag<(K, B)>, out: &mut Bag<O>) {
        // on failure the tx has failed, and the graph stops after this node
        let _ = self.run(tx, l, r, out);
    }
}
