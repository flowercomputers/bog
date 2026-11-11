//! [`Multiset`]: rows with signed multiplicities, kept as counts.

use super::*;

/// Rows with signed multiplicities, stored as each row's count; a row
/// whose count returns to zero is gone. Its graph source gives the
/// changes in count, as a [`Bag`].
pub struct Multiset<T>(KeyspaceId<T, Tally>);
handle!([T] Multiset[T], T, Tally);

impl<T: Key> Multiset<T> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> NodeRef<Bag<T>> {
        graph.multiset(self.0)
    }

    /// How many of `row` there are.
    pub fn get(&self, tx: &mut DbTx<'_>, row: &T) -> Result<Delta> {
        Ok(tx.table(self.0).get(row)?.map_or(0, |n| *n))
    }

    /// Add `times` of `row`; negative to take away. Reads nothing: the
    /// count is the sum of what was added, and a row whose count is zero
    /// is not there.
    pub fn add(&self, tx: &mut DbTx<'_>, row: T, times: Delta) -> Result<()> {
        tx.table(self.0).add(row, times)
    }

    /// Take away `times` of `row`.
    pub fn remove(&self, tx: &mut DbTx<'_>, row: T, times: Delta) -> Result<()> {
        self.add(tx, row, -times)
    }

    /// Take away every `row`.
    pub fn clear(&self, tx: &mut DbTx<'_>, row: &T) -> Result<()> {
        let mut ks = tx.table(self.0);
        match ks.get(row)?.map(|n| *n) {
            Some(n) => ks.add(row.clone(), n.checked_neg().expect("negate overflowed")),
            None => Ok(()),
        }
    }
}
