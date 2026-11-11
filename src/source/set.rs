//! [`Set`]: distinct membership, a table of `()`.

use super::*;

/// Distinct membership: a [`Table`] of `()`.
pub struct Set<T>(KeyspaceId<T, Mergeless<()>>);
handle!([T] Set[T], T, Mergeless<()>);

impl<T: Key> Set<T> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<T, ()> {
        graph.table(self.0)
    }

    pub fn contains(&self, tx: &mut DbTx<'_>, value: &T) -> Result<bool> {
        Ok(tx.table(self.0).get(value)?.is_some())
    }

    /// Add `value`. Returns whether it was new.
    pub fn insert(&self, tx: &mut DbTx<'_>, value: T) -> Result<bool> {
        let mut ks = tx.table(self.0);
        if ks.get(&value)?.is_some() {
            return Ok(false);
        }
        ks.absorb(value, Change::Insert(()))?;
        Ok(true)
    }

    /// Remove `value`. Returns whether it was there.
    pub fn remove(&self, tx: &mut DbTx<'_>, value: &T) -> Result<bool> {
        let mut ks = tx.table(self.0);
        if ks.get(value)?.is_none() {
            return Ok(false);
        }
        ks.absorb(value.clone(), Change::Delete(()))?;
        Ok(true)
    }

    /// [`insert`](Self::insert) every value.
    pub fn extend(&self, tx: &mut DbTx<'_>, values: impl IntoIterator<Item = T>) -> Result<()> {
        for v in values {
            self.insert(tx, v)?;
        }
        Ok(())
    }
}
