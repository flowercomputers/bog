//! [`Table`]: one value per key, last writer wins.

use super::*;

/// One value per key: upsert and delete, last writer wins.
pub struct Table<K, V: Value>(KeyspaceId<K, Mergeless<V>>);
handle!([K, V: Value] Table[K, V], K, Mergeless<V>);

impl<K: Key, V: Value> Table<K, V> {
    /// Add it to `graph` as a source: each run's changes, and the table
    /// as state.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<K, V> {
        graph.table(self.0)
    }

    /// The value of `key`, with this transaction's writes.
    pub fn get(&self, tx: &mut DbTx<'_>, key: &K) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(key)?.map(|v| v.into_owned()))
    }

    /// Set `key` to `value`, replacing what was there.
    pub fn upsert(&self, tx: &mut DbTx<'_>, key: K, value: V) -> Result<()> {
        tx.table(self.0).set(key, Some(value))
    }

    /// Delete `key`. Returns whether it was there.
    pub fn remove(&self, tx: &mut DbTx<'_>, key: &K) -> Result<bool> {
        let mut ks = tx.table(self.0);
        let Some(old) = ks.get(key)?.map(|v| v.into_owned()) else {
            return Ok(false);
        };
        ks.absorb(key.clone(), Change::Delete(old))?;
        Ok(true)
    }

    /// Change `key`'s value in place. Returns whether it was there.
    pub fn update(&self, tx: &mut DbTx<'_>, key: &K, f: impl FnOnce(&mut V)) -> Result<bool> {
        let mut ks = tx.table(self.0);
        let Some(old) = ks.get(key)?.map(|v| v.into_owned()) else {
            return Ok(false);
        };
        let mut new = old.clone();
        f(&mut new);
        if new != old {
            ks.absorb(key.clone(), Change::Replace(old, new))?;
        }
        Ok(true)
    }

    /// [`upsert`](Self::upsert) every row.
    pub fn extend(&self, tx: &mut DbTx<'_>, rows: impl IntoIterator<Item = (K, V)>) -> Result<()> {
        let mut ks = tx.table(self.0);
        for (k, v) in rows {
            ks.set(k, Some(v))?;
        }
        Ok(())
    }
}

impl<V: Value> Table<Id<V>, V> {
    /// Add `value` under a freshly minted id.
    pub fn insert(&self, tx: &mut DbTx<'_>, value: V) -> Result<Id<V>> {
        let id = tx.mint(self.0.raw());
        tx.table(self.0).absorb(id, Change::Insert(value))?;
        Ok(id)
    }
}
