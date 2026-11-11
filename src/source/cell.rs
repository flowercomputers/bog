//! [`Cell`]: one value, a table keyed by `()`.

use super::*;

/// One value: a [`Table`] keyed by `()`.
pub struct Cell<V: Value>(KeyspaceId<(), Mergeless<V>>);
handle!([V: Value] Cell[V], (), Mergeless<V>);

impl<V: Value> Cell<V> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<(), V> {
        graph.table(self.0)
    }

    pub fn get(&self, tx: &mut DbTx<'_>) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(&())?.map(|v| v.into_owned()))
    }

    pub fn set(&self, tx: &mut DbTx<'_>, value: V) -> Result<()> {
        tx.table(self.0).set((), Some(value))
    }

    pub fn clear(&self, tx: &mut DbTx<'_>) -> Result<()> {
        tx.table(self.0).set((), None)
    }
}
