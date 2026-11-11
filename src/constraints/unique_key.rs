//! The index behind a table's `unique_key`: an alternate key no two rows
//! may share, kept and checked by the table's write methods.

use super::*;

/// An alternate key of a table: an index from it to the primary key,
/// maintained in the table's own transaction and checked on every
/// write.
pub(crate) trait AltKey<K, V>: Send + Sync {
    /// `key` goes from `old` to `new` (`None` is absent). `Err` refuses
    /// the write before anything is recorded.
    fn apply(&self, tx: &mut DbTx<'_>, key: &K, old: Option<&V>, new: Option<&V>) -> Result<()>;
}

pub(super) struct Alt<K: Data, V, K2: Data, F> {
    pub(super) table: Arc<str>,
    pub(super) key: F,
    pub(super) index: KeyspaceId<K2, Mergeless<K>>,
    pub(super) _p: PhantomData<fn(&V)>,
}

impl<K, V, K2, F> AltKey<K, V> for Alt<K, V, K2, F>
where
    K: Data + Debug,
    V: Data,
    K2: Data + Debug,
    F: Fn(&V) -> K2 + Send + Sync + 'static,
{
    fn apply(&self, tx: &mut DbTx<'_>, key: &K, old: Option<&V>, new: Option<&V>) -> Result<()> {
        let old_k2 = old.map(&self.key);
        let new_k2 = new.map(&self.key);
        let mut index = tx.table(self.index);
        if let Some(k2) = &new_k2
            && let Some(holder) = index.get(k2)?
            && *holder != *key
        {
            let holder = holder.into_owned();
            return Err(Error::reject(Rejected::new(
                &*self.table,
                format!("unique key {k2:?} is already held by {holder:?}"),
            )));
        }
        if let Some(k2) = old_k2
            && new_k2.as_ref() != Some(&k2)
        {
            index.set(k2, None)?;
        }
        if let Some(k2) = new_k2 {
            index.set(k2, Some(key.clone()))?;
        }
        Ok(())
    }
}
