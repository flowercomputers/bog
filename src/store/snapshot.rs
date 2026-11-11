//! Snapshots: reads of one committed version, across layers and tables.

use super::*;

pub struct Snapshot<'s> {
    pub(super) version: Arc<Version>,
    pub(super) store: &'s Store,
}
impl Drop for Snapshot<'_> {
    fn drop(&mut self) {
        let mut map = self.store.snapshots();
        match map.get_mut(&self.version.epoch) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                map.remove(&self.version.epoch);
            }
            None => unreachable!("snapshot epoch missing from map"),
        }
    }
}
impl<'s> Snapshot<'s> {
    /// The newest epoch this snapshot can see.
    pub fn epoch(&self) -> u64 {
        self.version.epoch
    }

    pub(crate) fn store(&self) -> &'s Store {
        self.store
    }

    /// The newest committed value of `key`: the first layer, newest first,
    /// that mentions the key decides, then the first table, newest first.
    /// A tombstone there means absent.
    pub fn get<K, M>(
        &self,
        keyspace: KeyspaceId<K, M>,
        key: &K,
    ) -> Result<Option<Cow<'_, M::Value>>>
    where
        K: OrderedCodec + 'static,
        M: Merge,
        M::Value: Codec,
    {
        let ks = keyspace.raw();
        let mut hits = self
            .version
            .layers
            .iter()
            .rev()
            .filter_map(|layer| layer.trees.get(&ks))
            .map(|t| {
                t.tree
                    .downcast_ref::<Resident<K, M::Value>>()
                    .expect("keyspace id reused with a different type")
            })
            .filter_map(|tree| tree.get(key));
        // an additive keyspace's value is the sum of every source's
        let mut sum: Option<Cow<'_, M::Value>> = None;
        if M::ADDITIVE {
            for amount in hits.flatten() {
                add_amount::<M>(&mut sum, Cow::Borrowed(amount));
            }
        } else if let Some(slot) = hits.next() {
            return Ok(slot.as_ref().map(Cow::Borrowed));
        }
        let mut tables = self.version.tables_for(ks).peekable();
        if tables.peek().is_some() {
            let mut probe = Vec::with_capacity(key.encode_ordered_size());
            key.encode_ordered(&mut probe);
            for table in tables {
                if let Some((block, range)) = table.get(ks, &probe, &self.store.cache)? {
                    let value = decode_all::<Option<M::Value>>(&block[range]).map_err(|e| {
                        Error::corrupt(table.path(), 0, format!("keyspace {ks}: {e}"))
                    })?;
                    if !M::ADDITIVE {
                        return Ok(value.map(Cow::Owned));
                    }
                    if let Some(amount) = value {
                        add_amount::<M>(&mut sum, Cow::Owned(amount));
                    }
                }
            }
        }
        Ok(sum.filter(|s| !M::consider_delete(s)))
    }

    /// A reader of `keyspace`'s values for keys asked in ascending order:
    /// one lookup per layer in memory, and one forward pass over each
    /// table, so a sorted batch reads each table block at most once.
    pub(crate) fn lookup<K, M>(&self, keyspace: KeyspaceId<K, M>) -> Lookup<'_, K, M::Value>
    where
        K: OrderedCodec + 'static,
        M: Merge,
    {
        let ks = keyspace.raw();
        Lookup {
            trees: self
                .version
                .layers
                .iter()
                .rev()
                .filter_map(|l| l.trees.get(&ks))
                .map(|t| {
                    t.tree
                        .downcast_ref::<Resident<K, M::Value>>()
                        .expect("keyspace id reused with a different type")
                })
                .collect(),
            tables: self
                .version
                .tables_for(ks)
                .map(|t| t.cursor(ks, Some(&self.store.cache)))
                .collect(),
            ks,
            probe: Vec::new(),
        }
    }

    /// The live entries of `keyspace` with keys in `range`, in key order.
    pub fn range<K, M>(
        &self,
        keyspace: KeyspaceId<K, M>,
        range: impl RangeBounds<K>,
    ) -> Cursor<'_, K, M>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
    {
        let (lower, upper) = (range.start_bound().cloned(), range.end_bound().cloned());
        self.cursor(keyspace, &[], lower, upper, false, None)
    }

    /// [`range`](Self::range), in reverse key order.
    pub fn range_rev<K, M>(
        &self,
        keyspace: KeyspaceId<K, M>,
        range: impl RangeBounds<K>,
    ) -> Cursor<'_, K, M>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
    {
        let (lower, upper) = (range.start_bound().cloned(), range.end_bound().cloned());
        self.cursor(keyspace, &[], lower, upper, true, None)
    }

    /// Every run of `keyspace`, newest first, each with its index of type
    /// `I`, which must be registered on the keyspace. Indexes not built yet
    /// are built, or loaded from their table, now.
    pub fn runs_indexed<K, M, I>(
        &self,
        keyspace: KeyspaceId<K, M>,
    ) -> Result<Vec<IndexedRun<'_, K, M, I>>>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
        I: 'static,
    {
        let ks = keyspace.raw();
        let kinds = &self.store.keyspaces[&ks].indexes;
        let slot = kinds
            .iter()
            .position(|k| k.id == std::any::TypeId::of::<I>())
            .unwrap_or_else(|| {
                panic!(
                    "index {} is not registered on keyspace {ks}",
                    type_name::<I>()
                )
            });
        let kind = &kinds[slot];
        let mut runs = Vec::new();
        for layer in self.version.layers.iter().rev() {
            if let Some(t) = layer.trees.get(&ks) {
                let tree = t
                    .tree
                    .downcast_ref::<Resident<K, M::Value>>()
                    .expect("keyspace id reused with a different type");
                runs.push(IndexedRun {
                    run: Run::Layer {
                        tree,
                        epoch: layer.epoch,
                    },
                    index: t.indexes.get(slot, kind, &*t.tree),
                    _i: PhantomData,
                });
            }
        }
        for table in self.version.tables_for(ks) {
            if !table.keyspaces().any(|k| k == ks) {
                continue;
            }
            runs.push(IndexedRun {
                index: table_index(table, ks, kind)?,
                run: Run::Table {
                    table,
                    ks,
                    cache: &self.store.cache,
                },
                _i: PhantomData,
            });
        }
        Ok(runs)
    }

    /// A merged cursor over `keyspace` within the bounds, forward or, with
    /// `rev`, backward, with `overlay`, a write tx's own runs of changes,
    /// oldest first, on top.
    pub(crate) fn cursor<'a, K, M>(
        &'a self,
        keyspace: KeyspaceId<K, M>,
        overlay: &'a [OverlayRun<K, M>],
        lower: Bound<K>,
        upper: Bound<K>,
        rev: bool,
        tx: Option<&'a TxState>,
    ) -> Cursor<'a, K, M>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
    {
        let ks = keyspace.raw();
        let mut sources = Vec::new();
        if !empty_bounds(lower.as_ref(), upper.as_ref()) {
            let (lo, up) = (lower.as_ref(), upper.as_ref());
            for run in overlay.iter().rev() {
                sources.push(Source::overlay(&run.entries, lo, up, rev));
            }
            for layer in self.version.layers.iter().rev() {
                if let Some(t) = layer.trees.get(&ks) {
                    let tree = t
                        .tree
                        .downcast_ref::<Resident<K, M::Value>>()
                        .expect("keyspace id reused with a different type");
                    sources.push(Source::layer(tree, lo, up, rev));
                }
            }
            for table in self.version.tables_for(ks) {
                sources.push(Source::table(
                    table.cursor(ks, Some(&self.store.cache)),
                    ks,
                    rev,
                ));
            }
        }
        Cursor::new(sources, lower, upper, rev, tx)
    }
}

/// Add `amount` to `sum`, which starts as nothing.
pub(super) fn add_amount<'a, M: Merge>(
    sum: &mut Option<Cow<'a, M::Value>>,
    amount: Cow<'a, M::Value>,
) {
    match sum {
        Some(s) => M::add(s.to_mut(), &amount),
        None => *sum = Some(amount),
    }
}

/// Point reads of one keyspace in ascending key order; see
/// [`Snapshot::lookup`].
pub(crate) struct Lookup<'a, K, V> {
    /// Layers holding the keyspace, newest first.
    pub(super) trees: Vec<&'a Resident<K, V>>,
    /// Its tables, newest first, each read forward.
    pub(super) tables: Vec<TableCursor<'a>>,
    pub(super) ks: u64,
    pub(super) probe: Vec<u8>,
}

impl<'a, K: OrderedCodec, V: Codec + Clone> Lookup<'a, K, V> {
    /// The value of `key`, which must not be less than the last key asked.
    pub(crate) fn get(&mut self, key: &K) -> Result<Option<Cow<'a, V>>> {
        for tree in &self.trees {
            if let Some(v) = tree.get(key) {
                return Ok(v.as_ref().map(Cow::Borrowed));
            }
        }
        if self.tables.is_empty() {
            return Ok(None);
        }
        self.probe.clear();
        key.encode_ordered(&mut self.probe);
        for t in &mut self.tables {
            t.seek(&self.probe)?;
            if let Some((k, v)) = t.current()
                && k == &self.probe[..]
            {
                let v = decode_all::<Option<V>>(v).map_err(|e| {
                    Error::corrupt(t.path(), 0, format!("keyspace {}: {e}", self.ks))
                })?;
                return Ok(v.map(Cow::Owned));
            }
        }
        Ok(None)
    }
}
