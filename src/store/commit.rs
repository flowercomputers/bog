//! Committing a transaction, and reconciling derived state with the graph at open.

use super::*;

impl Store {
    /// Commit `tx`: log it, then publish it to readers. Returns the epoch
    /// it committed as, or `None` if it changed nothing, in which case no
    /// epoch is spent and nothing is logged.
    ///
    /// The WAL write and fsync happen before readers can see the tx, so
    /// under a strict policy nothing visible can be lost to a crash. If they
    /// fail, nothing is published and the store is poisoned.
    ///
    /// Checkpointing, folding layers and compacting tables are left to the
    /// maintenance thread. If it falls far enough behind (see
    /// `throttle`), this waits for it, and if the pass
    /// it waited on failed, returns that as [`Error::AfterCommit`]: the
    /// commit stands.
    pub fn finish(&self, tx: WriteTx<'_>) -> Result<Option<u64>> {
        let (epoch, overlay, snapshot, mut writer) = tx.into_parts();
        debug_assert!(
            std::ptr::eq(snapshot.store(), self),
            "tx from another store"
        );
        let overlay = overlay?;

        // opening a keyspace only to read it leaves an empty handle behind
        let mut sections: Vec<_> = overlay.into_iter().filter(|(_, u)| !u.is_empty()).collect();
        if sections.is_empty() {
            return Ok(None);
        }
        // deterministic WAL bytes for the same tx
        sections.sort_unstable_by_key(|(ks, _)| *ks);

        // subscribed keyspaces' changes are copied before sealing consumes
        // them, and sent once the commit is visible
        let mut subscribers = self
            .subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let staged: Vec<_> = sections
            .iter()
            .filter_map(|(ks, u)| subscribers.get(ks).map(|p| (*ks, p.stage(&**u))))
            .collect();

        // payload: repeated `keyspace u64 | len u64 | (ordered key, Option<value>)*`
        let mut trees = HashMap::with_capacity(sections.len());
        let mut bytes = 0;
        let logged = writer.log_with(epoch, |mut wal| {
            for (ks, underlay) in sections {
                let (tree, n) = match wal.as_deref_mut() {
                    Some(out) => encode_section(ks, out, |out| underlay.seal(Some(out))),
                    None => underlay.seal(None),
                };
                bytes += n;
                // built by the first reader that wants them
                let indexes = Arc::new(LayerIndexes::new(self.keyspaces[&ks].indexes.len()));
                trees.insert(
                    ks,
                    LayerTree {
                        tree: Arc::from(tree),
                        bytes: n,
                        indexes,
                    },
                );
            }
        })?;
        self.wal_bytes.fetch_add(logged, Relaxed);
        self.wal_buffer_bytes.store(
            writer.wal.as_ref().map_or(0, |w| w.buffer.capacity()),
            Relaxed,
        );

        let current = self.current.load();
        debug_assert_eq!(current.epoch + 1, epoch, "commit out of order");
        let mut layers = current.layers.clone();
        layers.push(Arc::new(Layer {
            epoch,
            trees,
            bytes,
        }));
        self.current.store(Arc::new(Version {
            epoch,
            resident_bytes: layers.iter().map(|l| l.bytes).sum(),
            layers,
            tables: current.tables.clone(),
            floors: current.floors.clone(),
            frozen: current.frozen,
        }));
        for (ks, changes) in staged {
            let live = subscribers
                .get_mut(&ks)
                .is_some_and(|p| p.publish(epoch, changes));
            if !live {
                subscribers.remove(&ks);
            }
        }
        drop(subscribers);

        // the tx's snapshot and our hold on the old version go first, so
        // nothing may see the new layer: then it folds straight into the
        // oldest, and no maintenance is needed for it. The writer is
        // released only after publishing, so the next tx's snapshot
        // includes this one.
        drop(current);
        drop(snapshot);
        self.fold_in_place_held(&self.snapshots(), INLINE_FOLD_LIMIT);
        drop(writer);
        if self.maintenance_due() {
            self.request_maintenance();
        }
        self.throttle().map_err(|e| Error::AfterCommit {
            epoch,
            source: Arc::new(e),
        })?;
        Ok(Some(epoch))
    }

    /// Bring derived state in line with graph fingerprint `fp`, at open.
    /// If it was recorded under another, truncate the derived and
    /// `dropped` keyspaces and rebuild the derived ones with `replay`,
    /// then record `fp`. Until it is recorded, every open rebuilds.
    pub(super) fn reconcile_graph(
        &self,
        fp: u64,
        dropped: &BTreeSet<u64>,
        replay: Option<&mut Replay<'_>>,
    ) -> Result<()> {
        let mut w = self.lock_writer();
        if w.manifest.graph == Some(fp) {
            debug_assert!(
                dropped.is_empty(),
                "dropping a keyspace changes the fingerprint"
            );
            return Ok(());
        }
        let derived: Vec<u64> = self
            .keyspaces
            .iter()
            .filter(|(_, d)| d.derived)
            .map(|(id, _)| *id)
            .collect();
        // with no commits yet there is nothing to rebuild from
        let rebuild = self.current.load().epoch > 0 && !derived.is_empty();
        if rebuild || !dropped.is_empty() {
            let replay = match replay {
                Some(replay) => Some(replay),
                None if rebuild => return Err(Error::RebuildRequired),
                None => None,
            };
            let mut doomed: Vec<u64> = derived;
            doomed.extend(dropped);
            self.truncate_locked(&mut w, &doomed, dropped)?;
            drop(w);
            if let (true, Some(replay)) = (rebuild, replay) {
                loop {
                    let mut tx = self.begin()?;
                    tx.set_rebuilding();
                    let more = replay(&mut tx)?;
                    self.finish(tx)?;
                    if !more {
                        break;
                    }
                }
            }
            w = self.lock_writer();
        }
        let mut next = w.manifest.clone();
        next.graph = Some(fp);
        w.update_manifest(next)
    }
}

impl Shared {
    /// Empty keyspaces `ids` as of the current epoch, and forget `dropped`
    /// ones entirely. Older data stays where it is, hidden by a floor:
    /// reads skip tables no newer than it, replay skips WAL sections no
    /// newer than it, and compaction leaves it behind. Clears the recorded
    /// graph fingerprint, which the caller records again once derived
    /// state is rebuilt.
    pub(super) fn truncate_locked(
        &self,
        w: &mut StoreWriter,
        ids: &[u64],
        dropped: &BTreeSet<u64>,
    ) -> Result<()> {
        w.check()?;
        let version = self.current.load_full();
        let mut next = w.manifest.clone();
        for &id in ids {
            next.floors.insert(id, version.epoch);
        }
        for id in dropped {
            next.keyspaces.remove(id);
            next.derived.remove(id);
        }
        next.graph = None;
        next.prune_floors();
        let floors = Arc::new(next.floors.clone());
        w.update_manifest(next)?;

        // layers hold nothing newer than the floor, so the trees go
        let layers: Vec<Arc<Layer>> = version
            .layers
            .iter()
            .filter_map(|l| {
                let trees: HashMap<u64, LayerTree> = l
                    .trees
                    .iter()
                    .filter(|(ks, _)| !ids.contains(ks))
                    .map(|(ks, t)| (*ks, t.clone()))
                    .collect();
                (!trees.is_empty()).then(|| {
                    Arc::new(Layer {
                        epoch: l.epoch,
                        bytes: trees.values().map(|t| t.bytes).sum(),
                        trees,
                    })
                })
            })
            .collect();
        self.current.store(Arc::new(Version {
            epoch: version.epoch,
            resident_bytes: layers.iter().map(|l| l.bytes).sum(),
            layers,
            tables: version.tables.clone(),
            floors,
            frozen: version.frozen,
        }));
        Ok(())
    }
}
