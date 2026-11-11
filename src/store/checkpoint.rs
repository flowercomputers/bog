//! Checkpoints: writing memory layers out as a table.

use super::*;

impl Shared {
    /// Write everything committed since the last checkpoint to a new table,
    /// retire the WAL segments it covers, and drop the covered layers from
    /// memory. Reads of that data then go through the block cache. Returns
    /// the epoch it covers through, or `None` if nothing was committed
    /// since the last one or the store is in memory.
    ///
    /// Each checkpoint is a delta over the previous one, until compaction
    /// merges them. Snapshots taken before a checkpoint keep their own
    /// layers alive until they are dropped.
    ///
    /// The maintenance thread checkpoints on its own when the WAL or the
    /// memtable crosses its limit; this is for checkpointing sooner.
    /// Commits continue while the table is written: it takes the writer
    /// lock only to switch to a new WAL segment at the start, and to
    /// install the table at the end.
    ///
    /// # Panics
    ///
    /// If this thread holds a write tx, which it would wait on forever.
    pub fn checkpoint(&self) -> Result<Option<u64>> {
        self.assert_no_tx_here("checkpoint");
        let done = self.checkpoint_now();
        if self.maintenance_due() {
            self.request_maintenance();
        }
        done
    }

    pub(super) fn checkpoint_now(&self) -> Result<Option<u64>> {
        let _layers = self
            .layer_work
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        // freeze: everything committed so far goes in the table, and the
        // WAL moves to a new segment, so the old ones hold only that
        let (dir, frozen, ckpt, retired, frozen_wal) = {
            let mut w = self.lock_writer();
            w.check()?;
            let Some(dir) = w.dir.clone() else {
                return Ok(None);
            };
            // stable: commits need the writer lock we hold
            let version = self.current.load_full();
            if version.layers.is_empty() {
                return Ok(None);
            }
            debug_assert!(
                version
                    .layers
                    .iter()
                    .all(|l| l.epoch > w.manifest.checkpointed()),
                "every layer is newer than every table"
            );
            debug_assert!(version.frozen.is_none(), "one checkpoint at a time");
            // reserve a file number; it reaches disk with a later manifest,
            // and a crash before then leaves an orphan the next open sweeps
            let ckpt = Checkpoint {
                number: w.manifest.next_file,
                epoch: version.epoch,
            };
            w.manifest.next_file += 1;
            let retired: Vec<u64> = w.manifest.segments.iter().map(|s| s.number).collect();
            if let Some(wal) = w.wal.as_mut() {
                // sealing promises the bytes are there, so sync them first
                let flushed = wal.flush(true, true).map(|()| wal.size as u64);
                let len = match flushed {
                    Ok(len) => len,
                    Err(e) => return Err(w.poison(e)),
                };
                let mut next = w.manifest.clone();
                next.segments
                    .last_mut()
                    .expect("the WAL has a segment")
                    .sealed = Some(len);
                let fresh = new_segment(&dir, &mut next)?;
                w.install_manifest(&dir, next)?;
                w.wal = Some(fresh);
            }
            // freeze every layer: commits keep folding into new ones above
            // them while the table is written
            self.current.store(Arc::new(Version {
                epoch: version.epoch,
                layers: version.layers.clone(),
                tables: version.tables.clone(),
                resident_bytes: version.resident_bytes,
                floors: version.floors.clone(),
                frozen: Some(version.epoch),
            }));
            let frozen: Vec<Arc<Layer>> = version.layers.clone();
            (dir, frozen, ckpt, retired, self.wal_bytes.load(Relaxed))
        };

        // write the table without locks; files left by a failure are
        // orphans the next open sweeps
        let path = dir.join(ckpt.file_name());
        let table = match self.write_checkpoint(&path, &frozen, ckpt) {
            Ok(table) => table,
            Err(e) => {
                self.thaw();
                return Err(e);
            }
        };
        drop(frozen);

        // install: the table replaces the frozen layers, which are all at
        // or below its epoch, and nothing has been folded into them
        let mut w = self.lock_writer();
        if let Err(e) = w.check() {
            self.thaw();
            return Err(e);
        }
        let mut next = w.manifest.clone();
        next.checkpoints.push(ckpt);
        next.segments.retain(|s| !retired.contains(&s.number));
        next.prune_floors();
        if let Err(e) = w.install_manifest(&dir, next) {
            self.thaw();
            return Err(e);
        }
        let now = self.current.load_full();
        let layers: Vec<Arc<Layer>> = now
            .layers
            .iter()
            .filter(|l| l.epoch > ckpt.epoch)
            .cloned()
            .collect();
        let mut tables = now.tables.clone();
        tables.push(Arc::new(table));
        self.current.store(Arc::new(Version {
            epoch: now.epoch,
            resident_bytes: layers.iter().map(|l| l.bytes).sum(),
            layers,
            tables,
            floors: Arc::new(w.manifest.floors.clone()),
            frozen: None,
        }));
        self.wal_bytes.fetch_sub(frozen_wal, Relaxed);
        drop(w);
        for hook in self
            .checkpoint_hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter_mut()
        {
            hook(ckpt.epoch);
        }

        for seg in retired {
            let seg = Segment {
                number: seg,
                sealed: None,
            };
            // a failure leaves an orphan for the next open to sweep
            let _ = fs::remove_file(dir.join(seg.file_name()));
        }
        Ok(Some(ckpt.epoch))
    }

    /// Unfreeze the layers a failed checkpoint froze, so folds resume;
    /// the next checkpoint freezes them again.
    pub(super) fn thaw(&self) {
        let _w = self.lock_writer();
        let now = self.current.load_full();
        self.current.store(Arc::new(Version {
            epoch: now.epoch,
            layers: now.layers.clone(),
            tables: now.tables.clone(),
            resident_bytes: now.resident_bytes,
            floors: now.floors.clone(),
            frozen: None,
        }));
    }

    /// Write `layers`, oldest first, to a checkpoint table at `path`.
    pub(super) fn write_checkpoint(
        &self,
        path: &Path,
        layers: &[Arc<Layer>],
        ckpt: Checkpoint,
    ) -> Result<Table> {
        let mut table = TableBuilder::create(path, self.opts.block_size)?;
        let mut built = BuiltIndexes::default();
        let ids: BTreeSet<u64> = layers
            .iter()
            .flat_map(|l| l.trees.keys().copied())
            .collect();
        for ks in ids {
            let trees: Vec<&(dyn Any + Send + Sync)> = layers
                .iter()
                .rev()
                .filter_map(|l| l.trees.get(&ks).map(|t| &*t.tree))
                .collect();
            let decl = &self.keyspaces[&ks];
            let mut indexing = RunIndexing::start(ks, &decl.indexes);
            table.begin_keyspace(ks)?;
            (decl.write_run)(&trees, &mut |k, v| {
                indexing.push(k, v, path)?;
                table.push(k, v)
            })?;
            indexing.seal(&mut table, &mut built)?;
        }
        table.finish(ckpt.epoch)?;
        let table = Table::open(path, ckpt.number)?;
        built.keep(&table);
        Ok(table)
    }
}
