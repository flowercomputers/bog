//! Compaction: merging checkpoint tables.

use super::*;

impl Shared {
    /// Merge windows of adjacent tables until no more than `max_tables`
    /// remain. Returns how many merges ran.
    pub(super) fn compact_while_over(&self) -> Result<usize> {
        let running = self
            .compaction
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.compact_over(&running)
    }

    /// [`compact_while_over`](Self::compact_while_over), with the compaction
    /// lock already held.
    pub(super) fn compact_over(&self, _running: &MutexGuard<'_, ()>) -> Result<usize> {
        let mut merges = 0;
        loop {
            let tables = self.current.load().tables.clone();
            if tables.len() <= self.opts.max_tables || self.shutdown.load(Relaxed) {
                return Ok(merges);
            }
            let window = cheapest_window(&tables, self.opts.compaction_width);
            self.merge_tables(&tables, window)?;
            merges += 1;
        }
    }

    /// Merge every table into one, dropping every tombstone. Returns
    /// whether there was anything to merge.
    ///
    /// Commits continue while it merges; it takes the writer lock only to
    /// install the result.
    ///
    /// # Panics
    ///
    /// If this thread holds a write tx, which it would wait on forever.
    pub fn compact(&self) -> Result<bool> {
        self.assert_no_tx_here("compact");
        let _running = self
            .compaction
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let tables = self.current.load().tables.clone();
        if tables.is_empty() {
            return Ok(false);
        }
        self.merge_tables(&tables, 0..tables.len())?;
        Ok(true)
    }

    /// Merge `tables[window]` into one table that replaces them. The window
    /// must be adjacent tables, so that newest-wins within it is right. If
    /// it starts at the oldest table nothing lies beneath it, so tombstones
    /// are dropped.
    ///
    /// The merge reads straight from disk, bypassing the cache, and holds no
    /// lock but the compaction lock. Only checkpoints change the table list
    /// meanwhile, and they only append newer tables, so the window is still
    /// in place when the result is installed.
    pub(super) fn merge_tables(&self, tables: &[Arc<Table>], window: Range<usize>) -> Result<()> {
        let inputs = &tables[window.clone()];
        let drop_tombstones = window.start == 0;
        let epoch = inputs.last().expect("a window holds tables").epoch();

        // reserve a file number; it reaches disk with the next manifest, and
        // a crash before then leaves an orphan the next open sweeps
        let (dir, number) = {
            let mut w = self.lock_writer();
            w.check()?;
            let Some(dir) = w.dir.clone() else {
                return Ok(());
            };
            let number = w.manifest.next_file;
            w.manifest.next_file += 1;
            (dir, number)
        };
        let ckpt = Checkpoint { number, epoch };
        let path = dir.join(ckpt.file_name());

        // floors only rise, and a later one covers every table that exists
        // when it is set, so reading them once here is safe
        let floors = self.current.load().floors.clone();
        #[cfg(test)]
        if self.fail_merges.load(Relaxed) {
            return Err(Error::io(
                format!("creating {}", path.display()),
                std::io::Error::other("injected merge failure"),
            ));
        }
        let mut out = TableBuilder::create(&path, self.opts.block_size)?;
        let ids: BTreeSet<u64> = inputs.iter().flat_map(|t| t.keyspaces()).collect();
        let (mut key, mut value) = (Vec::new(), Vec::new());
        let mut built = BuiltIndexes::default();
        for ks in ids {
            // newest first, so the first cursor holding a key wins; inputs
            // at or below the keyspace's floor hold nothing visible for it
            let floor = floors.get(&ks).copied().unwrap_or(0);
            let mut cursors: Vec<TableCursor> = inputs
                .iter()
                .rev()
                .take_while(|t| t.epoch() > floor)
                .map(|t| t.cursor(ks, None))
                .collect();
            if cursors.is_empty() {
                continue;
            }
            let mut indexing = RunIndexing::start(ks, &self.keyspaces[&ks].indexes);
            let add = self.keyspaces[&ks].add_encoded;
            out.begin_keyspace(ks)?;
            for c in &mut cursors {
                c.start()?;
            }
            loop {
                let mut win: Option<usize> = None;
                for (i, c) in cursors.iter().enumerate() {
                    if let Some((k, _)) = c.current()
                        && win.is_none_or(|w| k < cursors[w].current().unwrap().0)
                    {
                        win = Some(i);
                    }
                }
                let Some(w) = win else { break };
                let (k, v) = cursors[w].current().unwrap();
                key.clear();
                key.extend_from_slice(k);
                value.clear();
                value.extend_from_slice(v);
                cursors[w].advance()?;
                // older tables holding the key are shadowed or, in an
                // additive keyspace, add to it
                let mut live = true;
                for c in &mut cursors {
                    if let Some((_, amount)) = c.current().filter(|(k, _)| *k == &key[..]) {
                        if let Some(add) = add {
                            live = add(&mut value, amount).map_err(|e| {
                                Error::corrupt(c.path(), 0, format!("keyspace {ks}: {e}"))
                            })?;
                        }
                        c.advance()?;
                    }
                }
                if !live {
                    continue;
                }
                if !(drop_tombstones && value == TOMBSTONE) {
                    indexing.push(&key, &value, &path)?;
                    out.push(&key, &value)?;
                }
            }
            indexing.seal(&mut out, &mut built)?;
        }
        out.finish(epoch)?;
        let merged = Table::open(&path, number)?;
        built.keep(&merged);
        let merged = Arc::new(merged);

        {
            let mut w = self.lock_writer();
            w.check()?;
            let first = inputs[0].number();
            let mut next = w.manifest.clone();
            let at = next
                .checkpoints
                .iter()
                .position(|c| c.number == first)
                .expect("compaction inputs are still listed");
            debug_assert!(
                next.checkpoints[at..at + inputs.len()]
                    .iter()
                    .zip(inputs)
                    .all(|(c, t)| c.number == t.number()),
                "compaction inputs are still adjacent"
            );
            next.checkpoints.splice(at..at + inputs.len(), [ckpt]);
            next.prune_floors();
            let floors = Arc::new(next.floors.clone());
            w.install_manifest(&dir, next)?;

            // publish under the writer lock, which every publisher holds
            let current = self.current.load_full();
            let mut tables = current.tables.clone();
            tables.splice(at..at + inputs.len(), [merged]);
            self.current.store(Arc::new(Version {
                epoch: current.epoch,
                layers: current.layers.clone(),
                tables,
                resident_bytes: current.resident_bytes,
                floors,
                frozen: current.frozen,
            }));
        }

        // snapshots still reading the inputs keep their files open, and an
        // open file outlives its name
        for t in inputs {
            let _ = fs::remove_file(t.path());
            self.cache.forget(t.number());
        }
        Ok(())
    }
}

/// Builds a keyspace's run indexes as its rows are written to a table.
pub(super) struct RunIndexing {
    pub(super) ks: u64,
    pub(super) builders: Vec<(&'static str, Box<dyn EncodedBuilder>)>,
}

/// Indexes built while writing a table, to keep once it is open.
#[derive(Default)]
pub(super) struct BuiltIndexes(pub(super) Vec<(u64, &'static str, Aux)>);

impl BuiltIndexes {
    pub(super) fn keep(self, table: &Table) {
        for (ks, name, aux) in self.0 {
            table.indexes().insert(ks, name, aux);
        }
    }
}

impl RunIndexing {
    pub(super) fn start(ks: u64, kinds: &[IndexKind]) -> Self {
        RunIndexing {
            ks,
            builders: kinds.iter().map(|k| (k.name, (k.start)())).collect(),
        }
    }

    pub(super) fn push(&mut self, key: &[u8], value: &[u8], path: &Path) -> Result<()> {
        for (_, b) in &mut self.builders {
            b.push(key, value)
                .map_err(|e| Error::corrupt(path, 0, format!("keyspace {}: {e}", self.ks)))?;
        }
        Ok(())
    }

    /// Finish every index, writing the frames of those that encode.
    pub(super) fn seal(self, table: &mut TableBuilder, built: &mut BuiltIndexes) -> Result<()> {
        for (name, b) in self.builders {
            let (aux, frame) = b.finish();
            if let Some(frame) = frame {
                table.index_frame(self.ks, name, &frame)?;
            }
            built.0.push((self.ks, name, aux));
        }
        Ok(())
    }
}

/// The `width` adjacent tables, or all of them if there are fewer, with the
/// fewest bytes between them: merging small recent tables often and large
/// old ones rarely keeps the bytes rewritten per merge low.
pub(super) fn cheapest_window(tables: &[Arc<Table>], width: usize) -> Range<usize> {
    let width = width.min(tables.len());
    let start = (0..=tables.len() - width)
        .min_by_key(|&i| tables[i..i + width].iter().map(|t| t.size()).sum::<u64>())
        .unwrap_or(0);
    start..start + width
}
