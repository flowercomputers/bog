//! Declaring keyspaces and opening a store.

use super::*;

pub struct StoreBuilder {
    pub(super) durability_policy: DurabilityPolicy,
    pub(super) dir: Option<PathBuf>,
    pub(super) keyspaces: HashMap<u64, KeyspaceDecl>,
    pub(super) opts: StoreOptions,
    pub(super) graph: u64,
}

impl StoreBuilder {
    /// An in-memory store unless [`dir`](Self::dir) is set. Policies that
    /// write need a directory.
    pub fn new(durability_policy: DurabilityPolicy) -> Self {
        StoreBuilder {
            durability_policy,
            dir: None,
            keyspaces: HashMap::new(),
            opts: StoreOptions::default(),
            graph: 0,
        }
    }

    /// Replace every size limit at once.
    pub fn options(mut self, opts: StoreOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Adjust the size limits in place.
    pub fn options_with(mut self, f: impl FnOnce(&mut StoreOptions)) -> Self {
        f(&mut self.opts);
        self
    }

    /// Checkpoint once the WAL reaches this many bytes. Defaults to 64 MiB.
    pub fn wal_limit(mut self, bytes: u64) -> Self {
        self.opts.wal_limit = bytes;
        self
    }

    /// Checkpoint once committed data in memory reaches this many bytes.
    /// Defaults to 64 MiB.
    pub fn memtable_limit(mut self, bytes: usize) -> Self {
        self.opts.memtable_limit = bytes;
        self
    }

    /// Kill a write tx once its writes hold this many bytes. Defaults to
    /// 64 MiB.
    pub fn tx_memory_limit(mut self, bytes: usize) -> Self {
        self.opts.tx_memory_limit = bytes;
        self
    }

    /// Target size of a checkpoint table's blocks. Defaults to 4 KiB.
    pub fn block_size(mut self, bytes: usize) -> Self {
        self.opts.block_size = bytes;
        self
    }

    /// Bytes of checkpoint blocks to keep in memory. Defaults to 32 MiB.
    pub fn cache_size(mut self, bytes: usize) -> Self {
        self.opts.cache_size = bytes;
        self
    }

    /// Size the store's memory limits from one number; see
    /// [`StoreOptions::resolved`]. No budget by default.
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.opts.memory_budget = Some(bytes);
        self
    }

    /// The size limits as the store will use them.
    pub fn resolved_options(&self) -> StoreOptions {
        self.opts.resolved()
    }

    /// Replace the durability policy.
    pub fn durability(mut self, policy: DurabilityPolicy) -> Self {
        self.durability_policy = policy;
        self
    }

    /// Keep the store in `dir`, creating it if needed.
    pub fn dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// Declare a base keyspace, one the application writes. Every keyspace
    /// a transaction opens must be declared, and every keyspace in the WAL
    /// must be declared with the types it was written with.
    pub fn keyspace<K, M>(self, id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        self.declare(id, false)
    }

    /// Declare a derived keyspace, one the graph computes from base
    /// keyspaces. When the store opens under a different
    /// [`graph`](Self::graph) fingerprint, or with different derived
    /// keyspaces, than it last recorded, every derived keyspace is
    /// truncated and rebuilt; see [`build_with`](Self::build_with).
    ///
    /// A derived keyspace that is no longer declared is dropped, where an
    /// undeclared base keyspace is an error.
    pub fn derived<K, M>(self, id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        self.declare(id, true)
    }

    /// The fingerprint of the graph that computes the derived keyspaces,
    /// typically [`OpDag::fingerprint`](crate::graph::OpDag::fingerprint)
    /// mixed with a schema version.
    pub fn graph(mut self, fingerprint: u64) -> Self {
        self.graph = fingerprint;
        self
    }

    pub(super) fn declare<K, M>(mut self, id: KeyspaceId<K, M>, derived: bool) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        let decl = KeyspaceDecl {
            key: type_name::<K>(),
            merge: type_name::<M>(),
            derived,
            replay: replay_section::<K, M>,
            write_run: write_run::<K, M>,
            measure: measure_tree::<K, M::Value>,
            fold: fold_trees::<K, M>,
            fold_into: fold_into::<K, M>,
            add_encoded: M::ADDITIVE.then_some(add_encoded::<M>),
            indexes: Vec::new(),
        };
        if let Some(prev) = self.keyspaces.get(&id.raw()) {
            assert!(
                (prev.key, prev.merge, prev.derived) == (decl.key, decl.merge, decl.derived),
                "keyspace {} declared twice differently",
                id.raw()
            );
            return self;
        }
        self.keyspaces.insert(id.raw(), decl);
        self
    }

    /// Keep a run index of type `I` on every run of keyspace `id`, which
    /// must already be declared. See [`RunIndex`].
    pub fn indexed<K, M, I>(mut self, id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
        I: RunIndex<K, M::Value>,
    {
        let decl = self
            .keyspaces
            .get_mut(&id.raw())
            .unwrap_or_else(|| panic!("keyspace {} indexed before it is declared", id.raw()));
        assert_eq!(
            (decl.key, decl.merge),
            (type_name::<K>(), type_name::<M>()),
            "keyspace {} indexed with other types than declared",
            id.raw()
        );
        let kind = IndexKind::of::<K, M::Value, I>();
        assert!(
            decl.indexes
                .iter()
                .all(|k| k.id != kind.id && k.name != kind.name),
            "index {} registered twice on keyspace {}",
            kind.name,
            id.raw()
        );
        decl.indexes.push(kind);
        self
    }

    /// Open the store. Fails with [`Error::RebuildRequired`] if derived
    /// state must be rebuilt; use [`build_with`](Self::build_with) then.
    pub fn build(self) -> Result<Store> {
        self.open(None)
    }

    /// Open the store, rebuilding derived state with `replay` if the graph
    /// fingerprint or the set of derived keyspaces changed since the last
    /// open recorded them.
    ///
    /// A rebuild truncates every derived keyspace, then commits write
    /// transactions until `replay` returns `false`. Each call gets a fresh
    /// tx and should feed the graph a chunk of the base keyspaces, read
    /// through the tx, as inserts, and run it; chunking keeps each tx
    /// within the tx memory limit. Chunks commit as they go, but the store
    /// is not handed out until the last, so nobody sees a partial rebuild.
    /// Each tx reports [`WriteTx::rebuilding`], and [`Check`](crate::Check)
    /// accepts everything in one: a chunk can hold a state the rows never
    /// passed through, such as a sum with only its negative rows so far.
    /// The rows were checked when first written; a check added since is
    /// enforced only on later writes.
    ///
    /// If `replay` or a commit fails, open fails, and the next open starts
    /// the rebuild over.
    pub fn build_with(
        self,
        mut replay: impl FnMut(&mut WriteTx<'_>) -> Result<bool>,
    ) -> Result<Store> {
        self.open(Some(&mut replay))
    }

    pub(super) fn open(self, rebuild: Option<&mut Replay<'_>>) -> Result<Store> {
        let StoreBuilder {
            durability_policy: dp,
            dir,
            keyspaces,
            opts,
            graph,
        } = self;
        dp.validate().map_err(Error::Policy)?;
        let opts = opts.resolved();
        opts.validate()?;

        // the fingerprint covers which keyspaces are derived, and as what
        let mut fp = Fingerprint::new();
        fp.write_u64(graph);
        let mut derived: Vec<(&u64, &KeyspaceDecl)> =
            keyspaces.iter().filter(|(_, d)| d.derived).collect();
        derived.sort_unstable_by_key(|(id, _)| **id);
        for (id, d) in derived {
            fp.write_u64(*id);
            fp.write_str(d.key);
            fp.write_str(d.merge);
        }
        let fp = fp.finish();

        let Some(dir) = dir else {
            if !matches!(dp.write, Never) {
                return Err(Error::Policy("writing a WAL needs a directory"));
            }
            let mut m = Manifest::new(dp);
            m.graph = Some(fp);
            let writer = StoreWriter::new(None, m, None, dp);
            return Ok(Store::new(
                Replayed::default(),
                Vec::new(),
                writer,
                keyspaces,
                opts,
                0,
                None,
            ));
        };

        fs::create_dir_all(&dir).ctx("creating", &dir)?;
        let lock_path = dir.join("LOCK");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .ctx("opening", &lock_path)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Locked(dir)),
            Err(TryLockError::Error(e)) => return Err(e).ctx("locking", &lock_path),
        }

        // replay whatever is on disk even if this run will not write: the
        // previous run's settings may have differed from these
        let mut m = match manifest::read(&dir).ctx("reading manifest in", &dir)? {
            Some(text) => Manifest::parse(&text).map_err(Error::Manifest)?,
            None => Manifest::new(dp),
        };
        for (id, decl) in &keyspaces {
            if let Some((key, merge)) = m.keyspaces.get(id)
                && (key.as_str(), merge.as_str()) != (decl.key, decl.merge)
            {
                return Err(Error::KeyspaceType {
                    id: *id,
                    recorded: (key.clone(), merge.clone()),
                    registered: (decl.key, decl.merge),
                });
            }
        }
        // derived keyspaces nobody declares any more; their data is ignored
        // until the rebuild truncates them
        let dropped: BTreeSet<u64> = m
            .derived
            .iter()
            .copied()
            .filter(|id| !keyspaces.contains_key(id))
            .collect();
        sweep_orphans(&dir, &m)?;
        let tables = open_tables(&dir, &m, &keyspaces, &dropped)?;
        let replayed = replay(&dir, &m, &keyspaces, &dropped)?;

        // appending after a torn tail would hide every later frame from
        // replay, so seal the segment at its last good frame instead
        if let Some(valid) = replayed.torn_active {
            m.segments.last_mut().expect("torn segment exists").sealed = Some(valid);
        }
        let wal = if matches!(dp.write, Never) {
            None
        } else {
            Some(match m.active() {
                Some(seg) => {
                    let path = dir.join(seg.file_name());
                    let file = OpenOptions::new()
                        .append(true)
                        .open(&path)
                        .ctx("opening WAL", &path)?;
                    let size = file.metadata().ctx("opening WAL", &path)?.len() as usize;
                    Wal {
                        path,
                        file,
                        #[cfg(test)]
                        fail_next_write: false,
                        buffer: Vec::new(),
                        size,
                    }
                }
                None => new_segment(&dir, &mut m)?,
            })
        };

        // what the next open would replay, which drives auto-checkpoints
        let mut wal_bytes = 0;
        for seg in &m.segments {
            let path = dir.join(seg.file_name());
            wal_bytes += match seg.sealed {
                Some(len) => len,
                None => fs::metadata(&path).ctx("reading", &path)?.len(),
            };
        }

        m.policy = dp;
        m.options = opts.named();
        // dropped keyspaces stay listed until truncated, so a crash before
        // then still knows to ignore them
        m.keyspaces.retain(|id, _| dropped.contains(id));
        m.keyspaces.extend(
            keyspaces
                .iter()
                .map(|(id, d)| (*id, (d.key.to_string(), d.merge.to_string()))),
        );
        m.derived = keyspaces
            .iter()
            .filter(|(_, d)| d.derived)
            .map(|(id, _)| *id)
            .chain(dropped.iter().copied())
            .collect();
        manifest::stage(&dir, &m).ctx("writing manifest in", &dir)?;
        manifest::install(&dir).map_err(|e| match e {
            InstallError::Unchanged(e) | InstallError::Uncertain(e) => {
                Error::io(format!("installing manifest in {}", dir.display()), e)
            }
        })?;

        let writer = StoreWriter::new(Some(dir), m, wal, dp);
        let store = Store::new(
            replayed,
            tables,
            writer,
            keyspaces,
            opts,
            wal_bytes,
            Some(lock),
        );
        // the previous run may have left too much WAL, memtable, or tables
        if store.maintenance_due() {
            store.request_maintenance();
        }
        store.reconcile_graph(fp, &dropped, rebuild)?;
        Ok(store)
    }
}

/// Feeds one chunk of base state to the graph; see
/// [`StoreBuilder::build_with`].
pub(super) type Replay<'r> = dyn FnMut(&mut WriteTx<'_>) -> Result<bool> + 'r;
