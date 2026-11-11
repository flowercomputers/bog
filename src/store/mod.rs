use std::{
    any::{Any, type_name},
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, btree_map},
    fs::{self, File, OpenOptions, TryLockError},
    io::Write,
    marker::PhantomData,
    ops::{Bound, Deref, Range, RangeBounds},
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

mod builder;
mod checkpoint;
mod commit;
mod compact;
mod fold;
mod keyspace;
mod layer;
mod maintenance;
mod options;
mod recovery;
mod snapshot;
mod wal;
mod writer;

pub use builder::*;
use compact::*;
use keyspace::*;
use layer::*;
use maintenance::*;
pub use options::*;
use recovery::*;
pub use snapshot::*;
use wal::*;
use writer::*;

pub mod codec;
pub mod cursor;
pub mod index;
pub(crate) mod manifest;
pub mod merge;
pub mod subscribe;
pub(crate) mod table;
pub mod txn;

use arc_swap::ArcSwap;

use crate::{
    codec::{Codec, DecodeError, OrderedCodec, ResidentSize, decode_all},
    cursor::{Cursor, Source, empty_bounds},
    error::{Error, IoContext, Result},
    graph::{DEFAULT_GRAPH_RETAIN, Fingerprint},
    index::{Aux, EncodedBuilder, IndexKind, IndexedRun, LayerIndexes, Run, RunIndex, table_index},
    merge::Merge,
    store::DurabilityPolicyCadence::Never,
    store::manifest::{Checkpoint, InstallError, MANIFEST_TMP, Manifest, Segment},
    store::table::{BlockCache, DEFAULT_BLOCK_SIZE, TOMBSTONE, Table, TableBuilder, TableCursor},
    subscribe::{Publish, Senders, Subscription},
    tx::{KeyspaceId, OverlayRun, TxState, WriteTx},
};

/// An open store: a handle on its [`Shared`] state. Dropping it stops and
/// joins the maintenance thread, if one was started.
pub struct Store {
    shared: Arc<Shared>,
}

impl Deref for Store {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.shared
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Relaxed);
        {
            let mut st = self.shared.maintenance.lock();
            st.shutdown = true;
            self.shared.maintenance.wake.notify_all();
            self.shared.maintenance.progress.notify_all();
        }
        let worker = self
            .shared
            .worker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            // a panic there was already recorded; nothing more to report
            let _ = worker.join();
        }
    }
}

/// Everything an open store holds, shared by the [`Store`] handle and its
/// maintenance thread.
pub struct Shared {
    /// The latest committed state. Commits, checkpoints, and maintenance
    /// store into it, always while holding `writer`.
    current: ArcSwap<Version>,
    writer: Mutex<StoreWriter>,
    /// Live snapshot count per epoch; the minimum bounds which layers
    /// could be folded together.
    active_snapshots: Mutex<BTreeMap<u64, usize>>,
    keyspaces: HashMap<u64, KeyspaceDecl>,
    /// Checkpoint blocks, shared by every table and snapshot.
    cache: BlockCache,
    opts: StoreOptions,
    /// WAL bytes the next open would replay. Written under the writer
    /// lock; atomic so stats can read it while a tx is open.
    wal_bytes: AtomicU64,
    /// Subscribers by keyspace id. Held from before a commit is
    /// published until its deltas are sent, and while a subscriber takes
    /// its snapshot, so each sees exactly the commits after its snapshot.
    /// Taken after the writer lock, never before.
    subscribers: Mutex<HashMap<u64, Box<dyn Publish>>>,
    /// Estimated bytes of changes sent to subscriptions and not yet
    /// received.
    subscription_bytes: Arc<AtomicUsize>,
    /// Estimated bytes the open write tx holds.
    tx_bytes: Arc<AtomicUsize>,
    /// The WAL's write buffer, as of the last commit.
    wal_buffer_bytes: AtomicUsize,
    /// Bytes of buffers a `Db`'s graph holds between runs, which it
    /// publishes here.
    graph_bytes: Arc<AtomicUsize>,
    /// Every version a live snapshot was taken of, so stats can count
    /// the layers only they still hold. Dead ones are dropped as new
    /// ones are added.
    snapshot_versions: Mutex<Vec<Weak<Version>>>,
    /// Called with the epoch each checkpoint covers through.
    checkpoint_hooks: Mutex<Vec<Box<dyn FnMut(u64) + Send>>>,
    /// Held while compacting, so one compaction runs at a time.
    compaction: Mutex<()>,
    /// Held while folding or checkpointing. A checkpoint drops the layers
    /// it wrote by epoch when it installs, so no fold may merge them with
    /// newer ones in between.
    layer_work: Mutex<()>,
    /// Whether there is a directory to checkpoint to.
    durable: bool,
    maintenance: Maintenance,
    /// The maintenance thread, started by the first request for it.
    worker: Mutex<Option<JoinHandle<()>>>,
    /// This, for handing to the maintenance thread.
    me: Weak<Shared>,
    /// Which thread holds the open write tx, if any, as a [`thread_tag`],
    /// so calls that would wait on that tx can panic instead of deadlock.
    tx_thread: AtomicU64,
    /// Set when the store is dropped; long maintenance work checks it.
    shutdown: AtomicBool,
    /// Make every compaction fail, to test what a failed one does.
    #[cfg(test)]
    fail_merges: AtomicBool,
    /// Held for the store's life; released when the file is dropped.
    _lock: Option<File>,
}

impl Store {
    fn new(
        replayed: Replayed,
        tables: Vec<Arc<Table>>,
        writer: StoreWriter,
        keyspaces: HashMap<u64, KeyspaceDecl>,
        opts: StoreOptions,
        wal_bytes: u64,
        lock: Option<File>,
    ) -> Self {
        let Replayed { epoch, wal, .. } = replayed;
        let wal: HashMap<u64, LayerTree> = wal
            .into_iter()
            .map(|(ks, tree)| {
                let decl = &keyspaces[&ks];
                let bytes = (decl.measure)(&*tree);
                let tree = Arc::from(tree);
                let indexes = Arc::new(LayerIndexes::new(decl.indexes.len()));
                (
                    ks,
                    LayerTree {
                        tree,
                        bytes,
                        indexes,
                    },
                )
            })
            .collect();
        let bytes = wal.values().map(|t| t.bytes).sum();
        let layers = if wal.is_empty() {
            Vec::new()
        } else {
            vec![Arc::new(Layer {
                epoch,
                trees: wal,
                bytes,
            })]
        };
        let shared = Arc::new_cyclic(|me| Shared {
            current: ArcSwap::from_pointee(Version {
                epoch,
                layers,
                tables,
                resident_bytes: bytes,
                floors: Arc::new(writer.manifest.floors.clone()),
                frozen: None,
            }),
            durable: writer.dir.is_some(),
            writer: Mutex::new(writer),
            active_snapshots: Mutex::new(BTreeMap::new()),
            keyspaces,
            cache: BlockCache::new(opts.cache_size),
            opts,
            wal_bytes: AtomicU64::new(wal_bytes),
            subscribers: Mutex::new(HashMap::new()),
            subscription_bytes: Arc::default(),
            tx_bytes: Arc::default(),
            wal_buffer_bytes: AtomicUsize::new(0),
            graph_bytes: Arc::default(),
            snapshot_versions: Mutex::new(Vec::new()),
            checkpoint_hooks: Mutex::new(Vec::new()),
            compaction: Mutex::new(()),
            layer_work: Mutex::new(()),
            maintenance: Maintenance::default(),
            worker: Mutex::new(None),
            me: me.clone(),
            tx_thread: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            #[cfg(test)]
            fail_merges: AtomicBool::new(false),
            _lock: lock,
        });
        Store { shared }
    }
}

impl Shared {
    /// The writer. A panic while a tx held it does not poison it: a tx
    /// changes the writer only inside `finish`, and real failures there
    /// poison the store explicitly.
    fn lock_writer(&self) -> MutexGuard<'_, StoreWriter> {
        self.writer.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn snapshots(&self) -> MutexGuard<'_, BTreeMap<u64, usize>> {
        self.active_snapshots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn stats(&self) -> MemoryStats {
        let version = self.current.load();
        MemoryStats {
            memtable_bytes: version.resident_bytes,
            wal_bytes: self.wal_bytes.load(Relaxed),
            cache_bytes: self.cache.used(),
            index_bytes: version.tables.iter().map(|t| t.index_bytes()).sum(),
            run_index_bytes: self.run_index_bytes(&version),
            tables: version.tables.len(),
            layers: version.layers.len(),
            pinned_bytes: self.pinned_bytes(&version),
            tx_bytes: self.tx_bytes.load(Relaxed),
            wal_buffer_bytes: self.wal_buffer_bytes.load(Relaxed),
            subscription_bytes: self.subscription_bytes.load(Relaxed),
            graph_bytes: self.graph_bytes.load(Relaxed),
        }
    }

    /// The size limits in force: the options the store was built with,
    /// [resolved](StoreOptions::resolved).
    pub fn options(&self) -> &StoreOptions {
        &self.opts
    }

    /// Where a `Db` publishes what its graph holds between runs.
    pub(crate) fn graph_gauge(&self) -> Arc<AtomicUsize> {
        self.graph_bytes.clone()
    }

    /// The estimated bytes of layers that are not in `current` and that
    /// a live snapshot still holds, each counted once.
    fn pinned_bytes(&self, current: &Version) -> usize {
        let versions = self
            .snapshot_versions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut seen: Vec<*const Layer> = current.layers.iter().map(Arc::as_ptr).collect();
        let mut bytes = 0;
        for version in versions.iter().filter_map(Weak::upgrade) {
            for layer in &version.layers {
                let ptr = Arc::as_ptr(layer);
                if !seen.contains(&ptr) {
                    seen.push(ptr);
                    bytes += layer.bytes;
                }
            }
        }
        bytes
    }

    fn run_index_bytes(&self, version: &Version) -> usize {
        let kinds =
            |ks: u64| -> &[IndexKind] { self.keyspaces.get(&ks).map_or(&[], |d| &d.indexes[..]) };
        let layers: usize = version
            .layers
            .iter()
            .flat_map(|l| l.trees.iter())
            .map(|(ks, t)| t.indexes.bytes(kinds(*ks)))
            .sum();
        let tables: usize = version
            .tables
            .iter()
            .map(|t| t.indexes().bytes(kinds))
            .sum();
        layers + tables
    }

    /// Whether the WAL or the memtable is at `factor` times its limit, and
    /// there is somewhere to checkpoint to. At 1 a checkpoint is due; at 2
    /// commits wait for one.
    fn over_limits(&self, factor: u64) -> bool {
        // WAL bytes imply layers to checkpoint, and are read without the
        // version, which a fold in place briefly replaces with an empty one
        let version = self.current.load();
        self.durable
            && (self.wal_bytes.load(Relaxed) >= self.opts.wal_limit.saturating_mul(factor)
                || version.resident_bytes as u64
                    >= (self.opts.memtable_limit as u64).saturating_mul(factor))
    }

    /// Panics if this thread holds the open write tx: `what` would wait
    /// for it to finish, which it never would.
    fn assert_no_tx_here(&self, what: &str) {
        assert!(
            self.tx_thread.load(Relaxed) != thread_tag(),
            "{what} called while this thread holds a write tx, which it would wait on forever"
        );
    }

    /// Panics unless `id` was declared on the builder with these types.
    pub(crate) fn assert_declared<K: 'static, M: Merge>(&self, id: u64) {
        match self.keyspaces.get(&id) {
            Some(d) if (d.key, d.merge) == (type_name::<K>(), type_name::<M>()) => {}
            Some(d) => panic!(
                "keyspace {id} declared as ({}, {}) but opened as ({}, {})",
                d.key,
                d.merge,
                type_name::<K>(),
                type_name::<M>()
            ),
            None => panic!("keyspace {id} not declared; register it with StoreBuilder::keyspace"),
        }
    }
}

impl Store {
    pub fn snapshot(&self) -> Snapshot<'_> {
        let mut map = self.snapshots();
        let version = self.current.load_full();
        *map.entry(version.epoch).or_default() += 1;
        drop(map);
        let mut versions = self
            .snapshot_versions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !versions
            .last()
            .is_some_and(|v| std::ptr::eq(v.as_ptr(), Arc::as_ptr(&version)))
        {
            versions.retain(|v| v.strong_count() > 0);
            versions.push(Arc::downgrade(&version));
        }
        drop(versions);
        Snapshot {
            version,
            store: self,
        }
    }

    /// Open a write transaction over the latest committed state. Blocks
    /// while another write transaction is open. Dropping the tx without
    /// [`finish`](Self::finish) aborts it. Fails if the store is poisoned.
    /// Subscribe to `keyspace`: the latest snapshot, and every commit's
    /// net change to the keyspace after it, in order. See
    /// [`subscribe`].
    pub fn subscribe<K, M>(&self, keyspace: KeyspaceId<K, M>) -> (Snapshot<'_>, Subscription<K, M>)
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Send + Sync,
        M::Diff: Send + Sync,
    {
        self.assert_declared::<K, M>(keyspace.raw());
        let mut subscribers = self
            .subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // taken under the lock: a commit not in it has yet to send
        let snapshot = self.snapshot();
        let after = snapshot.epoch();
        let sub = self.subscribe_locked(&mut subscribers, keyspace, after);
        (snapshot, sub)
    }

    /// Subscribe to `keyspace` from a snapshot already held, at epoch
    /// `after`: every commit after it that has not yet been published.
    /// A commit between the snapshot and this call is published only if
    /// its publishing has not yet happened, so take the snapshot and
    /// subscribe close together.
    pub(crate) fn subscribe_from<K, M>(
        &self,
        keyspace: KeyspaceId<K, M>,
        after: u64,
    ) -> Subscription<K, M>
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Send + Sync,
        M::Diff: Send + Sync,
    {
        self.assert_declared::<K, M>(keyspace.raw());
        let mut subscribers = self
            .subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.subscribe_locked(&mut subscribers, keyspace, after)
    }

    fn subscribe_locked<K, M>(
        &self,
        subscribers: &mut HashMap<u64, Box<dyn Publish>>,
        keyspace: KeyspaceId<K, M>,
        after: u64,
    ) -> Subscription<K, M>
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Send + Sync,
        M::Diff: Send + Sync,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        subscribers
            .entry(keyspace.raw())
            .or_insert_with(|| Box::new(Senders::<K, M>::new(self.subscription_bytes.clone())))
            .as_any_mut()
            .downcast_mut::<Senders<K, M>>()
            .expect("keyspace id reused with a different type")
            .senders
            .push(tx);
        Subscription { rx, after }
    }

    /// Call `hook` after every checkpoint, with the epoch it covers
    /// through, on whichever thread wrote it: usually the maintenance
    /// thread. It must not start a transaction or a checkpoint.
    pub fn on_checkpoint(&self, hook: impl FnMut(u64) + Send + 'static) {
        self.checkpoint_hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::new(hook));
    }

    pub fn begin(&self) -> Result<WriteTx<'_>> {
        self.assert_no_tx_here("begin");
        // lock first: the snapshot must not be older than the last commit
        let guard = self.lock_writer();
        self.tx_thread.store(thread_tag(), Relaxed);
        let writer = WriterGuard {
            guard,
            owner: &self.tx_thread,
        };
        writer.check()?;
        Ok(WriteTx::new(
            self.snapshot(),
            writer,
            self.opts.tx_memory_limit,
            self.tx_bytes.clone(),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::drop_non_drop)] // drops mark where a borrow's use ends
mod tests;
