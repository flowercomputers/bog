use std::{
    any::{Any, type_name},
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{self, File, OpenOptions, TryLockError},
    io::Write,
    ops::{Bound, Range, RangeBounds},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use arc_swap::ArcSwap;

use crate::{
    Change,
    codec::{Codec, DecodeError, OrderedCodec, ResidentSize, decode_all},
    cursor::{Cursor, Source},
    error::{Error, IoContext, Result},
    graph::Fingerprint,
    manifest::{self, Checkpoint, InstallError, MANIFEST_TMP, Manifest, Segment},
    merge::Merge,
    store::DurabilityPolicyCadence::Never,
    table::{BlockCache, DEFAULT_BLOCK_SIZE, TOMBSTONE, Table, TableBuilder, TableCursor},
    tx::{KeyspaceId, TxState, WriteTx},
};

/// Size limits. Memory use is estimated with [`ResidentSize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StoreOptions {
    /// Checkpoint once the WAL that open would replay reaches this many
    /// bytes.
    pub wal_limit: u64,
    /// Checkpoint once committed data held in memory reaches this many
    /// bytes. In-memory stores have nowhere to checkpoint to, so for them
    /// this does nothing.
    pub memtable_limit: usize,
    /// Kill a write tx once its writes hold this many bytes.
    pub tx_memory_limit: usize,
    /// Target size of a checkpoint table's data blocks: the unit read from
    /// disk and cached.
    pub block_size: usize,
    /// Bytes of table blocks to keep cached.
    pub cache_size: usize,
    /// Compact once there are more checkpoint tables than this. Every
    /// table is checked on a read that misses memory, so this bounds read
    /// cost.
    pub max_tables: usize,
    /// How many adjacent tables one compaction merges.
    pub compaction_width: usize,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            wal_limit: 64 << 20,
            memtable_limit: 64 << 20,
            tx_memory_limit: 64 << 20,
            block_size: DEFAULT_BLOCK_SIZE,
            cache_size: 32 << 20,
            max_tables: 8,
            compaction_width: 4,
        }
    }
}

impl StoreOptions {
    fn validate(&self) -> Result<()> {
        if self.max_tables == 0 {
            return Err(Error::InvalidOption("max tables must be at least one"));
        }
        if self.compaction_width < 2 {
            return Err(Error::InvalidOption(
                "compaction must merge at least two tables",
            ));
        }
        if self.block_size == 0 {
            return Err(Error::InvalidOption("block size must be at least one byte"));
        }
        if self.block_size > u32::MAX as usize / 2 {
            return Err(Error::InvalidOption("block size must fit a u32 block"));
        }
        Ok(())
    }

    /// As recorded in the manifest.
    fn named(&self) -> BTreeMap<String, u64> {
        [
            ("wal-limit", self.wal_limit),
            ("memtable-limit", self.memtable_limit as u64),
            ("tx-memory-limit", self.tx_memory_limit as u64),
            ("block-size", self.block_size as u64),
            ("cache-size", self.cache_size as u64),
            ("max-tables", self.max_tables as u64),
            ("compaction-width", self.compaction_width as u64),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
}

/// Where the store's memory goes, and how much WAL the next open replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemoryStats {
    /// Committed data held in memory, not yet checkpointed. Layers pinned
    /// only by old snapshots are not counted.
    pub memtable_bytes: usize,
    /// WAL bytes the next open would replay.
    pub wal_bytes: u64,
    /// Table blocks in the cache.
    pub cache_bytes: usize,
    /// Table indexes, which stay in memory while the table is open.
    pub index_bytes: usize,
    pub tables: usize,
}

/// Estimated bookkeeping a `BTreeMap` spends per entry beyond the key and
/// value themselves.
pub(crate) const ENTRY_OVERHEAD: usize = 16;

/// Estimated resident bytes of one committed entry.
pub(crate) fn resident_entry_bytes<K: ResidentSize, V: ResidentSize>(
    key: &K,
    value: &Option<V>,
) -> usize {
    key.resident_size() + value.resident_size() + ENTRY_OVERHEAD
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DurabilityPolicyCadence {
    Never,
    Always,
    Windowed(core::time::Duration),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DurabilityPolicy {
    // Write(), i.e. the os has the data, durable to process crash
    pub(crate) write: DurabilityPolicyCadence,
    // Fsync/Fullsync; data is on the disk/durable to power loss
    pub(crate) fsync: DurabilityPolicyCadence,
}
impl DurabilityPolicy {
    pub const NONE: Self = Self {
        write: DurabilityPolicyCadence::Never,
        fsync: DurabilityPolicyCadence::Never,
    };
    pub const RELAXED: Self = Self {
        write: DurabilityPolicyCadence::Always,
        fsync: DurabilityPolicyCadence::Never,
    };
    pub const STRICT: Self = Self {
        write: DurabilityPolicyCadence::Always,
        fsync: DurabilityPolicyCadence::Always,
    };
    pub const fn periodic_sync(window: std::time::Duration) -> Self {
        Self {
            write: DurabilityPolicyCadence::Always,
            fsync: DurabilityPolicyCadence::Windowed(window),
        }
    }
    pub const fn batched(w: std::time::Duration) -> Self {
        Self {
            write: DurabilityPolicyCadence::Windowed(w),
            fsync: DurabilityPolicyCadence::Never,
        }
    }
    fn validate(&self) -> Result<(), &'static str> {
        if matches!(&self.write, DurabilityPolicyCadence::Never)
            && !matches!(&self.fsync, DurabilityPolicyCadence::Never)
        {
            return Err("invalid durability policy: cannot fsync unwritten data");
            // Err(Error::DurabilityPolicy(self.disk))
        }

        match (&self.write, &self.fsync) {
            (DurabilityPolicyCadence::Windowed(w), DurabilityPolicyCadence::Windowed(f))
                if f < w =>
            {
                return Err("fsync duration is less than write duration");
            }
            _ => (),
        }

        Ok(())
    }

    // returns (should_write, should_fsync)
    fn should(
        &self,
        now: Instant,
        oldest_unwritten: Option<Instant>,
        oldest_unsynced: Option<Instant>,
    ) -> (bool, bool) {
        let should_write = match &self.write {
            DurabilityPolicyCadence::Never => return (false, false),
            DurabilityPolicyCadence::Always => true,
            DurabilityPolicyCadence::Windowed(duration) => {
                if let Some(last) = oldest_unwritten {
                    now.duration_since(last) >= *duration
                } else {
                    true
                }
            }
        };
        let should_fsync = match &self.fsync {
            DurabilityPolicyCadence::Never => false,
            DurabilityPolicyCadence::Always => true,
            DurabilityPolicyCadence::Windowed(duration) => {
                if let Some(last) = oldest_unsynced {
                    now.duration_since(last) >= *duration
                } else {
                    true
                }
            }
        };
        (should_write || should_fsync, should_fsync)
    }
}
impl Default for DurabilityPolicy {
    fn default() -> Self {
        Self::RELAXED
    }
}

/// One committed keyspace in one epoch: every key the epoch wrote, with
/// `None` as a tombstone. Commit builds these; reads downcast to them.
pub(crate) type Resident<K, V> = BTreeMap<K, Option<V>>;

/// A type-erased `Resident<K, V>`. Shared, so folding layers reuses the
/// trees of keyspaces only one side touched.
type Tree = Arc<dyn Any + Send + Sync>;

/// One keyspace's tree in a layer, with its estimated resident bytes.
#[derive(Clone)]
struct LayerTree {
    tree: Tree,
    bytes: usize,
}

/// What one or more consecutive epochs committed, keyed by keyspace id:
/// the newest value of every key they wrote. Commits push a layer per
/// epoch, and [`fold_layers`](Store::fold_layers) merges neighbors so
/// there are only logarithmically many.
struct Layer {
    /// The newest epoch it holds.
    epoch: u64,
    trees: HashMap<u64, LayerTree>,
    /// Estimated resident bytes of `trees`.
    bytes: usize,
}

/// Fold layers while the older of the last two is no more than this many
/// times the newer's size. Layer sizes then grow geometrically from newest
/// to oldest, so there are O(log n) of them, and each entry is copied
/// O(log n) times over its life in memory.
const FOLD_RATIO: usize = 2;

/// An immutable view of committed state. Commit builds a new one and swaps
/// it in; a snapshot holds one for its whole life, so its epoch and its
/// contents always agree.
///
/// Every layer is newer than every table: tables hold epochs through the
/// last checkpoint, and layers hold epochs after it.
struct Version {
    epoch: u64,
    /// In memory, oldest first.
    layers: Vec<Arc<Layer>>,
    /// Checkpoints on disk, oldest first, read through the block cache.
    tables: Vec<Arc<Table>>,
    /// Estimated resident bytes of `layers`.
    resident_bytes: usize,
    /// Keyspace id to the epoch it was truncated at; tables no newer than
    /// that hold nothing visible for it. Mirrors the manifest's floors.
    floors: Arc<BTreeMap<u64, u64>>,
}

impl Version {
    /// The tables that can hold visible data for `ks`, newest first.
    fn tables_for(&self, ks: u64) -> impl Iterator<Item = &Arc<Table>> {
        let floor = self.floors.get(&ks).copied().unwrap_or(0);
        self.tables
            .iter()
            .rev()
            .take_while(move |t| t.epoch() > floor)
    }
}

struct Wal {
    path: PathBuf,
    file: File,
    /// Make the next write fail, to test what a failed write does.
    #[cfg(test)]
    fail_next_write: bool,
    buffer: Vec<u8>,
    size: usize,
}
impl Wal {
    /// Buffer one frame holding a whole transaction, so replay sees either
    /// all of a tx or none of it: `crc32 | epoch u64 | len u64 | payload`,
    /// little endian, with the crc covering everything after it. `payload`
    /// writes straight into the buffer, so the tx is never copied whole.
    fn append_with(&mut self, epoch: u64, payload: impl FnOnce(&mut Vec<u8>)) {
        const HEAD: usize = 4 + 8 + 8;
        let start = self.buffer.len();
        self.buffer.extend([0u8; HEAD]);
        payload(&mut self.buffer);
        let len = (self.buffer.len() - start - HEAD) as u64;
        self.buffer[start + 4..start + 12].copy_from_slice(&epoch.to_le_bytes());
        self.buffer[start + 12..start + 20].copy_from_slice(&len.to_le_bytes());
        let crc = crc32fast::hash(&self.buffer[start + 4..]);
        self.buffer[start..start + 4].copy_from_slice(&crc.to_le_bytes());
        self.size += self.buffer.len() - start;
    }

    fn flush(&mut self, write: bool, sync: bool) -> Result<()> {
        #[cfg(test)]
        if write && std::mem::take(&mut self.fail_next_write) {
            return Err(std::io::Error::other("injected write failure"))
                .ctx("writing WAL", &self.path);
        }
        if write && !self.buffer.is_empty() {
            self.file
                .write_all(&self.buffer)
                .ctx("writing WAL", &self.path)?;
            self.buffer.clear();
        }
        if sync {
            self.file.sync_data().ctx("syncing WAL", &self.path)?;
        }
        Ok(())
    }
}
/// Append one payload section, `keyspace u64 | len u64 | body`, with the
/// body written by `body`.
fn encode_section<R>(ks: u64, out: &mut Vec<u8>, body: impl FnOnce(&mut Vec<u8>) -> R) -> R {
    out.extend(ks.to_le_bytes());
    let len_at = out.len();
    out.extend(0u64.to_le_bytes());
    let r = body(out);
    let len = (out.len() - len_at - 8) as u64;
    out[len_at..len_at + 8].copy_from_slice(&len.to_le_bytes());
    r
}

impl Drop for Wal {
    /// A clean shutdown hands buffered frames to the OS. It does not fsync:
    /// that is the policy's call.
    fn drop(&mut self) {
        // nowhere to report a failure; the next open replays what landed
        let _ = self.flush(true, false);
    }
}

/// State only the writer touches. Held locked for the life of a
/// [`WriteTx`], which is what makes the store single-writer.
pub(crate) struct StoreWriter {
    /// `None` for an in-memory store.
    dir: Option<PathBuf>,
    /// The manifest as last written. Only open and checkpoint change it,
    /// and checkpoint does so under this lock.
    manifest: Manifest,
    wal: Option<Wal>,
    dp: DurabilityPolicy,
    last_write: Option<Instant>,
    last_fsync: Option<Instant>,
    /// Set when a failure leaves what is on disk uncertain; every later
    /// write fails with it until the store is reopened.
    poisoned: Option<Error>,
}
impl StoreWriter {
    fn new(
        dir: Option<PathBuf>,
        manifest: Manifest,
        wal: Option<Wal>,
        dp: DurabilityPolicy,
    ) -> Self {
        StoreWriter {
            dir,
            manifest,
            wal,
            dp,
            last_write: None,
            last_fsync: None,
            poisoned: None,
        }
    }

    fn check(&self) -> Result<()> {
        match &self.poisoned {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Refuse every later write: `cause` left what is on disk uncertain.
    ///
    /// Buffered WAL frames are discarded rather than written on drop: after
    /// a failed write the file may end in a partial frame, and replay stops
    /// there, so nothing appended after it would be seen anyway, and the tx
    /// that failed must not reappear on reopen.
    fn poison(&mut self, cause: Error) -> Error {
        if let Some(wal) = self.wal.as_mut() {
            wal.buffer.clear();
        }
        let e = self
            .poisoned
            .get_or_insert(Error::Poisoned(Arc::new(cause)))
            .clone();
        e
    }

    /// Make `next` the live manifest. A failure that leaves the old one
    /// live is an ordinary error; one that leaves it unknown which survives
    /// a crash poisons the store.
    fn install_manifest(&mut self, dir: &Path, next: Manifest) -> Result<()> {
        manifest::stage(dir, &next).ctx("writing manifest in", dir)?;
        match manifest::install(dir) {
            Ok(()) => {}
            Err(InstallError::Unchanged(e)) => {
                return Err(Error::io(
                    format!("installing manifest in {}", dir.display()),
                    e,
                ));
            }
            Err(InstallError::Uncertain(e)) => {
                let cause = Error::io(format!("syncing {}", dir.display()), e);
                return Err(self.poison(cause));
            }
        }
        self.manifest = next;
        Ok(())
    }

    /// Make `next` the manifest: installed on disk, or just kept for an
    /// in-memory store.
    fn update_manifest(&mut self, next: Manifest) -> Result<()> {
        match self.dir.clone() {
            Some(dir) => self.install_manifest(&dir, next),
            None => {
                self.manifest = next;
                Ok(())
            }
        }
    }

    /// Log one transaction, its payload written by `payload` (given `None`
    /// when there is no WAL), then write/fsync as the policy says. Returns
    /// the bytes appended to the WAL.
    // todo: windowed policies only flush when the next tx commits; a quiet
    // store needs a clock source to flush on its own.
    ///
    /// A failed write or fsync poisons the store: the frame, and frames
    /// buffered before it, may or may not be on disk.
    fn log_with(&mut self, epoch: u64, payload: impl FnOnce(Option<&mut Vec<u8>>)) -> Result<u64> {
        let Some(wal) = self.wal.as_mut() else {
            payload(None);
            return Ok(0);
        };
        let before = wal.size;
        wal.append_with(epoch, |buf| payload(Some(buf)));
        let appended = (wal.size - before) as u64;
        let now = Instant::now();
        let (write, sync) = self.dp.should(now, self.last_write, self.last_fsync);
        if let Err(e) = wal.flush(write, sync) {
            return Err(self.poison(e));
        }
        if write {
            self.last_write = Some(now);
        }
        if sync {
            // to accurately reflect the time it actually took to fsync
            self.last_fsync = Some(Instant::now());
        }
        Ok(appended)
    }
}

pub struct Snapshot<'s> {
    version: Arc<Version>,
    store: &'s Store,
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
    pub(crate) fn get<K, M>(
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
        let hit = self
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
            .find_map(|tree| tree.get(key));
        if let Some(slot) = hit {
            return Ok(slot.as_ref().map(Cow::Borrowed));
        }
        let mut tables = self.version.tables_for(ks).peekable();
        if tables.peek().is_none() {
            return Ok(None);
        }

        let mut probe = Vec::with_capacity(key.encode_ordered_size());
        key.encode_ordered(&mut probe);
        for table in tables {
            if let Some((block, range)) = table.get(ks, &probe, &self.store.cache)? {
                let value = decode_all::<Option<M::Value>>(&block[range])
                    .map_err(|e| Error::corrupt(table.path(), 0, format!("keyspace {ks}: {e}")))?;
                return Ok(value.map(Cow::Owned));
            }
        }
        Ok(None)
    }

    /// The live entries of `keyspace` with keys in `range`, in key order.
    pub(crate) fn range<K, M>(
        &self,
        keyspace: KeyspaceId<K, M>,
        range: impl RangeBounds<K>,
    ) -> Cursor<'_, K, M>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
    {
        self.cursor(
            keyspace,
            None,
            range.start_bound(),
            range.end_bound().cloned(),
            None,
        )
    }

    /// A merged cursor over `keyspace`, with `overlay`, a write tx's own
    /// changes, on top when given.
    pub(crate) fn cursor<'a, K, M>(
        &'a self,
        keyspace: KeyspaceId<K, M>,
        overlay: Option<&'a BTreeMap<K, Change<M>>>,
        lower: Bound<&K>,
        upper: Bound<K>,
        tx: Option<&'a TxState>,
    ) -> Cursor<'a, K, M>
    where
        K: OrderedCodec + Clone + 'static,
        M: Merge,
        M::Value: Codec,
    {
        let ks = keyspace.raw();
        let mut sources = Vec::new();
        if let Some(overlay) = overlay {
            sources.push(Source::overlay(overlay, lower));
        }
        for layer in self.version.layers.iter().rev() {
            if let Some(t) = layer.trees.get(&ks) {
                let tree = t
                    .tree
                    .downcast_ref::<Resident<K, M::Value>>()
                    .expect("keyspace id reused with a different type");
                sources.push(Source::layer(tree, lower));
            }
        }
        for table in self.version.tables_for(ks) {
            sources.push(Source::table(table.cursor(ks, Some(&self.store.cache)), ks));
        }
        Cursor::new(sources, lower, upper, tx)
    }
}

/// A keyspace's types, captured when it is registered with the builder.
/// Replay uses it to decode WAL sections; transactions use it to refuse
/// keyspaces nobody declared.
struct KeyspaceDecl {
    key: &'static str,
    merge: &'static str,
    /// Whether a graph computes it from other keyspaces, so it can be
    /// truncated and recomputed when the graph changes.
    derived: bool,
    /// Decode one WAL section and apply it over the keyspace's replayed
    /// tree, creating the tree on first use.
    replay: fn(&mut Option<Box<dyn Any + Send + Sync>>, &[u8]) -> Result<(), DecodeError>,
    /// Merge this keyspace's trees, newest first, into one sorted run of
    /// encoded rows.
    write_run:
        fn(&[&(dyn Any + Send + Sync)], &mut dyn FnMut(&[u8], &[u8]) -> Result<()>) -> Result<()>,
    /// Estimated resident bytes of one of this keyspace's trees.
    measure: fn(&(dyn Any + Send + Sync)) -> usize,
    /// Fold a newer tree over an older one; see [`fold_trees`].
    fold: fn(Option<&LayerTree>, Option<&LayerTree>, bool) -> LayerTree,
}

/// One tree holding `older` overwritten by `newer`, either of which may
/// be missing. With `drop_tombstones`, nothing lies beneath, so deletes
/// are applied rather than kept.
fn fold_trees<K, V>(
    older: Option<&LayerTree>,
    newer: Option<&LayerTree>,
    drop_tombstones: bool,
) -> LayerTree
where
    K: Ord + Clone + ResidentSize + Send + Sync + 'static,
    V: Clone + ResidentSize + Send + Sync + 'static,
{
    fn cast<K: 'static, V: 'static>(t: &LayerTree) -> &Resident<K, V> {
        t.tree
            .downcast_ref::<Resident<K, V>>()
            .expect("layer tree has the registered type")
    }
    let (mut tree, mut bytes) = match older {
        Some(t) => (cast::<K, V>(t).clone(), t.bytes),
        None => (Resident::new(), 0),
    };
    if drop_tombstones {
        tree.retain(|k, v| {
            if v.is_none() {
                bytes -= resident_entry_bytes(k, v);
            }
            v.is_some()
        });
    }
    for (k, v) in newer.map(cast::<K, V>).into_iter().flatten() {
        if drop_tombstones && v.is_none() {
            if let Some(old) = tree.remove(k) {
                bytes -= resident_entry_bytes(k, &old);
            }
            continue;
        }
        bytes += resident_entry_bytes(k, v);
        if let Some(old) = tree.insert(k.clone(), v.clone()) {
            bytes -= resident_entry_bytes(k, &old);
        }
    }
    LayerTree {
        tree: Arc::new(tree),
        bytes,
    }
}

fn measure_tree<K, V>(tree: &(dyn Any + Send + Sync)) -> usize
where
    K: ResidentSize + 'static,
    V: ResidentSize + 'static,
{
    tree.downcast_ref::<Resident<K, V>>()
        .expect("tree has the registered type")
        .iter()
        .map(|(k, v)| resident_entry_bytes(k, v))
        .sum()
}

fn replay_section<K, V>(
    tree: &mut Option<Box<dyn Any + Send + Sync>>,
    mut body: &[u8],
) -> Result<(), DecodeError>
where
    K: OrderedCodec + Send + Sync + 'static,
    V: Codec + Send + Sync + 'static,
{
    let tree = tree
        .get_or_insert_with(|| Box::new(Resident::<K, V>::new()))
        .downcast_mut::<Resident<K, V>>()
        .expect("replay tree has the registered type");
    while !body.is_empty() {
        let k = K::decode_ordered(&mut body)?;
        let v = Option::<V>::decode(&mut body)?;
        // keep tombstones: once disk runs exist they shadow older data
        tree.insert(k, v);
    }
    Ok(())
}

/// Feed `rows` the newest value of every key in `trees`, newest tree first,
/// as encoded `(key, Option<value>)` in key order.
fn write_run<K, V>(
    trees: &[&(dyn Any + Send + Sync)],
    rows: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
) -> Result<()>
where
    K: OrderedCodec + Send + Sync + 'static,
    V: Codec + Send + Sync + 'static,
{
    let mut merged: BTreeMap<&K, &Option<V>> = BTreeMap::new();
    for tree in trees {
        let tree = tree
            .downcast_ref::<Resident<K, V>>()
            .expect("layer tree has the registered type");
        for (k, v) in tree {
            // newest first, so the first sighting of a key wins
            merged.entry(k).or_insert(v);
        }
    }
    let (mut key, mut value) = (Vec::new(), Vec::new());
    for (k, v) in merged {
        key.clear();
        value.clear();
        k.encode_ordered(&mut key);
        // tombstones stay: older checkpoints below still hold the key
        v.encode(&mut value);
        rows(&key, &value)?;
    }
    Ok(())
}

pub(crate) struct StoreBuilder {
    durability_policy: DurabilityPolicy,
    dir: Option<PathBuf>,
    keyspaces: HashMap<u64, KeyspaceDecl>,
    opts: StoreOptions,
    graph: u64,
}

impl StoreBuilder {
    /// An in-memory store unless [`dir`](Self::dir) is set. Policies that
    /// write need a directory.
    pub(crate) fn new(durability_policy: DurabilityPolicy) -> Self {
        StoreBuilder {
            durability_policy,
            dir: None,
            keyspaces: HashMap::new(),
            opts: StoreOptions::default(),
            graph: 0,
        }
    }

    /// Replace every size limit at once.
    pub(crate) fn options(mut self, opts: StoreOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Adjust the size limits in place.
    pub(crate) fn options_with(mut self, f: impl FnOnce(&mut StoreOptions)) -> Self {
        f(&mut self.opts);
        self
    }

    /// Checkpoint once the WAL reaches this many bytes. Defaults to 64 MiB.
    pub(crate) fn wal_limit(mut self, bytes: u64) -> Self {
        self.opts.wal_limit = bytes;
        self
    }

    /// Checkpoint once committed data in memory reaches this many bytes.
    /// Defaults to 64 MiB.
    pub(crate) fn memtable_limit(mut self, bytes: usize) -> Self {
        self.opts.memtable_limit = bytes;
        self
    }

    /// Kill a write tx once its writes hold this many bytes. Defaults to
    /// 64 MiB.
    pub(crate) fn tx_memory_limit(mut self, bytes: usize) -> Self {
        self.opts.tx_memory_limit = bytes;
        self
    }

    /// Target size of a checkpoint table's blocks. Defaults to 4 KiB.
    pub(crate) fn block_size(mut self, bytes: usize) -> Self {
        self.opts.block_size = bytes;
        self
    }

    /// Bytes of checkpoint blocks to keep in memory. Defaults to 32 MiB.
    pub(crate) fn cache_size(mut self, bytes: usize) -> Self {
        self.opts.cache_size = bytes;
        self
    }

    /// Keep the store in `dir`, creating it if needed.
    pub(crate) fn dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// Declare a base keyspace, one the application writes. Every keyspace
    /// a transaction opens must be declared, and every keyspace in the WAL
    /// must be declared with the types it was written with.
    pub(crate) fn keyspace<K, M>(self, id: KeyspaceId<K, M>) -> Self
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
    pub(crate) fn derived<K, M>(self, id: KeyspaceId<K, M>) -> Self
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
    pub(crate) fn graph(mut self, fingerprint: u64) -> Self {
        self.graph = fingerprint;
        self
    }

    fn declare<K, M>(mut self, id: KeyspaceId<K, M>, derived: bool) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        let decl = KeyspaceDecl {
            key: type_name::<K>(),
            merge: type_name::<M>(),
            derived,
            replay: replay_section::<K, M::Value>,
            write_run: write_run::<K, M::Value>,
            measure: measure_tree::<K, M::Value>,
            fold: fold_trees::<K, M::Value>,
        };
        if let Some(prev) = self.keyspaces.get(&id.raw()) {
            assert!(
                (prev.key, prev.merge, prev.derived) == (decl.key, decl.merge, decl.derived),
                "keyspace {} declared twice differently",
                id.raw()
            );
        }
        self.keyspaces.insert(id.raw(), decl);
        self
    }

    /// Open the store. Fails with [`Error::RebuildRequired`] if derived
    /// state must be rebuilt; use [`build_with`](Self::build_with) then.
    pub(crate) fn build(self) -> Result<Store> {
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
    /// Constraints that depend on the order rows arrive in must be met by
    /// the order `replay` feeds them, or by feeding everything at once.
    ///
    /// If `replay` or a commit fails, open fails, and the next open starts
    /// the rebuild over.
    pub(crate) fn build_with(
        self,
        mut replay: impl FnMut(&mut WriteTx<'_>) -> Result<bool>,
    ) -> Result<Store> {
        self.open(Some(&mut replay))
    }

    fn open(self, rebuild: Option<&mut Replay<'_>>) -> Result<Store> {
        let StoreBuilder {
            durability_policy: dp,
            dir,
            keyspaces,
            opts,
            graph,
        } = self;
        dp.validate().map_err(Error::Policy)?;
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
        // the previous run may have left more than these limits allow
        {
            let mut writer = store.lock_writer();
            if store.over_limits(&writer) {
                store.checkpoint_locked(&mut writer)?;
            }
        }
        store.compact_while_over()?;
        store.reconcile_graph(fp, &dropped, rebuild)?;
        Ok(store)
    }
}

/// Feeds one chunk of base state to the graph; see
/// [`StoreBuilder::build_with`].
type Replay<'r> = dyn FnMut(&mut WriteTx<'_>) -> Result<bool> + 'r;

/// Create the next WAL segment and list it in `m` as active. The file is
/// created before the manifest names it; the manifest write syncs the
/// directory entry.
fn new_segment(dir: &Path, m: &mut Manifest) -> Result<Wal> {
    let seg = Segment {
        number: m.next_file,
        sealed: None,
    };
    let path = dir.join(seg.file_name());
    let file = OpenOptions::new()
        .append(true)
        .create_new(true)
        .open(&path)
        .ctx("creating WAL", &path)?;
    m.next_file += 1;
    m.segments.push(seg);
    Ok(Wal {
        path,
        file,
        #[cfg(test)]
        fail_next_write: false,
        buffer: Vec::new(),
        size: 0,
    })
}

/// Delete WAL segments and checkpoints the manifest does not name, and any
/// half-written manifest. A crash between creating a file and naming it,
/// or between dropping a file from the manifest and deleting it, leaves
/// these behind. Files with other names are left alone.
fn sweep_orphans(dir: &Path, m: &Manifest) -> Result<()> {
    let live: HashSet<String> = m.live_files().collect();
    for entry in fs::read_dir(dir).ctx("listing", dir)? {
        let entry = entry.ctx("listing", dir)?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let numbered = name.split_once('.').is_some_and(|(num, ext)| {
            matches!(ext, "wal" | "ckpt")
                && !num.is_empty()
                && num.bytes().all(|b| b.is_ascii_digit())
        });
        if (numbered && !live.contains(name)) || name == MANIFEST_TMP {
            fs::remove_file(entry.path()).ctx("removing orphan", &entry.path())?;
        }
    }
    Ok(())
}

/// Everything replay recovered.
#[derive(Default)]
struct Replayed {
    /// The newest epoch recovered, from the WAL or else the last checkpoint.
    epoch: u64,
    /// Everything in the WAL, folded into one tree per keyspace.
    wal: HashMap<u64, Box<dyn Any + Send + Sync>>,
    /// Valid length of the active segment, if it ends in a torn frame.
    torn_active: Option<u64>,
}

fn corrupt(path: &Path, offset: usize, why: impl Into<String>) -> Error {
    Error::corrupt(path, offset as u64, why)
}

fn frame_error(path: &Path, offset: usize) -> impl Fn(FrameError) -> Error + '_ {
    move |e| match e {
        FrameError::Unknown(id) => Error::UnknownKeyspace(id),
        FrameError::Decode(why) => corrupt(path, offset, why),
    }
}

/// Open every checkpoint's table. Only footers and indexes are read; data
/// blocks are paged in through the cache as reads need them.
fn open_tables(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
    dropped: &BTreeSet<u64>,
) -> Result<Vec<Arc<Table>>> {
    let mut tables = Vec::with_capacity(m.checkpoints.len());
    for ckpt in &m.checkpoints {
        let path = dir.join(ckpt.file_name());
        let table = Table::open(&path, ckpt.number)?;
        if table.epoch() != ckpt.epoch {
            return Err(corrupt(
                &path,
                0,
                format!(
                    "holds epoch {}, manifest says {}",
                    table.epoch(),
                    ckpt.epoch
                ),
            ));
        }
        let unknown = |ks: &u64| {
            !decls.contains_key(ks) && !dropped.contains(ks) && m.floor(*ks) < table.epoch()
        };
        if let Some(ks) = table.keyspaces().find(unknown) {
            return Err(Error::UnknownKeyspace(ks));
        }
        tables.push(Arc::new(table));
    }
    Ok(tables)
}

/// Replay every WAL segment, folding all frames into one tree per
/// keyspace. Nothing can hold a snapshot during open, so one layer is
/// equivalent to a layer per epoch, and reads stay cheap. The WAL holds
/// only epochs after the last checkpoint, so replay starts there.
fn replay(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
    dropped: &BTreeSet<u64>,
) -> Result<Replayed> {
    let mut out = Replayed {
        epoch: m.checkpointed(),
        ..Replayed::default()
    };
    for seg in &m.segments {
        let path = dir.join(seg.file_name());
        let bytes = fs::read(&path).ctx("reading WAL", &path)?;
        let limit = match seg.sealed {
            Some(len) if (len as usize) > bytes.len() => {
                return Err(corrupt(
                    &path,
                    bytes.len(),
                    format!("sealed at {len} bytes but shorter"),
                ));
            }
            Some(len) => len as usize,
            None => bytes.len(),
        };

        let mut at = 0;
        while at < limit {
            let (epoch, payload, used) = match parse_frame(&bytes[at..limit]) {
                Ok(frame) => frame,
                // a crash mid-append leaves a partial or unchecksummed frame
                // at the end of the active segment; everything before it is good
                Err(_) if seg.sealed.is_none() => {
                    out.torn_active = Some(at as u64);
                    break;
                }
                Err(why) => return Err(corrupt(&path, at, why)),
            };
            if epoch != out.epoch + 1 {
                return Err(corrupt(
                    &path,
                    at,
                    format!("epoch {epoch} follows epoch {}", out.epoch),
                ));
            }
            // truncated or dropped keyspaces' sections are skipped
            let skip = |ks: u64| dropped.contains(&ks) || m.floor(ks) >= epoch;
            apply_frame(payload, decls, &skip, &mut out.wal).map_err(frame_error(&path, at))?;
            out.epoch = epoch;
            at += used;
        }
    }
    Ok(out)
}

/// Split one frame off the front of `b`: `(epoch, payload, bytes used)`.
fn parse_frame(b: &[u8]) -> Result<(u64, &[u8], usize), String> {
    const HEAD: usize = 4 + 8 + 8;
    if b.len() < HEAD {
        return Err("truncated frame header".into());
    }
    let crc = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let epoch = u64::from_le_bytes(b[4..12].try_into().unwrap());
    let len = u64::from_le_bytes(b[12..20].try_into().unwrap());
    let end = usize::try_from(len)
        .ok()
        .and_then(|len| HEAD.checked_add(len))
        .filter(|&end| end <= b.len())
        .ok_or("truncated frame payload")?;
    if crc32fast::hash(&b[4..end]) != crc {
        return Err("frame checksum mismatch".into());
    }
    Ok((epoch, &b[HEAD..end], end))
}

enum FrameError {
    Unknown(u64),
    Decode(String),
}

/// Apply one checksummed frame: repeated `keyspace u64 | len u64 | body`.
fn apply_frame(
    mut payload: &[u8],
    decls: &HashMap<u64, KeyspaceDecl>,
    skip: &dyn Fn(u64) -> bool,
    trees: &mut HashMap<u64, Box<dyn Any + Send + Sync>>,
) -> Result<(), FrameError> {
    let bad = |why: &str| FrameError::Decode(why.to_string());
    while !payload.is_empty() {
        if payload.len() < 16 {
            return Err(bad("truncated section header"));
        }
        let ks = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let len = u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let end = usize::try_from(len)
            .ok()
            .and_then(|len| 16usize.checked_add(len))
            .filter(|&end| end <= payload.len())
            .ok_or_else(|| bad("truncated section body"))?;
        if skip(ks) {
            payload = &payload[end..];
            continue;
        }
        let decl = decls.get(&ks).ok_or(FrameError::Unknown(ks))?;
        let mut tree = trees.remove(&ks);
        let res = (decl.replay)(&mut tree, &payload[16..end]);
        if let Some(tree) = tree {
            trees.insert(ks, tree);
        }
        res.map_err(|e| FrameError::Decode(format!("keyspace {ks}: {e}")))?;
        payload = &payload[end..];
    }
    Ok(())
}

pub struct Store {
    /// The latest committed state. Only `finish` stores into it, and only
    /// while holding `writer`.
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
    /// Held while compacting, so one compaction runs at a time.
    compaction: Mutex<()>,
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
                let bytes = (keyspaces[&ks].measure)(&*tree);
                let tree = Arc::from(tree);
                (ks, LayerTree { tree, bytes })
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
        Store {
            current: ArcSwap::from_pointee(Version {
                epoch,
                layers,
                tables,
                resident_bytes: bytes,
                floors: Arc::new(writer.manifest.floors.clone()),
            }),
            writer: Mutex::new(writer),
            active_snapshots: Mutex::new(BTreeMap::new()),
            keyspaces,
            cache: BlockCache::new(opts.cache_size),
            opts,
            wal_bytes: AtomicU64::new(wal_bytes),
            compaction: Mutex::new(()),
            _lock: lock,
        }
    }

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

    pub(crate) fn stats(&self) -> MemoryStats {
        let version = self.current.load();
        MemoryStats {
            memtable_bytes: version.resident_bytes,
            wal_bytes: self.wal_bytes.load(Relaxed),
            cache_bytes: self.cache.used(),
            index_bytes: version.tables.iter().map(|t| t.index_bytes()).sum(),
            tables: version.tables.len(),
        }
    }

    /// Whether a checkpoint is due: the WAL or the memtable is over its
    /// limit, and there is somewhere to checkpoint to.
    fn over_limits(&self, w: &StoreWriter) -> bool {
        let version = self.current.load();
        w.dir.is_some()
            && !version.layers.is_empty()
            && (self.wal_bytes.load(Relaxed) >= self.opts.wal_limit
                || version.resident_bytes >= self.opts.memtable_limit)
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

    pub(crate) fn snapshot(&self) -> Snapshot<'_> {
        let mut map = self.snapshots();
        let version = self.current.load_full();
        *map.entry(version.epoch).or_default() += 1;
        drop(map);
        Snapshot {
            version,
            store: self,
        }
    }

    /// Open a write transaction over the latest committed state. Blocks
    /// while another write transaction is open. Dropping the tx without
    /// [`finish`](Self::finish) aborts it. Fails if the store is poisoned.
    pub(crate) fn begin(&self) -> Result<WriteTx<'_>> {
        // lock first: the snapshot must not be older than the last commit
        let writer = self.lock_writer();
        writer.check()?;
        Ok(WriteTx::new(
            self.snapshot(),
            writer,
            self.opts.tx_memory_limit,
        ))
    }

    /// Commit `tx`: log it, then publish it to readers. Returns the epoch
    /// it committed as, or `None` if it changed nothing, in which case no
    /// epoch is spent and nothing is logged.
    ///
    /// The WAL write and fsync happen before readers can see the tx, so
    /// under a strict policy nothing visible can be lost to a crash. If they
    /// fail, nothing is published and the store is poisoned.
    ///
    /// If the commit takes the WAL or the memtable over its limit, a
    /// checkpoint runs before this returns, and then compaction if there
    /// are too many tables. A failure there is [`Error::AfterCommit`]: the
    /// commit stands.
    pub(crate) fn finish(&self, tx: WriteTx<'_>) -> Result<Option<u64>> {
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
                trees.insert(
                    ks,
                    LayerTree {
                        tree: Arc::from(tree),
                        bytes: n,
                    },
                );
            }
        })?;
        self.wal_bytes.fetch_add(logged, Relaxed);

        let current = self.current.load();
        debug_assert_eq!(current.epoch + 1, epoch, "commit out of order");
        let mut layers = current.layers.clone();
        layers.push(Arc::new(Layer {
            epoch,
            trees,
            bytes,
        }));
        self.fold_layers(&mut layers, current.tables.is_empty());
        self.current.store(Arc::new(Version {
            epoch,
            resident_bytes: layers.iter().map(|l| l.bytes).sum(),
            layers,
            tables: current.tables.clone(),
            floors: current.floors.clone(),
        }));

        let checkpointed = if self.over_limits(&writer) {
            self.checkpoint_locked(&mut writer).map(|_| ())
        } else {
            Ok(())
        };
        // release the writer only after publishing, so the next tx's
        // snapshot includes this one; compaction does not need it until
        // it installs its result
        drop(writer);
        drop(snapshot);
        checkpointed
            .and_then(|()| self.compact_if_due())
            .map_err(|e| Error::AfterCommit {
                epoch,
                source: Arc::new(e),
            })?;
        Ok(Some(epoch))
    }

    /// Bring derived state in line with graph fingerprint `fp`, at open.
    /// If it was recorded under another, truncate the derived and
    /// `dropped` keyspaces and rebuild the derived ones with `replay`,
    /// then record `fp`. Until it is recorded, every open rebuilds.
    fn reconcile_graph(
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

    /// Empty keyspaces `ids` as of the current epoch, and forget `dropped`
    /// ones entirely. Older data stays where it is, hidden by a floor:
    /// reads skip tables no newer than it, replay skips WAL sections no
    /// newer than it, and compaction leaves it behind. Clears the recorded
    /// graph fingerprint, which the caller records again once derived
    /// state is rebuilt.
    fn truncate_locked(
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
        }));
        Ok(())
    }

    /// Fold the newest layers together while the older of the last two is
    /// no more than [`FOLD_RATIO`] times the newer's size, so a run of
    /// small commits does not leave a layer each for reads to probe.
    ///
    /// Snapshots hold the layers they were taken with, so folding never
    /// changes what one sees; it only stops sharing memory with them.
    /// With `bottomless`, nothing lies beneath the oldest layer, so a fold
    /// that reaches it applies deletes instead of keeping tombstones.
    fn fold_layers(&self, layers: &mut Vec<Arc<Layer>>, bottomless: bool) {
        while let [.., older, newer] = &layers[..]
            && older.bytes <= FOLD_RATIO * newer.bytes.max(1)
        {
            let drop_tombstones = bottomless && layers.len() == 2;
            let ids: BTreeSet<u64> = older
                .trees
                .keys()
                .chain(newer.trees.keys())
                .copied()
                .collect();
            let mut trees = HashMap::with_capacity(ids.len());
            for ks in ids {
                let (o, n) = (older.trees.get(&ks), newer.trees.get(&ks));
                let folded = match (o, n) {
                    // untouched by the other side and nothing to drop: share it
                    (Some(t), None) | (None, Some(t)) if !drop_tombstones => t.clone(),
                    _ => (self.keyspaces[&ks].fold)(o, n, drop_tombstones),
                };
                // every entry costs at least ENTRY_OVERHEAD, so no bytes
                // means deletes emptied the tree
                if folded.bytes > 0 {
                    trees.insert(ks, folded);
                }
            }
            let epoch = newer.epoch;
            layers.truncate(layers.len() - 2);
            if !trees.is_empty() {
                layers.push(Arc::new(Layer {
                    epoch,
                    bytes: trees.values().map(|t| t.bytes).sum(),
                    trees,
                }));
            }
        }
    }

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
    /// Holds the writer lock throughout, so commits wait for it, and it
    /// must not be called while this thread holds a write tx.
    pub(crate) fn checkpoint(&self) -> Result<Option<u64>> {
        let mut writer = self.lock_writer();
        writer.check()?;
        self.checkpoint_locked(&mut writer)
    }

    fn checkpoint_locked(&self, w: &mut StoreWriter) -> Result<Option<u64>> {
        let Some(dir) = w.dir.clone() else {
            return Ok(None);
        };
        let dir = &*dir;
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

        // build the next manifest aside, so a failure leaves the writer's
        // state matching the manifest on disk; files created before a
        // failure are orphans the next open sweeps
        let mut next = w.manifest.clone();
        let ckpt = Checkpoint {
            number: next.next_file,
            epoch: version.epoch,
        };
        next.next_file += 1;
        let path = dir.join(ckpt.file_name());
        let mut table = TableBuilder::create(&path, self.opts.block_size)?;
        let ids: BTreeSet<u64> = version
            .layers
            .iter()
            .flat_map(|l| l.trees.keys().copied())
            .collect();
        for ks in ids {
            let trees: Vec<&(dyn Any + Send + Sync)> = version
                .layers
                .iter()
                .rev()
                .filter_map(|l| l.trees.get(&ks).map(|t| &*t.tree))
                .collect();
            table.begin_keyspace(ks)?;
            (self.keyspaces[&ks].write_run)(&trees, &mut |k, v| table.push(k, v))?;
        }
        table.finish(ckpt.epoch)?;
        let table = Table::open(&path, ckpt.number)?;
        next.checkpoints.push(ckpt);

        // every segment holds only epochs the checkpoint now covers
        let retired = std::mem::take(&mut next.segments);
        let wal = match w.wal {
            Some(_) => Some(new_segment(dir, &mut next)?),
            None => None,
        };
        next.prune_floors();
        let floors = Arc::new(next.floors.clone());
        w.install_manifest(dir, next)?;

        // the checkpoint is durable: swap its table in for the layers it
        // covers, which is all of them
        let mut tables = version.tables.clone();
        tables.push(Arc::new(table));
        self.current.store(Arc::new(Version {
            epoch: version.epoch,
            layers: Vec::new(),
            tables,
            resident_bytes: 0,
            floors,
        }));
        self.wal_bytes.store(0, Relaxed);

        // the old segment's buffered frames, if any, are covered too; its
        // drop flushes them into a file that is about to be deleted
        w.wal = wal;
        for seg in retired {
            // a failure leaves an orphan for the next open to sweep
            let _ = fs::remove_file(dir.join(seg.file_name()));
        }
        Ok(Some(version.epoch))
    }

    /// Compact if there are more tables than `max_tables`, unless another
    /// compaction is already running.
    fn compact_if_due(&self) -> Result<()> {
        if self.current.load().tables.len() <= self.opts.max_tables {
            return Ok(());
        }
        let running = match self.compaction.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            // another thread is compacting; it will get there
            Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
        };
        self.compact_over(&running).map(|_| ())
    }

    /// Merge windows of adjacent tables until no more than `max_tables`
    /// remain. Returns how many merges ran.
    fn compact_while_over(&self) -> Result<usize> {
        let running = self
            .compaction
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.compact_over(&running)
    }

    /// [`compact_while_over`](Self::compact_while_over), with the compaction
    /// lock already held.
    fn compact_over(&self, _running: &MutexGuard<'_, ()>) -> Result<usize> {
        let mut merges = 0;
        loop {
            let tables = self.current.load().tables.clone();
            if tables.len() <= self.opts.max_tables {
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
    /// install the result, so it must not be called while this thread
    /// holds a write tx.
    pub(crate) fn compact(&self) -> Result<bool> {
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
    fn merge_tables(&self, tables: &[Arc<Table>], window: Range<usize>) -> Result<()> {
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
        let mut out = TableBuilder::create(&path, self.opts.block_size)?;
        let ids: BTreeSet<u64> = inputs.iter().flat_map(|t| t.keyspaces()).collect();
        let (mut key, mut value) = (Vec::new(), Vec::new());
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
                for c in &mut cursors {
                    if c.current().is_some_and(|(k, _)| k == &key[..]) {
                        c.advance()?;
                    }
                }
                if !(drop_tombstones && value == TOMBSTONE) {
                    out.push(&key, &value)?;
                }
            }
        }
        out.finish(epoch)?;
        let merged = Arc::new(Table::open(&path, number)?);

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

/// The `width` adjacent tables, or all of them if there are fewer, with the
/// fewest bytes between them: merging small recent tables often and large
/// old ones rarely keeps the bytes rewritten per merge low.
fn cheapest_window(tables: &[Arc<Table>], width: usize) -> Range<usize> {
    let width = width.min(tables.len());
    let start = (0..=tables.len() - width)
        .min_by_key(|&i| tables[i..i + width].iter().map(|t| t.size()).sum::<u64>())
        .unwrap_or(0);
    start..start + width
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Change, Register};

    const KS: KeyspaceId<u8, i64> = KeyspaceId::new(3);

    fn mem() -> Store {
        StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(KS)
            .build()
            .unwrap()
    }

    /// A fresh, empty directory unique to this test.
    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fold3-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn open(dir: &Path, dp: DurabilityPolicy) -> Result<Store> {
        StoreBuilder::new(dp).dir(dir).keyspace(KS).build()
    }

    fn put(store: &Store, kvs: &[(u8, Option<i64>)]) -> Option<u64> {
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        for &(k, v) in kvs {
            let change = match v {
                Some(v) => Change::Insert(v),
                None => Change::Delete(0),
            };
            if let Some(r) = ks.resolve(&k, change).unwrap() {
                ks.absorb(k, r).unwrap();
            }
        }
        store.finish(tx).unwrap()
    }

    fn read(store: &Store, k: u8) -> Option<i64> {
        store.snapshot().get(KS, &k).unwrap().map(|v| *v)
    }

    fn manifest_of(dir: &Path) -> Manifest {
        Manifest::parse(&manifest::read(dir).unwrap().unwrap()).unwrap()
    }

    #[test]
    fn commits_are_visible_in_order_and_tombstones_shadow() {
        let store = mem();
        assert_eq!(put(&store, &[(1, Some(10)), (2, Some(20))]), Some(1));
        assert_eq!(put(&store, &[(2, None), (3, Some(30))]), Some(2));
        assert_eq!(put(&store, &[(1, Some(11))]), Some(3));

        assert_eq!(read(&store, 1), Some(11), "newest layer wins");
        assert_eq!(read(&store, 2), None, "tombstone shadows epoch 1");
        assert_eq!(read(&store, 3), Some(30));
        assert_eq!(read(&store, 4), None);
    }

    #[test]
    fn snapshots_are_isolated_from_later_commits() {
        let store = mem();
        put(&store, &[(1, Some(10))]);
        let before = store.snapshot();
        put(&store, &[(1, Some(11))]);
        assert_eq!(before.get(KS, &1).unwrap().as_deref(), Some(&10));
        assert_eq!(before.epoch(), 1);
        assert_eq!(read(&store, 1), Some(11));
    }

    #[test]
    fn empty_and_aborted_txs_spend_no_epoch() {
        let store = mem();
        // read-only: the keyspace is opened but nothing is written
        let mut tx = store.begin().unwrap();
        assert_eq!(tx.keyspace(KS).get(&1).unwrap(), None);
        assert_eq!(store.finish(tx).unwrap(), None);
        // aborted: dropped without finish
        {
            let mut tx = store.begin().unwrap();
            tx.keyspace(KS).absorb(1, Change::Insert(5)).unwrap();
        }
        assert_eq!(read(&store, 1), None);
        assert_eq!(put(&store, &[(1, Some(10))]), Some(1));
        assert!(store.active_snapshots.lock().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "keyspace 9 not declared")]
    fn undeclared_keyspaces_are_refused() {
        let store = mem();
        let mut tx = store.begin().unwrap();
        tx.keyspace(KeyspaceId::<u8, i64>::new(9));
    }

    #[test]
    fn writing_policies_need_a_directory() {
        let err = StoreBuilder::new(DurabilityPolicy::RELAXED).build();
        assert!(matches!(err, Err(Error::Policy(_))));
    }

    #[test]
    fn wal_gets_one_checksummed_frame_per_tx() {
        let dir = tmpdir("frames");
        {
            let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
            put(&store, &[(1, Some(10)), (2, Some(20))]);
            put(&store, &[(2, None)]);
            // deleting a key that never existed writes nothing: no frame
            assert_eq!(put(&store, &[(9, None)]), None);
        }
        let bytes = fs::read(dir.join("000001.wal")).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        let mut rest = &bytes[..];
        let mut frames = Vec::new();
        while !rest.is_empty() {
            let (epoch, payload, used) = parse_frame(rest).unwrap();
            // one keyspace section: id, length, then (u8 key, Option<i64>)*
            assert_eq!(
                u64::from_le_bytes(payload[..8].try_into().unwrap()),
                KS.raw()
            );
            let body_len = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
            assert_eq!(16 + body_len, payload.len());
            frames.push((epoch, payload[16..].to_vec()));
            rest = &rest[used..];
        }
        // epoch 1: key 1 -> Some(10), key 2 -> Some(20)
        let mut one = vec![1u8, 1];
        one.extend(10i64.to_be_bytes());
        one.extend([2u8, 1]);
        one.extend(20i64.to_be_bytes());
        // epoch 2: key 2 -> None
        assert_eq!(frames, [(1, one), (2, vec![2u8, 0])]);
    }

    #[test]
    fn reopen_replays_the_wal_and_continues_epochs() {
        let dir = tmpdir("reopen");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10)), (2, Some(20))]);
            put(&store, &[(2, None), (3, Some(30))]);
            put(&store, &[(1, Some(11))]);
        }
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            assert_eq!(store.snapshot().epoch(), 3);
            assert_eq!(read(&store, 1), Some(11));
            assert_eq!(read(&store, 2), None);
            assert_eq!(read(&store, 3), Some(30));
            // a clean tail is reused, not replaced
            assert_eq!(put(&store, &[(4, Some(40))]), Some(4));
        }
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, 4), Some(40));
        let m = manifest_of(&dir);
        assert_eq!(
            m.segments,
            [Segment {
                number: 1,
                sealed: None
            }]
        );
        assert_eq!(m.keyspaces[&3], ("u8".to_string(), "i64".to_string()));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn torn_tail_is_sealed_and_later_commits_survive() {
        let dir = tmpdir("torn");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10))]);
            put(&store, &[(2, Some(20))]);
        }
        // a crash mid-append: chop the last frame in half
        let wal = dir.join("000001.wal");
        let full = fs::read(&wal).unwrap();
        let (_, _, first) = parse_frame(&full).unwrap();
        let torn = first + (full.len() - first) / 2;
        fs::write(&wal, &full[..torn]).unwrap();
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            assert_eq!(read(&store, 1), Some(10));
            assert_eq!(read(&store, 2), None, "torn frame is not replayed");
            assert_eq!(put(&store, &[(3, Some(30))]), Some(2));
        }
        let m = manifest_of(&dir);
        assert_eq!(
            m.segments,
            [
                Segment {
                    number: 1,
                    sealed: Some(first as u64)
                },
                Segment {
                    number: 2,
                    sealed: None
                },
            ]
        );
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            assert_eq!(read(&store, 1), Some(10));
            assert_eq!(
                read(&store, 3),
                Some(30),
                "frames after the torn tail replay"
            );
        }
        // the torn bytes are still on disk and still ignored
        assert_eq!(fs::read(&wal).unwrap().len(), torn);

        // damage inside a sealed segment is corruption, not a torn tail
        let mut bytes = fs::read(&wal).unwrap();
        bytes[first - 1] ^= 0xff;
        fs::write(&wal, &bytes).unwrap();
        let err = open(&dir, DurabilityPolicy::RELAXED).err().unwrap();
        assert!(matches!(err, Error::Corrupt { offset: 0, .. }), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keyspaces_must_be_registered_with_their_written_types() {
        let dir = tmpdir("types");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10))]);
        }
        let other = StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .keyspace(KeyspaceId::<u8, (i64, i64)>::new(3))
            .build();
        assert!(matches!(other, Err(Error::KeyspaceType { id: 3, .. })));

        let missing = StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .build();
        assert!(matches!(missing, Err(Error::UnknownKeyspace(3))));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_directory_opens_once_at_a_time() {
        let dir = tmpdir("lock");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert!(matches!(
            open(&dir, DurabilityPolicy::RELAXED),
            Err(Error::Locked(_))
        ));
        drop(store);
        assert!(open(&dir, DurabilityPolicy::RELAXED).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn non_writing_policy_still_replays_but_logs_nothing() {
        let dir = tmpdir("none");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10))]);
        }
        {
            let store = open(&dir, DurabilityPolicy::NONE).unwrap();
            assert_eq!(read(&store, 1), Some(10));
            assert_eq!(put(&store, &[(1, Some(11))]), Some(2));
        }
        assert_eq!(manifest_of(&dir).policy, DurabilityPolicy::NONE);
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, 1), Some(10), "unlogged commit is gone");
        assert_eq!(put(&store, &[(1, Some(12))]), Some(2));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The rows of keyspace `KS` in a checkpoint's table.
    fn checkpoint_rows(path: &Path) -> Vec<(u8, Option<i64>)> {
        let table = Table::open(path, 0).unwrap();
        let mut c = table.cursor(KS.raw(), None);
        c.start().unwrap();
        let mut rows = Vec::new();
        while let Some((k, v)) = c.current() {
            rows.push((
                crate::codec::decode_all_ordered::<u8>(k).unwrap(),
                decode_all::<Option<i64>>(v).unwrap(),
            ));
            c.advance().unwrap();
        }
        rows
    }

    fn files_in(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect()
    }

    #[test]
    fn checkpoint_retires_the_wal_and_reopen_loads_it() {
        let dir = tmpdir("ckpt");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10)), (2, Some(20))]);
            put(&store, &[(1, Some(11))]);
            assert_eq!(store.checkpoint().unwrap(), Some(2));
            assert_eq!(store.checkpoint().unwrap(), None, "nothing new");
            // after the checkpoint: one overwrite, one delete of checkpointed data
            put(&store, &[(2, None), (3, Some(30))]);
        }
        let m = manifest_of(&dir);
        assert_eq!(
            m.checkpoints,
            [Checkpoint {
                number: 2,
                epoch: 2
            }]
        );
        assert_eq!(
            m.segments,
            [Segment {
                number: 3,
                sealed: None
            }]
        );
        assert_eq!(
            files_in(&dir),
            BTreeSet::from(["LOCK", "MANIFEST", "000002.ckpt", "000003.wal"].map(String::from)),
            "the covered segment is deleted"
        );
        assert_eq!(
            checkpoint_rows(&dir.join("000002.ckpt")),
            [(1, Some(11)), (2, Some(20))],
            "newest value per key"
        );

        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(store.snapshot().epoch(), 3);
        let v = store.current.load();
        assert_eq!(
            (v.layers.len(), v.tables.len()),
            (1, 1),
            "WAL in memory, checkpoint on disk"
        );
        assert_eq!(read(&store, 1), Some(11));
        assert_eq!(
            read(&store, 2),
            None,
            "WAL tombstone shadows the checkpoint"
        );
        assert_eq!(read(&store, 3), Some(30));
        assert_eq!(put(&store, &[(4, Some(40))]), Some(4));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoints_are_deltas_and_load_in_order() {
        let dir = tmpdir("deltas");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10)), (2, Some(20))]);
            assert_eq!(store.checkpoint().unwrap(), Some(1));
            put(&store, &[(2, None)]);
            put(&store, &[(3, Some(30))]);
            assert_eq!(store.checkpoint().unwrap(), Some(3));
        }
        let m = manifest_of(&dir);
        assert_eq!(
            m.checkpoints,
            [
                Checkpoint {
                    number: 2,
                    epoch: 1
                },
                Checkpoint {
                    number: 4,
                    epoch: 3
                },
            ]
        );
        assert_eq!(
            checkpoint_rows(&dir.join("000004.ckpt")),
            [(2, None), (3, Some(30))],
            "only what changed since the first, tombstones included"
        );

        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(store.snapshot().epoch(), 3);
        let v = store.current.load();
        assert_eq!(
            (v.layers.len(), v.tables.len()),
            (0, 2),
            "no WAL frames left"
        );
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(read(&store, 2), None, "later checkpoint's tombstone wins");
        assert_eq!(read(&store, 3), Some(30));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoint_persists_commits_a_non_writing_policy_never_logged() {
        let dir = tmpdir("ckpt-none");
        {
            let store = open(&dir, DurabilityPolicy::NONE).unwrap();
            put(&store, &[(1, Some(10))]);
            assert_eq!(store.checkpoint().unwrap(), Some(1));
            put(&store, &[(2, Some(20))]);
        }
        assert!(manifest_of(&dir).segments.is_empty());
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(read(&store, 2), None, "never logged, never checkpointed");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn in_memory_checkpoint_is_a_no_op() {
        let store = mem();
        put(&store, &[(1, Some(10))]);
        assert_eq!(store.checkpoint().unwrap(), None);
    }

    #[test]
    fn open_sweeps_orphans_but_not_foreign_files() {
        let dir = tmpdir("orphans");
        drop(open(&dir, DurabilityPolicy::RELAXED).unwrap());
        for junk in [
            "000099.wal",
            "000098.ckpt",
            "MANIFEST.tmp",
            "notes.txt",
            "x.wal",
        ] {
            fs::write(dir.join(junk), b"junk").unwrap();
        }
        drop(open(&dir, DurabilityPolicy::RELAXED).unwrap());
        assert_eq!(
            files_in(&dir),
            BTreeSet::from(
                ["LOCK", "MANIFEST", "000001.wal", "notes.txt", "x.wal"].map(String::from)
            )
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_damaged_checkpoint_is_corruption() {
        let dir = tmpdir("ckpt-bad");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10))]);
            store.checkpoint().unwrap();
        }
        let path = dir.join("000002.ckpt");
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        let err = open(&dir, DurabilityPolicy::RELAXED).err().unwrap();
        assert!(matches!(err, Error::Corrupt { .. }), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoint_moves_data_out_of_memory_and_reads_page_it_in() {
        let dir = tmpdir("evict");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        put(&store, &[(2, None)]);
        store.checkpoint().unwrap();

        let v = store.current.load_full();
        assert!(v.layers.is_empty(), "covered layers are dropped");
        assert_eq!(store.cache.stats(), (0, 0), "nothing read yet");
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(read(&store, 2), None, "tombstone read from disk");
        assert_eq!(read(&store, 3), None);
        assert_eq!(store.cache.stats().1, 1, "one block, read once");

        // reads in a write tx fall through to the table too
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        let r = ks.resolve(&1, Change::Modify(5)).unwrap().unwrap();
        assert!(matches!(r, Change::Replace(10, 15)));
        ks.absorb(1, r).unwrap();
        store.finish(tx).unwrap();
        assert_eq!(read(&store, 1), Some(15), "memory layer shadows the table");
        drop((store, v));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snapshots_keep_their_view_across_a_checkpoint() {
        let dir = tmpdir("ckpt-snap");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
        let first = store.snapshot();
        put(&store, &[(1, Some(11)), (2, Some(20))]);
        let second = store.snapshot();
        store.checkpoint().unwrap();
        put(&store, &[(2, None)]);

        assert_eq!(first.get(KS, &1).unwrap().as_deref(), Some(&10));
        assert_eq!(first.get(KS, &2).unwrap().as_deref(), None);
        assert_eq!(second.get(KS, &1).unwrap().as_deref(), Some(&11));
        assert_eq!(second.get(KS, &2).unwrap().as_deref(), Some(&20));
        assert_eq!(read(&store, 1), Some(11));
        assert_eq!(read(&store, 2), None);
        drop((first, second));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn large_checkpoints_read_through_a_small_cache() {
        const BIG: KeyspaceId<u32, i64> = KeyspaceId::new(8);
        let dir = tmpdir("big");
        let build = || {
            StoreBuilder::new(DurabilityPolicy::RELAXED)
                .dir(&dir)
                .keyspace(KS)
                .keyspace(BIG)
                .cache_size(16 << 10)
                .build()
                .unwrap()
        };
        {
            let store = build();
            for chunk in (0..20_000u32).collect::<Vec<_>>().chunks(5_000) {
                let mut tx = store.begin().unwrap();
                let mut ks = tx.keyspace(BIG);
                for &i in chunk {
                    ks.absorb(i, Change::Insert(i as i64 * 3)).unwrap();
                }
                store.finish(tx).unwrap();
            }
            put(&store, &[(1, Some(10))]);
            store.checkpoint().unwrap();
        }
        let store = build();
        let snap = store.snapshot();
        for i in (0..20_000u32).step_by(7) {
            assert_eq!(snap.get(BIG, &i).unwrap().as_deref(), Some(&(i as i64 * 3)));
        }
        assert_eq!(snap.get(BIG, &20_000).unwrap().as_deref(), None);
        assert_eq!(snap.get(KS, &1).unwrap().as_deref(), Some(&10));
        let (cached, loads) = store.cache.stats();
        assert!(cached <= 16 << 10, "cache over budget: {cached}");
        assert!(loads > 4, "blocks were paged in and evicted: {loads}");
        drop(snap);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    fn open_with(
        dir: &Path,
        dp: DurabilityPolicy,
        f: impl FnOnce(StoreBuilder) -> StoreBuilder,
    ) -> Result<Store> {
        f(StoreBuilder::new(dp).dir(dir).keyspace(KS)).build()
    }

    #[test]
    fn a_tx_that_outgrows_its_limit_is_killed_and_commits_nothing() {
        let store = StoreBuilder::new(DurabilityPolicy::NONE)
            .keyspace(KS)
            .tx_memory_limit(1024)
            .build()
            .unwrap();
        put(&store, &[(0, Some(1))]);

        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        let mut killed = None;
        for k in 1..=255u8 {
            if let Err(oom) = ks.absorb(k, Change::Insert(k as i64)) {
                killed = Some((k, oom));
                break;
            }
        }
        let (at, err) = killed.expect("1 KiB cannot hold 255 entries");
        assert!(at > 1, "a few entries fit");
        let Error::OutOfMemory(oom) = err else {
            panic!("{err}")
        };
        assert!(oom.used > 1024 && oom.limit == 1024, "{oom}");
        let is_oom = |r: Result<()>| matches!(r, Err(Error::OutOfMemory(e)) if e == oom);
        assert!(is_oom(ks.absorb(250, Change::Insert(1))), "stays dead");
        assert!(ks.get(&1).is_err(), "reads fail too");
        drop(ks);
        assert!(matches!(tx.failure(), Some(Error::OutOfMemory(e)) if e == oom));
        assert_eq!(tx.memory_used(), 0);
        assert!(matches!(store.finish(tx), Err(Error::OutOfMemory(e)) if e == oom));

        assert_eq!(store.snapshot().epoch(), 1, "no epoch spent");
        assert_eq!(read(&store, 1), None);
        assert_eq!(
            put(&store, &[(1, Some(1))]),
            Some(2),
            "the writer is free again"
        );
    }

    #[test]
    fn tx_memory_tracks_entries_that_change_and_cancel() {
        let store = mem();
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Insert(10)).unwrap();
        ks.absorb(2, Change::Insert(20)).unwrap();
        drop(ks);
        let two = tx.memory_used();
        assert!(two > 0);
        let mut ks = tx.keyspace(KS);
        // insert then delete cancels: the entry is gone, and so are its bytes
        ks.absorb(2, Change::Delete(20)).unwrap();
        drop(ks);
        assert_eq!(tx.memory_used(), two / 2);
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Delete(10)).unwrap();
        drop(ks);
        assert_eq!(tx.memory_used(), 0);
    }

    #[test]
    fn crossing_the_wal_limit_checkpoints_automatically() {
        let dir = tmpdir("auto-wal");
        let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(200)).unwrap();
        let mut epochs = Vec::new();
        for i in 0..20u8 {
            epochs.push(put(&store, &[(i, Some(i as i64))]).unwrap());
            let stats = store.stats();
            assert!(
                stats.wal_bytes < 200,
                "checkpointed by the time finish returns"
            );
            if stats.tables > 0 {
                break;
            }
        }
        let stats = store.stats();
        assert_eq!(stats.tables, 1, "the limit was crossed");
        assert_eq!((stats.wal_bytes, stats.memtable_bytes), (0, 0));
        let n = epochs.len() as u8;
        assert_eq!(manifest_of(&dir).checkpointed(), n as u64);
        for i in 0..n {
            assert_eq!(read(&store, i), Some(i as i64));
        }
        drop(store);
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, n - 1), Some(n as i64 - 1));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn crossing_the_memtable_limit_checkpoints_even_without_a_wal() {
        let dir = tmpdir("auto-mem");
        let store = open_with(&dir, DurabilityPolicy::NONE, |b| b.memtable_limit(512)).unwrap();
        for i in 0..50u8 {
            put(&store, &[(i, Some(i as i64))]);
            assert!(store.stats().memtable_bytes < 512);
        }
        assert!(store.stats().tables > 1, "{:?}", store.stats());
        drop(store);
        let store = open(&dir, DurabilityPolicy::NONE).unwrap();
        let unlogged = (0..50u8).filter(|&i| read(&store, i).is_none()).count();
        assert!(
            unlogged < 50 && unlogged > 0,
            "checkpointed data survives, the tail does not"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_checkpoints_when_the_last_run_left_too_much() {
        let dir = tmpdir("auto-open");
        {
            let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
            put(&store, &[(1, Some(10))]);
            put(&store, &[(2, Some(20))]);
            assert_eq!(store.stats().tables, 0);
        }
        let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(10)).unwrap();
        let stats = store.stats();
        assert_eq!(
            (stats.tables, stats.wal_bytes, stats.memtable_bytes),
            (1, 0, 0)
        );
        assert_eq!(read(&store, 2), Some(20));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wal_bytes_match_the_files_and_survive_reopen() {
        let dir = tmpdir("wal-bytes");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        put(&store, &[(3, Some(30))]);
        let len = fs::metadata(dir.join("000001.wal")).unwrap().len();
        assert_eq!(store.stats().wal_bytes, len);
        drop(store);
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(store.stats().wal_bytes, len);
        assert!(
            store.stats().memtable_bytes > 0,
            "replayed data is accounted"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_auto_checkpoint_reports_the_commit_and_retries() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmpdir("auto-fail");
        let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(100)).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        if fs::write(dir.join("probe"), b"").is_ok() {
            // permissions do not bind (running as root): nothing to test
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
            drop(store);
            fs::remove_dir_all(&dir).unwrap();
            return;
        }

        let mut failed = None;
        for i in 0..20u8 {
            let mut tx = store.begin().unwrap();
            tx.keyspace(KS).absorb(i, Change::Insert(i as i64)).unwrap();
            match store.finish(tx) {
                Ok(_) => {}
                Err(Error::AfterCommit { epoch, .. }) => {
                    failed = Some((i, epoch));
                    break;
                }
                Err(e) => panic!("{e}"),
            }
        }
        let (i, epoch) = failed.expect("the limit was crossed");
        assert_eq!(read(&store, i), Some(i as i64), "the commit still happened");
        assert_eq!(store.snapshot().epoch(), epoch);
        assert_eq!(store.stats().tables, 0);

        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        put(&store, &[(99, Some(99))]);
        assert_eq!(store.stats().tables, 1, "the next commit retried it");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn options_are_validated_and_recorded() {
        let dir = tmpdir("options");
        let bad = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.block_size(0));
        assert!(matches!(bad, Err(Error::InvalidOption(_))));
        drop(
            open_with(&dir, DurabilityPolicy::RELAXED, |b| {
                b.block_size(512).cache_size(1 << 20)
            })
            .unwrap(),
        );
        let m = manifest_of(&dir);
        assert_eq!(m.options["block-size"], 512);
        assert_eq!(m.options["cache-size"], 1 << 20);
        assert_eq!(m.options["wal-limit"], 64 << 20);
        fs::remove_dir_all(&dir).unwrap();
    }

    fn live_rows(store: &Store) -> Vec<(u8, i64)> {
        store
            .snapshot()
            .range(KS, ..)
            .map(|r| r.map(|(k, v)| (*k, *v)))
            .collect::<Result<_>>()
            .unwrap()
    }

    /// Keys 1..=6 spread over two tables and a memory layer, with
    /// overwrites and deletes at every level.
    fn layered(store: &Store) {
        put(
            store,
            &[(1, Some(1)), (2, Some(2)), (3, Some(3)), (4, Some(4))],
        );
        store.checkpoint().unwrap();
        put(store, &[(2, Some(20)), (3, None), (5, Some(5))]);
        store.checkpoint().unwrap();
        put(store, &[(4, Some(40)), (5, None), (6, Some(6))]);
    }

    #[test]
    fn small_commits_fold_into_logarithmically_many_layers() {
        let store = mem();
        let mut model = BTreeMap::new();
        for i in 0..2000u32 {
            let k = (i % 251) as u8;
            let v = (i % 7 != 0).then_some(i as i64);
            put(&store, &[(k, v)]);
            match v {
                Some(v) => model.insert(k, v),
                None => model.remove(&k),
            };
            let layers = store.current.load().layers.len();
            assert!(layers <= 16, "{layers} layers after {i} commits");
        }
        for k in 0..=255u8 {
            assert_eq!(read(&store, k), model.get(&k).copied(), "key {k}");
        }
        let live: Vec<(u8, i64)> = model.into_iter().collect();
        assert_eq!(live_rows(&store), live);
    }

    #[test]
    fn folding_without_tables_applies_deletes() {
        let store = mem();
        for k in 0..100u8 {
            put(&store, &[(k, Some(k as i64))]);
        }
        for k in 0..100u8 {
            put(&store, &[(k, None)]);
        }
        let v = store.current.load();
        let tombstones: usize = v
            .layers
            .iter()
            .flat_map(|l| l.trees.values())
            .map(|t| {
                t.tree
                    .downcast_ref::<Resident<u8, i64>>()
                    .unwrap()
                    .values()
                    .filter(|v| v.is_none())
                    .count()
            })
            .sum();
        // only the newest few layers, not yet folded to the bottom, keep any
        assert!(tombstones < 20, "{tombstones} tombstones left");
        assert!(
            v.resident_bytes < 20 * 64,
            "{} bytes left",
            v.resident_bytes
        );
        assert_eq!(
            v.resident_bytes,
            v.layers.iter().map(|l| l.bytes).sum::<usize>()
        );
    }

    #[test]
    fn folding_above_tables_keeps_tombstones() {
        let dir = tmpdir("fold-over-tables");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(1)), (2, Some(2))]);
        store.checkpoint().unwrap();
        put(&store, &[(1, None)]);
        for i in 0..50 {
            put(&store, &[(3, Some(i))]);
        }
        assert_eq!(read(&store, 1), None, "the folded tombstone still shadows");
        assert_eq!(live_rows(&store), vec![(2, 2), (3, 49)]);
        drop(store);
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(live_rows(&store), vec![(2, 2), (3, 49)]);
    }

    #[test]
    fn snapshots_keep_their_view_across_folds() {
        let store = mem();
        put(&store, &[(1, Some(1)), (2, Some(2))]);
        let before = store.snapshot();
        for i in 0..100 {
            put(&store, &[(1, Some(100 + i)), (2, None)]);
        }
        assert_eq!(before.get(KS, &1).unwrap().as_deref(), Some(&1));
        assert_eq!(before.get(KS, &2).unwrap().as_deref(), Some(&2));
        assert_eq!(read(&store, 1), Some(199));
        assert_eq!(read(&store, 2), None);
    }

    #[test]
    fn range_merges_memory_and_tables_newest_first() {
        let dir = tmpdir("range");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        layered(&store);
        let v = store.current.load();
        assert_eq!((v.layers.len(), v.tables.len()), (1, 2));
        drop(v);

        assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
        let snap = store.snapshot();
        let keys = |c: Cursor<'_, u8, i64>| -> Vec<u8> { c.map(|r| *r.unwrap().0).collect() };
        assert_eq!(keys(snap.range(KS, 2..=4)), [2, 4]);
        assert_eq!(keys(snap.range(KS, 2..4)), [2]);
        assert_eq!(
            keys(snap.range(KS, (Bound::Excluded(2), Bound::Unbounded))),
            [4, 6]
        );
        assert_eq!(keys(snap.range(KS, 7..)), Vec::<u8>::new());

        // a write tx's cursor sees its own writes on top
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        ks.absorb(1, Change::Delete(1)).unwrap();
        ks.absorb(3, Change::Insert(30)).unwrap();
        let rows: Vec<(u8, i64)> = ks
            .range(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(rows, [(2, 20), (3, 30), (4, 40), (6, 6)]);
        drop(ks);
        drop(tx);
        drop(snap);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn batch_resolve_reads_each_block_at_most_once() {
        const BIG: KeyspaceId<u32, i64> = KeyspaceId::new(8);
        let dir = tmpdir("resolve-pass");
        let store = StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .keyspace(BIG)
            .cache_size(DEFAULT_BLOCK_SIZE)
            .build()
            .unwrap();
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(BIG);
        for i in 0..20_000u32 {
            ks.absorb(i, Change::Insert(i as i64)).unwrap();
        }
        store.finish(tx).unwrap();
        store.checkpoint().unwrap();
        let blocks = store.current.load().tables[0].blocks(BIG.raw());
        let loads_before = store.cache.stats().1;

        let mut changes = Register::new();
        for i in (0..20_000u32).step_by(7) {
            changes.absorb_one(i, Change::Modify(1));
        }
        changes.absorb_one(30_000, Change::Modify(1));
        let mut tx = store.begin().unwrap();
        let ks = tx.keyspace(BIG);
        let mut out = Vec::new();
        ks.resolve_sorted(changes, &mut out).unwrap();
        let loads = store.cache.stats().1 - loads_before;
        assert!(loads <= blocks as u64, "{loads} loads for {blocks} blocks");
        assert!(matches!(out[1], (7, Change::Replace(7, 8))));
        assert!(matches!(out.last(), Some((30_000, Change::Insert(1)))));
        drop(ks);
        drop(tx);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn too_many_tables_are_compacted_after_commits() {
        let dir = tmpdir("auto-compact");
        let opts = |b: StoreBuilder| {
            b.memtable_limit(300).options_with(|o| {
                o.max_tables = 3;
                o.compaction_width = 2;
            })
        };
        {
            let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
            for round in 0..6u8 {
                for i in 0..10u8 {
                    put(&store, &[(i, Some(round as i64 * 100 + i as i64))]);
                    assert!(store.stats().tables <= 3, "{:?}", store.stats());
                }
            }
            put(&store, &[(3, None)]);
            assert_eq!(live_rows(&store).len(), 9);
            assert_eq!(read(&store, 4), Some(504));
        }
        let m = manifest_of(&dir);
        let on_disk = files_in(&dir)
            .into_iter()
            .filter(|f| f.ends_with(".ckpt"))
            .count();
        assert_eq!(on_disk, m.checkpoints.len(), "merged inputs are deleted");
        let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
        assert_eq!(read(&store, 3), None);
        assert_eq!(read(&store, 9), Some(509));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn full_compaction_drops_tombstones() {
        let dir = tmpdir("compact-all");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        layered(&store);
        store.checkpoint().unwrap();
        assert_eq!(store.stats().tables, 3);
        assert!(store.compact().unwrap());
        assert_eq!(store.stats().tables, 1);
        let path = store.current.load().tables[0].path().to_path_buf();
        assert_eq!(
            checkpoint_rows(&path),
            [(1, Some(1)), (2, Some(20)), (4, Some(40)), (6, Some(6))],
            "no tombstones left"
        );
        assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
        drop(store);
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compacting_above_the_oldest_table_keeps_tombstones() {
        let dir = tmpdir("compact-mid");
        let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| {
            b.options_with(|o| {
                o.max_tables = 2;
                o.compaction_width = 2;
            })
        })
        .unwrap();
        // a big oldest table makes the newer two the cheapest window
        let big: Vec<_> = (0..200u8).map(|i| (i, Some(i as i64))).collect();
        put(&store, &big);
        store.checkpoint().unwrap();
        put(&store, &[(1, None)]);
        store.checkpoint().unwrap();
        put(&store, &[(250, Some(250))]);
        store.checkpoint().unwrap();
        // the explicit checkpoint path does not compact; a commit does
        put(&store, &[(251, Some(251))]);
        let v = store.current.load_full();
        assert_eq!(v.tables.len(), 2);
        assert!(checkpoint_rows(v.tables[1].path()).contains(&(1, None)));
        assert_eq!(
            read(&store, 1),
            None,
            "the tombstone still hides the oldest table's value"
        );
        assert_eq!(read(&store, 2), Some(2));
        drop(v);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snapshots_read_tables_that_compaction_deleted() {
        let dir = tmpdir("compact-snap");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        layered(&store);
        store.checkpoint().unwrap();
        let old = store.snapshot();
        let old_paths: Vec<_> = old
            .version
            .tables
            .iter()
            .map(|t| t.path().to_path_buf())
            .collect();
        put(&store, &[(1, Some(100))]);
        store.compact().unwrap();
        assert!(old_paths.iter().all(|p| !p.exists()), "inputs are deleted");
        assert_eq!(old.get(KS, &1).unwrap().as_deref(), Some(&1));
        let rows: Vec<u8> = old.range(KS, ..).map(|r| *r.unwrap().0).collect();
        assert_eq!(rows, [1, 2, 4, 6]);
        assert_eq!(read(&store, 1), Some(100));
        drop(old);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_wal_write_poisons_the_store_until_reopened() {
        let dir = tmpdir("poison");
        {
            let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
            put(&store, &[(1, Some(10))]);
            store.lock_writer().wal.as_mut().unwrap().fail_next_write = true;

            let mut tx = store.begin().unwrap();
            tx.keyspace(KS).absorb(2, Change::Insert(20)).unwrap();
            let err = store.finish(tx).unwrap_err();
            assert!(matches!(err, Error::Poisoned(_)), "{err}");
            assert_eq!(read(&store, 2), None, "nothing was published");
            assert!(matches!(store.begin(), Err(Error::Poisoned(_))));
            assert!(matches!(store.checkpoint(), Err(Error::Poisoned(_))));
            assert_eq!(read(&store, 1), Some(10), "reads still work");
        }
        let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(read(&store, 2), None);
        assert_eq!(put(&store, &[(2, Some(20))]), Some(2));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_read_error_fails_the_tx_but_not_the_store() {
        let dir = tmpdir("read-err");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        store.checkpoint().unwrap();
        // damage the only data block; open handles see the new bytes
        let path = store.current.load().tables[0].path().to_path_buf();
        let mut bytes = fs::read(&path).unwrap();
        bytes[12] ^= 0xff;
        fs::write(&path, &bytes).unwrap();

        assert!(matches!(
            store.snapshot().get(KS, &1),
            Err(Error::Corrupt { .. })
        ));
        let snap = store.snapshot();
        let mut cursor_errs = snap.range(KS, ..).collect::<Vec<_>>();
        assert_eq!(cursor_errs.len(), 1, "a cursor stops at its first error");
        assert!(cursor_errs.pop().unwrap().is_err());
        drop(cursor_errs);
        drop(snap);

        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        assert!(ks.get(&1).is_err());
        assert!(ks.absorb(3, Change::Insert(30)).is_err(), "the tx failed");
        drop(ks);
        assert!(matches!(store.finish(tx), Err(Error::Corrupt { .. })));
        // the store is fine: a tx that does not read the damage commits
        let mut tx = store.begin().unwrap();
        tx.keyspace(KS).absorb(3, Change::Insert(30)).unwrap();
        assert_eq!(store.finish(tx).unwrap(), Some(2));
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }
}
