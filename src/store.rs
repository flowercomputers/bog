use std::{
    any::{Any, type_name},
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use arc_swap::ArcSwap;

use crate::{
    codec::{Codec, DecodeError, OrderedCodec, decode_all},
    manifest::{self, Checkpoint, MANIFEST_TMP, Manifest, Segment},
    merge::Merge,
    store::DurabilityPolicyCadence::Never,
    table::{BlockCache, Table, TableBuilder, TableError},
    tx::{KeyspaceId, WriteTx},
};

/// Why a store could not be opened.
#[derive(Debug)]
pub enum OpenError {
    Io(io::Error),
    /// The durability policy is invalid, or needs a directory.
    Policy(&'static str),
    /// Another `Store` holds this directory.
    Locked,
    Manifest(String),
    /// A keyspace is registered with different types than it was written with.
    KeyspaceType {
        id: u64,
        recorded: (String, String),
        registered: (&'static str, &'static str),
    },
    /// The WAL holds data for a keyspace that was not registered.
    UnknownKeyspace(u64),
    /// A logged frame is damaged somewhere it cannot be a torn tail.
    Corrupt {
        file: PathBuf,
        offset: u64,
        why: String,
    },
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Policy(why) => write!(f, "durability policy: {why}"),
            Self::Locked => f.write_str("store directory is locked by another open store"),
            Self::Manifest(why) => write!(f, "bad manifest: {why}"),
            Self::KeyspaceType {
                id,
                recorded,
                registered,
            } => write!(
                f,
                "keyspace {id} was written as {recorded:?} but is registered as {registered:?}"
            ),
            Self::UnknownKeyspace(id) => {
                write!(f, "the WAL holds keyspace {id}, which is not registered")
            }
            Self::Corrupt { file, offset, why } => {
                write!(f, "{} is corrupt at byte {offset}: {why}", file.display())
            }
        }
    }
}

impl std::error::Error for OpenError {}

impl From<io::Error> for OpenError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
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

/// Everything one epoch committed, keyed by keyspace id. Each value is a
/// `Resident<K, V>` for that keyspace's types.
struct Layer {
    epoch: u64,
    trees: HashMap<u64, Box<dyn Any + Send + Sync>>,
}

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
}

struct Wal {
    file: File,
    buffer: Vec<u8>,
    size: usize,
}
impl Wal {
    // todo: make this fallible!
    /// Buffer one frame holding a whole transaction, so replay sees either
    /// all of a tx or none of it. See [`encode_frame`].
    fn append(&mut self, epoch: u64, payload: &[u8]) {
        let before = self.buffer.len();
        encode_frame(epoch, payload, &mut self.buffer);
        // todo: check if wal + this frame would exceed max wal size
        self.size += self.buffer.len() - before;
    }

    fn flush(&mut self, write: bool, sync: bool) {
        if write && !self.buffer.is_empty() {
            self.file.write_all(&self.buffer).unwrap();
            self.buffer.clear();
        }
        if sync {
            self.file.sync_data().unwrap();
        }
    }

    fn size(&self) -> usize {
        debug_assert_eq!(
            self.size,
            self.file.metadata().unwrap().len() as usize + self.buffer.len()
        );
        self.size
    }
}
/// `crc32 | epoch u64 | len u64 | payload`, little endian, with the crc
/// covering everything after it. WAL segments are a run of these, and a
/// checkpoint is a magic number followed by exactly one.
fn encode_frame(epoch: u64, payload: &[u8], out: &mut Vec<u8>) {
    let epoch = epoch.to_le_bytes();
    let len = (payload.len() as u64).to_le_bytes();
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&epoch);
    hasher.update(&len);
    hasher.update(payload);
    out.extend(hasher.finalize().to_le_bytes());
    out.extend(epoch);
    out.extend(len);
    out.extend(payload);
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
        self.flush(true, false);
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
        }
    }

    /// Log one transaction and write/fsync as the policy says.
    // todo: windowed policies only flush when the next tx commits; a quiet
    // store needs a clock source to flush on its own.
    fn log(&mut self, epoch: u64, payload: &[u8]) {
        let Some(wal) = self.wal.as_mut() else { return };
        wal.append(epoch, payload);
        let now = Instant::now();
        let (write, sync) = self.dp.should(now, self.last_write, self.last_fsync);
        wal.flush(write, sync);
        if write {
            self.last_write = Some(now);
        }
        if sync {
            // to accurately reflect the time it actually took to fsync
            self.last_fsync = Some(Instant::now());
        }
    }
}

pub struct Snapshot<'s> {
    version: Arc<Version>,
    store: &'s Store,
}
impl Drop for Snapshot<'_> {
    fn drop(&mut self) {
        let mut map = self.store.active_snapshots.lock().unwrap();
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
    /// that mentions the key decides, and a tombstone there means absent.
    pub(crate) fn get<K, M>(&self, keyspace: KeyspaceId<K, M>, key: &K) -> Option<Cow<'_, M::Value>>
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
            .map(|tree| {
                tree.downcast_ref::<Resident<K, M::Value>>()
                    .expect("keyspace id reused with a different type")
            })
            .find_map(|tree| tree.get(key));
        if let Some(slot) = hit {
            return slot.as_ref().map(Cow::Borrowed);
        }
        if self.version.tables.is_empty() {
            return None;
        }

        let mut probe = Vec::with_capacity(key.encode_ordered_size());
        key.encode_ordered(&mut probe);
        // todo: surface read errors instead of panicking once reads are fallible
        for table in self.version.tables.iter().rev() {
            let found = table
                .get(ks, &probe, &self.store.cache)
                .unwrap_or_else(|e| panic!("reading {}: {e}", table.path().display()));
            if let Some((block, range)) = found {
                let value = decode_all::<Option<M::Value>>(&block[range]).unwrap_or_else(|e| {
                    panic!("decoding keyspace {ks} in {}: {e}", table.path().display())
                });
                return value.map(Cow::Owned);
            }
        }
        None
    }
}

/// A keyspace's types, captured when it is registered with the builder.
/// Replay uses it to decode WAL sections; transactions use it to refuse
/// keyspaces nobody declared.
struct KeyspaceDecl {
    key: &'static str,
    merge: &'static str,
    /// Decode one WAL section and apply it over the keyspace's replayed
    /// tree, creating the tree on first use.
    replay: fn(&mut Option<Box<dyn Any + Send + Sync>>, &[u8]) -> Result<(), DecodeError>,
    /// Merge this keyspace's trees, newest first, into one sorted run of
    /// encoded rows.
    write_run: fn(
        &[&(dyn Any + Send + Sync)],
        &mut dyn FnMut(&[u8], &[u8]) -> io::Result<()>,
    ) -> io::Result<()>,
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
    rows: &mut dyn FnMut(&[u8], &[u8]) -> io::Result<()>,
) -> io::Result<()>
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
    cache_bytes: usize,
}

impl StoreBuilder {
    /// An in-memory store unless [`dir`](Self::dir) is set. Policies that
    /// write need a directory.
    pub(crate) fn new(durability_policy: DurabilityPolicy) -> Self {
        StoreBuilder {
            durability_policy,
            dir: None,
            keyspaces: HashMap::new(),
            cache_bytes: 32 << 20,
        }
    }

    /// Bytes of checkpoint blocks to keep in memory. Defaults to 32 MiB.
    pub(crate) fn cache_bytes(mut self, bytes: usize) -> Self {
        self.cache_bytes = bytes;
        self
    }

    /// Keep the store in `dir`, creating it if needed.
    pub(crate) fn dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// Declare a keyspace. Every keyspace a transaction opens must be
    /// declared, and every keyspace in the WAL must be declared with the
    /// types it was written with.
    pub(crate) fn keyspace<K, M>(mut self, id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        let decl = KeyspaceDecl {
            key: type_name::<K>(),
            merge: type_name::<M>(),
            replay: replay_section::<K, M::Value>,
            write_run: write_run::<K, M::Value>,
        };
        if let Some(prev) = self.keyspaces.get(&id.raw()) {
            assert!(
                (prev.key, prev.merge) == (decl.key, decl.merge),
                "keyspace {} declared twice with different types",
                id.raw()
            );
        }
        self.keyspaces.insert(id.raw(), decl);
        self
    }

    pub(crate) fn build(self) -> Result<Store, OpenError> {
        let StoreBuilder {
            durability_policy: dp,
            dir,
            keyspaces,
            cache_bytes,
        } = self;
        let cache = BlockCache::new(cache_bytes);
        dp.validate().map_err(OpenError::Policy)?;

        let Some(dir) = dir else {
            if !matches!(dp.write, Never) {
                return Err(OpenError::Policy("writing a WAL needs a directory"));
            }
            let writer = StoreWriter::new(None, Manifest::new(dp), None, dp);
            return Ok(Store::new(
                Replayed::default(),
                Vec::new(),
                writer,
                keyspaces,
                cache,
                None,
            ));
        };

        fs::create_dir_all(&dir)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join("LOCK"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(OpenError::Locked),
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }

        // replay whatever is on disk even if this run will not write: the
        // previous run's settings may have differed from these
        let mut m = match manifest::read(&dir)? {
            Some(text) => Manifest::parse(&text).map_err(OpenError::Manifest)?,
            None => Manifest::new(dp),
        };
        for (id, decl) in &keyspaces {
            if let Some((key, merge)) = m.keyspaces.get(id)
                && (key.as_str(), merge.as_str()) != (decl.key, decl.merge)
            {
                return Err(OpenError::KeyspaceType {
                    id: *id,
                    recorded: (key.clone(), merge.clone()),
                    registered: (decl.key, decl.merge),
                });
            }
        }
        sweep_orphans(&dir, &m)?;
        let tables = open_tables(&dir, &m, &keyspaces)?;
        let replayed = replay(&dir, &m, &keyspaces)?;

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
                    let file = OpenOptions::new()
                        .append(true)
                        .open(dir.join(seg.file_name()))?;
                    let size = file.metadata()?.len() as usize;
                    Wal {
                        file,
                        buffer: Vec::new(),
                        size,
                    }
                }
                None => new_segment(&dir, &mut m)?,
            })
        };

        m.policy = dp;
        m.keyspaces = keyspaces
            .iter()
            .map(|(id, d)| (*id, (d.key.to_string(), d.merge.to_string())))
            .collect();
        manifest::write(&dir, &m)?;

        let writer = StoreWriter::new(Some(dir), m, wal, dp);
        Ok(Store::new(
            replayed,
            tables,
            writer,
            keyspaces,
            cache,
            Some(lock),
        ))
    }
}

/// Create the next WAL segment and list it in `m` as active. The file is
/// created before the manifest names it; the manifest write syncs the
/// directory entry.
fn new_segment(dir: &Path, m: &mut Manifest) -> io::Result<Wal> {
    let seg = Segment {
        number: m.next_file,
        sealed: None,
    };
    let file = OpenOptions::new()
        .append(true)
        .create_new(true)
        .open(dir.join(seg.file_name()))?;
    m.next_file += 1;
    m.segments.push(seg);
    Ok(Wal {
        file,
        buffer: Vec::new(),
        size: 0,
    })
}

/// Delete WAL segments and checkpoints the manifest does not name, and any
/// half-written manifest. A crash between creating a file and naming it,
/// or between dropping a file from the manifest and deleting it, leaves
/// these behind. Files with other names are left alone.
fn sweep_orphans(dir: &Path, m: &Manifest) -> io::Result<()> {
    let live: HashSet<String> = m.live_files().collect();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let numbered = name.split_once('.').is_some_and(|(num, ext)| {
            matches!(ext, "wal" | "ckpt")
                && !num.is_empty()
                && num.bytes().all(|b| b.is_ascii_digit())
        });
        if (numbered && !live.contains(name)) || name == MANIFEST_TMP {
            fs::remove_file(entry.path())?;
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

fn corrupt(path: &Path, offset: usize, why: impl Into<String>) -> OpenError {
    OpenError::Corrupt {
        file: path.to_path_buf(),
        offset: offset as u64,
        why: why.into(),
    }
}

fn frame_error(path: &Path, offset: usize) -> impl Fn(FrameError) -> OpenError + '_ {
    move |e| match e {
        FrameError::Unknown(id) => OpenError::UnknownKeyspace(id),
        FrameError::Decode(why) => corrupt(path, offset, why),
    }
}

/// Open every checkpoint's table. Only footers and indexes are read; data
/// blocks are paged in through the cache as reads need them.
fn open_tables(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
) -> Result<Vec<Arc<Table>>, OpenError> {
    let mut tables = Vec::with_capacity(m.checkpoints.len());
    for ckpt in &m.checkpoints {
        let path = dir.join(ckpt.file_name());
        let table = Table::open(&path, ckpt.number).map_err(|e| table_error(&path, e))?;
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
        if let Some(ks) = table.keyspaces().find(|ks| !decls.contains_key(ks)) {
            return Err(OpenError::UnknownKeyspace(ks));
        }
        tables.push(Arc::new(table));
    }
    Ok(tables)
}

fn table_error(path: &Path, e: TableError) -> OpenError {
    match e {
        TableError::Io(e) => OpenError::Io(e),
        TableError::Corrupt { offset, why } => corrupt(path, offset as usize, why),
    }
}

/// Replay every WAL segment, folding all frames into one tree per
/// keyspace. Nothing can hold a snapshot during open, so one layer is
/// equivalent to a layer per epoch, and reads stay cheap. The WAL holds
/// only epochs after the last checkpoint, so replay starts there.
fn replay(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
) -> Result<Replayed, OpenError> {
    let mut out = Replayed {
        epoch: m.checkpointed(),
        ..Replayed::default()
    };
    for seg in &m.segments {
        let path = dir.join(seg.file_name());
        let bytes = fs::read(&path)?;
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
            apply_frame(payload, decls, &mut out.wal).map_err(frame_error(&path, at))?;
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
    /// Held for the store's life; released when the file is dropped.
    _lock: Option<File>,
}

impl Store {
    fn new(
        replayed: Replayed,
        tables: Vec<Arc<Table>>,
        writer: StoreWriter,
        keyspaces: HashMap<u64, KeyspaceDecl>,
        cache: BlockCache,
        lock: Option<File>,
    ) -> Self {
        let Replayed { epoch, wal, .. } = replayed;
        let layers = if wal.is_empty() {
            Vec::new()
        } else {
            vec![Arc::new(Layer { epoch, trees: wal })]
        };
        Store {
            current: ArcSwap::from_pointee(Version {
                epoch,
                layers,
                tables,
            }),
            writer: Mutex::new(writer),
            active_snapshots: Mutex::new(BTreeMap::new()),
            keyspaces,
            cache,
            _lock: lock,
        }
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
        let mut map = self.active_snapshots.lock().unwrap();
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
    /// [`finish`](Self::finish) aborts it.
    pub(crate) fn begin(&self) -> WriteTx<'_> {
        // lock first: the snapshot must not be older than the last commit
        let writer = self.writer.lock().unwrap();
        WriteTx::new(self.snapshot(), writer)
    }

    /// Commit `tx`: log it, then publish it to readers. Returns the epoch
    /// it committed as, or `None` if it changed nothing, in which case no
    /// epoch is spent and nothing is logged.
    ///
    /// The WAL write and fsync happen before readers can see the tx, so
    /// under a strict policy nothing visible can be lost to a crash.
    pub(crate) fn finish(&self, tx: WriteTx<'_>) -> Option<u64> {
        let (epoch, overlay, snapshot, mut writer) = tx.into_parts();
        debug_assert!(
            std::ptr::eq(snapshot.store(), self),
            "tx from another store"
        );

        // opening a keyspace only to read it leaves an empty handle behind
        let mut sections: Vec<_> = overlay.into_iter().filter(|(_, u)| !u.is_empty()).collect();
        if sections.is_empty() {
            return None;
        }
        // deterministic WAL bytes for the same tx
        sections.sort_unstable_by_key(|(ks, _)| *ks);

        // payload: repeated `keyspace u64 | len u64 | (ordered key, Option<value>)*`
        let mut payload = writer.wal.is_some().then(Vec::new);
        let mut trees = HashMap::with_capacity(sections.len());
        for (ks, underlay) in sections {
            let tree = match payload.as_mut() {
                Some(p) => encode_section(ks, p, |out| underlay.seal(Some(out))),
                None => underlay.seal(None),
            };
            trees.insert(ks, tree);
        }

        if let Some(p) = payload {
            writer.log(epoch, &p);
        }

        let current = self.current.load();
        debug_assert_eq!(current.epoch + 1, epoch, "commit out of order");
        let mut layers = current.layers.clone();
        layers.push(Arc::new(Layer { epoch, trees }));
        self.current.store(Arc::new(Version {
            epoch,
            layers,
            tables: current.tables.clone(),
        }));

        // release the writer only after publishing, so the next tx's
        // snapshot includes this one
        drop(writer);
        drop(snapshot);
        Some(epoch)
    }

    /// Write everything committed since the last checkpoint to a new table,
    /// retire the WAL segments it covers, and drop the covered layers from
    /// memory. Reads of that data then go through the block cache. Returns
    /// the epoch it covers through, or `None` if nothing was committed
    /// since the last one or the store is in memory.
    ///
    /// Each checkpoint is a delta over the previous one; they accumulate
    /// until compaction exists. Snapshots taken before a checkpoint keep
    /// their own layers alive until they are dropped.
    ///
    /// Holds the writer lock throughout, so commits wait for it.
    pub(crate) fn checkpoint(&self) -> io::Result<Option<u64>> {
        let mut writer = self.writer.lock().unwrap();
        let w = &mut *writer;
        let Some(dir) = w.dir.as_deref() else {
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
        let mut table = TableBuilder::create(&path)?;
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
                .filter_map(|l| l.trees.get(&ks).map(|t| &**t))
                .collect();
            table.begin_keyspace(ks)?;
            (self.keyspaces[&ks].write_run)(&trees, &mut |k, v| table.push(k, v))?;
        }
        table.finish(ckpt.epoch)?;
        let table = Table::open(&path, ckpt.number).map_err(|e| match e {
            TableError::Io(e) => e,
            e => io::Error::other(e.to_string()),
        })?;
        next.checkpoints.push(ckpt);

        // every segment holds only epochs the checkpoint now covers
        let retired = std::mem::take(&mut next.segments);
        let wal = match w.wal {
            Some(_) => Some(new_segment(dir, &mut next)?),
            None => None,
        };
        manifest::write(dir, &next)?;

        // the checkpoint is durable: swap its table in for the layers it
        // covers, which is all of them
        let mut tables = version.tables.clone();
        tables.push(Arc::new(table));
        self.current.store(Arc::new(Version {
            epoch: version.epoch,
            layers: Vec::new(),
            tables,
        }));

        // the old segment's buffered frames, if any, are covered too; its
        // drop flushes them into a file that is about to be deleted
        w.wal = wal;
        w.manifest = next;
        for seg in retired {
            // a failure leaves an orphan for the next open to sweep
            let _ = fs::remove_file(dir.join(seg.file_name()));
        }
        Ok(Some(version.epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Change;

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

    fn open(dir: &Path, dp: DurabilityPolicy) -> Result<Store, OpenError> {
        StoreBuilder::new(dp).dir(dir).keyspace(KS).build()
    }

    fn put(store: &Store, kvs: &[(u8, Option<i64>)]) -> Option<u64> {
        let mut tx = store.begin();
        let mut ks = tx.keyspace(KS);
        for &(k, v) in kvs {
            let change = match v {
                Some(v) => Change::Insert(v),
                None => Change::Delete(0),
            };
            if let Some(r) = ks.resolve(&k, change) {
                ks.absorb(k, r);
            }
        }
        store.finish(tx)
    }

    fn read(store: &Store, k: u8) -> Option<i64> {
        store.snapshot().get(KS, &k).map(|v| *v)
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
        assert_eq!(before.get(KS, &1).as_deref(), Some(&10));
        assert_eq!(before.epoch(), 1);
        assert_eq!(read(&store, 1), Some(11));
    }

    #[test]
    fn empty_and_aborted_txs_spend_no_epoch() {
        let store = mem();
        // read-only: the keyspace is opened but nothing is written
        let mut tx = store.begin();
        assert_eq!(tx.keyspace(KS).get(&1), None);
        assert_eq!(store.finish(tx), None);
        // aborted: dropped without finish
        {
            let mut tx = store.begin();
            tx.keyspace(KS).absorb(1, Change::Insert(5));
        }
        assert_eq!(read(&store, 1), None);
        assert_eq!(put(&store, &[(1, Some(10))]), Some(1));
        assert!(store.active_snapshots.lock().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "keyspace 9 not declared")]
    fn undeclared_keyspaces_are_refused() {
        let store = mem();
        let mut tx = store.begin();
        tx.keyspace(KeyspaceId::<u8, i64>::new(9));
    }

    #[test]
    fn writing_policies_need_a_directory() {
        let err = StoreBuilder::new(DurabilityPolicy::RELAXED).build();
        assert!(matches!(err, Err(OpenError::Policy(_))));
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
        assert!(matches!(err, OpenError::Corrupt { offset: 0, .. }), "{err}");
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
        assert!(matches!(other, Err(OpenError::KeyspaceType { id: 3, .. })));

        let missing = StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .build();
        assert!(matches!(missing, Err(OpenError::UnknownKeyspace(3))));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_directory_opens_once_at_a_time() {
        let dir = tmpdir("lock");
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert!(matches!(
            open(&dir, DurabilityPolicy::RELAXED),
            Err(OpenError::Locked)
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
        table
            .scan(KS.raw(), &BlockCache::new(1 << 20))
            .into_iter()
            .map(|(k, v)| {
                (
                    crate::codec::decode_all_ordered::<u8>(&k).unwrap(),
                    decode_all::<Option<i64>>(&v).unwrap(),
                )
            })
            .collect()
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
        assert!(matches!(err, OpenError::Corrupt { .. }), "{err}");
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
        let mut tx = store.begin();
        let mut ks = tx.keyspace(KS);
        let r = ks.resolve(&1, Change::Modify(5)).unwrap();
        assert!(matches!(r, Change::Replace(10, 15)));
        ks.absorb(1, r);
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

        assert_eq!(first.get(KS, &1).as_deref(), Some(&10));
        assert_eq!(first.get(KS, &2).as_deref(), None);
        assert_eq!(second.get(KS, &1).as_deref(), Some(&11));
        assert_eq!(second.get(KS, &2).as_deref(), Some(&20));
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
                .cache_bytes(16 << 10)
                .build()
                .unwrap()
        };
        {
            let store = build();
            for chunk in (0..20_000u32).collect::<Vec<_>>().chunks(5_000) {
                let mut tx = store.begin();
                let mut ks = tx.keyspace(BIG);
                for &i in chunk {
                    ks.absorb(i, Change::Insert(i as i64 * 3));
                }
                store.finish(tx).unwrap();
            }
            put(&store, &[(1, Some(10))]);
            store.checkpoint().unwrap();
        }
        let store = build();
        let snap = store.snapshot();
        for i in (0..20_000u32).step_by(7) {
            assert_eq!(snap.get(BIG, &i).as_deref(), Some(&(i as i64 * 3)));
        }
        assert_eq!(snap.get(BIG, &20_000).as_deref(), None);
        assert_eq!(snap.get(KS, &1).as_deref(), Some(&10));
        let (cached, loads) = store.cache.stats();
        assert!(cached <= 16 << 10, "cache over budget: {cached}");
        assert!(loads > 4, "blocks were paged in and evicted: {loads}");
        drop(snap);
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }
}
