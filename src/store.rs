use std::{
    any::Any,
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs::{File, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use arc_swap::ArcSwap;

use crate::{
    merge::Merge,
    store::DurabilityPolicyCadence::Never,
    tx::{KeyspaceId, WriteTx},
};

#[derive(Clone, Copy)]
enum DurabilityPolicyCadence {
    Never,
    Always,
    Windowed(core::time::Duration),
}

#[derive(Clone, Copy)]
pub(crate) struct DurabilityPolicy {
    // Write(), i.e. the os has the data, durable to process crash
    write: DurabilityPolicyCadence,
    // Fsync/Fullsync; data is on the disk/durable to power loss
    fsync: DurabilityPolicyCadence,
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
struct Version {
    epoch: u64,
    /// Oldest first.
    layers: Vec<Arc<Layer>>,
}

struct Wal {
    file: File,
    buffer: Vec<u8>,
    size: usize,
}
impl Wal {
    // todo: make this fallible!
    /// Buffer one frame holding a whole transaction, so replay sees either
    /// all of a tx or none of it:
    /// `crc32 | epoch u64 | len u64 | payload`, little endian, with the crc
    /// covering everything after it.
    fn append(&mut self, epoch: u64, payload: &[u8]) {
        let epoch = epoch.to_le_bytes();
        let len = (payload.len() as u64).to_le_bytes();

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&epoch);
        hasher.update(&len);
        hasher.update(payload);

        // todo: check if wal + this frame would exceed max wal size
        self.size += 4 + epoch.len() + len.len() + payload.len();

        self.buffer.extend(hasher.finalize().to_le_bytes());
        self.buffer.extend(epoch);
        self.buffer.extend(len);
        self.buffer.extend(payload);
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
impl Drop for Wal {
    /// A clean shutdown hands buffered frames to the OS. It does not fsync:
    /// that is the policy's call.
    fn drop(&mut self) {
        self.flush(true, false);
    }
}

/// State only the writer touches. Held locked for the life of a
/// [`WriteTx`], which is what makes the store single-writer.
pub(crate) struct Writer {
    wal: Option<Wal>,
    dp: DurabilityPolicy,
    last_write: Option<Instant>,
    last_fsync: Option<Instant>,
}
impl Writer {
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
    pub(crate) fn get<K: 'static + Ord, M: Merge>(
        &self,
        keyspace: KeyspaceId<K, M>,
        key: &K,
    ) -> Option<Cow<'_, M::Value>> {
        let hit = self
            .version
            .layers
            .iter()
            .rev()
            .filter_map(|layer| layer.trees.get(&keyspace.raw()))
            .map(|tree| {
                tree.downcast_ref::<Resident<K, M::Value>>()
                    .expect("keyspace id reused with a different type")
            })
            .find_map(|tree| tree.get(key));
        match hit {
            Some(slot) => slot.as_ref().map(Cow::Borrowed),
            // todo: check cached pages, then uncached pages.
            // there is no disk layer yet, so a resident miss is a miss.
            None => None,
        }
    }
}

pub(crate) struct StoreBuilder {
    durability_policy: DurabilityPolicy,
    wal_path: PathBuf,
}

impl StoreBuilder {
    pub(crate) fn new(durability_policy: DurabilityPolicy) -> Self {
        StoreBuilder {
            durability_policy,
            wal_path: PathBuf::from("WAL"),
        }
    }
    pub(crate) fn wal_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.wal_path = path.into();
        self
    }
    pub(crate) fn build(self) -> Result<Store, &'static str> {
        let StoreBuilder {
            durability_policy,
            wal_path,
        } = self;

        durability_policy.validate()?;
        let wal = if let Never = durability_policy.write {
            None
        } else {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(wal_path)
                .unwrap();
            let size = file.metadata().unwrap().len() as usize;
            Some(Wal {
                file,
                buffer: Vec::new(),
                size,
            })
        };

        // todo: replay the wal
        // even if we're configured with no wal,
        // we should try to find and replay it!
        // this goes for any on disk-data;
        // we can't assume the current settings
        // were the previous run's settings
        let version = Version {
            epoch: 0,
            layers: Vec::new(),
        };

        Ok(Store {
            current: ArcSwap::from_pointee(version),
            writer: Mutex::new(Writer {
                wal,
                dp: durability_policy,
                last_write: None,
                last_fsync: None,
            }),
            active_snapshots: Mutex::new(BTreeMap::new()),
        })
    }
}

pub struct Store {
    /// The latest committed state. Only `finish` stores into it, and only
    /// while holding `writer`.
    current: ArcSwap<Version>,
    writer: Mutex<Writer>,
    /// Live snapshot count per epoch; the minimum bounds which layers
    /// could be folded together.
    active_snapshots: Mutex<BTreeMap<u64, usize>>,
}

impl Store {
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
                Some(p) => {
                    p.extend(ks.to_le_bytes());
                    let len_at = p.len();
                    p.extend(0u64.to_le_bytes());
                    let tree = underlay.seal(Some(p));
                    let len = (p.len() - len_at - 8) as u64;
                    p[len_at..len_at + 8].copy_from_slice(&len.to_le_bytes());
                    tree
                }
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
        self.current.store(Arc::new(Version { epoch, layers }));

        // release the writer only after publishing, so the next tx's
        // snapshot includes this one
        drop(writer);
        drop(snapshot);
        Some(epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Change;

    const KS: KeyspaceId<u8, i64> = KeyspaceId::new(3);

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

    #[test]
    fn commits_are_visible_in_order_and_tombstones_shadow() {
        let store = StoreBuilder::new(DurabilityPolicy::NONE).build().unwrap();
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
        let store = StoreBuilder::new(DurabilityPolicy::NONE).build().unwrap();
        put(&store, &[(1, Some(10))]);
        let before = store.snapshot();
        put(&store, &[(1, Some(11))]);
        assert_eq!(before.get(KS, &1).as_deref(), Some(&10));
        assert_eq!(before.epoch(), 1);
        assert_eq!(read(&store, 1), Some(11));
    }

    #[test]
    fn empty_and_aborted_txs_spend_no_epoch() {
        let store = StoreBuilder::new(DurabilityPolicy::NONE).build().unwrap();
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
    fn wal_gets_one_checksummed_frame_per_tx() {
        let path = std::env::temp_dir().join(format!("fold3-wal-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let store = StoreBuilder::new(DurabilityPolicy::STRICT)
                .wal_path(&path)
                .build()
                .unwrap();
            put(&store, &[(1, Some(10)), (2, Some(20))]);
            put(&store, &[(2, None)]);
            // deleting a key that never existed writes nothing: no frame
            assert_eq!(put(&store, &[(9, None)]), None);
        }
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let mut rest = &bytes[..];
        let mut frames = Vec::new();
        while !rest.is_empty() {
            let crc = u32::from_le_bytes(rest[..4].try_into().unwrap());
            let epoch = u64::from_le_bytes(rest[4..12].try_into().unwrap());
            let len = u64::from_le_bytes(rest[12..20].try_into().unwrap()) as usize;
            assert_eq!(crc, crc32fast::hash(&rest[4..20 + len]));
            let payload = &rest[20..20 + len];
            // one keyspace section: id, length, then (u8 key, Option<i64>)*
            assert_eq!(
                u64::from_le_bytes(payload[..8].try_into().unwrap()),
                KS.raw()
            );
            let body_len = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
            assert_eq!(16 + body_len, len);
            frames.push((epoch, payload[16..].to_vec()));
            rest = &rest[20 + len..];
        }
        // epoch 1: key 1 -> Some(10), key 2 -> Some(20)
        let mut one = vec![1u8, 1];
        one.extend(10i64.to_be_bytes());
        one.extend([2u8, 1]);
        one.extend(20i64.to_be_bytes());
        // epoch 2: key 2 -> None
        assert_eq!(frames, [(1, one), (2, vec![2u8, 0])]);
    }
}
