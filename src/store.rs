use std::{
    any::Any,
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering::Acquire},
    },
    time::Instant,
};

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
            return Err("cannot fsync unwritten data");
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

struct Wal {
    file: File,
    dp: DurabilityPolicy,
    last_write: Option<Instant>,
    last_fsync: Option<Instant>,
    buffer: Vec<u8>,
    size: usize,
}
impl Wal {
    // todo: park the WAL on its own thread? only matters if we go multi-writer...
    // todo: make this fallible!
    fn append(&mut self, keyspace: u64, data: &[u8]) {
        let now = Instant::now();
        let (should_write, should_fsync) = self.dp.should(now, self.last_write, self.last_fsync);

        self.buffer.extend(&crc32fast::hash(data).to_le_bytes());
        self.buffer.extend(keyspace.to_le_bytes());
        self.buffer.extend((data.len() as u64).to_le_bytes());
        self.buffer.extend(data);

        self.size += 4 + 8 + 8 + data.len();

        // todo: check if we would exceed max wal size

        if should_write {
            self.file.write_all(&self.buffer).unwrap();
            self.buffer.clear();
            self.last_write = Some(now);
        }
        if should_fsync {
            self.file.sync_data().unwrap();
            // to accurately reflect the time it actually took to fsync
            self.last_fsync = Some(Instant::now());
        }
    }

    fn size(&self) -> usize {
        debug_assert_eq!(self.size, self.file.metadata().unwrap().len() as usize);
        self.size
    }
}

pub struct Snapshot<'s> {
    epoch: u64,
    store: &'s Store,
}
impl Drop for Snapshot<'_> {
    fn drop(&mut self) {
        let mut map = self.store.active_snapshots.lock().unwrap();
        match map.get_mut(&self.epoch) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                map.remove(&self.epoch);
            }
            None => unreachable!("snapshot epoch missing from map"),
        }
    }
}
impl<'s> Snapshot<'s> {
    /// The newest epoch this snapshot can see.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub(crate) fn get<K: 'static + Ord, M: Merge>(
        &'s self,
        keyspace: KeyspaceId<K, M>,
        key: &K,
    ) -> Option<Cow<'s, M::Value>> {
        self.store.get(self.epoch, keyspace, key)
    }
}

pub(crate) struct StoreBuilder {
    durability_policy: DurabilityPolicy,
}

impl StoreBuilder {
    pub(crate) fn new(durability_policy: DurabilityPolicy) -> Self {
        StoreBuilder { durability_policy }
    }
    pub(crate) fn build(self) -> Result<Store, &'static str> {
        let StoreBuilder { durability_policy } = self;

        durability_policy.validate()?;
        let wal = if let Never = durability_policy.write {
            None
        } else {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open("WAL")
                .unwrap();
            Some(Wal {
                file,
                dp: durability_policy,
                last_write: None,
                last_fsync: None,
                buffer: Vec::new(),
                size: 0,
            })
        };

        let frontier = AtomicU64::default();
        let snapshots = Mutex::new(BTreeMap::new());

        // todo: replay the wal
        // even if we're configured with no wal,
        // we should try to find and replay it!
        let residents = Default::default();

        Ok(Store {
            wal,
            frontier,
            residents,
            active_snapshots: snapshots,
        })
    }
}

pub struct Store {
    wal: Option<Wal>,
    frontier: AtomicU64,
    active_snapshots: Mutex<BTreeMap<u64, usize>>,
    // HM(keyspace, BTM(epoch, BTM(key, Option<V>)))
    residents: HashMap<u64, BTreeMap<u64, Box<dyn Any>>>,
}

impl Store {
    pub(crate) fn snapshot(&self) -> Snapshot<'_> {
        let mut map = self.active_snapshots.lock().unwrap();
        let epoch = self.frontier.load(Acquire);
        *map.entry(epoch).or_default() += 1;
        drop(map);
        Snapshot { epoch, store: self }
    }

    /// Test hook: plant a committed epoch directly in residents.
    #[cfg(test)]
    pub(crate) fn insert_resident<K: 'static + Ord, M: Merge>(
        &mut self,
        epoch: u64,
        keyspace: KeyspaceId<K, M>,
        tree: BTreeMap<K, Option<M::Value>>,
    ) {
        self.residents
            .entry(keyspace.raw())
            .or_default()
            .insert(epoch, Box::new(tree));
    }

    /// Open a write transaction over a snapshot of the latest committed
    /// state. Single writer: a second open tx would materialize against
    /// stale values. todo: enforce that with a writer lock.
    pub(crate) fn begin(&self) -> WriteTx<'_> {
        WriteTx::new(self.snapshot())
    }
    pub(crate) fn finish(&mut self, tx: WriteTx) {
        let WriteTx { overlay, epoch, .. } = tx;
        for (keyspace, underlay) in overlay {
            let ksp = self.residents.entry(keyspace).or_default();
            // todo: must convert changes into Option here...
            ksp.insert(epoch, underlay);
        }
    }

    fn get<'v, K: 'static + Ord, M: Merge>(
        &'v self,
        epoch: u64,
        keyspace: KeyspaceId<K, M>,
        key: &K,
    ) -> Option<Cow<'v, M::Value>> {
        if let Some(ksp) = self.residents.get(&keyspace.raw())
            && let Some(vop) = ksp
                .range(..=epoch)
                .rev()
                .map(|(_, et)| et.downcast_ref::<BTreeMap<K, Option<M::Value>>>().unwrap())
                .filter_map(|t| t.get(&key))
                .next()
        {
            if let Some(v) = vop {
                return Some(Cow::Borrowed(v));
            } else {
                return None;
            }
        }

        // todo: check cached pages, then uncached pages.
        // there is no disk layer yet, so a resident miss is a miss.
        None
    }
}
