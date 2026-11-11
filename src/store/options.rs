//! Size limits, memory statistics and the durability policy.

use super::*;

/// Size limits. Memory use is estimated with [`ResidentSize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoreOptions {
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
    /// Commits wait for the maintenance thread once there are more memory
    /// layers than this. Every layer is checked on a read, so this bounds
    /// read cost when maintenance falls behind.
    pub max_layers: usize,
    /// Bytes of buffers a [`Db`](crate::Db)'s graph may keep between
    /// runs; see [`OpDag::retain`](crate::graph::OpDag::retain).
    pub graph_retain: usize,
    /// How much memory the store should plan to use in all, or `None`
    /// for no plan. It is not enforced as such: it sets every memory
    /// limit above that was left at its default to a share of it, so one
    /// number sizes the store. See [`resolved`](Self::resolved).
    pub memory_budget: Option<usize>,
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
            max_layers: 8,
            graph_retain: DEFAULT_GRAPH_RETAIN,
            memory_budget: None,
        }
    }
}

impl StoreOptions {
    /// The options as the store uses them: with a
    /// [`memory_budget`](Self::memory_budget), each memory limit still at
    /// its default becomes a share of the budget, and one set to anything
    /// else is kept.
    ///
    /// | limit | share |
    /// |---|---|
    /// | `memtable_limit` | 1/4 (commits wait at twice it) |
    /// | `tx_memory_limit` | 1/4 |
    /// | `cache_size` | 1/8 |
    /// | `graph_retain` | 1/32 |
    /// | `wal_limit` | at most 1/4, since open replays it into memory |
    ///
    /// Together those are at most 29/32 of the budget; the rest is for
    /// what has no limit of its own: table and run indexes, layers held
    /// by old snapshots, and unread subscriptions. [`MemoryStats::total`]
    /// is the figure to hold against the budget.
    pub fn resolved(&self) -> StoreOptions {
        let Some(budget) = self.memory_budget else {
            return *self;
        };
        let default = StoreOptions::default();
        let mut o = *self;
        if o.memtable_limit == default.memtable_limit {
            o.memtable_limit = budget / 4;
        }
        if o.tx_memory_limit == default.tx_memory_limit {
            o.tx_memory_limit = budget / 4;
        }
        if o.cache_size == default.cache_size {
            o.cache_size = budget / 8;
        }
        if o.graph_retain == default.graph_retain {
            o.graph_retain = budget / 32;
        }
        if o.wal_limit == default.wal_limit {
            o.wal_limit = o.wal_limit.min(budget as u64 / 4);
        }
        o
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.max_tables == 0 {
            return Err(Error::InvalidOption("max tables must be at least one"));
        }
        if self.compaction_width < 2 {
            return Err(Error::InvalidOption(
                "compaction must merge at least two tables",
            ));
        }
        if self.max_layers < 2 {
            return Err(Error::InvalidOption("max layers must be at least two"));
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
    pub(super) fn named(&self) -> BTreeMap<String, u64> {
        [
            ("wal-limit", self.wal_limit),
            ("memtable-limit", self.memtable_limit as u64),
            ("tx-memory-limit", self.tx_memory_limit as u64),
            ("block-size", self.block_size as u64),
            ("cache-size", self.cache_size as u64),
            ("max-tables", self.max_tables as u64),
            ("compaction-width", self.compaction_width as u64),
            ("max-layers", self.max_layers as u64),
            ("graph-retain", self.graph_retain as u64),
            ("memory-budget", self.memory_budget.unwrap_or(0) as u64),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
}

/// Where the store's memory goes, and how much WAL the next open replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStats {
    /// Committed data held in memory, not yet checkpointed. Layers pinned
    /// only by old snapshots are not counted.
    pub memtable_bytes: usize,
    /// WAL bytes the next open would replay.
    pub wal_bytes: u64,
    /// Table blocks in the cache.
    pub cache_bytes: usize,
    /// Table indexes, which stay in memory while the table is open.
    pub index_bytes: usize,
    /// Run indexes built or loaded so far, over layers and tables. Those
    /// of layers pinned only by old snapshots are not counted.
    pub run_index_bytes: usize,
    pub tables: usize,
    /// Layers in memory: commits not yet folded together or checkpointed.
    pub layers: usize,
    /// Layers no longer current that a snapshot still holds. They are
    /// freed when the last snapshot that can see them is dropped.
    pub pinned_bytes: usize,
    /// What the open write transaction, if any, has written so far.
    pub tx_bytes: usize,
    /// The WAL's write buffer as of the last commit: frames the
    /// durability policy has yet to write out.
    pub wal_buffer_bytes: usize,
    /// Changes sent to subscriptions and not yet received, counted once
    /// however many subscribers share them.
    pub subscription_bytes: usize,
    /// Buffers the [`Db`](crate::Db)'s graph holds between runs; see
    /// [`OpDag::retain`](crate::graph::OpDag::retain). Zero for a store
    /// without one.
    pub graph_bytes: usize,
}

impl MemoryStats {
    /// Every byte of memory counted here, together: the figure to hold
    /// against a [memory budget](StoreOptions::memory_budget). The WAL is
    /// on disk, so it is not part of it.
    pub fn total(&self) -> usize {
        self.memtable_bytes
            + self.cache_bytes
            + self.index_bytes
            + self.run_index_bytes
            + self.pinned_bytes
            + self.tx_bytes
            + self.wal_buffer_bytes
            + self.subscription_bytes
            + self.graph_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DurabilityPolicyCadence {
    Never,
    Always,
    Windowed(core::time::Duration),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DurabilityPolicy {
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
    pub(super) fn validate(&self) -> Result<(), &'static str> {
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
    pub(super) fn should(
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
