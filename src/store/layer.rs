//! Committed data held in memory: layers of trees, and the version that names the current ones.

use super::*;

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

/// One committed keyspace in one epoch: every key the epoch wrote, with
/// `None` as a tombstone. Commit builds these; reads downcast to them.
pub(crate) type Resident<K, V> = BTreeMap<K, Option<V>>;

/// A type-erased `Resident<K, V>`. Shared, so folding layers reuses the
/// trees of keyspaces only one side touched.
pub(super) type Tree = Arc<dyn Any + Send + Sync>;

/// One keyspace's tree in a layer, with its estimated resident bytes and
/// its run indexes, which are shared wherever the tree is.
#[derive(Clone)]
pub(super) struct LayerTree {
    pub(super) tree: Tree,
    pub(super) bytes: usize,
    pub(super) indexes: Arc<LayerIndexes>,
}

/// What one or more consecutive epochs committed, keyed by keyspace id:
/// the newest value of every key they wrote. Commits push a layer per
/// epoch, and the maintenance thread folds neighbors together so there
/// are only logarithmically many; see [`fold_plan`].
pub(super) struct Layer {
    /// The newest epoch it holds.
    pub(super) epoch: u64,
    pub(super) trees: HashMap<u64, LayerTree>,
    /// Estimated resident bytes of `trees`.
    pub(super) bytes: usize,
}

/// A commit folds its layer into the oldest itself, rather than leaving
/// it to the maintenance thread, while the layers above the oldest hold
/// no more than this: a backlog left by a long-lived snapshot is the
/// thread's to clear.
pub(super) const INLINE_FOLD_LIMIT: usize = 4 << 20;

/// A layer should be more than this many times the size of every newer
/// layer together. Layer sizes then grow geometrically from newest to
/// oldest, so there are O(log n) of them, and each entry is copied
/// O(log n) times over its life in memory.
pub(super) const FOLD_RATIO: usize = 2;

/// Where the newest layers should be folded from: every layer from the
/// returned index on merges into one. That is the oldest layer no more
/// than [`FOLD_RATIO`] times the size of all newer layers together, so
/// after the fold every layer again outweighs everything newer. Past
/// `max_layers`, at least the newest half is folded regardless.
pub(super) fn fold_plan(layers: &[Arc<Layer>], max_layers: usize) -> Option<usize> {
    let mut start = None;
    let mut newer = 0usize;
    for (i, layer) in layers.iter().enumerate().rev() {
        if i + 1 < layers.len() && layer.bytes <= FOLD_RATIO * newer.max(1) {
            start = Some(i);
        }
        newer += layer.bytes;
    }
    if layers.len() > max_layers {
        let half = layers.len() / 2;
        start = Some(start.map_or(half, |s| s.min(half)));
    }
    start
}

/// An immutable view of committed state. Commit builds a new one and swaps
/// it in; a snapshot holds one for its whole life, so its epoch and its
/// contents always agree.
///
/// Every layer is newer than every table: tables hold epochs through the
/// last checkpoint, and layers hold epochs after it.
pub(super) struct Version {
    pub(super) epoch: u64,
    /// In memory, oldest first.
    pub(super) layers: Vec<Arc<Layer>>,
    /// Checkpoints on disk, oldest first, read through the block cache.
    pub(super) tables: Vec<Arc<Table>>,
    /// Estimated resident bytes of `layers`.
    pub(super) resident_bytes: usize,
    /// Keyspace id to the epoch it was truncated at; tables no newer than
    /// that hold nothing visible for it. Mirrors the manifest's floors.
    pub(super) floors: Arc<BTreeMap<u64, u64>>,
    /// Layers through this epoch are being written to a table: a
    /// checkpoint in flight. They are read as before, but no fold may
    /// change them or merge newer layers into them, since the checkpoint
    /// drops them by epoch when it installs.
    pub(super) frozen: Option<u64>,
}

impl Version {
    /// The index of the first layer a fold may touch.
    pub(super) fn live_from(&self) -> usize {
        match self.frozen {
            Some(through) => self.layers.partition_point(|l| l.epoch <= through),
            None => 0,
        }
    }

    /// The tables that can hold visible data for `ks`, newest first.
    pub(super) fn tables_for(&self, ks: u64) -> impl Iterator<Item = &Arc<Table>> {
        let floor = self.floors.get(&ks).copied().unwrap_or(0);
        self.tables
            .iter()
            .rev()
            .take_while(move |t| t.epoch() > floor)
    }
}
