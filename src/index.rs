//! Per-run indexes.
//!
//! A keyspace's committed data lives in runs: layers in memory and tables
//! on disk. Each is immutable, and they merge geometrically, so they are
//! the seam for Lucene-style per-segment structures: zone maps, max-score
//! blocks, term dictionaries, vector graphs. A [`RunIndex`] is built once
//! per run, shared by every snapshot that holds the run, and dropped when
//! the run is merged away.
//!
//! Indexes are built on the maintenance thread as it folds layers,
//! checkpoints and compacts, so commits never pay for them. The small
//! layers commits create are indexed lazily, by the first reader that
//! asks. A table's index is written into the table when it
//! [encodes](RunIndex::encode), and decoded on first use; one that does
//! not, or fails to decode, is rebuilt from the table on first use.
//!
//! A run holds the newest value of every key it wrote, including
//! tombstones, as of its epoch. A newer run may overwrite or delete any of
//! them, so an index's answers are candidates: confirm them against the
//! snapshot, or take only bounds from them. Readers walk
//! [`Snapshot::runs_indexed`](crate::store::Snapshot::runs_indexed) and
//! merge the per-run answers.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    marker::PhantomData,
    ops::RangeBounds,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use crate::{
    codec::{Codec, DecodeError, OrderedCodec, decode_all, decode_all_ordered},
    cursor::{Cursor, Source, empty_bounds},
    error::{Error, Result},
    merge::Merge,
    store::Resident,
    table::{BlockCache, Table},
};

/// An auxiliary structure over one run of one keyspace. Starts as
/// [`Default`], is fed every row of the run in key order, then finished.
pub trait RunIndex<K, V>: Default + Send + Sync + 'static {
    /// Names this index's frames in table files. Change it when the
    /// encoding changes: frames under another name are ignored and the
    /// index is rebuilt.
    const NAME: &'static str;

    /// One row of the run, in key order. `None` is a tombstone: the key
    /// was deleted as of this run, hiding it in older runs.
    fn push(&mut self, key: &K, value: Option<&V>);

    /// Every row was pushed. Returning `false` declines to keep the index,
    /// for a structure not worth its memory for a run this small; readers
    /// then scan the run instead.
    fn finish(&mut self) -> bool {
        true
    }

    /// Estimated bytes the index holds in memory.
    fn resident_bytes(&self) -> usize;

    /// Bytes for a table to keep beside its run, or `false` to rebuild the
    /// index from the table whenever it is opened and first used, which
    /// suits anything cheaper to build than to read.
    fn encode(&self, out: &mut Vec<u8>) -> bool {
        let _ = out;
        false
    }

    /// The index from bytes [`encode`](Self::encode) wrote. `None` falls
    /// back to rebuilding it, so a bad or stale frame never bricks a
    /// keyspace.
    fn decode(bytes: &[u8]) -> Option<Self> {
        let _ = bytes;
        None
    }
}

/// A built index, type-erased: an `Arc<I>`, or `None` where the index
/// declined the run.
pub(crate) type Aux = Option<Arc<dyn Any + Send + Sync>>;

/// Builds one index from encoded rows.
pub(crate) trait EncodedBuilder: Send {
    fn push(&mut self, key: &[u8], value: &[u8]) -> Result<(), DecodeError>;
    /// The index, and its frame if it encodes.
    fn finish(self: Box<Self>) -> (Aux, Option<Vec<u8>>);
}

/// One index kind registered on a keyspace, with its types erased.
#[derive(Clone, Copy)]
pub(crate) struct IndexKind {
    pub id: TypeId,
    pub name: &'static str,
    /// A builder fed encoded rows, for tables.
    pub start: fn() -> Box<dyn EncodedBuilder>,
    /// Build over a layer's tree.
    pub build_tree: fn(&(dyn Any + Send + Sync)) -> Aux,
    pub decode: fn(&[u8]) -> Aux,
    pub bytes: fn(&(dyn Any + Send + Sync)) -> usize,
}

impl IndexKind {
    pub fn of<K, V, I>() -> Self
    where
        K: OrderedCodec + Send + Sync + 'static,
        V: Codec + Send + Sync + 'static,
        I: RunIndex<K, V>,
    {
        IndexKind {
            id: TypeId::of::<I>(),
            name: I::NAME,
            start: || Box::new(Encoded::<K, V, I>(I::default(), PhantomData)),
            build_tree: |tree| {
                let tree = tree
                    .downcast_ref::<Resident<K, V>>()
                    .expect("layer tree has the registered type");
                let mut index = I::default();
                for (k, v) in tree {
                    index.push(k, v.as_ref());
                }
                index.finish().then(|| Arc::new(index) as _)
            },
            decode: |bytes| I::decode(bytes).map(|i| Arc::new(i) as _),
            bytes: |index| {
                index
                    .downcast_ref::<I>()
                    .expect("index has its registered type")
                    .resident_bytes()
            },
        }
    }

    /// Bytes `aux` holds.
    pub fn size(&self, aux: &Aux) -> usize {
        aux.as_deref().map_or(0, |a| (self.bytes)(a))
    }
}

struct Encoded<K, V, I>(I, PhantomData<fn() -> (K, V)>);

impl<K, V, I> EncodedBuilder for Encoded<K, V, I>
where
    K: OrderedCodec + Send + Sync + 'static,
    V: Codec + Send + Sync + 'static,
    I: RunIndex<K, V>,
{
    fn push(&mut self, key: &[u8], value: &[u8]) -> Result<(), DecodeError> {
        let k = decode_all_ordered::<K>(key)?;
        let v = decode_all::<Option<V>>(value)?;
        self.0.push(&k, v.as_ref());
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> (Aux, Option<Vec<u8>>) {
        if !self.0.finish() {
            return (None, None);
        }
        let mut frame = Vec::new();
        let frame = self.0.encode(&mut frame).then_some(frame);
        (Some(Arc::new(self.0)), frame)
    }
}

/// The indexes of one layer tree, one slot per kind registered on its
/// keyspace, each built at most once: by the fold that made the tree, or
/// by the first reader to ask.
#[derive(Default)]
pub(crate) struct LayerIndexes(Box<[OnceLock<Aux>]>);

impl LayerIndexes {
    pub fn new(kinds: usize) -> Self {
        LayerIndexes((0..kinds).map(|_| OnceLock::new()).collect())
    }

    /// Build every slot now, over `tree`.
    pub fn build_all(&self, kinds: &[IndexKind], tree: &(dyn Any + Send + Sync)) {
        for (slot, kind) in self.0.iter().zip(kinds) {
            slot.get_or_init(|| (kind.build_tree)(tree));
        }
    }

    pub fn get(&self, slot: usize, kind: &IndexKind, tree: &(dyn Any + Send + Sync)) -> Aux {
        self.0[slot].get_or_init(|| (kind.build_tree)(tree)).clone()
    }

    /// Bytes held by the slots built so far.
    pub fn bytes(&self, kinds: &[IndexKind]) -> usize {
        self.0
            .iter()
            .zip(kinds)
            .filter_map(|(slot, kind)| slot.get().map(|aux| kind.size(aux)))
            .sum()
    }
}

/// The indexes of a table's runs, by keyspace and index name, built,
/// decoded or rebuilt as they are first wanted.
#[derive(Default)]
pub(crate) struct TableIndexes(Mutex<HashMap<(u64, &'static str), Aux>>);

impl TableIndexes {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(u64, &'static str), Aux>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Keep an index built while the table was written.
    pub fn insert(&self, ks: u64, name: &'static str, aux: Aux) {
        self.lock().insert((ks, name), aux);
    }

    /// Bytes held by the indexes loaded so far.
    pub fn bytes<'k>(&self, kinds: impl Fn(u64) -> &'k [IndexKind]) -> usize {
        self.lock()
            .iter()
            .filter_map(|((ks, name), aux)| {
                let kind = kinds(*ks).iter().find(|k| k.name == *name)?;
                Some(kind.size(aux))
            })
            .sum()
    }
}

/// `kind`'s index over keyspace `ks` of `table`: kept from when it was
/// written, decoded from its frame, or else rebuilt by reading the run.
/// Concurrent first uses may each load it; the first stored wins.
pub(crate) fn table_index(table: &Table, ks: u64, kind: &IndexKind) -> Result<Aux> {
    if let Some(aux) = table.indexes().lock().get(&(ks, kind.name)) {
        return Ok(aux.clone());
    }
    // a damaged frame is rebuilt from the run, like a stale one
    let decoded = match table.aux_frame(ks, kind.name) {
        Ok(Some(frame)) => (kind.decode)(&frame),
        Ok(None) | Err(Error::Corrupt { .. }) => None,
        Err(e) => return Err(e),
    };
    let aux = match decoded {
        Some(aux) => Some(aux),
        None => {
            let mut builder = (kind.start)();
            let mut cursor = table.cursor(ks, None);
            cursor.start()?;
            while let Some((k, v)) = cursor.current() {
                builder
                    .push(k, v)
                    .map_err(|e| Error::corrupt(table.path(), 0, format!("keyspace {ks}: {e}")))?;
                cursor.advance()?;
            }
            builder.finish().0
        }
    };
    Ok(table
        .indexes()
        .lock()
        .entry((ks, kind.name))
        .or_insert(aux)
        .clone())
}

/// One run of a keyspace, with its index of type `I`, as
/// [`Snapshot::runs_indexed`](crate::store::Snapshot::runs_indexed) yields
/// them.
pub struct IndexedRun<'a, K, M: Merge, I> {
    pub(crate) run: Run<'a, K, M::Value>,
    pub(crate) index: Aux,
    pub(crate) _i: PhantomData<fn() -> I>,
}

pub(crate) enum Run<'a, K, V> {
    Layer {
        tree: &'a Resident<K, V>,
        epoch: u64,
    },
    Table {
        table: &'a Table,
        ks: u64,
        cache: &'a BlockCache,
    },
}

impl<'a, K, M, I> IndexedRun<'a, K, M, I>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
    I: 'static,
{
    /// The run's index, or `None` where it declined the run.
    pub fn index(&self) -> Option<&I> {
        self.index
            .as_deref()
            .map(|i| i.downcast_ref().expect("index has its registered type"))
    }

    /// The newest epoch the run holds.
    pub fn epoch(&self) -> u64 {
        match &self.run {
            Run::Layer { epoch, .. } => *epoch,
            Run::Table { table, .. } => table.epoch(),
        }
    }

    /// What this run holds for `key`: `None` if it does not mention it,
    /// `Some(None)` if it deleted it. Newer runs may say otherwise.
    pub fn get(&self, key: &K) -> Result<Option<Option<M::Value>>>
    where
        M::Value: Clone,
    {
        match &self.run {
            Run::Layer { tree, .. } => Ok(tree.get(key).cloned()),
            Run::Table { table, ks, cache } => {
                let mut probe = Vec::with_capacity(key.encode_ordered_size());
                key.encode_ordered(&mut probe);
                match table.get(*ks, &probe, cache)? {
                    None => Ok(None),
                    Some((block, range)) => decode_all::<Option<M::Value>>(&block[range])
                        .map(Some)
                        .map_err(|e| {
                            Error::corrupt(table.path(), 0, format!("keyspace {ks}: {e}"))
                        }),
                }
            }
        }
    }

    /// The entries this run holds with keys in `range`, in key order,
    /// without the keys it deleted. Newer runs may overwrite or delete
    /// any of them.
    pub fn range(&self, range: impl RangeBounds<K>) -> Cursor<'a, K, M> {
        self.cursor(range, false)
    }

    /// [`range`](Self::range), in reverse key order.
    pub fn range_rev(&self, range: impl RangeBounds<K>) -> Cursor<'a, K, M> {
        self.cursor(range, true)
    }

    fn cursor(&self, range: impl RangeBounds<K>, rev: bool) -> Cursor<'a, K, M> {
        let (lower, upper) = (range.start_bound().cloned(), range.end_bound().cloned());
        let mut sources = Vec::new();
        if !empty_bounds(lower.as_ref(), upper.as_ref()) {
            sources.push(match &self.run {
                Run::Layer { tree, .. } => Source::layer(tree, lower.as_ref(), upper.as_ref(), rev),
                Run::Table { table, ks, cache } => {
                    Source::table(table.cursor(*ks, Some(cache)), *ks, rev)
                }
            });
        }
        Cursor::new(sources, lower, upper, rev, None)
    }
}

impl<K, M: Merge, I> std::fmt::Debug for IndexedRun<'_, K, M, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.run {
            Run::Layer { .. } => "layer",
            Run::Table { .. } => "table",
        };
        f.debug_struct("IndexedRun")
            .field("run", &kind)
            .field("indexed", &self.index.is_some())
            .finish()
    }
}
