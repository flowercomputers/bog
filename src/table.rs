//! Sorted tables: the on-disk form of a checkpoint.
//!
//! A table holds sorted runs for any number of keyspaces. Keys are stored in
//! their [`OrderedCodec`](crate::codec::OrderedCodec) encoding, whose byte
//! order matches the key type's `Ord`, so the table compares raw bytes and
//! never needs to know key or value types.
//!
//! ```text
//! magic "fold3sst"
//! data block*      entries, then crc32 of the entries
//! index            per keyspace: the offset, length and first key of each block
//! footer           index offset u64 | index len u64 | epoch u64 | index crc32 | magic
//! ```
//!
//! An entry is `key len u32 | key | value len u32 | value`, the value being
//! the encoded `Option<V>` (a tombstone is `None`). All integers are little
//! endian.
//!
//! Opening a table reads only its footer and index. Data blocks are read on
//! demand through a [`BlockCache`] shared by every table in the store, and
//! their checksums are verified when they are loaded.

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const MAGIC: &[u8; 8] = b"fold3sst";
const FOOTER: usize = 8 + 8 + 8 + 4 + 8;
/// The default block size: a block is cut once its entries reach this.
pub(crate) const DEFAULT_BLOCK_SIZE: usize = 4096;

#[derive(Debug)]
pub(crate) enum TableError {
    Io(io::Error),
    Corrupt { offset: u64, why: String },
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Corrupt { offset, why } => write!(f, "corrupt at byte {offset}: {why}"),
        }
    }
}

impl From<io::Error> for TableError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

fn corrupt(offset: u64, why: impl Into<String>) -> TableError {
    TableError::Corrupt {
        offset,
        why: why.into(),
    }
}

struct BlockHandle {
    offset: u64,
    /// Including the trailing crc.
    len: u32,
    first_key: Box<[u8]>,
}

/// Writes a table. Keyspaces must be begun in ascending id order and keys
/// pushed in ascending encoded order within each.
pub(crate) struct TableBuilder {
    out: BufWriter<File>,
    block_size: usize,
    offset: u64,
    index: Vec<(u64, Vec<BlockHandle>)>,
    block: Vec<u8>,
    block_first: Option<Vec<u8>>,
}

impl TableBuilder {
    /// Create a new table file; fails if it already exists. Blocks are cut
    /// once their entries reach `block_size` bytes.
    pub fn create(path: &Path, block_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut out = BufWriter::new(file);
        out.write_all(MAGIC)?;
        Ok(TableBuilder {
            out,
            block_size,
            offset: MAGIC.len() as u64,
            index: Vec::new(),
            block: Vec::with_capacity(block_size * 2),
            block_first: None,
        })
    }

    pub fn begin_keyspace(&mut self, ks: u64) -> io::Result<()> {
        self.cut_block()?;
        if let Some((prev, _)) = self.index.last() {
            assert!(*prev < ks, "keyspaces out of order");
        }
        self.index.push((ks, Vec::new()));
        Ok(())
    }

    pub fn push(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        assert!(!self.index.is_empty(), "push before begin_keyspace");
        if self.block_first.is_none() {
            self.block_first = Some(key.to_vec());
        }
        self.block.extend((key.len() as u32).to_le_bytes());
        self.block.extend(key);
        self.block.extend((value.len() as u32).to_le_bytes());
        self.block.extend(value);
        if self.block.len() >= self.block_size {
            self.cut_block()?;
        }
        Ok(())
    }

    fn cut_block(&mut self) -> io::Result<()> {
        let Some(first_key) = self.block_first.take() else {
            return Ok(());
        };
        self.block
            .extend(crc32fast::hash(&self.block).to_le_bytes());
        self.out.write_all(&self.block)?;
        let len = self.block.len() as u32;
        self.index
            .last_mut()
            .expect("a block belongs to a keyspace")
            .1
            .push(BlockHandle {
                offset: self.offset,
                len,
                first_key: first_key.into(),
            });
        self.offset += len as u64;
        self.block.clear();
        Ok(())
    }

    /// Write the index and footer, then fsync. `epoch` is the newest epoch
    /// the table covers.
    pub fn finish(mut self, epoch: u64) -> io::Result<()> {
        self.cut_block()?;
        let mut index = Vec::new();
        let keyspaces: Vec<_> = self.index.iter().filter(|(_, b)| !b.is_empty()).collect();
        index.extend((keyspaces.len() as u32).to_le_bytes());
        for (ks, blocks) in keyspaces {
            index.extend(ks.to_le_bytes());
            index.extend((blocks.len() as u32).to_le_bytes());
            for b in blocks {
                index.extend(b.offset.to_le_bytes());
                index.extend(b.len.to_le_bytes());
                index.extend((b.first_key.len() as u32).to_le_bytes());
                index.extend(&b.first_key);
            }
        }
        self.out.write_all(&index)?;
        self.out.write_all(&self.offset.to_le_bytes())?;
        self.out.write_all(&(index.len() as u64).to_le_bytes())?;
        self.out.write_all(&epoch.to_le_bytes())?;
        self.out.write_all(&crc32fast::hash(&index).to_le_bytes())?;
        self.out.write_all(MAGIC)?;
        let file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()
    }
}

/// An open table: its index in memory, its blocks on disk.
pub(crate) struct Table {
    /// Unique within the store; names this table's blocks in the cache.
    number: u64,
    path: PathBuf,
    file: File,
    epoch: u64,
    index: HashMap<u64, Vec<BlockHandle>>,
}

/// A little-endian cursor over the index bytes.
struct Reader<'a> {
    b: &'a [u8],
    base: u64,
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], TableError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.b.len())
            .ok_or_else(|| corrupt(self.base + self.at as u64, "index truncated"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, TableError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, TableError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

impl Table {
    /// Open a table, reading and checking its footer and index.
    pub fn open(path: &Path, number: u64) -> Result<Table, TableError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len < (MAGIC.len() + FOOTER) as u64 {
            return Err(corrupt(0, "too short to be a table"));
        }
        let mut head = [0u8; 8];
        file.read_exact_at(&mut head, 0)?;
        if &head != MAGIC {
            return Err(corrupt(0, "not a table"));
        }

        let footer_at = len - FOOTER as u64;
        let mut footer = [0u8; FOOTER];
        file.read_exact_at(&mut footer, footer_at)?;
        let field = |at: usize| u64::from_le_bytes(footer[at..at + 8].try_into().unwrap());
        let (index_at, index_len, epoch) = (field(0), field(8), field(16));
        let crc = u32::from_le_bytes(footer[24..28].try_into().unwrap());
        if &footer[28..] != MAGIC {
            return Err(corrupt(footer_at, "bad footer magic"));
        }
        if index_at
            .checked_add(index_len)
            .is_none_or(|end| end != footer_at)
        {
            return Err(corrupt(footer_at, "index does not end at the footer"));
        }

        let mut raw = vec![0u8; index_len as usize];
        file.read_exact_at(&mut raw, index_at)?;
        if crc32fast::hash(&raw) != crc {
            return Err(corrupt(index_at, "index checksum mismatch"));
        }
        let mut r = Reader {
            b: &raw,
            base: index_at,
            at: 0,
        };
        let mut index = HashMap::new();
        for _ in 0..r.u32()? {
            let ks = r.u64()?;
            let n = r.u32()?;
            let mut blocks = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let offset = r.u64()?;
                let len = r.u32()?;
                let key_len = r.u32()? as usize;
                let first_key: Box<[u8]> = r.take(key_len)?.into();
                if offset
                    .checked_add(len as u64)
                    .is_none_or(|end| end > index_at)
                    || len < 4
                {
                    return Err(corrupt(index_at, "block outside the data region"));
                }
                blocks.push(BlockHandle {
                    offset,
                    len,
                    first_key,
                });
            }
            index.insert(ks, blocks);
        }
        if r.at != raw.len() {
            return Err(corrupt(index_at, "trailing bytes in index"));
        }

        Ok(Table {
            number,
            path: path.to_path_buf(),
            file,
            epoch,
            index,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Estimated bytes the in-memory index holds.
    pub fn index_bytes(&self) -> usize {
        self.index
            .values()
            .flatten()
            .map(|b| size_of::<BlockHandle>() + b.first_key.len())
            .sum()
    }

    pub fn keyspaces(&self) -> impl Iterator<Item = u64> + '_ {
        self.index.keys().copied()
    }

    /// The encoded value stored for `key` in keyspace `ks`, as a block and
    /// the value's byte range within it. `None` if this table does not
    /// mention the key; a tombstone is a hit whose value decodes to `None`.
    pub fn get(
        &self,
        ks: u64,
        key: &[u8],
        cache: &BlockCache,
    ) -> Result<Option<(Arc<[u8]>, Range<usize>)>, TableError> {
        let Some(blocks) = self.index.get(&ks) else {
            return Ok(None);
        };
        // the last block starting at or before the key is the only one
        // that can hold it
        let i = blocks.partition_point(|b| &*b.first_key <= key);
        let Some(handle) = i.checked_sub(1).map(|i| &blocks[i]) else {
            return Ok(None);
        };
        let block = self.block(handle, cache)?;
        let mut entries = Entries {
            b: &block,
            at: 0,
            base: handle.offset,
        };
        while let Some((k, v)) = entries.next()? {
            match k.cmp(key) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Ok(Some((block.clone(), v))),
                std::cmp::Ordering::Greater => break,
            }
        }
        Ok(None)
    }

    /// A block's entries, from the cache or from disk.
    fn block(&self, handle: &BlockHandle, cache: &BlockCache) -> Result<Arc<[u8]>, TableError> {
        cache.get_or_load((self.number, handle.offset), || {
            let mut buf = vec![0u8; handle.len as usize];
            self.file.read_exact_at(&mut buf, handle.offset)?;
            let (entries, crc) = buf.split_at(buf.len() - 4);
            if crc32fast::hash(entries) != u32::from_le_bytes(crc.try_into().unwrap()) {
                return Err(corrupt(handle.offset, "block checksum mismatch"));
            }
            buf.truncate(buf.len() - 4);
            Ok(buf.into())
        })
    }

    /// Every entry in keyspace `ks`, in order, as raw bytes.
    #[cfg(test)]
    pub fn scan(&self, ks: u64, cache: &BlockCache) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::new();
        for handle in self.index.get(&ks).into_iter().flatten() {
            let block = self.block(handle, cache).unwrap();
            let mut entries = Entries {
                b: &block,
                at: 0,
                base: handle.offset,
            };
            while let Some((k, v)) = entries.next().unwrap() {
                out.push((k.to_vec(), block[v].to_vec()));
            }
        }
        out
    }
}

/// Walks the entries of a verified block.
struct Entries<'a> {
    b: &'a [u8],
    at: usize,
    base: u64,
}

impl<'a> Entries<'a> {
    /// The next `(key, value range)`, or `None` at the end of the block.
    fn next(&mut self) -> Result<Option<(&'a [u8], Range<usize>)>, TableError> {
        if self.at == self.b.len() {
            return Ok(None);
        }
        let key = self.field()?;
        let value = self.field()?;
        Ok(Some((&self.b[key], value)))
    }

    fn field(&mut self) -> Result<Range<usize>, TableError> {
        let bad = |at: usize| corrupt(self.base + at as u64, "entry runs past its block");
        let len_end = self.at.checked_add(4).filter(|&e| e <= self.b.len());
        let len_end = len_end.ok_or_else(|| bad(self.at))?;
        let len = u32::from_le_bytes(self.b[self.at..len_end].try_into().unwrap()) as usize;
        let end = len_end
            .checked_add(len)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| bad(len_end))?;
        self.at = end;
        Ok(len_end..end)
    }
}

/// Verified data blocks shared across every table in a store, bounded by
/// total bytes and evicted least recently used first.
pub(crate) struct BlockCache {
    capacity: usize,
    inner: Mutex<CacheInner>,
}

#[derive(Default)]
struct CacheInner {
    used: usize,
    tick: u64,
    /// (table number, block offset) -> (block, last use)
    blocks: HashMap<(u64, u64), (Arc<[u8]>, u64)>,
    /// last use -> block key, oldest first
    lru: BTreeMap<u64, (u64, u64)>,
    /// Blocks read from disk, for tests.
    loads: u64,
}

impl BlockCache {
    pub fn new(capacity: usize) -> Self {
        BlockCache {
            capacity,
            inner: Mutex::default(),
        }
    }

    fn get_or_load<E>(
        &self,
        key: (u64, u64),
        load: impl FnOnce() -> Result<Arc<[u8]>, E>,
    ) -> Result<Arc<[u8]>, E> {
        {
            let mut inner = self.inner.lock().unwrap();
            let inner = &mut *inner;
            inner.tick += 1;
            if let Some((block, used)) = inner.blocks.get_mut(&key) {
                inner.lru.remove(used);
                *used = inner.tick;
                inner.lru.insert(inner.tick, key);
                return Ok(block.clone());
            }
        }

        // read outside the lock; a racing reader may load the same block,
        // and the later insert simply replaces the earlier one
        let block = load()?;

        let mut inner = self.inner.lock().unwrap();
        let inner = &mut *inner;
        inner.loads += 1;
        inner.tick += 1;
        if let Some((old, used)) = inner.blocks.remove(&key) {
            inner.lru.remove(&used);
            inner.used -= old.len();
        }
        inner.used += block.len();
        inner.blocks.insert(key, (block.clone(), inner.tick));
        inner.lru.insert(inner.tick, key);
        while inner.used > self.capacity {
            let Some((_, oldest)) = inner.lru.pop_first() else {
                break;
            };
            let (evicted, _) = inner.blocks.remove(&oldest).expect("lru tracks blocks");
            inner.used -= evicted.len();
        }
        Ok(block)
    }

    /// Bytes of blocks currently cached.
    pub fn used(&self) -> usize {
        self.inner.lock().unwrap().used
    }

    /// (bytes cached, blocks read from disk so far)
    #[cfg(test)]
    pub fn stats(&self) -> (usize, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.used, inner.loads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("fold3-table-{name}-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn key(i: u32) -> [u8; 4] {
        i.to_be_bytes()
    }

    /// Keyspace 1 holds even keys 0..2000 over many blocks; keyspace 5 one key.
    fn build(path: &Path) {
        let mut b = TableBuilder::create(path, DEFAULT_BLOCK_SIZE).unwrap();
        b.begin_keyspace(1).unwrap();
        for i in (0..2000).step_by(2) {
            b.push(&key(i), &[1, i as u8]).unwrap();
        }
        b.begin_keyspace(5).unwrap();
        b.push(&key(7), &[0]).unwrap();
        b.finish(42).unwrap();
    }

    #[test]
    fn finds_every_key_and_nothing_else() {
        let path = tmp("find");
        build(&path);
        let t = Table::open(&path, 1).unwrap();
        let cache = BlockCache::new(1 << 20);
        assert_eq!(t.epoch(), 42);
        assert!(t.index[&1].len() > 1, "spans several blocks");

        let get = |ks, k| {
            t.get(ks, &key(k), &cache)
                .unwrap()
                .map(|(b, r)| b[r].to_vec())
        };
        for i in 0..2000 {
            let want = (i % 2 == 0).then(|| vec![1, i as u8]);
            assert_eq!(get(1, i), want, "key {i}");
        }
        assert_eq!(get(1, 5000), None, "past the end");
        assert_eq!(get(5, 7), Some(vec![0]), "tombstone is a hit");
        assert_eq!(get(5, 6), None, "before the first key");
        assert_eq!(get(9, 0), None, "unknown keyspace");
        assert_eq!(t.scan(1, &cache).len(), 1000);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cache_serves_repeat_reads_and_stays_in_budget() {
        let path = tmp("cache");
        build(&path);
        let t = Table::open(&path, 1).unwrap();

        let cache = BlockCache::new(1 << 20);
        t.get(1, &key(0), &cache).unwrap();
        t.get(1, &key(2), &cache).unwrap();
        assert_eq!(cache.stats().1, 1, "same block: one disk read");

        let small = BlockCache::new(DEFAULT_BLOCK_SIZE * 2);
        for i in (0..2000).step_by(2) {
            assert!(t.get(1, &key(i), &small).unwrap().is_some());
            assert!(small.stats().0 <= DEFAULT_BLOCK_SIZE * 2);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn damage_is_caught_at_open_or_at_read() {
        let path = tmp("damage");
        build(&path);
        let good = std::fs::read(&path).unwrap();

        // a flipped data byte fails only the read of that block
        let mut bytes = good.clone();
        bytes[MAGIC.len() + 10] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let t = Table::open(&path, 1).unwrap();
        let cache = BlockCache::new(1 << 20);
        assert!(matches!(
            t.get(1, &key(0), &cache),
            Err(TableError::Corrupt { .. })
        ));
        assert!(t.get(1, &key(1998), &cache).unwrap().is_some());

        // a flipped index byte fails the open
        let mut bytes = good.clone();
        let at = bytes.len() - FOOTER - 3;
        bytes[at] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            Table::open(&path, 1),
            Err(TableError::Corrupt { .. })
        ));
        std::fs::remove_file(&path).unwrap();
    }
}
