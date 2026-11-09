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
//! the encoded `Option<V>`: a tombstone is `None`, the single byte `0`. All
//! integers are little endian.
//!
//! Opening a table reads only its footer and index. Data blocks are read on
//! demand, through a [`BlockCache`] shared by every table in the store for
//! reads, or straight from disk for compaction. Their checksums are
//! verified whenever they are read from disk.

use std::{
    collections::{BTreeMap, HashMap},
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
};

use crate::error::{Error, IoContext, Result};

const MAGIC: &[u8; 8] = b"fold3sst";
const FOOTER: usize = 8 + 8 + 8 + 4 + 8;
/// The default block size: a block is cut once its entries reach this.
pub(crate) const DEFAULT_BLOCK_SIZE: usize = 4096;
/// The encoded value of a tombstone, `None::<V>`.
pub(crate) const TOMBSTONE: &[u8] = &[0];

struct BlockHandle {
    offset: u64,
    /// Including the trailing crc.
    len: u32,
    first_key: Box<[u8]>,
}

/// Writes a table. Keyspaces must be begun in ascending id order and keys
/// pushed in ascending encoded order within each.
pub(crate) struct TableBuilder {
    path: PathBuf,
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
    pub fn create(path: &Path, block_size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .ctx("creating table", path)?;
        let mut out = BufWriter::new(file);
        out.write_all(MAGIC).ctx("writing table", path)?;
        Ok(TableBuilder {
            path: path.to_path_buf(),
            out,
            block_size,
            offset: MAGIC.len() as u64,
            index: Vec::new(),
            block: Vec::with_capacity(block_size * 2),
            block_first: None,
        })
    }

    pub fn begin_keyspace(&mut self, ks: u64) -> Result<()> {
        self.cut_block()?;
        if let Some((prev, _)) = self.index.last() {
            assert!(*prev < ks, "keyspaces out of order");
        }
        self.index.push((ks, Vec::new()));
        Ok(())
    }

    pub fn push(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
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

    fn cut_block(&mut self) -> Result<()> {
        let Some(first_key) = self.block_first.take() else {
            return Ok(());
        };
        self.block
            .extend(crc32fast::hash(&self.block).to_le_bytes());
        self.out
            .write_all(&self.block)
            .ctx("writing table", &self.path)?;
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
    pub fn finish(mut self, epoch: u64) -> Result<()> {
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
        let mut footer = Vec::with_capacity(FOOTER);
        footer.extend(self.offset.to_le_bytes());
        footer.extend((index.len() as u64).to_le_bytes());
        footer.extend(epoch.to_le_bytes());
        footer.extend(crc32fast::hash(&index).to_le_bytes());
        footer.extend(MAGIC);

        let path = self.path;
        self.out.write_all(&index).ctx("writing table", &path)?;
        self.out.write_all(&footer).ctx("writing table", &path)?;
        let file = self
            .out
            .into_inner()
            .map_err(|e| e.into_error())
            .ctx("writing table", &path)?;
        file.sync_all().ctx("syncing table", &path)
    }
}

/// An open table: its index in memory, its blocks on disk.
pub(crate) struct Table {
    /// Unique within the store; names this table's blocks in the cache.
    number: u64,
    path: PathBuf,
    file: File,
    /// Bytes on disk.
    size: u64,
    epoch: u64,
    index: HashMap<u64, Vec<BlockHandle>>,
}

/// A little-endian cursor over the index bytes.
struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(n).filter(|&end| end <= self.b.len())?;
        let s = &self.b[self.at..end];
        self.at = end;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// One entry of a verified block: key and value ranges, and where the next
/// entry starts. `None` at the end of the block.
fn entry_at(block: &[u8], at: usize) -> Result<Option<(Range<usize>, Range<usize>, usize)>, usize> {
    if at == block.len() {
        return Ok(None);
    }
    let field = |at: usize| -> Result<(Range<usize>, usize), usize> {
        let len_end = at.checked_add(4).filter(|&e| e <= block.len()).ok_or(at)?;
        let len = u32::from_le_bytes(block[at..len_end].try_into().unwrap()) as usize;
        let end = len_end
            .checked_add(len)
            .filter(|&e| e <= block.len())
            .ok_or(len_end)?;
        Ok((len_end..end, end))
    };
    let (key, at) = field(at)?;
    let (value, next) = field(at)?;
    Ok(Some((key, value, next)))
}

impl Table {
    /// Open a table, reading and checking its footer and index.
    pub fn open(path: &Path, number: u64) -> Result<Table> {
        let corrupt = |offset: u64, why: &str| Error::corrupt(path, offset, why);
        let file = File::open(path).ctx("opening table", path)?;
        let size = file.metadata().ctx("opening table", path)?.len();
        if size < (MAGIC.len() + FOOTER) as u64 {
            return Err(corrupt(0, "too short to be a table"));
        }
        let mut head = [0u8; 8];
        file.read_exact_at(&mut head, 0)
            .ctx("reading table", path)?;
        if &head != MAGIC {
            return Err(corrupt(0, "not a table"));
        }

        let footer_at = size - FOOTER as u64;
        let mut footer = [0u8; FOOTER];
        file.read_exact_at(&mut footer, footer_at)
            .ctx("reading table", path)?;
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
        file.read_exact_at(&mut raw, index_at)
            .ctx("reading table", path)?;
        if crc32fast::hash(&raw) != crc {
            return Err(corrupt(index_at, "index checksum mismatch"));
        }
        let truncated = || corrupt(index_at, "index truncated");
        let mut r = Reader { b: &raw, at: 0 };
        let mut index = HashMap::new();
        for _ in 0..r.u32().ok_or_else(truncated)? {
            let ks = r.u64().ok_or_else(truncated)?;
            let n = r.u32().ok_or_else(truncated)?;
            let mut blocks: Vec<BlockHandle> = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let offset = r.u64().ok_or_else(truncated)?;
                let len = r.u32().ok_or_else(truncated)?;
                let key_len = r.u32().ok_or_else(truncated)? as usize;
                let first_key: Box<[u8]> = r.take(key_len).ok_or_else(truncated)?.into();
                if offset
                    .checked_add(len as u64)
                    .is_none_or(|end| end > index_at)
                    || len < 4
                {
                    return Err(corrupt(index_at, "block outside the data region"));
                }
                if blocks.last().is_some_and(|b| b.first_key >= first_key) {
                    return Err(corrupt(index_at, "blocks out of order"));
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
            size,
            epoch,
            index,
        })
    }

    pub fn number(&self) -> u64 {
        self.number
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes on disk.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Estimated bytes the in-memory index holds.
    pub fn index_bytes(&self) -> usize {
        self.index
            .values()
            .flatten()
            .map(|b| size_of::<BlockHandle>() + b.first_key.len())
            .sum()
    }

    /// Data blocks in keyspace `ks`.
    #[cfg(test)]
    pub fn blocks(&self, ks: u64) -> usize {
        self.index.get(&ks).map_or(0, Vec::len)
    }

    pub fn keyspaces(&self) -> impl Iterator<Item = u64> + '_ {
        self.index.keys().copied()
    }

    fn corrupt(&self, offset: u64, why: &str) -> Error {
        Error::corrupt(&self.path, offset, why)
    }

    /// The encoded value stored for `key` in keyspace `ks`, as a block and
    /// the value's byte range within it. `None` if this table does not
    /// mention the key; a tombstone is a hit whose value is [`TOMBSTONE`].
    pub fn get(
        &self,
        ks: u64,
        key: &[u8],
        cache: &BlockCache,
    ) -> Result<Option<(Arc<[u8]>, Range<usize>)>> {
        let Some(blocks) = self.index.get(&ks) else {
            return Ok(None);
        };
        // the last block starting at or before the key is the only one
        // that can hold it
        let i = blocks.partition_point(|b| &*b.first_key <= key);
        let Some(handle) = i.checked_sub(1).map(|i| &blocks[i]) else {
            return Ok(None);
        };
        let block = self.block(handle, Some(cache))?;
        let mut at = 0;
        while let Some((k, v, next)) =
            entry_at(&block, at).map_err(|o| self.bad_entry(handle, o))?
        {
            match block[k].cmp(key) {
                std::cmp::Ordering::Less => at = next,
                std::cmp::Ordering::Equal => return Ok(Some((block.clone(), v))),
                std::cmp::Ordering::Greater => break,
            }
        }
        Ok(None)
    }

    fn bad_entry(&self, handle: &BlockHandle, at: usize) -> Error {
        self.corrupt(handle.offset + at as u64, "entry runs past its block")
    }

    /// A block's entries, from the cache if one is given, else from disk.
    fn block(&self, handle: &BlockHandle, cache: Option<&BlockCache>) -> Result<Arc<[u8]>> {
        let read = || {
            let mut buf = vec![0u8; handle.len as usize];
            self.file
                .read_exact_at(&mut buf, handle.offset)
                .ctx("reading table", &self.path)?;
            let (entries, crc) = buf.split_at(buf.len() - 4);
            if crc32fast::hash(entries) != u32::from_le_bytes(crc.try_into().unwrap()) {
                return Err(self.corrupt(handle.offset, "block checksum mismatch"));
            }
            buf.truncate(buf.len() - 4);
            Ok(buf.into())
        };
        match cache {
            Some(cache) => cache.get_or_load((self.number, handle.offset), read),
            None => read(),
        }
    }

    /// A cursor over keyspace `ks`, before its first entry. With no cache,
    /// blocks are read straight from disk and not kept.
    pub fn cursor<'a>(&'a self, ks: u64, cache: Option<&'a BlockCache>) -> TableCursor<'a> {
        TableCursor {
            table: self,
            blocks: self.index.get(&ks).map_or(&[], |b| &b[..]),
            cache,
            bi: 0,
            block: None,
            cur: None,
            started: false,
        }
    }
}

/// Walks one keyspace of a table in key order, a block at a time. It only
/// moves forward.
pub(crate) struct TableCursor<'a> {
    table: &'a Table,
    blocks: &'a [BlockHandle],
    cache: Option<&'a BlockCache>,
    /// The loaded block's index; `blocks.len()` once exhausted.
    bi: usize,
    block: Option<Arc<[u8]>>,
    /// The current entry: key range, value range, next entry's offset.
    cur: Option<(Range<usize>, Range<usize>, usize)>,
    started: bool,
}

impl<'a> TableCursor<'a> {
    pub fn path(&self) -> &'a Path {
        &self.table.path
    }

    /// The current entry's encoded key and value.
    pub fn current(&self) -> Option<(&[u8], &[u8])> {
        let (k, v, _) = self.cur.as_ref()?;
        let block = self.block.as_deref()?;
        Some((&block[k.clone()], &block[v.clone()]))
    }

    /// Move to the first entry, if not yet started.
    pub fn start(&mut self) -> Result<()> {
        if !self.started {
            self.started = true;
            self.load(0)?;
        }
        Ok(())
    }

    /// Move to the next entry.
    pub fn advance(&mut self) -> Result<()> {
        self.start()?;
        match self.cur.take() {
            Some((_, _, next)) => self.read_from(next),
            None => Ok(()),
        }
    }

    /// Move forward to the first entry whose key is at or past `probe`.
    /// Never moves backward.
    pub fn seek(&mut self, probe: &[u8]) -> Result<()> {
        self.start()?;
        if self.current().is_none_or(|(k, _)| k >= probe) {
            return Ok(());
        }
        // the last block starting at or before the probe; later blocks all
        // start past it
        let target = self
            .blocks
            .partition_point(|b| &*b.first_key <= probe)
            .saturating_sub(1)
            .max(self.bi);
        if target != self.bi {
            self.load(target)?;
        }
        while self.current().is_some_and(|(k, _)| k < probe) {
            self.advance()?;
        }
        Ok(())
    }

    /// Load block `bi` and position at its first entry, skipping to later
    /// blocks if it is empty.
    fn load(&mut self, bi: usize) -> Result<()> {
        self.bi = bi;
        self.cur = None;
        self.block = None;
        if let Some(handle) = self.blocks.get(bi) {
            self.block = Some(self.table.block(handle, self.cache)?);
            self.read_from(0)?;
        }
        Ok(())
    }

    /// Read the entry at `at` in the loaded block, or move to the next block.
    fn read_from(&mut self, at: usize) -> Result<()> {
        let Some(block) = self.block.as_deref() else {
            return Ok(());
        };
        match entry_at(block, at) {
            Ok(Some(entry)) => {
                self.cur = Some(entry);
                Ok(())
            }
            Ok(None) => self.load(self.bi + 1),
            Err(o) => Err(self.table.bad_entry(&self.blocks[self.bi], o)),
        }
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

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheInner> {
        // the cache is consistent between statements; a panic elsewhere
        // while holding it leaves nothing half-done
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn get_or_load(
        &self,
        key: (u64, u64),
        load: impl FnOnce() -> Result<Arc<[u8]>>,
    ) -> Result<Arc<[u8]>> {
        {
            let mut inner = self.lock();
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

        let mut inner = self.lock();
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

    /// Drop every cached block of table `number`, which no longer exists.
    pub fn forget(&self, number: u64) {
        let mut inner = self.lock();
        let inner = &mut *inner;
        let dead: Vec<_> = inner
            .blocks
            .iter()
            .filter(|((n, _), _)| *n == number)
            .map(|(k, (b, used))| (*k, b.len(), *used))
            .collect();
        for (key, len, used) in dead {
            inner.blocks.remove(&key);
            inner.lru.remove(&used);
            inner.used -= len;
        }
    }

    /// Bytes of blocks currently cached.
    pub fn used(&self) -> usize {
        self.lock().used
    }

    /// (bytes cached, blocks read from disk so far)
    #[cfg(test)]
    pub fn stats(&self) -> (usize, u64) {
        let inner = self.lock();
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
        b.push(&key(7), TOMBSTONE).unwrap();
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
        assert_eq!(get(5, 7), Some(TOMBSTONE.to_vec()), "tombstone is a hit");
        assert_eq!(get(5, 6), None, "before the first key");
        assert_eq!(get(9, 0), None, "unknown keyspace");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cursor_walks_and_seeks_across_blocks() {
        let path = tmp("cursor");
        build(&path);
        let t = Table::open(&path, 1).unwrap();
        let cache = BlockCache::new(1 << 20);

        let mut c = t.cursor(1, Some(&cache));
        c.start().unwrap();
        let mut n = 0;
        while let Some((k, _)) = c.current() {
            assert_eq!(k, key(n * 2));
            n += 1;
            c.advance().unwrap();
        }
        assert_eq!(n, 1000);

        let mut c = t.cursor(1, None);
        c.seek(&key(501)).unwrap();
        assert_eq!(c.current().unwrap().0, key(502), "next key at or past");
        c.seek(&key(1700)).unwrap();
        assert_eq!(c.current().unwrap().0, key(1700), "skips blocks");
        c.seek(&key(10)).unwrap();
        assert_eq!(c.current().unwrap().0, key(1700), "never moves back");
        c.seek(&key(9999)).unwrap();
        assert!(c.current().is_none());

        let mut empty = t.cursor(9, Some(&cache));
        empty.seek(&key(0)).unwrap();
        assert!(empty.current().is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cache_serves_repeat_reads_stays_in_budget_and_forgets() {
        let path = tmp("cache");
        build(&path);
        let t = Table::open(&path, 1).unwrap();

        let cache = BlockCache::new(1 << 20);
        t.get(1, &key(0), &cache).unwrap();
        t.get(1, &key(2), &cache).unwrap();
        assert_eq!(cache.stats().1, 1, "same block: one disk read");
        cache.forget(1);
        assert_eq!(cache.used(), 0);

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

        // a flipped data byte fails only the reads of that block
        let mut bytes = good.clone();
        bytes[MAGIC.len() + 10] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let t = Table::open(&path, 1).unwrap();
        let cache = BlockCache::new(1 << 20);
        assert!(matches!(
            t.get(1, &key(0), &cache),
            Err(Error::Corrupt { .. })
        ));
        let mut c = t.cursor(1, None);
        assert!(matches!(c.start(), Err(Error::Corrupt { .. })));
        assert!(t.get(1, &key(1998), &cache).unwrap().is_some());

        // a flipped index byte fails the open
        let mut bytes = good.clone();
        let at = bytes.len() - FOOTER - 3;
        bytes[at] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(Table::open(&path, 1), Err(Error::Corrupt { .. })));
        std::fs::remove_file(&path).unwrap();
    }
}
