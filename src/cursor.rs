//! Merged, sorted iteration over one keyspace.
//!
//! A keyspace's data is spread over a write tx's overlay, the committed
//! layers in memory, and the checkpoint tables on disk. A [`Cursor`] merges
//! them in key order. Where several sources hold the same key, the newest
//! wins, and a tombstone there hides the key entirely.
//!
//! Sources are ordered newest first, so priority is position: the overlay,
//! then layers newest to oldest, then tables newest to oldest. Table
//! entries are compared as encoded bytes when seeking and decoded only when
//! they become a source's head.

use std::{
    borrow::Cow,
    collections::{BTreeMap, btree_map},
    ops::Bound,
};

use crate::{
    Change,
    codec::{Codec, OrderedCodec, decode_all, decode_all_ordered},
    error::{Error, Result},
    merge::Merge,
    store::Resident,
    table::TableCursor,
    tx::TxState,
};

type Entry<'a, K, V> = (Cow<'a, K>, Option<Cow<'a, V>>);

pub(crate) enum Source<'a, K, M: Merge> {
    /// A write tx's own changes; a Delete is a tombstone.
    Overlay {
        map: &'a BTreeMap<K, Change<M>>,
        it: btree_map::Range<'a, K, Change<M>>,
        head: Option<(&'a K, &'a Change<M>)>,
    },
    /// A write tx's changes as of the start of a graph run: `map`, with
    /// each key in `prior` as it was before the run. A key the overlay
    /// did not hold then is skipped, so the snapshot below decides it.
    OverlayAsOf {
        map: &'a BTreeMap<K, Change<M>>,
        prior: &'a BTreeMap<K, Option<Change<M>>>,
        it: std::iter::Peekable<btree_map::Range<'a, K, Change<M>>>,
        pit: std::iter::Peekable<btree_map::Range<'a, K, Option<Change<M>>>>,
        head: Option<(&'a K, &'a Change<M>)>,
    },
    /// A committed layer in memory.
    Layer {
        map: &'a Resident<K, M::Value>,
        it: btree_map::Range<'a, K, Option<M::Value>>,
        head: Option<(&'a K, &'a Option<M::Value>)>,
    },
    /// A checkpoint table, started lazily since reading it can fail.
    Table {
        cursor: TableCursor<'a>,
        ks: u64,
        head: Option<(K, Option<M::Value>)>,
    },
}

impl<'a, K, M> Source<'a, K, M>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
{
    pub fn overlay(map: &'a BTreeMap<K, Change<M>>, lower: Bound<&K>) -> Self {
        let mut it = map.range::<K, _>((lower, Bound::Unbounded));
        let head = it.next();
        Source::Overlay { map, it, head }
    }

    pub fn overlay_as_of(
        map: &'a BTreeMap<K, Change<M>>,
        prior: &'a BTreeMap<K, Option<Change<M>>>,
        lower: Bound<&K>,
    ) -> Self {
        let mut s = Source::OverlayAsOf {
            map,
            prior,
            it: map.range::<K, _>((lower, Bound::Unbounded)).peekable(),
            pit: prior.range::<K, _>((lower, Bound::Unbounded)).peekable(),
            head: None,
        };
        s.next_as_of();
        s
    }

    /// Move an [`OverlayAsOf`](Source::OverlayAsOf) to its next entry.
    fn next_as_of(&mut self) {
        let Source::OverlayAsOf { it, pit, head, .. } = self else {
            unreachable!("only for overlays as of a run")
        };
        *head = loop {
            match (it.peek(), pit.peek()) {
                (None, None) => break None,
                (Some(_), None) => break it.next(),
                (Some((k, _)), Some((p, _))) if k < p => break it.next(),
                (a, Some((p, prior))) => {
                    let (p, prior) = (*p, *prior);
                    if a.is_some_and(|(k, _)| *k == p) {
                        it.next();
                    }
                    pit.next();
                    match prior {
                        Some(c) => break Some((p, c)),
                        None => continue,
                    }
                }
            }
        };
    }

    pub fn layer(map: &'a Resident<K, M::Value>, lower: Bound<&K>) -> Self {
        let mut it = map.range::<K, _>((lower, Bound::Unbounded));
        let head = it.next();
        Source::Layer { map, it, head }
    }

    pub fn table(cursor: TableCursor<'a>, ks: u64) -> Self {
        Source::Table {
            cursor,
            ks,
            head: None,
        }
    }

    fn head_key(&self) -> Option<&K> {
        match self {
            Source::Overlay { head, .. } | Source::OverlayAsOf { head, .. } => head.map(|(k, _)| k),
            Source::Layer { head, .. } => head.map(|(k, _)| k),
            Source::Table { head, .. } => head.as_ref().map(|(k, _)| k),
        }
    }

    /// The head's value without consuming it; `Some(None)` is a tombstone.
    fn peek_value(&self) -> Option<Option<Cow<'a, M::Value>>> {
        match self {
            Source::Overlay { head, .. } | Source::OverlayAsOf { head, .. } => {
                head.map(|(_, c)| c.new_value().map(Cow::Borrowed))
            }
            Source::Layer { head, .. } => head.map(|(_, v)| v.as_ref().map(Cow::Borrowed)),
            Source::Table { head, .. } => head
                .as_ref()
                .map(|(_, v)| v.as_ref().map(|v| Cow::Owned(v.clone()))),
        }
    }

    /// Consume the head; the source must then [`advance`](Self::advance).
    fn take_head(&mut self) -> Option<Entry<'a, K, M::Value>> {
        match self {
            Source::Overlay { head, .. } | Source::OverlayAsOf { head, .. } => head
                .take()
                .map(|(k, c)| (Cow::Borrowed(k), c.new_value().map(Cow::Borrowed))),
            Source::Layer { head, .. } => head
                .take()
                .map(|(k, v)| (Cow::Borrowed(k), v.as_ref().map(Cow::Borrowed))),
            Source::Table { head, .. } => {
                head.take().map(|(k, v)| (Cow::Owned(k), v.map(Cow::Owned)))
            }
        }
    }

    /// Position a table source at the lower bound. Memory sources start
    /// positioned.
    fn start(&mut self, lower: &Bound<Vec<u8>>) -> Result<()> {
        if let Source::Table { cursor, .. } = self {
            match lower {
                Bound::Unbounded => cursor.start()?,
                Bound::Included(probe) => cursor.seek(probe)?,
                Bound::Excluded(probe) => {
                    cursor.seek(probe)?;
                    if cursor.current().is_some_and(|(k, _)| k == &probe[..]) {
                        cursor.advance()?;
                    }
                }
            }
            self.decode_head()?;
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<()> {
        match self {
            Source::Overlay { it, head, .. } => *head = it.next(),
            Source::OverlayAsOf { .. } => self.next_as_of(),
            Source::Layer { it, head, .. } => *head = it.next(),
            Source::Table { cursor, .. } => {
                cursor.advance()?;
                self.decode_head()?;
            }
        }
        Ok(())
    }

    /// Move forward to the first key at or past `key` (`probe` is its
    /// encoding). Never moves backward.
    fn seek(&mut self, key: &K, probe: &[u8]) -> Result<()> {
        if self.head_key().is_none_or(|k| k >= key) {
            return Ok(());
        }
        match self {
            Source::Overlay { map, it, head } => {
                *it = map.range::<K, _>((Bound::Included(key), Bound::Unbounded));
                *head = it.next();
            }
            Source::OverlayAsOf {
                map,
                prior,
                it,
                pit,
                ..
            } => {
                *it = map
                    .range::<K, _>((Bound::Included(key), Bound::Unbounded))
                    .peekable();
                *pit = prior
                    .range::<K, _>((Bound::Included(key), Bound::Unbounded))
                    .peekable();
                self.next_as_of();
            }
            Source::Layer { map, it, head } => {
                *it = map.range::<K, _>((Bound::Included(key), Bound::Unbounded));
                *head = it.next();
            }
            Source::Table { cursor, .. } => {
                cursor.seek(probe)?;
                self.decode_head()?;
            }
        }
        Ok(())
    }

    fn decode_head(&mut self) -> Result<()> {
        let Source::Table { cursor, ks, head } = self else {
            return Ok(());
        };
        *head = match cursor.current() {
            None => None,
            Some((k, v)) => {
                let bad = |e| Error::corrupt(cursor.path(), 0, format!("keyspace {ks}: {e}"));
                Some((
                    decode_all_ordered::<K>(k).map_err(bad)?,
                    decode_all::<Option<M::Value>>(v).map_err(bad)?,
                ))
            }
        };
        Ok(())
    }
}

/// Merged iteration over one keyspace, in key order, skipping tombstones.
/// Yields `(key, value)`; values from memory are borrowed, values from disk
/// are owned.
///
/// Iterating a write tx's keyspace sees the tx's own writes. An error from
/// a tx's cursor fails the tx, as any read error does.
pub struct Cursor<'a, K, M: Merge> {
    sources: Vec<Source<'a, K, M>>,
    lower: Bound<Vec<u8>>,
    upper: Bound<K>,
    started: bool,
    done: bool,
    probe: Vec<u8>,
    tx: Option<&'a TxState>,
}

impl<'a, K, M> Cursor<'a, K, M>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
{
    /// `sources` newest first, memory sources already at `lower`.
    pub(crate) fn new(
        sources: Vec<Source<'a, K, M>>,
        lower: Bound<&K>,
        upper: Bound<K>,
        tx: Option<&'a TxState>,
    ) -> Self {
        let encode = |k: &K| {
            let mut b = Vec::with_capacity(k.encode_ordered_size());
            k.encode_ordered(&mut b);
            b
        };
        let lower = match lower {
            Bound::Included(k) => Bound::Included(encode(k)),
            Bound::Excluded(k) => Bound::Excluded(encode(k)),
            Bound::Unbounded => Bound::Unbounded,
        };
        Cursor {
            sources,
            lower,
            upper,
            started: false,
            done: false,
            probe: Vec::new(),
            tx,
        }
    }

    /// Put `source` above every other, before the cursor starts.
    pub(crate) fn push_newest(&mut self, source: Source<'a, K, M>) {
        debug_assert!(!self.started, "sources are fixed once started");
        self.sources.insert(0, source);
    }

    fn start(&mut self) -> Result<()> {
        if let Some(tx) = self.tx {
            tx.check()?;
        }
        if !self.started {
            self.started = true;
            for s in &mut self.sources {
                s.start(&self.lower)?;
            }
        }
        Ok(())
    }

    /// Record a failure in the tx this cursor reads for, if any.
    fn fail(&mut self, e: Error) -> Error {
        self.done = true;
        if let Some(tx) = self.tx {
            tx.fail(&e);
        }
        e
    }

    /// The next entry in key order, tombstones included.
    fn next_entry(&mut self) -> Result<Option<Entry<'a, K, M::Value>>> {
        self.start()?;
        let mut win: Option<usize> = None;
        for (i, s) in self.sources.iter().enumerate() {
            if let Some(k) = s.head_key()
                && win.is_none_or(|w| k < self.sources[w].head_key().unwrap())
            {
                win = Some(i);
            }
        }
        let Some(w) = win else {
            return Ok(None);
        };
        let past_end = {
            let k = self.sources[w].head_key().unwrap();
            match &self.upper {
                Bound::Included(u) => k > u,
                Bound::Excluded(u) => k >= u,
                Bound::Unbounded => false,
            }
        };
        if past_end {
            self.done = true;
            return Ok(None);
        }
        // older sources holding the same key are shadowed
        for i in w + 1..self.sources.len() {
            let same = self.sources[i].head_key().is_some()
                && self.sources[i].head_key() == self.sources[w].head_key();
            if same {
                self.sources[i].advance()?;
            }
        }
        let entry = self.sources[w].take_head();
        self.sources[w].advance()?;
        Ok(entry)
    }

    /// The value of `key`, moving the cursor forward to it. Lookups must
    /// come in ascending key order; that is what makes a batch of them one
    /// sequential pass.
    pub fn seek_get(&mut self, key: &K) -> Result<Option<Cow<'a, M::Value>>> {
        let res = self.seek_get_inner(key);
        res.map_err(|e| self.fail(e))
    }

    fn seek_get_inner(&mut self, key: &K) -> Result<Option<Cow<'a, M::Value>>> {
        self.start()?;
        self.probe.clear();
        key.encode_ordered(&mut self.probe);
        for s in &mut self.sources {
            s.seek(key, &self.probe)?;
        }
        Ok(self
            .sources
            .iter()
            .find(|s| s.head_key() == Some(key))
            .and_then(|s| s.peek_value())
            .flatten())
    }
}

impl<'a, K, M> Iterator for Cursor<'a, K, M>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
{
    type Item = Result<(Cow<'a, K>, Cow<'a, M::Value>)>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.done {
            match self.next_entry() {
                Ok(Some((k, Some(v)))) => return Some(Ok((k, v))),
                Ok(Some((_, None))) => continue,
                Ok(None) => self.done = true,
                Err(e) => return Some(Err(self.fail(e))),
            }
        }
        None
    }
}
