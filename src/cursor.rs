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

/// A map range walked forward or backward.
pub(crate) struct Walk<'a, K, V> {
    range: btree_map::Range<'a, K, V>,
    rev: bool,
}

impl<'a, K, V> Iterator for Walk<'a, K, V> {
    type Item = (&'a K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.rev {
            self.range.next_back()
        } else {
            self.range.next()
        }
    }
}

type Pairs<'a, K, V> = std::iter::Peekable<Walk<'a, K, V>>;

/// `map`'s entries within the bounds, walked forward or, with `rev`,
/// backward.
fn pairs<'a, K: Ord, V>(
    map: &'a BTreeMap<K, V>,
    lower: Bound<&K>,
    upper: Bound<&K>,
    rev: bool,
) -> Pairs<'a, K, V> {
    Walk {
        range: map.range::<K, _>((lower, upper)),
        rev,
    }
    .peekable()
}

/// Entries a memory source steps through before a seek searches its map
/// again instead: dense batches of seeks mostly land a few entries on.
const SEEK_STEPS: usize = 8;

/// Whether no key can lie within the bounds.
pub(crate) fn empty_bounds<K: Ord>(lower: Bound<&K>, upper: Bound<&K>) -> bool {
    match (lower, upper) {
        (Bound::Included(l), Bound::Included(u)) => l > u,
        (Bound::Included(l) | Bound::Excluded(l), Bound::Excluded(u))
        | (Bound::Excluded(l), Bound::Included(u)) => l >= u,
        _ => false,
    }
}

pub(crate) enum Source<'a, K, M: Merge> {
    /// One run of a write tx's own changes, sorted; a Delete is a
    /// tombstone. `entries[lo..hi]` is what remains within the bounds.
    Overlay {
        entries: &'a [(K, Change<M>)],
        lo: usize,
        hi: usize,
        rev: bool,
    },
    /// A committed layer in memory.
    Layer {
        map: &'a Resident<K, M::Value>,
        it: Pairs<'a, K, Option<M::Value>>,
        head: Option<(&'a K, &'a Option<M::Value>)>,
        rev: bool,
    },
    /// A checkpoint table, started lazily since reading it can fail.
    Table {
        cursor: TableCursor<'a>,
        ks: u64,
        head: Option<(K, Option<M::Value>)>,
        rev: bool,
    },
}

impl<'a, K, M> Source<'a, K, M>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
{
    pub fn overlay(
        entries: &'a [(K, Change<M>)],
        lower: Bound<&K>,
        upper: Bound<&K>,
        rev: bool,
    ) -> Self {
        let lo = match lower {
            Bound::Unbounded => 0,
            Bound::Included(k) => entries.partition_point(|(e, _)| e < k),
            Bound::Excluded(k) => entries.partition_point(|(e, _)| e <= k),
        };
        let hi = match upper {
            Bound::Unbounded => entries.len(),
            Bound::Included(k) => entries.partition_point(|(e, _)| e <= k),
            Bound::Excluded(k) => entries.partition_point(|(e, _)| e < k),
        };
        Source::Overlay {
            entries,
            lo,
            hi: hi.max(lo),
            rev,
        }
    }

    pub fn layer(
        map: &'a Resident<K, M::Value>,
        lower: Bound<&K>,
        upper: Bound<&K>,
        rev: bool,
    ) -> Self {
        let mut it = pairs(map, lower, upper, rev);
        let head = it.next();
        Source::Layer { map, it, head, rev }
    }

    pub fn table(cursor: TableCursor<'a>, ks: u64, rev: bool) -> Self {
        Source::Table {
            cursor,
            ks,
            head: None,
            rev,
        }
    }

    fn head_key(&self) -> Option<&K> {
        match self {
            Source::Overlay {
                entries,
                lo,
                hi,
                rev,
            } => (lo < hi).then(|| &entries[if *rev { hi - 1 } else { *lo }].0),
            Source::Layer { head, .. } => head.map(|(k, _)| k),
            Source::Table { head, .. } => head.as_ref().map(|(k, _)| k),
        }
    }

    /// The head's value without consuming it; `Some(None)` is a tombstone.
    fn peek_value(&self) -> Option<Option<Cow<'a, M::Value>>> {
        match self {
            Source::Overlay {
                entries,
                lo,
                hi,
                rev,
            } => (lo < hi).then(|| {
                entries[if *rev { hi - 1 } else { *lo }]
                    .1
                    .new_value()
                    .map(Cow::Borrowed)
            }),
            Source::Layer { head, .. } => head.map(|(_, v)| v.as_ref().map(Cow::Borrowed)),
            Source::Table { head, .. } => head
                .as_ref()
                .map(|(_, v)| v.as_ref().map(|v| Cow::Owned(v.clone()))),
        }
    }

    /// Consume the head; the source must then [`advance`](Self::advance).
    fn take_head(&mut self) -> Option<Entry<'a, K, M::Value>> {
        match self {
            Source::Overlay {
                entries,
                lo,
                hi,
                rev,
            } => (lo < hi).then(|| {
                let (k, c) = &entries[if *rev { *hi - 1 } else { *lo }];
                (Cow::Borrowed(k), c.new_value().map(Cow::Borrowed))
            }),
            Source::Layer { head, .. } => head
                .take()
                .map(|(k, v)| (Cow::Borrowed(k), v.as_ref().map(Cow::Borrowed))),
            Source::Table { head, .. } => {
                head.take().map(|(k, v)| (Cow::Owned(k), v.map(Cow::Owned)))
            }
        }
    }

    /// Position a table source at its first entry within the bounds,
    /// given encoded. Memory sources start positioned.
    fn start(&mut self, lower: &Bound<Vec<u8>>, upper: &Bound<Vec<u8>>) -> Result<()> {
        if let Source::Table { cursor, rev, .. } = self {
            if *rev {
                match upper {
                    Bound::Unbounded => cursor.start_back()?,
                    Bound::Included(probe) => cursor.seek_back(probe)?,
                    Bound::Excluded(probe) => {
                        cursor.seek_back(probe)?;
                        if cursor.current().is_some_and(|(k, _)| k == &probe[..]) {
                            cursor.retreat()?;
                        }
                    }
                }
            } else {
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
            }
            self.decode_head()?;
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<()> {
        match self {
            Source::Overlay { lo, hi, rev, .. } => {
                if lo < hi {
                    if *rev {
                        *hi -= 1;
                    } else {
                        *lo += 1;
                    }
                }
            }
            Source::Layer { it, head, .. } => *head = it.next(),
            Source::Table { cursor, rev, .. } => {
                if *rev {
                    cursor.retreat()?;
                } else {
                    cursor.advance()?;
                }
                self.decode_head()?;
            }
        }
        Ok(())
    }

    /// Move to the first key at or past `key` (`probe` is its encoding),
    /// in the source's direction. Never moves back.
    fn seek(&mut self, key: &K, probe: &[u8]) -> Result<()> {
        let rev = match self {
            Source::Overlay { rev, .. } | Source::Layer { rev, .. } | Source::Table { rev, .. } => {
                *rev
            }
        };
        let reached = |s: &Self| {
            s.head_key()
                .is_none_or(|k| if rev { k <= key } else { k >= key })
        };
        if reached(self) {
            return Ok(());
        }
        if !matches!(self, Source::Table { .. }) {
            for _ in 0..SEEK_STEPS {
                self.advance()?;
                if reached(self) {
                    return Ok(());
                }
            }
        }
        let (lower, upper) = if rev {
            (Bound::Unbounded, Bound::Included(key))
        } else {
            (Bound::Included(key), Bound::Unbounded)
        };
        match self {
            Source::Overlay {
                entries, lo, hi, ..
            } => {
                if rev {
                    *hi = entries[..*hi].partition_point(|(e, _)| e <= key).max(*lo);
                } else {
                    *lo = (*lo + entries[*lo..*hi].partition_point(|(e, _)| e < key)).min(*hi);
                }
            }
            Source::Layer { map, it, head, .. } => {
                *it = pairs(map, lower, upper, rev);
                *head = it.next();
            }
            Source::Table { cursor, .. } => {
                if rev {
                    cursor.seek_back(probe)?;
                } else {
                    cursor.seek(probe)?;
                }
                self.decode_head()?;
            }
        }
        Ok(())
    }

    fn decode_head(&mut self) -> Result<()> {
        let Source::Table {
            cursor, ks, head, ..
        } = self
        else {
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

/// Merged iteration over one keyspace, in key order or, made with `rev`,
/// in reverse, skipping tombstones. Yields `(key, value)`; values from
/// memory are borrowed, values from disk are owned.
///
/// Iterating a write tx's keyspace sees the tx's own writes. An error from
/// a tx's cursor fails the tx, as any read error does.
pub struct Cursor<'a, K, M: Merge> {
    sources: Vec<Source<'a, K, M>>,
    lower: Bound<K>,
    upper: Bound<K>,
    rev: bool,
    started: bool,
    done: bool,
    probe: Vec<u8>,
    tx: Option<&'a TxState>,
}

fn encode_bound<K: OrderedCodec>(b: &Bound<K>) -> Bound<Vec<u8>> {
    let encode = |k: &K| {
        let mut b = Vec::with_capacity(k.encode_ordered_size());
        k.encode_ordered(&mut b);
        b
    };
    match b {
        Bound::Included(k) => Bound::Included(encode(k)),
        Bound::Excluded(k) => Bound::Excluded(encode(k)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

impl<'a, K, M> Cursor<'a, K, M>
where
    K: OrderedCodec + Clone + 'static,
    M: Merge,
    M::Value: Codec,
{
    /// `sources` newest first, memory sources already positioned within
    /// the bounds in the cursor's direction.
    pub(crate) fn new(
        sources: Vec<Source<'a, K, M>>,
        lower: Bound<K>,
        upper: Bound<K>,
        rev: bool,
        tx: Option<&'a TxState>,
    ) -> Self {
        Cursor {
            sources,
            lower,
            upper,
            rev,
            started: false,
            done: false,
            probe: Vec::new(),
            tx,
        }
    }

    fn start(&mut self) -> Result<()> {
        if let Some(tx) = self.tx {
            tx.check()?;
        }
        if !self.started {
            self.started = true;
            let (lower, upper) = (encode_bound(&self.lower), encode_bound(&self.upper));
            for s in &mut self.sources {
                s.start(&lower, &upper)?;
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

    /// The next entry in the cursor's order, tombstones included.
    fn next_entry(&mut self) -> Result<Option<Entry<'a, K, M::Value>>> {
        self.start()?;
        let rev = self.rev;
        let mut win: Option<usize> = None;
        for (i, s) in self.sources.iter().enumerate() {
            if let Some(k) = s.head_key()
                && win.is_none_or(|w| {
                    let best = self.sources[w].head_key().unwrap();
                    if rev { k > best } else { k < best }
                })
            {
                win = Some(i);
            }
        }
        let Some(w) = win else {
            return Ok(None);
        };
        let past_end = {
            let k = self.sources[w].head_key().unwrap();
            if rev {
                match &self.lower {
                    Bound::Included(l) => k < l,
                    Bound::Excluded(l) => k <= l,
                    Bound::Unbounded => false,
                }
            } else {
                match &self.upper {
                    Bound::Included(u) => k > u,
                    Bound::Excluded(u) => k >= u,
                    Bound::Unbounded => false,
                }
            }
        };
        if past_end {
            self.done = true;
            return Ok(None);
        }
        // older sources holding the same key are shadowed or, in an
        // additive keyspace, add to it
        let mut sum: Option<M::Value> = None;
        for i in w + 1..self.sources.len() {
            let same = self.sources[i].head_key().is_some()
                && self.sources[i].head_key() == self.sources[w].head_key();
            if same {
                if M::ADDITIVE
                    && let Some(Some(amount)) = self.sources[i].peek_value()
                {
                    match &mut sum {
                        Some(s) => M::add(s, &amount),
                        None => sum = Some(amount.into_owned()),
                    }
                }
                self.sources[i].advance()?;
            }
        }
        let mut entry = self.sources[w].take_head();
        self.sources[w].advance()?;
        if M::ADDITIVE
            && let Some((_, value)) = &mut entry
        {
            if let Some(mut s) = sum {
                if let Some(amount) = value.as_deref() {
                    M::add(&mut s, amount);
                }
                *value = Some(Cow::Owned(s));
            }
            // a sum of nothing is absent
            if value.as_deref().is_some_and(M::consider_delete) {
                *value = None;
            }
        }
        Ok(entry)
    }

    /// Move to `key`, or past it in the cursor's direction if it is absent,
    /// so iteration resumes there. Seeks must follow the cursor's
    /// direction; that is what makes a batch of them one sequential pass.
    pub(crate) fn seek(&mut self, key: &K) -> Result<()> {
        let res = self.seek_inner(key);
        res.map_err(|e| self.fail(e))
    }

    fn seek_inner(&mut self, key: &K) -> Result<()> {
        self.start()?;
        self.probe.clear();
        key.encode_ordered(&mut self.probe);
        for s in &mut self.sources {
            s.seek(key, &self.probe)?;
        }
        Ok(())
    }

    /// The value of `key`, moving the cursor to it. Lookups must come in
    /// the cursor's order.
    pub fn seek_get(&mut self, key: &K) -> Result<Option<Cow<'a, M::Value>>> {
        self.seek(key)?;
        let mut at = self.sources.iter().filter(|s| s.head_key() == Some(key));
        if !M::ADDITIVE {
            return Ok(at.next().and_then(|s| s.peek_value()).flatten());
        }
        let mut sum: Option<M::Value> = None;
        for amount in at.filter_map(|s| s.peek_value().flatten()) {
            match &mut sum {
                Some(s) => M::add(s, &amount),
                None => sum = Some(amount.into_owned()),
            }
        }
        Ok(sum.filter(|s| !M::consider_delete(s)).map(Cow::Owned))
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
