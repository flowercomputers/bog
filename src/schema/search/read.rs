//! The read side: BM25 scoring, block-max top-k, completion.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use crate::{
    Delta, Snapshot,
    codec::{Codec, OrderedCodec},
    error::Result,
    merge::Merge,
    state::Edge,
    subscribe as raw,
    tx::{KeyspaceId, WriteTx},
};

use super::{
    DOCS, Data, Posting, SearchHandle, TOTAL_LEN,
    index::{Block, PostingBlocks},
    tokenize::{Tokenizer, intern},
};
use crate::schema::{
    At,
    subscribe::{Commit, Recv, Subscription},
};

/// What keys a tally keyspace read here has.
trait TallyKey: OrderedCodec + Clone + Send + Sync + 'static {}
impl<K: OrderedCodec + Clone + Send + Sync + 'static> TallyKey for K {}

/// Sum two document-ordered score lists.
fn merge_scores<D: Ord>(a: Vec<(D, f64)>, b: Vec<(D, f64)>) -> Vec<(D, f64)> {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut a, mut b) = (a.into_iter().peekable(), b.into_iter().peekable());
    while let (Some((da, _)), Some((db, _))) = (a.peek(), b.peek()) {
        match da.cmp(db) {
            std::cmp::Ordering::Less => out.push(a.next().unwrap()),
            std::cmp::Ordering::Greater => out.push(b.next().unwrap()),
            std::cmp::Ordering::Equal => {
                let (d, sa) = a.next().unwrap();
                let (_, sb) = b.next().unwrap();
                out.push((d, sa + sb));
            }
        }
    }
    out.extend(a);
    out.extend(b);
    out
}

/// BM25 parameters. `k1` saturates term frequency; `b` scales length
/// normalization (0 = none, 1 = full).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25 {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25 {
    fn default() -> Self {
        Bm25 { k1: 1.2, b: 0.75 }
    }
}

impl Bm25 {
    /// One term's contribution, given its inverse document frequency.
    #[inline]
    fn score(&self, idf: f64, tf: f64, dl: f64, avg: f64) -> f64 {
        let norm = self.k1 * (1.0 - self.b + self.b * dl / avg);
        idf * (tf * (self.k1 + 1.0)) / (tf + norm)
    }
}

/// A scored document ordered so that the *worst* is the least: lower
/// score first, then the later document.
struct Scored<D>(f64, D);

impl<D: Ord> PartialEq for Scored<D> {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl<D: Ord> Eq for Scored<D> {}
impl<D: Ord> PartialOrd for Scored<D> {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl<D: Ord> Ord for Scored<D> {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then_with(|| o.1.cmp(&self.1))
    }
}

/// The read facade of a [`SearchView`](super::SearchView).
pub struct SearchRead<'a, D, T> {
    at: At<'a>,
    h: &'a SearchHandle<D>,
    _t: PhantomData<fn() -> T>,
}

impl<'a, D, T> Clone for SearchRead<'a, D, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, D, T> Copy for SearchRead<'a, D, T> {}

/// Postings of one term: `(doc, tf, dl)`, in document order.
type Postings<D> = Vec<(D, usize, usize)>;

impl<'a, D: Data, T: Tokenizer> SearchRead<'a, D, T> {
    pub(super) fn new(h: &'a SearchHandle<D>, at: At<'a>) -> Self {
        SearchRead {
            at,
            h,
            _t: PhantomData,
        }
    }

    /// An amount in a tally keyspace, zero when absent.
    fn amount<K: TallyKey>(&self, id: KeyspaceId<K, crate::Tally>, key: &K) -> Result<Delta> {
        self.at.with(
            |s| Ok(s.get(id, key)?.map_or(0, |n| *n)),
            |tx| Ok(tx.keyspace(id).get(key)?.map_or(0, |n| *n)),
        )
    }

    /// The rows of a tally keyspace in `range`, collected.
    fn rows<K: TallyKey>(
        &self,
        id: KeyspaceId<K, crate::Tally>,
        range: impl RangeBounds<K>,
    ) -> Result<Vec<(K, Delta)>> {
        let (lo, hi) = (range.start_bound().cloned(), range.end_bound().cloned());
        let (lo2, hi2) = (lo.clone(), hi.clone());
        self.at.with(
            move |s| {
                s.range(id, (lo2, hi2))
                    .map(|r| r.map(|(k, n)| (k.into_owned(), *n)))
                    .collect()
            },
            move |tx| {
                tx.keyspace(id)
                    .range((lo, hi))
                    .map(|r| r.map(|(k, n)| (k.into_owned(), *n)))
                    .collect()
            },
        )
    }

    /// Every committed change to the index from the state this facade
    /// reads on: each posting's change in term frequency, as
    /// `(term id, doc, change)`; see [`intern`].
    pub fn subscribe(&self) -> Subscription<(u64, D, Delta)> {
        let (store, epoch) = self.at.store_and_epoch();
        Subscription::new(PostingChanges {
            sub: store.subscribe_from(self.h.postings, epoch),
        })
    }

    /// The postings keyspace: [`Posting`] keys at term frequency, the
    /// escape hatch to [`Snapshot::runs_indexed`].
    pub fn postings_keyspace(&self) -> KeyspaceId<Posting<D>, crate::Tally> {
        self.h.postings
    }

    /// Documents indexed.
    pub fn len(&self) -> Result<usize> {
        Ok(self.amount(self.h.corpus, &DOCS)?.max(0) as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Mean document length in tokens.
    pub fn avg_len(&self) -> Result<f64> {
        let n = self.len()?;
        if n == 0 {
            return Ok(0.0);
        }
        Ok(self.amount(self.h.corpus, &TOTAL_LEN)? as f64 / n as f64)
    }

    /// A document's length in tokens; zero if unknown.
    pub fn doc_len(&self, doc: &D) -> Result<usize> {
        Ok(self.amount(self.h.lengths, doc)?.max(0) as usize)
    }

    /// The documents containing term `id`, each with its frequency and
    /// length, in document order. Adjacent rows of one document (a
    /// document mid-update briefly holds rows at two lengths) fold
    /// together, and documents whose frequency nets to nothing drop.
    fn postings_in(&self, lo: Bound<Posting<D>>, hi: Bound<Posting<D>>) -> Result<Postings<D>> {
        let mut out: Postings<D> = Vec::new();
        let mut held: Option<(D, Delta, usize)> = None;
        for ((_, doc, dl), tf) in self.rows(self.h.postings, (lo, hi))? {
            let Edge::Val(doc) = doc else {
                unreachable!("postings are kept by document")
            };
            match held.take() {
                Some((d, t, l)) if d == doc => held = Some((d, t + tf, l.max(dl as usize))),
                prev => {
                    if let Some((d, t, l)) = prev
                        && t > 0
                    {
                        out.push((d, t as usize, l));
                    }
                    held = Some((doc, tf, dl as usize));
                }
            }
        }
        if let Some((d, t, l)) = held
            && t > 0
        {
            out.push((d, t as usize, l));
        }
        Ok(out)
    }

    fn postings_of(&self, id: u64) -> Result<Postings<D>> {
        self.postings_in(
            Bound::Included((id, Edge::Lo, 0)),
            Bound::Included((id, Edge::Hi, u32::MAX)),
        )
    }

    /// The documents containing `term` with its frequency in each, in
    /// document order.
    pub fn postings(&self, term: &str) -> Result<Vec<(D, usize)>> {
        Ok(self
            .postings_of(intern(term))?
            .into_iter()
            .map(|(d, tf, _)| (d, tf))
            .collect())
    }

    /// Documents containing `term`, from the term dictionary.
    pub fn doc_frequency(&self, term: &str) -> Result<usize> {
        Ok(self.amount(self.h.terms, &term.to_string())?.max(0) as usize)
    }

    /// Terms beginning with `prefix`, in term order, with their document
    /// frequencies. Linear in the matching terms: never touches the
    /// postings.
    pub fn terms_with_prefix(&self, prefix: &str) -> Result<Vec<(String, usize)>> {
        let rows = self.rows(self.h.terms, prefix.to_string()..)?;
        Ok(rows
            .into_iter()
            .take_while(|(t, _)| t.starts_with(prefix))
            .filter(|(_, df)| *df > 0)
            .map(|(t, df)| (t, df as usize))
            .collect())
    }

    /// Documents containing every term of `query`, in document order.
    pub fn matching_all(&self, query: &str) -> Result<Vec<D>> {
        let mut terms = T::tokenize(query);
        terms.sort();
        terms.dedup();
        let Some(first) = terms.first() else {
            return Ok(Vec::new());
        };
        let mut hits: Vec<D> = self.postings(first)?.into_iter().map(|(d, _)| d).collect();
        for term in &terms[1..] {
            let these: Vec<D> = self.postings(term)?.into_iter().map(|(d, _)| d).collect();
            hits.retain(|d| these.binary_search(d).is_ok());
            if hits.is_empty() {
                break;
            }
        }
        Ok(hits)
    }

    /// The `k` best documents for `query` under default [`Bm25`]: any
    /// term may match, and the score sums over matching terms. Ties
    /// break by document order.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(D, f64)>> {
        self.search_with(query, k, Bm25::default())
    }

    pub fn search_with(&self, query: &str, k: usize, p: Bm25) -> Result<Vec<(D, f64)>> {
        let mut terms = T::tokenize(query);
        terms.sort();
        terms.dedup();
        let n = self.len()? as f64;
        let avg = self.avg_len()?;
        if n == 0.0 || terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let idf_of = |df: usize| ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();

        let mut query: Vec<(u64, f64)> = Vec::with_capacity(terms.len());
        for t in &terms {
            let df = self.doc_frequency(t)?;
            if df > 0 {
                query.push((intern(t), idf_of(df)));
            }
        }
        if query.is_empty() {
            return Ok(Vec::new());
        }
        if let At::Snapshot(snap) = self.at
            && let Some(hits) = self.top_k(snap, &query, k, p, avg)?
        {
            return Ok(hits);
        }
        // Some run lacks its block index, or this is a transaction:
        // score everything. Each term's postings arrive in document
        // order, so a term merges in linearly.
        let mut scores: Vec<(D, f64)> = Vec::new();
        for &(id, idf) in &query {
            let these: Vec<(D, f64)> = self
                .postings_of(id)?
                .into_iter()
                .map(|(doc, tf, dl)| (doc, p.score(idf, tf as f64, dl as f64, avg)))
                .collect();
            scores = merge_scores(scores, these);
        }
        let mut ranked = scores;
        ranked.sort_by(|(da, sa), (db, sb)| sb.total_cmp(sa).then_with(|| da.cmp(db)));
        ranked.truncate(k);
        Ok(ranked)
    }

    /// Top-k by block maxima. Document ranges are visited best-bound
    /// first and the walk stops once no remaining range can beat the
    /// k-th score. `None` if some run lacks the index.
    ///
    /// A document's score sums over query terms, and its frequency for
    /// a term sums over runs, so the bound over a document range is the
    /// sum, over terms, of the score at the summed block maxima of that
    /// term's runs. Sweeping the block boundaries of every term gives
    /// the elementary ranges on which that set of blocks is fixed.
    fn top_k(
        &self,
        snap: &Snapshot<'_>,
        query: &[(u64, f64)],
        k: usize,
        p: Bm25,
        avg: f64,
    ) -> Result<Option<Vec<(D, f64)>>> {
        // every block of every query term across runs, tagged by term
        let runs =
            snap.runs_indexed::<Posting<D>, crate::Tally, PostingBlocks<D>>(self.h.postings)?;
        let mut indexes = Vec::with_capacity(runs.len());
        for run in &runs {
            let Some(index) = run.index() else {
                return Ok(None);
            };
            indexes.push(index);
        }
        let mut blocks: Vec<(usize, &Block<D>)> = Vec::new();
        for index in indexes {
            for (t, &(id, _)) in query.iter().enumerate() {
                blocks.extend(index.blocks_for(id).iter().map(|b| (t, b)));
            }
        }
        blocks.sort_by(|a, b| a.1.first.cmp(&b.1.first));
        let mut points: Vec<&D> = blocks
            .iter()
            .flat_map(|(_, b)| [&b.first, &b.last])
            .collect();
        points.sort();
        points.dedup();

        let term_bound = |t: usize, max_tf: u32, min_dl: u32| {
            if max_tf == 0 {
                0.0
            } else {
                p.score(query[t].1, max_tf as f64, min_dl as f64, avg)
            }
        };
        // (from, to, closed at `to`, bound)
        let mut ranges: Vec<(&D, &D, bool, f64)> = Vec::with_capacity(points.len());
        let mut active: Vec<(usize, &Block<D>)> = Vec::new();
        let mut per_term: Vec<(u32, u32)> = vec![(0, u32::MAX); query.len()];
        // each term's best possible contribution anywhere
        let mut best: Vec<f64> = vec![0.0; query.len()];
        let mut next = 0;
        for (i, &pt) in points.iter().enumerate() {
            while next < blocks.len() && blocks[next].1.first <= *pt {
                active.push(blocks[next]);
                next += 1;
            }
            active.retain(|(_, b)| b.last >= *pt);
            per_term.iter_mut().for_each(|e| *e = (0, u32::MAX));
            for (t, b) in &active {
                let e = &mut per_term[*t];
                e.0 += b.max_tf;
                e.1 = e.1.min(b.min_dl);
            }
            let mut ub = 0.0;
            for (t, &(tf, dl)) in per_term.iter().enumerate() {
                let b = term_bound(t, tf, dl);
                best[t] = best[t].max(b);
                ub += b;
            }
            match points.get(i + 1) {
                Some(&q) => ranges.push((pt, q, false, ub)),
                None => ranges.push((pt, pt, true, ub)),
            }
        }
        ranges.sort_by(|a, b| b.3.total_cmp(&a.3));

        // MaxScore: terms in ascending order of best contribution. Once
        // the k-th score exceeds the summed bests of a prefix, a
        // document holding only those terms cannot place, so they stop
        // being scanned and are merely probed for candidates the rest
        // produce.
        let mut order: Vec<usize> = (0..query.len()).collect();
        order.sort_by(|&a, &b| best[a].total_cmp(&best[b]));

        let mut heap: BinaryHeap<Reverse<Scored<D>>> = BinaryHeap::with_capacity(k + 1);
        for (from, to, closed, ub) in ranges {
            let threshold = if heap.len() == k {
                heap.peek()
                    .map(|Reverse(w)| w.0)
                    .unwrap_or(f64::NEG_INFINITY)
            } else {
                f64::NEG_INFINITY
            };
            if ub == 0.0 || ub <= threshold {
                break;
            }
            let mut acc = 0.0;
            let mut essential: Vec<bool> = vec![true; query.len()];
            for &t in &order {
                acc += best[t];
                if acc <= threshold {
                    essential[t] = false;
                } else {
                    break;
                }
            }
            let mut scores: Vec<(D, f64)> = Vec::new();
            for (t, &(id, idf)) in query.iter().enumerate() {
                if !essential[t] {
                    continue;
                }
                let lo = Bound::Included((id, Edge::Val(from.clone()), 0u32));
                let hi = if closed {
                    Bound::Included((id, Edge::Val(to.clone()), u32::MAX))
                } else {
                    Bound::Excluded((id, Edge::Val(to.clone()), 0u32))
                };
                let these: Vec<(D, f64)> = self
                    .postings_in(lo, hi)?
                    .into_iter()
                    .map(|(doc, tf, dl)| (doc, p.score(idf, tf as f64, dl as f64, avg)))
                    .collect();
                scores = merge_scores(scores, these);
            }
            // complete each candidate with the terms not scanned
            for (t, &(id, idf)) in query.iter().enumerate() {
                if essential[t] {
                    continue;
                }
                for (doc, s) in scores.iter_mut() {
                    let one = self.postings_in(
                        Bound::Included((id, Edge::Val(doc.clone()), 0)),
                        Bound::Included((id, Edge::Val(doc.clone()), u32::MAX)),
                    )?;
                    if let Some((_, tf, dl)) = one.first() {
                        *s += p.score(idf, *tf as f64, *dl as f64, avg);
                    }
                }
            }
            for (doc, s) in scores {
                if heap.len() < k {
                    heap.push(Reverse(Scored(s, doc)));
                } else if heap
                    .peek()
                    .is_some_and(|Reverse(w)| Scored(s, doc.clone()) > *w)
                {
                    heap.pop();
                    heap.push(Reverse(Scored(s, doc)));
                }
            }
        }
        let mut ranked: Vec<(D, f64)> = heap
            .into_iter()
            .map(|Reverse(Scored(s, d))| (d, s))
            .collect();
        ranked.sort_by(|(da, sa), (db, sb)| sb.total_cmp(sa).then_with(|| da.cmp(db)));
        Ok(Some(ranked))
    }

    /// The `k` most common terms beginning with `prefix`, with their
    /// document frequencies, most common first.
    pub fn complete(&self, prefix: &str, k: usize) -> Result<Vec<(String, usize)>> {
        let prefix = prefix.to_lowercase();
        let mut terms = self.terms_with_prefix(&prefix)?;
        terms.sort_by(|(ta, da), (tb, db)| db.cmp(da).then_with(|| ta.cmp(tb)));
        terms.truncate(k);
        Ok(terms)
    }
}

/// The postings keyspace's commits as `(term id, doc, change in tf)`.
struct PostingChanges<D> {
    sub: raw::Subscription<Posting<D>, crate::Tally>,
}

impl<D: Clone> PostingChanges<D> {
    fn commit(&self, d: raw::Delta<Posting<D>, crate::Tally>) -> Commit<(u64, D, Delta)> {
        let changes = d
            .changes
            .iter()
            .map(|((term, doc, _), c)| {
                let (crate::Change::Insert(by), Edge::Val(doc)) = (c, doc) else {
                    unreachable!("postings are amounts by document")
                };
                (*term, doc.clone(), *by)
            })
            .collect();
        Commit {
            epoch: d.epoch,
            changes,
        }
    }
}

impl<D: Data> Recv<(u64, D, Delta)> for PostingChanges<D> {
    fn recv(&self) -> Option<Commit<(u64, D, Delta)>> {
        self.sub.recv().map(|d| self.commit(d))
    }
    fn try_recv(
        &self,
    ) -> std::result::Result<Option<Commit<(u64, D, Delta)>>, crate::subscribe::Closed> {
        Ok(self.sub.try_recv()?.map(|d| self.commit(d)))
    }
    fn recv_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> std::result::Result<Option<Commit<(u64, D, Delta)>>, crate::subscribe::Closed> {
        Ok(self.sub.recv_timeout(timeout)?.map(|d| self.commit(d)))
    }
}

#[allow(dead_code)]
fn _bounds<K: Codec, M: Merge>(_: &KeyspaceId<K, M>, _: &mut WriteTx) {}
