//! [`SearchView`]: full-text search over a keyed stream of documents,
//! maintained incrementally with no operator state.
//!
//! Every statistic BM25 needs is additive, so the whole index is four
//! tally keyspaces, written blind by one node:
//!
//! | keyspace   | key                       | amount             |
//! |------------|---------------------------|--------------------|
//! | `postings` | `(term id, doc, length)`  | term frequency     |
//! | `terms`    | `term`                    | document frequency |
//! | `lengths`  | `doc`                     | document length    |
//! | `corpus`   | `0`, `1`                  | documents, total length |
//!
//! Terms are interned to a 64-bit hash, so a posting is three machine
//! words rather than a heap string, and it carries its document's length
//! so scoring never leaves the postings. Maintenance is linear in the
//! tokens of the delta per commit; retracting a document retracts its
//! postings at negative weight and every count cancels in storage.
//!
//! Every run of the postings keyspace carries a [`PostingBlocks`] index:
//! per term, per block of postings, the largest term frequency and the
//! shortest document. A top-k query for one term visits document ranges
//! in order of their score bound and stops once the bound cannot beat
//! the k-th result, which is how a common term costs a fraction of its
//! postings. Completion and inverse document frequency read the term
//! dictionary and never touch postings.
//!
//! The tokenizer is a *type*: its name is folded into the keyspace
//! names, so changing it changes the graph fingerprint and the index is
//! rebuilt rather than left stale.

use std::{cell::OnceCell, collections::HashMap, marker::PhantomData, ops::Deref, sync::Arc};

use crate::{
    Bag, Delta, Nary, Tally,
    state::{Declare, Edge},
    tx::{KeyspaceId, WriteTx},
};

use crate::schema::{
    At, Data, Grouped, KeyedFeed, Root, Rows, Shared,
    view::{SlotCore, ViewDecl},
};

mod index;
pub use index::{Block, PostingBlocks};

mod read;
pub use read::{Bm25, SearchRead};

mod tokenize;
pub use tokenize::{Simple, Tokenizer, intern};

/// A posting's key: `(term id, document, document length)`; its amount
/// is the term's frequency in the document.
pub type Posting<D> = (u64, Edge<D>, u32);

pub(super) const DOCS: u8 = 0;
pub(super) const TOTAL_LEN: u8 = 1;

/// A full-text index over documents `D`, defined from a [`Keyed`]
/// stream of `(D, String)`: each document's text.
///
/// [`Keyed`]: crate::stream::Keyed
pub struct SearchView<D, T = Simple>(PhantomData<fn() -> (D, T)>);

/// The keyspaces of a search view.
pub struct SearchHandle<D> {
    postings: KeyspaceId<Posting<D>, Tally>,
    terms: KeyspaceId<String, Tally>,
    lengths: KeyspaceId<D, Tally>,
    corpus: KeyspaceId<u8, Tally>,
}

impl<D> Clone for SearchHandle<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for SearchHandle<D> {}

/// The node maintaining the index: each document's rows, at the
/// document's multiplicity, added to the four keyspaces. Emits one
/// `(term, doc)` row per distinct term of each document.
struct Index<D, T> {
    h: SearchHandle<D>,
    /// Scratch, reused across runs.
    tf: HashMap<String, Delta>,
    postings: Vec<(Posting<D>, Delta)>,
    terms: Vec<(String, Delta)>,
    lengths: Vec<(D, Delta)>,
    corpus: [Delta; 2],
    _t: PhantomData<fn() -> T>,
}

impl<D: Data, T: Tokenizer> Index<D, T> {
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&Bag<(D, String)>],
        out: &mut Bag<(String, D)>,
    ) -> crate::error::Result<()> {
        self.postings.clear();
        self.terms.clear();
        self.lengths.clear();
        self.corpus = [0, 0];
        for b in batches {
            for ((doc, text), d) in b.diffs() {
                let tokens = T::tokenize(text);
                let len = tokens.len();
                self.tf.clear();
                for t in tokens {
                    *self.tf.entry(t).or_insert(0) += 1;
                }
                for (term, n) in self.tf.drain() {
                    self.postings
                        .push(((intern(&term), Edge::Val(doc.clone()), len as u32), n * d));
                    self.terms.push((term.clone(), d));
                    out.push((term, doc.clone()), d);
                }
                self.lengths.push((doc.clone(), len as Delta * d));
                self.corpus[DOCS as usize] += d;
                self.corpus[TOTAL_LEN as usize] += len as Delta * d;
            }
        }
        let mut ks = tx.keyspace(self.h.postings);
        for (k, n) in self.postings.drain(..) {
            ks.add(k, n)?;
        }
        let mut ks = tx.keyspace(self.h.terms);
        for (k, n) in self.terms.drain(..) {
            ks.add(k, n)?;
        }
        let mut ks = tx.keyspace(self.h.lengths);
        for (k, n) in self.lengths.drain(..) {
            ks.add(k, n)?;
        }
        let mut ks = tx.keyspace(self.h.corpus);
        ks.add(DOCS, self.corpus[DOCS as usize])?;
        ks.add(TOTAL_LEN, self.corpus[TOTAL_LEN as usize])?;
        Ok(())
    }
}

impl<D: Data, T: Tokenizer> Nary for Index<D, T> {
    type Takes = Bag<(D, String)>;
    type Gives = Bag<(String, D)>;
    type State = ();

    fn state(&self) {}

    fn keyspaces(&self) -> Vec<Declare> {
        vec![
            Declare::derived_indexed::<Posting<D>, Tally, PostingBlocks<D>>(self.h.postings),
            Declare::derived(self.h.terms),
            Declare::derived(self.h.lengths),
            Declare::derived(self.h.corpus),
        ]
    }

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        fp.write_str(T::NAME);
    }

    fn scratch(&mut self, release: bool) -> usize {
        if release {
            self.tf = HashMap::new();
        }
        self.tf.capacity() * size_of::<(String, Delta)>()
            + crate::vec_scratch(&mut self.postings, release)
            + crate::vec_scratch(&mut self.terms, release)
            + crate::vec_scratch(&mut self.lengths, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Self::Takes], out: &mut Self::Gives) {
        if self.run(tx, batches, out).is_err() {
            crate::Buffer::clear(out);
        }
    }
}

/// The builder slot of a [`SearchView`]: `define` it, then it is the
/// index as a [`Grouped`] stream of `(term, doc)` rows, one per distinct
/// term of each document.
pub struct SearchSlot<D: Data, T> {
    core: SlotCore<String, D>,
    head: OnceCell<Grouped<String, D>>,
    handle: OnceCell<SearchHandle<D>>,
    _t: PhantomData<fn() -> T>,
}

impl<D: Data, T: Tokenizer> SearchSlot<D, T> {
    /// Index every document's text.
    pub fn define(&self, from: &impl KeyedFeed<D, String>) {
        self.core.defining();
        let inputs = from.inputs();
        let mut b = self.core.build.borrow_mut();
        let part = |p: &str| format!("{}.{}.{p}", self.core.name, T::NAME);
        let h = SearchHandle {
            postings: b.id(&part("postings")),
            terms: b.id(&part("terms")),
            lengths: b.id(&part("lengths")),
            corpus: b.id(&part("corpus")),
        };
        let node = b.graph.nary(
            &inputs,
            Index::<D, T> {
                h,
                tf: HashMap::new(),
                postings: Vec::new(),
                terms: Vec::new(),
                lengths: Vec::new(),
                corpus: [0, 0],
                _t: PhantomData,
            },
        );
        drop(b);
        let head = Rows::rooted(
            Root::at(node),
            self.core.build.clone(),
            self.core.name.clone(),
            None,
        );
        let _ = self.head.set(head);
        let _ = self.handle.set(h);
    }
}

impl<D: Data, T> Deref for SearchSlot<D, T> {
    type Target = Grouped<String, D>;
    fn deref(&self) -> &Grouped<String, D> {
        self.head.get().unwrap_or_else(|| self.core.undefined())
    }
}

impl<D: Data, T: Tokenizer> ViewDecl for SearchView<D, T> {
    type Slot = SearchSlot<D, T>;
    type Handle = SearchHandle<D>;
    type Read<'a> = SearchRead<'a, D, T>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot {
        SearchSlot {
            core: SlotCore::new(name, build),
            head: OnceCell::new(),
            handle: OnceCell::new(),
            _t: PhantomData,
        }
    }

    fn finish(slot: Self::Slot) -> Self::Handle {
        match slot.handle.into_inner() {
            Some(h) => h,
            None => slot.core.undefined(),
        }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        SearchRead::new(h, at)
    }
}
