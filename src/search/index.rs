//! [`PostingBlocks`]: the per-run block-max index on the postings
//! keyspace.

use crate::{
    dataflow::Delta,
    dataflow::state::Edge,
    store::codec::{Codec, OrderedCodec},
    store::index::RunIndex,
};

use super::{Data, Posting};

/// Postings per block.
const BLOCK: usize = 128;

/// One block of a term's postings in one run: its document range, the
/// largest term frequency, and the shortest document. Since BM25 grows
/// with term frequency and shrinks with length, `(max_tf, min_dl)`
/// bounds every score in the block under any parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block<D> {
    pub term: u64,
    pub first: D,
    pub last: D,
    pub max_tf: u32,
    pub min_dl: u32,
}

/// The [`RunIndex`] every search view registers on its postings: blocks
/// in `(term, first)` order, so a term's blocks are one contiguous slice.
pub struct PostingBlocks<D> {
    blocks: Vec<Block<D>>,
    /// The block being filled, and how many postings it holds.
    open: Option<(Block<D>, usize)>,
}

impl<D> Default for PostingBlocks<D> {
    fn default() -> Self {
        PostingBlocks {
            blocks: Vec::new(),
            open: None,
        }
    }
}

impl<D: Data> PostingBlocks<D> {
    /// The blocks of `term` in this run, in document order.
    pub fn blocks_for(&self, term: u64) -> &[Block<D>] {
        let lo = self.blocks.partition_point(|b| b.term < term);
        let hi = lo + self.blocks[lo..].partition_point(|b| b.term == term);
        &self.blocks[lo..hi]
    }

    pub fn blocks(&self) -> &[Block<D>] {
        &self.blocks
    }
}

impl<D: Data> RunIndex<Posting<D>, Delta> for PostingBlocks<D> {
    const NAME: &'static str = "posting-blocks";

    fn push(&mut self, key: &Posting<D>, value: Option<&Delta>) {
        let (term, doc, dl) = key;
        let Edge::Val(doc) = doc else {
            unreachable!("postings are kept by document")
        };
        // a block closes at `BLOCK` postings or at a term change
        let fresh = match &self.open {
            Some((b, n)) => b.term != *term || *n == BLOCK,
            None => true,
        };
        if fresh {
            if let Some((b, _)) = self.open.take() {
                self.blocks.push(b);
            }
            self.open = Some((
                Block {
                    term: *term,
                    first: doc.clone(),
                    last: doc.clone(),
                    max_tf: 0,
                    min_dl: u32::MAX,
                },
                0,
            ));
        }
        let (b, n) = self.open.as_mut().expect("opened above");
        b.last = doc.clone();
        // only positive rows can raise a score; a retraction in this run
        // only lowers what an older run holds
        if let Some(&tf) = value
            && tf > 0
        {
            b.max_tf = b.max_tf.max(tf as u32);
            b.min_dl = b.min_dl.min(*dl);
        }
        *n += 1;
    }

    fn finish(&mut self) -> bool {
        if let Some((b, _)) = self.open.take() {
            self.blocks.push(b);
        }
        true
    }

    fn resident_bytes(&self) -> usize {
        self.blocks.capacity() * size_of::<Block<D>>()
    }

    fn encode(&self, out: &mut Vec<u8>) -> bool {
        (self.blocks.len() as u32).encode(out);
        for b in &self.blocks {
            b.term.encode(out);
            b.first.encode(out);
            b.last.encode(out);
            b.max_tf.encode(out);
            b.min_dl.encode(out);
        }
        true
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut p = bytes;
        let n = u32::decode(&mut p).ok()?;
        let mut blocks = Vec::with_capacity(n as usize);
        for _ in 0..n {
            blocks.push(Block {
                term: u64::decode(&mut p).ok()?,
                first: D::decode(&mut p).ok()?,
                last: D::decode(&mut p).ok()?,
                max_tf: u32::decode(&mut p).ok()?,
                min_dl: u32::decode(&mut p).ok()?,
            });
        }
        p.is_empty().then_some(PostingBlocks { blocks, open: None })
    }
}

#[allow(dead_code)]
fn _ordered<D: OrderedCodec>(d: &D) -> &D {
    d
}
