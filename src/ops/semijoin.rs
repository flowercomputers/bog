//! [`Semijoin`]: rows whose key the other side has ([`matching`]), or
//! lacks ([`without`]).
//!
//! [`matching`]: Semijoin::matching
//! [`without`]: Semijoin::without

use crate::{
    dataflow::state::{State, When},
    dataflow::{Bag, Binary, Delta},
    error::Result,
    store::txn::WriteTx,
};

use super::{Groups, Take, read_group};
use crate::stream::prelude::*;

/// Passes left rows `(k, a)` while the right side has a row of key `k`
/// ([`matching`](Self::matching), SQL's `EXISTS`), or while it has none
/// ([`without`](Self::without), `NOT EXISTS`), each at its own
/// multiplicity.
///
/// It keeps nothing: both parents' states must be grouped by the key.
/// For a key whose standing does not change, its new left rows pass or
/// not; for one that starts or stops passing, its whole left group is
/// emitted or retracted. A right row at a nonpositive multiplicity does
/// not count.
pub struct Semijoin<L: State, R> {
    anti: bool,
    left: L,
    right: R,
    /// Scratch, reused across runs.
    rows: Groups<L::Key, L::Value>,
    dirty: Vec<L::Key>,
    group: Vec<(L::Value, Delta)>,
}

impl<L: State, R> Semijoin<L, R> {
    /// Left rows whose key the right side has.
    pub fn matching(left: L, right: R) -> Self {
        Self::new(false, left, right)
    }

    /// Left rows whose key the right side lacks.
    pub fn without(left: L, right: R) -> Self {
        Self::new(true, left, right)
    }

    fn new(anti: bool, left: L, right: R) -> Self {
        Semijoin {
            anti,
            left,
            right,
            rows: Groups::default(),
            dirty: Vec::new(),
            group: Vec::new(),
        }
    }
}

impl<L, R> Semijoin<L, R>
where
    L: State,
    L::Value: Ord,
    R: State<Key = L::Key>,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &L::Delta,
        r: &R::Delta,
        out: &mut Bag<(L::Key, L::Value)>,
    ) -> Result<()> {
        self.rows.fill::<L>(l);
        let dirty = &mut self.dirty;
        dirty.clear();
        dirty.extend_from_slice(self.rows.keys());
        R::rows(r, &mut |k, _, _| dirty.push(k.clone()));
        dirty.sort();
        dirty.dedup();
        let mut left = self.rows.iter().peekable();
        for k in &self.dirty {
            let rows = left.next_if(|(lk, _)| *lk == k).map(|(_, rows)| rows);
            let was = self.right.exists(tx, k, When::Old)? != self.anti;
            let now = self.right.exists(tx, k, When::New)? != self.anti;
            if was == now {
                if now {
                    for (a, d) in rows.into_iter().flatten() {
                        out.push((k.clone(), a.clone()), *d);
                    }
                }
                continue;
            }
            let (when, sign) = if now { (When::New, 1) } else { (When::Old, -1) };
            read_group(&self.left, tx, k, when, Take::Whole, &mut self.group)?;
            for (a, d) in self.group.drain(..) {
                out.push((k.clone(), a), sign * d);
            }
        }
        Ok(())
    }
}

impl<L, R> Binary for Semijoin<L, R>
where
    L: State,
    L::Value: Ord,
    R: State<Key = L::Key>,
{
    type Left = L::Delta;
    type Right = R::Delta;
    type Gives = Bag<(L::Key, L::Value)>;
    type State = ();

    fn state(&self) {}

    fn fingerprint(&self, fp: &mut crate::dataflow::graph::Fingerprint) {
        fp.write_u64(self.anti as u64);
    }

    fn scratch(&mut self, release: bool) -> usize {
        crate::dataflow::vec_scratch(&mut self.dirty, release)
            + crate::dataflow::vec_scratch(&mut self.group, release)
            + self.rows.scratch(release)
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &L::Delta, r: &R::Delta, out: &mut Self::Gives) {
        let _ = self.run(tx, l, r, out);
    }
}

// The stream methods: how a schema declares this operator.

impl<K, V, M, C> Rows<K, V, M, C>
where
    K: Data,
    V: Data,
    M: Kind,
    C: Chain<Out = (K, V)>,
{
    /// The rows whose key the other side has: SQL's `EXISTS`.
    pub fn matching<B: Data>(&self, other: &impl GroupedFeed<K, B>) -> Rows<K, V, M> {
        self.semijoin(other, false)
    }

    /// The rows whose key the other side lacks: `NOT EXISTS`.
    pub fn without<B: Data>(&self, other: &impl GroupedFeed<K, B>) -> Rows<K, V, M> {
        self.semijoin(other, true)
    }

    pub(crate) fn semijoin<B: Data>(
        &self,
        other: &impl GroupedFeed<K, B>,
        anti: bool,
    ) -> Rows<K, V, M> {
        let kind = if anti { "without" } else { "matching" };
        let name = self.flow.name(kind, Some(&other.label()));
        let (l, ls) = arranged(self, &format!("{name}.left"));
        let (r, rs) = arranged(other, &format!("{name}.right"));
        let op = if anti {
            Semijoin::without(ls, rs)
        } else {
            Semijoin::matching(ls, rs)
        };
        let node = self.flow.build.borrow_mut().graph.binary(l, r, op);
        Rows::rooted(Root::at(node), self.flow.build.clone(), name.into(), None)
    }
}
