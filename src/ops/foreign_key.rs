//! [`ForeignKey`]: references to keys the parent side lacks.

use crate::stream::prelude::*;
use crate::{
    dataflow::state::{State, When},
    dataflow::{Bag, Binary},
    error::Result,
    store::txn::WriteTx,
};

/// Emits, at `+1`, every key that after a run is referenced by a child
/// row `(k, a)` on the left but has no parent row `(k, b)` on the right:
/// the run's foreign-key violations. Feed it to a
/// [`Reject`] to refuse transactions that have any.
///
/// It keeps nothing: the children must be grouped by the key they
/// reference, typically by an [`Arrange`] node, and the
/// parents are usually a table keyed by it. Its output is each run's
/// violations, not a maintained relation: a rejected tx leaves nothing to
/// retract.
pub struct ForeignKey<L: State, R> {
    dirty: Vec<L::Key>,
    children: L,
    parents: R,
}

impl<L: State, R> ForeignKey<L, R> {
    pub fn new(children: L, parents: R) -> Self {
        ForeignKey {
            dirty: Vec::new(),
            children,
            parents,
        }
    }
}

impl<L, R> ForeignKey<L, R>
where
    L: State,
    R: State<Key = L::Key>,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &L::Delta,
        r: &R::Delta,
        out: &mut Bag<L::Key>,
    ) -> Result<()> {
        let dirty = &mut self.dirty;
        L::rows(l, &mut |k, _, _| dirty.push(k.clone()));
        R::rows(r, &mut |k, _, _| dirty.push(k.clone()));
        dirty.sort();
        dirty.dedup();
        for k in dirty.drain(..) {
            if self.children.exists(tx, &k, When::New)?
                && !self.parents.exists(tx, &k, When::New)?
            {
                out.push(k, 1);
            }
        }
        Ok(())
    }
}

impl<L, R> Binary for ForeignKey<L, R>
where
    L: State,
    R: State<Key = L::Key>,
{
    type Left = L::Delta;
    type Right = R::Delta;
    type Gives = Bag<L::Key>;
    type State = ();

    fn state(&self) {}

    fn scratch(&mut self, release: bool) -> usize {
        crate::dataflow::vec_scratch(&mut self.dirty, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &L::Delta, r: &R::Delta, out: &mut Bag<L::Key>) {
        let _ = self.run(tx, l, r, out);
    }
}

// The stream methods: how a schema declares this operator.

impl<K: Data, C: Chain<Out = K>> Stream<K, C> {
    /// The keys here that `parent` lacks, after each run, and the
    /// operator's name: the rows a foreign key refuses. The keys are
    /// counted, so the state is one count per distinct key.
    pub(crate) fn dangling<B: Data>(
        &self,
        parent: &impl GroupedFeed<K, B>,
    ) -> (NodeRef<Bag<K>>, String) {
        let name = self.flow.name("foreign_key", Some(&parent.label()));
        let inputs = self.flow.inputs();
        let counts = {
            let mut b = self.flow.build.borrow_mut();
            let id: KeyspaceId<K, Tally> = b.id(&format!("{name}.refs"));
            b.graph.nary(&inputs, crate::ops::CountRows::new(id))
        };
        let (r, rs) = arranged(parent, &format!("{name}.parents"));
        let mut b = self.flow.build.borrow_mut();
        let refs = b.graph.state(counts);
        let node = b
            .graph
            .binary(counts.rows(), r, crate::ops::ForeignKey::new(refs, rs));
        (node, name)
    }
}
