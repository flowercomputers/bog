use crate::{Binary, Buffer, Nary, Node, Unary, WriteTx};
use smallvec::SmallVec;
use std::{any::Any, collections::BTreeSet, marker::PhantomData};

type OpDagNodeId = usize;

/// Typed, copyable handle to a node's output. `B` is the buffer type the
/// node gives (`Bag<_>` or `Register<_, _>`), so wiring a child whose
/// `Takes` differs is a compile error.
pub struct NodeRef<B>(OpDagNodeId, PhantomData<fn() -> B>);

impl<B> Clone for NodeRef<B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B> Copy for NodeRef<B> {}

enum Kind {
    /// Fed by `give`; `staging` accumulates between runs and is swapped
    /// into `out` when the source runs.
    Source {
        staging: Box<dyn Buffer>,
    },
    Op(Box<dyn Node>),
}

struct OpDagNode {
    kind: Kind,
    parents: SmallVec<[OpDagNodeId; 2]>,
    children: SmallVec<[OpDagNodeId; 4]>,
    /// This node's delta for the current epoch. Always allocated; cleared
    /// after every run, keeping the allocation.
    out: Box<dyn Buffer>,
}

// TODO:
// better memory accounting—we should always know how many bytes are allocated
// for this OpDag, ditto for txn, ditto for readers?
#[derive(Default)]
pub struct OpDag {
    nodes: Vec<OpDagNode>,
    pending: BTreeSet<OpDagNodeId>,
    /// Nodes that ran this epoch. Cleared after every graph run.
    touched: Vec<OpDagNodeId>,
}

impl OpDag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source<B: Buffer>(&mut self) -> NodeRef<B> {
        self.push(
            Kind::Source {
                staging: Box::new(B::empty()),
            },
            &[],
            Box::new(B::empty()),
        )
    }

    pub fn unary<U: Unary + 'static>(
        &mut self,
        input: NodeRef<U::Takes>,
        op: U,
    ) -> NodeRef<U::Gives> {
        self.push(
            Kind::Op(Box::new(UnaryNode(op))),
            &[input.0],
            Box::new(U::Gives::empty()),
        )
    }

    pub fn binary<B: Binary + 'static>(
        &mut self,
        l: NodeRef<B::Left>,
        r: NodeRef<B::Right>,
        op: B,
    ) -> NodeRef<B::Gives> {
        self.push(
            Kind::Op(Box::new(BinaryNode(op))),
            &[l.0, r.0],
            Box::new(B::Gives::empty()),
        )
    }

    /// One node over any number of same-typed inputs. Several inputs are
    /// unioned, which is what `Nary` means.
    pub fn nary<N: Nary + 'static>(
        &mut self,
        inputs: &[NodeRef<N::Takes>],
        op: N,
    ) -> NodeRef<N::Gives> {
        assert!(!inputs.is_empty(), "nary node needs at least one input");
        let ids: SmallVec<[OpDagNodeId; 4]> = inputs.iter().map(|r| r.0).collect();
        self.push(
            Kind::Op(Box::new(NaryNode(op))),
            &ids,
            Box::new(N::Gives::empty()),
        )
    }

    fn push<B>(&mut self, kind: Kind, parents: &[OpDagNodeId], out: Box<dyn Buffer>) -> NodeRef<B> {
        let id = self.nodes.len();
        for &p in parents {
            debug_assert!(p < id, "parent ids always precede the child: {p} < {id}");
            self.nodes[p].children.push(id);
        }
        self.nodes.push(OpDagNode {
            kind,
            parents: parents.into(),
            children: Default::default(),
            out,
        });
        NodeRef(id, PhantomData)
    }

    /// Stage input at a source. Drains `input` so the caller keeps its
    /// allocation. Bags append; registers compose per key.
    pub fn give<B: Buffer>(&mut self, src: NodeRef<B>, input: &mut B) {
        let Kind::Source { staging } = &mut self.nodes[src.0].kind else {
            panic!("NodeRef is not a source");
        };
        let staging: &mut B = staging
            .as_any_mut()
            .downcast_mut()
            .expect("source type mismatch"); // unreachable given NodeRef<B>
        staging.absorb(input);
        if staging.len() > 0 {
            self.pending.insert(src.0);
        }
    }

    /// Run every dirty node once, in dependency order.
    ///
    /// Open questions, in rough priority:
    /// - directed feed (on-demand backfill just to out-of-date leaves):
    ///   when an operator's closed epoch is lower than the db's epoch, flush
    ///   every historical batch from its most up-to-date stateful upstream.
    ///   Needs ops to know whether children are out of date, or ops to track
    ///   their close in their own keyspace. Doesn't work for every op type,
    ///   so it's best-effort, or nodes are marked restorable.
    /// - fusing unary chains as in fold2.
    /// - keyspace wiring: a second dependency layer beyond the batches.
    ///   Possibly just hand `eval` the parents' keyspaces.
    /// - ops that always need to run (windowed, etc.): a scheduler they
    ///   subscribe to, or mark them always dirty, or a `touch` method that
    ///   takes no inputs but can output.
    pub fn run(&mut self, tx: &mut WriteTx) {
        // pop_first always yields the smallest pending id. Every child id is
        // larger than its parents', so a node runs only after every dirty
        // parent has produced its delta, and never more than once per epoch.
        while let Some(id) = self.pending.pop_first() {
            self.touched.push(id);
            // Parents all have index < id, so split there: shared borrows
            // on the left, exclusive on the right.
            let (before, rest) = self.nodes.split_at_mut(id);
            let node = &mut rest[0];

            match &mut node.kind {
                // Source: the staged batch *becomes* this epoch's output.
                // The (empty) old output becomes next epoch's staging buffer.
                Kind::Source { staging } => std::mem::swap(staging, &mut node.out),
                Kind::Op(op) => {
                    let inputs: SmallVec<[&dyn Any; 8]> = node
                        .parents
                        .iter()
                        .map(|&p| before[p].out.as_any())
                        .collect();
                    op.eval(tx, &inputs, node.out.as_any_mut());
                }
            }

            if node.out.len() > 0 {
                self.pending.extend(node.children.iter());
            }
        }

        // Every consumed delta is now stale. Clear (keep allocations).
        for id in self.touched.drain(..) {
            self.nodes[id].out.clear();
        }
    }
}

// node wrapper types

struct UnaryNode<U>(U);
struct BinaryNode<B>(B);
struct NaryNode<N>(N);

impl<U: Unary> Node for UnaryNode<U> {
    #[inline(always)]
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any) {
        debug_assert_eq!(inputs.len(), 1);
        self.0.eval(
            tx,
            inputs[0].downcast_ref().unwrap(),
            out.downcast_mut().unwrap(),
        )
    }
}

impl<B: Binary> Node for BinaryNode<B> {
    #[inline(always)]
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any) {
        debug_assert_eq!(inputs.len(), 2);
        self.0.eval(
            tx,
            inputs[0].downcast_ref().unwrap(),
            inputs[1].downcast_ref().unwrap(),
            out.downcast_mut().unwrap(),
        )
    }
}

impl<N: Nary> Node for NaryNode<N> {
    #[inline(always)]
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any) {
        let typed_inputs: SmallVec<[&N::Takes; 8]> =
            inputs.iter().map(|i| i.downcast_ref().unwrap()).collect();
        self.0.eval(tx, &typed_inputs, out.downcast_mut().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bag, Diff};

    struct Double;
    impl Nary for Double {
        type Takes = Bag<u32>;
        type Gives = Bag<u32>;
        fn eval(&mut self, _: &mut WriteTx, batches: &[&Bag<u32>], out: &mut Bag<u32>) {
            for b in batches {
                for Diff(x, d) in b.iter() {
                    out.push(Diff(x * 2, *d));
                }
            }
        }
    }

    #[test]
    fn union_then_chain_runs_in_order_and_clears() {
        let mut g = OpDag::new();
        let a = g.source::<Bag<u32>>();
        let b = g.source::<Bag<u32>>();
        let d = g.nary(&[a, b], Double);
        let dd = g.unary(d, Double);

        g.give(a, &mut vec![Diff(1, 1)]);
        g.give(b, &mut vec![Diff(2, -1)]);
        g.run(&mut WriteTx::default());

        assert!(g.pending.is_empty());
        assert!(g.touched.is_empty());
        for n in &g.nodes {
            assert_eq!(n.out.len(), 0, "outputs are cleared after a run");
        }
        let _ = dd;
    }
}
