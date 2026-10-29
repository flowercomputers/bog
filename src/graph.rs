use crate::{Batch, Binary, Nary, Node, Unary, WriteTx};
use smallvec::{SmallVec, ToSmallVec};
use std::{any::Any, collections::BTreeSet, marker::PhantomData};

type OpDagNodeId = usize;

/// Typed, copyable handle to a node's output.
#[derive(Clone, Copy)]
pub struct NodeRef<T>(OpDagNodeId, PhantomData<fn() -> T>);

struct OpDagNode {
    /// `None` marks a source: no operator, fed by `give`.
    op: Option<Box<dyn Node>>,
    parents: SmallVec<[OpDagNodeId; 2]>,
    children: SmallVec<[OpDagNodeId; 4]>,
}

// TODO:
// better memory accounting—we should always know how many bytes are allocated
// for this OpDag, ditto for txn, ditto for readers?
pub struct OpDag {
    ops: Vec<OpDagNode>,
    /// outputs[id] is node `id`'s delta for the current epoch. Always allocated.
    outputs: Vec<Box<dyn Buffer>>,
    /// staging[id] is `Some` only for sources: the batch accumulated by `give`
    /// since the last `run`. Swapped into `outputs[id]` when the source runs.
    staging: Vec<Option<Box<dyn Buffer>>>,
    pending: BTreeSet<OpDagNodeId>,
    /// Cleared after every graph run
    touched_buf: Vec<OpDagNodeId>,
}

impl OpDag {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            outputs: Vec::new(),
            staging: Vec::new(),
            pending: BTreeSet::new(),
            touched_buf: Vec::new(),
        }
    }

    pub fn source<T: Send + 'static>(&mut self) -> NodeRef<T> {
        let id = self.push(None, &[], Box::new(Batch::<T>::new()));
        self.staging[id.0] = Some(Box::new(Batch::<T>::new()));
        id
    }

    pub fn unary<U: Unary + 'static>(&mut self, input: NodeRef<U::In>, op: U) -> NodeRef<U::Out>
    where
        U::Out: Send,
    {
        self.push(
            Some(Box::new(UnaryNode(op))),
            &[input.0],
            Box::new(Batch::<U::Out>::new()),
        )
    }

    pub fn binary<B: Binary + 'static>(
        &mut self,
        l: NodeRef<B::Left>,
        r: NodeRef<B::Right>,
        op: B,
    ) -> NodeRef<B::Out>
    where
        B::Out: Send,
    {
        self.push(
            Some(Box::new(BinaryNode(op))),
            &[l.0, r.0],
            Box::new(Batch::<B::Out>::new()),
        )
    }

    pub fn nary<N: Nary + 'static>(&mut self, inputs: &[NodeRef<N::In>], op: N) -> NodeRef<N::Out>
    where
        N::Out: Send,
    {
        assert!(!inputs.is_empty(), "nary node needs at least one input");
        let ids: SmallVec<[OpDagNodeId; 4]> = inputs.iter().map(|r| r.0).collect();
        self.push(
            Some(Box::new(NaryNode(op))),
            &ids,
            Box::new(Batch::<N::Out>::new()),
        )
    }

    fn push<T>(
        &mut self,
        op: Option<Box<dyn Node>>,
        parents: &[OpDagNodeId],
        out: Box<dyn Buffer>,
    ) -> NodeRef<T> {
        let id = self.ops.len();
        for &p in parents {
            debug_assert!(p < id, "parent ids always precede the child: {p} < {id}");
            self.ops[p].children.push(id);
        }
        self.ops.push(OpDagNode {
            op,
            parents: parents.to_smallvec(),
            children: Default::default(),
        });
        self.outputs.push(out);
        self.staging.push(None);
        NodeRef(id, PhantomData)
    }

    /// Stage a batch at a source. Drains `input` so the caller keeps its allocation.
    pub fn give<T: 'static>(&mut self, src: NodeRef<T>, input: &mut Batch<T>) {
        let buf = self.staging[src.0]
            .as_mut()
            .expect("NodeRef is not a source")
            .as_any_mut()
            .downcast_mut::<Batch<T>>()
            .expect("source type mismatch"); // unreachable given NodeRef<T>
        buf.append(input);
        if !buf.is_empty() {
            self.pending.insert(src.0);
        }
    }

    pub fn run(&mut self, tx: &mut WriteTx) {
        // todo: figure out a few things like...
        // directed feed (on-demand backfill just to out-of-date leaves)
        //
        // when an operator's closed epoch is lower than the db's epoch
        // we flush every historical batch from its most up to date stateful
        // upstream, et voila it's up to date!
        // so, maybe we need ops to know if their children are out of date?
        // or maybe, the ops keep track of their close in their own keyspace?
        // or... I don't know!
        // caveat, this doesn't really work with some op types I don't think,
        // so it's kind of a best-effort thing? or we mark nodes as restorable?

        // todo: we can still fuse unary chains as in Fold2

        // todo: keyspace wiring;
        // this adds another dependency layer beyond just the batches, so
        // we need to be careful about this!
        // or, we just add a keyspace param to eval that hands the node the
        // parents' keyspaces?

        // pop_first always yields the smallest pending id. Every child id is
        // larger than its parents', so a node runs only after every dirty
        // parent has produced its delta, and never more than once per epoch.
        while let Some(id) = self.pending.pop_first() {
            self.touched_buf.push(id);
            let OpDagNode {
                op,
                parents,
                children,
            } = &mut self.ops[id];

            match op {
                None => {
                    // Source: the staged batch *becomes* this epoch's output.
                    // The (empty) old output becomes next epoch's staging buffer.
                    let staged = self.staging[id]
                        .as_mut()
                        .expect("source has no staging buffer");
                    std::mem::swap(staged, &mut self.outputs[id]);
                }
                Some(op) => {
                    let (parents_out, rest) = self.outputs.split_at_mut(id);
                    let inputs: SmallVec<[&dyn Any; 8]> =
                        parents.iter().map(|&p| parents_out[p].as_any()).collect();
                    op.eval(tx, &inputs, rest[0].as_any_mut());
                }
            }

            // TODO: ops that always need to run... windowed, etc...
            // or maybe we assign them some kind of scheduler they subscribe to?
            // or we can just mark them always dirty!
            // maybe a "touch" method on node that takes no inputs but can output?
            if self.outputs[id].len() > 0 {
                self.pending.extend(children.iter());
            }
        }

        // Every consumed delta is now stale. Clear (keep allocations).
        for id in self.touched_buf.drain(..) {
            self.outputs[id].clear();
        }
    }
}

// buffer

pub trait Buffer: Any + Send {
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Send + 'static> Buffer for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn clear(&mut self) {
        Vec::clear(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
        let typed: SmallVec<[&Batch<N::In>; 8]> =
            inputs.iter().map(|i| i.downcast_ref().unwrap()).collect();
        self.0.eval(tx, &typed, out.downcast_mut().unwrap())
    }
}
