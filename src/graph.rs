use crate::{
    Bag, Binary, Buffer, Change, Mergeless, Nary, Node, Register, Tally, Unary, WriteTx,
    codec::{Codec, OrderedCodec},
    error::Result,
    merge::Merge,
    state::{Declare, Delivered, Filtered, KeyedState, Mapped, State, Stored},
    store::StoreBuilder,
    tx::KeyspaceId,
};
use smallvec::SmallVec;
use std::{
    any::{Any, type_name},
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    ops::Bound,
    sync::{Arc, Mutex, PoisonError},
};

/// A stable 64-bit hash (FNV-1a) for fingerprinting what derived state
/// was computed by. Unlike `std`'s hashers, its output is fixed across
/// builds and Rust releases, so it can be recorded on disk.
#[derive(Clone, Copy, Debug)]
pub struct Fingerprint(u64);

impl Default for Fingerprint {
    fn default() -> Self {
        Fingerprint(0xcbf2_9ce4_8422_2325)
    }
}

impl Fingerprint {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    pub fn write_u64(&mut self, n: u64) {
        self.write(&n.to_le_bytes());
    }
    /// Length-prefixed, so consecutive strings cannot run together.
    pub fn write_str(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.write(s.as_bytes());
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

type OpDagNodeId = usize;

/// Typed, copyable handle to a node's output. `B` is the buffer type the
/// node gives (`Bag<_>` or `Register<_, _>`), so wiring a child whose
/// `Takes` differs is a compile error. `S` is its [state](crate::state)
/// handle, `()` for none, which children read through
/// [`unary_with`](OpDag::unary_with) and its siblings.
pub struct NodeRef<B, S = ()>(OpDagNodeId, PhantomData<fn() -> (B, S)>);

impl<B, S> Clone for NodeRef<B, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B, S> Copy for NodeRef<B, S> {}

impl<B, S> NodeRef<B, S> {
    /// The same output, without its state, for operators that only take
    /// deltas.
    pub fn rows(self) -> NodeRef<B> {
        NodeRef(self.0, PhantomData)
    }
}

/// Fills a table source's delta: the table's changes this run, or in a
/// rebuild, its next chunk.
trait TableFeed: Send {
    /// This run's changes to the table, into `out`.
    fn pull(&mut self, tx: &mut WriteTx, out: &mut dyn Any);
    /// Start a rebuild: the table shows nothing until delivered.
    fn hide(&mut self);
    /// The next `n` rows as inserts, into `out`, showing them. Returns
    /// whether rows remain.
    fn deliver(&mut self, tx: &mut WriteTx, n: usize, out: &mut dyn Any) -> Result<bool>;
    /// Show the whole table again.
    fn show(&mut self);
}

struct Table<K, M: Merge> {
    id: KeyspaceId<K, M>,
    delivered: Arc<Mutex<Delivered<K>>>,
    after: Option<K>,
}

impl<K, M: Merge> Table<K, M> {
    fn set(&self, d: Delivered<K>) {
        *self
            .delivered
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = d;
    }
}

impl<K, M> TableFeed for Table<K, M>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge + Send,
    M::Value: Codec + Sync + Send + PartialEq,
    M::Diff: Send,
{
    fn pull(&mut self, tx: &mut WriteTx, out: &mut dyn Any) {
        let out: &mut Register<K, M> = out.downcast_mut().expect("table delta type");
        tx.keyspace(self.id).run_changes(out);
    }

    fn hide(&mut self) {
        self.after = None;
        self.set(Delivered::UpTo {
            old: None,
            new: None,
        });
    }

    fn deliver(&mut self, tx: &mut WriteTx, n: usize, out: &mut dyn Any) -> Result<bool> {
        let out: &mut Register<K, M> = out.downcast_mut().expect("table delta type");
        let lower = match &self.after {
            Some(k) => Bound::Excluded(k.clone()),
            None => Bound::Unbounded,
        };
        let mut rows = 0;
        let mut last = None;
        for row in tx
            .keyspace(self.id)
            .range((lower, Bound::Unbounded))
            .take(n)
        {
            let (k, v) = row?;
            out.absorb_one(k.clone().into_owned(), Change::Insert(v.into_owned()));
            last = Some(k.into_owned());
            rows += 1;
        }
        let old = self.after.clone();
        if last.is_some() {
            self.after = last;
        }
        self.set(Delivered::UpTo {
            old,
            new: self.after.clone(),
        });
        Ok(rows == n)
    }

    fn show(&mut self) {
        self.set(Delivered::All);
    }
}

/// A stored multiset as a source: its counts' changes as a bag.
struct Counts<T> {
    id: KeyspaceId<T, Tally>,
    after: Option<T>,
    /// Scratch, reused across runs.
    changes: Register<T, Tally>,
}

impl<T> TableFeed for Counts<T>
where
    T: OrderedCodec + Clone + Send + Sync + 'static,
{
    fn pull(&mut self, tx: &mut WriteTx, out: &mut dyn Any) {
        let out: &mut Bag<T> = out.downcast_mut().expect("multiset delta type");
        tx.keyspace(self.id).run_changes(&mut self.changes);
        for (row, change) in self.changes.drain() {
            // a tally's change is the amount added
            let Change::Insert(by) = change else {
                unreachable!("a tally holds amounts")
            };
            out.push(row, by);
        }
    }

    // nothing reads a multiset's state through the graph
    fn hide(&mut self) {
        self.after = None;
    }

    fn deliver(&mut self, tx: &mut WriteTx, n: usize, out: &mut dyn Any) -> Result<bool> {
        let out: &mut Bag<T> = out.downcast_mut().expect("multiset delta type");
        let lower = match &self.after {
            Some(k) => Bound::Excluded(k.clone()),
            None => Bound::Unbounded,
        };
        let mut rows = 0;
        for row in tx
            .keyspace(self.id)
            .range((lower, Bound::Unbounded))
            .take(n)
        {
            let (k, count) = row?;
            out.push(k.clone().into_owned(), *count);
            self.after = Some(k.into_owned());
            rows += 1;
        }
        Ok(rows == n)
    }

    fn show(&mut self) {}
}

enum Kind {
    /// Fed by `give`, or by a table's changes; `staging` accumulates
    /// between runs and is swapped into `out` when the source runs.
    Source {
        staging: Box<dyn Buffer>,
        table: Option<Box<dyn TableFeed>>,
    },
    Op(Box<dyn Node>),
}

struct OpDagNode {
    kind: Kind,
    /// The source's buffer type, or the operator's type.
    name: &'static str,
    parents: SmallVec<[OpDagNodeId; 2]>,
    children: SmallVec<[OpDagNodeId; 4]>,
    /// This node's delta for the current epoch. Always allocated; cleared
    /// after every run.
    out: Box<dyn Buffer>,
    /// Its state handle, cloned for children that read it.
    state: Box<dyn Any + Send>,
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
    /// Mixed into the fingerprint; see [`OpDag::version`].
    version: u64,
    /// Every keyspace a node owns, by id, with the node.
    owned: BTreeMap<u64, OpDagNodeId>,
    /// How to declare them, until [`declare`](Self::declare) takes them.
    declares: Vec<Declare>,
    /// Table sources, in node order.
    tables: Vec<OpDagNodeId>,
    /// Sources fed by `give`, whose history a rebuild cannot read.
    given: usize,
}

impl OpDag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the version mixed into the [`fingerprint`](Self::fingerprint).
    /// Changing it rebuilds derived state at the next open, which is how to
    /// make a change the fingerprint cannot see, such as to a closure's
    /// body or a captured constant, take effect on existing data.
    pub fn version(&mut self, version: u64) -> &mut Self {
        self.version = version;
        self
    }

    /// A source fed by [`give`](Self::give), for rows the store does not
    /// keep, such as events.
    pub fn source<B: Buffer>(&mut self) -> NodeRef<B> {
        self.given += 1;
        self.push(
            Kind::Source {
                staging: Box::new(B::empty()),
                table: None,
            },
            type_name::<B>(),
            &[],
            Box::new(B::empty()),
            Box::new(()),
            Vec::new(),
        )
    }

    /// A base table: the application writes keyspace `id`, and the node
    /// gives each run's changes to it. Its state is the table.
    pub fn table<K, M>(&mut self, id: KeyspaceId<K, M>) -> NodeRef<Register<K, M>, Stored<K, M>>
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge + Send,
        M::Value: Codec + Sync + Send + PartialEq,
        M::Diff: Send,
    {
        let delivered = Arc::new(Mutex::new(Delivered::All));
        let feed = Table {
            id,
            delivered: delivered.clone(),
            after: None,
        };
        let node = self.push(
            Kind::Source {
                staging: Box::new(Register::<K, M>::empty()),
                table: Some(Box::new(feed)),
            },
            type_name::<Register<K, M>>(),
            &[],
            Box::new(Register::<K, M>::empty()),
            Box::new(Stored::table(id, delivered)),
            vec![Declare::base(id)],
        );
        self.tables.push(node.0);
        node
    }

    /// A stored multiset: the application writes each row's count to
    /// keyspace `id`, and the node gives each run's changes in count.
    pub fn multiset<T>(&mut self, id: KeyspaceId<T, Tally>) -> NodeRef<Bag<T>>
    where
        T: OrderedCodec + Clone + Send + Sync + 'static,
    {
        let feed = Counts {
            id,
            after: None,
            changes: Register::new(),
        };
        let node = self.push(
            Kind::Source {
                staging: Box::new(Bag::<T>::empty()),
                table: Some(Box::new(feed)),
            },
            type_name::<Bag<T>>(),
            &[],
            Box::new(Bag::<T>::empty()),
            Box::new(()),
            vec![Declare::base(id)],
        );
        self.tables.push(node.0);
        node
    }

    pub fn unary<U: Unary + 'static, S>(
        &mut self,
        input: NodeRef<U::Takes, S>,
        op: U,
    ) -> NodeRef<U::Gives, U::State> {
        let (state, owns) = (Box::new(op.state()), op.keyspaces());
        self.push(
            Kind::Op(Box::new(UnaryNode(op))),
            type_name::<U>(),
            &[input.0],
            Box::new(U::Gives::empty()),
            state,
            owns,
        )
    }

    /// [`unary`](Self::unary), making the operator with its input's
    /// state handle.
    pub fn unary_with<U: Unary + 'static, S: Clone + 'static>(
        &mut self,
        input: NodeRef<U::Takes, S>,
        make: impl FnOnce(S) -> U,
    ) -> NodeRef<U::Gives, U::State> {
        let op = make(self.state(input));
        self.unary(input, op)
    }

    pub fn binary<B: Binary + 'static, LS, RS>(
        &mut self,
        l: NodeRef<B::Left, LS>,
        r: NodeRef<B::Right, RS>,
        op: B,
    ) -> NodeRef<B::Gives, B::State> {
        let (state, owns) = (Box::new(op.state()), op.keyspaces());
        self.push(
            Kind::Op(Box::new(BinaryNode(op))),
            type_name::<B>(),
            &[l.0, r.0],
            Box::new(B::Gives::empty()),
            state,
            owns,
        )
    }

    /// [`binary`](Self::binary), making the operator with its inputs'
    /// state handles.
    pub fn binary_with<B: Binary + 'static, LS: Clone + 'static, RS: Clone + 'static>(
        &mut self,
        l: NodeRef<B::Left, LS>,
        r: NodeRef<B::Right, RS>,
        make: impl FnOnce(LS, RS) -> B,
    ) -> NodeRef<B::Gives, B::State> {
        let op = make(self.state(l), self.state(r));
        self.binary(l, r, op)
    }

    /// One node over any number of same-typed inputs. Several inputs are
    /// unioned, which is what `Nary` means.
    pub fn nary<N: Nary + 'static, S>(
        &mut self,
        inputs: &[NodeRef<N::Takes, S>],
        op: N,
    ) -> NodeRef<N::Gives, N::State> {
        assert!(!inputs.is_empty(), "nary node needs at least one input");
        let ids: SmallVec<[OpDagNodeId; 4]> = inputs.iter().map(|r| r.0).collect();
        let (state, owns) = (Box::new(op.state()), op.keyspaces());
        self.push(
            Kind::Op(Box::new(NaryNode(op))),
            type_name::<N>(),
            &ids,
            Box::new(N::Gives::empty()),
            state,
            owns,
        )
    }

    /// [`nary`](Self::nary), making the operator with its inputs' state
    /// handles, in input order.
    pub fn nary_with<N: Nary + 'static, S: Clone + 'static>(
        &mut self,
        inputs: &[NodeRef<N::Takes, S>],
        make: impl FnOnce(Vec<S>) -> N,
    ) -> NodeRef<N::Gives, N::State> {
        let states = inputs.iter().map(|&i| self.state(i)).collect();
        let op = make(states);
        self.nary(inputs, op)
    }

    /// A view of a register with every value mapped by `f`. Stores
    /// nothing: its children read its state through `f`.
    pub fn map_values<S, W>(
        &mut self,
        input: NodeRef<S::Delta, S>,
        f: impl Fn(&S::Value) -> W + Send + Sync + 'static,
    ) -> NodeRef<Register<S::Key, Mergeless<W>>, Mapped<S, W>>
    where
        S: KeyedState,
        W: Clone + PartialEq + Send + 'static,
    {
        self.unary_with(input, |parent| MapValues::new(parent, Arc::new(f)))
    }

    /// A view of a register keeping only the values `p` accepts.
    pub fn filter_values<S>(
        &mut self,
        input: NodeRef<S::Delta, S>,
        p: impl Fn(&S::Value) -> bool + Send + Sync + 'static,
    ) -> NodeRef<Register<S::Key, Mergeless<S::Value>>, Filtered<S>>
    where
        S: KeyedState,
        S::Value: PartialEq,
    {
        self.unary_with(input, |parent| FilterValues::new(parent, Arc::new(p)))
    }

    /// The state handle of `r`'s node, as children read it.
    pub fn state<B, S: Clone + 'static>(&self, r: NodeRef<B, S>) -> S {
        self.nodes[r.0]
            .state
            .downcast_ref::<S>()
            .expect("NodeRef state type matches its node")
            .clone()
    }

    fn push<B, S>(
        &mut self,
        kind: Kind,
        name: &'static str,
        parents: &[OpDagNodeId],
        out: Box<dyn Buffer>,
        state: Box<dyn Any + Send>,
        owns: Vec<Declare>,
    ) -> NodeRef<B, S> {
        let id = self.nodes.len();
        for &p in parents {
            debug_assert!(p < id, "parent ids always precede the child: {p} < {id}");
            self.nodes[p].children.push(id);
        }
        for d in owns {
            if let Some(other) = self.owned.insert(d.id, id) {
                panic!(
                    "keyspace {} is written by node {other} ({}) and node {id} ({name})",
                    d.id, self.nodes[other].name
                );
            }
            self.declares.push(d);
        }
        self.nodes.push(OpDagNode {
            kind,
            name,
            parents: parents.into(),
            children: Default::default(),
            out,
            state,
        });
        NodeRef(id, PhantomData)
    }

    /// Declare every keyspace the graph's nodes own to `b`: tables as
    /// base keyspaces, the rest as derived. Takes them, so call it once.
    pub(crate) fn declare(&mut self, mut b: StoreBuilder) -> StoreBuilder {
        for d in self.declares.drain(..) {
            b = (d.apply)(b);
        }
        b
    }

    /// Whether node-owned keyspace `id` is derived, if a node owns it.
    pub(crate) fn owner_is_table(&self, id: u64) -> Option<bool> {
        let node = *self.owned.get(&id)?;
        Some(self.tables.contains(&node))
    }

    /// Whether some source is fed by `give`, so its history is not kept.
    pub(crate) fn has_given_sources(&self) -> bool {
        self.given > 0
    }

    /// A fingerprint of the graph's shape: every node's type, its inputs in
    /// order, the keyspaces it owns, and whatever its operator adds.
    /// Derived state recorded under one fingerprint is rebuilt when the
    /// graph opens under another.
    ///
    /// Closures appear only by type name, which names where they are
    /// defined but not their bodies, and nor are values they capture. Bump
    /// [`version`](Self::version) when only those change.
    pub fn fingerprint(&self) -> u64 {
        let mut fp = Fingerprint::new();
        fp.write_u64(self.version);
        fp.write_u64(self.nodes.len() as u64);
        let mut owned: Vec<Vec<u64>> = vec![Vec::new(); self.nodes.len()];
        for (&ks, &node) in &self.owned {
            owned[node].push(ks);
        }
        for (node, owns) in self.nodes.iter().zip(&owned) {
            fp.write_str(node.name);
            fp.write_u64(node.parents.len() as u64);
            for &p in &node.parents {
                fp.write_u64(p as u64);
            }
            fp.write_u64(owns.len() as u64);
            for &ks in owns {
                fp.write_u64(ks);
            }
            if let Kind::Op(op) = &node.kind {
                op.fingerprint(&mut fp);
            }
        }
        fp.finish()
    }

    /// Stage input at a source. Drains `input` so the caller keeps its
    /// allocation. Bags append; registers compose per key.
    pub fn give<B: Buffer>(&mut self, src: NodeRef<B>, input: &mut B) {
        let Kind::Source { staging, table } = &mut self.nodes[src.0].kind else {
            panic!("NodeRef is not a source");
        };
        assert!(table.is_none(), "tables are fed by writing them");
        let staging: &mut B = staging
            .as_any_mut()
            .downcast_mut()
            .expect("source type mismatch"); // unreachable given NodeRef<B>
        staging.absorb(input);
        if staging.len() > 0 {
            self.pending.insert(src.0);
        }
    }

    /// Forget input staged for a transaction that will not commit.
    pub(crate) fn abort(&mut self) {
        self.pending.clear();
        for node in &mut self.nodes {
            if let Kind::Source { staging, .. } = &mut node.kind {
                staging.clear();
            }
            node.out.clear();
        }
    }

    /// Start a rebuild of derived state from the tables: each shows
    /// nothing until [`rebuild_step`](Self::rebuild_step) delivers it.
    pub(crate) fn rebuild_begin(&mut self) {
        for &t in &self.tables {
            if let Kind::Source {
                table: Some(feed), ..
            } = &mut self.nodes[t].kind
            {
                feed.hide();
            }
        }
    }

    /// Deliver the next `chunk` rows of the first table with rows left
    /// and run the graph over them. Returns whether any table has rows
    /// left; once none has, every table shows whole again.
    pub(crate) fn rebuild_step(
        &mut self,
        tx: &mut WriteTx,
        chunk: usize,
        next: &mut usize,
    ) -> Result<bool> {
        while let Some(&t) = self.tables.get(*next) {
            let Kind::Source {
                staging,
                table: Some(feed),
            } = &mut self.nodes[t].kind
            else {
                unreachable!("tables are table sources");
            };
            let more = feed.deliver(tx, chunk, staging.as_any_mut())?;
            if !staging.is_empty() {
                self.pending.insert(t);
            }
            self.run(tx);
            tx.failure().map_or(Ok(()), Err)?;
            if more {
                return Ok(true);
            }
            if let Kind::Source {
                table: Some(feed), ..
            } = &mut self.nodes[t].kind
            {
                feed.show();
            }
            *next += 1;
            if *next < self.tables.len() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Run every dirty node once, in dependency order: first the tables
    /// written since the last run, then everything downstream of them and
    /// of the sources given input. Ends the run, so the next one's "old"
    /// state is what this one left.
    ///
    /// Open questions, in rough priority:
    /// - fusing unary chains as in fold2.
    /// - ops that always need to run (windowed, etc.): a scheduler they
    ///   subscribe to, or mark them always dirty, or a `touch` method that
    ///   takes no inputs but can output.
    pub fn run(&mut self, tx: &mut WriteTx) {
        for &t in &self.tables {
            if let Kind::Source {
                staging,
                table: Some(feed),
            } = &mut self.nodes[t].kind
            {
                feed.pull(tx, staging.as_any_mut());
                if !staging.is_empty() {
                    self.pending.insert(t);
                }
            }
        }
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
                Kind::Source { staging, .. } => std::mem::swap(staging, &mut node.out),
                Kind::Op(op) => {
                    let inputs: SmallVec<[&dyn Any; 8]> = node
                        .parents
                        .iter()
                        .map(|&p| before[p].out.as_any())
                        .collect();
                    op.eval(tx, &inputs, node.out.as_any_mut());
                }
            }
            node.out.consolidate();

            // a failed tx will not commit, so nothing downstream needs to run
            if tx.failure().is_some() {
                self.pending.clear();
                break;
            }

            if !node.out.is_empty() {
                self.pending.extend(node.children.iter());
            }
        }

        // Every consumed delta is now stale. Clear (keep allocations).
        for id in self.touched.drain(..) {
            self.nodes[id].out.clear();
        }
        tx.end_run();
    }
}

// node wrapper types

struct UnaryNode<U>(U);
struct BinaryNode<B>(B);
struct NaryNode<N>(N);

impl<U: Unary> Node for UnaryNode<U> {
    fn fingerprint(&self, fp: &mut Fingerprint) {
        self.0.fingerprint(fp)
    }
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
    fn fingerprint(&self, fp: &mut Fingerprint) {
        self.0.fingerprint(fp)
    }
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
    fn fingerprint(&self, fp: &mut Fingerprint) {
        self.0.fingerprint(fp)
    }
    #[inline(always)]
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any) {
        let typed_inputs: SmallVec<[&N::Takes; 8]> =
            inputs.iter().map(|i| i.downcast_ref().unwrap()).collect();
        self.0.eval(tx, &typed_inputs, out.downcast_mut().unwrap())
    }
}

/// The node behind [`OpDag::map_values`].
struct MapValues<S: State, W> {
    view: Mapped<S, W>,
    f: Arc<dyn Fn(&S::Value) -> W + Send + Sync>,
}

impl<S: State, W> MapValues<S, W> {
    fn new(parent: S, f: Arc<dyn Fn(&S::Value) -> W + Send + Sync>) -> Self {
        MapValues {
            view: Mapped::new(parent, f.clone()),
            f,
        }
    }
}

impl<S, W> Unary for MapValues<S, W>
where
    S: KeyedState,
    W: Clone + PartialEq + Send + 'static,
{
    type Takes = S::Delta;
    type Gives = Register<S::Key, Mergeless<W>>;
    type State = Mapped<S, W>;

    fn state(&self) -> Self::State {
        self.view.clone()
    }

    fn eval(&mut self, _: &mut WriteTx, batch: &S::Delta, out: &mut Self::Gives) {
        S::changes(batch, &mut |k, old, new| {
            let (old, new) = (old.map(|v| (self.f)(v)), new.map(|v| (self.f)(v)));
            if let Some(c) = change(old, new) {
                out.absorb_one(k.clone(), c);
            }
        });
    }
}

/// The node behind [`OpDag::filter_values`].
struct FilterValues<S: State> {
    view: Filtered<S>,
    p: Arc<dyn Fn(&S::Value) -> bool + Send + Sync>,
}

impl<S: State> FilterValues<S> {
    fn new(parent: S, p: Arc<dyn Fn(&S::Value) -> bool + Send + Sync>) -> Self {
        FilterValues {
            view: Filtered::new(parent, p.clone()),
            p,
        }
    }
}

impl<S> Unary for FilterValues<S>
where
    S: KeyedState,
    S::Value: PartialEq,
{
    type Takes = S::Delta;
    type Gives = Register<S::Key, Mergeless<S::Value>>;
    type State = Filtered<S>;

    fn state(&self) -> Self::State {
        self.view.clone()
    }

    fn eval(&mut self, _: &mut WriteTx, batch: &S::Delta, out: &mut Self::Gives) {
        S::changes(batch, &mut |k, old, new| {
            let keep = |v: Option<&S::Value>| v.filter(|v| (self.p)(v)).cloned();
            if let Some(c) = change(keep(old), keep(new)) {
                out.absorb_one(k.clone(), c);
            }
        });
    }
}

/// The change from `old` to `new`, if they differ.
fn change<T: Clone + PartialEq + Send + 'static>(
    old: Option<T>,
    new: Option<T>,
) -> Option<Change<Mergeless<T>>> {
    match (old, new) {
        (None, None) => None,
        (Some(o), Some(n)) if o == n => None,
        (None, Some(n)) => Some(Change::Insert(n)),
        (Some(o), None) => Some(Change::Delete(o)),
        (Some(o), Some(n)) => Some(Change::Replace(o, n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bag;
    use crate::store::{DurabilityPolicy, StoreBuilder};

    struct Double;
    impl Nary for Double {
        type Takes = Bag<u32>;
        type Gives = Bag<u32>;
        type State = ();
        fn state(&self) {}
        fn eval(&mut self, _: &mut WriteTx, batches: &[&Bag<u32>], out: &mut Bag<u32>) {
            for b in batches {
                for (x, d) in b.diffs() {
                    out.push(x * 2, d);
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

        let mut in_a = Bag::new();
        in_a.push(1, 1);
        let mut in_b = Bag::new();
        in_b.push(2, -1);
        g.give(a, &mut in_a);
        g.give(b, &mut in_b);
        let store = StoreBuilder::new(DurabilityPolicy::NONE).build().unwrap();
        g.run(&mut store.begin().unwrap());

        assert!(g.pending.is_empty());
        assert!(g.touched.is_empty());
        for n in &g.nodes {
            assert_eq!(n.out.len(), 0, "outputs are cleared after a run");
        }
        let _ = dd;
    }
}
