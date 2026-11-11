//! Stream kinds: [`Stream`], [`Keyed`], [`Grouped`], the typed heads
//! views are declared over, and their combinators.
//!
//! - [`Stream<T>`]: a bag of `T`.
//! - [`Grouped<K, V>`]: `(K, V)` rows, any number per key.
//! - [`Keyed<K, V>`]: `(K, V)` rows, at most one per key: a register.
//!   [`Value<V>`] is the register with the one key `()`.
//!
//! The two keyed kinds are one type, [`Rows`], under a kind marker, so
//! combinators that keep the kind are written once. `map` on a keyed or
//! grouped stream maps the *value* and keeps the key; regrouping is
//! `key_by`; leaving the keyed world is `entries`. A [`Keyed`] is
//! accepted wherever a [`Grouped`] is, never the reverse.
//!
//! Every kind wraps a [`Flow`]: a [`Chain`] of linear stages from a root
//! graph node. Linear combinators extend the chain; whatever consumes it
//! builds the whole chain as one fused graph node, shared by every
//! consumer of that stream. Stateful combinators build a graph node
//! with keyspaces named after the stream (`orders.sum`) and return a
//! stream rooted there.
//!
//! An operator that reads its inputs' whole state (a join, a reduction)
//! reads what already exists where it can: a table, or the output of an
//! operator that keeps a register or an arrangement. A chain with no
//! such state behind it is arranged first, which costs a keyspace of its
//! rows.

use std::{cell::OnceCell, marker::PhantomData, rc::Rc, sync::Arc};

use crate::{
    dataflow::graph::{NodeRef, OpDag},
    dataflow::state::{Arranged, Arrangement, Stored},
    dataflow::{Bag, Emit, FusedLinears, Ident, Register, Then},
    ops::{Arrange, DistinctCounts, Distinctly, Flatten, linear::ReKeyWith},
    store::codec::{Codec, OrderedCodec},
    store::merge::Merge,
    store::merge::{Mergeless, Tally},
    store::txn::KeyspaceId,
};

use crate::{
    dataflow::Shared,
    schema::read::{FilterMappedReader, KeyedReader, RegisterReader},
};

pub(crate) mod dynstate;

/// Everything the operators' stream methods (in [`ops`](crate::ops))
/// build with: the stream kinds and their internals, and what this
/// module itself imports.
pub(crate) mod prelude {
    pub(crate) use super::dynstate::*;
    pub(crate) use super::*;
    pub(crate) use crate::{
        dataflow::{Delta, Register, graph::NodeRef, state::Stored},
        ops::*,
        ops::{aggregate::*, linear::*},
        schema::read::RegisterReader,
        store::{
            merge::{Mergeless, Tally},
            txn::KeyspaceId,
        },
    };
    pub(crate) use std::sync::Arc;
}

use dynstate::{DynKeyed, DynState, FilterMapValues, FilterMapped, ValueMap};

/// What a stream can carry: ordered, encodable, and sendable. Keys and
/// values alike.
pub trait Data: OrderedCodec + Codec + Clone + Send + Sync + 'static {}
impl<T: OrderedCodec + Codec + Clone + Send + Sync + 'static> Data for T {}

/// What [`sum`](Grouped::sum) can total: the signed integers.
pub trait Additive: Data + Merge<Diff = Self, Value = Self> {}
impl<T: Data + Merge<Diff = T, Value = T>> Additive for T {}

// ---------------------------------------------------------------------------
// Chain: a fused linear path from a root node
// ---------------------------------------------------------------------------

/// The graph node a chain starts from, giving rows of `T`.
pub struct Root<T> {
    pub(crate) node: RootNode<T>,
}

pub(crate) enum RootNode<T> {
    Node(NodeRef<Bag<T>>),
    /// Made when first needed: the rows of a register node, which costs
    /// a node of its own.
    Lazy(Rc<LazyNode<T>>),
    /// Several nodes, unioned: a consumer that takes any number of
    /// inputs takes them all; one that takes one gets a node made of
    /// them when first needed.
    Union(Rc<UnionNode<T>>),
}

pub(crate) struct LazyNode<T> {
    pub(crate) made: OnceCell<NodeRef<Bag<T>>>,
    pub(crate) make: Box<dyn Fn(&mut OpDag) -> NodeRef<Bag<T>>>,
}

pub(crate) struct UnionNode<T> {
    pub(crate) inputs: Vec<NodeRef<Bag<T>>>,
    pub(crate) made: OnceCell<NodeRef<Bag<T>>>,
}

impl<T> Clone for Root<T> {
    fn clone(&self) -> Self {
        Root {
            node: match &self.node {
                RootNode::Node(n) => RootNode::Node(*n),
                RootNode::Lazy(l) => RootNode::Lazy(l.clone()),
                RootNode::Union(u) => RootNode::Union(u.clone()),
            },
        }
    }
}

impl<T: 'static> Root<T> {
    pub(crate) fn at(node: NodeRef<Bag<T>>) -> Self {
        Root {
            node: RootNode::Node(node),
        }
    }

    pub(crate) fn lazy(make: impl Fn(&mut OpDag) -> NodeRef<Bag<T>> + 'static) -> Self {
        Root {
            node: RootNode::Lazy(Rc::new(LazyNode {
                made: OnceCell::new(),
                make: Box::new(make),
            })),
        }
    }

    pub(crate) fn union(inputs: Vec<NodeRef<Bag<T>>>) -> Self {
        Root {
            node: RootNode::Union(Rc::new(UnionNode {
                inputs,
                made: OnceCell::new(),
            })),
        }
    }

    pub(crate) fn node(&self, g: &mut OpDag) -> NodeRef<Bag<T>>
    where
        T: Data,
    {
        match &self.node {
            RootNode::Node(n) => *n,
            RootNode::Lazy(l) => *l.made.get_or_init(|| (l.make)(g)),
            RootNode::Union(u) => *u
                .made
                .get_or_init(|| g.nary(&u.inputs, FusedLinears::new(Ident))),
        }
    }

    pub(crate) fn inputs(&self, g: &mut OpDag) -> Vec<NodeRef<Bag<T>>> {
        match &self.node {
            RootNode::Node(n) => vec![*n],
            RootNode::Lazy(l) => vec![*l.made.get_or_init(|| (l.make)(g))],
            RootNode::Union(u) => u.inputs.clone(),
        }
    }
}

/// A linear path from a root node to a stream head. Attaching a consumer
/// builds the entire path as one fused node, however many stages it
/// holds; the chain itself is a cheap, cloneable description.
pub trait Chain: Clone + 'static {
    /// The rows the root gives.
    type In: Data;
    /// The rows that arrive at the end of the chain.
    type Out: Data;
    /// The whole path as one fused stage.
    type Lin: Emit<Self::In, Emits = Self::Out> + Clone + Send + 'static;

    fn root(&self) -> &Root<Self::In>;
    fn linear(&self) -> Self::Lin;
    /// The node giving the chain's rows, built in `g` if the chain has
    /// stages.
    fn attach(&self, g: &mut OpDag) -> NodeRef<Bag<Self::Out>>;
    /// The nodes giving the chain's rows, for a consumer that unions
    /// its inputs.
    fn inputs(&self, g: &mut OpDag) -> Vec<NodeRef<Bag<Self::Out>>> {
        vec![self.attach(g)]
    }
}

impl<T: Data> Chain for Root<T> {
    type In = T;
    type Out = T;
    type Lin = Ident;
    fn root(&self) -> &Root<T> {
        self
    }
    fn linear(&self) -> Ident {
        Ident
    }
    fn attach(&self, g: &mut OpDag) -> NodeRef<Bag<T>> {
        self.node(g)
    }
    fn inputs(&self, g: &mut OpDag) -> Vec<NodeRef<Bag<T>>> {
        Root::inputs(self, g)
    }
}

/// A chain extended by one linear stage.
#[derive(Clone)]
pub struct Ext<C, Y>(C, Y);

impl<C, Y> Chain for Ext<C, Y>
where
    C: Chain,
    Y: Emit<C::Out> + Clone + Send + 'static,
    Y::Emits: Data,
{
    type In = C::In;
    type Out = Y::Emits;
    type Lin = Then<C::Lin, Y>;
    fn root(&self) -> &Root<C::In> {
        self.0.root()
    }
    fn linear(&self) -> Self::Lin {
        Then(self.0.linear(), self.1.clone())
    }
    fn attach(&self, g: &mut OpDag) -> NodeRef<Bag<Self::Out>> {
        let inputs = self.root().inputs(g);
        g.nary(&inputs, FusedLinears::new(self.linear()))
    }
}

// ---------------------------------------------------------------------------
// Flow: a chain plus the bookkeeping the schema layer needs
// ---------------------------------------------------------------------------

/// A [`Chain`] with its build and its name. See the module docs.
pub struct Flow<T, C> {
    pub(crate) chain: C,
    pub(crate) build: Shared,
    /// The name of this flow's root: a source or view name, or an
    /// operator name like `transfers.sum`. Operator keyspaces are named
    /// from it.
    pub(crate) label: Arc<str>,
    /// The node this chain was built as, shared by its consumers.
    pub(crate) attached: Rc<OnceCell<NodeRef<Bag<T>>>>,
}

impl<T, C: Clone> Clone for Flow<T, C> {
    fn clone(&self) -> Self {
        Flow {
            chain: self.chain.clone(),
            build: self.build.clone(),
            label: self.label.clone(),
            attached: self.attached.clone(),
        }
    }
}

impl<T: Data> Flow<T, Root<T>> {
    pub(crate) fn at_root(root: Root<T>, build: Shared, label: Arc<str>) -> Self {
        Flow {
            chain: root,
            build,
            label,
            attached: Rc::new(OnceCell::new()),
        }
    }
}

impl<T: Data, C: Chain<Out = T>> Flow<T, C> {
    pub(crate) fn extend<Y>(&self, y: Y) -> Flow<Y::Emits, Ext<C, Y>>
    where
        Y: Emit<T> + Clone + Send + 'static,
        Y::Emits: Data,
    {
        Flow {
            chain: Ext(self.chain.clone(), y),
            build: self.build.clone(),
            label: self.label.clone(),
            attached: Rc::new(OnceCell::new()),
        }
    }

    /// The node giving this flow's rows.
    pub(crate) fn attach(&self) -> NodeRef<Bag<T>> {
        *self.attached.get_or_init(|| {
            let mut b = self.build.borrow_mut();
            self.chain.attach(&mut b.graph)
        })
    }

    /// The nodes giving this flow's rows, for a consumer that unions
    /// its inputs: a union root's members, else the one node.
    pub(crate) fn inputs(&self) -> Vec<NodeRef<Bag<T>>> {
        match self.attached.get() {
            Some(n) => vec![*n],
            None => {
                let mut b = self.build.borrow_mut();
                self.chain.inputs(&mut b.graph)
            }
        }
    }

    /// A fresh, unique operator name under this flow's label.
    pub(crate) fn name(&self, kind: &str, other: Option<&str>) -> String {
        let base = match other {
            Some(o) => format!("{}+{o}.{kind}", self.label),
            None => format!("{}.{kind}", self.label),
        };
        self.build.borrow_mut().unique(&base)
    }
}

// ---------------------------------------------------------------------------
// Feeds: what a stream can be connected as
// ---------------------------------------------------------------------------

/// Where the rows of a keyed stream already live, when a stateful
/// operator keeps them: what joins and reductions read, and what views
/// adopt.
pub struct KeyedOrigin<K: Data, V: Data> {
    /// The rows grouped by key, to read.
    pub(crate) state: DynState<K, V>,
    pub(crate) kind: OriginKind<K, V>,
}

pub(crate) enum OriginKind<K: Data, V: Data> {
    /// A register: a table, an operator's one-value-per-key output, or
    /// a filtered or mapped view of one.
    Register {
        /// The register, read by key.
        keyed: DynKeyed<K, V>,
        /// Its changes, as a register.
        changes: Changes<K, V>,
        /// Reads it from a snapshot or a transaction.
        reader: Arc<dyn KeyedReader<K, V>>,
    },
    /// An arrangement an operator keeps.
    Arranged {
        id: Arrangement<K, V>,
        totals: Option<KeyspaceId<K, Tally>>,
    },
    /// The counts a [`DistinctRows`] node keeps.
    Distinct { id: DistinctCounts<K, V> },
}

impl<K: Data, V: Data> Clone for KeyedOrigin<K, V> {
    fn clone(&self) -> Self {
        KeyedOrigin {
            state: self.state.clone(),
            kind: match &self.kind {
                OriginKind::Register {
                    keyed,
                    changes,
                    reader,
                } => OriginKind::Register {
                    keyed: keyed.clone(),
                    changes: changes.clone(),
                    reader: reader.clone(),
                },
                OriginKind::Arranged { id, totals } => OriginKind::Arranged {
                    id: *id,
                    totals: *totals,
                },
                OriginKind::Distinct { id } => OriginKind::Distinct { id: *id },
            },
        }
    }
}

/// A register node's changes, made when first needed: a filtered or
/// mapped view of a register costs a node only if something consumes
/// its changes.
pub(crate) struct Changes<K: Data, V: Data>(Rc<LazyChanges<K, V>>);

pub(crate) struct LazyChanges<K: Data, V: Data> {
    pub(crate) made: OnceCell<NodeRef<Register<K, Mergeless<V>>>>,
    pub(crate) make: Box<dyn Fn(&mut OpDag) -> NodeRef<Register<K, Mergeless<V>>>>,
}

impl<K: Data, V: Data> Clone for Changes<K, V> {
    fn clone(&self) -> Self {
        Changes(self.0.clone())
    }
}

impl<K: Data, V: Data> Changes<K, V> {
    pub(crate) fn at(node: NodeRef<Register<K, Mergeless<V>>>) -> Self {
        let made = OnceCell::new();
        let _ = made.set(node);
        Changes(Rc::new(LazyChanges {
            made,
            make: Box::new(move |_| node),
        }))
    }

    pub(crate) fn lazy(
        make: impl Fn(&mut OpDag) -> NodeRef<Register<K, Mergeless<V>>> + 'static,
    ) -> Self {
        Changes(Rc::new(LazyChanges {
            made: OnceCell::new(),
            make: Box::new(make),
        }))
    }

    pub(crate) fn get(&self, g: &mut OpDag) -> NodeRef<Register<K, Mergeless<V>>> {
        *self.0.made.get_or_init(|| (self.0.make)(g))
    }
}

impl<K: Data, V: Data> KeyedOrigin<K, V> {
    /// A register's state, changes and reader.
    pub(crate) fn register(
        keyed: DynKeyed<K, V>,
        changes: Changes<K, V>,
        reader: Arc<dyn KeyedReader<K, V>>,
    ) -> Self {
        KeyedOrigin {
            state: DynState::new(keyed.clone()),
            kind: OriginKind::Register {
                keyed,
                changes,
                reader,
            },
        }
    }

    /// A plain register keyspace an operator keeps.
    pub(crate) fn stored(
        id: KeyspaceId<K, Mergeless<V>>,
        changes: NodeRef<Register<K, Mergeless<V>>>,
    ) -> Self {
        Self::register(
            DynKeyed::new(Stored::new(id)),
            Changes::at(changes),
            Arc::new(RegisterReader::plain(id)),
        )
    }

    /// This register with each value mapped by `f`, dropping those it
    /// maps to `None`: stores nothing. `None` for an arrangement.
    pub(crate) fn filter_map<W: Data>(&self, f: ValueMap<K, V, W>) -> Option<KeyedOrigin<K, W>> {
        let OriginKind::Register {
            keyed,
            changes,
            reader,
        } = &self.kind
        else {
            return None;
        };
        let mapped = DynKeyed::new(FilterMapped {
            parent: keyed.clone(),
            f: f.clone(),
        });
        let (parent, f2) = (changes.clone(), f.clone());
        let changes = Changes::lazy(move |g| {
            let input = parent.get(g);
            g.unary(input, FilterMapValues { f: f2.clone() })
        });
        let reader = Arc::new(FilterMappedReader {
            parent: reader.clone(),
            f,
        });
        Some(KeyedOrigin::register(mapped, changes, reader))
    }

    pub(crate) fn arranged(id: Arrangement<K, V>, totals: Option<KeyspaceId<K, Tally>>) -> Self {
        KeyedOrigin {
            state: DynState::new(Arranged::new(id, totals)),
            kind: OriginKind::Arranged { id, totals },
        }
    }

    pub(crate) fn distinct(state: Distinctly<K, V>) -> Self {
        let id = state.id();
        KeyedOrigin {
            state: DynState::new(state),
            kind: OriginKind::Distinct { id },
        }
    }
}

mod sealed {
    use super::*;

    pub trait FeedInner<T> {
        fn build(&self) -> &Shared;
        fn label(&self) -> Arc<str>;
        fn attach(&self) -> NodeRef<Bag<T>>;
        fn inputs(&self) -> Vec<NodeRef<Bag<T>>>;
    }

    pub trait GroupedInner<K: Data, V: Data> {
        fn origin(&self) -> Option<KeyedOrigin<K, V>>;
    }
}

pub(crate) use sealed::{FeedInner, GroupedInner};

/// Anything that gives rows of `T` to an operator or a view: the kinds,
/// and the builder heads and view slots that deref to them.
pub trait Feed<T: Data>: FeedInner<T> {}
/// A grouped feed: any rows per key. [`Grouped`], and [`Keyed`] by
/// upcast.
pub trait GroupedFeed<K: Data, V: Data>: Feed<(K, V)> + GroupedInner<K, V> {}
/// A register-shaped feed: at most one row per key. Only [`Keyed`].
pub trait KeyedFeed<K: Data, V: Data>: GroupedFeed<K, V> {}
/// A bag feed. Only [`Stream`].
pub trait StreamFeed<T: Data>: Feed<T> {}

impl<T: Data, W> FeedInner<T> for W
where
    W: std::ops::Deref,
    W::Target: FeedInner<T>,
{
    fn build(&self) -> &Shared {
        (**self).build()
    }
    fn label(&self) -> Arc<str> {
        (**self).label()
    }
    fn attach(&self) -> NodeRef<Bag<T>> {
        (**self).attach()
    }
    fn inputs(&self) -> Vec<NodeRef<Bag<T>>> {
        (**self).inputs()
    }
}
impl<K: Data, V: Data, W> GroupedInner<K, V> for W
where
    W: std::ops::Deref,
    W::Target: GroupedInner<K, V>,
{
    fn origin(&self) -> Option<KeyedOrigin<K, V>> {
        (**self).origin()
    }
}
impl<T: Data, W> Feed<T> for W
where
    W: std::ops::Deref,
    W::Target: Feed<T>,
{
}
impl<K: Data, V: Data, W> GroupedFeed<K, V> for W
where
    W: std::ops::Deref,
    W::Target: GroupedFeed<K, V>,
{
}
impl<K: Data, V: Data, W> KeyedFeed<K, V> for W
where
    W: std::ops::Deref,
    W::Target: KeyedFeed<K, V>,
{
}
impl<T: Data, W> StreamFeed<T> for W
where
    W: std::ops::Deref,
    W::Target: StreamFeed<T>,
{
}

/// The rows of `feed` grouped by key, to read: its own state where it
/// has one, otherwise an arrangement of them named `name`, and the node
/// giving its rows.
pub(crate) fn arranged<K: Data, V: Data>(
    feed: &(impl GroupedFeed<K, V> + ?Sized),
    name: &str,
) -> (NodeRef<Bag<(K, V)>>, DynState<K, V>) {
    if let Some(origin) = feed.origin() {
        return (feed.attach(), origin.state);
    }
    let inputs = feed.inputs();
    let mut b = feed.build().borrow_mut();
    let id = b.id(name);
    let totals = b.id(&format!("{name}.totals"));
    let node = b.graph.nary(&inputs, Arrange::new(id, totals));
    (node.rows(), DynState::new(Arranged::new(id, Some(totals))))
}

// ---------------------------------------------------------------------------
// The kinds
// ---------------------------------------------------------------------------

/// A bag of `T`.
pub struct Stream<T, C = Root<T>> {
    pub(crate) flow: Flow<T, C>,
}

/// How many rows a key may hold: the kind marker of [`Rows`].
pub trait Kind: Send + Sync + 'static {}
/// At most one row per key: a register.
pub enum One {}
/// Any number of rows per key.
pub enum Many {}
impl Kind for One {}
impl Kind for Many {}

/// `(K, V)` rows of kind `M`. Spelled [`Keyed`] or [`Grouped`].
pub struct Rows<K: Data, V: Data, M, C = Root<(K, V)>> {
    pub(crate) flow: Flow<(K, V), C>,
    /// Where the rows live, when an operator keeps them. Only a stream
    /// rooted at that operator has one: any combinator clears it.
    pub(crate) origin: Option<KeyedOrigin<K, V>>,
    pub(crate) _m: PhantomData<fn() -> M>,
}

/// `(K, V)` rows, any number per key.
pub type Grouped<K, V, C = Root<(K, V)>> = Rows<K, V, Many, C>;

/// `(K, V)` rows, at most one per key.
pub type Keyed<K, V, C = Root<(K, V)>> = Rows<K, V, One, C>;

/// A single value: the register with the one key `()`.
pub type Value<V, C = Root<((), V)>> = Keyed<(), V, C>;

impl<T, C: Clone> Clone for Stream<T, C> {
    fn clone(&self) -> Self {
        Stream {
            flow: self.flow.clone(),
        }
    }
}

impl<K: Data, V: Data, M, C: Clone> Clone for Rows<K, V, M, C> {
    fn clone(&self) -> Self {
        Rows {
            flow: self.flow.clone(),
            origin: self.origin.clone(),
            _m: PhantomData,
        }
    }
}

impl<T, C> Stream<T, C> {
    pub(crate) fn from_flow(flow: Flow<T, C>) -> Self {
        Stream { flow }
    }
}

impl<T: Data> Stream<T> {
    /// A stream rooted at `root`.
    pub(crate) fn rooted(root: Root<T>, build: Shared, label: Arc<str>) -> Self {
        Stream::from_flow(Flow::at_root(root, build, label))
    }
}

impl<K: Data, V: Data, M, C> Rows<K, V, M, C> {
    pub(crate) fn from_flow(flow: Flow<(K, V), C>) -> Self {
        Rows {
            flow,
            origin: None,
            _m: PhantomData,
        }
    }
}

impl<K: Data, V: Data, M> Rows<K, V, M> {
    /// Rows rooted at `root`, kept by `origin`'s operator if any.
    pub(crate) fn rooted(
        root: Root<(K, V)>,
        build: Shared,
        label: Arc<str>,
        origin: Option<KeyedOrigin<K, V>>,
    ) -> Self {
        Rows {
            flow: Flow::at_root(root, build, label),
            origin,
            _m: PhantomData,
        }
    }

    /// Rows rooted at a register, kept by `origin`.
    pub(crate) fn at_register(
        changes: Changes<K, V>,
        origin: KeyedOrigin<K, V>,
        build: Shared,
        label: Arc<str>,
    ) -> Self {
        let root = Root::lazy(move |g| {
            let node = changes.get(g);
            g.unary(node, Flatten::default())
        });
        Self::rooted(root, build, label, Some(origin))
    }

    /// Rows rooted at an operator's register node, kept by it.
    pub(crate) fn at_stored(
        id: KeyspaceId<K, Mergeless<V>>,
        node: NodeRef<Register<K, Mergeless<V>>>,
        build: Shared,
        label: Arc<str>,
    ) -> Self {
        Self::at_register(
            Changes::at(node),
            KeyedOrigin::stored(id, node),
            build,
            label,
        )
    }
}

impl<T: Data, C: Chain<Out = T>> FeedInner<T> for Stream<T, C> {
    fn build(&self) -> &Shared {
        &self.flow.build
    }
    fn label(&self) -> Arc<str> {
        self.flow.label.clone()
    }
    fn attach(&self) -> NodeRef<Bag<T>> {
        self.flow.attach()
    }
    fn inputs(&self) -> Vec<NodeRef<Bag<T>>> {
        self.flow.inputs()
    }
}
impl<T: Data, C: Chain<Out = T>> Feed<T> for Stream<T, C> {}
impl<T: Data, C: Chain<Out = T>> StreamFeed<T> for Stream<T, C> {}

impl<K: Data, V: Data, M: Kind, C: Chain<Out = (K, V)>> FeedInner<(K, V)> for Rows<K, V, M, C> {
    fn build(&self) -> &Shared {
        &self.flow.build
    }
    fn label(&self) -> Arc<str> {
        self.flow.label.clone()
    }
    fn attach(&self) -> NodeRef<Bag<(K, V)>> {
        self.flow.attach()
    }
    fn inputs(&self) -> Vec<NodeRef<Bag<(K, V)>>> {
        self.flow.inputs()
    }
}
impl<K: Data, V: Data, M: Kind, C: Chain<Out = (K, V)>> GroupedInner<K, V> for Rows<K, V, M, C> {
    fn origin(&self) -> Option<KeyedOrigin<K, V>> {
        self.origin.clone()
    }
}
impl<K: Data, V: Data, M: Kind, C: Chain<Out = (K, V)>> Feed<(K, V)> for Rows<K, V, M, C> {}
impl<K: Data, V: Data, M: Kind, C: Chain<Out = (K, V)>> GroupedFeed<K, V> for Rows<K, V, M, C> {}
impl<K: Data, V: Data, C: Chain<Out = (K, V)>> KeyedFeed<K, V> for Keyed<K, V, C> {}

// ---------------------------------------------------------------------------
// Linear combinators
// ---------------------------------------------------------------------------

impl<K: Data, C: Chain<Out = K>> Stream<K, C> {
    /// The build this stream belongs to.
    pub(crate) fn shared(&self) -> &Shared {
        &self.flow.build
    }
}

impl<K, V, C> Keyed<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Re-key by a function the caller guarantees injective over live
    /// rows: a table's alternate key, enforced at write time. Rooted at
    /// its own node, so the result is a plain `Keyed<K2, V>`.
    pub(crate) fn rekeyed_unchecked<K2: Data, F>(&self, f: F) -> Keyed<K2, V>
    where
        F: Fn(&K, &V) -> K2 + Clone + Send + 'static,
    {
        let ext = self.flow.extend(ReKeyWith(f));
        let node = ext.attach();
        Rows::rooted(
            Root::at(node),
            self.flow.build.clone(),
            self.flow.label.clone(),
            None,
        )
    }
}
