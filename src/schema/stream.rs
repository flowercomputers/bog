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
    Bag, Delta, Distinct, Emit, Explode, Filter, FilterMap, FlatMap, FusedLinears, Ident, Map,
    Mergeless, Negate, Register, Sum, Tally, Then,
    codec::{Codec, OrderedCodec},
    graph::{NodeRef, OpDag},
    merge::Merge,
    ops::{
        Arrange, Flatten, Join, ReduceMany, Semijoin,
        linear::{
            ExplodeVal, FilterKV, FlatMapVal, KeyBy, Keys, MapVal, MapWith, ReKey, ReKeyWith,
            Reindex, Unkey, Values,
        },
        reduce::{Aggregate, Take},
    },
    state::{Arranged, Arrangement, Stored},
    tx::KeyspaceId,
};

use super::{
    Shared,
    dynstate::{DynKeyed, DynState, FilterMapValues, FilterMapped, ValueMap},
    read::{FilterMappedReader, KeyedReader, RegisterReader},
};

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
    node: RootNode<T>,
}

enum RootNode<T> {
    Node(NodeRef<Bag<T>>),
    /// Made when first needed: the rows of a register node, which costs
    /// a node of its own.
    Lazy(Rc<LazyNode<T>>),
    /// Several nodes, unioned: a consumer that takes any number of
    /// inputs takes them all; one that takes one gets a node made of
    /// them when first needed.
    Union(Rc<UnionNode<T>>),
}

struct LazyNode<T> {
    made: OnceCell<NodeRef<Bag<T>>>,
    make: Box<dyn Fn(&mut OpDag) -> NodeRef<Bag<T>>>,
}

struct UnionNode<T> {
    inputs: Vec<NodeRef<Bag<T>>>,
    made: OnceCell<NodeRef<Bag<T>>>,
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

    fn node(&self, g: &mut OpDag) -> NodeRef<Bag<T>>
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

    fn inputs(&self, g: &mut OpDag) -> Vec<NodeRef<Bag<T>>> {
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
    chain: C,
    build: Shared,
    /// The name of this flow's root: a source or view name, or an
    /// operator name like `transfers.sum`. Operator keyspaces are named
    /// from it.
    label: Arc<str>,
    /// The node this chain was built as, shared by its consumers.
    attached: Rc<OnceCell<NodeRef<Bag<T>>>>,
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
    fn extend<Y>(&self, y: Y) -> Flow<Y::Emits, Ext<C, Y>>
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
    fn attach(&self) -> NodeRef<Bag<T>> {
        *self.attached.get_or_init(|| {
            let mut b = self.build.borrow_mut();
            self.chain.attach(&mut b.graph)
        })
    }

    /// The nodes giving this flow's rows, for a consumer that unions
    /// its inputs: a union root's members, else the one node.
    fn inputs(&self) -> Vec<NodeRef<Bag<T>>> {
        match self.attached.get() {
            Some(n) => vec![*n],
            None => {
                let mut b = self.build.borrow_mut();
                self.chain.inputs(&mut b.graph)
            }
        }
    }

    /// A fresh, unique operator name under this flow's label.
    fn name(&self, kind: &str, other: Option<&str>) -> String {
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
            },
        }
    }
}

/// A register node's changes, made when first needed: a filtered or
/// mapped view of a register costs a node only if something consumes
/// its changes.
pub(crate) struct Changes<K: Data, V: Data>(Rc<LazyChanges<K, V>>);

struct LazyChanges<K: Data, V: Data> {
    made: OnceCell<NodeRef<Register<K, Mergeless<V>>>>,
    make: Box<dyn Fn(&mut OpDag) -> NodeRef<Register<K, Mergeless<V>>>>,
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

    fn lazy(make: impl Fn(&mut OpDag) -> NodeRef<Register<K, Mergeless<V>>> + 'static) -> Self {
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
    flow: Flow<T, C>,
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
    flow: Flow<(K, V), C>,
    /// Where the rows live, when an operator keeps them. Only a stream
    /// rooted at that operator has one: any combinator clears it.
    origin: Option<KeyedOrigin<K, V>>,
    _m: PhantomData<fn() -> M>,
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
    fn from_flow(flow: Flow<T, C>) -> Self {
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
    fn from_flow(flow: Flow<(K, V), C>) -> Self {
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

impl<T: Data, C: Chain<Out = T>> Stream<T, C> {
    pub fn map<U: Data, F>(&self, f: F) -> Stream<U, Ext<C, Map<F>>>
    where
        F: Fn(&T) -> U + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Map(f)))
    }

    pub fn filter<P>(&self, p: P) -> Stream<T, Ext<C, Filter<P>>>
    where
        P: Fn(&T) -> bool + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Filter(p)))
    }

    pub fn flat_map<I, F>(&self, f: F) -> Stream<I::Item, Ext<C, FlatMap<F>>>
    where
        I: IntoIterator + 'static,
        I::Item: Data,
        F: Fn(&T) -> I + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(FlatMap(f)))
    }

    pub fn filter_map<U: Data, F>(&self, f: F) -> Stream<U, Ext<C, FilterMap<F>>>
    where
        F: Fn(&T) -> Option<U> + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(FilterMap(f)))
    }

    /// Any number of weighted rows per row, each at the row's
    /// multiplicity times its weight.
    pub fn explode<I, U: Data, F>(&self, f: F) -> Stream<U, Ext<C, Explode<F>>>
    where
        I: IntoIterator<Item = (U, Delta)> + 'static,
        F: Fn(&T) -> I + Clone + Send + 'static,
    {
        Stream::from_flow(self.flow.extend(Explode(f)))
    }

    /// Every row at the opposite multiplicity.
    pub fn negate(&self) -> Stream<T, Ext<C, Negate>> {
        Stream::from_flow(self.flow.extend(Negate))
    }

    /// Group the rows by `f`: `t` becomes `(f(&t), t)`.
    pub fn key_by<K: Data, F>(&self, f: F) -> Grouped<K, T, Ext<C, KeyBy<F>>>
    where
        F: Fn(&T) -> K + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(KeyBy(f)))
    }
}

impl<K, V, M, C> Rows<K, V, M, C>
where
    K: Data,
    V: Data,
    M: Kind,
    C: Chain<Out = (K, V)>,
{
    /// Map each value, keeping its key. Over a register an operator
    /// keeps, the result is still one, read through `f`: a view defined
    /// from it stores nothing.
    pub fn map<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, MapVal<F>>>
    where
        F: Fn(&V) -> W + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(MapVal(f), Arc::new(move |_: &K, v: &V| Some(g(v))))
    }

    /// Map each value with its key, keeping the key. See [`map`](Self::map).
    pub fn map_with<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, MapWith<F>>>
    where
        F: Fn(&K, &V) -> W + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(MapWith(f), Arc::new(move |k: &K, v: &V| Some(g(k, v))))
    }

    /// Keep the rows `p` accepts. See [`map`](Self::map).
    pub fn filter<P>(&self, p: P) -> Rows<K, V, M, Ext<C, FilterKV<P>>>
    where
        P: Fn(&K, &V) -> bool + Clone + Send + Sync + 'static,
    {
        let q = p.clone();
        self.through(
            FilterKV(p),
            Arc::new(move |k: &K, v: &V| q(k, v).then(|| v.clone())),
        )
    }

    /// Map each value, dropping those `f` maps to `None`. See
    /// [`map`](Self::map).
    pub fn filter_map<W: Data, F>(&self, f: F) -> Rows<K, W, M, Ext<C, FlatMapVal<F>>>
    where
        F: Fn(&V) -> Option<W> + Clone + Send + Sync + 'static,
    {
        let g = f.clone();
        self.through(FlatMapVal(f), Arc::new(move |_: &K, v: &V| g(v)))
    }

    /// Extend the chain by `stage`, which maps each value as `f` says,
    /// keeping a register origin as a mapped register.
    fn through<W: Data, Y>(&self, stage: Y, f: ValueMap<K, V, W>) -> Rows<K, W, M, Ext<C, Y>>
    where
        Y: Emit<(K, V), Emits = (K, W)> + Clone + Send + 'static,
    {
        Rows {
            flow: self.flow.extend(stage),
            origin: self.origin.as_ref().and_then(|o| o.filter_map(f)),
            _m: PhantomData,
        }
    }

    /// Regroup by `f` of the value. Two rows may now share a key, so
    /// the result is grouped.
    pub fn key_by<K2: Data, F>(&self, f: F) -> Grouped<K2, V, Ext<C, ReKey<F>>>
    where
        F: Fn(&V) -> K2 + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ReKey(f)))
    }

    /// Regroup by `f` of the key and value.
    pub fn key_by_with<K2: Data, F>(&self, f: F) -> Grouped<K2, V, Ext<C, ReKeyWith<F>>>
    where
        F: Fn(&K, &V) -> K2 + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ReKeyWith(f)))
    }

    /// Just the keys, as a bag: one per row.
    pub fn keys(&self) -> Stream<K, Ext<C, Keys>> {
        Stream::from_flow(self.flow.extend(Keys))
    }

    /// Just the values, as a bag: one per row.
    pub fn values(&self) -> Stream<V, Ext<C, Values>> {
        Stream::from_flow(self.flow.extend(Values))
    }

    /// Leave the keyed world: the rows as a bag of pairs.
    pub fn entries(&self) -> Stream<(K, V), C> {
        Stream::from_flow(self.flow.clone())
    }

    /// The same rows, viewed as grouped: a register is a group of at
    /// most one.
    pub fn grouped(&self) -> Grouped<K, V, C> {
        Rows {
            flow: self.flow.clone(),
            origin: self.origin.clone(),
            _m: PhantomData,
        }
    }
}

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Any number of values per value, each keeping the key.
    pub fn flat_map<I, F>(&self, f: F) -> Grouped<K, I::Item, Ext<C, FlatMapVal<F>>>
    where
        I: IntoIterator + 'static,
        I::Item: Data,
        F: Fn(&V) -> I + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(FlatMapVal(f)))
    }

    /// Weighted values per value, each keeping the key; see
    /// [`Stream::explode`].
    pub fn explode<I, W: Data, F>(&self, f: F) -> Grouped<K, W, Ext<C, ExplodeVal<F>>>
    where
        I: IntoIterator<Item = (W, Delta)> + 'static,
        F: Fn(&V) -> I + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(ExplodeVal(f)))
    }
}

impl<K, V, C> Keyed<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Re-key by a derived prefix, keeping the key as the last
    /// component: `(k, v)` becomes `((f(&k, &v), k), v)`. Injective by
    /// construction, so still a register: a secondary index, whose rows
    /// sort by the prefix first.
    pub fn index_by<X: Data, F>(&self, f: F) -> Keyed<(X, K), V, Ext<C, Reindex<F>>>
    where
        F: Fn(&K, &V) -> X + Clone + Send + 'static,
    {
        Rows::from_flow(self.flow.extend(Reindex(f)))
    }

    /// Every row under the one key `()`, for reductions over the whole
    /// register.
    fn unkeyed(&self) -> Grouped<(), V, Ext<C, Unkey>> {
        Rows::from_flow(self.flow.extend(Unkey))
    }
}

// ---------------------------------------------------------------------------
// Stateful combinators
// ---------------------------------------------------------------------------

impl<T: Data, C: Chain<Out = T>> Stream<T, C> {
    /// Bag union with another stream. Costs nothing by itself: what
    /// consumes the union takes both as inputs.
    pub fn concat(&self, other: &impl StreamFeed<T>) -> Stream<T> {
        let mut inputs = self.flow.inputs();
        inputs.extend(other.inputs());
        let name = self.flow.name("concat", Some(&other.label()));
        Stream::rooted(Root::union(inputs), self.flow.build.clone(), name.into())
    }

    /// Each row once, however many times it occurs: set semantics.
    pub fn distinct(&self) -> Stream<T> {
        let inputs = self.flow.inputs();
        let name = self.flow.name("distinct", None);
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<T, Tally> = b.id(&name);
        let node = b.graph.nary(&inputs, Distinct::new(id));
        drop(b);
        Stream::rooted(Root::at(node.rows()), self.flow.build.clone(), name.into())
    }

    /// Total multiplicity across the whole stream.
    pub fn count(&self) -> Value<Delta> {
        self.key_by(|_| ()).count()
    }
}

/// The output of a [`Sum`] node, read through `pick`.
fn summed<K, V, W, C>(
    flow: &Flow<(K, V), C>,
    kind: &str,
    pick: impl Fn(&(Delta, V)) -> W + Send + Sync + 'static,
) -> Keyed<K, W>
where
    K: Data,
    V: Additive,
    W: Data,
    C: Chain<Out = (K, V)>,
{
    let inputs = flow.inputs();
    let name = flow.name(kind, None);
    let pick: Arc<dyn Fn(&(Delta, V)) -> W + Send + Sync> = Arc::new(pick);
    let mut b = flow.build.borrow_mut();
    let id: KeyspaceId<K, (Delta, V)> = b.id(&name);
    let node = b.graph.nary(&inputs, Sum::<K, V>::new(id));
    let p = pick.clone();
    let mapped = b.graph.map_values(node, move |v| p(v));
    let keyed = DynKeyed::new(b.graph.state(mapped));
    drop(b);
    let changes = Changes::at(mapped.rows());
    let origin = KeyedOrigin::register(
        keyed,
        changes.clone(),
        Arc::new(RegisterReader::new(id, pick)),
    );
    Rows::at_register(changes, origin, flow.build.clone(), name.into())
}

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Union with another grouped (or keyed) stream on the same key.
    /// Costs nothing by itself: what consumes the union takes both as
    /// inputs.
    pub fn concat(&self, other: &impl GroupedFeed<K, V>) -> Grouped<K, V> {
        let mut inputs = self.flow.inputs();
        inputs.extend(other.inputs());
        let name = self.flow.name("concat", Some(&other.label()));
        Rows::rooted(
            Root::union(inputs),
            self.flow.build.clone(),
            name.into(),
            None,
        )
    }

    /// Each `(key, value)` once, however many times it occurs.
    pub fn distinct(&self) -> Grouped<K, V> {
        let inputs = self.flow.inputs();
        let name = self.flow.name("distinct", None);
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<(K, V), Tally> = b.id(&name);
        let node = b.graph.nary(&inputs, Distinct::new(id));
        drop(b);
        Rows::rooted(
            Root::at(node.rows()),
            self.flow.build.clone(),
            name.into(),
            None,
        )
    }

    /// Inner join against a grouped stream on the same key: every row
    /// pairs with every right row of its key, at the product of their
    /// multiplicities. Reads both sides' state; see the module docs.
    pub fn join<B: Data, O: Data, F>(
        &self,
        other: &impl GroupedFeed<K, B>,
        mut f: F,
    ) -> Grouped<K, O>
    where
        F: FnMut(&K, &V, &B) -> O + Send + 'static,
    {
        let name = self.flow.name("join", Some(&other.label()));
        let (l, ls) = arranged(self, &format!("{name}.left"));
        let (r, rs) = arranged(other, &format!("{name}.right"));
        let logic = move |k: &K, a: &V, b: &B| (k.clone(), f(k, a, b));
        let node = self
            .flow
            .build
            .borrow_mut()
            .graph
            .binary(l, r, Join::new(ls, rs, logic));
        Rows::rooted(Root::at(node), self.flow.build.clone(), name.into(), None)
    }

    /// Total of the values per key. A key with values is present even
    /// when they total zero; a key whose values are all retracted is
    /// not. One accumulator per key, so a dirty key costs one lookup
    /// whatever its group size.
    pub fn sum(&self) -> Keyed<K, V>
    where
        V: Additive,
    {
        summed(&self.flow, "sum", |(_, v)| v.clone())
    }

    /// Multiplicity per key.
    pub fn count(&self) -> Keyed<K, Delta> {
        let unit: Grouped<K, (), _> = Rows::from_flow(self.flow.extend(MapVal(|_: &V| ())));
        summed(&unit.flow, "count", |(n, _)| *n)
    }

    /// One value per key from the key's whole group, which arrives
    /// sorted ascending, consolidated, each value with its multiplicity.
    /// `f` is not called for a key whose group emptied.
    pub fn aggregate<O: Data, F>(&self, f: F) -> Keyed<K, O>
    where
        F: FnMut(&K, &[(V, Delta)]) -> O + Send + 'static,
    {
        self.aggregated("aggregate", f, Take::Whole)
    }

    /// The least value of each key.
    pub fn min(&self) -> Keyed<K, V> {
        fn least<K, V: Clone>(_: &K, vals: &[(V, Delta)]) -> V {
            vals[0].0.clone()
        }
        self.aggregated("min", least::<K, V>, Take::First(1))
    }

    /// The greatest value of each key.
    pub fn max(&self) -> Keyed<K, V> {
        fn greatest<K, V: Clone>(_: &K, vals: &[(V, Delta)]) -> V {
            vals[vals.len() - 1].0.clone()
        }
        self.aggregated("max", greatest::<K, V>, Take::Last(1))
    }

    fn aggregated<O: Data, F>(&self, kind: &str, logic: F, take: Take) -> Keyed<K, O>
    where
        F: FnMut(&K, &[(V, Delta)]) -> O + Send + 'static,
    {
        let name = self.flow.name(kind, None);
        let (input, state) = arranged(self, &format!("{name}.state"));
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<K, Mergeless<O>> = b.id(&name);
        let node = b
            .graph
            .nary(&[input], Aggregate::new(state, id, logic).reading(take));
        drop(b);
        Rows::at_stored(id, node.rows(), self.flow.build.clone(), name.into())
    }

    /// Any number of outputs per key from the key's whole group: the
    /// most general reduction. The outputs are kept, so downstream
    /// always holds `f` of the current groups.
    pub fn reduce<O: Data, F>(&self, f: F) -> Grouped<K, O>
    where
        F: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send + 'static,
    {
        self.reduced("reduce", f, Take::Whole)
    }

    /// The `k` greatest distinct values of each key, each once.
    pub fn top_k(&self, k: usize) -> Grouped<K, V> {
        let logic = move |_: &K, vals: &[(V, Delta)], out: &mut Vec<(V, Delta)>| {
            for (v, _) in vals.iter().rev().take(k) {
                out.push((v.clone(), 1));
            }
        };
        self.reduced("top_k", logic, Take::Last(k))
    }

    fn reduced<O: Data, F>(&self, kind: &str, logic: F, take: Take) -> Grouped<K, O>
    where
        F: FnMut(&K, &[(V, Delta)], &mut Vec<(O, Delta)>) + Send + 'static,
    {
        let name = self.flow.name(kind, None);
        let (input, state) = arranged(self, &format!("{name}.state"));
        let mut b = self.flow.build.borrow_mut();
        let id: Arrangement<K, O> = b.id(&name);
        let node = b
            .graph
            .nary(&[input], ReduceMany::new(state, id, logic).reading(take));
        drop(b);
        let origin = KeyedOrigin::arranged(id, None);
        Rows::rooted(
            Root::at(node.rows()),
            self.flow.build.clone(),
            name.into(),
            Some(origin),
        )
    }
}

impl<K, V, M, C> Rows<K, V, M, C>
where
    K: Data,
    V: Data,
    M: Kind,
    C: Chain<Out = (K, V)>,
{
    /// Inner join against a register on the same key: each row pairs
    /// with the key's one right value, so the kind is kept.
    pub fn with<B: Data, O: Data, F>(&self, other: &impl KeyedFeed<K, B>, mut f: F) -> Rows<K, O, M>
    where
        F: FnMut(&K, &V, &B) -> O + Send + 'static,
    {
        let name = self.flow.name("with", Some(&other.label()));
        let (l, ls) = arranged(self, &format!("{name}.left"));
        let (r, rs) = arranged(other, &format!("{name}.right"));
        let logic = move |k: &K, a: &V, b: &B| (k.clone(), f(k, a, b));
        let node = self
            .flow
            .build
            .borrow_mut()
            .graph
            .binary(l, r, Join::new(ls, rs, logic));
        Rows::rooted(Root::at(node), self.flow.build.clone(), name.into(), None)
    }

    /// The rows whose key the other side has: SQL's `EXISTS`.
    pub fn matching<B: Data>(&self, other: &impl GroupedFeed<K, B>) -> Rows<K, V, M> {
        self.semijoin(other, false)
    }

    /// The rows whose key the other side lacks: `NOT EXISTS`.
    pub fn without<B: Data>(&self, other: &impl GroupedFeed<K, B>) -> Rows<K, V, M> {
        self.semijoin(other, true)
    }

    fn semijoin<B: Data>(&self, other: &impl GroupedFeed<K, B>, anti: bool) -> Rows<K, V, M> {
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

impl<K, V, C> Keyed<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Number of rows in the register.
    pub fn count(&self) -> Value<Delta> {
        self.unkeyed().count()
    }

    pub fn min(&self) -> Value<V> {
        self.unkeyed().min()
    }

    pub fn max(&self) -> Value<V> {
        self.unkeyed().max()
    }

    /// Total of every value in the register.
    pub fn sum(&self) -> Value<V>
    where
        V: Additive,
    {
        self.unkeyed().sum()
    }

    /// One value from every row of the register.
    pub fn aggregate<O: Data, F>(&self, mut f: F) -> Value<O>
    where
        F: FnMut(&[((K, V), Delta)]) -> O + Send + 'static,
    {
        let unit: Grouped<(), (K, V), _> =
            Rows::from_flow(self.flow.extend(KeyBy(|_: &(K, V)| ())));
        unit.aggregate(move |_, rows| f(rows))
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

impl<K, V, C> Grouped<K, V, C>
where
    K: Data + std::fmt::Debug,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Treat this grouped stream as a register, enforcing that it is
    /// one row per key. A transaction that leaves a key with more than
    /// one row, or one row at a multiplicity other than one, is refused
    /// with a [`Rejected`](super::Rejected) naming the operator. Prefer
    /// [`min`](Self::min) or [`max`](Self::max) when a key may
    /// legitimately have several rows.
    pub fn unique(&self) -> Keyed<K, V> {
        let name = self.flow.name("unique", None);
        let (input, state) = arranged(self, &format!("{name}.state"));
        let mut b = self.flow.build.borrow_mut();
        let id: KeyspaceId<K, Mergeless<V>> = b.id(&name);
        let node = b.graph.nary(
            &[input],
            super::constraints::Unique::new(name.as_str().into(), state, id),
        );
        drop(b);
        Rows::at_stored(id, node.rows(), self.flow.build.clone(), name.into())
    }
}

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

    /// The build this stream belongs to.
    pub(crate) fn shared(&self) -> &Shared {
        &self.flow.build
    }
}
