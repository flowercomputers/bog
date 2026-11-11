//! View kinds as schema fields: [`MapView`], [`GroupView`],
//! [`OrderedView`], [`ValueView`]. Each is a declaration marker with a
//! builder slot (defined once, then a stream head), a handle and a read
//! facade, tied together by [`ViewDecl`].

use std::{
    cell::{Cell, OnceCell, RefCell},
    marker::PhantomData,
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use crate::{
    dataflow::state::Edge,
    dataflow::{Bag, Nary, Register},
    ops::IndexBy,
    ops::{Arrange, Materialize},
    store::merge::Mergeless,
    store::txn::{KeyspaceId, WriteTx},
};

use super::{
    At, Data, GroupRead, Grouped, GroupedFeed, Keyed, KeyedFeed, KeyedOrigin, OrderedRead, Read,
    Root, Rows, Shared, Value, ValueRead,
    read::{GroupReader, KeyedReader, OrderReader},
};
use crate::stream::{GroupedInner, OriginKind};

/// A view field of a schema.
pub trait ViewDecl: 'static {
    /// The builder slot: `define` it, then it is a stream head.
    type Slot;
    type Handle: Send + Sync + 'static;
    type Read<'a>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot;

    /// The handle, once `declare` has run.
    ///
    /// # Panics
    ///
    /// If the view was never defined.
    fn finish(slot: Self::Slot) -> Self::Handle;

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a>;
}

/// A post-graph check on a row the view holds.
pub(crate) type Guard<K, V> =
    Box<dyn Fn(&K, &V) -> Result<(), Arc<dyn std::error::Error + Send + Sync>> + Send>;

/// What every slot shares: its name, whether it is defined, and the
/// guards to run over its rows.
pub(crate) struct SlotCore<K: Data, V: Data> {
    pub(crate) name: Arc<str>,
    pub(crate) build: Shared,
    defined: Rc<Cell<bool>>,
    pub(crate) guards: RefCell<Vec<Guard<K, V>>>,
}

impl<K: Data, V: Data> SlotCore<K, V> {
    pub(crate) fn new(name: Arc<str>, build: &Shared) -> Self {
        SlotCore {
            defined: build.borrow_mut().slot(name.clone()),
            name,
            build: build.clone(),
            guards: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn defining(&self) {
        assert!(
            !self.defined.replace(true),
            "view `{}` is defined twice",
            self.name
        );
    }

    pub(crate) fn undefined(&self) -> ! {
        panic!(
            "view `{}` was used before it was defined; define it first",
            self.name
        )
    }

    /// Run the guards over the rows of `head` after every graph run.
    fn guard(&self, head: &impl GroupedFeed<K, V>) {
        let guards = self.guards.take();
        if guards.is_empty() {
            return;
        }
        let guards = Guards(guards);
        // a register's changes say what arrived without a flattened copy
        if let Some(KeyedOrigin {
            kind: OriginKind::Register { changes, .. },
            ..
        }) = head.origin()
        {
            let mut b = self.build.borrow_mut();
            let changes = changes.get(&mut b.graph);
            b.graph.nary(&[changes], GuardChanges(guards));
        } else {
            let inputs = head.inputs();
            self.build
                .borrow_mut()
                .graph
                .nary(&inputs, GuardRows(guards));
        }
    }
}

/// A view's guards, run over each row that arrives.
struct Guards<K, V>(Vec<Guard<K, V>>);

impl<K, V> Guards<K, V> {
    fn check(&self, tx: &WriteTx, k: &K, v: &V) -> bool {
        for guard in &self.0 {
            if let Err(e) = guard(k, v) {
                tx.reject_with(e);
                return false;
            }
        }
        true
    }
}

/// Rejects the tx when a row arriving at a view fails a guard.
struct GuardRows<K, V>(Guards<K, V>);

impl<K, V> Nary for GuardRows<K, V>
where
    K: Ord + Send + 'static,
    V: Ord + Send + 'static,
{
    type Takes = Bag<(K, V)>;
    type Gives = Bag<()>;
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Bag<(K, V)>], _: &mut Bag<()>) {
        // a rebuild replays rows accepted when they were written
        if tx.rebuilding() {
            return;
        }
        for b in batches {
            for ((k, v), d) in b.diffs() {
                if d > 0 && !self.0.check(tx, k, v) {
                    return;
                }
            }
        }
    }
}

/// [`GuardRows`] over a register's changes: each key's new value.
struct GuardChanges<K, V>(Guards<K, V>);

impl<K, V> Nary for GuardChanges<K, V>
where
    K: Ord + Send + 'static,
    V: Clone + Send + 'static,
{
    type Takes = Register<K, Mergeless<V>>;
    type Gives = Bag<()>;
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Register<K, Mergeless<V>>], _: &mut Bag<()>) {
        if tx.rebuilding() {
            return;
        }
        for b in batches {
            for (k, change) in b.iter() {
                if let Some(v) = change.new_value()
                    && !self.0.check(tx, k, v)
                {
                    return;
                }
            }
        }
    }
}

/// A register kept for a view: adopted from the operator that keeps
/// it, or materialized under the view's name.
fn register_for<K: Data, V: Data>(
    core: &SlotCore<K, V>,
    from: &impl KeyedFeed<K, V>,
) -> (Keyed<K, V>, Arc<dyn KeyedReader<K, V>>) {
    if let Some(origin) = from.origin()
        && let OriginKind::Register {
            changes, reader, ..
        } = &origin.kind
    {
        let (changes, reader) = (changes.clone(), reader.clone());
        let head = Rows::at_register(changes, origin, core.build.clone(), core.name.clone());
        return (head, reader);
    }
    let inputs = from.inputs();
    let mut b = core.build.borrow_mut();
    let id: KeyspaceId<K, Mergeless<V>> = b.id(&core.name);
    let node = b.graph.nary(&inputs, Materialize::new(id));
    drop(b);
    let reader: Arc<dyn KeyedReader<K, V>> = Arc::new(super::read::RegisterReader::plain(id));
    (
        Rows::at_stored(id, node.rows(), core.build.clone(), core.name.clone()),
        reader,
    )
}

// ---------------------------------------------------------------------------
// MapView
// ---------------------------------------------------------------------------

/// A register: one value per key, defined from a [`Keyed`] stream.
pub struct MapView<K, V>(PhantomData<fn() -> (K, V)>);

pub struct MapSlot<K: Data, V: Data> {
    pub(crate) core: SlotCore<K, V>,
    head: OnceCell<Keyed<K, V>>,
    reader: OnceCell<Arc<dyn KeyedReader<K, V>>>,
}

impl<K: Data, V: Data> MapSlot<K, V> {
    fn open(name: Arc<str>, build: &Shared) -> Self {
        MapSlot {
            core: SlotCore::new(name, build),
            head: OnceCell::new(),
            reader: OnceCell::new(),
        }
    }

    /// Define the view as the rows of `from`.
    pub fn define(&self, from: &impl KeyedFeed<K, V>) {
        self.core.defining();
        let (head, reader) = register_for(&self.core, from);
        let _ = self.head.set(head);
        let _ = self.reader.set(reader);
    }

    fn finish(self) -> MapHandle<K, V> {
        let Some(reader) = self.reader.into_inner() else {
            self.core.undefined()
        };
        let head = self.head.into_inner().expect("defined with the reader");
        self.core.guard(&head);
        MapHandle { reader }
    }
}

impl<K: Data, V: Data> Deref for MapSlot<K, V> {
    type Target = Keyed<K, V>;
    fn deref(&self) -> &Keyed<K, V> {
        self.head.get().unwrap_or_else(|| self.core.undefined())
    }
}

pub struct MapHandle<K, V> {
    reader: Arc<dyn KeyedReader<K, V>>,
}

pub type MapRead<'a, K, V> = Read<'a, K, V>;

impl<K: Data, V: Data> ViewDecl for MapView<K, V> {
    type Slot = MapSlot<K, V>;
    type Handle = MapHandle<K, V>;
    type Read<'a> = MapRead<'a, K, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot {
        MapSlot::open(name, build)
    }

    fn finish(slot: Self::Slot) -> Self::Handle {
        slot.finish()
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        Read::new(&*h.reader, at)
    }
}

// ---------------------------------------------------------------------------
// GroupView
// ---------------------------------------------------------------------------

/// Any number of values per key, each at a multiplicity, defined from a
/// [`Grouped`] stream.
pub struct GroupView<K, V>(PhantomData<fn() -> (K, V)>);

pub struct GroupSlot<K: Data, V: Data> {
    pub(crate) core: SlotCore<K, V>,
    head: OnceCell<Grouped<K, V>>,
    reader: OnceCell<GroupReader<K, V>>,
}

impl<K: Data, V: Data> GroupSlot<K, V> {
    /// Define the view as the rows of `from`. Rows an operator already
    /// keeps by key (an arrangement, a keyed `distinct`) are read in
    /// place; any others are arranged under the view's name.
    pub fn define(&self, from: &impl GroupedFeed<K, V>) {
        self.core.defining();
        let core = &self.core;
        let adopt = |origin: KeyedOrigin<K, V>, reader| {
            let head = Rows::rooted(
                Root::at(from.attach()),
                core.build.clone(),
                core.name.clone(),
                Some(origin),
            );
            (head, reader)
        };
        let (head, reader) = match from.origin() {
            Some(
                origin @ KeyedOrigin {
                    kind: OriginKind::Arranged { id, totals },
                    ..
                },
            ) => adopt(origin.clone(), GroupReader::arranged(id, totals)),
            Some(
                origin @ KeyedOrigin {
                    kind: OriginKind::Distinct { id },
                    ..
                },
            ) => adopt(origin.clone(), GroupReader::distinct(id)),
            _ => {
                let inputs = from.inputs();
                let mut b = core.build.borrow_mut();
                let id = b.id(&core.name);
                let totals = b.id(&format!("{}.totals", core.name));
                let node = b.graph.nary(&inputs, Arrange::new(id, totals));
                drop(b);
                (
                    Rows::rooted(
                        Root::at(node.rows()),
                        core.build.clone(),
                        core.name.clone(),
                        Some(KeyedOrigin::arranged(id, Some(totals))),
                    ),
                    GroupReader::arranged(id, Some(totals)),
                )
            }
        };
        let _ = self.head.set(head);
        let _ = self.reader.set(reader);
    }
}

impl<K: Data, V: Data> Deref for GroupSlot<K, V> {
    type Target = Grouped<K, V>;
    fn deref(&self) -> &Grouped<K, V> {
        self.head.get().unwrap_or_else(|| self.core.undefined())
    }
}

pub struct GroupHandle<K, V> {
    reader: GroupReader<K, V>,
}

impl<K: Data, V: Data> ViewDecl for GroupView<K, V> {
    type Slot = GroupSlot<K, V>;
    type Handle = GroupHandle<K, V>;
    type Read<'a> = GroupRead<'a, K, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot {
        GroupSlot {
            core: SlotCore::new(name, build),
            head: OnceCell::new(),
            reader: OnceCell::new(),
        }
    }

    fn finish(slot: Self::Slot) -> Self::Handle {
        let Some(reader) = slot.reader.into_inner() else {
            slot.core.undefined()
        };
        let head = slot.head.into_inner().expect("defined with the reader");
        slot.core.guard(&head);
        GroupHandle { reader }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        GroupRead::new(&h.reader, at)
    }
}

// ---------------------------------------------------------------------------
// OrderedView
// ---------------------------------------------------------------------------

/// A register read in value order as well: a leaderboard. Defined from
/// a [`Keyed`] stream; keeps an index of `(value, key)` beside it.
pub struct OrderedView<K, V>(PhantomData<fn() -> (K, V)>);

pub struct OrderedSlot<K: Data, V: Data> {
    map: MapSlot<K, V>,
    order: OnceCell<OrderReader<K, V>>,
}

impl<K: Data, V: Data> OrderedSlot<K, V> {
    pub(crate) fn map_core(&self) -> &SlotCore<K, V> {
        &self.map.core
    }

    /// Define the view as the rows of `from`.
    pub fn define(&self, from: &impl KeyedFeed<K, V>) {
        self.map.define(from);
        let head = self.map.head.get().expect("just defined");
        let Some(KeyedOrigin {
            kind: OriginKind::Register { changes, .. },
            ..
        }) = head.origin()
        else {
            unreachable!("a map view is rooted at a register")
        };
        let core = &self.map.core;
        let mut b = core.build.borrow_mut();
        let id: KeyspaceId<(V, Edge<K>), ()> = b.id(&format!("{}.order", core.name));
        let changes = changes.get(&mut b.graph);
        b.graph.nary(
            &[changes],
            IndexBy::new(id, |k: &K, v: &V| Some((v.clone(), Edge::Val(k.clone())))),
        );
        let _ = self.order.set(OrderReader { id });
    }
}

impl<K: Data, V: Data> Deref for OrderedSlot<K, V> {
    type Target = Keyed<K, V>;
    fn deref(&self) -> &Keyed<K, V> {
        &self.map
    }
}

pub struct OrderedHandle<K: Data, V: Data> {
    map: MapHandle<K, V>,
    order: OrderReader<K, V>,
}

impl<K: Data, V: Data> ViewDecl for OrderedView<K, V> {
    type Slot = OrderedSlot<K, V>;
    type Handle = OrderedHandle<K, V>;
    type Read<'a> = OrderedRead<'a, K, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot {
        OrderedSlot {
            map: MapSlot::open(name, build),
            order: OnceCell::new(),
        }
    }

    fn finish(slot: Self::Slot) -> Self::Handle {
        let Some(order) = slot.order.into_inner() else {
            slot.map.core.undefined()
        };
        OrderedHandle {
            map: slot.map.finish(),
            order,
        }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        OrderedRead::new(Read::new(&*h.map.reader, at), &h.order)
    }
}

// ---------------------------------------------------------------------------
// ValueView
// ---------------------------------------------------------------------------

/// A single value, defined from a [`Value`] stream: a count, a total.
pub struct ValueView<V>(PhantomData<fn() -> V>);

pub struct ValueSlot<V: Data> {
    pub(crate) map: MapSlot<(), V>,
}

impl<V: Data> ValueSlot<V> {
    /// Define the view as `from`'s one value.
    pub fn define(&self, from: &impl KeyedFeed<(), V>) {
        self.map.define(from)
    }
}

impl<V: Data> Deref for ValueSlot<V> {
    type Target = Value<V>;
    fn deref(&self) -> &Value<V> {
        &self.map
    }
}

pub struct ValueHandle<V> {
    map: MapHandle<(), V>,
}

impl<V: Data> ViewDecl for ValueView<V> {
    type Slot = ValueSlot<V>;
    type Handle = ValueHandle<V>;
    type Read<'a> = ValueRead<'a, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Slot {
        ValueSlot {
            map: MapSlot::open(name, build),
        }
    }

    fn finish(slot: Self::Slot) -> Self::Handle {
        ValueHandle {
            map: slot.map.finish(),
        }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        ValueRead::new(Read::new(&*h.map.reader, at))
    }
}

#[allow(dead_code)]
fn _uses<K: Data, V: Data>(
    r: &Register<K, crate::store::merge::Mergeless<V>>,
) -> &Register<K, crate::store::merge::Mergeless<V>> {
    r
}
