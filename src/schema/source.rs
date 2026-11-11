//! Source kinds as schema fields: [`Table`], [`Events`], [`Set`],
//! [`Cell`], [`Multiset`]. Each is a declaration marker with a builder
//! head (a stream, plus constraints), a handle, a write facade and a
//! read facade, tied together by [`SourceDecl`].

use std::{cell::RefCell, marker::PhantomData, ops::Deref, rc::Rc, sync::Arc};

use crate::{
    database::DbTx,
    dataflow::Delta,
    error::Result,
    id::Id,
    source as raw,
    source::Retain,
    store::merge::{Mergeless, Tally},
    store::txn::KeyspaceId,
};

use super::{
    At, Data, Keyed, KeyedOrigin, MultisetRead, Read, Root, SetRead, Shared, Stream, TxCell, Value,
    ValueRead, read::RegisterReader,
};
use crate::{
    constraints::AltKey,
    stream::{Changes, dynstate::DynKeyed},
};

/// A write-time check on a row; see the heads' `check`.
pub(crate) type Check<V> = Box<dyn Fn(&V) -> Result<()> + Send + Sync>;

/// Checks as a head collects them.
pub(crate) type Checks<V> = Rc<RefCell<Vec<Check<V>>>>;

fn run_checks<V>(checks: &[Check<V>], v: &V) -> Result<()> {
    for check in checks {
        check(v)?;
    }
    Ok(())
}

/// A source field of a schema: its faces and how to open them.
pub trait SourceDecl: 'static {
    /// The builder head: the source as a stream, with its constraints.
    type Head;
    type Handle: Send + Sync + 'static;
    type Write<'a>;
    type Read<'a>;

    /// Open the base keyspace `name` in the build.
    fn open(name: Arc<str>, build: &Shared) -> Self::Head;

    /// The handle, once `declare` has run.
    fn finish(head: Self::Head) -> Self::Handle;

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a>;

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a>;
}

/// A register-shaped source's stream head and keyspace in the build.
fn open_register<K: Data, V: Data>(
    name: &Arc<str>,
    build: &Shared,
    id: KeyspaceId<K, Mergeless<V>>,
) -> Keyed<K, V> {
    let mut b = build.borrow_mut();
    let node = b.graph.table(id);
    let keyed = DynKeyed::new(b.graph.state(node));
    drop(b);
    let changes = Changes::at(node.rows());
    let origin = KeyedOrigin::register(keyed, changes.clone(), Arc::new(RegisterReader::plain(id)));
    Keyed::at_register(changes, origin, build.clone(), name.clone())
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

/// One value per key, keyed by the caller: last writer wins.
pub struct Table<K, V>(PhantomData<fn() -> (K, V)>);

/// The builder head of a [`Table`]: the table as a [`Keyed`] stream,
/// and its constraints.
pub struct TableHead<K: Data, V: Data> {
    head: Keyed<K, V>,
    pub(crate) name: Arc<str>,
    pub(crate) raw: raw::Table<K, V>,
    pub(crate) checks: Checks<V>,
    pub(crate) alts: Rc<RefCell<Vec<Box<dyn AltKey<K, V>>>>>,
    pub(crate) build: Shared,
}

impl<K: Data, V: Data> Deref for TableHead<K, V> {
    type Target = Keyed<K, V>;
    fn deref(&self) -> &Keyed<K, V> {
        &self.head
    }
}

pub struct TableHandle<K: Data, V: Data> {
    raw: raw::Table<K, V>,
    checks: Arc<[Check<V>]>,
    alts: Arc<[Box<dyn AltKey<K, V>>]>,
    reader: RegisterReader<K, Mergeless<V>, V>,
}

/// The write facade of a [`Table`].
pub struct TableWrite<'a, K: Data, V: Data> {
    tx: &'a TxCell<'a>,
    h: &'a TableHandle<K, V>,
}

pub type TableRead<'a, K, V> = Read<'a, K, V>;

impl<K: Data, V: Data> TableWrite<'_, K, V> {
    /// The value of `key`, with this transaction's writes.
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        self.h.raw.get(&mut self.tx.borrow_mut(), key)
    }

    pub fn contains(&self, key: &K) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    /// Set `key` to `value`, replacing what was there.
    pub fn upsert(&self, key: K, value: V) -> Result<()> {
        run_checks(&self.h.checks, &value)?;
        let mut tx = self.tx.borrow_mut();
        if !self.h.alts.is_empty() {
            let old = self.h.raw.get(&mut tx, &key)?;
            self.alts(&mut tx, &key, old.as_ref(), Some(&value))?;
        }
        self.h.raw.upsert(&mut tx, key, value)
    }

    /// Delete `key`. Returns whether it was there.
    pub fn remove(&self, key: &K) -> Result<bool> {
        let mut tx = self.tx.borrow_mut();
        if !self.h.alts.is_empty() {
            let Some(old) = self.h.raw.get(&mut tx, key)? else {
                return Ok(false);
            };
            self.alts(&mut tx, key, Some(&old), None)?;
        }
        self.h.raw.remove(&mut tx, key)
    }

    /// Change `key`'s value in place. Returns whether it was there.
    pub fn update(&self, key: &K, f: impl FnOnce(&mut V)) -> Result<bool> {
        let mut tx = self.tx.borrow_mut();
        let Some(old) = self.h.raw.get(&mut tx, key)? else {
            return Ok(false);
        };
        let mut new = old.clone();
        f(&mut new);
        if new != old {
            run_checks(&self.h.checks, &new)?;
            self.alts(&mut tx, key, Some(&old), Some(&new))?;
            self.h.raw.upsert(&mut tx, key.clone(), new)?;
        }
        Ok(true)
    }

    /// Delete every row.
    pub fn clear(&self) -> Result<()> {
        let keys: Vec<K> = {
            let mut tx = self.tx.borrow_mut();
            tx.table(self.h.raw.id())
                .range(..)
                .map(|r| r.map(|(k, _)| k.into_owned()))
                .collect::<Result<_>>()?
        };
        for k in keys {
            self.remove(&k)?;
        }
        Ok(())
    }

    /// [`upsert`](Self::upsert) every row.
    pub fn extend(&self, rows: impl IntoIterator<Item = (K, V)>) -> Result<()> {
        for (k, v) in rows {
            self.upsert(k, v)?;
        }
        Ok(())
    }

    fn alts(&self, tx: &mut DbTx<'_>, key: &K, old: Option<&V>, new: Option<&V>) -> Result<()> {
        for alt in self.h.alts.iter() {
            alt.apply(tx, key, old, new)?;
        }
        Ok(())
    }
}

impl<V: Data> TableWrite<'_, Id<V>, V> {
    /// Add `value` under a freshly minted id.
    pub fn insert(&self, value: V) -> Result<Id<V>> {
        run_checks(&self.h.checks, &value)?;
        let mut tx = self.tx.borrow_mut();
        let id = tx.mint(self.h.raw.id().raw());
        self.alts(&mut tx, &id, None, Some(&value))?;
        tx.table(self.h.raw.id())
            .absorb(id, crate::store::merge::Change::Insert(value))?;
        Ok(id)
    }

    /// [`insert`](Self::insert) every value.
    pub fn insert_all(&self, values: impl IntoIterator<Item = V>) -> Result<Vec<Id<V>>> {
        values.into_iter().map(|v| self.insert(v)).collect()
    }
}

impl<K: Data, V: Data> SourceDecl for Table<K, V> {
    type Head = TableHead<K, V>;
    type Handle = TableHandle<K, V>;
    type Write<'a> = TableWrite<'a, K, V>;
    type Read<'a> = TableRead<'a, K, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Head {
        let id = build.borrow_mut().id(&name);
        TableHead {
            head: open_register(&name, build, id),
            name,
            raw: raw::Table::new(id.raw()),
            checks: Rc::new(RefCell::new(Vec::new())),
            alts: Rc::new(RefCell::new(Vec::new())),
            build: build.clone(),
        }
    }

    fn finish(head: Self::Head) -> Self::Handle {
        TableHandle {
            raw: head.raw,
            checks: head.checks.take().into(),
            alts: head.alts.take().into(),
            reader: RegisterReader::plain(head.raw.id()),
        }
    }

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a> {
        TableWrite { tx, h }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        Read::new(&h.reader, at)
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// An append-only log: each row gets a minted [`Id`], increasing across
/// commits and within one. What it keeps is set with
/// [`EventsHead::retain`]; evicted rows leave as deletes, so every view
/// over the log sees them go.
pub struct Events<V>(PhantomData<fn() -> V>);

pub struct EventsHead<V: Data> {
    head: Keyed<Id<V>, V>,
    pub(crate) name: Arc<str>,
    pub(crate) raw: raw::Events<V>,
    pub(crate) checks: Checks<V>,
    build: Shared,
}

impl<V: Data> Deref for EventsHead<V> {
    type Target = Keyed<Id<V>, V>;
    fn deref(&self) -> &Keyed<Id<V>, V> {
        &self.head
    }
}

impl<V: Data> EventsHead<V> {
    /// What the log keeps; [`Retain::Forever`] by default.
    pub fn retain(&self, retain: Retain) {
        let raw = self.raw;
        self.build
            .borrow_mut()
            .after_open
            .push(Box::new(move |db| db.retain(raw, retain)));
    }
}

pub struct EventsHandle<V: Data> {
    raw: raw::Events<V>,
    checks: Arc<[Check<V>]>,
    reader: RegisterReader<Id<V>, Mergeless<V>, V>,
}

pub struct EventsWrite<'a, V: Data> {
    tx: &'a TxCell<'a>,
    h: &'a EventsHandle<V>,
}

pub type EventsRead<'a, V> = Read<'a, Id<V>, V>;

impl<V: Data> EventsWrite<'_, V> {
    /// Append `value`, returning its id.
    pub fn push(&self, value: V) -> Result<Id<V>> {
        run_checks(&self.h.checks, &value)?;
        self.h.raw.push(&mut self.tx.borrow_mut(), value)
    }

    pub fn get(&self, id: Id<V>) -> Result<Option<V>> {
        self.h.raw.get(&mut self.tx.borrow_mut(), id)
    }
}

impl<V: Data> SourceDecl for Events<V> {
    type Head = EventsHead<V>;
    type Handle = EventsHandle<V>;
    type Write<'a> = EventsWrite<'a, V>;
    type Read<'a> = EventsRead<'a, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Head {
        let id = build.borrow_mut().id(&name);
        EventsHead {
            head: open_register(&name, build, id),
            name,
            raw: raw::Events::new(id.raw()),
            checks: Rc::new(RefCell::new(Vec::new())),
            build: build.clone(),
        }
    }

    fn finish(head: Self::Head) -> Self::Handle {
        EventsHandle {
            raw: head.raw,
            checks: head.checks.take().into(),
            reader: RegisterReader::plain(head.raw.id()),
        }
    }

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a> {
        EventsWrite { tx, h }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        Read::new(&h.reader, at)
    }
}

// ---------------------------------------------------------------------------
// Set
// ---------------------------------------------------------------------------

/// Distinct membership: a [`Table`] of `()`, and a [`Keyed<T, ()>`]
/// stream.
pub struct Set<T>(PhantomData<fn() -> T>);

pub struct SetHead<T: Data> {
    head: Keyed<T, ()>,
    pub(crate) name: Arc<str>,
    raw: raw::Set<T>,
    pub(crate) checks: Checks<T>,
}

impl<T: Data> Deref for SetHead<T> {
    type Target = Keyed<T, ()>;
    fn deref(&self) -> &Keyed<T, ()> {
        &self.head
    }
}

pub struct SetHandle<T: Data> {
    raw: raw::Set<T>,
    checks: Arc<[Check<T>]>,
    reader: RegisterReader<T, Mergeless<()>, ()>,
}

pub struct SetWrite<'a, T: Data> {
    tx: &'a TxCell<'a>,
    h: &'a SetHandle<T>,
}

impl<T: Data> SetWrite<'_, T> {
    /// Add `value`. Returns whether it was new.
    pub fn insert(&self, value: T) -> Result<bool> {
        run_checks(&self.h.checks, &value)?;
        self.h.raw.insert(&mut self.tx.borrow_mut(), value)
    }

    /// Remove `value`. Returns whether it was there.
    pub fn remove(&self, value: &T) -> Result<bool> {
        self.h.raw.remove(&mut self.tx.borrow_mut(), value)
    }

    pub fn contains(&self, value: &T) -> Result<bool> {
        self.h.raw.contains(&mut self.tx.borrow_mut(), value)
    }

    /// Remove every member.
    pub fn clear(&self) -> Result<()> {
        let members: Vec<T> = {
            let mut tx = self.tx.borrow_mut();
            tx.table(self.h.raw.id())
                .range(..)
                .map(|r| r.map(|(k, _)| k.into_owned()))
                .collect::<Result<_>>()?
        };
        for m in members {
            self.remove(&m)?;
        }
        Ok(())
    }

    /// [`insert`](Self::insert) every value.
    pub fn extend(&self, values: impl IntoIterator<Item = T>) -> Result<()> {
        for v in values {
            self.insert(v)?;
        }
        Ok(())
    }
}

impl<T: Data> SourceDecl for Set<T> {
    type Head = SetHead<T>;
    type Handle = SetHandle<T>;
    type Write<'a> = SetWrite<'a, T>;
    type Read<'a> = SetRead<'a, T>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Head {
        let id = build.borrow_mut().id(&name);
        SetHead {
            head: open_register(&name, build, id),
            name,
            raw: raw::Set::new(id.raw()),
            checks: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn finish(head: Self::Head) -> Self::Handle {
        SetHandle {
            raw: head.raw,
            checks: head.checks.take().into(),
            reader: RegisterReader::plain(head.raw.id()),
        }
    }

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a> {
        SetWrite { tx, h }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        SetRead::new(Read::new(&h.reader, at))
    }
}

// ---------------------------------------------------------------------------
// Cell
// ---------------------------------------------------------------------------

/// One value: a [`Table`] keyed by `()`, and a [`Value<V>`] stream.
pub struct Cell<V>(PhantomData<fn() -> V>);

pub struct CellHead<V: Data> {
    head: Value<V>,
    pub(crate) name: Arc<str>,
    raw: raw::Cell<V>,
    pub(crate) checks: Checks<V>,
}

impl<V: Data> Deref for CellHead<V> {
    type Target = Value<V>;
    fn deref(&self) -> &Value<V> {
        &self.head
    }
}

pub struct CellHandle<V: Data> {
    raw: raw::Cell<V>,
    checks: Arc<[Check<V>]>,
    reader: RegisterReader<(), Mergeless<V>, V>,
}

pub struct CellWrite<'a, V: Data> {
    tx: &'a TxCell<'a>,
    h: &'a CellHandle<V>,
}

pub type CellRead<'a, V> = ValueRead<'a, V>;

impl<V: Data> CellWrite<'_, V> {
    pub fn set(&self, value: V) -> Result<()> {
        run_checks(&self.h.checks, &value)?;
        self.h.raw.set(&mut self.tx.borrow_mut(), value)
    }

    pub fn clear(&self) -> Result<()> {
        self.h.raw.clear(&mut self.tx.borrow_mut())
    }

    pub fn get(&self) -> Result<Option<V>> {
        self.h.raw.get(&mut self.tx.borrow_mut())
    }
}

impl<V: Data> SourceDecl for Cell<V> {
    type Head = CellHead<V>;
    type Handle = CellHandle<V>;
    type Write<'a> = CellWrite<'a, V>;
    type Read<'a> = CellRead<'a, V>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Head {
        let id = build.borrow_mut().id(&name);
        CellHead {
            head: open_register(&name, build, id),
            name,
            raw: raw::Cell::new(id.raw()),
            checks: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn finish(head: Self::Head) -> Self::Handle {
        CellHandle {
            raw: head.raw,
            checks: head.checks.take().into(),
            reader: RegisterReader::plain(head.raw.id()),
        }
    }

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a> {
        CellWrite { tx, h }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        ValueRead::new(Read::new(&h.reader, at))
    }
}

// ---------------------------------------------------------------------------
// Multiset
// ---------------------------------------------------------------------------

/// Rows with signed multiplicities, kept as each row's count, and a
/// [`Stream<T>`] of them. Adding reads nothing.
pub struct Multiset<T>(PhantomData<fn() -> T>);

pub struct MultisetHead<T: Data> {
    head: Stream<T>,
    pub(crate) name: Arc<str>,
    raw: raw::Multiset<T>,
    pub(crate) checks: Checks<T>,
}

impl<T: Data> Deref for MultisetHead<T> {
    type Target = Stream<T>;
    fn deref(&self) -> &Stream<T> {
        &self.head
    }
}

pub struct MultisetHandle<T: Data> {
    raw: raw::Multiset<T>,
    checks: Arc<[Check<T>]>,
    reader: RegisterReader<T, Tally, Delta>,
}

pub struct MultisetWrite<'a, T: Data> {
    tx: &'a TxCell<'a>,
    h: &'a MultisetHandle<T>,
}

impl<T: Data> MultisetWrite<'_, T> {
    /// Add `times` of `row`; negative to take away.
    pub fn add(&self, row: T, times: Delta) -> Result<()> {
        run_checks(&self.h.checks, &row)?;
        self.h.raw.add(&mut self.tx.borrow_mut(), row, times)
    }

    /// Take away `times` of `row`.
    pub fn remove(&self, row: T, times: Delta) -> Result<()> {
        self.h.raw.remove(&mut self.tx.borrow_mut(), row, times)
    }

    /// How many of `row` there are.
    pub fn get(&self, row: &T) -> Result<Delta> {
        self.h.raw.get(&mut self.tx.borrow_mut(), row)
    }

    /// Take away every `row`.
    pub fn clear(&self, row: &T) -> Result<()> {
        self.h.raw.clear(&mut self.tx.borrow_mut(), row)
    }
}

impl<T: Data> SourceDecl for Multiset<T> {
    type Head = MultisetHead<T>;
    type Handle = MultisetHandle<T>;
    type Write<'a> = MultisetWrite<'a, T>;
    type Read<'a> = MultisetRead<'a, T>;

    fn open(name: Arc<str>, build: &Shared) -> Self::Head {
        let mut b = build.borrow_mut();
        let id: KeyspaceId<T, Tally> = b.id(&name);
        let node = b.graph.multiset(id);
        drop(b);
        MultisetHead {
            head: Stream::rooted(Root::at(node), build.clone(), name.clone()),
            name,
            raw: raw::Multiset::new(id.raw()),
            checks: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn finish(head: Self::Head) -> Self::Handle {
        MultisetHandle {
            raw: head.raw,
            checks: head.checks.take().into(),
            reader: RegisterReader::tally(head.raw.id()),
        }
    }

    fn write<'a>(h: &'a Self::Handle, tx: &'a TxCell<'a>) -> Self::Write<'a> {
        MultisetWrite { tx, h }
    }

    fn read<'a>(h: &'a Self::Handle, at: At<'a>) -> Self::Read<'a> {
        MultisetRead::new(Read::new(&h.reader, at))
    }
}
