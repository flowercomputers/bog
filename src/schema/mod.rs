//! The schema layer: declare sources and views as the fields of one
//! struct, define the views as dataflow over typed stream heads, and get
//! a database whose write and read sides mirror the struct.
//!
//! ```ignore
//! #[derive(Schema)]
//! struct Ledger {
//!     users: Table<Id<User>, User>,
//!     transfers: Events<Transfer>,
//!     #[view]
//!     balances: MapView<Id<User>, i64>,
//! }
//!
//! impl Declare for Ledger {
//!     fn declare(b: &mut LedgerBuilder) {
//!         b.transfers.reject(|t| t.amount == 0);
//!         b.transfers.foreign_key(&b.users, |t| t.from);
//!         let sent = b.transfers.key_by(|t| t.from).map(|t| -t.amount);
//!         let got = b.transfers.key_by(|t| t.to).map(|t| t.amount);
//!         b.balances.define(&sent.concat(&got).sum());
//!         b.balances.reject(|_, balance| *balance < 0);
//!     }
//! }
//!
//! let db = Ledger::memory().build()?;
//! db.write(|tx| tx.users.insert(User { .. }))?;
//! db.read(|r| r.balances.get(&id))?;
//! ```
//!
//! The derive is mechanical: it mirrors the struct's fields into a write
//! struct, a builder with one stream head or view slot per field, a
//! bundle of handles, and a read struct bound to one snapshot or
//! transaction. Everything with behaviour is ordinary code here: the
//! stream kinds and their combinators ([`Stream`], [`Keyed`],
//! [`Grouped`]), the source kinds ([`Table`], [`Events`], [`Set`],
//! [`Cell`], [`Multiset`]), the view kinds ([`MapView`], [`GroupView`],
//! [`OrderedView`], [`ValueView`]), and the database ([`Db`]).
//!
//! # How it maps onto the graph
//!
//! A source is a stored keyspace and a graph source node. A linear chain
//! of combinators (`map`, `filter`, `key_by`, ...) is one fused graph
//! node, built when something consumes it. A stateful combinator
//! (`sum`, `with`, `min`, ...) is a graph node with its own keyspaces,
//! named after the stream it was built from (`transfers.sum`), and reads
//! its inputs' state rather than copying it: a join of two tables keyed
//! alike stores nothing. A view adopts the keyspace of the operator it
//! was defined from where there is one, and otherwise keeps its own.
//!
//! Keyspaces are named by field path (`users`, `ledger.balances`) and
//! derived names; the name hashes to the keyspace id, so renaming a
//! field or a view rebuilds it and nothing else.
//!
//! # Order
//!
//! The graph is built in declaration order, so a view's head is usable
//! only after `define`: define a view before building on it. A view
//! that is never defined is an error at open.
//!
//! # Reads
//!
//! Every read returns a `Result`: tables on disk can fail to read.
//! Scans over a snapshot are lazy; inside a transaction they are
//! collected first. Combinator closures take their rows by reference,
//! since a row is borrowed from the batch it arrives in.
//!
//! # Constraints
//!
//! `reject` and `check` on a source run at write time, in the write
//! method, which returns the error. `foreign_key` and `unique` are
//! dataflow, and `reject`/`check` on a view run after the graph: a
//! violation fails the whole transaction. Every refusal arrives as
//! [`Error::Rejected`](crate::Error::Rejected) carrying the constraint's
//! error, a [`Rejected`] for the predicate forms; recover it with
//! [`Error::rejected`](crate::Error::rejected). A write closure refuses
//! its own transaction the same way, with [`Error::reject`](crate::Error::reject).
//!
//! # Composition
//!
//! A `#[schema]` field nests another schema whole: its sources, views
//! and constraints, under its own [`Declare`] impl, with its keyspaces
//! named under the field (`ledger.users`). The nested `declare` runs
//! first, so the outer one can build on the nested views.

#![allow(clippy::type_complexity)]

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::Path,
    ptr::NonNull,
    rc::Rc,
    sync::Arc,
};

use crate::{
    DurabilityPolicy, Snapshot, Store, StoreBuilder, StoreOptions,
    db::{Db as RawDb, DbTx},
    error::Result,
    graph::{Fingerprint, OpDag},
    merge::Merge,
    state::Declare as DeclareKeyspace,
    tx::KeyspaceId,
};

mod constraints;
mod dynstate;
mod read;
mod source;
mod stream;
mod subscribe;
mod view;

pub use read::*;
pub use source::*;
pub use stream::*;
pub use subscribe::*;
pub use view::*;

pub use crate::Delta;
pub use crate::id::Id;
pub use crate::source::Retain;

#[cfg(feature = "derive")]
pub use fold3_derive::{Codec, Schema};

/// A refused write: which source or view refused, and why. What the
/// predicate forms of the constraints (`reject`, `foreign_key`,
/// `unique`, `unique_key`) refuse with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    pub at: String,
    pub reason: String,
}

impl Rejected {
    pub fn new(at: impl Into<String>, reason: impl Into<String>) -> Self {
        Rejected {
            at: at.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.at, self.reason)
    }
}

impl std::error::Error for Rejected {}

/// What one schema tree builds: the graph, the keyspace names, the
/// views waiting to be defined.
#[doc(hidden)]
pub struct Build {
    pub(crate) graph: OpDag,
    /// Every keyspace by id, for collision checks.
    names: HashMap<u64, String>,
    /// Every operator name handed out.
    taken: HashSet<String>,
    /// Base keyspaces outside the graph: the indexes of alternate keys.
    pub(crate) bases: Vec<DeclareKeyspace>,
    /// Every view slot, with whether it has been defined.
    slots: Vec<(Arc<str>, Rc<std::cell::Cell<bool>>)>,
    /// What to set up on the database once open: retention policies.
    pub(crate) after_open: Vec<Box<dyn FnOnce(&RawDb)>>,
}

/// The build, shared by every head, slot and stream of one tree.
#[doc(hidden)]
pub type Shared = Rc<RefCell<Build>>;

impl Build {
    fn new() -> Shared {
        Rc::new(RefCell::new(Build {
            graph: OpDag::new(),
            names: HashMap::new(),
            taken: HashSet::new(),
            bases: Vec::new(),
            slots: Vec::new(),
            after_open: Vec::new(),
        }))
    }

    /// `base`, or `base#2`, `base#3`, ... if it was handed out before.
    pub(crate) fn unique(&mut self, base: &str) -> String {
        if self.taken.insert(base.to_string()) {
            return base.to_string();
        }
        let mut n = 2;
        loop {
            let name = format!("{base}#{n}");
            if self.taken.insert(name.clone()) {
                return name;
            }
            n += 1;
        }
    }

    /// The keyspace named `name`: its id is the name's hash.
    pub(crate) fn id<K, M: Merge>(&mut self, name: &str) -> KeyspaceId<K, M> {
        let mut fp = Fingerprint::new();
        fp.write_str(name);
        let raw = fp.finish();
        if let Some(other) = self.names.insert(raw, name.to_string()) {
            if other == name {
                panic!("keyspace {name:?} is declared twice");
            }
            panic!("keyspaces {other:?} and {name:?} hash alike; rename one");
        }
        KeyspaceId::new(raw)
    }

    pub(crate) fn slot(&mut self, name: Arc<str>) -> Rc<std::cell::Cell<bool>> {
        let defined = Rc::new(std::cell::Cell::new(false));
        self.slots.push((name, defined.clone()));
        defined
    }

    /// Every view must be defined by the end of `declare`.
    fn check_defined(&self) {
        let undefined: Vec<&str> = self
            .slots
            .iter()
            .filter(|(_, defined)| !defined.get())
            .map(|(name, _)| &**name)
            .collect();
        assert!(
            undefined.is_empty(),
            "views declared but never defined: {undefined:?}"
        );
    }
}

/// A write transaction, shared by the write facades of one
/// transaction: a `RefCell` over a borrowed [`DbTx`], covariant in its
/// lifetime so a facade can name one lifetime for both.
#[doc(hidden)]
pub struct TxCell<'a> {
    tx: NonNull<DbTx<'a>>,
    /// Outstanding borrows: `-1` while mutably borrowed.
    borrows: std::cell::Cell<isize>,
    _p: PhantomData<&'a ()>,
}

impl<'a> TxCell<'a> {
    fn new<'t: 'a>(tx: &'a mut DbTx<'t>) -> Self {
        TxCell {
            tx: NonNull::from(tx),
            borrows: std::cell::Cell::new(0),
            _p: PhantomData,
        }
    }

    pub(crate) fn borrow(&self) -> TxRef<'_, 'a> {
        let n = self.borrows.get();
        assert!(n >= 0, "transaction already borrowed mutably");
        self.borrows.set(n + 1);
        TxRef { cell: self }
    }

    pub(crate) fn borrow_mut(&self) -> TxRefMut<'_, 'a> {
        assert!(self.borrows.get() == 0, "transaction already borrowed");
        self.borrows.set(-1);
        TxRefMut { cell: self }
    }

    /// The epoch the transaction commits as.
    pub fn epoch(&self) -> u64 {
        self.borrow().epoch()
    }
}

pub(crate) struct TxRef<'c, 'a> {
    cell: &'c TxCell<'a>,
}

impl<'a> Deref for TxRef<'_, 'a> {
    type Target = DbTx<'a>;
    fn deref(&self) -> &DbTx<'a> {
        // the cell was made from an exclusive borrow of the tx that
        // outlives it, and the count says nobody holds it mutably
        unsafe { self.cell.tx.as_ref() }
    }
}

impl Drop for TxRef<'_, '_> {
    fn drop(&mut self) {
        self.cell.borrows.set(self.cell.borrows.get() - 1);
    }
}

pub(crate) struct TxRefMut<'c, 'a> {
    cell: &'c TxCell<'a>,
}

impl<'a> Deref for TxRefMut<'_, 'a> {
    type Target = DbTx<'a>;
    fn deref(&self) -> &DbTx<'a> {
        unsafe { self.cell.tx.as_ref() }
    }
}

impl<'a> DerefMut for TxRefMut<'_, 'a> {
    fn deref_mut(&mut self) -> &mut DbTx<'a> {
        // as above, and the count says this is the only borrow
        unsafe { &mut *self.cell.tx.as_ptr() }
    }
}

impl Drop for TxRefMut<'_, '_> {
    fn drop(&mut self) {
        self.cell.borrows.set(0);
    }
}

/// What a read facade reads from.
#[derive(Clone, Copy)]
pub enum At<'a> {
    /// A snapshot: committed state.
    Snapshot(&'a Snapshot<'a>),
    /// The committed state a transaction started from.
    Before(&'a TxCell<'a>),
    /// A transaction, with its own writes: the graph runs before each
    /// read, so derived state reflects them.
    During(&'a TxCell<'a>),
}

/// A struct of sources, `#[view]` fields and `#[schema]` fields nesting
/// other schemas. Derived, never hand-written; the application's half is
/// [`Declare`].
pub trait Schema: Sized + 'static {
    /// Every field's handle: keyspace ids, constraints and readers.
    type Handles: Send + Sync + 'static;
    /// One stream head per source, one slot per view, one nested
    /// builder per `#[schema]` field.
    type Builder;
    /// The write side: a write facade per source, a read facade per
    /// view that sees the transaction's own writes.
    type Write<'a>;
    /// Every field's read facade, bound to one snapshot or transaction.
    type Read<'a>;

    /// Open every field under `prefix` (`""` at the root, `"ledger."`
    /// for a nested schema).
    fn open(prefix: &str, build: &Shared) -> Self::Builder;

    /// Run every nested schema's [`Declare::declare`], depth first, so
    /// the root may build on nested views.
    fn declare_nested(b: &mut Self::Builder);

    /// The handles, once `declare` has run over the builder.
    fn finish(b: Self::Builder) -> Self::Handles;

    fn write<'a>(h: &'a Self::Handles, tx: &'a TxCell<'a>) -> Self::Write<'a>;

    fn read<'a>(h: &'a Self::Handles, at: At<'a>) -> Self::Read<'a>;
}

/// The application's half: constraints and view definitions over the
/// builder the derive generated.
pub trait Declare: Schema {
    fn declare(b: &mut Self::Builder);
}

/// Implemented by the derive to mark a schema struct's fields as used:
/// the struct is a declaration and is never instantiated, which would
/// otherwise trip `dead_code` on every field.
#[doc(hidden)]
pub trait Fields {
    fn __touch(&self);
}

/// Configure and open a database of schema `S`.
pub struct DbBuilder<S: Schema> {
    store: StoreBuilder,
    version: u64,
    commit_hooks: Vec<CommitHook<S>>,
    checkpoint_hooks: Vec<Box<dyn FnMut(u64) + Send>>,
}

type CommitHook<S> = Box<dyn FnMut(&<S as Schema>::Read<'_>) + Send>;

impl<S: Declare> DbBuilder<S> {
    /// A durable database at `dir`, written through on every commit;
    /// see [`durability`](Self::durability) for syncing.
    pub fn open(dir: impl AsRef<Path>) -> Self {
        Self::new(StoreBuilder::new(DurabilityPolicy::RELAXED).dir(dir.as_ref()))
    }

    /// A memory-only database: nothing survives drop.
    pub fn memory() -> Self {
        Self::new(StoreBuilder::new(DurabilityPolicy::NONE))
    }

    /// A database over a store configured by hand. Its keyspaces are
    /// the schema's; declare none.
    pub fn with(store: StoreBuilder) -> Self {
        Self::new(store)
    }

    fn new(store: StoreBuilder) -> Self {
        DbBuilder {
            store,
            version: 0,
            commit_hooks: Vec::new(),
            checkpoint_hooks: Vec::new(),
        }
    }

    /// When commits are written and synced; see [`DurabilityPolicy`].
    pub fn durability(mut self, policy: DurabilityPolicy) -> Self {
        self.store = self.store.durability(policy);
        self
    }

    /// The store's sizes and limits.
    pub fn options(mut self, opts: StoreOptions) -> Self {
        self.store = self.store.options(opts);
        self
    }

    pub fn options_with(mut self, f: impl FnOnce(&mut StoreOptions)) -> Self {
        self.store = self.store.options_with(f);
        self
    }

    /// A version mixed into the graph's fingerprint. Derived state is
    /// rebuilt when the graph's shape changes; bump the version to
    /// rebuild it when only a closure's body did.
    pub fn version(mut self, version: u64) -> Self {
        self.version = version;
        self
    }

    /// Call `hook` after every commit, on the committing thread, with
    /// the committed state.
    pub fn commit_hook(mut self, hook: impl FnMut(&S::Read<'_>) + Send + 'static) -> Self {
        self.commit_hooks.push(Box::new(hook));
        self
    }

    /// Call `hook` after every checkpoint, with the epoch it covers
    /// through, on the thread that wrote it.
    pub fn checkpoint_hook(mut self, hook: impl FnMut(u64) + Send + 'static) -> Self {
        self.checkpoint_hooks.push(Box::new(hook));
        self
    }

    /// Open the store, run [`Declare::declare`] over the generated
    /// builder, and rebuild derived state if the graph changed.
    ///
    /// # Panics
    ///
    /// If a view is left undefined, or used before it was defined.
    pub fn build(self) -> Result<Db<S>> {
        let shared = Build::new();
        let mut builder = S::open("", &shared);
        S::declare_nested(&mut builder);
        S::declare(&mut builder);
        let handles = Arc::new(S::finish(builder));
        let build = Rc::try_unwrap(shared)
            .ok()
            .expect("streams do not outlive declare")
            .into_inner();
        build.check_defined();
        let Build {
            mut graph,
            bases,
            after_open,
            ..
        } = build;
        graph.version(self.version);
        let mut store = self.store;
        for base in bases {
            store = (base.apply)(store);
        }
        let inner = RawDb::open(store, graph)?;
        for setup in after_open {
            setup(&inner);
        }
        for mut hook in self.commit_hooks {
            let handles = handles.clone();
            inner.on_commit(move |snap| hook(&S::read(&handles, At::Snapshot(snap))));
        }
        for hook in self.checkpoint_hooks {
            inner.on_checkpoint(hook);
        }
        Ok(Db { inner, handles })
    }
}

/// A database over a declared schema. Shared between threads as it is:
/// writes serialize on the store's writer lock, reads take snapshots.
pub struct Db<S: Schema> {
    inner: RawDb,
    handles: Arc<S::Handles>,
}

impl<S: Declare> Db<S> {
    /// Run a write transaction: `f` writes through the sources' facades,
    /// then the graph runs over what it wrote, then it commits. `f`'s
    /// `Err` aborts; a constraint refuses at the write call or after the
    /// graph, as [`Error::Rejected`](crate::Error::Rejected). Blocks
    /// while another write is in progress.
    pub fn write<T>(&self, f: impl FnOnce(&mut S::Write<'_>) -> Result<T>) -> Result<T> {
        let mut tx = self.inner.begin()?;
        let r = {
            let cell = TxCell::new(&mut tx);
            let mut w = S::write(&self.handles, &cell);
            f(&mut w)
        };
        match r {
            Ok(v) => self.inner.commit(tx).map(|()| v),
            Err(e) => {
                self.inner.abort(tx);
                Err(e)
            }
        }
    }

    /// Read every source and view from one consistent snapshot.
    pub fn read<T>(&self, f: impl FnOnce(&S::Read<'_>) -> T) -> T {
        let snap = self.inner.snapshot();
        f(&S::read(&self.handles, At::Snapshot(&snap)))
    }

    /// The read facades bound to a snapshot you hold, for reads that
    /// outlive one closure.
    pub fn at<'a>(&'a self, snap: &'a Snapshot<'a>) -> S::Read<'a> {
        S::read(&self.handles, At::Snapshot(snap))
    }

    /// The latest committed state.
    pub fn snapshot(&self) -> Snapshot<'_> {
        self.inner.snapshot()
    }

    /// Write every layer to tables now, and return the epoch the tables
    /// cover through.
    pub fn checkpoint(&self) -> Result<Option<u64>> {
        self.inner.store().checkpoint()
    }

    /// Checkpoint and release the store.
    pub fn close(self) -> Result<()> {
        self.checkpoint()?;
        Ok(())
    }

    /// The store underneath.
    pub fn store(&self) -> &Store {
        self.inner.store()
    }

    /// The database underneath, for raw keyspace access.
    pub fn raw(&self) -> &RawDb {
        &self.inner
    }
}
