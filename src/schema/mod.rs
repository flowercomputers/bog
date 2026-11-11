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
//! [`OrderedView`], [`ValueView`], [`SearchView`]), and the database
//! ([`Db`]).
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
//! [`Error::Rejected`](crate::error::Error::Rejected) carrying the constraint's
//! error, a [`Rejected`] for the predicate forms; recover it with
//! [`Error::rejected`](crate::error::Error::rejected). A write closure refuses
//! its own transaction the same way, with [`Error::reject`](crate::error::Error::reject).
//!
//! # Composition
//!
//! A `#[schema]` field nests another schema whole: its sources, views
//! and constraints, under its own [`Declare`] impl, with its keyspaces
//! named under the field (`ledger.users`). The nested `declare` runs
//! first, so the outer one can build on the nested views.

#![allow(clippy::type_complexity)]

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{database::DbTx, store::Snapshot};

mod db;
pub(crate) mod read;
pub(crate) mod source;
pub(crate) mod subscribe;
pub(crate) mod view;

pub use db::*;
pub use read::*;
pub use source::*;

pub use crate::dataflow::{Build, Shared};
pub use crate::rejected::Rejected;
pub use crate::search::*;
pub use crate::stream::*;
pub use subscribe::*;
pub use view::*;

pub use crate::dataflow::Delta;
pub use crate::id::Id;
pub use crate::source::Retain;

#[cfg(feature = "derive")]
pub use fold3_derive::{Codec, Schema};

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
