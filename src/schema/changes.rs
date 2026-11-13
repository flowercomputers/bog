//! A commit as one value: what it wrote to every source of a schema.
//!
//! The derive gives each schema a `<Name>Changes` struct with one vec
//! of changes per source and one nested struct per `#[schema]` field.
//! [`DbBuilder::on_changes`](super::DbBuilder::on_changes) hands one to
//! a hook after every commit; the write facade's `apply` writes one to
//! another database of the same schema, whose graph then derives the
//! same views. [`ChangeSet::invert`] turns one into its undo, and
//! [`ChangeSet::encode`] into bytes for a wire.
//!
//! Views are left out: a replica derives them from the sources. To
//! follow one view's changes, subscribe to it.

use crate::{
    dataflow::Delta,
    store::codec,
};

use super::subscribe::{Change, Commit, Subscription};

/// A change that can be undone by applying another.
pub trait Invert {
    /// The change that undoes this one.
    fn invert(self) -> Self;
}

impl<V> Invert for Change<V> {
    fn invert(self) -> Self {
        match self {
            Change::Insert(v) => Change::Delete(v),
            Change::Delete(v) => Change::Insert(v),
            Change::Replace(old, new) => Change::Replace(new, old),
        }
    }
}

impl<K, V> Invert for (K, Change<V>) {
    fn invert(self) -> Self {
        (self.0, self.1.invert())
    }
}

/// A multiset row's change in multiplicity.
impl<T> Invert for (T, Delta) {
    fn invert(self) -> Self {
        (self.0, -self.1)
    }
}

/// One commit's changes to a schema's sources: the derived
/// `<Name>Changes`.
pub trait ChangeSet: Sized {
    /// The epoch the commit committed as, in the database it came from.
    fn epoch(&self) -> u64;

    /// Whether the commit changed no source.
    fn is_empty(&self) -> bool;

    /// The changes that undo these, under the same epoch.
    fn invert(self) -> Self;

    fn encode_into(&self, out: &mut Vec<u8>);

    fn decode_from(b: &mut &[u8]) -> codec::Result<Self>;

    /// These changes as bytes, for [`decode`](Self::decode) elsewhere.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
    }

    /// The changes [`encode`](Self::encode) wrote.
    fn decode(mut b: &[u8]) -> codec::Result<Self> {
        Self::decode_from(&mut b)
    }
}

/// The commits of a schema's sources, stitched back together by epoch.
/// Only read on the committing thread, after a commit: every source's
/// changes for it are queued by then.
#[doc(hidden)]
pub trait CommitFeed {
    type Changes;
    /// The oldest epoch with changes waiting.
    fn epoch(&mut self) -> Option<u64>;
    /// Take the changes of `epoch`.
    fn take(&mut self, epoch: u64) -> Self::Changes;
}

/// One source's subscription, with its next commit held back until the
/// feed reaches that epoch.
#[doc(hidden)]
pub struct Tap<T> {
    sub: Subscription<T>,
    head: Option<Commit<T>>,
}

impl<T: 'static> Tap<T> {
    pub fn new(sub: Subscription<T>) -> Self {
        Tap { sub, head: None }
    }

    pub fn epoch(&mut self) -> Option<u64> {
        if self.head.is_none() {
            self.head = self.sub.try_recv().ok().flatten();
        }
        self.head.as_ref().map(|c| c.epoch)
    }

    pub fn take(&mut self, epoch: u64) -> Vec<T> {
        if self.epoch() == Some(epoch) {
            self.head.take().map(|c| c.changes).unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

/// The least of the epochs waiting.
#[doc(hidden)]
pub fn first_epoch(epochs: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    epochs.into_iter().flatten().min()
}
