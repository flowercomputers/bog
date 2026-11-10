//! Change subscriptions on the schema's read facades: every committed
//! transaction's net change to a source or view, from a snapshot on.
//! See [`subscribe`](crate::subscribe) for the guarantees.

use std::{sync::Arc, time::Duration};

use crate::{
    Change as RawChange, Delta, Tally,
    codec::{Codec, OrderedCodec},
    merge::Merge,
    state::Edge,
    subscribe as raw,
};

pub use crate::subscribe::Closed;

/// One committed transaction's changes to a source or view.
#[derive(Clone, Debug)]
pub struct Commit<T> {
    /// The epoch the transaction committed as.
    pub epoch: u64,
    /// What changed, in key order.
    pub changes: Vec<T>,
}

/// A change to one key of a register-shaped source or view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change<V> {
    Insert(V),
    /// The value that was there.
    Delete(V),
    /// The value before, and after.
    Replace(V, V),
}

impl<V> Change<V> {
    /// The value after the change, if any.
    pub fn after(&self) -> Option<&V> {
        match self {
            Change::Insert(v) | Change::Replace(_, v) => Some(v),
            Change::Delete(_) => None,
        }
    }

    /// The value before the change, if any.
    pub fn before(&self) -> Option<&V> {
        match self {
            Change::Delete(v) | Change::Replace(v, _) => Some(v),
            Change::Insert(_) => None,
        }
    }
}

pub(crate) trait Recv<T>: Send {
    fn recv(&self) -> Option<Commit<T>>;
    fn try_recv(&self) -> Result<Option<Commit<T>>, Closed>;
    fn recv_timeout(&self, timeout: Duration) -> Result<Option<Commit<T>>, Closed>;
}

/// A stream of [`Commit`]s from one source or view. Register-shaped
/// ones deliver `(key, Change<value>)`; grouped ones `(key, value,
/// multiplicity)`.
pub struct Subscription<T> {
    inner: Box<dyn Recv<T>>,
}

impl<T: 'static> Subscription<T> {
    pub(crate) fn new(inner: impl Recv<T> + 'static) -> Self {
        Subscription {
            inner: Box::new(inner),
        }
    }

    /// The next commit, blocking. `None` once the database is gone.
    pub fn recv(&self) -> Option<Commit<T>> {
        self.inner.recv()
    }

    /// The next commit if one is waiting. `Ok(None)` when nothing is
    /// queued; `Err(Closed)` once the database is gone.
    pub fn try_recv(&self) -> Result<Option<Commit<T>>, Closed> {
        self.inner.try_recv()
    }

    /// The next commit within `timeout`. `Ok(None)` on timeout;
    /// `Err(Closed)` once the database is gone.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<Commit<T>>, Closed> {
        self.inner.recv_timeout(timeout)
    }

    /// Every commit until the database is gone.
    pub fn iter(&self) -> impl Iterator<Item = Commit<T>> + '_ {
        std::iter::from_fn(|| self.recv())
    }

    /// The same commits with each change mapped by `f`.
    pub fn map<U: 'static>(self, f: impl Fn(T) -> U + Send + 'static) -> Subscription<U> {
        self.filter_map(move |t| Some(f(t)))
    }

    /// The same commits with each change mapped by `f`, keeping those
    /// it maps to `Some`.
    pub fn filter_map<U: 'static>(
        self,
        f: impl Fn(T) -> Option<U> + Send + 'static,
    ) -> Subscription<U> {
        Subscription::new(Mapped {
            inner: self.inner,
            f: Box::new(f),
        })
    }
}

struct Mapped<T, U> {
    inner: Box<dyn Recv<T>>,
    f: Box<dyn Fn(T) -> Option<U> + Send>,
}

impl<T, U> Recv<U> for Mapped<T, U> {
    fn recv(&self) -> Option<Commit<U>> {
        self.inner.recv().map(|c| self.commit(c))
    }
    fn try_recv(&self) -> Result<Option<Commit<U>>, Closed> {
        Ok(self.inner.try_recv()?.map(|c| self.commit(c)))
    }
    fn recv_timeout(&self, timeout: Duration) -> Result<Option<Commit<U>>, Closed> {
        Ok(self.inner.recv_timeout(timeout)?.map(|c| self.commit(c)))
    }
}

impl<T, U> Mapped<T, U> {
    fn commit(&self, c: Commit<T>) -> Commit<U> {
        Commit {
            epoch: c.epoch,
            changes: c.changes.into_iter().filter_map(&self.f).collect(),
        }
    }
}

/// A register keyspace's commits, each value read through `pick`.
pub(crate) struct Registered<K, M: Merge, V> {
    pub sub: raw::Subscription<K, M>,
    pub pick: Arc<dyn Fn(&M::Value) -> V + Send + Sync>,
}

impl<K, M, V> Registered<K, M, V>
where
    K: Clone,
    M: Merge,
    V: PartialEq,
{
    fn commit(&self, d: raw::Delta<K, M>) -> Commit<(K, Change<V>)> {
        let pick = &self.pick;
        let changes = d
            .changes
            .iter()
            .filter_map(|(k, c)| {
                let change = match c {
                    RawChange::Insert(v) => Change::Insert(pick(v)),
                    RawChange::Delete(o) => Change::Delete(pick(o)),
                    RawChange::Replace(o, n) => {
                        let (o, n) = (pick(o), pick(n));
                        if o == n {
                            return None;
                        }
                        Change::Replace(o, n)
                    }
                    RawChange::Modify(_) => unreachable!("committed changes are resolved"),
                };
                Some((k.clone(), change))
            })
            .collect();
        Commit {
            epoch: d.epoch,
            changes,
        }
    }
}

impl<K, M, V> Recv<(K, Change<V>)> for Registered<K, M, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Send + Sync,
    M::Diff: Send + Sync,
    V: PartialEq,
{
    fn recv(&self) -> Option<Commit<(K, Change<V>)>> {
        self.sub.recv().map(|d| self.commit(d))
    }
    fn try_recv(&self) -> Result<Option<Commit<(K, Change<V>)>>, Closed> {
        Ok(self.sub.try_recv()?.map(|d| self.commit(d)))
    }
    fn recv_timeout(&self, timeout: Duration) -> Result<Option<Commit<(K, Change<V>)>>, Closed> {
        Ok(self.sub.recv_timeout(timeout)?.map(|d| self.commit(d)))
    }
}

/// An arrangement's commits: each row's change in multiplicity.
pub(crate) struct Arranged<K, V> {
    pub sub: raw::Subscription<(K, Edge<V>), Tally>,
}

impl<K: Clone, V: Clone> Arranged<K, V> {
    fn commit(&self, d: raw::Delta<(K, Edge<V>), Tally>) -> Commit<(K, V, Delta)> {
        let changes = d
            .changes
            .iter()
            .map(|((k, v), c)| {
                let (RawChange::Insert(by), Edge::Val(v)) = (c, v) else {
                    unreachable!("an arrangement holds amounts by value")
                };
                (k.clone(), v.clone(), *by)
            })
            .collect();
        Commit {
            epoch: d.epoch,
            changes,
        }
    }
}

impl<K, V> Recv<(K, V, Delta)> for Arranged<K, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    V: OrderedCodec + Clone + Send + Sync + 'static,
{
    fn recv(&self) -> Option<Commit<(K, V, Delta)>> {
        self.sub.recv().map(|d| self.commit(d))
    }
    fn try_recv(&self) -> Result<Option<Commit<(K, V, Delta)>>, Closed> {
        Ok(self.sub.try_recv()?.map(|d| self.commit(d)))
    }
    fn recv_timeout(&self, timeout: Duration) -> Result<Option<Commit<(K, V, Delta)>>, Closed> {
        Ok(self.sub.recv_timeout(timeout)?.map(|d| self.commit(d)))
    }
}
