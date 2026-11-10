//! Change subscriptions: every committed transaction's net change to a
//! keyspace, delivered to any number of subscribers.
//!
//! [`Store::subscribe`](crate::Store::subscribe) hands back a snapshot
//! and a [`Subscription`] that starts right after it, so the snapshot's
//! state plus every [`Delta`] received is the live state, with no gap and
//! no double count, whichever thread reads. A keyspace nobody subscribes
//! to costs a commit nothing; a subscribed one costs it a copy of the
//! changes, shared by its subscribers.
//!
//! Deltas are sent once a commit is visible and before the next can
//! start, so they arrive in epoch order. The rebuild of derived state at
//! open is not delivered: nobody can subscribe before the store is open.

use std::{
    any::Any,
    sync::{
        Arc,
        mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError},
    },
    time::{Duration, Instant},
};

use crate::{
    Change,
    codec::{Codec, OrderedCodec},
    merge::Merge,
    tx::{ErasedUnderlay, net_changes},
};

/// One committed transaction's net change to a keyspace.
pub struct Delta<K, M: Merge> {
    /// The epoch the transaction committed as.
    pub epoch: u64,
    /// Every key it changed, in key order, with the value before and
    /// after: an `Insert`, a `Delete` of the old value, or a `Replace`.
    /// In a [`Tally`](crate::Tally) keyspace, an `Insert` of the amount
    /// added, which may be negative.
    pub changes: Arc<Vec<(K, Change<M>)>>,
}

impl<K, M: Merge> Clone for Delta<K, M> {
    fn clone(&self) -> Self {
        Delta {
            epoch: self.epoch,
            changes: self.changes.clone(),
        }
    }
}

/// The store behind a [`Subscription`] has been dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Closed;

impl std::fmt::Display for Closed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the subscribed store has been closed")
    }
}

impl std::error::Error for Closed {}

/// A stream of [`Delta`]s from one keyspace. See the module docs.
pub struct Subscription<K, M: Merge> {
    pub(crate) rx: Receiver<Delta<K, M>>,
    /// Deltas through this epoch were already in the subscriber's
    /// snapshot.
    pub(crate) after: u64,
}

impl<K, M: Merge> Subscription<K, M> {
    fn admit(&self, d: Delta<K, M>) -> Option<Delta<K, M>> {
        (d.epoch > self.after).then_some(d)
    }

    /// The next delta, blocking. `None` once the store is gone.
    pub fn recv(&self) -> Option<Delta<K, M>> {
        loop {
            let d = self.rx.recv().ok()?;
            if let Some(d) = self.admit(d) {
                return Some(d);
            }
        }
    }

    /// The next delta if one is waiting. `Ok(None)` when nothing is
    /// queued; `Err(Closed)` once the store is gone.
    pub fn try_recv(&self) -> Result<Option<Delta<K, M>>, Closed> {
        loop {
            match self.rx.try_recv() {
                Ok(d) => {
                    if let Some(d) = self.admit(d) {
                        return Ok(Some(d));
                    }
                }
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => return Err(Closed),
            }
        }
    }

    /// The next delta within `timeout`. `Ok(None)` on timeout;
    /// `Err(Closed)` once the store is gone.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<Delta<K, M>>, Closed> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(d) => {
                    if let Some(d) = self.admit(d) {
                        return Ok(Some(d));
                    }
                }
                Err(RecvTimeoutError::Timeout) => return Ok(None),
                Err(RecvTimeoutError::Disconnected) => return Err(Closed),
            }
        }
    }

    /// Every delta until the store is gone.
    pub fn iter(&self) -> impl Iterator<Item = Delta<K, M>> + '_ {
        std::iter::from_fn(|| self.recv())
    }
}

/// One keyspace's subscribers, with its types erased for the commit path.
pub(crate) trait Publish: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Copy the net changes a tx is about to commit to this keyspace.
    fn stage(&self, handle: &dyn ErasedUnderlay) -> Box<dyn Any + Send>;
    /// Send what [`stage`](Self::stage) copied, now committed as `epoch`,
    /// to every live subscriber. Returns whether any are left.
    fn publish(&mut self, epoch: u64, staged: Box<dyn Any + Send>) -> bool;
}

pub(crate) struct Senders<K, M: Merge>(pub Vec<Sender<Delta<K, M>>>);

impl<K, M> Publish for Senders<K, M>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Send + Sync,
    M::Diff: Send + Sync,
{
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn stage(&self, handle: &dyn ErasedUnderlay) -> Box<dyn Any + Send> {
        Box::new(Arc::new(net_changes::<K, M>(handle)))
    }

    fn publish(&mut self, epoch: u64, staged: Box<dyn Any + Send>) -> bool {
        let changes = *staged
            .downcast::<Arc<Vec<(K, Change<M>)>>>()
            .expect("staged by the same subscribers");
        self.0.retain(|s| {
            s.send(Delta {
                epoch,
                changes: changes.clone(),
            })
            .is_ok()
        });
        !self.0.is_empty()
    }
}
