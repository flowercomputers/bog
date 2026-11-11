//! [`Events`]: an append-only log keyed by minted ids, with retention.

use super::*;

/// An append-only log: each row gets a minted [`Id`], increasing across
/// commits and within one. What is kept is set with
/// [`Db::retain`](crate::raw::Db::retain), and evicted rows leave as deletes
/// in the transaction that pushed them out, so the graph sees them go.
pub struct Events<V: Value>(KeyspaceId<Id<V>, Mergeless<V>>);
handle!([V: Value] Events[V], Id<V>, Mergeless<V>);

impl<V: Value> Events<V> {
    /// Add it to `graph` as a source.
    pub fn source(&self, graph: &mut OpDag) -> TableNode<Id<V>, V> {
        graph.table(self.0)
    }

    /// Append `value`, returning its id.
    pub fn push(&self, tx: &mut DbTx<'_>, value: V) -> Result<Id<V>> {
        let id = tx.mint(self.0.raw());
        tx.table(self.0).absorb(id, Change::Insert(value))?;
        Ok(id)
    }

    pub fn get(&self, tx: &mut DbTx<'_>, id: Id<V>) -> Result<Option<V>> {
        Ok(tx.table(self.0).get(&id)?.map(|v| v.into_owned()))
    }
}

/// What an [`Events`] log keeps. Evaluated against the log itself, never
/// a wall clock, so replaying it gives the same state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retain {
    Forever,
    /// At most this many rows; the oldest go first.
    Count(usize),
    /// The rows pushed in the last this many epochs, counting back from
    /// the newest row's.
    Epochs(u64),
}

/// A retention policy applied to one log at every commit.
pub(crate) trait Retention: Send {
    /// The keyspace it keeps.
    fn keyspace(&self) -> u64;
    /// Evict what the policy no longer keeps, given how many rows this
    /// transaction pushed.
    fn apply(&mut self, tx: &mut WriteTx<'_>, pushed: u32) -> Result<()>;
    /// The transaction did not commit: forget what was learned from it.
    fn reset(&mut self);
}

pub(crate) struct EventsRetention<V: Value> {
    pub id: KeyspaceId<Id<V>, Mergeless<V>>,
    pub retain: Retain,
    /// Rows in the log as of the last commit; counted on first use, then
    /// kept up with each commit's pushes and evictions.
    pub live: Option<usize>,
    /// Whether the log already satisfies the policy, so a commit that
    /// pushes nothing has nothing to evict.
    pub settled: bool,
}

impl<V: Value> Retention for EventsRetention<V> {
    fn keyspace(&self) -> u64 {
        self.id.raw()
    }

    fn apply(&mut self, tx: &mut WriteTx<'_>, pushed: u32) -> Result<()> {
        if pushed == 0 && self.settled {
            return Ok(());
        }
        let mut ks = tx.keyspace(self.id);
        let evict: Vec<Id<V>> = match self.retain {
            Retain::Forever => return Ok(()),
            Retain::Count(keep) => {
                let live = match self.live {
                    Some(live) => live + pushed as usize,
                    None => {
                        let mut n = 0;
                        for row in ks.range(..) {
                            row?;
                            n += 1;
                        }
                        n
                    }
                };
                let excess = live.saturating_sub(keep);
                self.live = Some(live - excess);
                ks.range(..)
                    .take(excess)
                    .map(|row| row.map(|(k, _)| *k))
                    .collect::<Result<_>>()?
            }
            Retain::Epochs(keep) => {
                let Some(newest) = ks.range_rev(..).next() else {
                    return Ok(());
                };
                let newest = *newest?.0;
                let first = (newest.epoch() + 1).saturating_sub(keep);
                ks.range((Bound::Unbounded, Bound::Excluded(Id::minted(first, 0))))
                    .map(|row| row.map(|(k, _)| *k))
                    .collect::<Result<_>>()?
            }
        };
        for id in evict {
            ks.set(id, None)?;
        }
        self.settled = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.live = None;
        self.settled = false;
    }
}
