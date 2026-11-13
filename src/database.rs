//! [`Db`]: a store and the graph that computes its derived state, kept in
//! step.
//!
//! The application writes base tables through a [`DbTx`]; the graph's
//! table sources turn each run's writes into deltas, and every commit
//! runs the graph first, so derived state never lags the tables. When the
//! graph changes, opening rebuilds derived state by reading the tables
//! back in chunks. Sources fed by [`give`](DbTx::give), whose rows the
//! store does not keep, need the application to replay them.
//!
//! The [`source`](crate::source) handles write tables through a `DbTx`
//! with the usual shapes: tables, sets, cells, multisets and event logs.
//! [`subscribe`](Db::subscribe) follows any keyspace's committed changes,
//! base or derived.

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard, PoisonError},
};

use crate::{
    dataflow::Buffer,
    dataflow::graph::{NodeRef, OpDag},
    error::{Error, Result},
    id::Id,
    source::{Events, EventsRetention, Retain, Retention, Value},
    store::codec::{Codec, OrderedCodec},
    store::merge::Merge,
    store::subscribe::Subscription,
    store::txn::{Keyspace, KeyspaceId, WriteTx},
    store::{Snapshot, Store, StoreBuilder},
};

/// Rows a rebuild reads from a table per transaction, by default.
pub const REBUILD_CHUNK: usize = 4096;

/// Replays rows of unstored sources during a rebuild: gives one chunk to
/// the graph, runs it, and returns whether there is more.
pub type ReplayGiven<'r> = dyn FnMut(&mut WriteTx<'_>, &mut OpDag) -> Result<bool> + 'r;

/// A store and its graph. See the module docs.
///
/// Writes take `&self`: the graph is behind a mutex and the store has its
/// own writer lock, so a `Db` is shared between threads as it is, and
/// one writes while others read snapshots.
pub struct Db {
    store: Store,
    graph: Mutex<OpDag>,
    /// What each event log keeps, applied at every commit.
    retention: Mutex<Vec<Box<dyn Retention>>>,
    commit_hooks: Mutex<Vec<Box<dyn FnMut(&Snapshot<'_>) + Send>>>,
}

impl Db {
    /// Open a store with `graph` computing its derived keyspaces, which
    /// the graph declares: tables as base keyspaces, everything its
    /// operators write as derived. If the graph changed since the last
    /// open, derived state is rebuilt from the tables. Fails with
    /// [`Error::RebuildRequired`] if the graph also has unstored sources;
    /// open those with [`open_with`](Self::open_with).
    pub fn open(builder: StoreBuilder, graph: OpDag) -> Result<Db> {
        Self::open_inner(builder, graph, REBUILD_CHUNK, None)
    }

    /// [`open`](Self::open), with `replay` to feed the graph's unstored
    /// sources during a rebuild, after the tables.
    pub fn open_with(
        builder: StoreBuilder,
        graph: OpDag,
        replay: &mut ReplayGiven<'_>,
    ) -> Result<Db> {
        Self::open_inner(builder, graph, REBUILD_CHUNK, Some(replay))
    }

    pub(crate) fn open_inner(
        builder: StoreBuilder,
        mut graph: OpDag,
        chunk: usize,
        mut replay: Option<&mut ReplayGiven<'_>>,
    ) -> Result<Db> {
        let fp = graph.fingerprint();
        let builder = graph.declare(builder).graph(fp);
        // unless the graph was given its own limit
        graph.retain_default(builder.resolved_options().graph_retain);
        let mut started = false;
        let mut next_table = 0;
        let mut tables_done = false;
        let store = builder.build_with(|tx| {
            if !started {
                if graph.has_given_sources() && replay.is_none() {
                    return Err(Error::RebuildRequired);
                }
                started = true;
                graph.rebuild_begin();
            }
            if !tables_done {
                if graph.rebuild_step(tx, chunk, &mut next_table)? {
                    return Ok(true);
                }
                tables_done = true;
                return Ok(replay.is_some());
            }
            let replay = replay.as_mut().expect("replay given when tables are done");
            let more = replay(tx, &mut graph)?;
            tx.failure().map_or(Ok(more), Err)
        })?;
        graph.gauge(store.graph_gauge());
        Ok(Db {
            store,
            graph: Mutex::new(graph),
            retention: Mutex::new(Vec::new()),
            commit_hooks: Mutex::new(Vec::new()),
        })
    }

    /// Run `f` in a write transaction, then the graph over what it wrote,
    /// then commit. If `f` fails, nothing commits and its error is
    /// returned. Blocks while another write is in progress.
    pub fn write<R>(&self, f: impl FnOnce(&mut DbTx<'_>) -> Result<R>) -> Result<R> {
        let mut tx = self.begin()?;
        match f(&mut tx) {
            Ok(r) => self.commit(tx).map(|()| r),
            Err(e) => {
                self.abort(tx);
                Err(e)
            }
        }
    }

    /// Open a write transaction: [`write`](Self::write) in parts, for a
    /// caller that keeps the transaction somewhere a closure cannot.
    /// Blocks while another write is in progress; finish with
    /// [`commit`](Self::commit) or [`abort`](Self::abort).
    pub fn begin(&self) -> Result<DbTx<'_>> {
        let graph = self.graph.lock().unwrap_or_else(PoisonError::into_inner);
        let retention = self
            .retention
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Ok(DbTx {
            tx: self.store.begin()?,
            graph,
            retention,
            minted: HashMap::new(),
            saved: Vec::new(),
        })
    }

    /// Run the graph over what `tx` wrote, then commit it.
    pub fn commit(&self, tx: DbTx<'_>) -> Result<()> {
        let DbTx {
            mut tx,
            mut graph,
            mut retention,
            minted,
            saved: _,
        } = tx;
        // evictions are part of the transaction that caused them
        let mut evicted = Ok(());
        for policy in retention.iter_mut() {
            let pushed = minted.get(&policy.keyspace()).copied().unwrap_or(0);
            evicted = evicted.and_then(|()| policy.apply(&mut tx, pushed));
        }
        graph.run(&mut tx);
        let committed = match evicted.and_then(|()| tx.failure().map_or(Ok(()), Err)) {
            Ok(()) => self.store.finish(tx),
            Err(e) => Err(e),
        };
        let epoch = match committed {
            Ok(epoch) => epoch,
            // committed, though something after it failed
            Err(e @ Error::AfterCommit { .. }) => {
                self.run_hooks();
                return Err(e);
            }
            Err(e) => {
                graph.abort();
                for policy in retention.iter_mut() {
                    policy.reset();
                }
                return Err(e);
            }
        };
        if epoch.is_some() {
            self.run_hooks();
        }
        Ok(())
    }

    /// Drop `tx` without committing it.
    pub fn abort(&self, mut tx: DbTx<'_>) {
        tx.graph.abort();
        for policy in tx.retention.iter_mut() {
            policy.reset();
        }
    }

    fn run_hooks(&self) {
        let mut hooks = self
            .commit_hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if hooks.is_empty() {
            return;
        }
        let snapshot = self.store.snapshot();
        for hook in hooks.iter_mut() {
            hook(&snapshot);
        }
    }

    /// Call `hook` after every commit, on the committing thread, with a
    /// snapshot that includes it.
    pub fn on_commit(&self, hook: impl FnMut(&Snapshot<'_>) + Send + 'static) {
        self.commit_hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::new(hook));
    }

    /// Call `hook` after every checkpoint, with the epoch it covers
    /// through; see [`Store::on_checkpoint`].
    pub fn on_checkpoint(&self, hook: impl FnMut(u64) + Send + 'static) {
        self.store.on_checkpoint(hook);
    }

    /// Follow keyspace `id`, base or derived: the latest snapshot, and
    /// every commit's net change after it; see [`Store::subscribe`].
    pub fn subscribe<K, M>(&self, id: KeyspaceId<K, M>) -> (Snapshot<'_>, Subscription<K, M>)
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Send + Sync,
        M::Diff: Send + Sync,
    {
        self.store.subscribe(id)
    }

    /// Set what `events` keeps, from the next commit on: each commit then
    /// evicts what the policy no longer keeps, as deletes the graph sees.
    /// It is not stored: set it again after opening.
    ///
    /// [`Retain::Count`] counts the rows pushed through the
    /// [`Events`] handle; rows written to its keyspace any other way are
    /// not counted until the next open.
    pub fn retain<V: Value>(&self, events: Events<V>, retain: Retain) {
        let id = events.id();
        let mut retention = self
            .retention
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        retention.retain(|p| p.keyspace() != id.raw());
        if retain != Retain::Forever {
            retention.push(Box::new(EventsRetention {
                id,
                retain,
                live: None,
                settled: false,
            }));
        }
    }

    /// The latest committed state.
    pub fn snapshot(&self) -> Snapshot<'_> {
        self.store.snapshot()
    }

    pub fn store(&self) -> &Store {
        &self.store
    }
}

/// A write transaction on a [`Db`].
pub struct DbTx<'t> {
    tx: WriteTx<'t>,
    graph: MutexGuard<'t, OpDag>,
    retention: MutexGuard<'t, Vec<Box<dyn Retention>>>,
    /// Ids minted so far, by the keyspace they were minted for.
    minted: HashMap<u64, u32>,
    /// `minted` as of each open savepoint, innermost last.
    saved: Vec<HashMap<u64, u32>>,
}

impl<'t> DbTx<'t> {
    /// Base table `id`, to read and write. Writes reach the graph at the
    /// next run: at commit, or at [`run`](Self::run).
    ///
    /// # Panics
    ///
    /// If a graph operator writes `id`: derived state is the graph's.
    pub fn table<K, M>(&mut self, id: KeyspaceId<K, M>) -> Keyspace<'_, 't, K, M>
    where
        K: OrderedCodec + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        assert!(
            self.graph.owner_is_table(id.raw()) != Some(false),
            "keyspace {} is derived; only the graph writes it",
            id.raw()
        );
        self.tx.keyspace(id)
    }

    /// Keyspace `id`, to read, typically derived state. Run the graph
    /// first to see this transaction's writes reflected in it.
    pub fn read<K, M>(&mut self, id: KeyspaceId<K, M>) -> Keyspace<'_, 't, K, M>
    where
        K: OrderedCodec + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        self.tx.keyspace(id)
    }

    /// The next id for `keyspace` in this transaction: its epoch, and one
    /// more than the last minted for that keyspace.
    pub(crate) fn mint<T>(&mut self, keyspace: u64) -> Id<T> {
        let seq = self.minted.entry(keyspace).or_insert(0);
        let id = Id::minted(self.tx.epoch(), *seq);
        *seq += 1;
        id
    }

    /// The epoch this transaction commits as, if it writes anything.
    pub fn epoch(&self) -> u64 {
        self.tx.epoch()
    }

    /// The committed state this transaction started from.
    pub fn snapshot(&self) -> &Snapshot<'t> {
        self.tx.snapshot()
    }

    /// The transaction itself, for raw keyspace access.
    pub fn raw(&mut self) -> &mut WriteTx<'t> {
        &mut self.tx
    }

    /// Stage rows at an unstored source.
    pub fn give<B: Buffer>(&mut self, src: NodeRef<B>, input: &mut B) {
        self.graph.give(src, input);
    }

    /// Open a savepoint: [`rollback`](Self::rollback) returns the
    /// transaction to this point, and [`release`](Self::release) keeps
    /// what was written since. Savepoints nest.
    pub fn savepoint(&mut self) {
        self.tx.savepoint();
        self.saved.push(self.minted.clone());
    }

    /// Close the innermost savepoint by undoing everything since it:
    /// writes to tables, what graph runs derived from them, and ids
    /// minted. A rejection since the savepoint is forgotten too, so the
    /// transaction carries on; any other failure stays and is returned.
    ///
    /// Rows [given](Self::give) to unstored sources and not yet run are
    /// dropped, whether given before the savepoint or after.
    ///
    /// # Panics
    ///
    /// If no savepoint is open.
    pub fn rollback(&mut self) -> Result<()> {
        self.minted = self.saved.pop().expect("rollback without a savepoint");
        self.graph.abort();
        self.tx.rollback()
    }

    /// Close the innermost savepoint, keeping everything since it.
    ///
    /// # Panics
    ///
    /// If no savepoint is open.
    pub fn release(&mut self) {
        self.saved.pop().expect("release without a savepoint");
        self.tx.release();
    }

    /// Run `f` as a unit within this transaction: then the graph, so a
    /// constraint on derived state that `f`'s writes break shows now. If
    /// either fails, everything since the call is undone and the error
    /// returned, and the transaction carries on as if `f` never ran.
    pub fn attempt<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        self.savepoint();
        match f(self).and_then(|r| self.run().map(|()| r)) {
            Ok(r) => {
                self.release();
                Ok(r)
            }
            Err(e) => {
                self.rollback()?;
                Err(e)
            }
        }
    }

    /// Run the graph now, so derived state reflects this transaction's
    /// writes so far. Commit runs it again for any written since.
    pub fn run(&mut self) -> Result<()> {
        self.graph.run(&mut self.tx);
        self.tx.failure().map_or(Ok(()), Err)
    }
}
