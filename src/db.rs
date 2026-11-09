//! [`Db`]: a store and the graph that computes its derived state, kept in
//! step.
//!
//! The application writes base tables through a [`DbTx`]; the graph's
//! table sources turn each run's writes into deltas, and every commit
//! runs the graph first, so derived state never lags the tables. When the
//! graph changes, opening rebuilds derived state by reading the tables
//! back in chunks. Sources fed by [`give`](DbTx::give), whose rows the
//! store does not keep, need the application to replay them.

use crate::{
    Buffer,
    codec::{Codec, OrderedCodec},
    error::{Error, Result},
    graph::{NodeRef, OpDag},
    merge::Merge,
    store::{Snapshot, Store, StoreBuilder},
    tx::{Keyspace, KeyspaceId, WriteTx},
};

/// Rows a rebuild reads from a table per transaction, by default.
pub const REBUILD_CHUNK: usize = 4096;

/// Replays rows of unstored sources during a rebuild: gives one chunk to
/// the graph, runs it, and returns whether there is more.
pub type ReplayGiven<'r> = dyn FnMut(&mut WriteTx<'_>, &mut OpDag) -> Result<bool> + 'r;

/// A store and its graph. See the module docs.
pub struct Db {
    store: Store,
    graph: OpDag,
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
        Ok(Db { store, graph })
    }

    /// Run `f` in a write transaction, then the graph over what it wrote,
    /// then commit. If `f` fails, nothing commits and its error is
    /// returned.
    pub fn write<R>(&mut self, f: impl FnOnce(&mut DbTx<'_, '_>) -> Result<R>) -> Result<R> {
        let mut tx = DbTx {
            tx: self.store.begin()?,
            graph: &mut self.graph,
        };
        let r = match f(&mut tx) {
            Ok(r) => r,
            Err(e) => {
                tx.graph.abort();
                return Err(e);
            }
        };
        tx.graph.run(&mut tx.tx);
        let DbTx { tx, graph } = tx;
        if tx.failure().is_some() {
            graph.abort();
        }
        self.store.finish(tx)?;
        Ok(r)
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
pub struct DbTx<'d, 's> {
    tx: WriteTx<'s>,
    graph: &'d mut OpDag,
}

impl<'s> DbTx<'_, 's> {
    /// Base table `id`, to read and write. Writes reach the graph at the
    /// next run: at commit, or at [`run`](Self::run).
    ///
    /// # Panics
    ///
    /// If a graph operator writes `id`: derived state is the graph's.
    pub fn table<K, M>(&mut self, id: KeyspaceId<K, M>) -> Keyspace<'_, 's, K, M>
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
    pub fn read<K, M>(&mut self, id: KeyspaceId<K, M>) -> Keyspace<'_, 's, K, M>
    where
        K: OrderedCodec + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        self.tx.keyspace(id)
    }

    /// Stage rows at an unstored source.
    pub fn give<B: Buffer>(&mut self, src: NodeRef<B>, input: &mut B) {
        self.graph.give(src, input);
    }

    /// Run the graph now, so derived state reflects this transaction's
    /// writes so far. Commit runs it again for any written since.
    pub fn run(&mut self) -> Result<()> {
        self.graph.run(&mut self.tx);
        self.tx.failure().map_or(Ok(()), Err)
    }
}
