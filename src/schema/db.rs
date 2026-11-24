//! The database a schema opens as: [`DbBuilder`] and [`Db`].

use std::{path::Path, rc::Rc, sync::Arc};

use super::{At, CommitFeed, Declare, Schema, TxCell};
use crate::{
    database::Db as RawDb,
    dataflow::Build,
    error::Result,
    store::{DurabilityPolicy, Snapshot, Store, StoreBuilder, StoreOptions},
};

/// Configure and open a database of schema `S`.
pub struct DbBuilder<S: Schema> {
    store: StoreBuilder,
    version: u64,
    commit_hooks: Vec<CommitHook<S>>,
    change_hooks: Vec<ChangeHook<S>>,
    checkpoint_hooks: Vec<Box<dyn FnMut(u64) + Send>>,
}

type ChangeHook<S> = Box<dyn FnMut(&<S as Schema>::Changes, &<S as Schema>::Read<'_>) + Send>;

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
            change_hooks: Vec::new(),
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

    /// Record the schema's layouts over those the store was written
    /// with, where they differ, instead of refusing to open. See
    /// [`StoreBuilder::adopt_layouts`].
    pub fn adopt_layouts(mut self) -> Self {
        self.store = self.store.adopt_layouts();
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

    /// Call `hook` after every commit that changed a source, on the
    /// committing thread, with what it changed and the committed state.
    /// See [`ChangeSet`](super::ChangeSet).
    pub fn on_changes(
        mut self,
        hook: impl FnMut(&S::Changes, &S::Read<'_>) + Send + 'static,
    ) -> Self {
        self.change_hooks.push(Box::new(hook));
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
        if !self.change_hooks.is_empty() {
            let mut hooks = self.change_hooks;
            let mut feed = {
                let snap = inner.snapshot();
                S::feed(&S::read(&handles, At::Snapshot(&snap)))
            };
            let handles = handles.clone();
            inner.on_commit(move |snap| {
                let read = S::read(&handles, At::Snapshot(snap));
                while let Some(epoch) = feed.epoch() {
                    let changes = feed.take(epoch);
                    for hook in &mut hooks {
                        hook(&changes, &read);
                    }
                }
            });
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
    /// graph, as [`Error::Rejected`](crate::error::Error::Rejected). Blocks
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
