//! Constraints: everything that can refuse a transaction.
//!
//! - On a source head, `reject`/`check` run at write time, in the write
//!   method; `foreign_key` is dataflow; `unique_key` is an index the
//!   table maintains and checks at write time.
//! - On a view slot, `reject`/`check` run after the graph, over the rows
//!   the transaction left in the view.
//! - [`unique`](super::Grouped::unique) enforces one row per key as an
//!   operator.
//!
//! Every refusal is an [`Error::Rejected`](crate::Error::Rejected); the
//! predicate forms carry a [`Rejected`].

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    Change, Delta, Mergeless, Nary, Register,
    db::DbTx,
    error::{Error, Result},
    ops::{Reject, Take, dirty_keys_into, read_group},
    state::{Declare as DeclareKeyspace, State, Stored, When},
    tx::{KeyspaceId, WriteTx},
};

use super::{
    Data, GroupedFeed, Keyed, Rejected,
    source::{CellHead, EventsHead, MultisetHead, SetHead, TableHead},
    view::{GroupSlot, MapSlot, OrderedSlot, SlotCore, ValueSlot},
};

type Shared = Arc<dyn std::error::Error + Send + Sync>;

/// Constraint methods every source head shares.
macro_rules! head_constraints {
    ($row:ty) => {
        /// Refuse rows for which `pred` holds, at write time, with a
        /// [`Rejected`] naming this source.
        pub fn reject(&self, pred: impl Fn(&$row) -> bool + Send + Sync + 'static) {
            let name = self.name.clone();
            self.check(move |v| {
                if pred(v) {
                    Err(Rejected::new(&*name, "rejected by predicate"))
                } else {
                    Ok(())
                }
            })
        }

        /// Run `f` on every row written, at write time; its `Err`
        /// refuses the write.
        pub fn check<E>(&self, f: impl Fn(&$row) -> Result<(), E> + Send + Sync + 'static)
        where
            E: std::error::Error + Send + Sync + 'static,
        {
            self.checks
                .borrow_mut()
                .push(Box::new(move |v| f(v).map_err(Error::reject)));
        }
    };
}

impl<K: Data, V: Data> TableHead<K, V> {
    head_constraints!(V);

    /// An alternate key: `key` of every live row must be distinct. A
    /// write that would make two rows share one is refused at the write
    /// call, with a [`Rejected`]. The table keeps an index from the
    /// alternate key to the primary beside it, and the returned stream
    /// is the table re-keyed by `key`: a register, since the index
    /// guarantees it.
    pub fn unique_key<K2, F>(&self, key: F) -> Keyed<K2, V>
    where
        K: Debug,
        K2: Data + Debug,
        F: Fn(&V) -> K2 + Clone + Send + Sync + 'static,
    {
        let mut alts = self.alts.borrow_mut();
        let name = format!("{}.unique{}", self.name, alts.len());
        let mut b = self.build.borrow_mut();
        let index: KeyspaceId<K2, Mergeless<K>> = b.id(&name);
        b.bases.push(DeclareKeyspace::base(index));
        drop(b);
        alts.push(Box::new(Alt {
            table: self.name.clone(),
            key: key.clone(),
            index,
            _p: PhantomData,
        }));
        drop(alts);
        self.rekeyed_unchecked(move |_, v| key(v))
    }

    /// A foreign key: every row's `key` must name a row of `parent`,
    /// and a parent row cannot be removed while a row here references
    /// it. Dataflow: the references are counted per key, so the state
    /// is the size of the parent, and a violation refuses the
    /// transaction after the graph runs.
    pub fn foreign_key<PK, PV>(
        &self,
        parent: &impl GroupedFeed<PK, PV>,
        key: impl Fn(&V) -> PK + Clone + Send + 'static,
    ) where
        PK: Data + Debug,
        PV: Data,
    {
        foreign_key(&self.name, &self.values().map(key), parent)
    }
}

impl<V: Data> EventsHead<V> {
    head_constraints!(V);

    /// A foreign key; see [`TableHead::foreign_key`]. A retained event
    /// keeps its parent referenced until it ages out.
    pub fn foreign_key<PK, PV>(
        &self,
        parent: &impl GroupedFeed<PK, PV>,
        key: impl Fn(&V) -> PK + Clone + Send + 'static,
    ) where
        PK: Data + Debug,
        PV: Data,
    {
        foreign_key(&self.name, &self.values().map(key), parent)
    }
}

impl<T: Data> SetHead<T> {
    head_constraints!(T);
}

impl<V: Data> CellHead<V> {
    head_constraints!(V);
}

impl<T: Data> MultisetHead<T> {
    head_constraints!(T);
}

/// Refuse any transaction after which a key in `children` is one
/// `parent` lacks.
fn foreign_key<K, PV>(
    child: &Arc<str>,
    children: &super::Stream<K, impl super::Chain<Out = K>>,
    parent: &impl GroupedFeed<K, PV>,
) where
    K: Data + Debug,
    PV: Data,
{
    let (dangling, _) = children.dangling(parent);
    let (child, pname) = (child.clone(), parent.label());
    children.shared().borrow_mut().graph.nary(
        &[dangling],
        Reject::new(move |k: &K| {
            Rejected::new(&*child, format!("foreign key to {pname}: no row {k:?}"))
        }),
    );
}

/// An alternate key of a table: an index from it to the primary key,
/// maintained in the table's own transaction and checked on every
/// write.
pub(crate) trait AltKey<K, V>: Send + Sync {
    /// `key` goes from `old` to `new` (`None` is absent). `Err` refuses
    /// the write before anything is recorded.
    fn apply(&self, tx: &mut DbTx<'_>, key: &K, old: Option<&V>, new: Option<&V>) -> Result<()>;
}

struct Alt<K: Data, V, K2: Data, F> {
    table: Arc<str>,
    key: F,
    index: KeyspaceId<K2, Mergeless<K>>,
    _p: PhantomData<fn(&V)>,
}

impl<K, V, K2, F> AltKey<K, V> for Alt<K, V, K2, F>
where
    K: Data + Debug,
    V: Data,
    K2: Data + Debug,
    F: Fn(&V) -> K2 + Send + Sync + 'static,
{
    fn apply(&self, tx: &mut DbTx<'_>, key: &K, old: Option<&V>, new: Option<&V>) -> Result<()> {
        let old_k2 = old.map(&self.key);
        let new_k2 = new.map(&self.key);
        let mut index = tx.table(self.index);
        if let Some(k2) = &new_k2
            && let Some(holder) = index.get(k2)?
            && *holder != *key
        {
            let holder = holder.into_owned();
            return Err(Error::reject(Rejected::new(
                &*self.table,
                format!("unique key {k2:?} is already held by {holder:?}"),
            )));
        }
        if let Some(k2) = old_k2
            && new_k2.as_ref() != Some(&k2)
        {
            index.set(k2, None)?;
        }
        if let Some(k2) = new_k2 {
            index.set(k2, Some(key.clone()))?;
        }
        Ok(())
    }
}

/// Constraints on view slots: post-graph `reject`/`check`.
impl<K: Data, V: Data> SlotCore<K, V> {
    fn check<E>(&self, check: impl Fn(&K, &V) -> Result<(), E> + Send + 'static)
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.guards.borrow_mut().push(Box::new(move |k, v| {
            check(k, v).map_err(|e| Arc::new(e) as Shared)
        }));
    }

    fn reject(&self, pred: impl Fn(&K, &V) -> bool + Send + 'static) {
        let name = self.name.clone();
        self.check(move |k, v| {
            if pred(k, v) {
                Err(Rejected::new(&*name, "rejected by predicate"))
            } else {
                Ok(())
            }
        })
    }
}

macro_rules! slot_constraints {
    () => {
        /// Refuse any transaction that leaves a row for which `pred`
        /// holds. Evaluated after the graph has run, over the rows the
        /// transaction changed.
        pub fn reject(&self, pred: impl Fn(&K, &V) -> bool + Send + 'static) {
            self.core.reject(pred)
        }

        /// Like [`reject`](Self::reject), with `f`'s error.
        pub fn check<E>(&self, f: impl Fn(&K, &V) -> Result<(), E> + Send + 'static)
        where
            E: std::error::Error + Send + Sync + 'static,
        {
            self.core.check(f)
        }
    };
}

impl<K: Data, V: Data> MapSlot<K, V> {
    slot_constraints!();
}

impl<K: Data, V: Data> GroupSlot<K, V> {
    slot_constraints!();
}

impl<K: Data, V: Data> OrderedSlot<K, V> {
    /// See [`MapSlot::reject`].
    pub fn reject(&self, pred: impl Fn(&K, &V) -> bool + Send + 'static) {
        self.map_core().reject(pred)
    }

    /// See [`MapSlot::check`].
    pub fn check<E>(&self, f: impl Fn(&K, &V) -> Result<(), E> + Send + 'static)
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.map_core().check(f)
    }
}

impl<V: Data> ValueSlot<V> {
    /// Refuse any transaction that leaves a value for which `pred`
    /// holds.
    pub fn reject(&self, pred: impl Fn(&V) -> bool + Send + 'static) {
        self.map.core.reject(move |_, v| pred(v))
    }

    /// Like [`reject`](Self::reject), with `f`'s error.
    pub fn check<E>(&self, f: impl Fn(&V) -> Result<(), E> + Send + 'static)
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.map.core.check(move |_, v| f(v))
    }
}

/// Keeps each key's one row in `output`, refusing the transaction when a
/// key is left with any other number of rows.
pub(crate) struct Unique<S: State> {
    name: Arc<str>,
    parent: S,
    output: KeyspaceId<S::Key, Mergeless<S::Value>>,
    dirty: Vec<S::Key>,
    vals: Vec<(S::Value, Delta)>,
}

impl<S: State> Unique<S> {
    pub(crate) fn new(
        name: Arc<str>,
        parent: S,
        output: KeyspaceId<S::Key, Mergeless<S::Value>>,
    ) -> Self {
        Unique {
            name,
            parent,
            output,
            dirty: Vec::new(),
            vals: Vec::new(),
        }
    }
}

impl<S> Unique<S>
where
    S: State,
    S::Key: Data + Debug,
    S::Value: Data,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        batches: &[&S::Delta],
        out: &mut Register<S::Key, Mergeless<S::Value>>,
    ) -> Result<()> {
        self.dirty.clear();
        for b in batches {
            dirty_keys_into::<S>(b, &mut self.dirty);
        }
        if batches.len() > 1 {
            self.dirty.sort();
            self.dirty.dedup();
        }
        for k in &self.dirty {
            read_group(&self.parent, tx, k, When::New, Take::Whole, &mut self.vals)?;
            let new = match &self.vals[..] {
                [] => None,
                [(v, 1)] => Some(v.clone()),
                // a rebuild replays rows accepted when they were written
                _ if tx.rebuilding() => None,
                vals => {
                    let total: Delta = vals.iter().map(|(_, d)| d).sum();
                    tx.reject(Rejected::new(
                        &*self.name,
                        format!(
                            "key {k:?} has {} rows (total multiplicity {total})",
                            vals.len()
                        ),
                    ));
                    return tx.failure().map_or(Ok(()), Err);
                }
            };
            let mut ks = tx.keyspace(self.output);
            let old = ks.get(k)?.map(|o| o.into_owned());
            let change = match (old, new) {
                (None, None) => continue,
                (Some(o), Some(n)) if o == n => continue,
                (None, Some(n)) => Change::Insert(n),
                (Some(o), None) => Change::Delete(o),
                (Some(o), Some(n)) => Change::Replace(o, n),
            };
            ks.absorb(k.clone(), change.clone())?;
            out.absorb_one(k.clone(), change);
        }
        Ok(())
    }
}

impl<S> Nary for Unique<S>
where
    S: State,
    S::Key: Data + Debug,
    S::Value: Data,
{
    type Takes = S::Delta;
    type Gives = Register<S::Key, Mergeless<S::Value>>;
    type State = Stored<S::Key, Mergeless<S::Value>>;

    fn state(&self) -> Self::State {
        Stored::new(self.output)
    }

    fn keyspaces(&self) -> Vec<DeclareKeyspace> {
        vec![DeclareKeyspace::derived(self.output)]
    }

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        fp.write_u64(self.output.raw());
    }

    fn scratch(&mut self, release: bool) -> usize {
        crate::vec_scratch(&mut self.dirty, release) + crate::vec_scratch(&mut self.vals, release)
    }

    fn eval(&mut self, tx: &mut WriteTx, batches: &[&S::Delta], out: &mut Self::Gives) {
        let _ = self.run(tx, batches, out);
    }
}
