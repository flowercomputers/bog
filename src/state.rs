//! What a graph node's children can read of its output.
//!
//! A node's delta is what changed in one run. Its *state* is its whole
//! output: a keyspace it owns, or a view computed over its parent's
//! state. Children get their parents' state handles when they are
//! built (see [`OpDag::binary_with`](crate::graph::OpDag::binary_with)),
//! so an operator reads what already exists rather than keeping a copy.
//! A handle reads the state as it is now ([`When::New`]) or as it was at
//! the start of the run ([`When::Old`]); parents run before children,
//! so `New` includes the parent's delta and `Old` excludes it.

use std::{
    borrow::Cow,
    sync::{Arc, Mutex, PoisonError},
};

use crate::{
    Bag, Buffer, Delta, Mergeless, Register,
    codec::{Codec, OrderedCodec},
    error::Result,
    merge::Merge,
    store::StoreBuilder,
    tx::{KeyspaceId, WriteTx},
};

/// Which state to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum When {
    /// As of the start of this graph run.
    Old,
    /// Including everything written so far.
    New,
}

/// A node's whole output, grouped by key: for each key, a multiset of
/// values.
pub trait State: Clone + Send + 'static {
    type Key: Ord + Clone + Send + 'static;
    type Value: Clone + Send + 'static;
    /// The node's delta, whose rows this state accumulates.
    type Delta: Buffer;

    /// Each row of `delta`, in key order, with its multiplicity.
    fn rows(delta: &Self::Delta, f: &mut dyn FnMut(&Self::Key, &Self::Value, Delta));

    /// Visit `key`'s values, ascending, each with its multiplicity, until
    /// `f` returns `false`.
    fn group(
        &self,
        tx: &mut WriteTx,
        key: &Self::Key,
        when: When,
        f: &mut dyn FnMut(&Self::Value, Delta) -> bool,
    ) -> Result<()>;

    /// [`group`](Self::group), descending.
    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &Self::Key,
        when: When,
        f: &mut dyn FnMut(&Self::Value, Delta) -> bool,
    ) -> Result<()> {
        let mut all = Vec::new();
        self.group(tx, key, when, &mut |v, d| {
            all.push((v.clone(), d));
            true
        })?;
        for (v, d) in all.iter().rev() {
            if !f(v, *d) {
                break;
            }
        }
        Ok(())
    }

    /// Every value of each of `keys`, which ascend, as
    /// `f(key, value, multiplicity)`. One pass where the state allows.
    fn groups(
        &self,
        tx: &mut WriteTx,
        keys: &[Self::Key],
        when: When,
        f: &mut dyn FnMut(&Self::Key, &Self::Value, Delta),
    ) -> Result<()> {
        for k in keys {
            self.group(tx, k, when, &mut |v, d| {
                f(k, v, d);
                true
            })?;
        }
        Ok(())
    }

    /// The total multiplicity of `key`'s values.
    fn total(&self, tx: &mut WriteTx, key: &Self::Key, when: When) -> Result<Delta> {
        let mut total: Delta = 0;
        self.group(tx, key, when, &mut |_, d| {
            total = total.checked_add(d).expect("multiplicity overflowed");
            true
        })?;
        Ok(total)
    }

    /// Whether `key` is present: its total multiplicity is positive.
    fn exists(&self, tx: &mut WriteTx, key: &Self::Key, when: When) -> Result<bool> {
        Ok(self.total(tx, key, when)? > 0)
    }
}

/// A state holding at most one value per key: a register.
pub trait KeyedState: State {
    fn get(&self, tx: &mut WriteTx, key: &Self::Key, when: When) -> Result<Option<Self::Value>>;

    /// Each change in `delta`, as the key's value before and after.
    fn changes(
        delta: &Self::Delta,
        f: &mut dyn FnMut(&Self::Key, Option<&Self::Value>, Option<&Self::Value>),
    );
}

/// Rows of a register's changes: the old value leaves, the new arrives.
fn register_rows<K, M: Merge>(delta: &Register<K, M>, f: &mut dyn FnMut(&K, &M::Value, Delta))
where
    K: Ord,
{
    for (k, c) in delta.iter() {
        if let Some(o) = c.old_value() {
            f(k, o, -1);
        }
        if let Some(n) = c.new_value() {
            f(k, n, 1);
        }
    }
}

fn register_changes<K: Ord, M: Merge>(
    delta: &Register<K, M>,
    f: &mut dyn FnMut(&K, Option<&M::Value>, Option<&M::Value>),
) {
    for (k, c) in delta.iter() {
        f(k, c.old_value(), c.new_value());
    }
}

/// Which keys of a table a rebuild has delivered so far. Outside a
/// rebuild, all of them.
#[derive(Clone, Debug)]
pub(crate) enum Delivered<K> {
    All,
    /// Keys up to `old` were delivered before this run, and up to `new`
    /// by its end; `None` is none.
    UpTo {
        old: Option<K>,
        new: Option<K>,
    },
}

impl<K: Ord> Delivered<K> {
    fn shows(&self, key: &K, when: When) -> bool {
        match self {
            Delivered::All => true,
            Delivered::UpTo { old, new } => {
                let bound = match when {
                    When::Old => old,
                    When::New => new,
                };
                bound.as_ref().is_some_and(|b| key <= b)
            }
        }
    }
}

/// The state of a node that owns a register keyspace: a table, or an
/// operator's output.
pub struct Stored<K, M: Merge> {
    id: KeyspaceId<K, M>,
    /// For a table: what a rebuild has delivered, so its children see
    /// the table grow chunk by chunk, as its deltas say it does.
    delivered: Option<Arc<Mutex<Delivered<K>>>>,
}

impl<K, M: Merge> Clone for Stored<K, M> {
    fn clone(&self) -> Self {
        Stored {
            id: self.id,
            delivered: self.delivered.clone(),
        }
    }
}

impl<K, M: Merge> Stored<K, M> {
    pub(crate) fn new(id: KeyspaceId<K, M>) -> Self {
        Stored {
            id,
            delivered: None,
        }
    }

    pub(crate) fn table(id: KeyspaceId<K, M>, delivered: Arc<Mutex<Delivered<K>>>) -> Self {
        Stored {
            id,
            delivered: Some(delivered),
        }
    }

    pub fn id(&self) -> KeyspaceId<K, M> {
        self.id
    }
}

impl<K, M> State for Stored<K, M>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge + Send,
    M::Value: Codec + Sync + Send,
    M::Diff: Send,
{
    type Key = K;
    type Value = M::Value;
    type Delta = Register<K, M>;

    fn rows(delta: &Register<K, M>, f: &mut dyn FnMut(&K, &M::Value, Delta)) {
        register_rows(delta, f)
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&M::Value, Delta) -> bool,
    ) -> Result<()> {
        if let Some(v) = self.get(tx, key, when)? {
            f(&v, 1);
        }
        Ok(())
    }
}

impl<K, M> KeyedState for Stored<K, M>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    M: Merge + Send,
    M::Value: Codec + Sync + Send,
    M::Diff: Send,
{
    fn get(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Option<M::Value>> {
        if let Some(d) = &self.delivered
            && !d
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .shows(key, when)
        {
            return Ok(None);
        }
        let mut ks = tx.keyspace(self.id);
        let v = match when {
            When::New => ks.get(key)?,
            When::Old => ks.get_old(key)?,
        };
        Ok(v.map(Cow::into_owned))
    }

    fn changes(
        delta: &Register<K, M>,
        f: &mut dyn FnMut(&K, Option<&M::Value>, Option<&M::Value>),
    ) {
        register_changes(delta, f)
    }
}

/// A bound or a value, for keys that group: `Lo < Val(v) < Hi` for every
/// `v`, so `(k, Lo)..=(k, Hi)` spans exactly the keys `(k, Val(_))`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edge<V> {
    Lo,
    Val(V),
    Hi,
}

/// Rows `(K, V)` at their multiplicity, grouped by `K`, as an
/// [`Arrange`](crate::ops::Arrange) node keeps them: stored as
/// `(k, Val(v)) -> count`, so a key's group is the range
/// `(k, Lo)..=(k, Hi)`, read in either direction. A count that reaches
/// zero removes the row.
pub type Arrangement<K, V> = KeyspaceId<(K, Edge<V>), Delta>;

/// The state of a node that keeps an [`Arrangement`], and maybe each
/// key's total multiplicity beside it, which makes
/// [`total`](State::total) and [`exists`](State::exists) one lookup.
pub struct Arranged<K, V> {
    id: Arrangement<K, V>,
    totals: Option<KeyspaceId<K, Delta>>,
}

impl<K, V> Clone for Arranged<K, V> {
    fn clone(&self) -> Self {
        Arranged {
            id: self.id,
            totals: self.totals,
        }
    }
}

impl<K, V> Arranged<K, V> {
    pub(crate) fn new(id: Arrangement<K, V>, totals: Option<KeyspaceId<K, Delta>>) -> Self {
        Arranged { id, totals }
    }
}

fn val<V>(edge: &Edge<V>) -> &V {
    match edge {
        Edge::Val(v) => v,
        _ => unreachable!("arrangements store values"),
    }
}

impl<K, V> State for Arranged<K, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    V: OrderedCodec + Clone + Send + Sync + 'static,
{
    type Key = K;
    type Value = V;
    type Delta = Bag<(K, V)>;

    fn rows(delta: &Bag<(K, V)>, f: &mut dyn FnMut(&K, &V, Delta)) {
        for ((k, v), d) in delta.diffs() {
            f(k, v, d);
        }
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        self.scan(tx, key, when, false, f)
    }

    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        self.scan(tx, key, when, true, f)
    }

    fn groups(
        &self,
        tx: &mut WriteTx,
        keys: &[K],
        when: When,
        f: &mut dyn FnMut(&K, &V, Delta),
    ) -> Result<()> {
        let (Some(first), Some(last)) = (keys.first(), keys.last()) else {
            return Ok(());
        };
        let mut ks = tx.keyspace(self.id);
        let span = (first.clone(), Edge::Lo)..=(last.clone(), Edge::Hi);
        let mut rows = match when {
            When::New => ks.range(span),
            When::Old => ks.range_old(span),
        };
        // the row read past the last group, which may open the next
        let mut ahead: Option<(K, V, Delta)> = None;
        for key in keys {
            match &ahead {
                Some((k, _, _)) if k > key => continue,
                Some((k, _, _)) if k < key => ahead = None,
                _ => {}
            }
            if ahead.is_none() {
                rows.seek(&(key.clone(), Edge::Lo))?;
            }
            if let Some((_, v, d)) = ahead.take() {
                f(key, &v, d);
            }
            for row in rows.by_ref() {
                let (k, d) = row?;
                let (kk, v) = &*k;
                if kk != key {
                    ahead = Some((kk.clone(), val(v).clone(), *d));
                    break;
                }
                f(key, val(v), *d);
            }
        }
        Ok(())
    }

    fn total(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Delta> {
        let Some(totals) = self.totals else {
            let mut total: Delta = 0;
            self.group(tx, key, when, &mut |_, d| {
                total = total.checked_add(d).expect("multiplicity overflowed");
                true
            })?;
            return Ok(total);
        };
        let mut ks = tx.keyspace(totals);
        let t = match when {
            When::New => ks.get(key)?,
            When::Old => ks.get_old(key)?,
        };
        Ok(t.map_or(0, |t| *t))
    }
}

impl<K, V> Arranged<K, V>
where
    K: OrderedCodec + Clone + Send + Sync + 'static,
    V: OrderedCodec + Clone + Send + Sync + 'static,
{
    fn scan(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        rev: bool,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        let mut ks = tx.keyspace(self.id);
        let span = (key.clone(), Edge::Lo)..=(key.clone(), Edge::Hi);
        let rows = match (when, rev) {
            (When::New, false) => ks.range(span),
            (When::New, true) => ks.range_rev(span),
            (When::Old, false) => ks.range_old(span),
            (When::Old, true) => ks.range_old_rev(span),
        };
        for row in rows {
            let (k, d) = row?;
            if !f(val(&k.1), *d) {
                break;
            }
        }
        Ok(())
    }
}

/// The state of a [`Distinct`](crate::Distinct) node: every row whose
/// count is positive, once, under its own key, with value `()`.
pub struct Counted<T> {
    id: KeyspaceId<T, Delta>,
}

impl<T> Clone for Counted<T> {
    fn clone(&self) -> Self {
        Counted { id: self.id }
    }
}

impl<T> Counted<T> {
    pub(crate) fn new(id: KeyspaceId<T, Delta>) -> Self {
        Counted { id }
    }
}

impl<T> State for Counted<T>
where
    T: OrderedCodec + Clone + Send + Sync + 'static,
{
    type Key = T;
    type Value = ();
    type Delta = Bag<T>;

    fn rows(delta: &Bag<T>, f: &mut dyn FnMut(&T, &(), Delta)) {
        for (t, d) in delta.diffs() {
            f(t, &(), d);
        }
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &T,
        when: When,
        f: &mut dyn FnMut(&(), Delta) -> bool,
    ) -> Result<()> {
        let mut ks = tx.keyspace(self.id);
        let count = match when {
            When::New => ks.get(key)?,
            When::Old => ks.get_old(key)?,
        };
        if count.is_some_and(|c| *c > 0) {
            f(&(), 1);
        }
        Ok(())
    }
}

/// A view of a register's state with every value mapped. Stores nothing:
/// reads map the parent's values as they go.
pub struct Mapped<S: State, W> {
    parent: S,
    f: Arc<dyn Fn(&S::Value) -> W + Send + Sync>,
}

impl<S: State, W> Clone for Mapped<S, W> {
    fn clone(&self) -> Self {
        Mapped {
            parent: self.parent.clone(),
            f: self.f.clone(),
        }
    }
}

impl<S: State, W> Mapped<S, W> {
    pub(crate) fn new(parent: S, f: Arc<dyn Fn(&S::Value) -> W + Send + Sync>) -> Self {
        Mapped { parent, f }
    }
}

impl<S, W> State for Mapped<S, W>
where
    S: KeyedState,
    W: Clone + Send + 'static,
{
    type Key = S::Key;
    type Value = W;
    type Delta = Register<S::Key, Mergeless<W>>;

    fn rows(delta: &Self::Delta, f: &mut dyn FnMut(&S::Key, &W, Delta)) {
        register_rows(delta, f)
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &S::Key,
        when: When,
        f: &mut dyn FnMut(&W, Delta) -> bool,
    ) -> Result<()> {
        if let Some(w) = self.get(tx, key, when)? {
            f(&w, 1);
        }
        Ok(())
    }
}

impl<S, W> KeyedState for Mapped<S, W>
where
    S: KeyedState,
    W: Clone + Send + 'static,
{
    fn get(&self, tx: &mut WriteTx, key: &S::Key, when: When) -> Result<Option<W>> {
        Ok(self.parent.get(tx, key, when)?.map(|v| (self.f)(&v)))
    }

    fn changes(delta: &Self::Delta, f: &mut dyn FnMut(&S::Key, Option<&W>, Option<&W>)) {
        register_changes(delta, f)
    }
}

/// A view of a register's state keeping only values `p` accepts.
pub struct Filtered<S: State> {
    parent: S,
    p: Arc<dyn Fn(&S::Value) -> bool + Send + Sync>,
}

impl<S: State> Clone for Filtered<S> {
    fn clone(&self) -> Self {
        Filtered {
            parent: self.parent.clone(),
            p: self.p.clone(),
        }
    }
}

impl<S: State> Filtered<S> {
    pub(crate) fn new(parent: S, p: Arc<dyn Fn(&S::Value) -> bool + Send + Sync>) -> Self {
        Filtered { parent, p }
    }
}

impl<S: KeyedState> State for Filtered<S> {
    type Key = S::Key;
    type Value = S::Value;
    type Delta = Register<S::Key, Mergeless<S::Value>>;

    fn rows(delta: &Self::Delta, f: &mut dyn FnMut(&S::Key, &S::Value, Delta)) {
        register_rows(delta, f)
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &S::Key,
        when: When,
        f: &mut dyn FnMut(&S::Value, Delta) -> bool,
    ) -> Result<()> {
        if let Some(v) = self.get(tx, key, when)? {
            f(&v, 1);
        }
        Ok(())
    }
}

impl<S: KeyedState> KeyedState for Filtered<S> {
    fn get(&self, tx: &mut WriteTx, key: &S::Key, when: When) -> Result<Option<S::Value>> {
        Ok(self.parent.get(tx, key, when)?.filter(|v| (self.p)(v)))
    }

    fn changes(
        delta: &Self::Delta,
        f: &mut dyn FnMut(&S::Key, Option<&S::Value>, Option<&S::Value>),
    ) {
        register_changes(delta, f)
    }
}

/// A keyspace a graph node owns, and how to declare it to the store.
pub struct Declare {
    pub(crate) id: u64,
    pub(crate) apply: Box<dyn FnOnce(StoreBuilder) -> StoreBuilder + Send>,
}

impl Declare {
    /// A keyspace the node computes, rebuilt when the graph changes.
    pub fn derived<K, M>(id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        Declare {
            id: id.raw(),
            apply: Box::new(move |b| b.derived(id)),
        }
    }

    /// A base table, written by the application.
    pub(crate) fn base<K, M>(id: KeyspaceId<K, M>) -> Self
    where
        K: OrderedCodec + Clone + Send + Sync + 'static,
        M: Merge,
        M::Value: Codec + Sync,
    {
        Declare {
            id: id.raw(),
            apply: Box::new(move |b| b.keyspace(id)),
        }
    }
}
