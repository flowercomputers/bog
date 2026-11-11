//! [`DynState`]: a node's state with its type erased, so the schema
//! layer's operators can read any parent through one type.

use std::sync::Arc;

use crate::{
    dataflow::state::{KeyedState, State, When},
    dataflow::{Bag, Delta, Register, Unary},
    error::Result,
    store::merge::Change,
    store::merge::Mergeless,
    store::txn::WriteTx,
};

use super::Data;

/// The object-safe part of [`State`].
pub(crate) trait StateObj<K, V>: Send + Sync {
    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()>;
    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()>;
    fn groups(
        &self,
        tx: &mut WriteTx,
        keys: &[K],
        when: When,
        f: &mut dyn FnMut(&K, &V, Delta),
    ) -> Result<()>;
    fn total(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Delta>;
    fn exists(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<bool>;
}

impl<S: State + Sync> StateObj<S::Key, S::Value> for S {
    fn group(
        &self,
        tx: &mut WriteTx,
        key: &S::Key,
        when: When,
        f: &mut dyn FnMut(&S::Value, Delta) -> bool,
    ) -> Result<()> {
        State::group(self, tx, key, when, f)
    }
    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &S::Key,
        when: When,
        f: &mut dyn FnMut(&S::Value, Delta) -> bool,
    ) -> Result<()> {
        State::group_rev(self, tx, key, when, f)
    }
    fn groups(
        &self,
        tx: &mut WriteTx,
        keys: &[S::Key],
        when: When,
        f: &mut dyn FnMut(&S::Key, &S::Value, Delta),
    ) -> Result<()> {
        State::groups(self, tx, keys, when, f)
    }
    fn total(&self, tx: &mut WriteTx, key: &S::Key, when: When) -> Result<Delta> {
        State::total(self, tx, key, when)
    }
    fn exists(&self, tx: &mut WriteTx, key: &S::Key, when: When) -> Result<bool> {
        State::exists(self, tx, key, when)
    }
}

/// Any grouped state, read through a bag of `(K, V)` rows as its delta.
pub struct DynState<K, V>(Arc<dyn StateObj<K, V>>);

impl<K, V> Clone for DynState<K, V> {
    fn clone(&self) -> Self {
        DynState(self.0.clone())
    }
}

impl<K, V> DynState<K, V> {
    pub(crate) fn new(state: impl StateObj<K, V> + 'static) -> Self {
        DynState(Arc::new(state))
    }
}

impl<K, V> State for DynState<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Ord + Clone + Send + Sync + 'static,
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
        self.0.group(tx, key, when, f)
    }

    fn group_rev(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        self.0.group_rev(tx, key, when, f)
    }

    fn groups(
        &self,
        tx: &mut WriteTx,
        keys: &[K],
        when: When,
        f: &mut dyn FnMut(&K, &V, Delta),
    ) -> Result<()> {
        self.0.groups(tx, keys, when, f)
    }

    fn total(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Delta> {
        self.0.total(tx, key, when)
    }

    fn exists(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<bool> {
        self.0.exists(tx, key, when)
    }
}

/// The object-safe part of [`KeyedState`].
pub(crate) trait KeyedObj<K, V>: StateObj<K, V> {
    fn get(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Option<V>>;
}

impl<S: KeyedState + Sync> KeyedObj<S::Key, S::Value> for S {
    fn get(&self, tx: &mut WriteTx, key: &S::Key, when: When) -> Result<Option<S::Value>> {
        KeyedState::get(self, tx, key, when)
    }
}

/// Any register state, read through a register of its changes as its
/// delta.
pub struct DynKeyed<K, V>(Arc<dyn KeyedObj<K, V>>);

impl<K, V> Clone for DynKeyed<K, V> {
    fn clone(&self) -> Self {
        DynKeyed(self.0.clone())
    }
}

impl<K, V> DynKeyed<K, V> {
    pub(crate) fn new(state: impl KeyedObj<K, V> + 'static) -> Self {
        DynKeyed(Arc::new(state))
    }
}

fn register_rows<K: Ord, V>(delta: &Register<K, Mergeless<V>>, f: &mut dyn FnMut(&K, &V, Delta))
where
    V: Clone + Send + 'static,
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

impl<K: Data, V: Data> State for DynKeyed<K, V> {
    type Key = K;
    type Value = V;
    type Delta = Register<K, Mergeless<V>>;

    fn rows(delta: &Self::Delta, f: &mut dyn FnMut(&K, &V, Delta)) {
        register_rows(delta, f)
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&V, Delta) -> bool,
    ) -> Result<()> {
        if let Some(v) = self.0.get(tx, key, when)? {
            f(&v, 1);
        }
        Ok(())
    }

    fn total(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Delta> {
        Ok(self.0.get(tx, key, when)?.is_some() as Delta)
    }
}

impl<K: Data, V: Data> KeyedState for DynKeyed<K, V> {
    fn get(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Option<V>> {
        self.0.get(tx, key, when)
    }

    fn changes(delta: &Self::Delta, f: &mut dyn FnMut(&K, Option<&V>, Option<&V>)) {
        for (k, c) in delta.iter() {
            f(k, c.old_value(), c.new_value());
        }
    }
}

/// What a value is mapped to, if it is kept.
pub(crate) type ValueMap<K, V, W> = Arc<dyn Fn(&K, &V) -> Option<W> + Send + Sync>;

/// A register state with every value mapped by `f`, dropping those it
/// maps to `None`. Stores nothing: reads map the parent's values as
/// they go.
pub(crate) struct FilterMapped<K, V, W> {
    pub parent: DynKeyed<K, V>,
    pub f: ValueMap<K, V, W>,
}

impl<K, V, W> Clone for FilterMapped<K, V, W> {
    fn clone(&self) -> Self {
        FilterMapped {
            parent: self.parent.clone(),
            f: self.f.clone(),
        }
    }
}

impl<K: Data, V: Data, W: Data> State for FilterMapped<K, V, W> {
    type Key = K;
    type Value = W;
    type Delta = Register<K, Mergeless<W>>;

    fn rows(delta: &Self::Delta, f: &mut dyn FnMut(&K, &W, Delta)) {
        register_rows(delta, f)
    }

    fn group(
        &self,
        tx: &mut WriteTx,
        key: &K,
        when: When,
        f: &mut dyn FnMut(&W, Delta) -> bool,
    ) -> Result<()> {
        if let Some(w) = KeyedState::get(self, tx, key, when)? {
            f(&w, 1);
        }
        Ok(())
    }

    fn total(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Delta> {
        Ok(KeyedState::get(self, tx, key, when)?.is_some() as Delta)
    }
}

impl<K: Data, V: Data, W: Data> KeyedState for FilterMapped<K, V, W> {
    fn get(&self, tx: &mut WriteTx, key: &K, when: When) -> Result<Option<W>> {
        Ok(self
            .parent
            .0
            .get(tx, key, when)?
            .and_then(|v| (self.f)(key, &v)))
    }

    fn changes(delta: &Self::Delta, f: &mut dyn FnMut(&K, Option<&W>, Option<&W>)) {
        for (k, c) in delta.iter() {
            f(k, c.old_value(), c.new_value());
        }
    }
}

/// The node computing a [`FilterMapped`] register's changes from its
/// parent's: each change's old and new values mapped.
pub(crate) struct FilterMapValues<K, V, W> {
    pub f: ValueMap<K, V, W>,
}

impl<K: Data, V: Data, W: Data> Unary for FilterMapValues<K, V, W> {
    type Takes = Register<K, Mergeless<V>>;
    type Gives = Register<K, Mergeless<W>>;
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, _: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives) {
        for (k, c) in batch.iter() {
            let old = c.old_value().and_then(|v| (self.f)(k, v));
            let new = c.new_value().and_then(|v| (self.f)(k, v));
            let change = match (old, new) {
                (None, None) => continue,
                (Some(o), Some(n)) if o == n => continue,
                (None, Some(n)) => Change::Insert(n),
                (Some(o), None) => Change::Delete(o),
                (Some(o), Some(n)) => Change::Replace(o, n),
            };
            out.absorb_one(k.clone(), change);
        }
    }
}
