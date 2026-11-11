//! Constraints on view slots: `reject` and `check` over the rows a
//! transaction left in the view, after the graph ran.

use super::*;

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
