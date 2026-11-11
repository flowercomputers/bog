//! Constraints on source heads: `reject` and `check` at write time,
//! `foreign_key` as dataflow, `unique_key` as an index the table checks.

use super::*;

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
pub(super) fn foreign_key<K, PV>(
    child: &Arc<str>,
    children: &crate::stream::Stream<K, impl crate::stream::Chain<Out = K>>,
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
