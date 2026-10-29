use std::{
    any::Any,
    collections::{BTreeMap, HashMap, btree_map::Entry},
};

use crate::{
    Batch, Delta, Diff,
    merge::{Merge, ValOp},
};

type Epoch = u64;

type KeyspaceId = u64;

pub trait ErasedUnderlay: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    fn absorb_erased(&mut self, other: Box<dyn ErasedUnderlay>);
    // fn flush_into(&mut self, out: &mut Vec<WalOp>);
    fn is_empty(&self) -> bool;
}

#[repr(transparent)]
struct Underlay<K, M: Merge> {
    inner: BTreeMap<K, ValOp<M>>,
}
impl<K, M: Merge> Default for Underlay<K, M> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<K: Ord + 'static, M: Merge> ErasedUnderlay for Underlay<K, M> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn absorb_erased(&mut self, other: Box<dyn ErasedUnderlay>) {
        let other = other
            .into_any()
            .downcast::<Underlay<K, M>>()
            .expect("absorb: keyspace id collision — two KS types share an id");
        self.absorb_underlay(*other);
    }

    // fn flush_into(&mut self, out: &mut Vec<WalOp>) {
    //     for (k, vop) in self.drain() {
    //         out.push(WalOp {
    //             ks: KS::ID,
    //             key: KS::encode_key(&k),
    //             op: vop.encode::<KS>(),
    //         });
    //     }
    // }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K: Ord + 'static, M: Merge> Underlay<K, M> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn absorb_underlay(&mut self, other: Self) {
        for (k, vop) in other.inner {
            self.absorb_value(k, vop);
        }
    }

    pub fn absorb_value(&mut self, key: K, vop: ValOp<M>) {
        match self.inner.entry(key) {
            Entry::Vacant(v) => {
                v.insert(vop);
            }
            Entry::Occupied(mut o) => o.get_mut().absorb_one(vop),
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = (K, ValOp<M>)> {
        std::mem::take(&mut self.inner).into_iter()
    }

    pub fn emit_changes(&self, target: &mut Batch<(K, M::Value)>) {
        self.inner.iter().flat_map(|(key, op)| match op {
            ValOp::Store(v) => std::iter::once(Diff((key.to_owned(), v.to_owned()), 1)),

            // todo: delete and modify need to perform a full
            // read to materialize their retracted values
            //
            // ok, but if a downstream op only uses the keyspace handles,
            // we actually don't need to push anything downstream at all?
            // becuase this btree is already the consolidated, keyed list
            // of relevant deltas...
            ValOp::Delete => std::iter::once(Diff((key.to_owned(), todo!()), -1)),
            ValOp::Modify(_) => todo!(),
        });
        // target.extend();
    }
}

pub struct WriteTx {
    epoch: Epoch,
    pub(crate) overlay: HashMap<KeyspaceId, Box<dyn ErasedUnderlay>>,
    // store: &'tx store
    // (not mut because we only use this for reads)
    // store should have its own &mut self method that finishes this tx
    // by flushing the overlay into packed runs & the wal!
}

impl WriteTx {
    pub fn keyspace<K: 'static + Ord, M: Merge>(&mut self, id: KeyspaceId) -> &mut Underlay<K, M> {
        self.overlay
            .entry(id)
            .or_insert(Box::new(Underlay::<K, M>::default()))
            .as_any_mut()
            .downcast_mut()
            .unwrap()
    }
}
