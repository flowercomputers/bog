use std::{
    any::Any,
    collections::{BTreeMap, HashMap, btree_map::Entry},
};

use crate::{
    Change, Register,
    merge::{Merge, ValOp},
};

type Epoch = u64;

pub type KeyspaceId = u64;

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
}

#[derive(Default)]
pub struct WriteTx {
    epoch: Epoch,
    pub(crate) overlay: HashMap<KeyspaceId, Box<dyn ErasedUnderlay>>,
    // store: &'tx store
    // (not mut because we only use this for reads)
    // store should have its own &mut self method that finishes this tx
    // by flushing the overlay into packed runs & the wal!
}

pub struct KeyspaceHandle<K, M: Merge> {
    inner: Register<K, M>,
}

impl<K: 'static + Ord, M: Merge> KeyspaceHandle<K, M> {
    fn absorb(&mut self, key: K, change: Change<M>) {
        match self.inner.entry(key) {
            Entry::Vacant(vac) => {
                vac.insert(change);
            }
            Entry::Occupied(occ) => {
                let (key, prev) = occ.remove_entry();
                if let Some(c) = prev.compose(change) {
                    self.inner.insert(key, c);
                }
                // check if the change is a noop?
            }
        }
    }
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

    pub fn keyspace2<K: 'static + Ord, M: Merge>(
        &mut self,
        id: KeyspaceId,
    ) -> &mut KeyspaceHandle<K, M> {
        self.overlay
            .entry(id)
            .or_insert(Box::new(Underlay::<K, M>::default()))
            .as_any_mut()
            .downcast_mut()
            .unwrap()
    }
}
