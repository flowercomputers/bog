//! What the store knows of each keyspace: its types, erased, and the typed work done through them.

use super::*;

/// A keyspace's types, captured when it is registered with the builder.
/// Replay uses it to decode WAL sections; transactions use it to refuse
/// keyspaces nobody declared.
pub(super) struct KeyspaceDecl {
    pub(super) key: &'static str,
    pub(super) merge: &'static str,
    /// Whether a graph computes it from other keyspaces, so it can be
    /// truncated and recomputed when the graph changes.
    pub(super) derived: bool,
    /// Decode one WAL section and apply it over the keyspace's replayed
    /// tree, creating the tree on first use.
    pub(super) replay:
        fn(&mut Option<Box<dyn Any + Send + Sync>>, &[u8]) -> Result<(), DecodeError>,
    /// Merge this keyspace's trees, newest first, into one sorted run of
    /// encoded rows.
    pub(super) write_run:
        fn(&[&(dyn Any + Send + Sync)], &mut dyn FnMut(&[u8], &[u8]) -> Result<()>) -> Result<()>,
    /// Estimated resident bytes of one of this keyspace's trees.
    pub(super) measure: fn(&(dyn Any + Send + Sync)) -> usize,
    /// Fold trees, oldest first, into one; see [`fold_trees`].
    pub(super) fold: fn(&[&LayerTree], bool) -> LayerTree,
    /// Fold a newer tree into an older one in place; see [`fold_into`].
    pub(super) fold_into: fn(&mut LayerTree, LayerTree, bool) -> Result<(), LayerTree>,
    /// For an additive keyspace: add an encoded amount to another,
    /// replacing it; false when they sum to nothing.
    pub(super) add_encoded: Option<fn(&mut Vec<u8>, &[u8]) -> Result<bool, DecodeError>>,
    /// Run indexes, in slot order.
    pub(super) indexes: Vec<IndexKind>,
}

pub(super) fn add_encoded<M: Merge>(sum: &mut Vec<u8>, amount: &[u8]) -> Result<bool, DecodeError>
where
    M::Value: Codec,
{
    let total = match (
        decode_all::<Option<M::Value>>(sum)?,
        decode_all::<Option<M::Value>>(amount)?,
    ) {
        (Some(mut a), Some(b)) => {
            M::add(&mut a, &b);
            Some(a)
        }
        (a, b) => a.or(b),
    };
    sum.clear();
    total.encode(sum);
    Ok(total.as_ref().is_some_and(|t| !M::consider_delete(t)))
}

/// One tree holding `trees`, oldest first, each overwriting the ones
/// before it. With `drop_tombstones`, nothing lies beneath, so deletes are
/// applied rather than kept.
///
/// In an additive keyspace the trees add up instead, and a key whose
/// amounts sum to nothing is dropped wherever it is.
pub(super) fn fold_trees<K, M>(trees: &[&LayerTree], drop_tombstones: bool) -> LayerTree
where
    K: Ord + Clone + ResidentSize + Send + Sync + 'static,
    M: Merge,
    M::Value: Clone + ResidentSize + Send + Sync + 'static,
{
    fn cast<K: 'static, V: 'static>(t: &LayerTree) -> &Resident<K, V> {
        t.tree
            .downcast_ref::<Resident<K, V>>()
            .expect("layer tree has the registered type")
    }
    // a k-way merge of the sorted trees, newest winning a shared key,
    // into a sorted run the map is then built from in one pass
    let mut its: Vec<_> = trees
        .iter()
        .map(|t| cast::<K, M::Value>(t).iter().peekable())
        .collect();
    let mut merged: Vec<(K, Option<M::Value>)> = Vec::with_capacity(
        trees
            .iter()
            .map(|t| cast::<K, M::Value>(t).len())
            .max()
            .unwrap_or(0),
    );
    let mut bytes = 0;
    loop {
        // the least head key; among equals, the newest tree's
        let mut win: Option<(usize, &K)> = None;
        for (i, it) in its.iter_mut().enumerate() {
            if let Some(&(k, _)) = it.peek()
                && win.is_none_or(|(_, best)| k <= best)
            {
                win = Some((i, k));
            }
        }
        let Some((w, _)) = win else { break };
        let (k, v) = its[w].next().expect("the winner has a head");
        if M::ADDITIVE {
            let mut sum = v.clone().expect("an additive keyspace holds amounts");
            for it in its.iter_mut() {
                if let Some((_, amount)) = it.next_if(|(other, _)| *other == k) {
                    M::add(&mut sum, amount.as_ref().expect("an amount"));
                }
            }
            if !M::consider_delete(&sum) {
                let sum = Some(sum);
                bytes += resident_entry_bytes(k, &sum);
                merged.push((k.clone(), sum));
            }
            continue;
        }
        for it in its.iter_mut() {
            it.next_if(|(other, _)| *other == k);
        }
        if drop_tombstones && v.is_none() {
            continue;
        }
        bytes += resident_entry_bytes(k, v);
        merged.push((k.clone(), v.clone()));
    }
    LayerTree {
        tree: Arc::new(merged.into_iter().collect::<Resident<K, M::Value>>()),
        bytes,
        // the caller registers and builds them
        indexes: Arc::default(),
    }
}

/// Apply `newer`'s entries to `base` in place, deleting rather than
/// keeping tombstones with `drop_tombstones`. Fails, handing `newer` back,
/// if `base`'s tree is shared, so it cannot change. The caller resets
/// `base`'s run indexes.
pub(super) fn fold_into<K, M>(
    base: &mut LayerTree,
    newer: LayerTree,
    drop_tombstones: bool,
) -> Result<(), LayerTree>
where
    K: Ord + Clone + ResidentSize + Send + Sync + 'static,
    M: Merge,
    M::Value: Clone + ResidentSize + Send + Sync + 'static,
{
    let Some(tree) = Arc::get_mut(&mut base.tree) else {
        return Err(newer);
    };
    let tree = tree
        .downcast_mut::<Resident<K, M::Value>>()
        .expect("layer tree has the registered type");
    let newer = newer
        .tree
        .downcast::<Resident<K, M::Value>>()
        .expect("layer tree has the registered type");
    // moved out when nothing else holds it, else copied
    let newer = Arc::try_unwrap(newer).unwrap_or_else(|shared| (*shared).clone());
    let mut bytes = base.bytes;
    if M::ADDITIVE {
        for (k, amount) in newer {
            match tree.entry(k) {
                btree_map::Entry::Vacant(e) => {
                    bytes += resident_entry_bytes(e.key(), &amount);
                    e.insert(amount);
                }
                btree_map::Entry::Occupied(mut e) => {
                    let sum = e.get_mut().as_mut().expect("an amount");
                    M::add(sum, amount.as_ref().expect("an amount"));
                    if M::consider_delete(sum) {
                        bytes -= resident_entry_bytes(e.key(), e.get());
                        e.remove();
                    }
                }
            }
        }
        base.bytes = bytes;
        return Ok(());
    }
    for (k, v) in newer {
        if drop_tombstones && v.is_none() {
            if let Some(old) = tree.remove(&k) {
                bytes -= resident_entry_bytes(&k, &old);
            }
            continue;
        }
        bytes += resident_entry_bytes(&k, &v);
        match tree.get_mut(&k) {
            Some(slot) => {
                bytes -= resident_entry_bytes(&k, slot);
                *slot = v;
            }
            None => {
                tree.insert(k, v);
            }
        }
    }
    base.bytes = bytes;
    Ok(())
}

pub(super) fn measure_tree<K, V>(tree: &(dyn Any + Send + Sync)) -> usize
where
    K: ResidentSize + 'static,
    V: ResidentSize + 'static,
{
    tree.downcast_ref::<Resident<K, V>>()
        .expect("tree has the registered type")
        .iter()
        .map(|(k, v)| resident_entry_bytes(k, v))
        .sum()
}

pub(super) fn replay_section<K, M>(
    tree: &mut Option<Box<dyn Any + Send + Sync>>,
    mut body: &[u8],
) -> Result<(), DecodeError>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Send + Sync + 'static,
{
    let tree = tree
        .get_or_insert_with(|| Box::new(Resident::<K, M::Value>::new()))
        .downcast_mut::<Resident<K, M::Value>>()
        .expect("replay tree has the registered type");
    while !body.is_empty() {
        let k = K::decode_ordered(&mut body)?;
        let v = Option::<M::Value>::decode(&mut body)?;
        match tree.entry(k) {
            // amounts add up, and are dropped when they cancel
            btree_map::Entry::Occupied(mut e) if M::ADDITIVE => {
                if let (Some(sum), Some(amount)) = (e.get_mut(), &v) {
                    M::add(sum, amount);
                    if M::consider_delete(sum) {
                        e.remove();
                    }
                }
            }
            // keep tombstones: once disk runs exist they shadow older data
            btree_map::Entry::Occupied(mut e) => {
                e.insert(v);
            }
            btree_map::Entry::Vacant(e) => {
                e.insert(v);
            }
        }
    }
    Ok(())
}

/// Feed `rows` the newest value of every key in `trees`, newest tree first,
/// as encoded `(key, Option<value>)` in key order.
///
/// In an additive keyspace a key's amounts are added, and it is left out
/// if they sum to nothing.
pub(super) fn write_run<K, M>(
    trees: &[&(dyn Any + Send + Sync)],
    rows: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
) -> Result<()>
where
    K: OrderedCodec + Send + Sync + 'static,
    M: Merge,
    M::Value: Codec + Send + Sync + 'static,
{
    // a k-way merge of the sorted trees, streamed out: nothing is built
    // first, so a checkpoint of a large layer costs no memory
    let mut its: Vec<_> = trees
        .iter()
        .map(|t| {
            t.downcast_ref::<Resident<K, M::Value>>()
                .expect("layer tree has the registered type")
                .iter()
                .peekable()
        })
        .collect();
    let (mut key, mut value) = (Vec::new(), Vec::new());
    loop {
        // the least head key; among equals, the newest tree's, which
        // comes first
        let mut win: Option<(usize, &K)> = None;
        for (i, it) in its.iter_mut().enumerate() {
            if let Some(&(k, _)) = it.peek()
                && win.is_none_or(|(_, best)| k < best)
            {
                win = Some((i, k));
            }
        }
        let Some((w, _)) = win else { break };
        let (k, v) = its[w].next().expect("the winner has a head");
        key.clear();
        value.clear();
        if M::ADDITIVE {
            let mut sum = v.clone();
            for it in its.iter_mut() {
                if let Some((_, amount)) = it.next_if(|(other, _)| *other == k)
                    && let (Some(sum), Some(amount)) = (&mut sum, amount)
                {
                    M::add(sum, amount);
                }
            }
            if sum.as_ref().is_none_or(M::consider_delete) {
                continue;
            }
            sum.encode(&mut value);
        } else {
            for it in its.iter_mut() {
                it.next_if(|(other, _)| *other == k);
            }
            // tombstones stay: older checkpoints below still hold the key
            v.encode(&mut value);
        }
        k.encode_ordered(&mut key);
        rows(&key, &value)?;
    }
    Ok(())
}
