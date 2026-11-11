//! Folding memory layers together.

use super::*;

impl Shared {
    /// Fold the layers [`fold_plan`] picks into one. Returns whether there
    /// was anything to do.
    ///
    /// The merge runs without locks, over the layers as they were when it
    /// started. Commits only append layers meanwhile, so it installs over
    /// the same layers, now with newer ones after them. Checkpoints wait
    /// for it; a truncation in between replaces every layer, and the
    /// result is thrown away.
    ///
    /// Snapshots hold the layers they were taken with, so folding never
    /// changes what one sees; it only stops sharing memory with them. With
    /// no tables, nothing lies beneath the oldest layer, so a fold that
    /// reaches it applies deletes instead of keeping tombstones.
    pub(super) fn fold_once(&self) -> bool {
        // folding in place needs no coordination with a checkpoint: one in
        // flight holds the version, so the fold finds it shared and stops
        if let Some(folded) = self.fold_in_place() {
            return folded;
        }
        let _layers = self
            .layer_work
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let before = self.current.load_full();
        let from = before.live_from();
        let Some(start) = fold_plan(&before.layers[from..], self.opts.max_layers) else {
            return false;
        };
        let start = from + start;
        let window = &before.layers[start..];
        let drop_tombstones = start == 0 && before.tables.is_empty();
        let merged = self.merge_layers(window, drop_tombstones);

        let w = self.lock_writer();
        let now = self.current.load_full();
        let in_place = now.layers.len() >= before.layers.len()
            && now.live_from() <= start
            && now.layers[start..before.layers.len()]
                .iter()
                .zip(window)
                .all(|(a, b)| Arc::ptr_eq(a, b));
        if in_place {
            let mut layers = now.layers[..start].to_vec();
            layers.extend(merged.map(Arc::new));
            layers.extend_from_slice(&now.layers[before.layers.len()..]);
            self.current.store(Arc::new(Version {
                epoch: now.epoch,
                resident_bytes: layers.iter().map(|l| l.bytes).sum(),
                layers,
                tables: now.tables.clone(),
                floors: now.floors.clone(),
                frozen: now.frozen,
            }));
        }
        drop(w);
        self.progress();
        true
    }

    /// Fold every layer into the oldest in place, if no snapshot can see
    /// them: then nothing but the current version holds them, and they
    /// need not be copied. Returns whether there was anything to fold, or
    /// `None` if they are shared and must be folded by copying.
    ///
    /// Holds the writer lock and the snapshot registry throughout, so no
    /// commit or snapshot can start meanwhile, and swaps the current
    /// version out to own it. Anything else still holding it, such as a
    /// stats read, sends it back to the copying fold.
    pub(super) fn fold_in_place(&self) -> Option<bool> {
        let w = self.lock_writer();
        let snapshots = self.snapshots();
        let folded = self.fold_in_place_held(&snapshots, usize::MAX);
        drop(snapshots);
        drop(w);
        if folded == Some(true) {
            self.progress();
        }
        folded
    }

    /// [`fold_in_place`](Self::fold_in_place) with the writer lock and the
    /// snapshot registry already held, folding only if the layers above
    /// the oldest hold no more than `limit` bytes.
    pub(super) fn fold_in_place_held(
        &self,
        snapshots: &BTreeMap<u64, usize>,
        limit: usize,
    ) -> Option<bool> {
        if !snapshots.is_empty() {
            return None;
        }
        let current = self.current.load_full();
        let from = current.live_from();
        if current.layers.len() - from < 2 {
            return Some(false);
        }
        if current.layers[from + 1..]
            .iter()
            .map(|l| l.bytes)
            .sum::<usize>()
            > limit
        {
            return None;
        }
        // what readers of stats see for the moment: nobody else can read
        // layers while the registry is held
        let stand_in = Arc::new(Version {
            epoch: current.epoch,
            layers: Vec::new(),
            tables: current.tables.clone(),
            resident_bytes: 0,
            floors: current.floors.clone(),
            frozen: current.frozen,
        });
        drop(current);
        let version = match Arc::try_unwrap(self.current.swap(stand_in)) {
            Ok(v) => v,
            Err(shared) => {
                self.current.store(shared);
                return None;
            }
        };
        let Version {
            epoch,
            layers,
            tables,
            floors,
            frozen,
            ..
        } = version;
        let mut layers = layers.into_iter();
        // frozen layers stay as they are, beneath the fold
        let mut folded: Vec<Arc<Layer>> = layers.by_ref().take(from).collect();
        // nothing lies beneath the oldest layer without tables
        let drop_tombstones = from == 0 && tables.is_empty();
        let restore = |this: &Self, folded: Vec<Arc<Layer>>, rest: Vec<Arc<Layer>>| {
            let mut layers = folded;
            layers.extend(rest);
            this.current.store(Arc::new(Version {
                epoch,
                resident_bytes: layers.iter().map(|l| l.bytes).sum(),
                layers,
                tables,
                floors,
                frozen,
            }));
        };
        let mut base = match Arc::try_unwrap(layers.next().expect("two live layers")) {
            Ok(base) => base,
            Err(shared) => {
                let rest: Vec<_> = std::iter::once(shared).chain(layers).collect();
                restore(self, folded, rest);
                return None;
            }
        };
        for layer in layers {
            let (newest, trees) = match Arc::try_unwrap(layer) {
                Ok(l) => (l.epoch, l.trees),
                Err(shared) => (shared.epoch, shared.trees.clone()),
            };
            for (ks, newer) in trees {
                let decl = &self.keyspaces[&ks];
                match base.trees.entry(ks) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(newer);
                    }
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        let slot = e.get_mut();
                        if let Err(newer) = (decl.fold_into)(slot, newer, drop_tombstones) {
                            // the base tree is shared after all: copy it
                            *slot = (decl.fold)(&[&*slot, &newer], drop_tombstones);
                        }
                        // rebuilt by the next reader that wants them
                        slot.indexes = Arc::new(LayerIndexes::new(decl.indexes.len()));
                        if slot.bytes == 0 {
                            e.remove();
                        }
                    }
                }
            }
            base.epoch = newest;
        }
        base.bytes = base.trees.values().map(|t| t.bytes).sum();
        if !base.trees.is_empty() {
            folded.push(Arc::new(base));
        }
        restore(self, folded, Vec::new());
        Some(true)
    }

    /// One layer holding `layers`, oldest first, or `None` if deletes
    /// emptied it. Trees only one layer has are shared, not copied.
    pub(super) fn merge_layers(
        &self,
        layers: &[Arc<Layer>],
        drop_tombstones: bool,
    ) -> Option<Layer> {
        let ids: BTreeSet<u64> = layers
            .iter()
            .flat_map(|l| l.trees.keys().copied())
            .collect();
        let mut trees = HashMap::with_capacity(ids.len());
        for ks in ids {
            let parts: Vec<&LayerTree> = layers.iter().filter_map(|l| l.trees.get(&ks)).collect();
            let decl = &self.keyspaces[&ks];
            let folded = match parts[..] {
                [only] if !drop_tombstones => only.clone(),
                _ => {
                    let mut folded = (decl.fold)(&parts, drop_tombstones);
                    // off the commit path, so build the indexes now
                    let indexes = LayerIndexes::new(decl.indexes.len());
                    indexes.build_all(&decl.indexes, &*folded.tree);
                    folded.indexes = Arc::new(indexes);
                    folded
                }
            };
            // every entry costs at least ENTRY_OVERHEAD, so no bytes means
            // deletes emptied the tree
            if folded.bytes > 0 {
                trees.insert(ks, folded);
            }
        }
        (!trees.is_empty()).then(|| Layer {
            epoch: layers.last().expect("merging at least one layer").epoch,
            bytes: trees.values().map(|t| t.bytes).sum(),
            trees,
        })
    }
}
