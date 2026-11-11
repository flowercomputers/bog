//! [`Build`]: the context a schema is declared in. It holds the graph
//! under construction and names every keyspace by its field path.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

use super::{Fingerprint, OpDag, state::Declare as DeclareKeyspace};
use crate::{database::Db as RawDb, merge::Merge, tx::KeyspaceId};

/// What one schema tree builds: the graph, the keyspace names, the
/// views waiting to be defined.
#[doc(hidden)]
pub struct Build {
    pub(crate) graph: OpDag,
    /// Every keyspace by id, for collision checks.
    names: HashMap<u64, String>,
    /// Every operator name handed out.
    taken: HashSet<String>,
    /// Base keyspaces outside the graph: the indexes of alternate keys.
    pub(crate) bases: Vec<DeclareKeyspace>,
    /// Every view slot, with whether it has been defined.
    slots: Vec<(Arc<str>, Rc<std::cell::Cell<bool>>)>,
    /// What to set up on the database once open: retention policies.
    pub(crate) after_open: Vec<Box<dyn FnOnce(&RawDb)>>,
}

/// The build, shared by every head, slot and stream of one tree.
#[doc(hidden)]
pub type Shared = Rc<RefCell<Build>>;

impl Build {
    pub(crate) fn new() -> Shared {
        Rc::new(RefCell::new(Build {
            graph: OpDag::new(),
            names: HashMap::new(),
            taken: HashSet::new(),
            bases: Vec::new(),
            slots: Vec::new(),
            after_open: Vec::new(),
        }))
    }

    /// `base`, or `base#2`, `base#3`, ... if it was handed out before.
    pub(crate) fn unique(&mut self, base: &str) -> String {
        if self.taken.insert(base.to_string()) {
            return base.to_string();
        }
        let mut n = 2;
        loop {
            let name = format!("{base}#{n}");
            if self.taken.insert(name.clone()) {
                return name;
            }
            n += 1;
        }
    }

    /// The keyspace named `name`: its id is the name's hash.
    pub(crate) fn id<K, M: Merge>(&mut self, name: &str) -> KeyspaceId<K, M> {
        let mut fp = Fingerprint::new();
        fp.write_str(name);
        let raw = fp.finish();
        if let Some(other) = self.names.insert(raw, name.to_string()) {
            if other == name {
                panic!("keyspace {name:?} is declared twice");
            }
            panic!("keyspaces {other:?} and {name:?} hash alike; rename one");
        }
        KeyspaceId::new(raw)
    }

    pub(crate) fn slot(&mut self, name: Arc<str>) -> Rc<std::cell::Cell<bool>> {
        let defined = Rc::new(std::cell::Cell::new(false));
        self.slots.push((name, defined.clone()));
        defined
    }

    /// Every view must be defined by the end of `declare`.
    pub(crate) fn check_defined(&self) {
        let undefined: Vec<&str> = self
            .slots
            .iter()
            .filter(|(_, defined)| !defined.get())
            .map(|(name, _)| &**name)
            .collect();
        assert!(
            undefined.is_empty(),
            "views declared but never defined: {undefined:?}"
        );
    }
}
