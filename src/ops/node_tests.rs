//! Unit tests of the operators that need no arrangement.

use crate::graph::OpDag;
use crate::store::{DurabilityPolicy, StoreBuilder};
use crate::*;

#[test]
fn sum_inserts_then_replaces_then_deletes() {
    let mut g = OpDag::new();
    let src = g.source::<Bag<(u8, i64)>>();
    const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);
    let sum = g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
    let _ = sum;
    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(TOTALS)
        .build()
        .unwrap();
    let mut tx = store.begin().unwrap();

    let mut b = Bag::new();
    b.push((1, 10), 1);
    b.push((1, 5), 1);
    g.give(src, &mut b);
    g.run(&mut tx);
    let mut ks = tx.keyspace(TOTALS);
    assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&(2, 15)));

    let mut b = Bag::new();
    b.push((1, 10), -1);
    g.give(src, &mut b);
    g.run(&mut tx);
    let mut ks = tx.keyspace(TOTALS);
    assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&(1, 5)));

    let mut b = Bag::new();
    b.push((1, 5), -1);
    g.give(src, &mut b);
    g.run(&mut tx);
    let mut ks = tx.keyspace(TOTALS);
    assert_eq!(ks.get(&1).unwrap().as_deref(), None);
    // insert then delete within one tx cancels out entirely
    assert!(ks.inner_is_empty());
}

mod rebuild {
    use crate::error::Error;
    use crate::graph::{NodeRef, OpDag};
    use crate::store::{DurabilityPolicy, Store, StoreBuilder};
    use crate::*;
    use std::{collections::BTreeMap, ops::Bound, path::Path};

    const BASE: KeyspaceId<u8, i64> = KeyspaceId::new(1);
    const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);

    /// Base rows `k -> v`, summed into TOTALS by `k % modulus`.
    struct App {
        g: OpDag,
        src: NodeRef<Bag<(u8, i64)>>,
        modulus: u8,
    }

    impl App {
        fn new(modulus: u8) -> Self {
            let mut g = OpDag::new();
            let src = g.source::<Bag<(u8, i64)>>();
            let m = modulus;
            let class = g.nary(
                &[src],
                FusedLinears::new(Map(move |&(k, v): &(u8, i64)| (k % m, v))),
            );
            g.nary(&[class], Sum::<u8, i64>::new(TOTALS));
            App { g, src, modulus }
        }

        /// Like [`new`](Self::new), but rejecting negative totals.
        fn checked(modulus: u8) -> Self {
            let mut g = OpDag::new();
            let src = g.source::<Bag<(u8, i64)>>();
            let m = modulus;
            let class = g.nary(
                &[src],
                FusedLinears::new(Map(move |&(k, v): &(u8, i64)| (k % m, v))),
            );
            let totals = g.nary(&[class], Sum::<u8, i64>::new(TOTALS));
            g.nary(
                &[totals],
                Check::new(
                    |_: &u8, &(_, sum): &(i64, i64)| {
                        if sum < 0 { Err(Negative) } else { Ok(()) }
                    },
                ),
            );
            App { g, src, modulus }
        }

        fn fingerprint(&self) -> u64 {
            let mut fp = crate::graph::Fingerprint::new();
            fp.write_u64(self.g.fingerprint());
            fp.write_u64(self.modulus as u64);
            fp.finish()
        }

        fn builder(&self, dir: &Path) -> StoreBuilder {
            StoreBuilder::new(DurabilityPolicy::RELAXED)
                .dir(dir)
                .keyspace(BASE)
                .derived(TOTALS)
                .graph(self.fingerprint())
        }

        /// Set base rows, feeding their changes to the graph.
        fn write(&mut self, store: &Store, rows: &[(u8, Option<i64>)]) {
            let mut tx = store.begin().unwrap();
            let mut feed = Bag::new();
            {
                let mut ks = tx.keyspace(BASE);
                for &(k, v) in rows {
                    let old = ks.get(&k).unwrap().map(|v| *v);
                    if let Some(o) = old {
                        feed.push((k, o), -1);
                    }
                    if let Some(n) = v {
                        feed.push((k, n), 1);
                    }
                    let change = match (old, v) {
                        (None, None) => continue,
                        (None, Some(n)) => Change::Insert(n),
                        (Some(o), None) => Change::Delete(o),
                        (Some(o), Some(n)) => Change::Replace(o, n),
                    };
                    ks.absorb(k, change).unwrap();
                }
            }
            self.g.give(self.src, &mut feed);
            self.g.run(&mut tx);
            store.finish(tx).unwrap();
        }

        /// Feed base rows to the graph `chunk` at a time.
        fn replay(
            &mut self,
            chunk: usize,
            fail_after: Option<usize>,
        ) -> impl FnMut(&mut WriteTx<'_>) -> crate::error::Result<bool> + '_ {
            let mut after: Option<u8> = None;
            let mut calls = 0;
            move |tx| {
                calls += 1;
                let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                let rows: Vec<(u8, i64)> = tx
                    .keyspace(BASE)
                    .range((lower, Bound::Unbounded))
                    .take(chunk)
                    .map(|r| r.map(|(k, v)| (*k, *v)))
                    .collect::<crate::error::Result<_>>()?;
                let mut feed = Bag::new();
                for &(k, v) in &rows {
                    feed.push((k, v), 1);
                }
                after = rows.last().map(|r| r.0).or(after);
                self.g.give(self.src, &mut feed);
                self.g.run(tx);
                if fail_after == Some(calls) {
                    return Err(Error::InvalidOption("replay failed on purpose"));
                }
                Ok(rows.len() == chunk)
            }
        }
    }

    fn totals(store: &Store) -> Vec<(u8, (i64, i64))> {
        store
            .snapshot()
            .range(TOTALS, ..)
            .map(|r| r.map(|(k, v)| (*k, *v)))
            .collect::<crate::error::Result<_>>()
            .unwrap()
    }

    fn expected(model: &BTreeMap<u8, i64>, modulus: u8) -> Vec<(u8, (i64, i64))> {
        let mut t: BTreeMap<u8, (i64, i64)> = BTreeMap::new();
        for (&k, &v) in model {
            let e = t.entry(k % modulus).or_default();
            e.0 += 1;
            e.1 += v;
        }
        t.into_iter().collect()
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fold3-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Base rows spread over checkpoints, the WAL, and deletes.
    fn populate(app: &mut App, store: &Store, model: &mut BTreeMap<u8, i64>) {
        for round in 0..4i64 {
            let rows: Vec<(u8, Option<i64>)> = (0..20u8)
                .map(|k| {
                    let v = (k as i64 * 7 + round) % 5 != 0;
                    (k, v.then_some(k as i64 * 10 + round))
                })
                .collect();
            for &(k, v) in &rows {
                match v {
                    Some(v) => model.insert(k, v),
                    None => model.remove(&k),
                };
            }
            app.write(store, &rows);
            if round % 2 == 0 {
                store.checkpoint().unwrap();
            }
        }
    }

    #[test]
    fn a_changed_graph_truncates_and_replays_derived_state() {
        let dir = tmpdir("rebuild");
        let mut model = BTreeMap::new();
        let mut v1 = App::new(2);
        let store = v1.builder(&dir).build().unwrap();
        populate(&mut v1, &store, &mut model);
        assert_eq!(totals(&store), expected(&model, 2));
        drop(store);

        // the same graph reopens without a rebuild
        let store = v1.builder(&dir).build().unwrap();
        assert_eq!(totals(&store), expected(&model, 2));
        drop(store);

        // a different graph cannot open without a replay
        let mut v2 = App::new(3);
        assert!(matches!(
            v2.builder(&dir).build(),
            Err(Error::RebuildRequired)
        ));

        // a failed replay leaves the rebuild to the next open
        let b = v2.builder(&dir);
        assert!(b.build_with(v2.replay(4, Some(2))).is_err());
        let mut v2 = App::new(3);
        assert!(matches!(
            v2.builder(&dir).build(),
            Err(Error::RebuildRequired)
        ));

        let b = v2.builder(&dir);
        let store = b.build_with(v2.replay(3, None)).unwrap();
        assert_eq!(totals(&store), expected(&model, 3));

        // the new graph keeps running incrementally over rebuilt state
        v2.write(&store, &[(5, None), (40, Some(1))]);
        model.remove(&5);
        model.insert(40, 1);
        assert_eq!(totals(&store), expected(&model, 3));
        drop(store);

        let store = v2.builder(&dir).build().unwrap();
        assert_eq!(totals(&store), expected(&model, 3));

        // compaction leaves truncated data behind, and once every table
        // is newer than the floor, the floor goes too
        store.checkpoint().unwrap();
        store.compact().unwrap();
        assert_eq!(totals(&store), expected(&model, 3));
        drop(store);
        let m = std::fs::read_to_string(dir.join("MANIFEST")).unwrap();
        assert!(!m.contains("floor\t"), "floor outlived its data:\n{m}");
        let store = v2.builder(&dir).build().unwrap();
        assert_eq!(totals(&store), expected(&model, 3));
    }

    #[derive(Debug)]
    struct Negative;
    impl std::fmt::Display for Negative {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("negative total")
        }
    }
    impl std::error::Error for Negative {}

    #[test]
    fn a_rebuild_is_not_held_to_checks_its_chunks_cannot_meet() {
        let dir = tmpdir("rebuild-check");
        let mut v1 = App::checked(2);
        let store = v1.builder(&dir).build().unwrap();
        // each class sums to 5, but key 0 alone would be -5
        v1.write(&store, &[(0, Some(-5)), (1, Some(5)), (2, Some(10))]);
        let mut tx = store.begin().unwrap();
        assert!(!tx.rebuilding());
        let mut feed = Bag::new();
        feed.push((4, -100), 1);
        v1.g.give(v1.src, &mut feed);
        v1.g.run(&mut tx);
        assert!(
            matches!(store.finish(tx), Err(Error::Rejected(_))),
            "ordinary writes are checked"
        );
        drop(store);

        // one class, fed a row at a time, starting with key 0
        let mut v2 = App::checked(1);
        let b = v2.builder(&dir);
        let store = b.build_with(v2.replay(1, None)).unwrap();
        assert_eq!(totals(&store), vec![(0, (3, 10))]);
    }

    #[test]
    fn an_undeclared_derived_keyspace_is_dropped() {
        let dir = tmpdir("drop-derived");
        let mut model = BTreeMap::new();
        let mut app = App::new(2);
        let store = app.builder(&dir).build().unwrap();
        populate(&mut app, &store, &mut model);
        drop(store);

        let base_only = || {
            StoreBuilder::new(DurabilityPolicy::RELAXED)
                .dir(&dir)
                .keyspace(BASE)
        };
        // no derived keyspace remains, so there is nothing to replay
        let store = base_only().build().unwrap();
        assert_eq!(store.snapshot().range(BASE, ..).count(), model.len());
        drop(store);
        let m = std::fs::read_to_string(dir.join("MANIFEST")).unwrap();
        assert!(!m.contains("derived"), "{m}");
        assert!(base_only().build().is_ok());

        // bringing it back rebuilds it from scratch
        let mut app = App::new(2);
        let b = app.builder(&dir);
        let store = b.build_with(app.replay(64, None)).unwrap();
        assert_eq!(totals(&store), expected(&model, 2));
    }

    #[test]
    fn graph_fingerprints_see_shape_and_keyspaces() {
        assert_eq!(App::new(2).g.fingerprint(), App::new(2).g.fingerprint());
        let mut g = OpDag::new();
        let src = g.source::<Bag<(u8, i64)>>();
        g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
        let mut h = OpDag::new();
        let src = h.source::<Bag<(u8, i64)>>();
        h.nary(&[src], Sum::<u8, i64>::new(KeyspaceId::new(8)));
        assert_ne!(g.fingerprint(), h.fingerprint());
        assert_ne!(g.fingerprint(), App::new(2).g.fingerprint());
        // closure bodies are invisible, so a version stands in for them
        assert_eq!(App::new(2).g.fingerprint(), App::new(3).g.fingerprint());
        let before = g.fingerprint();
        g.version(1);
        assert_ne!(g.fingerprint(), before);
    }
}

#[test]
fn a_killed_tx_stops_the_graph_and_commits_nothing() {
    let mut g = OpDag::new();
    let src = g.source::<Bag<(u8, i64)>>();
    const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(7);
    let _sum = g.nary(&[src], Sum::<u8, i64>::new(TOTALS));
    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(TOTALS)
        .tx_memory_limit(2048)
        .build()
        .unwrap();

    let mut b = Bag::new();
    for k in 0..=255u8 {
        b.push((k, 1), 1);
    }
    g.give(src, &mut b);
    let mut tx = store.begin().unwrap();
    g.run(&mut tx);
    assert!(tx.failure().is_some());
    assert!(store.finish(tx).is_err());
    assert_eq!(store.snapshot().get(TOTALS, &0).unwrap(), None);

    // the graph is clean for the next epoch
    let mut b = Bag::new();
    b.push((3, 5), 1);
    g.give(src, &mut b);
    let mut tx = store.begin().unwrap();
    g.run(&mut tx);
    assert_eq!(store.finish(tx).unwrap(), Some(1));
    assert_eq!(
        store.snapshot().get(TOTALS, &3).unwrap().as_deref(),
        Some(&(1, 5))
    );
}
