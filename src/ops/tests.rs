//! Model tests: every operator's accumulated output must equal its
//! definition recomputed from scratch, after every transaction, while
//! its state moves through layers, checkpoints, compaction, reopens and
//! rebuilds.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use super::{linear::*, reduce, *};
use crate::{
    FusedLinears, Map, Mergeless, Then,
    db::Db,
    error::Error,
    graph::{NodeRef, OpDag},
    state::Stored,
    store::{DurabilityPolicy, StoreBuilder},
};

type Rows<T> = BTreeMap<T, Delta>;

/// Accumulates what reaches it; a run's rows count once its tx commits.
struct Sink<T> {
    shared: Arc<Mutex<(Rows<T>, Rows<T>)>>,
}

impl<T> Clone for Sink<T> {
    fn clone(&self) -> Self {
        Sink {
            shared: self.shared.clone(),
        }
    }
}

fn add<T: Ord + Clone>(rows: &mut Rows<T>, t: &T, d: Delta) {
    let e = rows.entry(t.clone()).or_default();
    *e += d;
    if *e == 0 {
        rows.remove(t);
    }
}

impl<T: Ord + Clone> Sink<T> {
    fn new() -> Self {
        Sink {
            shared: Arc::new(Mutex::new((Rows::new(), Rows::new()))),
        }
    }
    fn commit(&self) {
        let mut s = self.shared.lock().unwrap();
        let pending = std::mem::take(&mut s.1);
        for (t, d) in pending {
            add(&mut s.0, &t, d);
        }
    }
    fn abort(&self) {
        self.shared.lock().unwrap().1.clear();
    }
    fn reset(&self) {
        *self.shared.lock().unwrap() = (Rows::new(), Rows::new());
    }
    fn rows(&self) -> Rows<T> {
        self.shared.lock().unwrap().0.clone()
    }
}

impl<T: Ord + Clone + Send + 'static> Nary for Sink<T> {
    type Takes = Bag<T>;
    type Gives = Bag<()>;
    type State = ();
    fn state(&self) {}
    fn eval(&mut self, _: &mut WriteTx, batches: &[&Bag<T>], _: &mut Bag<()>) {
        let mut s = self.shared.lock().unwrap();
        for b in batches {
            for (t, d) in b.diffs() {
                add(&mut s.1, t, d);
            }
        }
    }
}

/// xorshift, so failures replay.
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fold3-ops-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn builder(dir: &Path) -> StoreBuilder {
    StoreBuilder::new(DurabilityPolicy::RELAXED)
        .dir(dir)
        .memtable_limit(1500)
        .options_with(|o| {
            o.max_tables = 3;
            o.compaction_width = 2;
        })
}

// ---------------------------------------------------------------------------
// Rows with multiplicities, given to unstored sources
// ---------------------------------------------------------------------------

type Src = NodeRef<Bag<(u8, u8)>>;

/// Drive random changes to two collections of `(key, value)` rows through
/// the graph `build` makes, checking the sink against `expected` of the
/// collections after every tx. Reopens twice on the way.
fn drive<O: Ord + Clone + Debug + Send + 'static>(
    name: &str,
    build: impl Fn(&mut OpDag, Src, Src) -> NodeRef<Bag<O>>,
    expected: impl Fn(&Rows<(u8, u8)>, &Rows<(u8, u8)>) -> Rows<O>,
) {
    let dir = tmpdir(name);
    let sink = Sink::<O>::new();
    let (mut left, mut right) = (Rows::new(), Rows::new());
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ name.len() as u64);
    for session in 0..3 {
        let mut g = OpDag::new();
        let (l, r) = (g.source(), g.source());
        let out = build(&mut g, l, r);
        g.nary(&[out], sink.clone());
        let mut db = Db::open(builder(&dir), g).unwrap();

        for step in 0..60 {
            let (mut nl, mut nr) = (left.clone(), right.clone());
            let abort = rng.below(8) == 0;
            let res = db.write(|tx| {
                for _ in 0..1 + rng.below(2) {
                    let (mut bl, mut br) = (Bag::new(), Bag::new());
                    for _ in 0..1 + rng.below(4) {
                        let row = (rng.below(5) as u8, rng.below(4) as u8);
                        let (model, bag) = if rng.below(2) == 0 {
                            (&mut nl, &mut bl)
                        } else {
                            (&mut nr, &mut br)
                        };
                        // rows never go negative
                        let d = if model.contains_key(&row) && rng.below(2) == 0 {
                            -1
                        } else {
                            1 + rng.below(2) as Delta
                        };
                        add(model, &row, d);
                        bag.push(row, d);
                    }
                    tx.give(l, &mut bl);
                    tx.give(r, &mut br);
                    tx.run()?;
                }
                if abort {
                    return Err(Error::InvalidOption("abort"));
                }
                Ok(())
            });
            match res {
                Ok(()) => {
                    sink.commit();
                    (left, right) = (nl, nr);
                }
                Err(_) => sink.abort(),
            }
            assert_eq!(
                sink.rows(),
                expected(&left, &right),
                "{name}: session {session}, step {step}\nleft {left:?}\nright {right:?}"
            );
        }
        // later sessions read whatever state there is from tables
        db.store().checkpoint().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const AL: Arrangement<u8, u8> = KeyspaceId::new(10);
const AR: Arrangement<u8, u8> = KeyspaceId::new(11);
const TL: KeyspaceId<u8, Delta> = KeyspaceId::new(16);
const TR: KeyspaceId<u8, Delta> = KeyspaceId::new(17);

fn arrange(
    g: &mut OpDag,
    l: Src,
    r: Src,
) -> (
    NodeRef<Bag<(u8, u8)>, Arranged<u8, u8>>,
    NodeRef<Bag<(u8, u8)>, Arranged<u8, u8>>,
) {
    (
        g.nary(&[l], Arrange::new(AL, TL)),
        g.nary(&[r], Arrange::new(AR, TR)),
    )
}

fn groups(rows: &Rows<(u8, u8)>) -> BTreeMap<u8, Vec<(u8, Delta)>> {
    let mut g: BTreeMap<u8, Vec<(u8, Delta)>> = BTreeMap::new();
    for (&(k, v), &d) in rows {
        g.entry(k).or_default().push((v, d));
    }
    g
}

#[test]
fn join_pairs_rows_from_both_sides() {
    drive(
        "join",
        |g, l, r| {
            let (l, r) = arrange(g, l, r);
            g.binary_with(l, r, |l, r| {
                Join::new(l, r, |k: &u8, a: &u8, b: &u8| (*k, *a, *b))
            })
        },
        |l, r| {
            let mut out = Rows::new();
            for (&(k, a), &da) in l {
                for (&(k2, b), &db) in r {
                    if k == k2 {
                        add(&mut out, &(k, a, b), da * db);
                    }
                }
            }
            out
        },
    );
}

fn semijoin_expected(l: &Rows<(u8, u8)>, r: &Rows<(u8, u8)>, anti: bool) -> Rows<(u8, u8)> {
    let has = |k: u8| r.keys().any(|&(k2, _)| k2 == k);
    l.iter()
        .filter(|((k, _), _)| has(*k) != anti)
        .map(|(row, d)| (*row, *d))
        .collect()
}

#[test]
fn matching_passes_rows_whose_key_the_other_side_has() {
    drive(
        "matching",
        |g, l, r| {
            let (l, r) = arrange(g, l, r);
            g.binary_with(l, r, Semijoin::matching)
        },
        |l, r| semijoin_expected(l, r, false),
    );
}

#[test]
fn without_passes_rows_whose_key_the_other_side_lacks() {
    drive(
        "without",
        |g, l, r| {
            let (l, r) = arrange(g, l, r);
            g.binary_with(l, r, Semijoin::without)
        },
        |l, r| semijoin_expected(l, r, true),
    );
}

#[test]
fn minus_negates_the_right() {
    drive(
        "minus",
        |g, l, r| g.binary(l, r, Minus::default()),
        |l, r| {
            let mut out = l.clone();
            for (row, d) in r {
                add(&mut out, row, -d);
            }
            out
        },
    );
}

/// A reduction over the arranged left rows, checked against `f` of each
/// key's group, ascending by value.
fn drive_reduce<O, F>(
    name: &str,
    build: impl Fn(&mut OpDag, NodeRef<Bag<(u8, u8)>, Arranged<u8, u8>>) -> NodeRef<Bag<(u8, O)>>,
    f: F,
) where
    O: Ord + Clone + Debug + Send + 'static,
    F: Fn(&[(u8, Delta)]) -> Vec<O>,
{
    drive(
        name,
        |g, l, _| {
            let l = g.nary(&[l], Arrange::new(AL, TL));
            build(g, l)
        },
        |l, _| {
            let mut out = Rows::new();
            for (k, vals) in groups(l) {
                for o in f(&vals) {
                    add(&mut out, &(k, o), 1);
                }
            }
            out
        },
    );
}

const ONE: KeyspaceId<u8, Mergeless<u8>> = KeyspaceId::new(13);
const MANY: Arrangement<u8, u8> = KeyspaceId::new(14);

fn flat<K, M>(
    g: &mut OpDag,
    n: NodeRef<Register<K, M>, impl Clone + 'static>,
) -> NodeRef<Bag<(K, M::Value)>>
where
    K: Ord + Clone + Send + 'static,
    M: Merge + Send,
    M::Value: Ord + Send + 'static,
    M::Diff: Send,
{
    g.unary(n, Flatten::default())
}

#[test]
fn min_resurfaces_the_runner_up() {
    drive_reduce(
        "min",
        |g, l| {
            let m = g.nary_with(&[l], |s| reduce::min(s[0].clone(), ONE));
            flat(g, m)
        },
        |vals| vec![vals[0].0],
    );
}

#[test]
fn max_resurfaces_the_runner_up() {
    drive_reduce(
        "max",
        |g, l| {
            let m = g.nary_with(&[l], |s| reduce::max(s[0].clone(), ONE));
            flat(g, m)
        },
        |vals| vec![vals.last().unwrap().0],
    );
}

#[test]
fn top_k_keeps_the_greatest_distinct_values() {
    drive_reduce(
        "top-k",
        |g, l| {
            g.nary_with(&[l], |s| reduce::top_k(s[0].clone(), MANY, 2))
                .rows()
        },
        |vals| vals.iter().rev().take(2).map(|(v, _)| *v).collect(),
    );
}

#[test]
fn aggregate_recomputes_each_group() {
    const SUMS: KeyspaceId<u8, Mergeless<i64>> = KeyspaceId::new(15);
    drive_reduce(
        "aggregate",
        |g, l| {
            let a = g.nary_with(&[l], |s| {
                reduce::aggregate(s[0].clone(), SUMS, |_: &u8, vals: &[(u8, Delta)]| {
                    vals.iter().map(|(v, d)| *v as i64 * d).sum::<i64>()
                })
            });
            flat(g, a)
        },
        |vals| vec![vals.iter().map(|(v, d)| *v as i64 * d).sum::<i64>()],
    );
}

#[test]
fn linear_operators_fuse_and_reshape_rows() {
    drive(
        "linear",
        |g, l, _| {
            let x = Then(
                FilterKV(|k: &u8, _: &u8| *k != 3),
                Then(
                    ExplodeVal(|v: &u8| [(*v, 1), (v + 10, 2)]),
                    Then(Reindex(|_: &u8, v: &u8| *v % 2), Keys),
                ),
            );
            g.nary(&[l], FusedLinears::new(x))
        },
        |l, _| {
            let mut out = Rows::new();
            for (&(k, v), &d) in l {
                if k != 3 {
                    add(&mut out, &(v % 2, k), d);
                    add(&mut out, &((v + 10) % 2, k), 2 * d);
                }
            }
            out
        },
    );
}

// ---------------------------------------------------------------------------
// Tables, read as state
// ---------------------------------------------------------------------------

const L: KeyspaceId<u8, Mergeless<u8>> = KeyspaceId::new(1);
const R: KeyspaceId<u8, Mergeless<u8>> = KeyspaceId::new(2);
const BY_VALUE: Arrangement<u8, u8> = KeyspaceId::new(20);
const BY_VALUE_TOTALS: KeyspaceId<u8, Delta> = KeyspaceId::new(23);
const MAX_KEY: KeyspaceId<u8, Mergeless<u8>> = KeyspaceId::new(21);
const TOP: Arrangement<u8, u8> = KeyspaceId::new(22);

type Table = BTreeMap<u8, u8>;

/// Sinks for every operator in [`table_graph`], and what each should
/// hold for tables `l` and `r`.
#[derive(Clone)]
struct Sinks {
    join: Sink<(u8, u8, u8)>,
    matching: Sink<(u8, u8)>,
    view_join: Sink<(u8, u8, u8)>,
    max_key: Sink<(u8, u8)>,
    top: Sink<(u8, u8)>,
    by_value_join: Sink<(u8, u8, u8)>,
}

impl Sinks {
    fn new() -> Self {
        Sinks {
            join: Sink::new(),
            matching: Sink::new(),
            view_join: Sink::new(),
            max_key: Sink::new(),
            top: Sink::new(),
            by_value_join: Sink::new(),
        }
    }
    fn commit(&self) {
        self.join.commit();
        self.matching.commit();
        self.view_join.commit();
        self.max_key.commit();
        self.top.commit();
        self.by_value_join.commit();
    }
    fn abort(&self) {
        self.join.abort();
        self.matching.abort();
        self.view_join.abort();
        self.max_key.abort();
        self.top.abort();
        self.by_value_join.abort();
    }
    fn reset(&self) {
        self.join.reset();
        self.matching.reset();
        self.view_join.reset();
        self.max_key.reset();
        self.top.reset();
        self.by_value_join.reset();
    }
    fn check(&self, l: &Table, r: &Table, at: &str) {
        let join = ones(
            l.iter()
                .filter_map(|(k, a)| Some((*k, *a, *r.get(k)?)))
                .collect(),
        );
        assert_eq!(self.join.rows(), join, "join {at}");
        let matching = ones(
            l.iter()
                .filter(|(k, _)| r.contains_key(k))
                .map(|(k, a)| (*k, *a))
                .collect(),
        );
        assert_eq!(self.matching.rows(), matching, "matching {at}");
        let view = ones(
            l.iter()
                .filter_map(|(k, a)| Some((*k, *a, *r.get(k)? % 3)))
                .collect(),
        );
        assert_eq!(self.view_join.rows(), view, "view join {at}");
        let mut by_value: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
        for (k, v) in l {
            by_value.entry(*v).or_default().push(*k);
        }
        let max_key = ones(
            by_value
                .iter()
                .map(|(v, ks)| (*v, *ks.last().unwrap()))
                .collect(),
        );
        assert_eq!(self.max_key.rows(), max_key, "max key {at}");
        let top = ones(
            by_value
                .iter()
                .flat_map(|(v, ks)| ks.iter().rev().take(2).map(|k| (*v, *k)))
                .collect(),
        );
        assert_eq!(self.top.rows(), top, "top {at}");
        let bvj = ones(
            l.iter()
                .filter_map(|(k, v)| Some((*v, *k, *r.get(v)?)))
                .collect(),
        );
        assert_eq!(self.by_value_join.rows(), bvj, "by-value join {at}");
    }
}

fn ones<T: Ord>(rows: Vec<T>) -> Rows<T> {
    rows.into_iter().map(|t| (t, 1)).collect()
}

/// Every table-reading operator at once, with `version` mixed into the
/// fingerprint.
fn table_graph(version: u64, sinks: &Sinks) -> OpDag {
    let mut g = OpDag::new();
    g.version(version);
    let l = g.table(L);
    let r = g.table(R);
    // both sides are tables keyed by the join key: no state at all
    let join = g.binary_with(l, r, |l, r| {
        Join::new(l, r, |k: &u8, a: &u8, b: &u8| (*k, *a, *b))
    });
    g.nary(&[join], sinks.join.clone());
    let matching = g.binary_with(l, r, Semijoin::matching);
    g.nary(&[matching], sinks.matching.clone());
    // a view stores nothing either
    let view = g.map_values(r, |b: &u8| b % 3);
    let vj = g.binary_with(l, view, |l, v| {
        Join::new(l, v, |k: &u8, a: &u8, b: &u8| (*k, *a, *b))
    });
    g.nary(&[vj], sinks.view_join.clone());
    // one arrangement by value, shared by three readers
    let rows = g.unary(l, Flatten::default());
    let swapped = g.nary(&[rows], FusedLinears::new(Map(|&(k, v): &(u8, u8)| (v, k))));
    let by_value = g.nary(&[swapped], Arrange::new(BY_VALUE, BY_VALUE_TOTALS));
    let max = g.nary_with(&[by_value], |s| reduce::max(s[0].clone(), MAX_KEY));
    let max = g.unary(max, Flatten::default());
    g.nary(&[max], sinks.max_key.clone());
    let top = g.nary_with(&[by_value], |s| reduce::top_k(s[0].clone(), TOP, 2));
    g.nary(&[top.rows()], sinks.top.clone());
    let bvj = g.binary_with(by_value, r, |a, r: Stored<u8, Mergeless<u8>>| {
        Join::new(a, r, |v: &u8, k: &u8, b: &u8| (*v, *k, *b))
    });
    g.nary(&[bvj], sinks.by_value_join.clone());
    g
}

#[test]
fn operators_read_tables_views_and_shared_arrangements() {
    let dir = tmpdir("tables");
    let sinks = Sinks::new();
    let (mut l, mut r) = (Table::new(), Table::new());
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    // the graph changes in sessions 2 and 4, so derived state is rebuilt
    // from the tables, two rows a transaction
    for (session, version) in [0, 0, 1, 1, 2].into_iter().enumerate() {
        let rebuilds = session > 0 && version != [0, 0, 1, 1, 2][session - 1];
        if rebuilds {
            sinks.reset();
        }
        let mut db = Db::open_inner(builder(&dir), table_graph(version, &sinks), 2, None).unwrap();
        sinks.commit();
        sinks.check(&l, &r, &format!("after opening session {session}"));
        for step in 0..50 {
            let (mut nl, mut nr) = (l.clone(), r.clone());
            let abort = rng.below(10) == 0;
            let res = db.write(|tx| {
                for _ in 0..1 + rng.below(2) {
                    for _ in 0..1 + rng.below(4) {
                        let (k, v) = (rng.below(6) as u8, rng.below(4) as u8);
                        let value = (rng.below(4) != 0).then_some(v);
                        let (model, id) = if rng.below(2) == 0 {
                            (&mut nl, L)
                        } else {
                            (&mut nr, R)
                        };
                        match value {
                            Some(v) => model.insert(k, v),
                            None => model.remove(&k),
                        };
                        tx.table(id).set(k, value)?;
                    }
                    tx.run()?;
                }
                if abort {
                    return Err(Error::InvalidOption("abort"));
                }
                Ok(())
            });
            match res {
                Ok(()) => {
                    sinks.commit();
                    (l, r) = (nl, nr);
                }
                Err(_) => sinks.abort(),
            }
            sinks.check(&l, &r, &format!("session {session}, step {step}"));
        }
        db.store().checkpoint().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derived_keyspaces_are_the_graphs_to_write() {
    let dir = tmpdir("derived-writes");
    let sinks = Sinks::new();
    let mut db = Db::open(builder(&dir), table_graph(0, &sinks)).unwrap();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.write(|tx| {
            tx.table(MAX_KEY).set(1, Some(1))?;
            Ok(())
        })
    }));
    assert!(res.is_err(), "writing a derived keyspace panics");
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[should_panic(expected = "is written by node")]
fn a_keyspace_has_one_writer() {
    let mut g = OpDag::new();
    let l = g.table(L);
    let rows = g.unary(l, Flatten::default());
    g.nary(&[rows], Arrange::new(BY_VALUE, BY_VALUE_TOTALS));
    g.nary(&[rows], Arrange::new(BY_VALUE, BY_VALUE_TOTALS));
}

#[test]
fn foreign_keys_report_dangling_references_and_reject() {
    const CHILDREN: KeyspaceId<u8, Mergeless<u8>> = KeyspaceId::new(30);
    const PARENTS: KeyspaceId<u8, Mergeless<()>> = KeyspaceId::new(31);
    const BY_PARENT: Arrangement<u8, u8> = KeyspaceId::new(32);
    #[derive(Debug)]
    struct Dangling(u8);
    impl std::fmt::Display for Dangling {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "no parent {}", self.0)
        }
    }
    impl std::error::Error for Dangling {}

    let mut g = OpDag::new();
    let children = g.table(CHILDREN);
    let parents = g.table(PARENTS);
    let rows = g.unary(children, Flatten::default());
    let swapped = g.nary(&[rows], FusedLinears::new(Map(|&(c, p): &(u8, u8)| (p, c))));
    let by_parent = g.nary(&[swapped], Arrange::new(BY_PARENT, KeyspaceId::new(33)));
    let dangling = g.binary_with(by_parent, parents, ForeignKey::new);
    let sink = Sink::new();
    g.nary(&[dangling], sink.clone());
    g.nary(&[dangling], Reject::new(|k: &u8| Dangling(*k)));
    let mut db = Db::open(StoreBuilder::new(DurabilityPolicy::NONE), g).unwrap();

    let mut write = |c: &[(u8, Option<u8>)], p: &[(u8, bool)]| {
        let res = db.write(|tx| {
            for &(child, parent) in c {
                tx.table(CHILDREN).set(child, parent)?;
            }
            for &(k, live) in p {
                tx.table(PARENTS).set(k, live.then_some(()))?;
            }
            Ok(())
        });
        // a refused tx still reported its violation
        sink.commit();
        res
    };

    // a parent and its child in one tx, then a second child
    write(&[(10, Some(1))], &[(1, true)]).unwrap();
    write(&[(11, Some(1))], &[]).unwrap();
    // a child without a parent is refused, and leaves nothing behind
    let err = write(&[(12, Some(2))], &[]).unwrap_err();
    assert!(
        matches!(&err, Error::Rejected(e) if e.to_string() == "no parent 2"),
        "{err}"
    );
    // removing a referenced parent is refused
    assert!(write(&[], &[(1, false)]).is_err());
    // removing its children first frees it
    write(&[(10, None), (11, None)], &[(1, false)]).unwrap();
    write(&[(12, Some(2))], &[(2, true)]).unwrap();
    assert_eq!(
        sink.rows(),
        [(2, 1), (1, 1)].into_iter().collect(),
        "each refused tx reported its violation"
    );
}

#[test]
fn unstored_sources_need_the_application_to_replay_them() {
    const EVENTS: Arrangement<u8, u8> = KeyspaceId::new(40);
    let dir = tmpdir("given");
    let graph = |version| {
        let mut g = OpDag::new();
        g.version(version);
        let src = g.source::<Bag<(u8, u8)>>();
        g.nary(&[src], Arrange::new(EVENTS, KeyspaceId::new(41)));
        (g, src)
    };
    let rows = [(1u8, 2u8), (1, 3), (4, 5)];
    {
        let (g, src) = graph(0);
        let mut db = Db::open(builder(&dir), g).unwrap();
        db.write(|tx| {
            let mut b = Bag::new();
            for r in rows {
                b.push(r, 1);
            }
            tx.give(src, &mut b);
            Ok(())
        })
        .unwrap();
    }
    let (g, src) = graph(1);
    assert!(matches!(
        Db::open(builder(&dir), g),
        Err(Error::RebuildRequired)
    ));
    let (g, _) = graph(1);
    let mut replayed = false;
    let db = Db::open_with(builder(&dir), g, &mut |tx, g| {
        let mut b = Bag::new();
        for r in rows {
            b.push(r, 1);
        }
        g.give(src, &mut b);
        g.run(tx);
        replayed = true;
        Ok(false)
    })
    .unwrap();
    assert!(replayed);
    let arranged: Vec<(u8, u8)> = db
        .snapshot()
        .range(EVENTS, ..)
        .map(|r| {
            r.map(|(k, _)| match &k.1 {
                crate::state::Edge::Val(v) => (k.0, *v),
                _ => unreachable!(),
            })
            .unwrap()
        })
        .collect();
    assert_eq!(arranged, rows);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn matching_reads_a_distincts_state() {
    const SEEN: KeyspaceId<u8, Delta> = KeyspaceId::new(18);
    drive(
        "distinct-state",
        |g, l, r| {
            let l = g.nary(&[l], Arrange::new(AL, TL));
            let keys = g.nary(&[r], FusedLinears::new(Keys));
            let seen = g.nary(&[keys], crate::Distinct::new(SEEN));
            g.binary_with(l, seen, Semijoin::matching)
        },
        |l, r| semijoin_expected(l, r, false),
    );
}

#[test]
fn a_key_is_present_while_its_rows_total_positive() {
    let dir = tmpdir("totals");
    let mut g = OpDag::new();
    let (ls, rs) = (g.source::<Bag<(u8, u8)>>(), g.source::<Bag<(u8, u8)>>());
    let l = g.nary(&[ls], Arrange::new(AL, TL));
    let r = g.nary(&[rs], Arrange::new(AR, TR));
    let m = g.binary_with(l, r, Semijoin::matching);
    let sink = Sink::new();
    g.nary(&[m], sink.clone());
    let mut db = Db::open(builder(&dir), g).unwrap();
    let mut step = |left: &[((u8, u8), Delta)], right: &[((u8, u8), Delta)]| {
        db.write(|tx| {
            let (mut bl, mut br) = (Bag::new(), Bag::new());
            for &(row, d) in left {
                bl.push(row, d);
            }
            for &(row, d) in right {
                br.push(row, d);
            }
            tx.give(ls, &mut bl);
            tx.give(rs, &mut br);
            Ok(())
        })
        .unwrap();
        sink.commit();
        sink.rows()
    };
    // key 1's right rows net to zero, so it is absent despite a positive row
    assert!(step(&[((1, 7), 1)], &[((1, 0), 1), ((1, 1), -1)]).is_empty());
    assert_eq!(step(&[], &[((1, 2), 1)]), ones(vec![(1, 7)]));
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
