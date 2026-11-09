//! Model tests: every operator's accumulated output must equal its
//! definition recomputed from scratch, after every transaction, while
//! its state moves through layers, checkpoints, compaction and reopens.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use super::{linear::*, reduce, *};
use crate::{
    FusedLinears, Nary, Then,
    error::Error,
    graph::{NodeRef, OpDag},
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
    fn rows(&self) -> Rows<T> {
        self.shared.lock().unwrap().0.clone()
    }
}

impl<T: Ord + Clone + Send + 'static> Nary for Sink<T> {
    type Takes = Bag<T>;
    type Gives = Bag<()>;
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

type Src = NodeRef<Bag<(u8, u8)>>;

/// Drive random changes to two collections of `(key, value)` rows through
/// the graph `build` makes, checking the sink against `expected` of the
/// collections after every tx. Reopens the store twice on the way.
fn drive<O: Ord + Clone + Debug + Send + 'static>(
    name: &str,
    declare: impl Fn(StoreBuilder) -> StoreBuilder,
    build: impl Fn(&mut OpDag, Src, Src) -> NodeRef<Bag<O>>,
    expected: impl Fn(&Rows<(u8, u8)>, &Rows<(u8, u8)>) -> Rows<O>,
) {
    let dir = tmpdir(name);
    let sink = Sink::<O>::new();
    let (mut left, mut right) = (Rows::new(), Rows::new());
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ name.len() as u64);
    for session in 0..3 {
        let store = declare(
            StoreBuilder::new(DurabilityPolicy::RELAXED)
                .dir(&dir)
                .memtable_limit(1500)
                .options_with(|o| {
                    o.max_tables = 3;
                    o.compaction_width = 2;
                }),
        )
        .build()
        .unwrap();
        let mut g = OpDag::new();
        let (l, r) = (g.source(), g.source());
        let out = build(&mut g, l, r);
        g.nary(&[out], sink.clone());

        for step in 0..60 {
            let mut tx = store.begin().unwrap();
            let (mut nl, mut nr) = (left.clone(), right.clone());
            let runs = 1 + rng.below(2);
            for _ in 0..runs {
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
                g.give(l, &mut bl);
                g.give(r, &mut br);
                g.run(&mut tx);
            }
            if rng.below(8) == 0 {
                drop(tx);
                sink.abort();
            } else {
                store.finish(tx).unwrap();
                sink.commit();
                (left, right) = (nl, nr);
            }
            assert_eq!(
                sink.rows(),
                expected(&left, &right),
                "{name}: session {session}, step {step}\nleft {left:?}\nright {right:?}"
            );
        }
        // later sessions read whatever state there is from tables
        store.checkpoint().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const AL: Arrangement<u8, u8> = KeyspaceId::new(10);
const AR: Arrangement<u8, u8> = KeyspaceId::new(11);
const KEYS: KeyspaceId<u8, Delta> = KeyspaceId::new(12);

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
        |b| b.derived(AL).derived(AR),
        |g, l, r| {
            g.binary(
                l,
                r,
                Join::new(AL, AR, |k: &u8, a: &u8, b: &u8| (*k, *a, *b)),
            )
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
        |b| b.derived(AL).derived(KEYS),
        |g, l, r| g.binary(l, r, Semijoin::matching(AL, KEYS)),
        |l, r| semijoin_expected(l, r, false),
    );
}

#[test]
fn without_passes_rows_whose_key_the_other_side_lacks() {
    drive(
        "without",
        |b| b.derived(AL).derived(KEYS),
        |g, l, r| g.binary(l, r, Semijoin::without(AL, KEYS)),
        |l, r| semijoin_expected(l, r, true),
    );
}

#[test]
fn minus_negates_the_right() {
    drive(
        "minus",
        |b| b,
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

/// A reduce over the left rows, checked against `f` of each key's
/// group, ascending by value.
fn drive_reduce<O, F>(
    name: &str,
    declare: impl Fn(StoreBuilder) -> StoreBuilder,
    build: impl Fn(&mut OpDag, Src) -> NodeRef<Bag<(u8, O)>>,
    f: F,
) where
    O: Ord + Clone + Debug + Send + 'static,
    F: Fn(&[(u8, Delta)]) -> Vec<O>,
{
    drive(
        name,
        declare,
        |g, l, _| build(g, l),
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

const OUT: Arrangement<u8, u8> = KeyspaceId::new(13);

#[test]
fn min_resurfaces_the_runner_up() {
    drive_reduce(
        "min",
        |b| b.derived(AL).derived(OUT),
        |g, l| g.nary(&[l], reduce::min(AL, OUT)),
        |vals| vec![vals[0].0],
    );
}

#[test]
fn max_resurfaces_the_runner_up() {
    drive_reduce(
        "max",
        |b| b.derived(AL).derived(OUT),
        |g, l| g.nary(&[l], reduce::max(AL, OUT)),
        |vals| vec![vals.last().unwrap().0],
    );
}

#[test]
fn top_k_keeps_the_greatest_distinct_values() {
    drive_reduce(
        "top-k",
        |b| b.derived(AL).derived(OUT),
        |g, l| g.nary(&[l], reduce::top_k(AL, OUT, 2)),
        |vals| vals.iter().rev().take(2).map(|(v, _)| *v).collect(),
    );
}

#[test]
fn aggregate_recomputes_each_group() {
    const SUMS: Arrangement<u8, i64> = KeyspaceId::new(14);
    drive_reduce(
        "aggregate",
        |b| b.derived(AL).derived(SUMS),
        |g, l| {
            g.nary(
                &[l],
                reduce::aggregate(AL, SUMS, |_, vals: &[(u8, Delta)]| {
                    vals.iter().map(|(v, d)| *v as i64 * d).sum::<i64>()
                }),
            )
        },
        |vals| vec![vals.iter().map(|(v, d)| *v as i64 * d).sum::<i64>()],
    );
}

#[test]
fn count_totals_each_key() {
    const UNITS: Arrangement<u8, ()> = KeyspaceId::new(15);
    const COUNTS: Arrangement<u8, Delta> = KeyspaceId::new(16);
    drive_reduce(
        "count",
        |b| b.derived(UNITS).derived(COUNTS),
        |g, l| {
            let units = g.nary(&[l], FusedLinears::new(MapVal(|_: &u8| ())));
            g.nary(&[units], reduce::count(UNITS, COUNTS))
        },
        |vals| vec![vals.iter().map(|(_, d)| d).sum::<Delta>()],
    );
}

#[test]
fn linear_operators_fuse_and_reshape_rows() {
    drive(
        "linear",
        |b| b,
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

#[test]
fn foreign_keys_report_dangling_references_and_reject() {
    const REFS: KeyspaceId<u8, Delta> = KeyspaceId::new(20);
    const PARENTS: KeyspaceId<u8, Delta> = KeyspaceId::new(21);
    #[derive(Debug)]
    struct Dangling(u8);
    impl std::fmt::Display for Dangling {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "no parent {}", self.0)
        }
    }
    impl std::error::Error for Dangling {}

    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(REFS)
        .keyspace(PARENTS)
        .build()
        .unwrap();
    let mut g = OpDag::new();
    let children = g.source::<Bag<(u8, &'static str)>>();
    let parents = g.source::<Bag<(u8, ())>>();
    let dangling = g.binary(children, parents, ForeignKey::new(REFS, PARENTS));
    let sink = Sink::new();
    g.nary(&[dangling], sink.clone());
    g.nary(&[dangling], Reject::new(|k: &u8| Dangling(*k)));

    let mut write = |c: &[((u8, &'static str), Delta)], p: &[(u8, Delta)]| {
        let mut tx = store.begin().unwrap();
        let (mut bc, mut bp) = (Bag::new(), Bag::new());
        for &(row, d) in c {
            bc.push(row, d);
        }
        for &(k, d) in p {
            bp.push((k, ()), d);
        }
        g.give(children, &mut bc);
        g.give(parents, &mut bp);
        g.run(&mut tx);
        let res = store.finish(tx);
        sink.commit();
        res
    };

    // a parent and its child in one tx, then a second child
    write(&[((1, "a"), 1)], &[(1, 1)]).unwrap();
    write(&[((1, "b"), 1)], &[]).unwrap();
    // a child without a parent is refused, and leaves nothing behind
    let err = write(&[((2, "c"), 1)], &[]).unwrap_err();
    assert!(
        matches!(&err, Error::Rejected(e) if e.to_string() == "no parent 2"),
        "{err}"
    );
    // removing a referenced parent is refused
    assert!(write(&[], &[(1, -1)]).is_err());
    // removing its children first frees it
    write(&[((1, "a"), -1), ((1, "b"), -1)], &[(1, -1)]).unwrap();
    write(&[((2, "c"), 1)], &[(2, 1)]).unwrap();
    assert_eq!(
        sink.rows(),
        [(2, 1), (1, 1)].into_iter().collect(),
        "each refused tx reported its violation"
    );
}

#[test]
fn flatten_turns_a_view_into_rows() {
    use crate::Sum;
    const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(30);
    drive(
        "flatten",
        |b| b.derived(TOTALS),
        |g, l, _| {
            let rows = g.nary(&[l], FusedLinears::new(MapVal(|v: &u8| *v as i64)));
            let totals = g.nary(&[rows], Sum::<u8, i64>::new(TOTALS));
            g.unary(totals, Flatten::default())
        },
        |l, _| {
            let mut out = Rows::new();
            for (k, vals) in groups(l) {
                let n: Delta = vals.iter().map(|(_, d)| d).sum();
                let s: i64 = vals.iter().map(|(v, d)| *v as i64 * d).sum();
                add(&mut out, &(k, (n, s)), 1);
            }
            out
        },
    );
}
