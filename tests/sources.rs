//! The stored sources, subscriptions and commit hooks, through the
//! public API: a table with minted ids, a set, a cell, a multiset and an
//! event log with retention feed a small graph, which a subscriber
//! follows.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use fold3::{
    Change, Db, Distinct, DurabilityPolicy, FlatMap, FusedLinears, Id, KeyspaceId, OpDag,
    StoreBuilder, Sum, Tally,
    ops::Flatten,
    source::{Cell, Events, Multiset, Retain, Set, Table},
};

/// A reading: (sensor, value).
type Reading = (u8, i64);

const NOTES: Table<Id<String>, String> = Table::new(1);
const SENSORS: Set<u8> = Set::new(2);
const LIMIT: Cell<i64> = Cell::new(3);
const TAGS: Multiset<String> = Multiset::new(4);
const READINGS: Events<Reading> = Events::new(5);

/// The sum and count of the readings kept, by sensor.
const TOTALS: KeyspaceId<u8, (i64, i64)> = KeyspaceId::new(10);
/// Each tag present, once.
const TAG_COUNTS: KeyspaceId<String, Tally> = KeyspaceId::new(11);

fn graph(version: u64) -> OpDag {
    let mut g = OpDag::new();
    g.version(version);
    NOTES.source(&mut g);
    SENSORS.source(&mut g);
    LIMIT.source(&mut g);
    let readings = READINGS.source(&mut g);
    let rows = g.unary(readings, Flatten::default());
    let by_sensor = g.nary(
        &[rows],
        FusedLinears::new(FlatMap(|&(_, reading): &(Id<Reading>, Reading)| [reading])),
    );
    g.nary(&[by_sensor], Sum::<u8, i64>::new(TOTALS));
    let tags = TAGS.source(&mut g);
    g.nary(&[tags], Distinct::new(TAG_COUNTS));
    g
}

fn dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fold3-sources-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn open(dir: &PathBuf, version: u64) -> Db {
    Db::open(
        StoreBuilder::new(DurabilityPolicy::RELAXED).dir(dir),
        graph(version),
    )
    .unwrap()
}

fn totals(db: &Db) -> BTreeMap<u8, (i64, i64)> {
    db.snapshot()
        .range(TOTALS, ..)
        .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
        .collect()
}

fn tags(db: &Db) -> Vec<String> {
    db.snapshot()
        .range(TAG_COUNTS, ..)
        .map(|r| r.unwrap().0.into_owned())
        .collect()
}

fn readings(db: &Db) -> Vec<Reading> {
    db.snapshot()
        .range(READINGS.id(), ..)
        .map(|r| *r.unwrap().1)
        .collect()
}

#[test]
fn tables_sets_cells_and_multisets_write_through_a_tx() {
    let dir = dir("shapes");
    let db = open(&dir, 0);
    let (first, second) = db
        .write(|tx| {
            let a = NOTES.insert(tx, "one".into())?;
            let b = NOTES.insert(tx, "two".into())?;
            assert!(SENSORS.insert(tx, 7)?);
            assert!(!SENSORS.insert(tx, 7)?, "already there");
            LIMIT.set(tx, 100)?;
            TAGS.add(tx, "hot".into(), 2)?;
            TAGS.add(tx, "cold".into(), 1)?;
            Ok((a, b))
        })
        .unwrap();
    assert!(first < second, "minted in order");
    assert_eq!((first.seq(), second.seq()), (0, 1));
    assert_eq!(first.epoch(), second.epoch());
    assert_eq!(tags(&db), ["cold", "hot"]);

    let third = db
        .write(|tx| {
            assert_eq!(NOTES.get(tx, &first)?.as_deref(), Some("one"));
            assert!(NOTES.update(tx, &first, |n| n.push('!'))?);
            assert!(NOTES.remove(tx, &second)?);
            assert!(!NOTES.remove(tx, &second)?, "already gone");
            assert!(SENSORS.contains(tx, &7)?);
            assert!(SENSORS.remove(tx, &7)?);
            assert_eq!(LIMIT.get(tx)?, Some(100));
            LIMIT.clear(tx)?;
            // one of two leaves the tag present; the last one removes it
            TAGS.remove(tx, "hot".into(), 1)?;
            TAGS.remove(tx, "cold".into(), 1)?;
            assert_eq!(TAGS.get(tx, &"hot".into())?, 1);
            assert_eq!(TAGS.get(tx, &"cold".into())?, 0);
            NOTES.insert(tx, "three".into())
        })
        .unwrap();
    assert!(third > second, "later commits mint greater ids");
    assert_eq!(tags(&db), ["hot"]);
    let snap = db.snapshot();
    assert_eq!(
        snap.get(NOTES.id(), &first)
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("one!")
    );
    assert_eq!(snap.get(NOTES.id(), &second).unwrap(), None);
    assert_eq!(snap.get(LIMIT.id(), &()).unwrap(), None);
    assert_eq!(snap.get(SENSORS.id(), &7).unwrap(), None);
    assert_eq!(snap.get(TAGS.id(), &"cold".into()).unwrap(), None);
    drop(snap);
    drop(db);

    // a changed graph rebuilds from the stored multiset as from a table
    let db = open(&dir, 1);
    assert_eq!(tags(&db), ["hot"]);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_event_log_keeps_what_its_policy_says() {
    let dir = dir("events");
    let db = open(&dir, 0);
    db.retain(READINGS, Retain::Count(3));
    for i in 0..5i64 {
        db.write(|tx| READINGS.push(tx, (1, i)).map(|_| ()))
            .unwrap();
    }
    assert_eq!(readings(&db), [(1, 2), (1, 3), (1, 4)]);
    // the graph saw the evictions as deletes
    assert_eq!(totals(&db), BTreeMap::from([(1, (3, 9))]));

    // several pushes in one commit, more than the log keeps
    db.write(|tx| {
        for i in 10..15i64 {
            READINGS.push(tx, (2, i))?;
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(readings(&db), [(2, 12), (2, 13), (2, 14)]);
    assert_eq!(totals(&db), BTreeMap::from([(2, (3, 39))]));

    // a refused commit evicts nothing, and the count stays right after it
    let err = db.write(|tx| {
        READINGS.push(tx, (3, 1))?;
        Err::<(), _>(fold3::Error::Rejected(Arc::new(std::fmt::Error)))
    });
    assert!(err.is_err());
    db.write(|tx| READINGS.push(tx, (3, 100)).map(|_| ()))
        .unwrap();
    assert_eq!(readings(&db), [(2, 13), (2, 14), (3, 100)]);
    drop(db);

    // the policy is not stored; the log is, and the count is recovered
    let db = open(&dir, 0);
    db.retain(READINGS, Retain::Count(2));
    db.write(|tx| READINGS.push(tx, (4, 1)).map(|_| ()))
        .unwrap();
    assert_eq!(readings(&db), [(3, 100), (4, 1)]);

    // by epoch: this commit's rows and the one before it
    db.retain(READINGS, Retain::Epochs(2));
    db.write(|tx| READINGS.push(tx, (5, 1)).map(|_| ()))
        .unwrap();
    assert_eq!(readings(&db), [(4, 1), (5, 1)]);
    db.write(|tx| {
        READINGS.push(tx, (6, 1))?;
        READINGS.push(tx, (6, 2)).map(|_| ())
    })
    .unwrap();
    assert_eq!(readings(&db), [(5, 1), (6, 1), (6, 2)]);
    assert_eq!(totals(&db), BTreeMap::from([(5, (1, 1)), (6, (2, 3))]));
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_subscription_follows_a_keyspace_from_its_snapshot() {
    let dir = dir("subscribe");
    let db = open(&dir, 0);
    db.write(|tx| READINGS.push(tx, (1, 5)).map(|_| ()))
        .unwrap();

    let commits = Arc::new(AtomicU64::new(0));
    let seen = commits.clone();
    db.on_commit(move |snap| seen.store(snap.epoch(), Ordering::SeqCst));

    // a derived keyspace: the snapshot holds what came before
    let (snap, sub) = db.subscribe(TOTALS);
    let mut state: BTreeMap<u8, (i64, i64)> = snap
        .range(TOTALS, ..)
        .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
        .collect();
    let since = snap.epoch();
    drop(snap);
    assert_eq!(state, BTreeMap::from([(1, (1, 5))]));
    assert!(sub.try_recv().unwrap().is_none(), "nothing committed since");

    db.write(|tx| {
        READINGS.push(tx, (1, 2))?;
        READINGS.push(tx, (2, 9)).map(|_| ())
    })
    .unwrap();
    // a commit that does not touch the keyspace sends nothing
    db.write(|tx| LIMIT.set(tx, 1)).unwrap();
    db.retain(READINGS, Retain::Count(1));
    db.write(|tx| READINGS.push(tx, (2, 1)).map(|_| ()))
        .unwrap();
    assert_eq!(commits.load(Ordering::SeqCst), db.snapshot().epoch());

    let first = sub.try_recv().unwrap().expect("the first commit's delta");
    assert_eq!(first.epoch, since + 1);
    assert_eq!(first.changes.len(), 2);
    assert!(matches!(
        first.changes[0],
        (1, Change::Replace((1, 5), (2, 7)))
    ));
    assert!(matches!(first.changes[1], (2, Change::Insert((1, 9)))));
    let second = sub.try_recv().unwrap().expect("the eviction's delta");
    assert_eq!(second.epoch, since + 3);
    assert!(sub.try_recv().unwrap().is_none());

    // the snapshot plus every delta is the live state
    for delta in [first, second] {
        for (k, change) in delta.changes.iter() {
            match change {
                Change::Insert(v) | Change::Replace(_, v) => state.insert(*k, *v),
                Change::Delete(_) => state.remove(k),
                Change::Modify(_) => unreachable!("committed changes are resolved"),
            };
        }
    }
    assert_eq!(state, totals(&db));
    assert_eq!(state, BTreeMap::from([(2, (1, 1))]));

    // a second subscriber starts from its own snapshot; a dropped store
    // closes both
    let (snap, late) = db.subscribe(TOTALS);
    drop(snap);
    db.write(|tx| READINGS.push(tx, (3, 3)).map(|_| ()))
        .unwrap();
    assert_eq!(
        late.try_recv().unwrap().unwrap().epoch,
        sub.try_recv().unwrap().unwrap().epoch
    );

    let checkpoints = Arc::new(AtomicU64::new(0));
    let seen = checkpoints.clone();
    db.on_checkpoint(move |epoch| seen.store(epoch, Ordering::SeqCst));
    let through = db.store().checkpoint().unwrap().unwrap();
    assert_eq!(checkpoints.load(Ordering::SeqCst), through);

    drop(db);
    assert!(matches!(sub.try_recv(), Err(fold3::subscribe::Closed)));
    assert!(late.recv().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
