//! A commit as a value: `on_changes` reports what each commit wrote to
//! the sources, and `apply` writes it to another database of the same
//! schema, which then derives the same views. Inverted, it is an undo.

use std::sync::mpsc;

use bog::schema::{
    Cell, ChangeSet, Declare, Events, MapView, Multiset, OrderedView, Schema, Set, Table, ValueView,
};

#[derive(Schema)]
struct Inner {
    members: Set<u32>,
    #[view]
    member_count: ValueView<i64>,
}

impl Declare for Inner {
    fn declare(b: &mut InnerBuilder) {
        b.member_count.define(&b.members.count());
    }
}

#[derive(Schema)]
struct World {
    /// Player -> (name, cell).
    players: Table<u32, (String, u16)>,
    log: Events<String>,
    title: Cell<String>,
    tags: Multiset<String>,
    #[schema]
    inner: Inner,
    /// Cell -> who stands there.
    #[view]
    at: MapView<u16, (String, u16)>,
    #[view]
    name_lengths: OrderedView<u32, i64>,
    #[view]
    tag_total: ValueView<i64>,
    #[view]
    log_count: ValueView<i64>,
}

impl Declare for World {
    fn declare(b: &mut WorldBuilder) {
        b.at.define(&b.players.unique_key(|(_, cell)| *cell));
        b.name_lengths
            .define(&b.players.map(|(name, _)| name.len() as i64));
        b.tag_total.define(&b.tags.count());
        b.log_count.define(&b.log.count());
    }
}

/// Everything readable, as one comparable value.
fn state(db: &bog::Db<World>) -> String {
    db.read(|r| {
        format!(
            "{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            r.players.iter().map(|r| r.unwrap()).collect::<Vec<_>>(),
            r.log.iter().map(|r| r.unwrap().1).collect::<Vec<_>>(),
            r.title.get().unwrap(),
            r.tags.iter().map(|r| r.unwrap()).collect::<Vec<_>>(),
            r.inner
                .members
                .iter()
                .map(|r| r.unwrap())
                .collect::<Vec<_>>(),
            r.inner.member_count.get().unwrap(),
            r.at.iter().map(|r| r.unwrap()).collect::<Vec<_>>(),
            r.name_lengths
                .desc()
                .map(|r| r.unwrap())
                .collect::<Vec<_>>(),
            r.tag_total.get().unwrap(),
            r.log_count.get().unwrap(),
        )
    })
}

fn primary() -> (bog::Db<World>, mpsc::Receiver<Vec<u8>>) {
    let (send, recv) = mpsc::channel();
    let db = World::memory()
        .on_changes(move |changes, _| send.send(changes.encode()).unwrap())
        .build()
        .unwrap();
    (db, recv)
}

#[test]
fn a_replica_fed_every_commit_derives_the_same_state() {
    let (db, wire) = primary();
    let replica = World::memory().build().unwrap();
    let sync = |db: &bog::Db<World>| {
        let mut epochs = Vec::new();
        for bytes in wire.try_iter() {
            let changes = WorldChanges::decode(&bytes).unwrap();
            assert!(!changes.is_empty());
            epochs.push(changes.epoch);
            replica.write(|tx| tx.apply(&changes)).unwrap();
        }
        assert_eq!(state(db), state(&replica));
        epochs
    };

    db.write(|tx| {
        tx.players.upsert(1, ("ann".into(), 10))?;
        tx.players.upsert(2, ("bob".into(), 20))?;
        tx.log.push("hello".into())?;
        tx.title.set("lobby".into())?;
        tx.tags.add("red".into(), 2)?;
        tx.inner.members.extend([1, 2, 3])
    })
    .unwrap();
    assert_eq!(sync(&db), [1]);

    // two rows trade unique keys in one commit
    db.write(|tx| {
        tx.players.remove(&1)?;
        tx.players.upsert(2, ("bob".into(), 10))?;
        tx.players.upsert(1, ("ann".into(), 20))
    })
    .unwrap();
    db.write(|tx| {
        tx.players.remove(&2)?;
        tx.log.push("bye bob".into())?;
        tx.title.clear()?;
        tx.tags.add("red".into(), -1)?;
        tx.tags.add("blue".into(), 5)?;
        tx.inner.members.remove(&2).map(drop)
    })
    .unwrap();
    assert_eq!(sync(&db), [2, 3]);

    // an attempt that fails leaves no trace in the commit's changes
    db.write(|tx| {
        assert!(
            tx.attempt(|tx| tx.players.upsert(7, ("zed".into(), 20)))
                .is_err()
        );
        tx.players.upsert(8, ("yan".into(), 30))
    })
    .unwrap();
    assert_eq!(sync(&db), [4]);

    // a commit that changes nothing reports nothing
    db.write(|tx| tx.players.upsert(8, ("yan".into(), 30)))
        .unwrap();
    assert_eq!(sync(&db), [] as [u64; 0]);
}

#[test]
fn inverted_changes_undo_a_commit() {
    let (db, wire) = primary();
    db.write(|tx| {
        tx.players.upsert(1, ("ann".into(), 10))?;
        tx.title.set("lobby".into())?;
        tx.tags.add("red".into(), 2)?;
        tx.inner.members.insert(1).map(drop)
    })
    .unwrap();
    let before = state(&db);
    wire.try_iter().count();

    db.write(|tx| {
        tx.players.upsert(1, ("anne".into(), 11))?;
        tx.players.upsert(2, ("bob".into(), 10))?;
        tx.log.push("moved".into())?;
        tx.title.set("arena".into())?;
        tx.tags.add("red".into(), -2)?;
        tx.tags.add("blue".into(), 1)?;
        tx.inner.members.remove(&1)?;
        tx.inner.members.insert(2).map(drop)
    })
    .unwrap();
    assert_ne!(state(&db), before);
    let changes = WorldChanges::decode(&wire.try_recv().unwrap()).unwrap();

    db.write(|tx| tx.apply(&changes.clone().invert())).unwrap();
    assert_eq!(state(&db), before);
    // and the undo was itself a commit, reported like any other
    let undo = WorldChanges::decode(&wire.try_recv().unwrap()).unwrap();
    assert_eq!(undo.players.len(), 2);
    db.write(|tx| tx.apply(&undo.invert())).unwrap();
    db.write(|tx| tx.apply(&changes.invert())).unwrap();
    assert_eq!(state(&db), before);
}
