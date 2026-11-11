use super::*;
use crate::{dataflow::Register, store::merge::Change, store::merge::Tally};

const KS: KeyspaceId<u8, i64> = KeyspaceId::new(3);

fn mem() -> Store {
    StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(KS)
        .build()
        .unwrap()
}

/// A fresh, empty directory unique to this test.
fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fold3-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn open(dir: &Path, dp: DurabilityPolicy) -> Result<Store> {
    StoreBuilder::new(dp).dir(dir).keyspace(KS).build()
}

fn put(store: &Store, kvs: &[(u8, Option<i64>)]) -> Option<u64> {
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    for &(k, v) in kvs {
        let change = match v {
            Some(v) => Change::Insert(v),
            None => Change::Delete(0),
        };
        if let Some(r) = ks.resolve(&k, change).unwrap() {
            ks.absorb(k, r).unwrap();
        }
    }
    store.finish(tx).unwrap()
}

fn read(store: &Store, k: u8) -> Option<i64> {
    store.snapshot().get(KS, &k).unwrap().map(|v| *v)
}

fn manifest_of(dir: &Path) -> Manifest {
    Manifest::parse(&manifest::read(dir).unwrap().unwrap()).unwrap()
}

#[test]
fn commits_are_visible_in_order_and_tombstones_shadow() {
    let store = mem();
    assert_eq!(put(&store, &[(1, Some(10)), (2, Some(20))]), Some(1));
    assert_eq!(put(&store, &[(2, None), (3, Some(30))]), Some(2));
    assert_eq!(put(&store, &[(1, Some(11))]), Some(3));

    assert_eq!(read(&store, 1), Some(11), "newest layer wins");
    assert_eq!(read(&store, 2), None, "tombstone shadows epoch 1");
    assert_eq!(read(&store, 3), Some(30));
    assert_eq!(read(&store, 4), None);
}

#[test]
fn snapshots_are_isolated_from_later_commits() {
    let store = mem();
    put(&store, &[(1, Some(10))]);
    let before = store.snapshot();
    put(&store, &[(1, Some(11))]);
    assert_eq!(before.get(KS, &1).unwrap().as_deref(), Some(&10));
    assert_eq!(before.epoch(), 1);
    assert_eq!(read(&store, 1), Some(11));
}

#[test]
fn empty_and_aborted_txs_spend_no_epoch() {
    let store = mem();
    // read-only: the keyspace is opened but nothing is written
    let mut tx = store.begin().unwrap();
    assert_eq!(tx.keyspace(KS).get(&1).unwrap(), None);
    assert_eq!(store.finish(tx).unwrap(), None);
    // aborted: dropped without finish
    {
        let mut tx = store.begin().unwrap();
        tx.keyspace(KS).absorb(1, Change::Insert(5)).unwrap();
    }
    assert_eq!(read(&store, 1), None);
    assert_eq!(put(&store, &[(1, Some(10))]), Some(1));
    assert!(store.active_snapshots.lock().unwrap().is_empty());
}

#[test]
#[should_panic(expected = "keyspace 9 not declared")]
fn undeclared_keyspaces_are_refused() {
    let store = mem();
    let mut tx = store.begin().unwrap();
    tx.keyspace(KeyspaceId::<u8, i64>::new(9));
}

#[test]
fn writing_policies_need_a_directory() {
    let err = StoreBuilder::new(DurabilityPolicy::RELAXED).build();
    assert!(matches!(err, Err(Error::Policy(_))));
}

#[test]
fn wal_gets_one_checksummed_frame_per_tx() {
    let dir = tmpdir("frames");
    {
        let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        put(&store, &[(2, None)]);
        // deleting a key that never existed writes nothing: no frame
        assert_eq!(put(&store, &[(9, None)]), None);
    }
    let bytes = fs::read(dir.join("000001.wal")).unwrap();
    fs::remove_dir_all(&dir).unwrap();

    let mut rest = &bytes[..];
    let mut frames = Vec::new();
    while !rest.is_empty() {
        let (epoch, payload, used) = parse_frame(rest).unwrap();
        // one keyspace section: id, length, then (u8 key, Option<i64>)*
        assert_eq!(
            u64::from_le_bytes(payload[..8].try_into().unwrap()),
            KS.raw()
        );
        let body_len = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
        assert_eq!(16 + body_len, payload.len());
        frames.push((epoch, payload[16..].to_vec()));
        rest = &rest[used..];
    }
    // epoch 1: key 1 -> Some(10), key 2 -> Some(20)
    let mut one = vec![1u8, 1];
    one.extend(10i64.to_be_bytes());
    one.extend([2u8, 1]);
    one.extend(20i64.to_be_bytes());
    // epoch 2: key 2 -> None
    assert_eq!(frames, [(1, one), (2, vec![2u8, 0])]);
}

#[test]
fn reopen_replays_the_wal_and_continues_epochs() {
    let dir = tmpdir("reopen");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        put(&store, &[(2, None), (3, Some(30))]);
        put(&store, &[(1, Some(11))]);
    }
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(store.snapshot().epoch(), 3);
        assert_eq!(read(&store, 1), Some(11));
        assert_eq!(read(&store, 2), None);
        assert_eq!(read(&store, 3), Some(30));
        // a clean tail is reused, not replaced
        assert_eq!(put(&store, &[(4, Some(40))]), Some(4));
    }
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(read(&store, 4), Some(40));
    let m = manifest_of(&dir);
    assert_eq!(
        m.segments,
        [Segment {
            number: 1,
            sealed: None
        }]
    );
    assert_eq!(m.keyspaces[&3], ("u8".to_string(), "i64".to_string()));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_tail_is_sealed_and_later_commits_survive() {
    let dir = tmpdir("torn");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
        put(&store, &[(2, Some(20))]);
    }
    // a crash mid-append: chop the last frame in half
    let wal = dir.join("000001.wal");
    let full = fs::read(&wal).unwrap();
    let (_, _, first) = parse_frame(&full).unwrap();
    let torn = first + (full.len() - first) / 2;
    fs::write(&wal, &full[..torn]).unwrap();
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(read(&store, 2), None, "torn frame is not replayed");
        assert_eq!(put(&store, &[(3, Some(30))]), Some(2));
    }
    let m = manifest_of(&dir);
    assert_eq!(
        m.segments,
        [
            Segment {
                number: 1,
                sealed: Some(first as u64)
            },
            Segment {
                number: 2,
                sealed: None
            },
        ]
    );
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(
            read(&store, 3),
            Some(30),
            "frames after the torn tail replay"
        );
    }
    // the torn bytes are still on disk and still ignored
    assert_eq!(fs::read(&wal).unwrap().len(), torn);

    // damage inside a sealed segment is corruption, not a torn tail
    let mut bytes = fs::read(&wal).unwrap();
    bytes[first - 1] ^= 0xff;
    fs::write(&wal, &bytes).unwrap();
    let err = open(&dir, DurabilityPolicy::RELAXED).err().unwrap();
    assert!(matches!(err, Error::Corrupt { offset: 0, .. }), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn keyspaces_must_be_registered_with_their_written_types() {
    let dir = tmpdir("types");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
    }
    let other = StoreBuilder::new(DurabilityPolicy::RELAXED)
        .dir(&dir)
        .keyspace(KeyspaceId::<u8, (i64, i64)>::new(3))
        .build();
    assert!(matches!(other, Err(Error::KeyspaceType { id: 3, .. })));

    let missing = StoreBuilder::new(DurabilityPolicy::RELAXED)
        .dir(&dir)
        .build();
    assert!(matches!(missing, Err(Error::UnknownKeyspace(3))));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_directory_opens_once_at_a_time() {
    let dir = tmpdir("lock");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert!(matches!(
        open(&dir, DurabilityPolicy::RELAXED),
        Err(Error::Locked(_))
    ));
    drop(store);
    assert!(open(&dir, DurabilityPolicy::RELAXED).is_ok());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn non_writing_policy_still_replays_but_logs_nothing() {
    let dir = tmpdir("none");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
    }
    {
        let store = open(&dir, DurabilityPolicy::NONE).unwrap();
        assert_eq!(read(&store, 1), Some(10));
        assert_eq!(put(&store, &[(1, Some(11))]), Some(2));
    }
    assert_eq!(manifest_of(&dir).policy, DurabilityPolicy::NONE);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(read(&store, 1), Some(10), "unlogged commit is gone");
    assert_eq!(put(&store, &[(1, Some(12))]), Some(2));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

/// The rows of keyspace `KS` in a checkpoint's table.
fn checkpoint_rows(path: &Path) -> Vec<(u8, Option<i64>)> {
    let table = Table::open(path, 0).unwrap();
    let mut c = table.cursor(KS.raw(), None);
    c.start().unwrap();
    let mut rows = Vec::new();
    while let Some((k, v)) = c.current() {
        rows.push((
            crate::store::codec::decode_all_ordered::<u8>(k).unwrap(),
            decode_all::<Option<i64>>(v).unwrap(),
        ));
        c.advance().unwrap();
    }
    rows
}

fn files_in(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect()
}

#[test]
fn checkpoint_retires_the_wal_and_reopen_loads_it() {
    let dir = tmpdir("ckpt");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        put(&store, &[(1, Some(11))]);
        assert_eq!(store.checkpoint().unwrap(), Some(2));
        assert_eq!(store.checkpoint().unwrap(), None, "nothing new");
        // after the checkpoint: one overwrite, one delete of checkpointed data
        put(&store, &[(2, None), (3, Some(30))]);
    }
    let m = manifest_of(&dir);
    assert_eq!(
        m.checkpoints,
        [Checkpoint {
            number: 2,
            epoch: 2
        }]
    );
    assert_eq!(
        m.segments,
        [Segment {
            number: 3,
            sealed: None
        }]
    );
    assert_eq!(
        files_in(&dir),
        BTreeSet::from(["LOCK", "MANIFEST", "000002.ckpt", "000003.wal"].map(String::from)),
        "the covered segment is deleted"
    );
    assert_eq!(
        checkpoint_rows(&dir.join("000002.ckpt")),
        [(1, Some(11)), (2, Some(20))],
        "newest value per key"
    );

    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(store.snapshot().epoch(), 3);
    let v = store.current.load();
    assert_eq!(
        (v.layers.len(), v.tables.len()),
        (1, 1),
        "WAL in memory, checkpoint on disk"
    );
    assert_eq!(read(&store, 1), Some(11));
    assert_eq!(
        read(&store, 2),
        None,
        "WAL tombstone shadows the checkpoint"
    );
    assert_eq!(read(&store, 3), Some(30));
    assert_eq!(put(&store, &[(4, Some(40))]), Some(4));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn checkpoints_are_deltas_and_load_in_order() {
    let dir = tmpdir("deltas");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10)), (2, Some(20))]);
        assert_eq!(store.checkpoint().unwrap(), Some(1));
        put(&store, &[(2, None)]);
        put(&store, &[(3, Some(30))]);
        assert_eq!(store.checkpoint().unwrap(), Some(3));
    }
    let m = manifest_of(&dir);
    assert_eq!(
        m.checkpoints,
        [
            Checkpoint {
                number: 2,
                epoch: 1
            },
            Checkpoint {
                number: 4,
                epoch: 3
            },
        ]
    );
    assert_eq!(
        checkpoint_rows(&dir.join("000004.ckpt")),
        [(2, None), (3, Some(30))],
        "only what changed since the first, tombstones included"
    );

    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(store.snapshot().epoch(), 3);
    let v = store.current.load();
    assert_eq!(
        (v.layers.len(), v.tables.len()),
        (0, 2),
        "no WAL frames left"
    );
    assert_eq!(read(&store, 1), Some(10));
    assert_eq!(read(&store, 2), None, "later checkpoint's tombstone wins");
    assert_eq!(read(&store, 3), Some(30));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn checkpoint_persists_commits_a_non_writing_policy_never_logged() {
    let dir = tmpdir("ckpt-none");
    {
        let store = open(&dir, DurabilityPolicy::NONE).unwrap();
        put(&store, &[(1, Some(10))]);
        assert_eq!(store.checkpoint().unwrap(), Some(1));
        put(&store, &[(2, Some(20))]);
    }
    assert!(manifest_of(&dir).segments.is_empty());
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(read(&store, 1), Some(10));
    assert_eq!(read(&store, 2), None, "never logged, never checkpointed");
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn in_memory_checkpoint_is_a_no_op() {
    let store = mem();
    put(&store, &[(1, Some(10))]);
    assert_eq!(store.checkpoint().unwrap(), None);
}

#[test]
fn open_sweeps_orphans_but_not_foreign_files() {
    let dir = tmpdir("orphans");
    drop(open(&dir, DurabilityPolicy::RELAXED).unwrap());
    for junk in [
        "000099.wal",
        "000098.ckpt",
        "MANIFEST.tmp",
        "notes.txt",
        "x.wal",
    ] {
        fs::write(dir.join(junk), b"junk").unwrap();
    }
    drop(open(&dir, DurabilityPolicy::RELAXED).unwrap());
    assert_eq!(
        files_in(&dir),
        BTreeSet::from(["LOCK", "MANIFEST", "000001.wal", "notes.txt", "x.wal"].map(String::from))
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_damaged_checkpoint_is_corruption() {
    let dir = tmpdir("ckpt-bad");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
        store.checkpoint().unwrap();
    }
    let path = dir.join("000002.ckpt");
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    let err = open(&dir, DurabilityPolicy::RELAXED).err().unwrap();
    assert!(matches!(err, Error::Corrupt { .. }), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn checkpoint_moves_data_out_of_memory_and_reads_page_it_in() {
    let dir = tmpdir("evict");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    put(&store, &[(1, Some(10)), (2, Some(20))]);
    put(&store, &[(2, None)]);
    store.checkpoint().unwrap();

    let v = store.current.load_full();
    assert!(v.layers.is_empty(), "covered layers are dropped");
    assert_eq!(store.cache.stats(), (0, 0), "nothing read yet");
    assert_eq!(read(&store, 1), Some(10));
    assert_eq!(read(&store, 2), None, "tombstone read from disk");
    assert_eq!(read(&store, 3), None);
    assert_eq!(store.cache.stats().1, 1, "one block, read once");

    // reads in a write tx fall through to the table too
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    let r = ks.resolve(&1, Change::Modify(5)).unwrap().unwrap();
    assert!(matches!(r, Change::Replace(10, 15)));
    ks.absorb(1, r).unwrap();
    store.finish(tx).unwrap();
    assert_eq!(read(&store, 1), Some(15), "memory layer shadows the table");
    drop((store, v));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn snapshots_keep_their_view_across_a_checkpoint() {
    let dir = tmpdir("ckpt-snap");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    put(&store, &[(1, Some(10))]);
    let first = store.snapshot();
    put(&store, &[(1, Some(11)), (2, Some(20))]);
    let second = store.snapshot();
    store.checkpoint().unwrap();
    put(&store, &[(2, None)]);

    assert_eq!(first.get(KS, &1).unwrap().as_deref(), Some(&10));
    assert_eq!(first.get(KS, &2).unwrap().as_deref(), None);
    assert_eq!(second.get(KS, &1).unwrap().as_deref(), Some(&11));
    assert_eq!(second.get(KS, &2).unwrap().as_deref(), Some(&20));
    assert_eq!(read(&store, 1), Some(11));
    assert_eq!(read(&store, 2), None);
    drop((first, second));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn large_checkpoints_read_through_a_small_cache() {
    const BIG: KeyspaceId<u32, i64> = KeyspaceId::new(8);
    let dir = tmpdir("big");
    let build = || {
        StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .keyspace(KS)
            .keyspace(BIG)
            .cache_size(16 << 10)
            .build()
            .unwrap()
    };
    {
        let store = build();
        for chunk in (0..20_000u32).collect::<Vec<_>>().chunks(5_000) {
            let mut tx = store.begin().unwrap();
            let mut ks = tx.keyspace(BIG);
            for &i in chunk {
                ks.absorb(i, Change::Insert(i as i64 * 3)).unwrap();
            }
            store.finish(tx).unwrap();
        }
        put(&store, &[(1, Some(10))]);
        store.checkpoint().unwrap();
    }
    let store = build();
    let snap = store.snapshot();
    for i in (0..20_000u32).step_by(7) {
        assert_eq!(snap.get(BIG, &i).unwrap().as_deref(), Some(&(i as i64 * 3)));
    }
    assert_eq!(snap.get(BIG, &20_000).unwrap().as_deref(), None);
    assert_eq!(snap.get(KS, &1).unwrap().as_deref(), Some(&10));
    let (cached, loads) = store.cache.stats();
    assert!(cached <= 16 << 10, "cache over budget: {cached}");
    assert!(loads > 4, "blocks were paged in and evicted: {loads}");
    drop(snap);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

fn open_with(
    dir: &Path,
    dp: DurabilityPolicy,
    f: impl FnOnce(StoreBuilder) -> StoreBuilder,
) -> Result<Store> {
    f(StoreBuilder::new(dp).dir(dir).keyspace(KS)).build()
}

#[test]
fn a_tx_that_outgrows_its_limit_is_killed_and_commits_nothing() {
    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(KS)
        .tx_memory_limit(1024)
        .build()
        .unwrap();
    put(&store, &[(0, Some(1))]);

    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    let mut killed = None;
    for k in 1..=255u8 {
        if let Err(oom) = ks.absorb(k, Change::Insert(k as i64)) {
            killed = Some((k, oom));
            break;
        }
    }
    let (at, err) = killed.expect("1 KiB cannot hold 255 entries");
    assert!(at > 1, "a few entries fit");
    let Error::OutOfMemory(oom) = err else {
        panic!("{err}")
    };
    assert!(oom.used > 1024 && oom.limit == 1024, "{oom}");
    let is_oom = |r: Result<()>| matches!(r, Err(Error::OutOfMemory(e)) if e == oom);
    assert!(is_oom(ks.absorb(250, Change::Insert(1))), "stays dead");
    assert!(ks.get(&1).is_err(), "reads fail too");
    drop(ks);
    assert!(matches!(tx.failure(), Some(Error::OutOfMemory(e)) if e == oom));
    assert_eq!(tx.memory_used(), 0);
    assert!(matches!(store.finish(tx), Err(Error::OutOfMemory(e)) if e == oom));

    assert_eq!(store.snapshot().epoch(), 1, "no epoch spent");
    assert_eq!(read(&store, 1), None);
    assert_eq!(
        put(&store, &[(1, Some(1))]),
        Some(2),
        "the writer is free again"
    );
}

#[test]
fn tx_memory_tracks_entries_that_change_and_cancel() {
    let store = mem();
    let mut tx = store.begin().unwrap();
    let settle = |tx: &mut WriteTx<'_>| {
        tx.end_run();
        drop(tx.keyspace(KS));
        tx.memory_used()
    };
    let mut ks = tx.keyspace(KS);
    ks.absorb(1, Change::Insert(10)).unwrap();
    ks.absorb(2, Change::Insert(20)).unwrap();
    drop(ks);
    let two = settle(&mut tx);
    assert!(two > 0);
    let one = two / 2;
    // a later run's change to a key is one more entry: each run keeps
    // its own, so the run's old and new values stay readable
    let mut ks = tx.keyspace(KS);
    ks.absorb(2, Change::Delete(20)).unwrap();
    drop(ks);
    assert_eq!(settle(&mut tx), one * 3);
    // within a run, insert then delete cancels: the entry is gone,
    // and so are its bytes
    let mut ks = tx.keyspace(KS);
    ks.absorb(5, Change::Insert(50)).unwrap();
    ks.absorb(5, Change::Delete(50)).unwrap();
    drop(ks);
    assert_eq!(settle(&mut tx), one * 3);
    // and so does an out-of-order write, once settled
    let mut ks = tx.keyspace(KS);
    ks.absorb(7, Change::Insert(70)).unwrap();
    ks.absorb(6, Change::Insert(60)).unwrap();
    ks.absorb(7, Change::Delete(70)).unwrap();
    assert_eq!(ks.get(&7).unwrap().as_deref(), None);
    assert_eq!(ks.get(&6).unwrap().as_deref(), Some(&60));
    drop(ks);
    assert_eq!(settle(&mut tx), one * 4);
}

#[test]
fn crossing_the_wal_limit_checkpoints_automatically() {
    let dir = tmpdir("auto-wal");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(200)).unwrap();
    let n = 20u8;
    for i in 0..n {
        put(&store, &[(i, Some(i as i64))]).unwrap();
        assert!(
            store.stats().wal_bytes < 400,
            "commits wait at twice the limit"
        );
    }
    store.quiesce().unwrap();
    let stats = store.stats();
    assert!(stats.tables >= 1, "the limit was crossed");
    assert!(stats.wal_bytes < 200, "{stats:?}");
    let m = manifest_of(&dir);
    assert!(m.checkpointed() > 0);
    assert!(m.segments.len() == 1, "covered segments are retired");
    for i in 0..n {
        assert_eq!(read(&store, i), Some(i as i64));
    }
    drop(store);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(read(&store, n - 1), Some(n as i64 - 1));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn crossing_the_memtable_limit_checkpoints_even_without_a_wal() {
    let dir = tmpdir("auto-mem");
    let store = open_with(&dir, DurabilityPolicy::NONE, |b| b.memtable_limit(512)).unwrap();
    for i in 0..50u8 {
        put(&store, &[(i, Some(i as i64))]);
        assert!(
            store.stats().memtable_bytes < 1024,
            "commits wait at twice the limit"
        );
    }
    store.quiesce().unwrap();
    let stats = store.stats();
    assert!(stats.memtable_bytes < 512);
    assert!(stats.tables >= 1, "{stats:?}");
    drop(store);
    let store = open(&dir, DurabilityPolicy::NONE).unwrap();
    let unlogged = (0..50u8).filter(|&i| read(&store, i).is_none()).count();
    assert!(unlogged < 50, "checkpointed data survives");
    assert_eq!(
        unlogged > 0,
        stats.memtable_bytes > 0,
        "the tail does not: {unlogged}"
    );
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn open_checkpoints_when_the_last_run_left_too_much() {
    let dir = tmpdir("auto-open");
    {
        let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
        put(&store, &[(1, Some(10))]);
        put(&store, &[(2, Some(20))]);
        assert_eq!(store.stats().tables, 0);
    }
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(10)).unwrap();
    store.quiesce().unwrap();
    let stats = store.stats();
    assert_eq!(
        (stats.tables, stats.wal_bytes, stats.memtable_bytes),
        (1, 0, 0)
    );
    assert_eq!(read(&store, 2), Some(20));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wal_bytes_match_the_files_and_survive_reopen() {
    let dir = tmpdir("wal-bytes");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    put(&store, &[(1, Some(10)), (2, Some(20))]);
    put(&store, &[(3, Some(30))]);
    let len = fs::metadata(dir.join("000001.wal")).unwrap().len();
    assert_eq!(store.stats().wal_bytes, len);
    drop(store);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(store.stats().wal_bytes, len);
    assert!(
        store.stats().memtable_bytes > 0,
        "replayed data is accounted"
    );
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_failed_auto_checkpoint_is_reported_to_a_waiting_commit_and_retried() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmpdir("auto-fail");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.wal_limit(100)).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
    if fs::write(dir.join("probe"), b"").is_ok() {
        // permissions do not bind (running as root): nothing to test
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
        return;
    }

    let mut failed = None;
    for i in 0..20u8 {
        let mut tx = store.begin().unwrap();
        tx.keyspace(KS).absorb(i, Change::Insert(i as i64)).unwrap();
        match store.finish(tx) {
            Ok(_) => {}
            Err(Error::AfterCommit { epoch, .. }) => {
                failed = Some((i, epoch));
                break;
            }
            Err(e) => panic!("{e}"),
        }
    }
    let (i, epoch) = failed.expect("the limit was crossed");
    assert_eq!(read(&store, i), Some(i as i64), "the commit still happened");
    assert_eq!(store.snapshot().epoch(), epoch);
    assert_eq!(store.stats().tables, 0);

    assert!(store.quiesce().is_err(), "still failing");

    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    put(&store, &[(99, Some(99))]);
    store.quiesce().unwrap();
    assert_eq!(store.stats().tables, 1, "the next pass retried it");
    assert_eq!(read(&store, 99), Some(99));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn options_are_validated_and_recorded() {
    let dir = tmpdir("options");
    let bad = open_with(&dir, DurabilityPolicy::RELAXED, |b| b.block_size(0));
    assert!(matches!(bad, Err(Error::InvalidOption(_))));
    drop(
        open_with(&dir, DurabilityPolicy::RELAXED, |b| {
            b.block_size(512).cache_size(1 << 20)
        })
        .unwrap(),
    );
    let m = manifest_of(&dir);
    assert_eq!(m.options["block-size"], 512);
    assert_eq!(m.options["cache-size"], 1 << 20);
    assert_eq!(m.options["wal-limit"], 64 << 20);
    fs::remove_dir_all(&dir).unwrap();
}

fn live_rows(store: &Store) -> Vec<(u8, i64)> {
    store
        .snapshot()
        .range(KS, ..)
        .map(|r| r.map(|(k, v)| (*k, *v)))
        .collect::<Result<_>>()
        .unwrap()
}

/// Keys 1..=6 spread over two tables and a memory layer, with
/// overwrites and deletes at every level.
fn layered(store: &Store) {
    put(
        store,
        &[(1, Some(1)), (2, Some(2)), (3, Some(3)), (4, Some(4))],
    );
    store.checkpoint().unwrap();
    put(store, &[(2, Some(20)), (3, None), (5, Some(5))]);
    store.checkpoint().unwrap();
    put(store, &[(4, Some(40)), (5, None), (6, Some(6))]);
}

#[test]
fn small_commits_fold_into_logarithmically_many_layers() {
    let store = mem();
    let mut model = BTreeMap::new();
    for i in 0..2000u32 {
        let k = (i % 251) as u8;
        let v = (i % 7 != 0).then_some(i as i64);
        put(&store, &[(k, v)]);
        match v {
            Some(v) => model.insert(k, v),
            None => model.remove(&k),
        };
        let layers = store.current.load().layers.len();
        assert!(layers <= 32, "{layers} layers after {i} commits");
    }
    store.quiesce().unwrap();
    let layers = store.current.load().layers.len();
    assert!(layers <= 12, "{layers} layers once maintenance catches up");
    for k in 0..=255u8 {
        assert_eq!(read(&store, k), model.get(&k).copied(), "key {k}");
    }
    let live: Vec<(u8, i64)> = model.into_iter().collect();
    assert_eq!(live_rows(&store), live);
}

#[test]
fn maintenance_races_commits_checkpoints_and_readers() {
    let dir = tmpdir("maintenance-race");
    let opts = |b: StoreBuilder| {
        b.memtable_limit(4096).options_with(|o| {
            o.max_tables = 3;
            o.compaction_width = 2;
            o.max_layers = 4;
        })
    };
    let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
    let mut model = BTreeMap::new();
    let done = AtomicBool::new(false);
    // stops the readers even if the writer below panics
    struct Done<'a>(&'a AtomicBool);
    impl Drop for Done<'_> {
        fn drop(&mut self) {
            self.0.store(true, Relaxed);
        }
    }
    std::thread::scope(|scope| {
        let _done = Done(&done);
        let readers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    // every key holds its epoch's commit number: within
                    // one snapshot, each value is at most the epoch
                    while !done.load(Relaxed) {
                        let snap = store.snapshot();
                        for r in snap.range(KS, ..) {
                            let (_, v) = r.unwrap();
                            assert!(*v as u64 <= snap.epoch(), "{v} past {}", snap.epoch());
                        }
                    }
                })
            })
            .collect();
        for i in 1..=3000u64 {
            let k = (i * 37 % 200) as u8;
            let v = (i % 5 != 0).then_some(i as i64);
            // key 255 changes every commit, so each one spends an epoch
            assert_eq!(put(&store, &[(k, v), (255, Some(i as i64))]), Some(i));
            for (k, v) in [(k, v), (255, Some(i as i64))] {
                match v {
                    Some(v) => model.insert(k, v),
                    None => model.remove(&k),
                };
            }
            if i % 500 == 0 {
                store.checkpoint().unwrap();
            }
        }
        done.store(true, Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    });
    store.quiesce().unwrap();
    let v = store.current.load_full();
    assert!(v.tables.len() <= 3, "{} tables", v.tables.len());
    assert!(v.layers.len() <= 4, "{} layers", v.layers.len());
    let live: Vec<(u8, i64)> = model.into_iter().collect();
    assert_eq!(live_rows(&store), live);
    drop(v);
    drop(store);
    let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
    assert_eq!(live_rows(&store), live);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn folding_without_tables_applies_deletes() {
    let store = mem();
    for k in 0..100u8 {
        put(&store, &[(k, Some(k as i64))]);
    }
    for k in 0..100u8 {
        put(&store, &[(k, None)]);
    }
    store.quiesce().unwrap();
    let v = store.current.load();
    let tombstones: usize = v
        .layers
        .iter()
        .flat_map(|l| l.trees.values())
        .map(|t| {
            t.tree
                .downcast_ref::<Resident<u8, i64>>()
                .unwrap()
                .values()
                .filter(|v| v.is_none())
                .count()
        })
        .sum();
    // only the newest few layers, not yet folded to the bottom, keep any
    assert!(tombstones < 20, "{tombstones} tombstones left");
    assert!(
        v.resident_bytes < 20 * 64,
        "{} bytes left",
        v.resident_bytes
    );
    assert_eq!(
        v.resident_bytes,
        v.layers.iter().map(|l| l.bytes).sum::<usize>()
    );
}

#[test]
fn folding_above_tables_keeps_tombstones() {
    let dir = tmpdir("fold-over-tables");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    put(&store, &[(1, Some(1)), (2, Some(2))]);
    store.checkpoint().unwrap();
    put(&store, &[(1, None)]);
    for i in 0..50 {
        put(&store, &[(3, Some(i))]);
    }
    assert_eq!(read(&store, 1), None, "the folded tombstone still shadows");
    assert_eq!(live_rows(&store), vec![(2, 2), (3, 49)]);
    drop(store);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(live_rows(&store), vec![(2, 2), (3, 49)]);
}

#[test]
fn snapshots_keep_their_view_across_folds() {
    let store = mem();
    put(&store, &[(1, Some(1)), (2, Some(2))]);
    let before = store.snapshot();
    for i in 0..100 {
        put(&store, &[(1, Some(100 + i)), (2, None)]);
    }
    assert_eq!(before.get(KS, &1).unwrap().as_deref(), Some(&1));
    assert_eq!(before.get(KS, &2).unwrap().as_deref(), Some(&2));
    assert_eq!(read(&store, 1), Some(199));
    assert_eq!(read(&store, 2), None);
}

#[test]
fn range_merges_memory_and_tables_newest_first() {
    let dir = tmpdir("range");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    layered(&store);
    let v = store.current.load();
    assert_eq!((v.layers.len(), v.tables.len()), (1, 2));
    drop(v);

    assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
    let snap = store.snapshot();
    let keys = |c: Cursor<'_, u8, i64>| -> Vec<u8> { c.map(|r| *r.unwrap().0).collect() };
    assert_eq!(keys(snap.range(KS, 2..=4)), [2, 4]);
    assert_eq!(keys(snap.range(KS, 2..4)), [2]);
    assert_eq!(
        keys(snap.range(KS, (Bound::Excluded(2), Bound::Unbounded))),
        [4, 6]
    );
    assert_eq!(keys(snap.range(KS, 7..)), Vec::<u8>::new());

    // a write tx's cursor sees its own writes on top
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    ks.absorb(1, Change::Delete(1)).unwrap();
    ks.absorb(3, Change::Insert(30)).unwrap();
    let rows: Vec<(u8, i64)> = ks
        .range(..)
        .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
        .collect();
    assert_eq!(rows, [(2, 20), (3, 30), (4, 40), (6, 6)]);
    drop(ks);
    drop(tx);
    drop(snap);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn batch_resolve_reads_each_block_at_most_once() {
    const BIG: KeyspaceId<u32, i64> = KeyspaceId::new(8);
    let dir = tmpdir("resolve-pass");
    let store = StoreBuilder::new(DurabilityPolicy::RELAXED)
        .dir(&dir)
        .keyspace(BIG)
        .cache_size(DEFAULT_BLOCK_SIZE)
        .build()
        .unwrap();
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(BIG);
    for i in 0..20_000u32 {
        ks.absorb(i, Change::Insert(i as i64)).unwrap();
    }
    store.finish(tx).unwrap();
    store.checkpoint().unwrap();
    let blocks = store.current.load().tables[0].blocks(BIG.raw());
    let loads_before = store.cache.stats().1;

    let mut changes = Register::new();
    for i in (0..20_000u32).step_by(7) {
        changes.absorb_one(i, Change::Modify(1));
    }
    changes.absorb_one(30_000, Change::Modify(1));
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(BIG);
    let mut out = Vec::new();
    ks.resolve_sorted(changes, &mut out).unwrap();
    let loads = store.cache.stats().1 - loads_before;
    assert!(loads <= blocks as u64, "{loads} loads for {blocks} blocks");
    assert!(matches!(out[1], (7, Change::Replace(7, 8))));
    assert!(matches!(out.last(), Some((30_000, Change::Insert(1)))));
    drop(ks);
    drop(tx);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn too_many_tables_are_compacted_after_commits() {
    let dir = tmpdir("auto-compact");
    let opts = |b: StoreBuilder| {
        b.memtable_limit(300).options_with(|o| {
            o.max_tables = 3;
            o.compaction_width = 2;
        })
    };
    {
        let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
        for round in 0..6u8 {
            for i in 0..10u8 {
                put(&store, &[(i, Some(round as i64 * 100 + i as i64))]);
                // commits wait for compaction past twice the limit
                assert!(store.stats().tables <= 6, "{:?}", store.stats());
            }
        }
        store.quiesce().unwrap();
        assert!(store.stats().tables <= 3, "{:?}", store.stats());
        put(&store, &[(3, None)]);
        assert_eq!(live_rows(&store).len(), 9);
        assert_eq!(read(&store, 4), Some(504));
    }
    let m = manifest_of(&dir);
    let on_disk = files_in(&dir)
        .into_iter()
        .filter(|f| f.ends_with(".ckpt"))
        .count();
    assert_eq!(on_disk, m.checkpoints.len(), "merged inputs are deleted");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, opts).unwrap();
    assert_eq!(read(&store, 3), None);
    assert_eq!(read(&store, 9), Some(509));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_failed_compaction_is_reported_and_retried() {
    let dir = tmpdir("compact-fail");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| {
        b.options_with(|o| {
            o.max_tables = 2;
            o.compaction_width = 2;
        })
    })
    .unwrap();
    // no tables yet, so nothing can have been merged before this
    store.fail_merges.store(true, Relaxed);
    for i in 0..5u8 {
        put(&store, &[(i, Some(i as i64))]);
        store.checkpoint().unwrap();
    }
    let err = store.quiesce().unwrap_err();
    assert!(err.to_string().contains("injected"), "{err}");
    assert!(store.stats().tables > 2, "nothing was merged");
    // commits that wait on it hear about it too, but still commit
    let mut tx = store.begin().unwrap();
    tx.keyspace(KS).absorb(9, Change::Insert(9)).unwrap();
    store.finish(tx).ok();
    assert_eq!(read(&store, 9), Some(9));

    store.fail_merges.store(false, Relaxed);
    store.quiesce().unwrap();
    assert!(store.stats().tables <= 2, "{:?}", store.stats());
    let on_disk = files_in(&dir)
        .into_iter()
        .filter(|f| f.ends_with(".ckpt"))
        .count();
    assert_eq!(on_disk, store.stats().tables, "failed outputs are swept");
    drop(store);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(live_rows(&store).len(), 6);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[should_panic(expected = "holds a write tx")]
fn checkpointing_under_ones_own_tx_panics_instead_of_deadlocking() {
    let store = mem();
    let _tx = store.begin().unwrap();
    let _ = store.checkpoint();
}

#[test]
fn a_tx_on_another_thread_is_waited_for() {
    let store = mem();
    let tx = store.begin().unwrap();
    thread::scope(|s| {
        let waiter = s.spawn(|| store.checkpoint().unwrap());
        store.finish(tx).unwrap();
        assert_eq!(waiter.join().unwrap(), None, "in memory");
    });
    // the tx is gone, so this thread may checkpoint again
    store.checkpoint().unwrap();
}

#[test]
fn the_maintenance_thread_starts_when_first_needed() {
    let store = mem();
    put(&store, &[(1, Some(1))]);
    put(&store, &[(2, Some(2))]);
    assert!(
        store.worker.lock().unwrap().is_none(),
        "commits nobody can see fold their own layers"
    );
    // a snapshot keeps the layers apart, so folding them is left to
    // the thread
    let snap = store.snapshot();
    put(&store, &[(3, Some(3))]);
    assert!(store.worker.lock().unwrap().is_some());
    drop(snap);
    store.quiesce().unwrap();
    assert_eq!(live_rows(&store).len(), 3);
}

/// The live keys of a run: how many, and the least and greatest.
#[derive(Default, Debug, PartialEq)]
struct Zone {
    live: u32,
    min: Option<u8>,
    max: Option<u8>,
}

static ZONE_DECODES: AtomicU64 = AtomicU64::new(0);

impl RunIndex<u8, i64> for Zone {
    const NAME: &'static str = "zone";
    fn push(&mut self, key: &u8, value: Option<&i64>) {
        if value.is_some() {
            self.live += 1;
            self.min = Some(self.min.map_or(*key, |m| m.min(*key)));
            self.max = Some(*key);
        }
    }
    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
    }
    fn encode(&self, out: &mut Vec<u8>) -> bool {
        out.extend(self.live.to_le_bytes());
        out.extend([self.min.unwrap_or(0), self.max.unwrap_or(0)]);
        true
    }
    fn decode(b: &[u8]) -> Option<Self> {
        ZONE_DECODES.fetch_add(1, Relaxed);
        let live = u32::from_le_bytes(b.get(..4)?.try_into().ok()?);
        let (min, max) = (*b.get(4)?, *b.get(5)?);
        let some = |k| (live > 0).then_some(k);
        Some(Zone {
            live,
            min: some(min),
            max: some(max),
        })
    }
}

/// Every row of a run, tombstones included; declines empty runs and
/// never persists.
#[derive(Default)]
struct Rows(u32);

impl RunIndex<u8, i64> for Rows {
    const NAME: &'static str = "rows";
    fn push(&mut self, _: &u8, _: Option<&i64>) {
        self.0 += 1;
    }
    fn finish(&mut self) -> bool {
        self.0 > 0
    }
    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

fn open_indexed(dir: &Path) -> Store {
    StoreBuilder::new(DurabilityPolicy::RELAXED)
        .dir(dir)
        .keyspace(KS)
        .indexed::<_, _, Zone>(KS)
        .indexed::<_, _, Rows>(KS)
        .options_with(|o| {
            o.max_tables = 3;
            o.compaction_width = 2;
        })
        .build()
        .unwrap()
}

/// Every run's indexes describe exactly that run.
fn check_runs(store: &Store) -> (usize, usize) {
    let snap = store.snapshot();
    let zones = snap.runs_indexed::<_, _, Zone>(KS).unwrap();
    let rows = snap.runs_indexed::<_, _, Rows>(KS).unwrap();
    assert_eq!(zones.len(), rows.len());
    let mut tables = 0;
    for (z, r) in zones.iter().zip(&rows) {
        let live: Vec<u8> = z.range(..).map(|e| *e.unwrap().0).collect();
        let want = Zone {
            live: live.len() as u32,
            min: live.first().copied(),
            max: live.last().copied(),
        };
        assert_eq!(z.index(), Some(&want), "{z:?}");
        let all = (0..=255u8).filter(|k| r.get(k).unwrap().is_some()).count();
        assert_eq!(r.index().map(|r| r.0 as usize), Some(all));
        tables += matches!(z.run, Run::Table { .. }) as usize;
    }
    (zones.len(), tables)
}

#[test]
fn run_indexes_follow_runs_through_folds_checkpoints_and_compaction() {
    let dir = tmpdir("run-index");
    let store = open_indexed(&dir);
    let mut model = BTreeMap::new();
    for round in 0..8u8 {
        for i in 0..12u8 {
            let k = i * 7 % 40 + round;
            let v = (i % 4 != 3).then_some(round as i64);
            put(&store, &[(k, v)]);
            match v {
                Some(v) => model.insert(k, v),
                None => model.remove(&k),
            };
            check_runs(&store);
        }
        if round % 2 == 1 {
            store.checkpoint().unwrap();
        }
        store.quiesce().unwrap();
        check_runs(&store);
    }
    put(&store, &[(200, Some(1))]);
    model.insert(200, 1);
    let (runs, tables) = check_runs(&store);
    assert!(tables > 1 && runs > tables, "{runs} runs, {tables} tables");
    assert!(store.stats().run_index_bytes > 0);
    let live: BTreeMap<u8, i64> = live_rows(&store).into_iter().collect();
    assert_eq!(live, model);
    drop(store);

    // zones are decoded from the tables; row counts are rebuilt
    let store = open_indexed(&dir);
    assert_eq!(store.stats().run_index_bytes, 0, "loaded on first use");
    let before = ZONE_DECODES.load(Relaxed);
    let (_, tables) = check_runs(&store);
    assert!(ZONE_DECODES.load(Relaxed) - before >= tables as u64);
    assert!(store.stats().run_index_bytes > 0);
    drop(store);

    // a store without the indexes ignores their frames
    let plain = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(live_rows(&plain).len(), model.len());
    drop(plain);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[should_panic(expected = "not registered")]
fn an_unregistered_index_is_a_mistake() {
    let store = mem();
    let _ = store.snapshot().runs_indexed::<_, _, Zone>(KS);
}

#[test]
fn reverse_ranges_mirror_forward_ones_across_every_source() {
    let dir = tmpdir("reverse");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| {
        b.block_size(64).options_with(|o| o.max_tables = 100)
    })
    .unwrap();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut rng = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let bound = |r: u64, k: u8| match r {
        0 => Bound::Unbounded,
        1 => Bound::Included(k),
        _ => Bound::Excluded(k),
    };
    let collect = |c: Cursor<'_, u8, i64>| -> Vec<(u8, i64)> {
        c.map(|r| r.map(|(k, v)| (*k, *v)).unwrap()).collect()
    };
    for round in 0..40 {
        let rows: Vec<(u8, Option<i64>)> = (0..1 + rng(12))
            .map(|_| (rng(60) as u8, (rng(4) != 0).then(|| rng(1000) as i64)))
            .collect();
        put(&store, &rows);
        if round % 7 == 3 {
            store.checkpoint().unwrap();
        }
        let snap = store.snapshot();
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(KS);
        for _ in 0..rng(6) {
            let k = rng(60) as u8;
            let v = (rng(3) != 0).then(|| rng(1000) as i64);
            ks.set(k, v).unwrap();
        }
        tx.end_run();
        let mut ks = tx.keyspace(KS);
        for _ in 0..rng(6) {
            let k = rng(60) as u8;
            let v = (rng(3) != 0).then(|| rng(1000) as i64);
            ks.set(k, v).unwrap();
        }
        for _ in 0..10 {
            let (a, b) = (rng(64) as u8, rng(64) as u8);
            let range = (bound(rng(3), a.min(b)), bound(rng(3), a.max(b)));
            let mut fwd = collect(snap.range(KS, range));
            fwd.reverse();
            assert_eq!(
                collect(snap.range_rev(KS, range)),
                fwd,
                "snapshot {range:?}"
            );
            let mut fwd = collect(ks.range(range));
            fwd.reverse();
            assert_eq!(collect(ks.range_rev(range)), fwd, "tx {range:?}");
            let mut fwd = collect(ks.range_old(range));
            fwd.reverse();
            assert_eq!(collect(ks.range_old_rev(range)), fwd, "old {range:?}");
        }
        // seeks follow the cursor's direction
        let want: Vec<Option<i64>> = (0..64u8).map(|k| ks.get(&k).unwrap().map(|v| *v)).collect();
        let mut down = ks.range_rev(..);
        for k in (0..64u8).rev() {
            let got = down.seek_get(&k).unwrap().map(|v| *v);
            assert_eq!(got, want[k as usize], "rev seek {k}");
        }
        drop(down);
        let mut up = ks.range(..);
        for k in 0..64u8 {
            let got = up.seek_get(&k).unwrap().map(|v| *v);
            assert_eq!(got, want[k as usize], "seek {k}");
        }
        drop(up);
        drop(ks);
        drop(tx);
    }
    assert!(store.stats().tables > 3);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn layers_no_snapshot_can_see_fold_in_place() {
    let store = mem();
    let base_tree = |store: &Store| {
        let v = store.current.load();
        assert_eq!(v.layers.len(), 1, "folded into one");
        let t = &v.layers[0].trees[&KS.raw()];
        let measured = (store.keyspaces[&KS.raw()].measure)(&*t.tree);
        assert_eq!(t.bytes, measured, "bytes stay exact");
        Arc::as_ptr(&t.tree) as *const () as usize
    };
    for i in 0..50u8 {
        put(&store, &[(i, Some(i as i64))]);
    }
    store.quiesce().unwrap();
    let tree = base_tree(&store);
    put(&store, &[(3, None), (60, Some(60))]);
    store.quiesce().unwrap();
    assert_eq!(base_tree(&store), tree, "the same tree, changed in place");
    assert_eq!(live_rows(&store).len(), 50);
    assert_eq!(read(&store, 3), None);

    // a snapshot can see the layers, so they fold by copying, and it
    // keeps its view
    let snap = store.snapshot();
    for i in 70..80u8 {
        put(&store, &[(i, Some(1))]);
    }
    store.quiesce().unwrap();
    assert_eq!(snap.range(KS, ..).count(), 50);
    assert_eq!(snap.get(KS, &3).unwrap(), None);
    drop(snap);
    put(&store, &[(90, Some(1))]);
    store.quiesce().unwrap();
    base_tree(&store);
    assert_eq!(live_rows(&store).len(), 61);
}

const TALLY: KeyspaceId<u8, Tally> = KeyspaceId::new(4);

/// A tally keyspace's amounts are summed wherever they meet: within a
/// tx, across its runs, across layers and folds, through a WAL
/// replay, into checkpoints, and across tables when they are merged;
/// a key whose amounts sum to zero is absent at every one.
#[test]
fn tallies_sum_at_every_level() {
    let dir = tmpdir("tally");
    let open = || {
        StoreBuilder::new(DurabilityPolicy::RELAXED)
            .dir(&dir)
            .keyspace(KS)
            .keyspace(TALLY)
            .options_with(|o| {
                o.max_tables = 2;
                o.compaction_width = 2;
            })
            .build()
            .unwrap()
    };
    let add = |store: &Store, rows: &[(u8, i64)]| {
        let mut tx = store.begin().unwrap();
        let mut ks = tx.keyspace(TALLY);
        for &(k, d) in rows {
            ks.add(k, d).unwrap();
        }
        store.finish(tx).unwrap();
    };
    let get = |store: &Store, k: u8| store.snapshot().get(TALLY, &k).unwrap().map(|v| *v);
    let rows = |store: &Store| {
        store
            .snapshot()
            .range(TALLY, ..)
            .map(|r| r.map(|(k, v)| (*k, *v)))
            .collect::<Result<Vec<_>>>()
            .unwrap()
    };
    let table_rows = |path: &Path| {
        let table = Table::open(path, 0).unwrap();
        let mut c = table.cursor(TALLY.raw(), None);
        c.start().unwrap();
        let mut rows = Vec::new();
        while let Some((k, v)) = c.current() {
            rows.push((
                crate::store::codec::decode_all_ordered::<u8>(k).unwrap(),
                decode_all::<Option<i64>>(v).unwrap(),
            ));
            c.advance().unwrap();
        }
        rows
    };

    let store = open();
    // within a tx: adds to a key sum, and a sum of zero is nothing
    let mut tx = store.begin().unwrap();
    {
        let mut ks = tx.keyspace(TALLY);
        ks.add(1, 2).unwrap();
        ks.add(2, 1).unwrap();
        ks.add(1, 3).unwrap();
        ks.add(2, -1).unwrap();
        ks.add(9, 0).unwrap();
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&5));
        assert_eq!(ks.get(&2).unwrap().as_deref(), None, "cancelled");
        assert_eq!(ks.get(&9).unwrap().as_deref(), None, "nothing added");
        let seen: Vec<(u8, i64)> = ks
            .range(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(seen, [(1, 5)]);
    }
    store.finish(tx).unwrap();
    assert_eq!(rows(&store), [(1, 5)]);

    // across a tx's runs: the overlay adds to the snapshot, and a run
    // sees what the runs before it added
    let mut tx = store.begin().unwrap();
    {
        let mut ks = tx.keyspace(TALLY);
        ks.add(1, -5).unwrap();
        ks.add(3, 7).unwrap();
        assert_eq!(
            ks.get(&1).unwrap().as_deref(),
            None,
            "overlay cancels the snapshot"
        );
        assert_eq!(ks.get_old(&1).unwrap().as_deref(), Some(&5));
        let mut lookup = ks.lookup();
        assert_eq!(lookup.get(&1).unwrap().as_deref(), None);
        assert_eq!(lookup.get(&3).unwrap().as_deref(), Some(&7));
    }
    tx.end_run();
    {
        let mut ks = tx.keyspace(TALLY);
        assert_eq!(ks.get_old(&1).unwrap().as_deref(), None);
        ks.add(1, 1).unwrap();
        assert_eq!(ks.get(&1).unwrap().as_deref(), Some(&1));
        let old: Vec<(u8, i64)> = ks
            .range_old(..)
            .map(|r| r.map(|(k, v)| (*k, *v)).unwrap())
            .collect();
        assert_eq!(old, [(3, 7)]);
    }
    store.finish(tx).unwrap();
    assert_eq!(rows(&store), [(1, 1), (3, 7)]);

    // across tables: a checkpoint sums the layers it writes, and a read
    // sums the tables
    store.checkpoint().unwrap();
    add(&store, &[(1, -1), (3, 1), (4, 4)]);
    store.checkpoint().unwrap();
    let second = store.current.load().tables[1].path().to_path_buf();
    assert_eq!(
        table_rows(&second),
        [(1, Some(-1)), (3, Some(1)), (4, Some(4))],
        "a table holds what was added since the one before"
    );
    assert_eq!(get(&store, 1), None, "tables cancel");
    assert_eq!(get(&store, 3), Some(8));
    assert_eq!(rows(&store), [(3, 8), (4, 4)]);

    // across layers and their folds, copying while a snapshot pins the
    // layers and in place once it is gone
    let pin = store.snapshot();
    add(&store, &[(4, -4), (5, 5)]);
    add(&store, &[(5, 5), (6, 1)]);
    add(&store, &[(3, -8)]);
    assert_eq!(rows(&store), [(5, 10), (6, 1)]);
    drop(pin);
    store.quiesce().unwrap();
    add(&store, &[(6, -1), (7, 2)]);
    add(&store, &[(7, 3)]);
    store.quiesce().unwrap();
    assert_eq!(rows(&store), [(5, 10), (7, 5)]);

    // through a WAL replay
    drop(store);
    let store = open();
    assert_eq!(rows(&store), [(5, 10), (7, 5)]);
    assert_eq!(
        get(&store, 3),
        None,
        "the replayed layer cancels the tables"
    );

    // across merged tables: a key whose tables cancel is dropped
    store.checkpoint().unwrap();
    assert!(store.compact().unwrap());
    assert_eq!(store.stats().tables, 1);
    let only = store.current.load().tables[0].path().to_path_buf();
    assert_eq!(table_rows(&only), [(5, Some(10)), (7, Some(5))]);
    assert_eq!(rows(&store), [(5, 10), (7, 5)]);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn full_compaction_drops_tombstones() {
    let dir = tmpdir("compact-all");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    layered(&store);
    store.checkpoint().unwrap();
    assert_eq!(store.stats().tables, 3);
    assert!(store.compact().unwrap());
    assert_eq!(store.stats().tables, 1);
    let path = store.current.load().tables[0].path().to_path_buf();
    assert_eq!(
        checkpoint_rows(&path),
        [(1, Some(1)), (2, Some(20)), (4, Some(40)), (6, Some(6))],
        "no tombstones left"
    );
    assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
    drop(store);
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    assert_eq!(live_rows(&store), [(1, 1), (2, 20), (4, 40), (6, 6)]);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn compacting_above_the_oldest_table_keeps_tombstones() {
    let dir = tmpdir("compact-mid");
    let store = open_with(&dir, DurabilityPolicy::RELAXED, |b| {
        b.options_with(|o| {
            o.max_tables = 2;
            o.compaction_width = 2;
        })
    })
    .unwrap();
    // a big oldest table makes the newer two the cheapest window
    let big: Vec<_> = (0..200u8).map(|i| (i, Some(i as i64))).collect();
    put(&store, &big);
    store.checkpoint().unwrap();
    put(&store, &[(1, None)]);
    store.checkpoint().unwrap();
    put(&store, &[(250, Some(250))]);
    store.checkpoint().unwrap();
    put(&store, &[(251, Some(251))]);
    store.quiesce().unwrap();
    let v = store.current.load_full();
    assert_eq!(v.tables.len(), 2);
    assert!(checkpoint_rows(v.tables[1].path()).contains(&(1, None)));
    assert_eq!(
        read(&store, 1),
        None,
        "the tombstone still hides the oldest table's value"
    );
    assert_eq!(read(&store, 2), Some(2));
    drop(v);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn snapshots_read_tables_that_compaction_deleted() {
    let dir = tmpdir("compact-snap");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    layered(&store);
    store.checkpoint().unwrap();
    let old = store.snapshot();
    let old_paths: Vec<_> = old
        .version
        .tables
        .iter()
        .map(|t| t.path().to_path_buf())
        .collect();
    put(&store, &[(1, Some(100))]);
    store.compact().unwrap();
    assert!(old_paths.iter().all(|p| !p.exists()), "inputs are deleted");
    assert_eq!(old.get(KS, &1).unwrap().as_deref(), Some(&1));
    let rows: Vec<u8> = old.range(KS, ..).map(|r| *r.unwrap().0).collect();
    assert_eq!(rows, [1, 2, 4, 6]);
    assert_eq!(read(&store, 1), Some(100));
    drop(old);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_failed_wal_write_poisons_the_store_until_reopened() {
    let dir = tmpdir("poison");
    {
        let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
        put(&store, &[(1, Some(10))]);
        store.lock_writer().wal.as_mut().unwrap().fail_next_write = true;

        let mut tx = store.begin().unwrap();
        tx.keyspace(KS).absorb(2, Change::Insert(20)).unwrap();
        let err = store.finish(tx).unwrap_err();
        assert!(matches!(err, Error::Poisoned(_)), "{err}");
        assert_eq!(read(&store, 2), None, "nothing was published");
        assert!(matches!(store.begin(), Err(Error::Poisoned(_))));
        assert!(matches!(store.checkpoint(), Err(Error::Poisoned(_))));
        assert_eq!(read(&store, 1), Some(10), "reads still work");
    }
    let store = open(&dir, DurabilityPolicy::STRICT).unwrap();
    assert_eq!(read(&store, 1), Some(10));
    assert_eq!(read(&store, 2), None);
    assert_eq!(put(&store, &[(2, Some(20))]), Some(2));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_read_error_fails_the_tx_but_not_the_store() {
    let dir = tmpdir("read-err");
    let store = open(&dir, DurabilityPolicy::RELAXED).unwrap();
    put(&store, &[(1, Some(10)), (2, Some(20))]);
    store.checkpoint().unwrap();
    // damage the only data block; open handles see the new bytes
    let path = store.current.load().tables[0].path().to_path_buf();
    let mut bytes = fs::read(&path).unwrap();
    bytes[12] ^= 0xff;
    fs::write(&path, &bytes).unwrap();

    assert!(matches!(
        store.snapshot().get(KS, &1),
        Err(Error::Corrupt { .. })
    ));
    let snap = store.snapshot();
    let mut cursor_errs = snap.range(KS, ..).collect::<Vec<_>>();
    assert_eq!(cursor_errs.len(), 1, "a cursor stops at its first error");
    assert!(cursor_errs.pop().unwrap().is_err());
    drop(cursor_errs);
    drop(snap);

    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    assert!(ks.get(&1).is_err());
    assert!(ks.absorb(3, Change::Insert(30)).is_err(), "the tx failed");
    drop(ks);
    assert!(matches!(store.finish(tx), Err(Error::Corrupt { .. })));
    // the store is fine: a tx that does not read the damage commits
    let mut tx = store.begin().unwrap();
    tx.keyspace(KS).absorb(3, Change::Insert(30)).unwrap();
    assert_eq!(store.finish(tx).unwrap(), Some(2));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_memory_budget_sizes_the_limits_left_at_their_defaults() {
    let plain = StoreOptions::default();
    assert_eq!(plain.resolved(), plain, "no budget, no change");

    let o = StoreOptions {
        memory_budget: Some(256 << 20),
        cache_size: 1 << 20,
        ..StoreOptions::default()
    }
    .resolved();
    assert_eq!(o.memtable_limit, 64 << 20);
    assert_eq!(o.tx_memory_limit, 64 << 20);
    assert_eq!(o.graph_retain, 8 << 20);
    assert_eq!(o.wal_limit, 64 << 20);
    assert_eq!(o.cache_size, 1 << 20, "set by hand, so kept");

    let small = StoreOptions {
        memory_budget: Some(32 << 20),
        ..StoreOptions::default()
    }
    .resolved();
    assert_eq!(small.memtable_limit, 8 << 20);
    assert_eq!(small.wal_limit, 8 << 20);

    // the store runs on the resolved limits
    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(KS)
        .memory_budget(32 << 20)
        .build()
        .unwrap();
    assert_eq!(store.options().tx_memory_limit, 8 << 20);
}

#[test]
fn stats_count_the_open_tx_pinned_layers_and_unread_subscriptions() {
    let store = mem();
    assert_eq!(store.stats().total(), 0);

    // an open tx's writes
    let mut tx = store.begin().unwrap();
    let mut ks = tx.keyspace(KS);
    ks.absorb(1, Change::Insert(10)).unwrap();
    ks.absorb(2, Change::Insert(20)).unwrap();
    assert!(store.stats().tx_bytes > 0);
    store.finish(tx).unwrap();
    let s = store.stats();
    assert_eq!(s.tx_bytes, 0);
    assert!(s.memtable_bytes > 0);
    assert_eq!(s.pinned_bytes, 0);
    assert_eq!(s.total(), s.memtable_bytes);

    // a snapshot keeps the layers it saw once a fold replaces them
    let snap = store.snapshot();
    put(&store, &[(3, Some(30))]);
    put(&store, &[(4, Some(40))]);
    while store.fold_once() {}
    let s = store.stats();
    assert_eq!(s.layers, 1, "folded beneath the snapshot");
    assert!(s.pinned_bytes > 0, "{s:?}");
    drop(snap);
    assert_eq!(store.stats().pinned_bytes, 0);

    // changes no subscriber has received yet, counted once for all
    let (_, one) = store.subscribe(KS);
    let (_, two) = store.subscribe(KS);
    put(&store, &[(5, Some(50))]);
    let unread = store.stats().subscription_bytes;
    assert!(unread > 0);
    put(&store, &[(6, Some(60))]);
    assert_eq!(store.stats().subscription_bytes, 2 * unread);
    drop(one.recv());
    assert_eq!(
        store.stats().subscription_bytes,
        2 * unread,
        "the other subscriber has yet to read it"
    );
    drop(two.recv());
    assert_eq!(store.stats().subscription_bytes, unread);
    drop((one, two));
    assert_eq!(store.stats().subscription_bytes, 0);
}
