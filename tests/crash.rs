//! Kill-9 crash harness.
//!
//! The parent re-executes this binary with `BOG_CRASH_DIR` set; the
//! child opens the database, commits strictly durable transactions in a
//! loop, checkpointing and compacting every few commits, and prints an
//! ack after each `write` returns. The parent kills it cold at a random
//! moment, reopens the database, and checks the recovered state:
//!
//! - every acknowledged commit is present (the strict guarantee),
//! - committed keys form a contiguous prefix: transactions are atomic,
//! - the derived views equal what the base implies: derived state is
//!   recovered or rebuilt exactly,
//! - the epoch equals the number of committed transactions.
//!
//! The directory is reused across rounds, so each round also exercises
//! recover-then-continue, and kills land inside WAL appends, layer folds,
//! and checkpoint table and manifest I/O alike.

use std::{
    io::{BufRead, BufReader, Write as _},
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bog::{
    DurabilityPolicy, StoreBuilder,
    schema::{Db, Declare, MapView, Schema, Set, ValueView},
};

const ROUNDS: u32 = 8;
/// Sentinel for "no ack seen this round".
const NO_ACK: u64 = u64::MAX;

/// A set of sequence numbers, a derived mirror of it and a derived
/// count: the smallest graph that distinguishes base recovery from
/// derived recovery.
#[derive(Schema)]
struct Journal {
    base: Set<u64>,
    #[view]
    mirror: MapView<u64, u64>,
    #[view]
    count: ValueView<i64>,
}

impl Declare for Journal {
    fn declare(b: &mut JournalBuilder) {
        b.mirror.define(&b.base.map_with(|k, _| *k * 2));
        b.count.define(&b.base.count());
    }
}

fn open(dir: &Path) -> Db<Journal> {
    Journal::with(
        StoreBuilder::new(DurabilityPolicy::STRICT)
            .dir(dir)
            .wal_limit(64 << 10),
    )
    .build()
    .unwrap()
}

/// Commit `next, next+1, ..` forever, acking each one, until killed.
fn child(dir: &Path) -> ! {
    let db = open(dir);
    let mut next = db
        .read(|r| r.base.iter_rev().next().transpose())
        .unwrap()
        .map_or(0, |k| k + 1);
    loop {
        db.write(|tx| tx.base.insert(next).map(|_| ())).unwrap();
        // the write returned: the commit is synced; ack it
        println!("acked {next}");
        std::io::stdout().flush().unwrap();
        next += 1;
        if next.is_multiple_of(5) {
            db.checkpoint().unwrap();
        }
        if next.is_multiple_of(20) {
            db.store().compact().unwrap();
        }
    }
}

fn main() {
    if let Ok(dir) = std::env::var("BOG_CRASH_DIR") {
        child(Path::new(&dir));
    }

    let dir = std::env::temp_dir().join(format!("bog-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let exe = std::env::current_exe().unwrap();
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64
        | 1;
    let mut prev_committed = 0u64;

    for round in 0..ROUNDS {
        let mut victim = Command::new(&exe)
            .env("BOG_CRASH_DIR", &dir)
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn child");
        let stdout = victim.stdout.take().unwrap();
        let acked = Arc::new(AtomicU64::new(NO_ACK));
        let reader = {
            let acked = acked.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if let Some(n) = line
                        .strip_prefix("acked ")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        acked.store(n, Ordering::SeqCst);
                    }
                }
            })
        };

        // xorshift over a wall-clock seed: kill at an arbitrary moment
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        std::thread::sleep(Duration::from_millis(20 + seed % 130));
        victim.kill().expect("kill child");
        victim.wait().expect("wait child");
        reader.join().unwrap();
        let last_acked = acked.load(Ordering::SeqCst);

        let db = open(&dir);
        let (keys, mirror, count, epoch) = db.read(|r| {
            let keys: Vec<u64> = r.base.iter().map(|k| k.unwrap()).collect();
            let mirror: Vec<(u64, u64)> = r.mirror.iter().map(|k| k.unwrap()).collect();
            (keys, mirror, r.count.get().unwrap(), db.snapshot().epoch())
        });
        let want: Vec<u64> = (0..keys.len() as u64).collect();
        assert_eq!(
            keys, want,
            "round {round}: committed keys must form a contiguous prefix"
        );
        let committed = keys.len() as u64;
        if last_acked != NO_ACK {
            assert!(
                committed > last_acked,
                "round {round}: acked commit {last_acked} was lost (recovered {committed})"
            );
        }
        assert!(
            committed >= prev_committed,
            "round {round}: history went backwards ({prev_committed} -> {committed})"
        );
        prev_committed = committed;

        let want: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k * 2)).collect();
        assert_eq!(
            mirror, want,
            "round {round}: the mirror disagrees with the base"
        );
        assert_eq!(
            count,
            (committed > 0).then_some(committed as i64),
            "round {round}: the count disagrees with the base"
        );
        assert_eq!(
            epoch, committed,
            "round {round}: epoch disagrees with the commit count"
        );
        drop(db);

        eprintln!(
            "round {round}: {committed} committed, last acked {}",
            if last_acked == NO_ACK {
                "none".to_string()
            } else {
                last_acked.to_string()
            }
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
