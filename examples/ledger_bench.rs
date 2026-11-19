//! The fold2 ledger benchmark's workload on bog, as a standalone binary
//! so each run gets a fresh process to measure: 10,000 users each minted a
//! large balance, then random transfers, each checked (amount, distinct
//! parties, both parties exist, no overdraft) and folded into balances,
//! money supply, a leaderboard and a distinct counterparty set.
//!
//! `cargo run --release --example ledger_bench -- <memory|deferred|strict> <per_tx> <txs>`
//! prints one CSV line: see `HEADER`.

use std::time::Instant;

use bog::{
    DurabilityPolicy, Mergeless, StoreBuilder, Tally,
    dataflow::state::Arrangement,
    raw::{
        Arrange, Change, Check, Db, Distinct, FlatMap, Flatten, ForeignKey, FusedLinears, IndexBy,
        KeyspaceId, Map, OpDag, Reject, Sum,
    },
};

type User = u64;
type Transfer = (User, User, u64);
type Mint = (User, u64);

const USERS: KeyspaceId<User, Mergeless<String>> = KeyspaceId::new(1);
const TRANSACTIONS: KeyspaceId<u64, Mergeless<Transfer>> = KeyspaceId::new(2);
const MINTS: KeyspaceId<u64, Mergeless<Mint>> = KeyspaceId::new(3);
const BALANCES: KeyspaceId<User, (i64, i128)> = KeyspaceId::new(10);
const SUPPLY: KeyspaceId<(), (i64, i128)> = KeyspaceId::new(11);
const REFS: Arrangement<User, u64> = KeyspaceId::new(12);
const REF_TOTALS: KeyspaceId<User, Tally> = KeyspaceId::new(13);
const LEADERBOARD: KeyspaceId<(i128, User), ()> = KeyspaceId::new(14);
const COUNTERPARTIES: KeyspaceId<(User, User), Tally> = KeyspaceId::new(15);

/// Mint ids share the reference arrangement with transfer ids.
const MINT_BIT: u64 = 1 << 63;
const N_USERS: u64 = 10_000;

macro_rules! error {
    ($name:ident, $msg:literal) => {
        #[derive(Debug)]
        struct $name;
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str($msg)
            }
        }
        impl std::error::Error for $name {}
    };
}
error!(Invalid, "zero amount or self transfer");
error!(Overdraft, "overdraft");
error!(NoSuchUser, "no such user");

fn ledger() -> OpDag {
    let mut g = OpDag::new();
    let users = g.table(USERS);
    let txs = g.table(TRANSACTIONS);
    let mints = g.table(MINTS);

    g.nary(
        &[txs],
        Check::new(|_: &u64, &(from, to, amt): &Transfer| {
            if amt == 0 || from == to {
                Err(Invalid)
            } else {
                Ok(())
            }
        }),
    );
    let tx_rows = g.unary(txs, Flatten::default());
    let mint_rows = g.unary(mints, Flatten::default());
    let user_rows = g.unary(users, Flatten::default());

    // both parties of every transfer, and every minted user, must exist
    let tx_refs = g.nary(
        &[tx_rows],
        FusedLinears::new(FlatMap(|&(id, (from, to, _)): &(u64, Transfer)| {
            [(from, id), (to, id)]
        })),
    );
    let mint_refs = g.nary(
        &[mint_rows],
        FusedLinears::new(Map(|&(id, (user, _)): &(u64, Mint)| (user, id | MINT_BIT))),
    );
    let refs = g.nary(&[tx_refs, mint_refs], Arrange::new(REFS, REF_TOTALS));
    let dangling = g.binary_with(refs, users, ForeignKey::new);
    g.nary(&[dangling], Reject::new(|_: &User| NoSuchUser));

    let supply = g.nary(
        &[mint_rows],
        FusedLinears::new(Map(|&(_, (_, amt)): &(u64, Mint)| ((), amt as i128))),
    );
    g.nary(&[supply], Sum::<(), i128>::new(SUPPLY));

    let sent_received = g.nary(
        &[tx_rows],
        FusedLinears::new(FlatMap(|&(_, (from, to, amt)): &(u64, Transfer)| {
            [(from, -(amt as i128)), (to, amt as i128)]
        })),
    );
    let minted = g.nary(
        &[mint_rows],
        FusedLinears::new(Map(|&(_, (user, amt)): &(u64, Mint)| (user, amt as i128))),
    );
    let opened = g.nary(
        &[user_rows],
        FusedLinears::new(Map(|&(user, _): &(User, String)| (user, 0i128))),
    );
    let balances = g.nary(
        &[sent_received, minted, opened],
        Sum::<User, i128>::new(BALANCES),
    );
    g.nary(
        &[balances],
        Check::new(
            |_: &User, &(_, bal): &(i64, i128)| if bal < 0 { Err(Overdraft) } else { Ok(()) },
        ),
    );
    g.nary(
        &[balances],
        IndexBy::new(LEADERBOARD, |&user: &User, &(_, bal): &(i64, i128)| {
            (bal != 0).then_some((-bal, user))
        }),
    );
    let pairs = g.nary(
        &[tx_rows],
        FusedLinears::new(FlatMap(|&(_, (from, to, _)): &(u64, Transfer)| {
            [(from, to), (to, from)]
        })),
    );
    g.nary(&[pairs], Distinct::new(COUNTERPARTIES));
    g
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

#[repr(C)]
struct Timeval {
    sec: i64,
    usec: i32,
}
#[repr(C)]
struct Rusage {
    utime: Timeval,
    stime: Timeval,
    /// `ru_maxrss` first: bytes on macOS.
    rest: [i64; 14],
}
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}
/// (cpu seconds, peak RSS bytes) of this process so far.
fn usage() -> (f64, i64) {
    let mut r = std::mem::MaybeUninit::<Rusage>::zeroed();
    // SAFETY: getrusage fills the struct, laid out as on darwin
    let r = unsafe {
        getrusage(0, r.as_mut_ptr());
        r.assume_init()
    };
    let secs = |t: &Timeval| t.sec as f64 + t.usec as f64 / 1e6;
    (secs(&r.utime) + secs(&r.stime), r.rest[0])
}

pub const HEADER: &str = "impl,mode,per_tx,txs,transfers,seconds,transfers_per_s,txs_per_s,cpu_s,cpu_us_per_transfer,peak_rss_mb,seed_s";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("header") {
        println!("{HEADER}");
        return;
    }
    let mode = args[1].clone();
    let per_tx: u64 = args[2].parse().unwrap();
    let txs: u64 = args[3].parse().unwrap();

    let dir = std::env::temp_dir().join(format!("bog-ledger-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let builder = match mode.as_str() {
        "memory" => StoreBuilder::new(DurabilityPolicy::NONE),
        "deferred" => StoreBuilder::new(DurabilityPolicy::RELAXED).dir(&dir),
        "strict" => StoreBuilder::new(DurabilityPolicy::STRICT).dir(&dir),
        m => panic!("unknown mode {m}"),
    }
    .wal_limit(256 << 20)
    .options_with(|o| {
        if let Some(n) = std::env::var("MAX_LAYERS")
            .ok()
            .and_then(|n| n.parse().ok())
        {
            o.max_layers = n;
        }
    });
    let db = Db::open(builder, ledger()).unwrap();

    let seed_start = Instant::now();
    db.write(|tx| {
        let mut users = tx.table(USERS);
        for u in 0..N_USERS {
            users.absorb(u, Change::Insert(format!("user-{u}")))?;
        }
        let mut mints = tx.table(MINTS);
        for u in 0..N_USERS {
            mints.absorb(u, Change::Insert((u, 1 << 40)))?;
        }
        Ok(())
    })
    .unwrap();
    let seed_s = seed_start.elapsed().as_secs_f64();

    let mut rng = Rng(per_tx | 1);
    let mut next_id = 0u64;
    let (cpu0, _) = usage();
    let start = Instant::now();
    for _ in 0..txs {
        db.write(|tx| {
            let mut t = tx.table(TRANSACTIONS);
            for _ in 0..per_tx {
                let from = rng.below(N_USERS);
                let mut to = rng.below(N_USERS);
                if to == from {
                    to = (from + 1 + rng.below(N_USERS - 1)) % N_USERS;
                }
                t.absorb(next_id, Change::Insert((from, to, 1 + rng.below(100))))?;
                next_id += 1;
            }
            Ok(())
        })
        .unwrap();
    }
    let secs = start.elapsed().as_secs_f64();
    if std::env::var_os("LEDGER_STATS").is_some() {
        eprintln!("{:?}", db.store().stats());
    }
    let (cpu1, rss) = usage();
    let transfers = txs * per_tx;
    let cpu = cpu1 - cpu0;
    println!(
        "bog,{mode},{per_tx},{txs},{transfers},{secs:.3},{:.0},{:.0},{cpu:.3},{:.2},{:.1},{seed_s:.3}",
        transfers as f64 / secs,
        txs as f64 / secs,
        cpu * 1e6 / transfers as f64,
        rss as f64 / (1 << 20) as f64,
    );
    // sanity: the ledger balances
    let supply = db.snapshot().get(SUPPLY, &()).unwrap().unwrap().1;
    let total: i128 = db
        .snapshot()
        .range(BALANCES, ..)
        .map(|r| r.unwrap().1.1)
        .sum();
    assert_eq!(total, supply);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
