//! The ledger benchmark through the schema layer: fold2's
//! `examples/ledger_bench.rs` schema, declared the same way, so the three
//! programs (fold2, this, and the hand-wired `ledger_bench`) measure the
//! same work.
//!
//! `cargo run --release --example ledger_schema_bench -- <memory|deferred|strict> <per_tx> <txs>`

use std::time::Instant;

use fold3::{
    DurabilityPolicy, StoreBuilder,
    schema::{
        Codec, Declare, Events, GroupView, Id, MapView, OrderedView, Schema, Table, ValueView,
    },
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
pub struct User {
    pub name: String,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
pub struct Transaction {
    pub from: Id<User>,
    pub to: Id<User>,
    pub amt: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
pub struct Mint {
    pub user: Id<User>,
    pub amt: u64,
}

#[derive(Schema)]
pub struct Ledger {
    pub users: Table<Id<User>, User>,
    pub transactions: Events<Transaction>,
    pub mint: Events<Mint>,
    #[view]
    pub balances: MapView<Id<User>, i128>,
    #[view]
    pub leaderboard: OrderedView<Id<User>, i128>,
    #[view]
    pub money_supply: ValueView<i128>,
    #[view]
    pub counterparties: GroupView<Id<User>, Id<User>>,
}

impl Declare for Ledger {
    fn declare(b: &mut LedgerBuilder) {
        b.transactions.reject(|t| t.amt == 0 || t.from == t.to);
        b.transactions.foreign_key(&b.users, |t| t.from);
        b.transactions.foreign_key(&b.users, |t| t.to);
        b.mint.foreign_key(&b.users, |m| m.user);
        b.money_supply.define(&b.mint.map(|m| m.amt as i128).sum());
        let sent = b.transactions.key_by(|t| t.from);
        let received = b.transactions.key_by(|t| t.to);
        let minted = b.mint.key_by(|m| m.user);
        let balances = sent
            .map(|t| -(t.amt as i128))
            .concat(&received.map(|t| t.amt as i128))
            .concat(&minted.map(|m| m.amt as i128))
            .concat(&b.users.map(|_| 0i128))
            .sum();
        b.balances.define(&balances);
        b.balances.reject(|_, bal| *bal < 0);
        b.leaderboard.define(&balances.filter(|_, bal| *bal != 0));
        b.counterparties.define(
            &sent
                .map(|t| t.to)
                .concat(&received.map(|t| t.from))
                .distinct(),
        );
    }
}

const N_USERS: u64 = 10_000;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args[1].clone();
    let per_tx: u64 = args[2].parse().unwrap();
    let txs: u64 = args[3].parse().unwrap();

    let dir =
        std::env::temp_dir().join(format!("fold3-ledger-schema-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = match mode.as_str() {
        "memory" => StoreBuilder::new(DurabilityPolicy::NONE),
        "deferred" => StoreBuilder::new(DurabilityPolicy::RELAXED).dir(&dir),
        "strict" => StoreBuilder::new(DurabilityPolicy::STRICT).dir(&dir),
        m => panic!("unknown mode {m}"),
    }
    .wal_limit(256 << 20);
    let db = Ledger::with(store).build().unwrap();

    let seed_start = Instant::now();
    let ids: Vec<Id<User>> = db
        .write(|tx| {
            let ids = tx.users.insert_all((0..N_USERS).map(|i| User {
                name: format!("user-{i}"),
            }))?;
            for id in &ids {
                tx.mint.push(Mint {
                    user: *id,
                    amt: 1 << 40,
                })?;
            }
            Ok(ids)
        })
        .unwrap();
    let seed_s = seed_start.elapsed().as_secs_f64();

    let mut rng = Rng(per_tx | 1);
    let (cpu0, _) = usage();
    let start = Instant::now();
    for _ in 0..txs {
        db.write(|tx| {
            for _ in 0..per_tx {
                let from = rng.below(N_USERS);
                let mut to = rng.below(N_USERS);
                if to == from {
                    to = (from + 1 + rng.below(N_USERS - 1)) % N_USERS;
                }
                tx.transactions.push(Transaction {
                    from: ids[from as usize],
                    to: ids[to as usize],
                    amt: 1 + rng.below(100),
                })?;
            }
            Ok(())
        })
        .unwrap();
    }
    let secs = start.elapsed().as_secs_f64();
    let (cpu1, rss) = usage();
    let transfers = txs * per_tx;
    let cpu = cpu1 - cpu0;
    println!(
        "fold3-schema,{mode},{per_tx},{txs},{transfers},{secs:.3},{:.0},{:.0},{cpu:.3},{:.2},{:.1},{seed_s:.3}",
        transfers as f64 / secs,
        txs as f64 / secs,
        cpu * 1e6 / transfers as f64,
        rss as f64 / (1 << 20) as f64,
    );
    // sanity: the ledger balances
    let supply = db.read(|r| r.money_supply.get()).unwrap().unwrap_or(0);
    let total: i128 = db.read(|r| r.balances.iter().map(|r| r.unwrap().1).sum());
    assert_eq!(total, supply);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
