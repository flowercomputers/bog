//! A small ledger, built only from the public API: accounts and transfers
//! are tables; balances, a per-account transfer index, a foreign-key
//! check and a names-with-balances join are derived.

use std::{borrow::Cow, collections::BTreeMap, path::PathBuf};

use bog::{
    DurabilityPolicy, Error, Mergeless, StoreBuilder, Tally,
    dataflow::state::{Arrangement, Edge},
    raw::{
        Arrange, Db, FlatMap, Flatten, ForeignKey, FusedLinears, Join, KeyspaceId, OpDag, Reject,
        Sum,
    },
};

type Account = u64;
type Transfer = (Account, Account, i64);

const ACCOUNTS: KeyspaceId<Account, Mergeless<String>> = KeyspaceId::new(1);
const TRANSFERS: KeyspaceId<u64, Mergeless<Transfer>> = KeyspaceId::new(2);
const BALANCES: KeyspaceId<Account, (i64, i64)> = KeyspaceId::new(10);
const BY_ACCOUNT: Arrangement<Account, u64> = KeyspaceId::new(11);
const BY_ACCOUNT_TOTALS: KeyspaceId<Account, Tally> = KeyspaceId::new(12);
const NAMED: Arrangement<String, i64> = KeyspaceId::new(13);
const NAMED_TOTALS: KeyspaceId<String, Tally> = KeyspaceId::new(14);

#[derive(Debug)]
struct NoSuchAccount(Account);
impl std::fmt::Display for NoSuchAccount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no account {}", self.0)
    }
}
impl std::error::Error for NoSuchAccount {}

/// The graph.
fn ledger(version: u64) -> OpDag {
    let mut g = OpDag::new();
    g.version(version);
    let accounts = g.table(ACCOUNTS);
    let transfers = g.table(TRANSFERS);
    let rows = g.unary(transfers, Flatten::default());

    // balances: each transfer debits one account and credits another
    let legs = g.nary(
        &[rows],
        FusedLinears::new(FlatMap(|&(_, (from, to, amount)): &(u64, Transfer)| {
            [(from, -amount), (to, amount)]
        })),
    );
    let balances = g.nary(&[legs], Sum::<Account, i64>::new(BALANCES));

    // every transfer, indexed by the accounts it touches, must name
    // accounts that exist
    let touches = g.nary(
        &[rows],
        FusedLinears::new(FlatMap(|&(id, (from, to, _)): &(u64, Transfer)| {
            [(from, id), (to, id)]
        })),
    );
    let by_account = g.nary(&[touches], Arrange::new(BY_ACCOUNT, BY_ACCOUNT_TOTALS));
    let dangling = g.binary_with(by_account, accounts, ForeignKey::new);
    g.nary(&[dangling], Reject::new(|a: &Account| NoSuchAccount(*a)));

    // names with balances: both sides are registers keyed by account, so
    // the join keeps nothing; arranging its output keeps it to read
    let named = g.binary_with(balances, accounts, |b, a| {
        Join::new(
            b,
            a,
            |_: &Account, &(_, balance): &(i64, i64), name: &String| (name.clone(), balance),
        )
    });
    g.nary(&[named], Arrange::new(NAMED, NAMED_TOTALS));
    g
}

fn dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bog-ledger-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn open(dir: &PathBuf, version: u64) -> Db {
    let graph = ledger(version);
    Db::open(StoreBuilder::new(DurabilityPolicy::RELAXED).dir(dir), graph).unwrap()
}

fn named(db: &Db) -> Vec<(String, i64)> {
    db.snapshot()
        .range(NAMED, ..)
        .map(|r| {
            let (k, _) = r.unwrap();
            match &k.1 {
                Edge::Val(balance) => (k.0.clone(), *balance),
                _ => unreachable!("arrangements hold values"),
            }
        })
        .collect()
}

fn balances(db: &Db) -> BTreeMap<Account, i64> {
    db.snapshot()
        .range(BALANCES, ..)
        .map(|r| r.map(|(k, v)| (*k, v.1)).unwrap())
        .collect()
}

#[test]
fn a_ledger_keeps_balances_and_refuses_unknown_accounts() {
    let dir = dir();
    let db = open(&dir, 0);
    db.write(|tx| {
        let mut accounts = tx.table(ACCOUNTS);
        accounts.set(1, Some("alice".into()))?;
        accounts.set(2, Some("bob".into()))?;
        tx.table(TRANSFERS).set(100, Some((1, 2, 30)))?;
        Ok(())
    })
    .unwrap();
    assert_eq!(balances(&db), BTreeMap::from([(1, -30), (2, 30)]));

    // derived state is readable inside the tx once the graph has run
    db.write(|tx| {
        tx.table(TRANSFERS).set(101, Some((2, 1, 5)))?;
        tx.run()?;
        let b = tx.read(BALANCES).get(&1)?.map(Cow::into_owned);
        assert_eq!(b, Some((2, -25)));
        Ok(())
    })
    .unwrap();

    // a transfer to an account that does not exist is refused whole
    let err = db
        .write(|tx| {
            tx.table(TRANSFERS).set(102, Some((1, 9, 1)))?;
            Ok(())
        })
        .unwrap_err();
    assert!(
        matches!(&err, Error::Rejected(e) if e.to_string() == "no account 9"),
        "{err}"
    );
    assert_eq!(balances(&db), BTreeMap::from([(1, -25), (2, 25)]));
    // and so is closing an account that transfers still name
    assert!(db.write(|tx| tx.table(ACCOUNTS).set(2, None)).is_err());

    // reversing a transfer is an update to its row
    db.write(|tx| tx.table(TRANSFERS).set(100, Some((2, 1, 30))))
        .unwrap();
    assert_eq!(balances(&db), BTreeMap::from([(1, 35), (2, -35)]));
    assert_eq!(named(&db), vec![("alice".into(), 35), ("bob".into(), -35)]);
    drop(db);

    // the same graph reopens as it was; a changed one rebuilds from the
    // tables
    assert_eq!(
        balances(&open(&dir, 0)),
        BTreeMap::from([(1, 35), (2, -35)])
    );
    let db = open(&dir, 1);
    assert_eq!(balances(&db), BTreeMap::from([(1, 35), (2, -35)]));
    assert_eq!(named(&db), vec![("alice".into(), 35), ("bob".into(), -35)]);
    let newest: Vec<Account> = db
        .snapshot()
        .range_rev(BALANCES, ..)
        .map(|r| *r.unwrap().0)
        .collect();
    assert_eq!(newest, vec![2, 1]);
    db.store().checkpoint().unwrap();
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
