//! `attempt`: a unit of writes inside a transaction that is undone on
//! its own when it fails, at a write-time check or at a constraint on
//! derived state, leaving the rest of the transaction to commit.

use std::collections::BTreeMap;

use bog::{
    Error,
    schema::{Change, Declare, Events, MapView, Rejected, Schema, Table, ValueView},
};

#[derive(Schema)]
struct Bank {
    /// Account -> owner name, unique.
    accounts: Table<u64, String>,
    /// (from, to, amount).
    transfers: Events<(u64, u64, i64)>,
    #[view]
    balances: MapView<u64, i64>,
    #[view]
    by_owner: MapView<String, String>,
    #[view]
    transfer_count: ValueView<i64>,
}

impl Declare for Bank {
    fn declare(b: &mut BankBuilder) {
        b.transfers
            .reject(|&(from, to, amount)| amount <= 0 || from == to);
        let out = b.transfers.key_by(|t| t.0).map(|t| -t.2);
        let into = b.transfers.key_by(|t| t.1).map(|t| t.2);
        b.balances.define(&out.concat(&into).sum());
        // checked after the graph runs: nobody but the mint (0) overdraws
        b.balances.reject(|acct, bal| *acct != 0 && *bal < 0);
        b.by_owner
            .define(&b.accounts.unique_key(|name| name.clone()));
        b.transfer_count.define(&b.transfers.count());
    }
}

fn balances(db: &bog::Db<Bank>) -> BTreeMap<u64, i64> {
    db.read(|r| r.balances.iter().map(|r| r.unwrap()).collect())
}

#[test]
fn a_failed_attempt_is_undone_and_the_rest_commits() {
    let db = Bank::memory().build().unwrap();
    let sub = db.read(|r| r.balances.subscribe());
    let outcomes = db
        .write(|tx| {
            let mut out = Vec::new();
            tx.accounts.upsert(1, "ann".into())?;
            // fine
            out.push(tx.attempt(|tx| tx.transfers.push((0, 1, 100)).map(drop)));
            // refused at the write
            out.push(tx.attempt(|tx| tx.transfers.push((1, 1, 5)).map(drop)));
            // refused after the graph: 2 would go negative, and the
            // account row written with it goes too
            out.push(tx.attempt(|tx| {
                tx.accounts.upsert(2, "bob".into())?;
                tx.transfers.push((2, 1, 30)).map(drop)
            }));
            // a unique key refused at the write, after an earlier write
            out.push(tx.attempt(|tx| {
                tx.transfers.push((1, 3, 10))?;
                tx.accounts.upsert(3, "ann".into())
            }));
            // the closure's own error
            out.push(tx.attempt(|tx| {
                tx.transfers.push((1, 4, 99))?;
                Err(Error::reject(Rejected::new("test", "changed my mind")))
            }));
            // fine, and reads see the survivors only
            out.push(tx.attempt(|tx| {
                assert_eq!(tx.balances.get(&1)?, Some(100));
                assert_eq!(tx.balances.get(&3)?, None);
                assert_eq!(tx.accounts.get(&2)?, None);
                tx.transfers.push((1, 5, 40)).map(drop)
            }));
            Ok(out)
        })
        .unwrap();
    let ok: Vec<bool> = outcomes.iter().map(|r| r.is_ok()).collect();
    assert_eq!(ok, [true, false, false, false, false, true]);
    assert!(
        outcomes[2]
            .as_ref()
            .unwrap_err()
            .rejected::<Rejected>()
            .is_some()
    );

    assert_eq!(balances(&db), BTreeMap::from([(0, -100), (1, 60), (5, 40)]));
    db.read(|r| {
        assert_eq!(r.transfer_count.get().unwrap(), Some(2));
        assert_eq!(r.accounts.len().unwrap(), 1);
        // ids minted by undone attempts were handed out again
        assert_eq!(r.transfers.iter().count(), 2);
    });
    // one commit, holding the net change only
    let commit = sub.try_recv().unwrap().expect("a commit");
    assert_eq!(
        commit.changes,
        vec![
            (0, Change::Insert(-100)),
            (1, Change::Insert(60)),
            (5, Change::Insert(40))
        ]
    );
    assert!(sub.try_recv().unwrap().is_none());
}

#[test]
fn attempts_nest() {
    let db = Bank::memory().build().unwrap();
    db.write(|tx| {
        tx.transfers.push((0, 1, 10))?;
        let r = tx.attempt(|tx| {
            tx.transfers.push((0, 2, 10))?;
            // the inner one fails alone
            assert!(
                tx.attempt(|tx| tx.transfers.push((3, 1, 1)).map(drop))
                    .is_err()
            );
            assert_eq!(tx.balances.get(&2)?, Some(10));
            tx.attempt(|tx| tx.transfers.push((0, 4, 10)).map(drop))?;
            // then the outer fails, taking the inner success with it
            tx.transfers.push((5, 1, 1)).map(drop)
        });
        assert!(r.is_err());
        tx.transfers.push((0, 6, 10)).map(drop)
    })
    .unwrap();
    assert_eq!(balances(&db), BTreeMap::from([(0, -20), (1, 10), (6, 10)]));
}

/// Many attempts in one transaction against a model: the overlay's runs
/// are merged as savepoints close, and reads stay right throughout.
#[test]
fn a_long_run_of_attempts_matches_a_model() {
    let db = Bank::memory().build().unwrap();
    let mut model: BTreeMap<u64, i64> = BTreeMap::new();
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..4 {
        db.write(|tx| {
            for i in 0..400 {
                let from = if i % 7 == 0 { 0 } else { rng() % 20 };
                let to = rng() % 20;
                let amount = (rng() % 50) as i64;
                let r = tx.attempt(|tx| tx.transfers.push((from, to, amount)).map(drop));
                let legal = amount > 0
                    && from != to
                    && (from == 0 || model.get(&from).copied().unwrap_or(0) >= amount);
                assert_eq!(
                    r.is_ok(),
                    legal,
                    "round {round} step {i}: {from}->{to} {amount}"
                );
                if legal {
                    *model.entry(from).or_default() -= amount;
                    *model.entry(to).or_default() += amount;
                }
                if i % 37 == 0 {
                    let k = rng() % 20;
                    assert_eq!(
                        tx.balances.get(&k)?.unwrap_or(0),
                        model.get(&k).copied().unwrap_or(0)
                    );
                }
            }
            Ok(())
        })
        .unwrap();
        let mut want = model.clone();
        want.retain(|_, v| *v != 0);
        assert_eq!(balances(&db), want, "round {round}");
    }
}
