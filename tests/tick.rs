//! A ticked database: queued units run as one commit per tick, each
//! undone alone when it fails.

use std::sync::{Arc, Mutex};

use bog::{
    schema::{Cell, ChangeSet, Declare, Events, MapView, Schema},
    tick::Ticker,
};

#[derive(Schema)]
struct Bank {
    clock: Cell<u64>,
    transfers: Events<(u64, u64, i64)>,
    #[view]
    balances: MapView<u64, i64>,
}

impl Declare for Bank {
    fn declare(b: &mut BankBuilder) {
        let out = b.transfers.key_by(|t| t.0).map(|t| -t.2);
        let into = b.transfers.key_by(|t| t.1).map(|t| t.2);
        b.balances.define(&out.concat(&into).sum());
        b.balances.reject(|acct, bal| *acct != 0 && *bal < 0);
    }
}

#[test]
fn a_tick_is_one_commit_and_bad_units_fail_alone() {
    let commits = Arc::new(Mutex::new(Vec::new()));
    let seen = commits.clone();
    let db = Bank::memory()
        .on_changes(move |c, r| {
            seen.lock()
                .unwrap()
                .push((c.epoch(), c.transfers.len(), r.clock.get().unwrap()));
        })
        .build()
        .unwrap();
    let ticker = Arc::new(Ticker::new(db, 0));
    ticker.system(|tx, tick| tx.clock.set(tick));

    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let transfer = |from: u64, to: u64, amount: i64| {
        let outcomes = outcomes.clone();
        ticker.submit_then(
            move |tx| tx.transfers.push((from, to, amount)).map(drop),
            move |r| outcomes.lock().unwrap().push(r.is_ok()),
        );
    };
    // submitted from other threads, in a known order
    for (from, to, amount) in [(0, 1, 50), (2, 1, 5), (1, 2, 20), (2, 3, 30)] {
        std::thread::scope(|s| {
            s.spawn(|| transfer(from, to, amount));
        });
    }
    assert_eq!(ticker.queued(), 4);
    assert!(
        commits.lock().unwrap().is_empty(),
        "nothing runs before the tick"
    );

    assert_eq!(ticker.tick().unwrap(), 1);
    // 2 has nothing to send at first, and too little later
    assert_eq!(*outcomes.lock().unwrap(), [true, false, true, false]);
    assert_eq!(*commits.lock().unwrap(), [(1, 2, Some(1))]);
    ticker.db().read(|r| {
        let balances: Vec<_> = r.balances.iter().map(|r| r.unwrap()).collect();
        assert_eq!(balances, [(0, -50), (1, 30), (2, 20)]);
    });

    // a tick with nothing queued still runs the systems
    assert_eq!(ticker.tick().unwrap(), 2);
    assert_eq!(commits.lock().unwrap().last(), Some(&(2, 0, Some(2))));
    assert_eq!(ticker.last_tick(), 2);
}
