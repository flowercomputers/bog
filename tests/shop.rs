//! The schema layer, through the public API: fold2's record shop (three
//! tables, chained joins, sums, an ordered view, anti-joins, a foreign
//! key, `unique`, live counts), plus a nested schema, constraints on
//! views and sources, subscriptions, reads inside a transaction, and
//! reopening with and without a changed graph.

use std::{collections::BTreeMap, path::PathBuf};

use fold3::{
    Error,
    schema::{
        Cell, Change, Codec, Declare, Events, GroupView, Id, MapView, Multiset, OrderedView,
        Rejected, Retain, Schema, Set, Table, ValueView,
    },
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
struct Customer {
    name: String,
    region: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
struct Album {
    title: String,
    price_cents: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
struct Order {
    customer: Id<Customer>,
    album: Id<Album>,
    qty: i64,
}

#[derive(Schema)]
struct Shop {
    customers: Table<Id<Customer>, Customer>,
    albums: Table<Id<Album>, Album>,
    orders: Table<Id<Order>, Order>,

    /// Region -> revenue in cents.
    #[view]
    revenue: MapView<String, i64>,
    /// Title -> units sold, best first.
    #[view]
    top_sellers: OrderedView<String, i64>,
    /// Customers who have never ordered.
    #[view]
    dormant: MapView<Id<Customer>, Customer>,
    /// Orders whose customer does not exist.
    #[view]
    orphans: MapView<Id<Order>, Id<Customer>>,
    /// Every order line by customer.
    #[view]
    lines: GroupView<Id<Customer>, (String, i64)>,
    #[view]
    customer_count: ValueView<i64>,
    #[view]
    order_count: ValueView<i64>,
}

impl Declare for Shop {
    fn declare(b: &mut ShopBuilder) {
        b.orders.reject(|o| o.qty <= 0);
        b.orders.foreign_key(&b.albums, |o| o.album);

        // orders ⋈ albums: one line per order
        let lines = b.orders.key_by(|o| o.album).with(&b.albums, |_, o, a| {
            (o.customer, o.qty * a.price_cents, a.title.clone(), o.qty)
        });

        // ⋈ customers, then sum per region
        b.revenue.define(
            &lines
                .key_by(|(cust, ..)| *cust)
                .with(&b.customers, |_, (_, cents, ..), c| {
                    (c.region.clone(), *cents)
                })
                .key_by(|(region, _)| region.clone())
                .map(|(_, cents)| *cents)
                .sum(),
        );

        b.top_sellers.define(
            &lines
                .key_by(|(.., title, _)| title.clone())
                .map(|(.., qty)| *qty)
                .sum(),
        );

        b.dormant
            .define(&b.customers.without(&b.orders.key_by(|o| o.customer)));

        b.orphans.define(
            &b.orders
                .map_with(|id, o| (*id, o.customer))
                .key_by(|(_, cust)| *cust)
                .without(&b.customers)
                .key_by(|(order, _)| *order)
                .map(|(_, cust)| *cust)
                .unique(),
        );

        b.lines.define(
            &lines
                .key_by(|(cust, ..)| *cust)
                .map(|(_, cents, title, _)| (title.clone(), *cents)),
        );

        b.customer_count.define(&b.customers.count());
        b.order_count.define(&b.orders.count());
    }
}

fn customer(name: &str, region: &str) -> Customer {
    Customer {
        name: name.into(),
        region: region.into(),
    }
}

fn album(title: &str, price_cents: i64) -> Album {
    Album {
        title: title.into(),
        price_cents,
    }
}

fn dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fold3-shop-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn revenue(r: &ShopRead<'_>) -> BTreeMap<String, i64> {
    r.revenue.iter().map(|r| r.unwrap()).collect()
}

fn sellers(r: &ShopRead<'_>) -> Vec<(String, i64)> {
    r.top_sellers.desc().map(|r| r.unwrap()).collect()
}

fn dormant(r: &ShopRead<'_>) -> Vec<String> {
    r.dormant.iter().map(|r| r.unwrap().1.name).collect()
}

#[test]
fn the_record_shop_keeps_every_view_current() {
    let dir = dir("views");
    let db = Shop::open(&dir).build().unwrap();

    let (ada, cy, blue_train, kind_of_blue, orphan) = db
        .write(|tx| {
            let ada = tx.customers.insert(customer("ada", "emea"))?;
            let bob = tx.customers.insert(customer("bob", "apac"))?;
            let cy = tx.customers.insert(customer("cy", "emea"))?;
            let blue_train = tx.albums.insert(album("Blue Train", 1999))?;
            let kind_of_blue = tx.albums.insert(album("Kind of Blue", 1499))?;
            let supreme = tx.albums.insert(album("A Love Supreme", 1799))?;
            tx.orders.insert(Order {
                customer: ada,
                album: blue_train,
                qty: 2,
            })?;
            tx.orders.insert(Order {
                customer: bob,
                album: kind_of_blue,
                qty: 1,
            })?;
            // customer 99 does not exist: observed by `orphans`, not refused
            let orphan = tx.orders.insert(Order {
                customer: Id::new(99),
                album: supreme,
                qty: 1,
            })?;
            // derived state is readable inside the transaction
            assert_eq!(tx.revenue.get(&"emea".to_string())?, Some(3998));
            assert_eq!(tx.customer_count.get()?, Some(3));
            // and the state the transaction started from is empty
            assert_eq!(tx.read().customer_count.get()?, None);
            Ok((ada, cy, blue_train, kind_of_blue, orphan))
        })
        .unwrap();

    db.read(|r| {
        assert_eq!(
            revenue(r),
            BTreeMap::from([("emea".into(), 3998), ("apac".into(), 1499)])
        );
        // ties order by title, descending
        assert_eq!(
            sellers(r),
            vec![
                ("Blue Train".into(), 2),
                ("Kind of Blue".into(), 1),
                ("A Love Supreme".into(), 1)
            ]
        );
        assert_eq!(
            r.top_sellers.get(&"Blue Train".to_string()).unwrap(),
            Some(2)
        );
        assert_eq!(dormant(r), vec!["cy"]);
        let orphans: Vec<_> = r.orphans.iter().map(|r| r.unwrap()).collect();
        assert_eq!(orphans, vec![(orphan, Id::new(99))]);
        assert_eq!(r.customer_count.get().unwrap(), Some(3));
        assert_eq!(r.order_count.get().unwrap(), Some(3));
        assert_eq!(
            r.lines.get(&ada).unwrap(),
            vec![(("Blue Train".to_string(), 3998), 1)]
        );
        assert!(r.lines.contains(&ada).unwrap());
        assert!(!r.lines.contains(&cy).unwrap());
        assert_eq!(r.customers.len().unwrap(), 3);
        assert_eq!(
            r.albums.get(&blue_train).unwrap().unwrap().title,
            "Blue Train"
        );
    });

    // a rejected row: the write call refuses it
    let err = db
        .write(|tx| {
            tx.orders.insert(Order {
                customer: ada,
                album: blue_train,
                qty: 0,
            })
        })
        .unwrap_err();
    let rejected = err.rejected::<Rejected>().expect("a predicate rejection");
    assert_eq!(rejected.at, "orders");

    // an order for a missing album: the foreign key refuses the commit
    let err = db
        .write(|tx| {
            tx.orders.insert(Order {
                customer: ada,
                album: Id::new(404),
                qty: 1,
            })?;
            // visible inside the transaction, since the graph has not
            // run the check yet
            assert_eq!(tx.order_count.get()?, Some(4));
            Ok(())
        })
        .unwrap_err();
    assert!(
        err.rejected::<Rejected>()
            .is_some_and(|r| r.reason.contains("foreign key to albums")),
        "{err}"
    );
    db.read(|r| assert_eq!(r.order_count.get().unwrap(), Some(3)));
    // and so does removing an album that orders still name
    assert!(db.write(|tx| tx.albums.remove(&blue_train)).is_err());

    // the dormant customer orders: dormancy retracts
    db.write(|tx| {
        tx.orders.insert(Order {
            customer: cy,
            album: kind_of_blue,
            qty: 1,
        })
    })
    .unwrap();
    db.read(|r| {
        assert_eq!(dormant(r), Vec::<String>::new());
        assert_eq!(
            sellers(r),
            vec![
                ("Kind of Blue".into(), 2),
                ("Blue Train".into(), 2),
                ("A Love Supreme".into(), 1)
            ]
        );
    });

    // repricing corrects both joins and every sum downstream
    db.write(|tx| tx.albums.upsert(blue_train, album("Blue Train", 2499)))
        .unwrap();
    db.read(|r| assert_eq!(revenue(r)["emea"], 4998 + 1499));

    // moving a customer re-homes her revenue
    db.write(|tx| tx.customers.upsert(ada, customer("ada", "apac")))
        .unwrap();
    db.read(|r| {
        assert_eq!(
            revenue(r),
            BTreeMap::from([("emea".into(), 1499), ("apac".into(), 1499 + 4998)])
        );
        assert_eq!(
            r.top_sellers.top(1).unwrap(),
            vec![("Kind of Blue".into(), 2)]
        );
        assert_eq!(
            r.top_sellers.bottom(1).unwrap(),
            vec![("A Love Supreme".into(), 1)]
        );
        assert_eq!(
            r.top_sellers
                .by_value(2..)
                .map(|r| r.unwrap())
                .collect::<Vec<_>>(),
            vec![("Blue Train".into(), 2), ("Kind of Blue".into(), 2)]
        );
    });

    // update in place, and a seek over sorted keys
    db.write(|tx| {
        assert!(tx.customers.update(&ada, |c| c.name.push('!'))?);
        Ok(())
    })
    .unwrap();
    db.read(|r| {
        let mut seek = r.customers.seek();
        assert_eq!(seek.get(&ada).unwrap().unwrap().name, "ada!");
        assert_eq!(seek.get(&cy).unwrap().unwrap().name, "cy");
        assert!(seek.get(&Id::new(1 << 60)).unwrap().is_none());
    });
    let before = db.read(|r| (revenue(r), sellers(r), dormant(r)));
    db.close().unwrap();

    // the same schema reopens as it was; a bumped version rebuilds
    for version in [0, 1] {
        let db = Shop::open(&dir).version(version).build().unwrap();
        db.read(|r| assert_eq!((revenue(r), sellers(r), dormant(r)), before));
        db.read(|r| assert_eq!(r.order_count.get().unwrap(), Some(4)));
        drop(db);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
struct Transfer {
    from: u64,
    to: u64,
    amount: i64,
}

#[derive(Debug)]
struct Overdraft(u64);
impl std::fmt::Display for Overdraft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "account {} overdrawn", self.0)
    }
}
impl std::error::Error for Overdraft {}

/// A ledger nested in a larger schema, with its own constraints.
#[derive(Schema)]
struct Ledger {
    accounts: Set<u64>,
    transfers: Events<Transfer>,
    #[view]
    balances: MapView<u64, i64>,
    #[view]
    supply: ValueView<i64>,
}

impl Declare for Ledger {
    fn declare(b: &mut LedgerBuilder) {
        b.transfers.reject(|t| t.amount <= 0 || t.from == t.to);
        b.transfers.foreign_key(&b.accounts, |t| t.from);
        b.transfers.foreign_key(&b.accounts, |t| t.to);
        let out = b.transfers.key_by(|t| t.from).map(|t| -t.amount);
        let into = b.transfers.key_by(|t| t.to).map(|t| t.amount);
        // the mint account may go negative; nobody else may
        b.balances.define(&out.concat(&into).sum());
        b.balances.check(|&account, &balance| {
            if account != 0 && balance < 0 {
                Err(Overdraft(account))
            } else {
                Ok(())
            }
        });
        b.supply.define(&b.balances.filter(|&k, _| k != 0).sum());
    }
}

#[derive(Schema)]
struct App {
    #[schema]
    ledger: Ledger,
    name: Cell<String>,
    tags: Multiset<String>,
    #[view]
    rich: OrderedView<u64, i64>,
    #[view]
    tag_count: ValueView<i64>,
}

impl Declare for App {
    fn declare(b: &mut AppBuilder) {
        b.name.reject(|n| n.is_empty());
        // the outer schema builds on the nested view
        b.rich
            .define(&b.ledger.balances.filter(|_, &bal| bal >= 100));
        b.tag_count.define(&b.tags.distinct().count());
    }
}

#[test]
fn a_nested_ledger_enforces_its_constraints_and_publishes_changes() {
    let db = App::memory().build().unwrap();
    db.write(|tx| {
        tx.ledger.accounts.extend([0, 1, 2])?;
        tx.name.set("shop".into())?;
        tx.tags.add("jazz".into(), 2)?;
        tx.tags.add("blues".into(), 1)?;
        Ok(())
    })
    .unwrap();

    // subscribe from a snapshot, then change things
    let (balances, rich, supply) = db.read(|r| {
        (
            r.ledger.balances.subscribe(),
            r.rich.subscribe(),
            r.ledger.supply.subscribe(),
        )
    });

    db.write(|tx| {
        tx.ledger.transfers.push(Transfer {
            from: 0,
            to: 1,
            amount: 150,
        })?;
        tx.ledger.transfers.push(Transfer {
            from: 1,
            to: 2,
            amount: 30,
        })?;
        Ok(())
    })
    .unwrap();
    db.read(|r| {
        assert_eq!(r.ledger.balances.get(&1).unwrap(), Some(120));
        assert_eq!(r.ledger.supply.get().unwrap(), Some(150));
        assert_eq!(
            r.rich.desc().map(|r| r.unwrap()).collect::<Vec<_>>(),
            vec![(1, 120)]
        );
        assert_eq!(r.tag_count.get().unwrap(), Some(2));
        assert_eq!(r.tags.get(&"jazz".to_string()).unwrap(), 2);
        assert_eq!(r.name.get().unwrap().as_deref(), Some("shop"));
    });

    // an overdraft is refused with the application's error
    let err = db
        .write(|tx| {
            tx.ledger.transfers.push(Transfer {
                from: 2,
                to: 1,
                amount: 31,
            })
        })
        .unwrap_err();
    assert_eq!(err.rejected::<Overdraft>().map(|o| o.0), Some(2));
    // a transfer to nobody, and an empty name, are refused too
    assert!(matches!(
        db.write(|tx| tx.ledger.transfers.push(Transfer {
            from: 0,
            to: 9,
            amount: 1
        })),
        Err(Error::Rejected(_))
    ));
    assert_eq!(
        db.write(|tx| tx.name.set(String::new()))
            .unwrap_err()
            .rejected::<Rejected>()
            .map(|r| r.at.as_str()),
        Some("name")
    );
    // a write closure refuses its own transaction the same way
    let err = db
        .write(|tx| {
            tx.tags.add("rock".into(), 1)?;
            Err::<(), _>(Error::reject(Overdraft(7)))
        })
        .unwrap_err();
    assert_eq!(err.rejected::<Overdraft>().map(|o| o.0), Some(7));
    db.read(|r| assert_eq!(r.tag_count.get().unwrap(), Some(2)));

    // only the one commit was published
    let commit = balances.try_recv().unwrap().expect("the transfers");
    assert_eq!(
        commit.changes,
        vec![
            (0, Change::Insert(-150)),
            (1, Change::Insert(120)),
            (2, Change::Insert(30))
        ]
    );
    assert!(balances.try_recv().unwrap().is_none());
    assert_eq!(
        rich.try_recv().unwrap().unwrap().changes,
        vec![(1, Change::Insert(120))]
    );
    assert_eq!(
        supply.try_recv().unwrap().unwrap().changes,
        vec![Change::Insert(150)]
    );

    // retention: the log keeps the last two transfers, and the views
    // see the evicted one go
    drop(db);
}

#[derive(Schema)]
struct Log {
    readings: Events<(u8, i64)>,
    #[view]
    by_sensor: GroupView<u8, i64>,
    #[view]
    peak: MapView<u8, i64>,
    #[view]
    latest: MapView<u8, i64>,
}

impl Declare for Log {
    fn declare(b: &mut LogBuilder) {
        b.readings.retain(Retain::Count(3));
        let by = b.readings.key_by(|&(sensor, _)| sensor).map(|&(_, v)| v);
        b.by_sensor.define(&by);
        b.peak.define(&b.by_sensor.max());
        // the greatest id per sensor is the latest reading
        let stamped = b
            .readings
            .map_with(|id, &(s, v)| (s, id.raw(), v))
            .key_by(|&(s, _, _)| s)
            .map(|&(_, id, v)| (id, v));
        b.latest.define(&stamped.max().map(|&(_, v)| v));
    }
}

#[test]
fn an_event_log_evicts_through_its_views() {
    let db = Log::memory().build().unwrap();
    for (i, v) in [5i64, 9, 2, 7].into_iter().enumerate() {
        db.write(|tx| tx.readings.push((1, v)).map(|_| ())).unwrap();
        let kept = (i + 1).min(3);
        db.read(|r| assert_eq!(r.readings.len().unwrap(), kept));
    }
    db.read(|r| {
        // 5 was evicted: the peak is 9, kept rows are 9, 2, 7
        assert_eq!(r.peak.get(&1).unwrap(), Some(9));
        assert_eq!(r.latest.get(&1).unwrap(), Some(7));
        assert_eq!(r.by_sensor.get(&1).unwrap(), vec![(2, 1), (7, 1), (9, 1)]);
        assert_eq!(r.by_sensor.top(&1, 2).unwrap(), vec![(9, 1), (7, 1)]);
    });
    db.write(|tx| tx.readings.push((1, 1)).map(|_| ())).unwrap();
    db.read(|r| assert_eq!(r.peak.get(&1).unwrap(), Some(7)));
}

#[derive(Schema)]
struct People {
    people: Table<Id<(String, u8)>, (String, u8)>,
    #[view]
    by_name: MapView<String, (String, u8)>,
    #[view]
    adults: GroupView<u8, String>,
}

impl Declare for People {
    fn declare(b: &mut PeopleBuilder) {
        let by_name = b.people.unique_key(|(name, _)| name.clone());
        b.by_name.define(&by_name);
        b.adults.define(
            &b.people
                .key_by(|&(_, age)| age)
                .filter(|&age, _| age >= 18)
                .map(|(name, _)| name.clone())
                .top_k(2),
        );
    }
}

#[test]
fn unique_keys_are_checked_at_the_write() {
    let db = People::memory().build().unwrap();
    let ann = db.write(|tx| tx.people.insert(("ann".into(), 30))).unwrap();
    let err = db
        .write(|tx| tx.people.insert(("ann".into(), 31)))
        .unwrap_err();
    assert!(err.to_string().contains("unique key"), "{err}");
    db.write(|tx| {
        tx.people.insert(("bob".into(), 17))?;
        tx.people.insert(("cat".into(), 40))?;
        tx.people.insert(("dan".into(), 40))?;
        tx.people.insert(("eve".into(), 40))?;
        // renaming frees the old key
        tx.people.upsert(ann, ("anne".into(), 30))?;
        tx.people.insert(("ann".into(), 1))
    })
    .unwrap();
    db.read(|r| {
        assert_eq!(
            r.by_name.get(&"anne".to_string()).unwrap(),
            Some(("anne".into(), 30))
        );
        assert_eq!(
            r.by_name.get(&"ann".to_string()).unwrap(),
            Some(("ann".into(), 1))
        );
        assert_eq!(
            r.adults.get(&40).unwrap(),
            vec![("dan".to_string(), 1), ("eve".to_string(), 1)]
        );
        assert!(!r.adults.contains(&17).unwrap());
    });
}

#[derive(Schema)]
struct Tagged {
    tags: Multiset<(u64, String)>,
    #[view]
    by_item: GroupView<u64, String>,
    #[view]
    tag_count: MapView<u64, i64>,
}

impl Declare for Tagged {
    fn declare(b: &mut TaggedBuilder) {
        let distinct = b
            .tags
            .key_by(|&(item, _)| item)
            .map(|(_, t)| t.clone())
            .distinct();
        b.by_item.define(&distinct);
        b.tag_count.define(&b.by_item.count());
    }
}

#[test]
fn a_keyed_distinct_is_read_in_place_and_published_once() {
    let db = Tagged::memory().build().unwrap();
    let sub = db.read(|r| r.by_item.subscribe());
    db.write(|tx| {
        tx.tags.add((1, "jazz".into()), 2)?;
        tx.tags.add((1, "blues".into()), 1)?;
        tx.tags.add((2, "jazz".into()), 1)?;
        Ok(())
    })
    .unwrap();
    db.read(|r| {
        assert_eq!(
            r.by_item.get(&1).unwrap(),
            vec![("blues".to_string(), 1), ("jazz".to_string(), 1)]
        );
        assert!(r.by_item.contains(&2).unwrap());
        assert_eq!(r.tag_count.get(&1).unwrap(), Some(2));
    });
    let c = sub.try_recv().unwrap().unwrap();
    assert_eq!(c.changes.len(), 3);
    assert!(c.changes.iter().all(|(_, _, d)| *d == 1));

    // a second copy changes nothing visible; taking both away does
    db.write(|tx| tx.tags.add((1, "blues".into()), 1)).unwrap();
    assert!(sub.try_recv().unwrap().unwrap().changes.is_empty());
    db.write(|tx| tx.tags.remove((1, "jazz".into()), 2))
        .unwrap();
    assert_eq!(
        sub.try_recv().unwrap().unwrap().changes,
        vec![(1, "jazz".to_string(), -1)]
    );
    db.read(|r| {
        assert_eq!(r.by_item.get(&1).unwrap(), vec![("blues".to_string(), 1)]);
        assert_eq!(r.tag_count.get(&1).unwrap(), Some(1));
    });
}
