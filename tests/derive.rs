//! `#[derive(Codec)]`: structs, tuples, units, enums and generics round
//! trip through both codecs, truncated input is an error, and a derived
//! type works as a key and a value of a stored keyspace.

use bog::{
    DurabilityPolicy, Mergeless, StoreBuilder,
    raw::KeyspaceId,
    schema::Codec as DeriveCodec,
    store::codec::{Codec, DecodeError, LayoutHash, OrderedCodec, ResidentSize},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Order {
    customer: String,
    cents: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Pair(u32, String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Unit;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
enum Event {
    Created { id: u64, name: String },
    Renamed(String, String),
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Wrap<T> {
    inner: Vec<T>,
    tag: u8,
}

/// The library under an alias, as a renamed dependency would see it.
mod renamed {
    pub use bog as thefold;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
#[bog(crate = "crate::renamed::thefold")]
struct Aliased {
    id: u64,
}

fn roundtrip<T: Codec + OrderedCodec + PartialEq + std::fmt::Debug>(value: &T) {
    let mut buf = Vec::new();
    value.encode(&mut buf);
    assert_eq!(buf.len(), value.encode_size());
    let mut input = buf.as_slice();
    assert_eq!(&T::decode(&mut input).unwrap(), value);
    assert!(input.is_empty(), "decode must consume exactly");

    let mut buf = Vec::new();
    value.encode_ordered(&mut buf);
    assert_eq!(buf.len(), value.encode_ordered_size());
    let mut input = buf.as_slice();
    assert_eq!(&T::decode_ordered(&mut input).unwrap(), value);
    assert!(input.is_empty());
}

/// Fields named like the generated code's own bindings (the buffers
/// `encode` writes to and `decode` reads from, an enum's tag): the
/// derive must not let them shadow its parameters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Tether {
    a: u32,
    b: u32,
    out: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
enum Shadowed {
    Named { b: i64, out: u8, tag: u8 },
    Tuple(u8, String),
}

#[test]
fn fields_named_like_the_generated_bindings() {
    roundtrip(&Tether {
        a: 1,
        b: 2,
        out: "x".into(),
    });
    roundtrip(&Shadowed::Named {
        b: -4,
        out: 5,
        tag: 6,
    });
    roundtrip(&Shadowed::Tuple(7, "y".into()));
}

#[test]
fn derived_types_roundtrip() {
    roundtrip(&Order {
        customer: "ada".into(),
        cents: -1250,
    });
    roundtrip(&Pair(7, "seven".into()));
    roundtrip(&Unit);
    roundtrip(&Wrap {
        inner: vec![1u32, 2, 3],
        tag: 9,
    });
    roundtrip(&Aliased { id: 7 });
    roundtrip(&Event::Created {
        id: 1,
        name: "x".into(),
    });
    roundtrip(&Event::Renamed("x".into(), "y".into()));
    roundtrip(&Event::Deleted);
    roundtrip(&vec![Pair(1, "a".into()), Pair(2, "b".into())]);
}

#[test]
fn ordered_encoding_sorts_like_the_derived_ord() {
    let mut rows = vec![
        Order {
            customer: "b".into(),
            cents: -5,
        },
        Order {
            customer: "a".into(),
            cents: 10,
        },
        Order {
            customer: "a".into(),
            cents: -10,
        },
        Order {
            customer: "ab".into(),
            cents: 0,
        },
    ];
    let mut encoded: Vec<(Vec<u8>, Order)> = rows
        .iter()
        .map(|o| {
            let mut b = Vec::new();
            o.encode_ordered(&mut b);
            (b, o.clone())
        })
        .collect();
    rows.sort();
    encoded.sort();
    let by_bytes: Vec<Order> = encoded.into_iter().map(|(_, o)| o).collect();
    assert_eq!(by_bytes, rows);
    assert!(
        Order {
            customer: "a".into(),
            cents: 1
        }
        .resident_size()
            >= std::mem::size_of::<Order>()
    );
}

#[test]
fn truncated_input_is_an_error() {
    let mut buf = Vec::new();
    Event::Renamed("abc".into(), "def".into()).encode(&mut buf);
    for cut in 0..buf.len() {
        let mut input = &buf[..cut];
        assert!(Event::decode(&mut input).is_err(), "cut at {cut}");
    }
    let mut input: &[u8] = &[42u8];
    assert_eq!(
        Event::decode(&mut input),
        Err(DecodeError::InvalidTag {
            what: "Event",
            tag: 42
        })
    );
}

#[test]
fn derived_types_key_and_fill_a_keyspace() {
    const ORDERS: KeyspaceId<Pair, Mergeless<Event>> = KeyspaceId::new(1);
    let store = StoreBuilder::new(DurabilityPolicy::NONE)
        .keyspace(ORDERS)
        .build()
        .unwrap();
    let mut tx = store.begin().unwrap();
    tx.keyspace(ORDERS)
        .set(Pair(2, "b".into()), Some(Event::Deleted))
        .unwrap();
    tx.keyspace(ORDERS)
        .set(
            Pair(1, "z".into()),
            Some(Event::Created {
                id: 3,
                name: "n".into(),
            }),
        )
        .unwrap();
    store.finish(tx).unwrap();
    let rows: Vec<(Pair, Event)> = store
        .snapshot()
        .range(ORDERS, ..)
        .map(|r| r.map(|(k, v)| (k.into_owned(), v.into_owned())).unwrap())
        .collect();
    assert_eq!(rows[0].0, Pair(1, "z".into()));
    assert_eq!(rows[1].1, Event::Deleted);
}

/// `Order` under another name, with its fields renamed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Purchase {
    buyer: String,
    amount: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Swapped {
    cents: i64,
    customer: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Wider {
    customer: String,
    cents: i64,
    note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Wrapped(Order);

/// `Event` with a variant appended.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
enum EventPlus {
    Created { id: u64, name: String },
    Renamed(String, String),
    Deleted,
    Restored,
}

/// `Event` with two variants' fields run together differently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
enum EventSplit {
    Created { id: u64 },
    Renamed(String, String, String),
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Tree {
    label: u8,
    children: Vec<Tree>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Bush {
    label: u8,
    children: Vec<Bush>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, DeriveCodec)]
struct Shrub {
    label: u8,
    children: Vec<Option<Shrub>>,
}

#[test]
fn a_layout_is_the_bytes_and_not_the_names() {
    let order = LayoutHash::of::<Order>();
    // names, field names, tuples and wrappers do not reach the bytes
    assert_eq!(order, LayoutHash::of::<Purchase>());
    assert_eq!(order, LayoutHash::of::<(String, i64)>());
    assert_eq!(order, LayoutHash::of::<Wrapped>());
    assert_eq!(order, LayoutHash::of::<(Box<str>, (i64, Unit))>());
    assert_eq!(
        LayoutHash::of::<bog::schema::Id<Order>>(),
        LayoutHash::of::<u64>()
    );
    // what a field is, and where, does
    assert_ne!(order, LayoutHash::of::<Swapped>());
    assert_ne!(order, LayoutHash::of::<Wider>());
    assert_ne!(order, LayoutHash::of::<(String, u64)>());
    assert_ne!(order, LayoutHash::of::<Pair>());
    // framing keeps parts from running together
    assert_ne!(
        LayoutHash::of::<Option<(u8, u8)>>(),
        LayoutHash::of::<(Option<u8>, u8)>()
    );
    assert_ne!(
        LayoutHash::of::<(Vec<u8>, u8)>(),
        LayoutHash::of::<Vec<(u8, u8)>>()
    );
    assert_ne!(LayoutHash::of::<[u8; 2]>(), LayoutHash::of::<[u8; 3]>());
    assert_ne!(LayoutHash::of::<[u8; 2]>(), LayoutHash::of::<(u8, u8)>());
    assert_eq!(
        LayoutHash::of::<Wrap<u8>>(),
        LayoutHash::of::<(Vec<u8>, u8)>()
    );
    assert_ne!(LayoutHash::of::<Wrap<u8>>(), LayoutHash::of::<Wrap<u16>>());
}

#[test]
fn an_enum_layout_is_its_variants_in_order() {
    let event = LayoutHash::of::<Event>();
    assert_ne!(event, LayoutHash::of::<EventPlus>());
    assert_ne!(event, LayoutHash::of::<EventSplit>());
    assert_ne!(event, LayoutHash::of::<(u8, u64, String)>());
}

#[test]
fn a_recursive_type_has_a_layout() {
    let tree = LayoutHash::of::<Tree>();
    assert_eq!(tree, LayoutHash::of::<Bush>());
    assert_ne!(tree, LayoutHash::of::<Shrub>());
    assert_ne!(tree, LayoutHash::of::<(u8, Vec<u8>)>());
}

#[test]
fn a_layout_is_fixed_across_builds() {
    // recorded on disk: a change here strands every existing store
    assert_eq!(LayoutHash::of::<u64>(), 0x90b5_6c24_3331_93d4);
    assert_eq!(LayoutHash::of::<Order>(), 0x9b17_b838_ed55_e96f);
    assert_eq!(LayoutHash::of::<Event>(), 0x6c69_8769_ae33_d670);
}

/// A schema's sources are held to their layouts across opens.
mod reopen {
    use bog::{
        Error,
        schema::{Codec, Declare, Schema, Table},
    };

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Codec)]
    pub struct User {
        pub name: String,
        pub age: u8,
    }

    /// `User`, renamed.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Codec)]
    pub struct Member {
        pub handle: String,
        pub years: u8,
    }

    /// `User` with a field added.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Codec)]
    pub struct UserWithEmail {
        pub name: String,
        pub age: u8,
        pub email: Option<String>,
    }

    #[derive(Schema)]
    pub struct Before {
        users: Table<u64, User>,
    }
    impl Declare for Before {
        fn declare(_: &mut BeforeBuilder) {}
    }

    #[derive(Schema)]
    pub struct Renamed {
        users: Table<u64, Member>,
    }
    impl Declare for Renamed {
        fn declare(_: &mut RenamedBuilder) {}
    }

    #[derive(Schema)]
    pub struct Widened {
        users: Table<u64, UserWithEmail>,
    }
    impl Declare for Widened {
        fn declare(_: &mut WidenedBuilder) {}
    }

    #[test]
    fn a_renamed_type_opens_and_a_reshaped_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("bog-reopen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let db = Before::open(&dir).build().unwrap();
            db.write(|w| {
                w.users.upsert(
                    1,
                    User {
                        name: "ada".into(),
                        age: 36,
                    },
                )
            })
            .unwrap();
            db.close().unwrap();
        }
        {
            let db = Renamed::open(&dir).build().unwrap();
            let ada = db.read(|r| r.users.get(&1)).unwrap();
            assert_eq!(
                ada,
                Some(Member {
                    handle: "ada".into(),
                    years: 36
                })
            );
            db.close().unwrap();
        }
        let widened = Widened::open(&dir).build();
        assert!(matches!(widened, Err(Error::KeyspaceLayout { .. })));
        // and the refusal left the store as it was
        let db = Before::open(&dir).build().unwrap();
        assert_eq!(db.read(|r| r.users.len()).unwrap(), 1);
        db.close().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
