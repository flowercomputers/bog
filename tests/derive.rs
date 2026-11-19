//! `#[derive(Codec)]`: structs, tuples, units, enums and generics round
//! trip through both codecs, truncated input is an error, and a derived
//! type works as a key and a value of a stored keyspace.

use bog::{
    DurabilityPolicy, Mergeless, StoreBuilder,
    raw::KeyspaceId,
    schema::Codec as DeriveCodec,
    store::codec::{Codec, DecodeError, OrderedCodec, ResidentSize},
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
