//! Full-text search as a view: BM25 ranking, boolean matching, prefix
//! completion, retraction correctness, recovery, a tokenizer as a type,
//! and block-max top-k against brute force.

use fold3::schema::{Codec, Declare, Id, MapView, Schema, SearchView, Table, Tokenizer, intern};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Codec)]
struct Album {
    title: String,
    blurb: String,
}

#[derive(Schema)]
struct Catalog {
    albums: Table<Id<Album>, Album>,
    #[view]
    search: SearchView<Id<Album>>,
    /// Built on the search view's rows: how many albums each term
    /// appears in.
    #[view]
    term_docs: MapView<String, i64>,
}

impl Declare for Catalog {
    fn declare(b: &mut CatalogBuilder) {
        b.search
            .define(&b.albums.map(|a| format!("{} {}", a.title, a.blurb)));
        b.term_docs.define(&b.search.distinct().count());
    }
}

fn album(title: &str, blurb: &str) -> Album {
    Album {
        title: title.into(),
        blurb: blurb.into(),
    }
}

fn seed(db: &fold3::schema::Db<Catalog>) -> Vec<Id<Album>> {
    db.write(|tx| {
        tx.albums.insert_all([
            album("Blue Train", "hard bop, Coltrane's Blue Note classic"),
            album("Kind of Blue", "modal jazz; the best selling jazz record"),
            album("A Love Supreme", "Coltrane's spiritual suite"),
            album("Giant Steps", "Coltrane, sheets of sound"),
        ])
    })
    .unwrap()
}

#[test]
fn bm25_ranks_and_boolean_matches() {
    let db = Catalog::memory().build().unwrap();
    let ids = seed(&db);
    let (blue_train, kind_of_blue, supreme, giant) = (ids[0], ids[1], ids[2], ids[3]);

    db.read(|r| {
        let s = &r.search;
        assert_eq!(s.len().unwrap(), 4);
        assert_eq!(s.doc_frequency("blue").unwrap(), 2);
        assert_eq!(s.doc_frequency("coltrane").unwrap(), 3);
        assert_eq!(s.doc_frequency("nope").unwrap(), 0);
        assert_eq!(s.doc_len(&giant).unwrap(), 6);

        // term frequency is the posting's amount
        assert_eq!(
            s.postings("blue").unwrap(),
            vec![(blue_train, 2), (kind_of_blue, 1)]
        );

        // "blue" twice in Blue Train beats once in Kind of Blue
        let hits = s.search("blue", 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, blue_train);
        assert_eq!(hits[1].0, kind_of_blue);
        assert!(hits[0].1 > hits[1].1);

        // a rare term outweighs a common one; every matching doc scores
        let hits = s.search("coltrane supreme", 10).unwrap();
        assert_eq!(hits[0].0, supreme);
        assert_eq!(hits.len(), 3);
        assert_eq!(s.search("coltrane supreme", 1).unwrap().len(), 1);

        // boolean AND
        assert_eq!(s.matching_all("blue coltrane").unwrap(), vec![blue_train]);
        assert_eq!(s.matching_all("blue jazz").unwrap(), vec![kind_of_blue]);
        assert!(s.matching_all("blue nope").unwrap().is_empty());
        assert!(s.matching_all("").unwrap().is_empty());

        // completion by document frequency, then alphabetically
        assert_eq!(
            s.complete("co", 10).unwrap(),
            vec![("coltrane".to_string(), 3)]
        );
        let b: Vec<String> = s
            .complete("b", 10)
            .unwrap()
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(b, vec!["blue", "best", "bop"]);
        assert_eq!(s.complete("b", 1).unwrap().len(), 1);

        // the index is a stream other views build on
        assert_eq!(r.term_docs.get(&"coltrane".to_string()).unwrap(), Some(3));
        assert_eq!(r.term_docs.get(&"blue".to_string()).unwrap(), Some(2));
    });

    // inside a transaction, the index reflects the writes so far
    db.write(|tx| {
        tx.albums.insert(album("Blue Moon", "standards"))?;
        assert_eq!(tx.search.doc_frequency("blue")?, 3);
        assert_eq!(tx.search.search("moon", 5)?.len(), 1);
        assert_eq!(tx.read().search.doc_frequency("blue")?, 2);
        Ok(())
    })
    .unwrap();
}

#[test]
fn updates_and_removals_retract_cleanly() {
    let db = Catalog::memory().build().unwrap();
    let ids = seed(&db);
    let blue_train = ids[0];

    // retitle: old tokens vanish, new ones appear, lengths follow
    db.write(|tx| {
        tx.albums
            .upsert(blue_train, album("Lush Life", "Coltrane ballads"))
    })
    .unwrap();
    db.read(|r| {
        assert_eq!(r.search.doc_frequency("blue").unwrap(), 1);
        assert_eq!(r.search.doc_frequency("train").unwrap(), 0);
        assert_eq!(r.search.doc_frequency("lush").unwrap(), 1);
        assert_eq!(r.search.doc_len(&blue_train).unwrap(), 4);
        assert_eq!(r.search.len().unwrap(), 4);
        assert_eq!(r.term_docs.get(&"train".to_string()).unwrap(), None);
    });

    // remove every document: every counter cancels to nothing
    db.write(|tx| tx.albums.clear()).unwrap();
    db.read(|r| {
        assert_eq!(r.search.len().unwrap(), 0);
        assert_eq!(r.search.avg_len().unwrap(), 0.0);
        assert!(r.search.search("coltrane", 5).unwrap().is_empty());
        assert!(r.search.complete("", 5).unwrap().is_empty());
        assert_eq!(r.search.doc_len(&blue_train).unwrap(), 0);
        assert!(r.term_docs.is_empty().unwrap());
    });
}

#[test]
fn index_persists_recovers_and_rebuilds() {
    let dir = std::env::temp_dir().join(format!("fold3-search-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let supreme = {
        let db = Catalog::open(&dir).build().unwrap();
        let ids = seed(&db);
        db.close().unwrap();
        ids[2]
    };
    for version in [0, 1] {
        let db = Catalog::open(&dir).version(version).build().unwrap();
        db.read(|r| {
            assert_eq!(r.search.len().unwrap(), 4);
            assert_eq!(r.search.search("supreme", 1).unwrap()[0].0, supreme);
            assert_eq!(r.term_docs.get(&"coltrane".to_string()).unwrap(), Some(3));
        });
        // the block index survives in the checkpoint tables
        db.checkpoint().unwrap();
        db.read(|r| assert_eq!(r.search.search("blue", 1).unwrap().len(), 1));
        drop(db);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A custom tokenizer is a type; its name is part of the keyspace names.
#[test]
fn tokenizer_is_a_type() {
    struct Words;
    impl Tokenizer for Words {
        const NAME: &'static str = "words";
        fn tokenize(text: &str) -> Vec<String> {
            text.split_whitespace().map(String::from).collect()
        }
    }
    #[derive(Schema)]
    struct Exact {
        docs: Table<u64, String>,
        #[view]
        search: SearchView<u64, Words>,
    }
    impl Declare for Exact {
        fn declare(b: &mut ExactBuilder) {
            b.search.define(&b.docs);
        }
    }
    let db = Exact::memory().build().unwrap();
    db.write(|tx| tx.docs.upsert(1, "Blue Train".into()))
        .unwrap();
    db.read(|r| {
        assert_eq!(r.search.doc_frequency("Blue").unwrap(), 1);
        assert_eq!(r.search.doc_frequency("blue").unwrap(), 0);
    });
}

#[test]
fn search_postings_are_subscribable() {
    let db = Catalog::memory().build().unwrap();
    let ids = seed(&db);
    let sub = db.read(|r| r.search.subscribe());
    db.write(|tx| tx.albums.upsert(ids[3], album("Giant Steps", "Coltrane")))
        .unwrap();
    let c = sub.recv().unwrap();
    // the document was re-indexed at a new length: kept terms leave at
    // the old length and arrive at the new, netting to nothing
    let mut net: std::collections::BTreeMap<u64, i64> = Default::default();
    for (t, _, d) in &c.changes {
        *net.entry(*t).or_insert(0) += d;
    }
    let retracted: Vec<u64> = net
        .iter()
        .filter(|(_, d)| **d < 0)
        .map(|(t, _)| *t)
        .collect();
    let mut want: Vec<u64> = ["of", "sheets", "sound"]
        .iter()
        .map(|t| intern(t))
        .collect();
    want.sort();
    assert_eq!(retracted, want);
    assert!(c.changes.iter().all(|(_, doc, _)| *doc == ids[3]));
}

/// Block-max top-k must agree with brute-force scoring exactly: same
/// documents, same scores, same order, across runs, updates, deletions,
/// and every k, with documents of very different lengths.
#[test]
fn block_max_top_k_matches_brute_force() {
    #[derive(Schema)]
    struct Corpus {
        docs: Table<u64, String>,
        #[view]
        search: SearchView<u64>,
    }
    impl Declare for Corpus {
        fn declare(b: &mut CorpusBuilder) {
            b.search.define(&b.docs);
        }
    }
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let words = [
        "alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta",
    ];
    let text = |rng: &mut dyn FnMut() -> u64| {
        let len = 1 + (rng() % 60) as usize;
        (0..len)
            .map(|_| {
                // skewed: alpha is very common, theta rare
                let u = (rng() % 1000) as f64 / 1000.0;
                words[((u * u * u) * words.len() as f64) as usize]
            })
            .collect::<Vec<_>>()
            .join(" ")
    };

    let dir = std::env::temp_dir().join(format!("fold3-search-topk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Corpus::open(&dir).build().unwrap();
    // one large commit, checkpointed, then small ones: updates and
    // deletions land in younger runs than the rows they touch
    db.write(|tx| {
        for id in 0..2_000u64 {
            tx.docs.upsert(id, text(&mut rng))?;
        }
        Ok(())
    })
    .unwrap();
    db.checkpoint().unwrap();
    for _ in 0..12u64 {
        db.write(|tx| {
            for _ in 0..40 {
                let id = rng() % 2_000;
                match rng() % 10 {
                    0 => tx.docs.remove(&id).map(|_| ())?,
                    _ => tx.docs.upsert(id, text(&mut rng))?,
                }
            }
            Ok(())
        })
        .unwrap();
    }

    db.read(|r| {
        let n = r.search.len().unwrap() as f64;
        let avg = r.search.avg_len().unwrap();
        let p = fold3::schema::Bm25::default();
        let idf = |df: usize| ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
        for query in [
            "alpha",
            "theta",
            "alpha theta",
            "beta gamma delta",
            "eps zeta eta theta alpha",
        ] {
            // brute force over every posting of every term
            let mut scores: std::collections::BTreeMap<u64, f64> = Default::default();
            for term in query.split(' ') {
                let df = r.search.doc_frequency(term).unwrap();
                for (doc, tf) in r.search.postings(term).unwrap() {
                    let dl = r.search.doc_len(&doc).unwrap() as f64;
                    let norm = p.k1 * (1.0 - p.b + p.b * dl / avg);
                    let s = idf(df) * (tf as f64 * (p.k1 + 1.0)) / (tf as f64 + norm);
                    *scores.entry(doc).or_insert(0.0) += s;
                }
            }
            let mut want: Vec<(u64, f64)> = scores.into_iter().collect();
            want.sort_by(|(da, sa), (db, sb)| sb.total_cmp(sa).then_with(|| da.cmp(db)));
            for k in [1, 3, 10, 50, 300] {
                let got = r.search.search(query, k).unwrap();
                let want: Vec<(u64, f64)> = want.iter().take(k).cloned().collect();
                assert_eq!(got.len(), want.len(), "{query} k={k}");
                for ((gd, gs), (wd, ws)) in got.iter().zip(&want) {
                    assert_eq!(gd, wd, "{query} k={k}");
                    assert!((gs - ws).abs() < 1e-9, "{query} k={k}: {gs} vs {ws}");
                }
            }
        }
    });
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
