//! [`Semijoin`]: rows whose key the other side has ([`matching`]), or
//! lacks ([`without`]).
//!
//! [`matching`]: Semijoin::matching
//! [`without`]: Semijoin::without

use std::marker::PhantomData;

use crate::{
    Bag, Binary, Change, Delta,
    error::Result,
    graph::Fingerprint,
    tx::{KeyspaceId, WriteTx},
};

use super::{Arrangement, Key, Resolved, Take, by_key, count_of, read_group, update_counts};

/// Passes left rows `(k, a)` through while the right side holds at least
/// one row of key `k` ([`matching`](Self::matching), SQL's `EXISTS`), or
/// while it holds none ([`without`](Self::without), `NOT EXISTS`). Each
/// row passes at its own multiplicity, however many right rows match.
///
/// The left rows are arranged, so that when a key starts or stops
/// matching its rows can be emitted or retracted; the right side is kept
/// only as a count per key.
pub struct Semijoin<K, A, B> {
    anti: bool,
    left: Arrangement<K, A>,
    keys: KeyspaceId<K, Delta>,
    flips: Vec<(K, bool)>,
    group: Vec<(A, Delta)>,
    resolved: Resolved<K, A>,
    resolved_keys: Vec<(K, Change<Delta>)>,
    _b: PhantomData<fn() -> B>,
}

impl<K, A, B> Semijoin<K, A, B> {
    /// Left rows whose key the right side has.
    pub fn matching(left: Arrangement<K, A>, keys: KeyspaceId<K, Delta>) -> Self {
        Self::new(false, left, keys)
    }

    /// Left rows whose key the right side lacks.
    pub fn without(left: Arrangement<K, A>, keys: KeyspaceId<K, Delta>) -> Self {
        Self::new(true, left, keys)
    }

    fn new(anti: bool, left: Arrangement<K, A>, keys: KeyspaceId<K, Delta>) -> Self {
        Semijoin {
            anti,
            left,
            keys,
            flips: Vec::new(),
            group: Vec::new(),
            resolved: Vec::new(),
            resolved_keys: Vec::new(),
            _b: PhantomData,
        }
    }
}

impl<K, A, B> Semijoin<K, A, B>
where
    K: Key,
    A: Key + Ord,
    B: Ord + Send + 'static,
{
    fn wanted(&self, count: Delta) -> bool {
        (count > 0) != self.anti
    }

    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &Bag<(K, A)>,
        r: &Bag<(K, B)>,
        out: &mut Bag<(K, A)>,
    ) -> Result<()> {
        // keys whose right count changes, and so whether they pass
        let mut dkeys = Bag::new();
        for ((k, _), d) in r.diffs() {
            dkeys.push(k.clone(), d);
        }
        self.flips.clear();
        for (k, d) in dkeys.diffs() {
            let old = count_of(tx, self.keys, k)?;
            let new = old.checked_add(d).expect("count overflowed");
            let (was, now) = (self.wanted(old), self.wanted(new));
            if was != now {
                self.flips.push((k.clone(), now));
            }
        }

        // new left rows pass if their key does; flipped keys are
        // handled whole below
        for (k, rows) in by_key(l) {
            if self.flips.binary_search_by(|(f, _)| f.cmp(k)).is_ok() {
                continue;
            }
            if self.wanted(count_of(tx, self.keys, k)?) {
                for (a, d) in rows {
                    out.push((k.clone(), a.clone()), d);
                }
            }
        }

        // keys that stop passing retract their rows as they were
        for i in 0..self.flips.len() {
            let (k, on) = &self.flips[i];
            if !on {
                read_group(tx, self.left, k, Take::Whole, &mut self.group)?;
                for (a, d) in self.group.drain(..) {
                    out.push((k.clone(), a), -d);
                }
            }
        }

        let mut arranged = Bag::new();
        for ((k, a), d) in l.diffs() {
            arranged.push((k.clone(), Some(a.clone())), d);
        }
        update_counts(tx, self.left, arranged, &mut self.resolved, |_, _, _| {})?;
        update_counts(tx, self.keys, dkeys, &mut self.resolved_keys, |_, _, _| {})?;

        // keys that start passing emit their rows as they are now
        for i in 0..self.flips.len() {
            let (k, on) = &self.flips[i];
            if *on {
                read_group(tx, self.left, k, Take::Whole, &mut self.group)?;
                for (a, d) in self.group.drain(..) {
                    out.push((k.clone(), a), d);
                }
            }
        }
        Ok(())
    }
}

impl<K, A, B> Binary for Semijoin<K, A, B>
where
    K: Key,
    A: Key + Ord,
    B: Ord + Send + 'static,
{
    type Left = Bag<(K, A)>;
    type Right = Bag<(K, B)>;
    type Gives = Bag<(K, A)>;

    fn fingerprint(&self, fp: &mut Fingerprint) {
        fp.write_u64(self.anti as u64);
        fp.write_u64(self.left.raw());
        fp.write_u64(self.keys.raw());
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &Bag<(K, A)>, r: &Bag<(K, B)>, out: &mut Bag<(K, A)>) {
        let _ = self.run(tx, l, r, out);
    }
}
