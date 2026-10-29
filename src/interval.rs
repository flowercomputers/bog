use crate::{
    interval::Bound::*,
    tx::{
        KeySpace, Merge,
        ValOp::{self, Modify},
    },
};

use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[derive(PartialEq)]
pub enum Bound<T> {
    Unbounded,
    Inclusive(T),
    Exclusive(T),
}

#[derive(PartialEq)]
pub struct Bounds<T> {
    /// None is unbounded
    /// Some is inclusive
    lo: Option<T>,
    hi: Bound<T>,
}

impl<T> From<RangeFull> for Bounds<T> {
    fn from(_: RangeFull) -> Self {
        Bounds {
            lo: None,
            hi: Unbounded,
        }
    }
}
impl<T> From<Range<T>> for Bounds<T> {
    fn from(range: Range<T>) -> Self {
        Bounds {
            lo: Some(range.start),
            hi: Exclusive(range.end),
        }
    }
}
impl<T> From<RangeInclusive<T>> for Bounds<T> {
    fn from(range: RangeInclusive<T>) -> Self {
        let (start, last) = range.into_inner();
        Bounds {
            lo: Some(start),
            hi: Inclusive(last),
        }
    }
}
impl<T> From<RangeFrom<T>> for Bounds<T> {
    fn from(range: RangeFrom<T>) -> Self {
        Bounds {
            lo: Some(range.start),
            hi: Unbounded,
        }
    }
}
impl<T> From<RangeTo<T>> for Bounds<T> {
    fn from(range: RangeTo<T>) -> Self {
        Bounds {
            lo: None,
            hi: Exclusive(range.end),
        }
    }
}
impl<T> From<RangeToInclusive<T>> for Bounds<T> {
    fn from(range: RangeToInclusive<T>) -> Self {
        Bounds {
            lo: None,
            hi: Inclusive(range.end),
        }
    }
}

struct Boundary<K, V> {
    /// ValOp applied to this range
    below: Option<V>,
    /// If matching is Some, this interval is inclusive of K
    /// All other Boundary<K,V> are exclusive.
    matching: Option<V>,
    /// Highest key in this range
    key: K,
}

pub struct BoundaryMap<KS: KeySpace> {
    inner: Vec<Boundary<KS::Key, ValOp<KS::Merge>>>,
    right: Option<ValOp<KS::Merge>>,
}

impl<KS: KeySpace> BoundaryMap<KS> {
    pub fn new() -> Self {
        BoundaryMap {
            inner: Vec::new(),
            right: None,
        }
    }

    /// Value of the open range whose upper boundary is `inner[idx]`
    /// (i.e. `inner[idx].below`), or `right` if `idx` is past the end.
    fn range_at(&self, idx: usize) -> Option<ValOp<KS::Merge>> {
        self.inner
            .get(idx)
            .map_or_else(|| self.right.clone(), |iv| iv.below.clone())
    }

    /// Apply `vop` to exactly one key.
    pub fn absorb(&mut self, key: KS::Key, vop: ValOp<KS::Merge>) {
        let pp = self.inner.partition_point(|i| i.key < key);
        match self.inner.get(pp) {
            Some(iv) if iv.key == key => {
                if iv.matching.is_none() {
                    let above = self
                        .inner
                        .get(pp + 1)
                        .map(|i| i.below.clone())
                        .unwrap_or_else(|| self.right.clone());
                    self.inner[pp].matching = above;
                }
            }
            _ => {
                let old = self
                    .inner
                    .get(pp)
                    .map(|i| i.below.clone())
                    .unwrap_or_else(|| self.right.clone());
                self.inner.insert(
                    pp,
                    Boundary {
                        key,
                        below: old.clone(),
                        matching: old,
                    },
                );
            }
        }
        let slot = &mut self.inner[pp].matching;
        if matches!(vop, Modify(..)) {
            absorb_or_set(slot, vop);
        } else {
            *slot = Some(vop);
        }
    }

    pub fn absorb_range(&mut self, range: impl Into<Bounds<KS::Key>>, vop: ValOp<KS::Merge>) {
        // todo: automagically combine runs

        let Bounds { lo, hi } = range.into();

        // degenerate ranges
        if let Some(l) = &lo {
            match &hi {
                Inclusive(h) if l > h => return,
                Inclusive(h) if l == h => return self.absorb(lo.unwrap(), vop),
                Exclusive(h) if l >= h => return,
                _ => {}
            }
        }
        let hi_inclusive = matches!(hi, Inclusive(_));

        // ---- Low end --------------------------------------------------
        // After this block, `inner[start..]` begins strictly above `lo`.
        // `low_point` is set iff the key `lo` is stored with its own value
        // (matching: Some) and so must be updated separately. Otherwise the
        // point either doesn't exist yet or has matching: None, in which
        // case it is carried by `inner[start].below` and the main loop
        // covers it.
        let mut low_point = None;
        let start = match lo {
            None => 0,
            Some(low) => {
                let pp = self.inner.partition_point(|i| i.key < low);
                match self.inner.get(pp) {
                    Some(iv) if iv.key == low => {
                        if iv.matching.is_some() {
                            low_point = Some(pp);
                        }
                    }
                    _ => {
                        let old = self.range_at(pp);
                        self.inner.insert(
                            pp,
                            Boundary {
                                key: low,
                                below: old,
                                matching: None,
                            },
                        );
                    }
                }
                pp + 1
            }
        };

        // ---- High end -------------------------------------------------
        // After this block, `inner[end]` is the interval keyed at `hi`
        // (or `end == len` if unbounded) with its own `matching`, so the
        // range `inner[end].below` is the last range covered, and
        // `inner[end].matching` is covered iff `hi_inclusive`.
        let end = match hi {
            Unbounded => self.inner.len(),
            Inclusive(high) | Exclusive(high) => {
                let pp = self.inner.partition_point(|i| i.key < high);
                match self.inner.get(pp) {
                    Some(iv) if iv.key == high => {
                        if hi_inclusive && iv.matching.is_none() {
                            // Split the point off the range above so it can
                            // be modified independently.
                            let above = self.range_at(pp + 1);
                            self.inner[pp].matching = above;
                        }
                    }
                    _ => {
                        let old = self.range_at(pp);
                        self.inner.insert(
                            pp,
                            Boundary {
                                key: high,
                                below: old.clone(),
                                matching: old,
                            },
                        );
                    }
                }
                pp
            }
        };
        debug_assert!(start <= end);

        // ---- Apply ----------------------------------------------------
        if matches!(vop, Modify(..)) {
            if let Some(i) = low_point {
                absorb_or_set(&mut self.inner[i].matching, vop.clone());
            }
            for iv in &mut self.inner[start..end] {
                absorb_or_set(&mut iv.below, vop.clone());
                if let Some(m) = &mut iv.matching {
                    m.absorb(vop.clone());
                }
            }
            match self.inner.get_mut(end) {
                Some(iv) => {
                    if hi_inclusive {
                        absorb_or_set(&mut iv.matching, vop.clone());
                    }
                    absorb_or_set(&mut iv.below, vop);
                }
                None => absorb_or_set(&mut self.right, vop),
            }
        } else {
            if let Some(i) = low_point {
                // Point joins the replaced range above it.
                self.inner[i].matching = None;
            }
            self.inner.drain(start..end);
            match self.inner.get_mut(start) {
                Some(iv) => {
                    if hi_inclusive {
                        iv.matching = Some(vop.clone());
                    }
                    iv.below = Some(vop);
                }
                None => self.right = Some(vop),
            }
        }

        debug_assert!(self.inner.is_sorted_by(|a, b| a.key < b.key));
    }

    pub fn get(&self, key: KS::Key) -> Option<&ValOp<KS::Merge>> {
        let pp = self.inner.partition_point(|interval| interval.key < key);
        match self.inner.get(pp) {
            Some(iv) if iv.key == key => iv.matching.as_ref().or_else(|| {
                self.inner
                    .get(pp + 1)
                    .map_or(self.right.as_ref(), |n| n.below.as_ref())
            }),
            Some(iv) => iv.below.as_ref(),
            None => self.right.as_ref(),
        }
    }
}

fn absorb_or_set<M: Merge>(old: &mut Option<ValOp<M>>, new: ValOp<M>) {
    if let Some(old) = old {
        old.absorb(new);
    } else {
        *old = Some(new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Harness - adapt these to your real types.
    // ------------------------------------------------------------------

    /// A `KeySpace` with `Key = u8` and whatever `Merge` type is convenient.
    /// A counter (value = i64, modify = add a delta) works well.
    struct TestKs;
    impl KeySpace for TestKs {
        type Key = u8;
        type Merge = i64;
        fn ident() -> String {
            "testks".into()
        }
    }

    type Vop = ValOp<<TestKs as KeySpace>::Merge>;

    /// A `ValOp` that replaces whatever was there, tagged with `n`.
    fn set(n: u64) -> Vop {
        ValOp::Store(n as i64)
    }

    /// A `ValOp::Modify` tagged with `n`.
    fn modify(n: u64) -> Vop {
        ValOp::Modify(n as i64)
    }

    // ------------------------------------------------------------------
    // Reference model: one slot per key, updated key-by-key.
    // ------------------------------------------------------------------

    const KEYS: usize = u8::MAX as usize + 1;

    struct Model(Vec<Option<Vop>>);

    impl Model {
        fn new() -> Self {
            Model(vec![None; KEYS])
        }

        fn apply_key(&mut self, k: u8, vop: &Vop) {
            let slot = &mut self.0[k as usize];
            if matches!(vop, Modify(..)) {
                absorb_or_set(slot, vop.clone());
            } else {
                *slot = Some(vop.clone());
            }
        }

        fn apply(&mut self, op: &Op) {
            match op {
                Op::Point(k, v) => self.apply_key(*k, v),
                Op::Range(lo, hi, v) => {
                    // Iterate over the keys the range *should* cover.
                    let lo = lo.unwrap_or(0) as usize;
                    let hi = match hi {
                        Unbounded => KEYS,
                        Inclusive(h) => *h as usize + 1,
                        Exclusive(h) => *h as usize,
                    };
                    for k in lo..hi {
                        self.apply_key(k as u8, v);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Operations and how to apply them to the real map.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug)]
    enum Op {
        Point(u8, Vop),
        /// Low is inclusive-or-unbounded, matching `Bounds::lo`.
        Range(Option<u8>, Bound<u8>, Vop),
    }

    impl std::fmt::Debug for Bound<u8> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Unbounded => write!(f, "Unbounded"),
                Inclusive(k) => write!(f, "Inclusive({k})"),
                Exclusive(k) => write!(f, "Exclusive({k})"),
            }
        }
    }

    impl Clone for Bound<u8> {
        fn clone(&self) -> Self {
            match self {
                Unbounded => Unbounded,
                Inclusive(k) => Inclusive(*k),
                Exclusive(k) => Exclusive(*k),
            }
        }
    }

    fn apply_real(map: &mut BoundaryMap<TestKs>, op: &Op) {
        match op {
            Op::Point(k, v) => map.absorb(*k, v.clone()),
            Op::Range(lo, hi, v) => {
                // Go through the public `From` impls where possible so those
                // get exercised too; `Bounds` fields are private to callers
                // but visible here via `super::*`.
                let v = v.clone();
                match (lo, hi) {
                    (None, Unbounded) => map.absorb_range(.., v),
                    (None, Inclusive(h)) => map.absorb_range(..=*h, v),
                    (None, Exclusive(h)) => map.absorb_range(..*h, v),
                    (Some(l), Unbounded) => map.absorb_range(*l.., v),
                    (Some(l), Inclusive(h)) => map.absorb_range(*l..=*h, v),
                    (Some(l), Exclusive(h)) => map.absorb_range(*l..*h, v),
                }
            }
        }
    }

    fn check(map: &BoundaryMap<TestKs>, model: &Model, context: &dyn Fn() -> String) {
        for k in 0..KEYS {
            let got = map.get(k as u8);
            let want = model.0[k].as_ref();
            assert_eq!(
                got,
                want,
                "mismatch at key {k}: got {got:?}, want {want:?}\n{}",
                context()
            );
        }
        // Structural sanity: keys strictly increasing.
        assert!(
            map.inner.windows(2).all(|w| w[0].key < w[1].key),
            "keys not strictly sorted\n{}",
            context()
        );
    }

    // ------------------------------------------------------------------
    // Tiny deterministic RNG so there's no test dependency.
    // ------------------------------------------------------------------

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Keys drawn from a small space so ranges collide and nest a lot.
        fn key(&mut self) -> u8 {
            self.below(12) as u8
        }

        fn vop(&mut self, tag: u64) -> Vop {
            if self.below(2) == 0 {
                set(tag)
            } else {
                modify(tag)
            }
        }

        fn op(&mut self, tag: u64) -> Op {
            let v = self.vop(tag);
            match self.below(8) {
                0 => Op::Point(self.key(), v),
                1 => Op::Range(None, Unbounded, v),
                2 => Op::Range(None, Inclusive(self.key()), v),
                3 => Op::Range(None, Exclusive(self.key()), v),
                4 => Op::Range(Some(self.key()), Unbounded, v),
                5 => Op::Range(Some(self.key()), Inclusive(self.key()), v),
                // Weighted towards the two most common shapes.
                _ => Op::Range(Some(self.key()), Exclusive(self.key()), v),
            }
        }
    }

    // ------------------------------------------------------------------
    // Randomised comparison against the model.
    // ------------------------------------------------------------------

    #[test]
    fn random_ops_match_model() {
        const SEEDS: u64 = 20_000;
        const OPS_PER_SEED: usize = 40;

        for seed in 1..=SEEDS {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut map = BoundaryMap::<TestKs>::new();
            let mut model = Model::new();
            let mut log: Vec<Op> = Vec::new();

            for i in 0..OPS_PER_SEED {
                let op = rng.op(i as u64);
                log.push(op.clone());
                apply_real(&mut map, &op);
                model.apply(&op);
                check(&map, &model, &|| {
                    format!("seed {seed}, after op {i}\nops so far:\n{:#?}", log)
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // Targeted regressions for the bugs found in review.
    // ------------------------------------------------------------------

    fn assert_keys(map: &BoundaryMap<TestKs>, expected: &[(u8, Option<Vop>)]) {
        for (k, want) in expected {
            assert_eq!(map.get(*k), want.as_ref(), "at key {k}");
        }
    }

    #[test]
    fn empty_map_is_empty() {
        let map = BoundaryMap::<TestKs>::new();
        for k in 0..=u8::MAX {
            assert_eq!(map.get(k), None);
        }
    }

    #[test]
    fn get_sees_right_past_last_key() {
        // `get` used to return None for any key above the last stored key.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(.., set(1));
        assert_keys(
            &map,
            &[(0, Some(set(1))), (100, Some(set(1))), (255, Some(set(1)))],
        );
    }

    #[test]
    fn get_follows_matching_none_to_range_above() {
        // `5..10` inserts key 5 with `matching: None`; `get(5)` must look
        // at the range above rather than reporting None.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(5..10, set(1));
        assert_keys(
            &map,
            &[(4, None), (5, Some(set(1))), (9, Some(set(1))), (10, None)],
        );
    }

    #[test]
    fn half_open_range_includes_low_key() {
        // The old exclusive path skipped the low key entirely.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(.., set(1));
        map.absorb_range(5..10, set(2));
        assert_keys(
            &map,
            &[
                (4, Some(set(1))),
                (5, Some(set(2))),
                (9, Some(set(2))),
                (10, Some(set(1))),
            ],
        );

        // Same thing when the low key already exists as its own point.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb(5, set(1));
        map.absorb_range(5..10, set(2));
        assert_keys(&map, &[(5, Some(set(2)))]);

        // And for `lo..`.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb(5, set(1));
        map.absorb_range(5.., set(2));
        assert_keys(&map, &[(4, None), (5, Some(set(2))), (255, Some(set(2)))]);
    }

    #[test]
    fn unbounded_modify_applies_once() {
        // The old `..` path absorbed into `right` twice.
        let mut map = BoundaryMap::<TestKs>::new();
        let mut want: Option<Vop> = None;
        absorb_or_set(&mut want, modify(1));
        map.absorb_range(.., modify(1));
        assert_eq!(map.right, want);
        assert_eq!(map.get(0), want.as_ref());

        // Also on top of an existing value.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(.., set(7));
        map.absorb_range(.., modify(1));
        let mut want = Some(set(7));
        absorb_or_set(&mut want, modify(1));
        assert_eq!(map.get(0), want.as_ref());
        assert_eq!(map.get(255), want.as_ref());
    }

    #[test]
    fn degenerate_ranges() {
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(.., set(1));

        map.absorb_range(5..5, set(2)); // empty
        map.absorb_range(7..3, set(2)); // reversed
        map.absorb_range(9..=3, set(2)); // reversed inclusive
        for k in 0..=u8::MAX {
            assert_eq!(map.get(k), Some(&set(1)), "at key {k}");
        }

        map.absorb_range(5..=5, set(2)); // single point
        assert_keys(
            &map,
            &[(4, Some(set(1))), (5, Some(set(2))), (6, Some(set(1)))],
        );
    }

    #[test]
    fn inclusive_high_splits_existing_point() {
        // Key 10 exists only as the exclusive end of an earlier range
        // (matching: None); an inclusive range ending at 10 must split it
        // off so the range above is not modified.
        let mut map = BoundaryMap::<TestKs>::new();
        map.absorb_range(.., set(1));
        map.absorb_range(0..10, set(2));
        map.absorb_range(5..=10, modify(3));

        let mut want_10 = Some(set(1));
        absorb_or_set(&mut want_10, modify(3));
        let mut want_5 = Some(set(2));
        absorb_or_set(&mut want_5, modify(3));

        assert_keys(
            &map,
            &[
                (4, Some(set(2))),
                (5, want_5),
                (10, want_10),
                (11, Some(set(1))),
            ],
        );
    }

    #[test]
    fn replace_collapses_covered_points() {
        // Points inside a replaced range should vanish from `inner`, not
        // merely be overwritten, so the map doesn't grow without bound.
        let mut map = BoundaryMap::<TestKs>::new();
        for k in 0..20 {
            map.absorb(k, set(k as u64));
        }
        let before = map.inner.len();
        map.absorb_range(2..18, set(99));
        assert!(map.inner.len() < before);
        assert_keys(
            &map,
            &[
                (1, Some(set(1))),
                (2, Some(set(99))),
                (17, Some(set(99))),
                (18, Some(set(18))),
            ],
        );
    }
}
