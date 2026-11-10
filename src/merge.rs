use std::borrow::Cow;

use bytes::Bytes;

use crate::Delta;

pub enum ValOp<M: Merge> {
    Store(M::Value),
    // todo: Remove modify
    Modify(M::Diff),
    Delete,
}

pub trait Merge: 'static {
    type Diff: 'static + Clone + Send;
    type Value: Clone + Send;

    fn base() -> Self::Value;
    fn merge<'a>(val: &mut Self::Value, diffs: impl Iterator<Item = Cow<'a, Self::Diff>>);
    fn combine<'a>(old: &mut Self::Diff, new: impl Iterator<Item = Cow<'a, Self::Diff>>);
    fn scale(diff: &mut Self::Diff, times: Delta);

    /// Optional fast path for determining when a value should automatically be considered deleted
    fn consider_delete(_val: &Self::Value) -> bool {
        false
    }
    /// Optional fast path for determining if a diff is a no-op
    fn noop(_diff: &Self::Diff) -> bool {
        false
    }

    /// Whether a keyspace of this type is additive: what is written to a
    /// key is an amount, and the key's value is the sum of every amount
    /// written, absent when [`consider_delete`](Self::consider_delete)
    /// says so. Writes to an additive keyspace are blind: nothing is read
    /// to make them, there are no deletes, and writes to a key in
    /// different transactions are summed when they are read or folded
    /// together. See [`Tally`].
    const ADDITIVE: bool = false;

    /// For an additive type, add `amount` to `sum`. The order amounts are
    /// added in must not matter.
    fn add(_sum: &mut Self::Value, _amount: &Self::Value) {
        unreachable!("only additive keyspaces sum their values")
    }
}

/// A count kept additively: each write is an amount to add, made without
/// reading the count, and a count of zero is absent. What
/// [arrangements](crate::state::Arrangement) and
/// [multisets](crate::source::Multiset) are kept as.
pub enum Tally {}

impl Merge for Tally {
    type Diff = Delta;
    type Value = Delta;
    const ADDITIVE: bool = true;
    fn base() -> Delta {
        0
    }
    #[inline]
    fn merge<'a>(val: &mut Delta, diffs: impl Iterator<Item = Cow<'a, Delta>>) {
        for d in diffs {
            Self::add(val, &d);
        }
    }
    #[inline]
    fn combine<'a>(old: &mut Delta, new: impl Iterator<Item = Cow<'a, Delta>>) {
        for d in new {
            Self::add(old, &d);
        }
    }
    fn scale(d: &mut Delta, n: Delta) {
        *d = d.checked_mul(n).expect("multiplicity overflowed");
    }
    #[inline]
    fn consider_delete(val: &Delta) -> bool {
        *val == 0
    }
    #[inline]
    fn noop(diff: &Delta) -> bool {
        *diff == 0
    }
    #[inline]
    fn add(sum: &mut Delta, amount: &Delta) {
        *sum = sum.checked_add(*amount).expect("multiplicity overflowed");
    }
}

impl Merge for () {
    type Diff = ();
    type Value = ();
    fn base() -> Self::Value {}
    fn merge<'a>(_: &mut Self::Value, _: impl Iterator<Item = Cow<'a, Self::Diff>>) {}
    fn combine<'a>(_: &mut Self::Diff, _: impl Iterator<Item = Cow<'a, Self::Diff>>) {}
    fn scale(_: &mut Self::Diff, _: Delta) {}
}

impl Merge for i128 {
    type Diff = i128;
    type Value = i128;
    fn base() -> i128 {
        0
    }
    #[inline]
    fn merge<'a>(val: &mut i128, diffs: impl Iterator<Item = Cow<'a, i128>>) {
        *val += diffs.map(|d| *d).sum::<i128>();
    }
    #[inline]
    fn combine<'a>(old: &mut i128, new: impl Iterator<Item = Cow<'a, i128>>) {
        *old += new.map(|d| *d).sum::<i128>();
    }
    #[inline]
    fn noop(diff: &Self::Diff) -> bool {
        *diff == 0
    }
    fn scale(d: &mut i128, n: Delta) {
        *d *= n as i128;
    }
}

impl Merge for i64 {
    type Diff = i64;
    type Value = i64;
    fn base() -> i64 {
        0
    }
    #[inline]
    fn merge<'a>(val: &mut i64, diffs: impl Iterator<Item = Cow<'a, i64>>) {
        *val += diffs.map(|d| *d).sum::<i64>();
    }
    #[inline]
    fn combine<'a>(old: &mut i64, new: impl Iterator<Item = Cow<'a, i64>>) {
        *old += new.map(|d| *d).sum::<i64>();
    }
    #[inline]
    fn noop(diff: &Self::Diff) -> bool {
        *diff == 0
    }
    fn scale(d: &mut i64, n: Delta) {
        *d *= n;
    }
}

impl<A: Merge, B: Merge> Merge for (A, B) {
    type Diff = (A::Diff, B::Diff);
    type Value = (A::Value, B::Value);
    fn base() -> Self::Value {
        (A::base(), B::base())
    }
    #[inline]
    fn merge<'a>(val: &mut Self::Value, diffs: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        for d in diffs {
            let (a, b) = match d {
                Cow::Borrowed((a, b)) => (Cow::Borrowed(a), Cow::Borrowed(b)),
                Cow::Owned((a, b)) => (Cow::Owned(a), Cow::Owned(b)),
            };
            A::merge(&mut val.0, std::iter::once(a));
            B::merge(&mut val.1, std::iter::once(b));
        }
    }
    #[inline]
    fn combine<'a>(old: &mut Self::Diff, new: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        for d in new {
            let (a, b) = match d {
                Cow::Borrowed((a, b)) => (Cow::Borrowed(a), Cow::Borrowed(b)),
                Cow::Owned((a, b)) => (Cow::Owned(a), Cow::Owned(b)),
            };
            A::combine(&mut old.0, std::iter::once(a));
            B::combine(&mut old.1, std::iter::once(b));
        }
    }
    #[inline]
    fn scale(d: &mut Self::Diff, n: Delta) {
        A::scale(&mut d.0, n);
        B::scale(&mut d.1, n);
    }
    #[inline]
    fn consider_delete(val: &Self::Value) -> bool {
        A::consider_delete(&val.0) && B::consider_delete(&val.1)
    }
    #[inline]
    fn noop(diff: &Self::Diff) -> bool {
        A::noop(&diff.0) && B::noop(&diff.1)
    }
}

pub struct Sparse<M>(M);
impl<M: Merge> Merge for Sparse<M> {
    type Diff = Option<M::Diff>;
    type Value = M::Value;
    fn base() -> Self::Value {
        M::base()
    }
    #[inline]
    fn merge<'a>(val: &mut Self::Value, diffs: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        M::merge(
            val,
            diffs.filter_map(|d| match d {
                Cow::Borrowed(o) => o.as_ref().map(Cow::Borrowed),
                Cow::Owned(o) => o.map(Cow::Owned),
            }),
        );
    }
    #[inline]
    fn combine<'a>(old: &mut Self::Diff, new: impl Iterator<Item = Cow<'a, Self::Diff>>) {
        for n in new.filter_map(|d| match d {
            Cow::Borrowed(o) => o.as_ref().map(Cow::Borrowed),
            Cow::Owned(o) => o.map(Cow::Owned),
        }) {
            match old {
                Some(o) => M::combine(o, std::iter::once(n)),
                None => *old = Some(n.into_owned()),
            }
        }
    }
    #[inline]
    fn scale(d: &mut Self::Diff, n: Delta) {
        if let Some(d) = d {
            M::scale(d, n)
        }
    }
    #[inline]
    fn consider_delete(val: &Self::Value) -> bool {
        M::consider_delete(val)
    }
    #[inline]
    fn noop(diff: &Self::Diff) -> bool {
        if let Some(inner) = diff {
            M::noop(inner)
        } else {
            true
        }
    }
}

impl<M: Merge> ValOp<M> {
    #[inline]
    fn apply<'a>(&mut self, diffs: impl Iterator<Item = Cow<'a, M::Diff>>) {
        match self {
            ValOp::Store(v) => M::merge(v, diffs),
            ValOp::Modify(s) => M::combine(s, diffs),
            ValOp::Delete => {
                let mut v = M::base();
                M::merge(&mut v, diffs);
                *self = ValOp::Store(v);
            }
        }
    }

    #[inline]
    pub fn absorb_one(&mut self, new: Self) {
        // this is the absorb used on the write path
        match new {
            ValOp::Delete | ValOp::Store(_) => *self = new,
            ValOp::Modify(d) => {
                if !M::noop(&d) {
                    self.apply(std::iter::once(Cow::Owned(d)))
                }
            }
        }
    }

    #[inline]
    pub fn absorb(&mut self, ops: &[Self]) {
        // this is the absorb used on the read path
        // to reconstruct values from storage
        let reset = ops.iter().rposition(|op| !matches!(op, ValOp::Modify(_)));
        let tail = match reset {
            Some(i) => {
                *self = match &ops[i] {
                    ValOp::Store(v) => ValOp::Store(v.clone()),
                    ValOp::Delete => ValOp::Delete,
                    ValOp::Modify(_) => unreachable!("rposition excluded Modify"),
                };
                &ops[i + 1..]
            }
            None => ops,
        };
        if tail.is_empty() {
            return;
        }
        self.apply(
            tail.iter()
                .map(|op| match op {
                    ValOp::Modify(d) => Cow::Borrowed(d),
                    _ => unreachable!("everything after the last reset is a Modify"),
                })
                .filter(|d| !M::noop(d)),
        );
    }
}

// TODO: figure this shit out

pub trait RunIndex<K, M: Merge>: 'static {
    const NAME: &'static str;
    type Builder: IndexBuilder<K, M>;
    /// Streams the run's rows in key order as the run is written.
    fn builder() -> Self::Builder;
    /// A zero-copy view over the sealed section bytes.
    fn open(bytes: Bytes) -> Self;
}

pub trait IndexBuilder<K, M: Merge> {
    fn push(&mut self, key: &K, op: &ValOp<M>);
    /// `None` declines: not worth a section for this run.
    fn finish(self) -> Option<Bytes>;
}
