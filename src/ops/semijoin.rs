//! [`Semijoin`]: rows whose key the other side has ([`matching`]), or
//! lacks ([`without`]).
//!
//! [`matching`]: Semijoin::matching
//! [`without`]: Semijoin::without

use crate::{
    Bag, Binary,
    error::Result,
    state::{State, When},
    tx::WriteTx,
};

use super::{Take, read_group, rows_by_key};

/// Passes left rows `(k, a)` while the right side has a row of key `k`
/// ([`matching`](Self::matching), SQL's `EXISTS`), or while it has none
/// ([`without`](Self::without), `NOT EXISTS`), each at its own
/// multiplicity.
///
/// It keeps nothing: both parents' states must be grouped by the key.
/// For a key whose standing does not change, its new left rows pass or
/// not; for one that starts or stops passing, its whole left group is
/// emitted or retracted. A right row at a nonpositive multiplicity does
/// not count.
pub struct Semijoin<L, R> {
    anti: bool,
    left: L,
    right: R,
}

impl<L, R> Semijoin<L, R> {
    /// Left rows whose key the right side has.
    pub fn matching(left: L, right: R) -> Self {
        Semijoin {
            anti: false,
            left,
            right,
        }
    }

    /// Left rows whose key the right side lacks.
    pub fn without(left: L, right: R) -> Self {
        Semijoin {
            anti: true,
            left,
            right,
        }
    }
}

impl<L, R> Semijoin<L, R>
where
    L: State,
    L::Value: Ord,
    R: State<Key = L::Key>,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &L::Delta,
        r: &R::Delta,
        out: &mut Bag<(L::Key, L::Value)>,
    ) -> Result<()> {
        let left = rows_by_key::<L>(l);
        let mut dirty: Vec<L::Key> = left.iter().map(|(k, _)| k.clone()).collect();
        R::rows(r, &mut |k, _, _| dirty.push(k.clone()));
        dirty.sort();
        dirty.dedup();
        let mut left = left.into_iter().peekable();
        let mut group = Vec::new();
        for k in dirty {
            let rows = left.next_if(|(lk, _)| *lk == k).map(|(_, rows)| rows);
            let was = self.right.exists(tx, &k, When::Old)? != self.anti;
            let now = self.right.exists(tx, &k, When::New)? != self.anti;
            if was == now {
                if now {
                    for (a, d) in rows.into_iter().flatten() {
                        out.push((k.clone(), a), d);
                    }
                }
                continue;
            }
            let (when, sign) = if now { (When::New, 1) } else { (When::Old, -1) };
            read_group(&self.left, tx, &k, when, Take::Whole, &mut group)?;
            for (a, d) in group.drain(..) {
                out.push((k.clone(), a), sign * d);
            }
        }
        Ok(())
    }
}

impl<L, R> Binary for Semijoin<L, R>
where
    L: State,
    L::Value: Ord,
    R: State<Key = L::Key>,
{
    type Left = L::Delta;
    type Right = R::Delta;
    type Gives = Bag<(L::Key, L::Value)>;
    type State = ();

    fn state(&self) {}

    fn fingerprint(&self, fp: &mut crate::graph::Fingerprint) {
        fp.write_u64(self.anti as u64);
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &L::Delta, r: &R::Delta, out: &mut Self::Gives) {
        let _ = self.run(tx, l, r, out);
    }
}
