//! [`Join`]: the inner join, reactive on both sides.

use crate::{
    Bag, Binary,
    error::Result,
    state::{State, When},
    tx::WriteTx,
};

use super::{Take, multiply, read_group, rows_by_key};

/// Pairs each left row `(k, a)` with every right row `(k, b)` and emits
/// `logic(&k, &a, &b)` at the product of their multiplicities: SQL's
/// INNER JOIN. Retracting a row on either side retracts its pairs.
///
/// It keeps nothing: it reads its parents' states, which must be grouped
/// by the join key: tables and other registers keyed by it, or
/// [`Arrange`](super::Arrange) nodes. A run emits
/// `ΔA ⋈ B_old + A_new ⋈ ΔB`, so rows arriving on both sides at once
/// pair exactly once. A dirty key costs reading its group on the other
/// side: one lookup for a register.
pub struct Join<L, R, F> {
    left: L,
    right: R,
    logic: F,
}

impl<L, R, F> Join<L, R, F> {
    /// Build with [`OpDag::binary_with`](crate::graph::OpDag::binary_with),
    /// which hands over the parents' states.
    pub fn new(left: L, right: R, logic: F) -> Self {
        Join { left, right, logic }
    }
}

impl<L, R, O, F> Join<L, R, F>
where
    L: State,
    R: State<Key = L::Key>,
    O: Ord + Send + 'static,
    F: FnMut(&L::Key, &L::Value, &R::Value) -> O + Send,
{
    fn run(
        &mut self,
        tx: &mut WriteTx,
        l: &L::Delta,
        r: &R::Delta,
        out: &mut Bag<O>,
    ) -> Result<()> {
        let mut group_b = Vec::new();
        for (k, rows) in rows_by_key::<L>(l) {
            read_group(&self.right, tx, &k, When::Old, Take::Whole, &mut group_b)?;
            for (a, da) in &rows {
                for (b, db) in &group_b {
                    out.push((self.logic)(&k, a, b), multiply(*da, *db));
                }
            }
        }
        let mut group_a = Vec::new();
        for (k, rows) in rows_by_key::<R>(r) {
            read_group(&self.left, tx, &k, When::New, Take::Whole, &mut group_a)?;
            for (b, db) in &rows {
                for (a, da) in &group_a {
                    out.push((self.logic)(&k, a, b), multiply(*da, *db));
                }
            }
        }
        Ok(())
    }
}

impl<L, R, O, F> Binary for Join<L, R, F>
where
    L: State,
    R: State<Key = L::Key>,
    O: Ord + Send + 'static,
    F: FnMut(&L::Key, &L::Value, &R::Value) -> O + Send,
{
    type Left = L::Delta;
    type Right = R::Delta;
    type Gives = Bag<O>;
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, tx: &mut WriteTx, l: &L::Delta, r: &R::Delta, out: &mut Bag<O>) {
        // on failure the tx has failed, and the graph stops after this node
        let _ = self.run(tx, l, r, out);
    }
}
