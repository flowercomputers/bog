//! [`Join`]: the inner join, reactive on both sides.

use crate::{
    dataflow::state::{State, When},
    dataflow::{Bag, Binary},
    error::Result,
    store::txn::WriteTx,
};

use super::{Groups, multiply};
use crate::stream::prelude::*;

/// Pairs each left row `(k, a)` with every right row `(k, b)` and emits
/// `logic(&k, &a, &b)` at the product of their multiplicities: SQL's
/// INNER JOIN. Retracting a row on either side retracts its pairs.
///
/// It keeps nothing: it reads its parents' states, which must be grouped
/// by the join key: tables and other registers keyed by it, or
/// [`Arrange`] nodes. A run emits
/// `ΔA ⋈ B_old + A_new ⋈ ΔB`, so rows arriving on both sides at once
/// pair exactly once. A dirty key costs reading its group on the other
/// side: one lookup for a register.
pub struct Join<L: State, R: State, F> {
    left: L,
    right: R,
    logic: F,
    /// Each side's delta grouped by key: scratch, reused across runs.
    left_rows: Groups<L::Key, L::Value>,
    right_rows: Groups<R::Key, R::Value>,
}

impl<L: State, R: State, F> Join<L, R, F> {
    /// Build with [`OpDag::binary_with`](crate::dataflow::graph::OpDag::binary_with),
    /// which hands over the parents' states.
    pub fn new(left: L, right: R, logic: F) -> Self {
        Join {
            left,
            right,
            logic,
            left_rows: Groups::default(),
            right_rows: Groups::default(),
        }
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
        // ΔA ⋈ B_old: every dirty left key's rows against the right group
        self.left_rows.fill::<L>(l);
        let mut rows = self.left_rows.iter().peekable();
        let logic = &mut self.logic;
        self.right
            .groups(tx, self.left_rows.keys(), When::Old, &mut |k, b, db| {
                while rows.next_if(|(lk, _)| *lk < k).is_some() {}
                if let Some((_, group)) = rows.peek() {
                    for (a, da) in *group {
                        out.push(logic(k, a, b), multiply(*da, db));
                    }
                }
            })?;
        // A_new ⋈ ΔB
        self.right_rows.fill::<R>(r);
        let mut rows = self.right_rows.iter().peekable();
        self.left
            .groups(tx, self.right_rows.keys(), When::New, &mut |k, a, da| {
                while rows.next_if(|(rk, _)| *rk < k).is_some() {}
                if let Some((_, group)) = rows.peek() {
                    for (b, db) in *group {
                        out.push(logic(k, a, b), multiply(da, *db));
                    }
                }
            })?;
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

    fn scratch(&mut self, release: bool) -> usize {
        self.left_rows.scratch(release) + self.right_rows.scratch(release)
    }

    fn eval(&mut self, tx: &mut WriteTx, l: &L::Delta, r: &R::Delta, out: &mut Bag<O>) {
        // on failure the tx has failed, and the graph stops after this node
        let _ = self.run(tx, l, r, out);
    }
}

// The stream methods: how a schema declares this operator.

impl<K, V, C> Grouped<K, V, C>
where
    K: Data,
    V: Data,
    C: Chain<Out = (K, V)>,
{
    /// Inner join against a grouped stream on the same key: every row
    /// pairs with every right row of its key, at the product of their
    /// multiplicities. Reads both sides' state; see the module docs.
    pub fn join<B: Data, O: Data, F>(
        &self,
        other: &impl GroupedFeed<K, B>,
        mut f: F,
    ) -> Grouped<K, O>
    where
        F: FnMut(&K, &V, &B) -> O + Send + 'static,
    {
        let name = self.flow.name("join", Some(&other.label()));
        let (l, ls) = arranged(self, &format!("{name}.left"));
        let (r, rs) = arranged(other, &format!("{name}.right"));
        let logic = move |k: &K, a: &V, b: &B| (k.clone(), f(k, a, b));
        let node = self
            .flow
            .build
            .borrow_mut()
            .graph
            .binary(l, r, Join::new(ls, rs, logic));
        Rows::rooted(Root::at(node), self.flow.build.clone(), name.into(), None)
    }
}

impl<K, V, M, C> Rows<K, V, M, C>
where
    K: Data,
    V: Data,
    M: Kind,
    C: Chain<Out = (K, V)>,
{
    /// Inner join against a register on the same key: each row pairs
    /// with the key's one right value, so the kind is kept.
    pub fn with<B: Data, O: Data, F>(&self, other: &impl KeyedFeed<K, B>, mut f: F) -> Rows<K, O, M>
    where
        F: FnMut(&K, &V, &B) -> O + Send + 'static,
    {
        let name = self.flow.name("with", Some(&other.label()));
        let (l, ls) = arranged(self, &format!("{name}.left"));
        let (r, rs) = arranged(other, &format!("{name}.right"));
        let logic = move |k: &K, a: &V, b: &B| (k.clone(), f(k, a, b));
        let node = self
            .flow
            .build
            .borrow_mut()
            .graph
            .binary(l, r, Join::new(ls, rs, logic));
        Rows::rooted(Root::at(node), self.flow.build.clone(), name.into(), None)
    }
}
