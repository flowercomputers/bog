//! What an operator is: [`Nary`], [`Unary`] or [`Binary`], by how many
//! inputs it takes. The graph erases them to [`Node`].

use std::any::Any;

use super::{Buffer, Fingerprint, state::Declare};
use crate::store::txn::WriteTx;

pub trait Node: Send {
    fn eval(&mut self, tx: &mut WriteTx, inputs: &[&dyn Any], out: &mut dyn Any);
    fn fingerprint(&self, fp: &mut Fingerprint);
    fn scratch(&mut self, release: bool) -> usize;
}

/// Commutative, variable-arity
pub trait Nary: Send {
    // TODO: In/Out are redundant with Takes/Gives
    type Takes: Buffer;
    type Gives: Buffer;
    /// What children can read of this node's output; see [`state`](super::state). `()`
    /// for none.
    type State: Clone + Send + 'static;
    fn eval(&mut self, tx: &mut WriteTx, batches: &[&Self::Takes], out: &mut Self::Gives);
    /// The handle children get to this node's state.
    fn state(&self) -> Self::State;
    /// The keyspaces this node writes. Each is owned by one node.
    fn keyspaces(&self) -> Vec<Declare> {
        Vec::new()
    }
    /// Add whatever the operator's type does not already say to the
    /// graph's fingerprint. Owned keyspaces are added for it.
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
    /// The bytes of allocation the operator keeps between runs to reuse,
    /// such as scratch vectors; with `release`, it gives them back first.
    /// An operator that keeps any should say so: see [`vec_scratch`](super::vec_scratch).
    fn scratch(&mut self, _release: bool) -> usize {
        0
    }
}

pub trait Unary: Send {
    type Takes: Buffer;
    type Gives: Buffer;
    /// See [`Nary::State`].
    type State: Clone + Send + 'static;
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives);
    fn state(&self) -> Self::State;
    fn keyspaces(&self) -> Vec<Declare> {
        Vec::new()
    }
    /// See [`Nary::fingerprint`].
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
    /// See [`Nary::scratch`].
    fn scratch(&mut self, _release: bool) -> usize {
        0
    }
}

impl<N: Nary> Unary for N {
    type Takes = N::Takes;
    type Gives = N::Gives;
    type State = N::State;
    #[inline]
    fn eval(&mut self, tx: &mut WriteTx, batch: &Self::Takes, out: &mut Self::Gives) {
        N::eval(self, tx, &[batch], out);
    }
    fn state(&self) -> N::State {
        N::state(self)
    }
    fn keyspaces(&self) -> Vec<Declare> {
        N::keyspaces(self)
    }
    fn fingerprint(&self, fp: &mut Fingerprint) {
        N::fingerprint(self, fp)
    }
    fn scratch(&mut self, release: bool) -> usize {
        N::scratch(self, release)
    }
}

pub trait Binary: Send {
    type Left: Buffer;
    type Right: Buffer;
    type Gives: Buffer;
    /// See [`Nary::State`].
    type State: Clone + Send + 'static;
    fn eval(&mut self, tx: &mut WriteTx, l: &Self::Left, r: &Self::Right, out: &mut Self::Gives);
    fn state(&self) -> Self::State;
    fn keyspaces(&self) -> Vec<Declare> {
        Vec::new()
    }
    /// See [`Nary::fingerprint`].
    fn fingerprint(&self, _fp: &mut Fingerprint) {}
    /// See [`Nary::scratch`].
    fn scratch(&mut self, _release: bool) -> usize {
        0
    }
}
