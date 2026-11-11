//! Linear stages fused into one node: a chain of [`Emit`]s, each handing
//! rows straight to the next, run by [`FusedLinears`].

use std::{borrow::Cow, marker::PhantomData};

use super::{Bag, Nary};
use crate::store::txn::WriteTx;

pub struct FusedLinears<In, X>(X, PhantomData<fn() -> In>);

/// The stage that changes nothing: an empty chain.
#[derive(Clone, Copy)]
pub struct Ident;
impl<In: Clone + 'static> Emit<In> for Ident {
    type Emits = In;
    #[inline]
    fn emit<S: FnMut(Cow<In>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        sink(data, diff)
    }
}

impl<In, X> FusedLinears<In, X> {
    /// One node running the chain of linear stages `x` over each row.
    pub fn new(x: X) -> Self {
        FusedLinears(x, PhantomData)
    }
}
impl<In, Out, X> Nary for FusedLinears<In, X>
where
    In: Clone + 'static + Send + Ord,
    Out: Clone + 'static + Send + Ord,
    X: Emit<In, Emits = Out>,
{
    type Takes = Bag<In>;
    type Gives = Bag<Out>;
    type State = ();

    fn state(&self) {}

    fn eval(&mut self, _tx: &mut WriteTx, batches: &[&Bag<In>], out: &mut Bag<Out>) {
        // implicitly concatenates batches
        for batch in batches {
            for (data, diff) in batch.diffs() {
                self.0.emit(Cow::Borrowed(data), diff, &mut |data, diff| {
                    out.push(data.into_owned(), diff)
                });
            }
        }
    }
}

pub trait Emit<In: Clone>: 'static + Send {
    type Emits: Clone + 'static;
    fn emit<S: FnMut(Cow<Self::Emits>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S);
}

#[derive(Clone)]
pub struct Then<A, B>(pub A, pub B);
impl<In, Mid, Out, A, B> Emit<In> for Then<A, B>
where
    In: Clone + 'static,
    Mid: Clone + 'static,
    Out: Clone + 'static,
    A: Emit<In, Emits = Mid>,
    B: Emit<Mid, Emits = Out>,
{
    type Emits = Out;
    #[inline]
    fn emit<S: FnMut(Cow<Out>, i64)>(&self, data: Cow<In>, diff: i64, sink: &mut S) {
        self.0
            .emit(data, diff, &mut |mid, d| self.1.emit(mid, d, &mut *sink));
    }
}
