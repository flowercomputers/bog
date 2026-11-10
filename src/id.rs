//! [`Id`]: a typed row identity.
//!
//! An `Id<T>` is a `u64` that names one `T`. It is `Copy`, orders as a
//! plain integer, and encodes as one. Construction is public, so a table
//! keyed by application-chosen ids works, and sources that mint ids
//! ([`Events::push`](crate::source::Events::push),
//! [`Table::insert`](crate::source::Table::insert)) hand back the same
//! type.
//!
//! A minted id packs the committing transaction's epoch in the high 40
//! bits and a per-source, per-transaction sequence number in the low 24.
//! Numeric order is therefore `(epoch, seq)` order: ids are monotone
//! across commits with no separate timestamp, which is what retention
//! windows key on.

use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use crate::codec::{self, BufMut, Codec, OrderedCodec, ResidentSize};

const SEQ_BITS: u32 = 24;
const SEQ_MASK: u64 = (1 << SEQ_BITS) - 1;

pub struct Id<T> {
    raw: u64,
    _p: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    pub const fn new(raw: u64) -> Self {
        Id {
            raw,
            _p: PhantomData,
        }
    }

    /// The id a source mints for the `seq`-th row it accepts in the
    /// transaction committing as `epoch`.
    ///
    /// # Panics
    ///
    /// If either half overflows its bits.
    pub fn minted(epoch: u64, seq: u32) -> Self {
        assert!(epoch < (1 << (64 - SEQ_BITS)), "epoch overflows Id layout");
        assert!(seq as u64 <= SEQ_MASK, "sequence overflows Id layout");
        Id::new((epoch << SEQ_BITS) | seq as u64)
    }

    pub const fn raw(self) -> u64 {
        self.raw
    }

    /// The epoch a minted id was created in.
    pub const fn epoch(self) -> u64 {
        self.raw >> SEQ_BITS
    }

    /// The sequence number of a minted id within its transaction.
    pub const fn seq(self) -> u32 {
        (self.raw & SEQ_MASK) as u32
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Id<T> {}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<T> Eq for Id<T> {}
impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}
impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state)
    }
}
impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({})", self.raw)
    }
}
impl<T> fmt::Display for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.raw.fmt(f)
    }
}
impl<T> From<u64> for Id<T> {
    fn from(raw: u64) -> Self {
        Id::new(raw)
    }
}

impl<T> ResidentSize for Id<T> {
    fn resident_size(&self) -> usize {
        size_of::<Self>()
    }
}
impl<T> Codec for Id<T> {
    fn encode(&self, out: &mut impl BufMut) {
        self.raw.encode(out)
    }
    fn decode(b: &mut &[u8]) -> codec::Result<Self> {
        u64::decode(b).map(Id::new)
    }
    fn encode_size(&self) -> usize {
        self.raw.encode_size()
    }
}
impl<T> OrderedCodec for Id<T> {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        self.raw.encode_ordered(out)
    }
    fn decode_ordered(b: &mut &[u8]) -> codec::Result<Self> {
        u64::decode_ordered(b).map(Id::new)
    }
    fn encode_ordered_size(&self) -> usize {
        self.raw.encode_ordered_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minted_ids_order_by_epoch_then_sequence() {
        let a: Id<()> = Id::minted(1, 5);
        let b: Id<()> = Id::minted(2, 0);
        assert!(a < b);
        assert_eq!((a.epoch(), a.seq()), (1, 5));
        let mut bytes = Vec::new();
        b.encode_ordered(&mut bytes);
        assert_eq!(Id::<()>::decode_ordered(&mut &bytes[..]).unwrap(), b);
    }
}
