use std::array;
use std::borrow::Cow;
use std::fmt;

pub use bytes::BufMut;

use crate::dataflow::Fingerprint;

/// Why a byte slice could not be decoded.
///
/// `decode` takes `&mut &[u8]` and advances it as it goes, so on error the
/// caller can recover the approximate failing offset as
/// `original.len() - remaining.len()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The input ended before the value did.
    UnexpectedEof,
    /// A discriminant byte was not one of the expected values.
    InvalidTag { what: &'static str, tag: u8 },
    /// A string's bytes were not valid UTF-8.
    InvalidUtf8,
    /// A `char`'s code point was not a valid scalar value.
    InvalidChar,
    /// An ordered string ran off the end of the input without its terminator.
    MissingTerminator,
    /// `decode_all` finished the value with bytes still left over.
    TrailingBytes { remaining: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("unexpected end of input"),
            Self::InvalidTag { what, tag } => write!(f, "invalid {what} tag {tag:#04x}"),
            Self::InvalidUtf8 => f.write_str("invalid UTF-8 in string"),
            Self::InvalidChar => f.write_str("invalid char code point"),
            Self::MissingTerminator => f.write_str("ordered string missing terminator"),
            Self::TrailingBytes { remaining } => {
                write!(f, "{remaining} trailing byte(s) after value")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

pub type Result<T, E = DecodeError> = std::result::Result<T, E>;

/// How much memory a value occupies: its own `size_of` plus every heap
/// allocation it owns. The store uses it to account for memory, so it is
/// an estimate: allocator overhead is not counted, and shared allocations
/// (`Rc`, `Arc`) are counted in full by every holder.
pub trait ResidentSize {
    fn resident_size(&self) -> usize;
}

/// The heap part of [`ResidentSize`]: what a value owns beyond its inline
/// bytes. Containers sum this over their elements, since the elements'
/// inline bytes are already in the container's own size.
pub fn heap_size<T: ResidentSize>(x: &T) -> usize {
    x.resident_size().saturating_sub(size_of::<T>())
}

macro_rules! impl_inline_resident {
    ($($t:ty),* $(,)?) => {$(
        impl ResidentSize for $t {
            #[inline]
            fn resident_size(&self) -> usize {
                size_of::<Self>()
            }
        }
    )*};
}
impl_inline_resident!(
    (),
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

impl<T: ResidentSize> ResidentSize for Option<T> {
    fn resident_size(&self) -> usize {
        size_of::<Self>() + self.as_ref().map_or(0, heap_size)
    }
}

impl<T: ResidentSize, const N: usize> ResidentSize for [T; N] {
    fn resident_size(&self) -> usize {
        size_of::<Self>() + self.iter().map(heap_size).sum::<usize>()
    }
}

impl<T: ResidentSize> ResidentSize for Vec<T> {
    fn resident_size(&self) -> usize {
        // the whole buffer is allocated, used or not
        size_of::<Self>()
            + self.capacity() * size_of::<T>()
            + self.iter().map(heap_size).sum::<usize>()
    }
}

impl ResidentSize for String {
    fn resident_size(&self) -> usize {
        size_of::<Self>() + self.capacity()
    }
}

impl ResidentSize for Box<str> {
    fn resident_size(&self) -> usize {
        size_of::<Self>() + self.len()
    }
}

impl ResidentSize for std::rc::Rc<str> {
    fn resident_size(&self) -> usize {
        // the allocation also holds the strong and weak counts
        size_of::<Self>() + 2 * size_of::<usize>() + self.len()
    }
}

impl ResidentSize for std::sync::Arc<str> {
    fn resident_size(&self) -> usize {
        size_of::<Self>() + 2 * size_of::<usize>() + self.len()
    }
}

/// The shape of a type's encoding, for telling whether stored bytes
/// still decode as the type that reads them.
///
/// A keyspace records the [`LayoutHash`] of its key and value types, and
/// opening it under types with another refuses, where decoding
/// positional bytes as the wrong shape would otherwise go unnoticed.
///
/// The hash covers what the bytes are, not what the types are called:
/// names, field names and module paths are left out, and a struct or
/// tuple contributes its fields one after another, as it encodes. So
/// renaming, moving or newtype-wrapping a type is free, while adding,
/// removing, reordering or retyping a field is a different layout.
///
/// The derive implements it. A hand-written codec describes itself with
/// [`LayoutHash::atom`], under a tag to change whenever its bytes do:
///
/// ```ignore
/// impl Layout for Money {
///     fn layout(l: &mut LayoutHash) {
///         l.atom("money/1");
///     }
/// }
/// ```
///
/// One layout covers both the plain and the ordered encoding, which are
/// two writings of the same fields.
pub trait Layout {
    fn layout(l: &mut LayoutHash);
}

/// What a [`Layout`] is written to: a stable hash of a token stream.
pub struct LayoutHash {
    fp: Fingerprint,
    /// The named types being described, outermost first, so one that
    /// contains itself is described once.
    open: Vec<&'static str>,
}

impl Default for LayoutHash {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutHash {
    const ATOM: u8 = 1;
    const OPEN: u8 = 2;
    const CLOSE: u8 = 3;
    const AGAIN: u8 = 4;

    pub fn new() -> Self {
        LayoutHash {
            fp: Fingerprint::new(),
            open: Vec::new(),
        }
    }

    /// The layout of `T`.
    pub fn of<T: Layout>() -> u64 {
        let mut l = Self::new();
        T::layout(&mut l);
        l.finish()
    }

    pub fn finish(&self) -> u64 {
        self.fp.finish()
    }

    /// An encoding with no parts: a primitive, or a hand-written codec.
    pub fn atom(&mut self, tag: &str) {
        self.fp.write(&[Self::ATOM]);
        self.fp.write_str(tag);
    }

    /// An encoding that frames its parts, so they do not run on from
    /// what is around them: an option, a list, one variant of an enum.
    /// `parts` writes the parts.
    pub fn group(&mut self, kind: &str, parts: impl FnOnce(&mut Self)) {
        self.fp.write(&[Self::OPEN]);
        self.fp.write_str(kind);
        parts(self);
        self.fp.write(&[Self::CLOSE]);
    }

    /// The fields of the named type `T`, written by `fields`. A type
    /// reached again inside itself is written as a reference back to
    /// the first, so a recursive type has a finite layout.
    pub fn named<T: ?Sized>(&mut self, fields: impl FnOnce(&mut Self)) {
        let name = std::any::type_name::<T>();
        if let Some(at) = self.open.iter().rposition(|open| *open == name) {
            self.fp.write(&[Self::AGAIN]);
            self.fp.write_u64((self.open.len() - at) as u64);
            return;
        }
        self.open.push(name);
        fields(self);
        self.open.pop();
    }
}

macro_rules! impl_atom_layout {
    ($($t:ty => $tag:literal),* $(,)?) => {$(
        impl Layout for $t {
            fn layout(l: &mut LayoutHash) {
                l.atom($tag);
            }
        }
    )*};
}
impl_atom_layout!(
    bool => "bool", char => "char",
    u8 => "u8", u16 => "u16", u32 => "u32", u64 => "u64", u128 => "u128",
    i8 => "i8", i16 => "i16", i32 => "i32", i64 => "i64", i128 => "i128",
    // every string type encodes alike
    String => "str", Box<str> => "str", std::rc::Rc<str> => "str", std::sync::Arc<str> => "str",
);

impl Layout for () {
    fn layout(_: &mut LayoutHash) {}
}

impl<T: Layout> Layout for Option<T> {
    fn layout(l: &mut LayoutHash) {
        l.group("option", T::layout);
    }
}

impl<T: Layout> Layout for crate::dataflow::state::Edge<T> {
    fn layout(l: &mut LayoutHash) {
        l.group("edge", T::layout);
    }
}

impl<T: Layout, const N: usize> Layout for [T; N] {
    fn layout(l: &mut LayoutHash) {
        l.group("array", |l| {
            l.atom(&N.to_string());
            T::layout(l);
        });
    }
}

impl<T: Layout> Layout for Vec<T> {
    fn layout(l: &mut LayoutHash) {
        l.group("vec", T::layout);
    }
}

impl<T: Layout> Layout for std::cmp::Reverse<T> {
    fn layout(l: &mut LayoutHash) {
        l.group("reverse", T::layout);
    }
}

pub trait Codec: Sized + ResidentSize + Layout {
    fn encode(&self, out: &mut impl BufMut);
    fn decode(b: &mut &[u8]) -> Result<Self>;
    fn encode_size(&self) -> usize;
}
pub trait OrderedCodec: Ord + Sized + ResidentSize + Layout {
    fn encode_ordered(&self, out: &mut impl BufMut);
    fn decode_ordered(b: &mut &[u8]) -> Result<Self>;
    fn encode_ordered_size(&self) -> usize;
}

/// Decode a value that must occupy the whole slice.
pub fn decode_all<T: Codec>(mut b: &[u8]) -> Result<T> {
    let v = T::decode(&mut b)?;
    check_empty(b)?;
    Ok(v)
}

/// Decode an ordered value that must occupy the whole slice.
pub fn decode_all_ordered<T: OrderedCodec>(mut b: &[u8]) -> Result<T> {
    let v = T::decode_ordered(&mut b)?;
    check_empty(b)?;
    Ok(v)
}

fn check_empty(b: &[u8]) -> Result<()> {
    if b.is_empty() {
        Ok(())
    } else {
        Err(DecodeError::TrailingBytes { remaining: b.len() })
    }
}

fn take_u8(b: &mut &[u8]) -> Result<u8> {
    let (&x, rest) = b.split_first().ok_or(DecodeError::UnexpectedEof)?;
    *b = rest;
    Ok(x)
}

fn take_chunk<const N: usize>(b: &mut &[u8]) -> Result<[u8; N]> {
    let (chunk, rest) = b
        .split_first_chunk::<N>()
        .ok_or(DecodeError::UnexpectedEof)?;
    *b = rest;
    Ok(*chunk)
}

fn take_bytes<'a>(b: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if n > b.len() {
        return Err(DecodeError::UnexpectedEof);
    }
    let (head, rest) = b.split_at(n);
    *b = rest;
    Ok(head)
}

impl Codec for () {
    fn encode(&self, _: &mut impl BufMut) {}
    fn decode(_: &mut &[u8]) -> Result<Self> {
        Ok(())
    }
    fn encode_size(&self) -> usize {
        0
    }
}
impl OrderedCodec for () {
    fn encode_ordered(&self, _: &mut impl BufMut) {}
    fn decode_ordered(_: &mut &[u8]) -> Result<Self> {
        Ok(())
    }
    fn encode_ordered_size(&self) -> usize {
        0
    }
}

// Option: 0x00 for None, 0x01 + payload for Some. None < Some matches Option's Ord.
impl<T: Codec> Codec for Option<T> {
    fn encode(&self, out: &mut impl BufMut) {
        match self {
            None => out.put_u8(0),
            Some(inner) => {
                out.put_u8(1);
                inner.encode(out);
            }
        }
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        match take_u8(b)? {
            0 => Ok(None),
            1 => Ok(Some(T::decode(b)?)),
            tag => Err(DecodeError::InvalidTag {
                what: "option",
                tag,
            }),
        }
    }
    fn encode_size(&self) -> usize {
        1 + match self {
            None => 0,
            Some(inner) => inner.encode_size(),
        }
    }
}
impl<T: OrderedCodec> OrderedCodec for Option<T> {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        match self {
            None => out.put_u8(0),
            Some(inner) => {
                out.put_u8(1);
                inner.encode_ordered(out);
            }
        }
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        match take_u8(b)? {
            0 => Ok(None),
            1 => Ok(Some(T::decode_ordered(b)?)),
            tag => Err(DecodeError::InvalidTag {
                what: "option",
                tag,
            }),
        }
    }
    fn encode_ordered_size(&self) -> usize {
        1 + match self {
            None => 0,
            Some(inner) => inner.encode_ordered_size(),
        }
    }
}

impl<T: ResidentSize> ResidentSize for crate::dataflow::state::Edge<T> {
    fn resident_size(&self) -> usize {
        size_of::<Self>()
            + match self {
                crate::dataflow::state::Edge::Val(v) => heap_size(v),
                _ => 0,
            }
    }
}

// Edge: Lo < Val(_) < Hi, matching tags 0 < 1 < 2.
impl<T: OrderedCodec> OrderedCodec for crate::dataflow::state::Edge<T> {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        use crate::dataflow::state::Edge;
        match self {
            Edge::Lo => out.put_u8(0),
            Edge::Val(v) => {
                out.put_u8(1);
                v.encode_ordered(out);
            }
            Edge::Hi => out.put_u8(2),
        }
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        use crate::dataflow::state::Edge;
        match take_u8(b)? {
            0 => Ok(Edge::Lo),
            1 => Ok(Edge::Val(T::decode_ordered(b)?)),
            2 => Ok(Edge::Hi),
            tag => Err(DecodeError::InvalidTag { what: "edge", tag }),
        }
    }
    fn encode_ordered_size(&self) -> usize {
        1 + match self {
            crate::dataflow::state::Edge::Val(v) => v.encode_ordered_size(),
            _ => 0,
        }
    }
}

// bool: Ord is false < true, matching 0 < 1.
impl Codec for bool {
    fn encode(&self, out: &mut impl BufMut) {
        out.put_u8(*self as u8);
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        match take_u8(b)? {
            0 => Ok(false),
            1 => Ok(true),
            tag => Err(DecodeError::InvalidTag { what: "bool", tag }),
        }
    }
    fn encode_size(&self) -> usize {
        1
    }
}
impl OrderedCodec for bool {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        Codec::encode(self, out)
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        Codec::decode(b)
    }
    fn encode_ordered_size(&self) -> usize {
        1
    }
}

// char: Ord is by code point, so the u32 big-endian encoding already sorts correctly.
impl Codec for char {
    fn encode(&self, out: &mut impl BufMut) {
        u32::from(*self).encode(out);
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        char::from_u32(u32::decode(b)?).ok_or(DecodeError::InvalidChar)
    }
    fn encode_size(&self) -> usize {
        4
    }
}
impl OrderedCodec for char {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        Codec::encode(self, out)
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        Codec::decode(b)
    }
    fn encode_ordered_size(&self) -> usize {
        Codec::encode_size(self)
    }
}

macro_rules! impl_int_codec {
    (@common $t:ty) => {
        impl Codec for $t {
            fn encode(&self, out: &mut impl BufMut) {
                out.put_slice(&self.to_be_bytes());
            }
            fn decode(b: &mut &[u8]) -> Result<Self> {
                Ok(<$t>::from_be_bytes(take_chunk(b)?))
            }
            fn encode_size(&self) -> usize {
                self.to_be_bytes().len()
            }
        }
    };
    (unsigned: $($t:ty),* $(,)?) => {$(
        impl_int_codec!(@common $t);
        impl OrderedCodec for $t {
            fn encode_ordered(&self, out: &mut impl BufMut) { Codec::encode(self, out) }
            fn decode_ordered(b: &mut &[u8]) -> Result<Self> { Codec::decode(b) }
            fn encode_ordered_size(&self) -> usize {
                self.to_be_bytes().len()
            }
        }
    )*};
    // Signed: flipping the sign bit maps MIN..=MAX onto 0..=UMAX in order.
    (signed: $($t:ty),* $(,)?) => {$(
        impl_int_codec!(@common $t);
        impl OrderedCodec for $t {
            fn encode_ordered(&self, out: &mut impl BufMut) {
                Codec::encode(&(self ^ <$t>::MIN), out)
            }
            fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
                Ok(<$t as Codec>::decode(b)? ^ <$t>::MIN)
            }
            fn encode_ordered_size(&self) -> usize {
                self.to_be_bytes().len()
            }
        }
    )*};
}
impl_int_codec!(unsigned: u8, u16, u32, u64, u128);
impl_int_codec!(signed: i8, i16, i32, i64, i128);

// ---- arrays / vectors -------------------------------------------------------

// [T; N]: fixed length, so no prefix or terminator needed. Array Ord is lexicographic,
// which concatenation of prefix-free element encodings preserves.
//
// `array::from_fn` calls the closure in index order 0..N but can't short-circuit, and
// `array::try_from_fn` is unstable. So the closure records the first failure, returns
// a placeholder for the rest without touching the input, and we check afterwards.
fn decode_array<T, const N: usize>(mut f: impl FnMut() -> Result<T>) -> Result<[T; N]> {
    let mut failed: Option<DecodeError> = None;
    let results: [Result<T>; N] = array::from_fn(|_| {
        if let Some(e) = failed {
            return Err(e);
        }
        f().inspect_err(|&e| failed = Some(e))
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(results.map(|r| r.expect("no failure recorded"))),
    }
}

impl<T: Codec, const N: usize> Codec for [T; N] {
    fn encode(&self, out: &mut impl BufMut) {
        for x in self {
            x.encode(out);
        }
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        decode_array(|| T::decode(b))
    }
    fn encode_size(&self) -> usize {
        let mut size = 0;
        for x in self {
            size += x.encode_size();
        }
        size
    }
}
impl<T: OrderedCodec, const N: usize> OrderedCodec for [T; N] {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        for x in self {
            x.encode_ordered(out);
        }
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        decode_array(|| T::decode_ordered(b))
    }
    fn encode_ordered_size(&self) -> usize {
        let mut size = 0;
        for x in self {
            size += x.encode_ordered_size();
        }
        size
    }
}

// Vec<T>, unordered: u32 length prefix, then elements.
impl<T: Codec> Codec for Vec<T> {
    fn encode(&self, out: &mut impl BufMut) {
        let len = u32::try_from(self.len()).expect("vec too long");
        len.encode(out);
        for x in self {
            x.encode(out);
        }
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        let len = u32::decode(b)? as usize;
        // The length prefix is untrusted; a corrupt 0xFFFF_FFFF must not allocate 4 GiB
        // up front. Every non-zero-sized element costs at least one byte, so the
        // remaining input bounds the plausible count. (Zero-sized elements make the
        // bound loose, but then there is nothing to allocate anyway.)
        let mut v = Vec::with_capacity(len.min(b.len()));
        for _ in 0..len {
            v.push(T::decode(b)?);
        }
        Ok(v)
    }
    fn encode_size(&self) -> usize {
        let mut elsize = 0;
        for x in self {
            elsize += x.encode_size();
        }
        4 + elsize
    }
}

// Vec<T>, ordered: a length prefix would sort [5] before [1, 2], so instead each element
// is preceded by 0x01 and the list ends with 0x00. Where a shorter list has its 0x00
// terminator, a longer list with the same prefix has 0x01, so the shorter one sorts
// first — matching Vec's lexicographic Ord.
impl<T: OrderedCodec> OrderedCodec for Vec<T> {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        // TODO: This doubles the size of Vec<u8> keys... Maybe a Key(V<u8>) or Bytes(V<u8>) newtype?
        for x in self {
            out.put_u8(1);
            x.encode_ordered(out);
        }
        out.put_u8(0);
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        let mut v = Vec::new();
        loop {
            match take_u8(b)? {
                0 => return Ok(v),
                1 => v.push(T::decode_ordered(b)?),
                tag => {
                    return Err(DecodeError::InvalidTag {
                        what: "vec marker",
                        tag,
                    });
                }
            }
        }
    }
    fn encode_ordered_size(&self) -> usize {
        let mut elsize = 0;
        for x in self {
            elsize += 1 + x.encode_ordered_size();
        }
        1 + elsize
    }
}

// Ordered strings are escaped rather than restricted: each 0x00 byte in the string is
// written as 0x00 0x01, and the string ends with 0x00 0x00. This keeps lexicographic
// order (at the first point where two strings differ, a 0x00 byte is always followed by
// 0x01 while a terminator is followed by 0x00, so "ends here" < "continues with NUL" <
// "continues with anything else"), and because the terminator is two bytes it is
// unambiguous no matter what the *next* field's first byte is. A one-byte 0x00
// terminator with a 0x00 0xFF escape would misread "a" followed by a u8 of 255.
fn put_escaped(s: &[u8], out: &mut impl BufMut) {
    for &byte in s {
        out.put_u8(byte);
        if byte == 0 {
            out.put_u8(1);
        }
    }
    out.put_slice(&[0, 0]);
}

fn take_escaped<'a>(b: &mut &'a [u8]) -> Result<Cow<'a, [u8]>> {
    let input: &'a [u8] = b;
    let mut owned: Option<Vec<u8>> = None; // only allocated if an escape is seen
    let mut run_start = 0; // start of the unflushed run in `input`
    let mut i = 0;
    loop {
        let Some(&byte) = input.get(i) else {
            return Err(DecodeError::MissingTerminator);
        };
        if byte != 0 {
            i += 1;
            continue;
        }
        match input.get(i + 1) {
            Some(1) => {
                // escaped NUL: keep the 0x00, drop the 0x01
                owned
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&input[run_start..=i]);
                i += 2;
                run_start = i;
            }
            Some(0) => {
                let bytes = match owned {
                    None => Cow::Borrowed(&input[..i]),
                    Some(mut v) => {
                        v.extend_from_slice(&input[run_start..i]);
                        Cow::Owned(v)
                    }
                };
                *b = &input[i + 2..];
                return Ok(bytes);
            }
            Some(&tag) => {
                return Err(DecodeError::InvalidTag {
                    what: "string escape",
                    tag,
                });
            }
            None => return Err(DecodeError::MissingTerminator),
        }
    }
}

fn escaped_size(s: &[u8]) -> usize {
    s.len() + s.iter().filter(|&&b| b == 0).count() + 2
}

macro_rules! impl_str_codec {
    ($($t:ty),* $(,)?) => {$(
        impl Codec for $t {
            fn encode(&self, out: &mut impl BufMut) {
                let len = u32::try_from(self.len()).expect("string too long");
                len.encode(out);
                out.put_slice(self.as_bytes());
            }
            fn decode(b: &mut &[u8]) -> Result<Self> {
                let len = u32::decode(b)? as usize;
                let bytes = take_bytes(b, len)?;
                let s = str::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)?;
                Ok(<$t>::from(s))
            }
            fn encode_size(&self) -> usize {
                4 + self.as_bytes().len()
            }
        }
        impl OrderedCodec for $t {
            fn encode_ordered(&self, out: &mut impl BufMut) {
                put_escaped(self.as_bytes(), out);
            }
            fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
                let bytes = take_escaped(b)?;
                let s = str::from_utf8(&bytes).map_err(|_| DecodeError::InvalidUtf8)?;
                Ok(<$t>::from(s))
            }
            fn encode_ordered_size(&self) -> usize {
                escaped_size(self.as_bytes())
            }
        }
    )*};
}
impl_str_codec!(String, Box<str>, std::rc::Rc<str>, std::sync::Arc<str>);

macro_rules! impl_tuple_codec {
    ($($name:ident $idx:tt),+) => {
        impl<$($name: ResidentSize),+> ResidentSize for ($($name,)+) {
            fn resident_size(&self) -> usize {
                size_of::<Self>() $( + heap_size(&self.$idx) )+
            }
        }
        // a tuple encodes as its fields, one after another
        impl<$($name: Layout),+> Layout for ($($name,)+) {
            fn layout(l: &mut LayoutHash) { $( $name::layout(l); )+ }
        }
        impl<$($name: Codec),+> Codec for ($($name,)+) {
            fn encode(&self, out: &mut impl BufMut) { $( self.$idx.encode(out); )+ }
            fn decode(b: &mut &[u8]) -> Result<Self> {
                Ok(($( <$name as Codec>::decode(b)?, )+))
            }
            fn encode_size(&self) -> usize {
                0 $( + self.$idx.encode_size() )+
            }
        }
        impl<$($name: OrderedCodec),+> OrderedCodec for ($($name,)+) {
            fn encode_ordered(&self, out: &mut impl BufMut) { $( self.$idx.encode_ordered(out); )+ }
            fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
                Ok(($( <$name as OrderedCodec>::decode_ordered(b)?, )+))
            }
            fn encode_ordered_size(&self) -> usize {
                0 $( + self.$idx.encode_ordered_size() )+
            }
        }
    };
}
impl_tuple_codec!(A 0);
impl_tuple_codec!(A 0, B 1);
impl_tuple_codec!(A 0, B 1, C 2);
impl_tuple_codec!(A 0, B 1, C 2, D 3);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
// std only implements Ord to arity 12
impl_tuple_codec!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

impl<T: ResidentSize> ResidentSize for std::cmp::Reverse<T> {
    fn resident_size(&self) -> usize {
        self.0.resident_size()
    }
}

/// Stored as `T` is.
impl<T: Codec> Codec for std::cmp::Reverse<T> {
    fn encode(&self, out: &mut impl BufMut) {
        self.0.encode(out)
    }
    fn decode(b: &mut &[u8]) -> Result<Self> {
        T::decode(b).map(std::cmp::Reverse)
    }
    fn encode_size(&self) -> usize {
        self.0.encode_size()
    }
}

/// `T`'s ordered bytes, each complemented, so keys sort in reverse.
///
/// Every ordered encoding is self-delimiting (its decoder knows where it
/// ends), so no value's bytes are a proper prefix of another's: two
/// values first differ at a byte inside both, and complementing that
/// byte reverses the comparison, whatever follows in a composite key.
impl<T: OrderedCodec> OrderedCodec for std::cmp::Reverse<T> {
    fn encode_ordered(&self, out: &mut impl BufMut) {
        let mut inner = Vec::with_capacity(self.0.encode_ordered_size());
        self.0.encode_ordered(&mut inner);
        for byte in &mut inner {
            *byte = !*byte;
        }
        out.put_slice(&inner);
    }
    fn decode_ordered(b: &mut &[u8]) -> Result<Self> {
        // T's decoder reads plain bytes and stops where its encoding
        // ends, so un-complement what follows and advance by what it took
        let plain: Vec<u8> = b.iter().map(|byte| !byte).collect();
        let mut rest = &plain[..];
        let v = T::decode_ordered(&mut rest)?;
        *b = &b[plain.len() - rest.len()..];
        Ok(std::cmp::Reverse(v))
    }
    fn encode_ordered_size(&self) -> usize {
        self.0.encode_ordered_size()
    }
}

#[cfg(test)]
mod reverse_tests {
    use std::cmp::Reverse;

    use super::*;

    fn ordered<T: OrderedCodec>(v: &T) -> Vec<u8> {
        let mut out = Vec::new();
        v.encode_ordered(&mut out);
        assert_eq!(out.len(), v.encode_ordered_size());
        out
    }

    #[test]
    fn sorts_backwards_and_round_trips_inside_a_key() {
        let mut keys: Vec<((u8, Reverse<i64>), String)> = Vec::new();
        for price in [-5i64, 0, 3, 3, 900, i64::MIN, i64::MAX] {
            for name in ["", "a", "ab"] {
                keys.push(((1, Reverse(price)), name.to_string()));
            }
        }
        let mut by_bytes = keys.clone();
        by_bytes.sort_by_key(ordered);
        keys.sort();
        assert_eq!(by_bytes, keys);
        for k in &keys {
            assert_eq!(
                decode_all_ordered::<((u8, Reverse<i64>), String)>(&ordered(k)).unwrap(),
                *k
            );
        }

        let names = [
            Reverse("".to_string()),
            Reverse("a".into()),
            Reverse("a\0".into()),
            Reverse("b".into()),
        ];
        let mut by_bytes = names.to_vec();
        by_bytes.sort_by_key(ordered);
        let mut sorted = names.to_vec();
        sorted.sort();
        assert_eq!(by_bytes, sorted);
        assert_eq!(
            decode_all::<Reverse<u32>>(&{
                let mut o = Vec::new();
                Reverse(7u32).encode(&mut o);
                o
            })
            .unwrap(),
            Reverse(7)
        );
    }
}

#[cfg(test)]
mod resident_size_tests {
    use super::*;

    #[test]
    fn counts_inline_bytes_once_and_heap_bytes_in_full() {
        assert_eq!(7u64.resident_size(), 8);
        assert_eq!(().resident_size(), 0);
        // padding is in the tuple's own size, not double counted
        assert_eq!((1u8, 2u64).resident_size(), size_of::<(u8, u64)>());

        let s = String::with_capacity(100);
        assert_eq!(s.resident_size(), size_of::<String>() + 100);
        let some = Some("x".repeat(10));
        let cap = some.as_ref().unwrap().capacity();
        assert_eq!(some.resident_size(), size_of::<Option<String>>() + cap);

        let v: Vec<String> = vec!["abc".into(), "de".into()];
        let want = size_of::<Vec<String>>()
            + v.capacity() * size_of::<String>()
            + v.iter().map(String::capacity).sum::<usize>();
        assert_eq!(v.resident_size(), want);

        let pair = (Box::<str>::from("hello"), 3i32);
        assert_eq!(pair.resident_size(), size_of::<(Box<str>, i32)>() + 5);
    }
}
