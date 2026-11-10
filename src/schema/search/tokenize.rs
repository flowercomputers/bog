//! Tokenization and term interning.

/// Splits text into terms. Stateless and deterministic by construction:
/// the same text must tokenize the same way at every rebuild.
pub trait Tokenizer: Send + Sync + 'static {
    /// Folded into the keyspace names; change it when the tokenization
    /// changes so a persisted index is rebuilt.
    const NAME: &'static str;
    fn tokenize(text: &str) -> Vec<String>;
}

/// Lowercase, split on anything that is not alphanumeric.
pub struct Simple;

impl Tokenizer for Simple {
    const NAME: &'static str = "simple";
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect()
    }
}

/// A term's interned id: FNV-1a over its bytes. Stateless, so a rebuild
/// reproduces it, and 64 bits wide, so collisions are not a practical
/// concern for any vocabulary.
pub fn intern(term: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in term.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
