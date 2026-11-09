//! The manifest: a small text file naming the store's config, its declared
//! keyspaces, and its live files. It is the root of recovery. Open reads it
//! to learn which WAL segments to replay and what types to decode them as,
//! and any file in the directory it does not name is ignored.
//!
//! It is written whole to `MANIFEST.tmp`, fsynced, and renamed over
//! `MANIFEST`, so a crash leaves either the old manifest or the new one.
//!
//! Format: one record per line, fields separated by tabs (type names can
//! contain spaces but never tabs).
//!
//! ```text
//! fold3 manifest v1
//! policy	write=always	fsync=every:1000000000ns
//! next-file	3
//! option	wal-limit	67108864
//! keyspace	7	u8	(i64, i64)
//! checkpoint	1	40
//! wal	2	sealed	1234
//! wal	4	active
//! ```

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    time::Duration,
};

use crate::store::{DurabilityPolicy, DurabilityPolicyCadence};

pub(crate) const MANIFEST: &str = "MANIFEST";
pub(crate) const MANIFEST_TMP: &str = "MANIFEST.tmp";
const HEADER: &str = "fold3 manifest v1";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Manifest {
    /// The policy the store was last opened with. Informational: the
    /// builder's policy always wins on open.
    pub policy: DurabilityPolicy,
    /// Number for the next file created in the directory.
    pub next_file: u64,
    /// The size limits the store was last opened with, by name.
    /// Informational, like the policy.
    pub options: BTreeMap<String, u64>,
    /// Keyspace id to (key type, merge type), as `std::any::type_name`.
    pub keyspaces: BTreeMap<u64, (String, String)>,
    /// Checkpoints oldest first. Each holds what changed after the previous
    /// one, through its own epoch; loading them in order rebuilds the state
    /// as of the last one.
    pub checkpoints: Vec<Checkpoint>,
    /// WAL segments in replay order, holding only epochs after the last
    /// checkpoint. Only the last may be active.
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Checkpoint {
    pub number: u64,
    /// The newest epoch it covers.
    pub epoch: u64,
}

impl Checkpoint {
    pub fn file_name(&self) -> String {
        format!("{:06}.ckpt", self.number)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Segment {
    pub number: u64,
    /// `Some(len)` once sealed: exactly the first `len` bytes are valid
    /// frames and anything after is a torn tail to ignore. `None` while it
    /// is the active segment being appended to.
    pub sealed: Option<u64>,
}

impl Segment {
    pub fn file_name(&self) -> String {
        format!("{:06}.wal", self.number)
    }
}

impl Manifest {
    pub fn new(policy: DurabilityPolicy) -> Self {
        Manifest {
            policy,
            next_file: 1,
            options: BTreeMap::new(),
            keyspaces: BTreeMap::new(),
            checkpoints: Vec::new(),
            segments: Vec::new(),
        }
    }

    /// The newest epoch covered by a checkpoint, 0 if there are none.
    pub fn checkpointed(&self) -> u64 {
        self.checkpoints.last().map_or(0, |c| c.epoch)
    }

    /// Every file the manifest keeps alive.
    pub fn live_files(&self) -> impl Iterator<Item = String> + '_ {
        let ckpts = self.checkpoints.iter().map(Checkpoint::file_name);
        ckpts.chain(self.segments.iter().map(Segment::file_name))
    }

    /// The active segment, if the last one is still open for appends.
    pub fn active(&self) -> Option<Segment> {
        self.segments.last().copied().filter(|s| s.sealed.is_none())
    }

    pub fn encode(&self) -> String {
        let mut s = String::new();
        s.push_str(HEADER);
        s.push('\n');
        s.push_str(&format!(
            "policy\twrite={}\tfsync={}\n",
            encode_cadence(&self.policy.write),
            encode_cadence(&self.policy.fsync)
        ));
        s.push_str(&format!("next-file\t{}\n", self.next_file));
        for (name, value) in &self.options {
            s.push_str(&format!("option\t{name}\t{value}\n"));
        }
        for (id, (key, merge)) in &self.keyspaces {
            s.push_str(&format!("keyspace\t{id}\t{key}\t{merge}\n"));
        }
        for c in &self.checkpoints {
            s.push_str(&format!("checkpoint\t{}\t{}\n", c.number, c.epoch));
        }
        for seg in &self.segments {
            match seg.sealed {
                Some(len) => s.push_str(&format!("wal\t{}\tsealed\t{len}\n", seg.number)),
                None => s.push_str(&format!("wal\t{}\tactive\n", seg.number)),
            }
        }
        s
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().enumerate().filter(|(_, l)| !l.is_empty());
        match lines.next() {
            Some((_, HEADER)) => {}
            Some((_, other)) => return Err(format!("unknown manifest header {other:?}")),
            None => return Err("empty manifest".into()),
        }

        let mut policy = None;
        let mut next_file = None;
        let mut keyspaces = BTreeMap::new();
        let mut options = BTreeMap::new();
        let mut checkpoints: Vec<Checkpoint> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();

        for (n, line) in lines {
            let at = |why: String| format!("manifest line {}: {why}", n + 1);
            let fields: Vec<&str> = line.split('\t').collect();
            match fields.as_slice() {
                ["policy", write, fsync] => {
                    let write = write
                        .strip_prefix("write=")
                        .ok_or_else(|| at("bad write".into()))?;
                    let fsync = fsync
                        .strip_prefix("fsync=")
                        .ok_or_else(|| at("bad fsync".into()))?;
                    let parsed = DurabilityPolicy {
                        write: parse_cadence(write).map_err(at)?,
                        fsync: parse_cadence(fsync).map_err(at)?,
                    };
                    if policy.replace(parsed).is_some() {
                        return Err(at("duplicate policy".into()));
                    }
                }
                ["next-file", num] => {
                    if next_file.replace(parse_u64(num).map_err(at)?).is_some() {
                        return Err(at("duplicate next-file".into()));
                    }
                }
                ["option", name, value] => {
                    let value = parse_u64(value).map_err(at)?;
                    if options.insert(name.to_string(), value).is_some() {
                        return Err(at(format!("duplicate option {name}")));
                    }
                }
                ["keyspace", id, key, merge] => {
                    let id = parse_u64(id).map_err(at)?;
                    if keyspaces
                        .insert(id, (key.to_string(), merge.to_string()))
                        .is_some()
                    {
                        return Err(at(format!("duplicate keyspace {id}")));
                    }
                }
                ["checkpoint", num, epoch] => {
                    let c = Checkpoint {
                        number: parse_u64(num).map_err(at)?,
                        epoch: parse_u64(epoch).map_err(at)?,
                    };
                    if let Some(prev) = checkpoints.last()
                        && (prev.number >= c.number || prev.epoch >= c.epoch)
                    {
                        return Err(at("checkpoints out of order".into()));
                    }
                    checkpoints.push(c);
                }
                ["wal", num, rest @ ..] => {
                    let number = parse_u64(num).map_err(at)?;
                    let sealed = match rest {
                        ["active"] => None,
                        ["sealed", len] => Some(parse_u64(len).map_err(at)?),
                        _ => return Err(at("bad wal record".into())),
                    };
                    if let Some(prev) = segments.last() {
                        if prev.sealed.is_none() {
                            return Err(at("only the last wal segment may be active".into()));
                        }
                        if prev.number >= number {
                            return Err(at("wal segments out of order".into()));
                        }
                    }
                    segments.push(Segment { number, sealed });
                }
                _ => return Err(at(format!("unknown record {line:?}"))),
            }
        }

        let manifest = Manifest {
            policy: policy.ok_or("manifest has no policy")?,
            next_file: next_file.ok_or("manifest has no next-file")?,
            options,
            keyspaces,
            checkpoints,
            segments,
        };
        let newest = |n: Option<u64>| n.is_some_and(|n| n >= manifest.next_file);
        if newest(manifest.segments.last().map(|s| s.number))
            || newest(manifest.checkpoints.last().map(|c| c.number))
        {
            return Err("file numbered at or past next-file".into());
        }
        Ok(manifest)
    }
}

/// The manifest in `dir`, or `None` if this is a new store.
pub(crate) fn read(dir: &Path) -> io::Result<Option<String>> {
    match fs::read_to_string(dir.join(MANIFEST)) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Atomically replace the manifest in `dir`.
pub(crate) fn write(dir: &Path, manifest: &Manifest) -> io::Result<()> {
    let tmp = dir.join(MANIFEST_TMP);
    let mut f = File::create(&tmp)?;
    f.write_all(manifest.encode().as_bytes())?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, dir.join(MANIFEST))?;
    // persists the rename, and any file created since the last dir sync
    sync_dir(dir)
}

pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

fn encode_cadence(c: &DurabilityPolicyCadence) -> String {
    match c {
        DurabilityPolicyCadence::Never => "never".into(),
        DurabilityPolicyCadence::Always => "always".into(),
        DurabilityPolicyCadence::Windowed(d) => format!("every:{}ns", d.as_nanos()),
    }
}

fn parse_cadence(s: &str) -> Result<DurabilityPolicyCadence, String> {
    match s {
        "never" => Ok(DurabilityPolicyCadence::Never),
        "always" => Ok(DurabilityPolicyCadence::Always),
        _ => {
            let nanos = s
                .strip_prefix("every:")
                .and_then(|s| s.strip_suffix("ns"))
                .ok_or_else(|| format!("bad cadence {s:?}"))?;
            Ok(DurabilityPolicyCadence::Windowed(Duration::from_nanos(
                parse_u64(nanos)?,
            )))
        }
    }
}

fn parse_u64(s: &str) -> Result<u64, String> {
    s.parse().map_err(|_| format!("bad number {s:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut m = Manifest::new(DurabilityPolicy::periodic_sync(Duration::from_millis(1500)));
        m.next_file = 3;
        m.options.insert("wal-limit".into(), 1 << 26);
        m.keyspaces.insert(7, ("u8".into(), "(i64, i64)".into()));
        m.checkpoints.push(Checkpoint {
            number: 0,
            epoch: 9,
        });
        m.segments.push(Segment {
            number: 1,
            sealed: Some(1234),
        });
        m.segments.push(Segment {
            number: 2,
            sealed: None,
        });
        assert_eq!(Manifest::parse(&m.encode()), Ok(m));
    }

    #[test]
    fn rejects_bad_manifests() {
        let ok = "fold3 manifest v1\npolicy\twrite=always\tfsync=never\nnext-file\t3\n";
        assert!(Manifest::parse(ok).is_ok());
        assert!(Manifest::parse("fold3 manifest v2\n").is_err());
        assert!(Manifest::parse(&format!("{ok}wal\t1\tactive\nwal\t2\tactive\n")).is_err());
        assert!(Manifest::parse(&format!("{ok}wal\t2\tsealed\t0\nwal\t1\tactive\n")).is_err());
        assert!(Manifest::parse(&format!("{ok}wal\t3\tactive\n")).is_err());
        assert!(Manifest::parse(&format!("{ok}bogus\n")).is_err());
        assert!(Manifest::parse(&format!("{ok}checkpoint\t1\t5\ncheckpoint\t2\t5\n")).is_err());
        assert!(Manifest::parse(&format!("{ok}checkpoint\t3\t5\n")).is_err());
    }
}
