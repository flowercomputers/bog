//! The write-ahead log: framing, the write buffer and segments.

use super::*;

/// The most the WAL's write buffer keeps allocated once it is written
/// out.
pub(super) const WAL_BUFFER_RETAIN: usize = 1 << 20;

pub(super) struct Wal {
    pub(super) path: PathBuf,
    pub(super) file: File,
    /// Make the next write fail, to test what a failed write does.
    #[cfg(test)]
    pub(super) fail_next_write: bool,
    pub(super) buffer: Vec<u8>,
    pub(super) size: usize,
}
impl Wal {
    /// Buffer one frame holding a whole transaction, so replay sees either
    /// all of a tx or none of it: `crc32 | epoch u64 | len u64 | payload`,
    /// little endian, with the crc covering everything after it. `payload`
    /// writes straight into the buffer, so the tx is never copied whole.
    pub(super) fn append_with(&mut self, epoch: u64, payload: impl FnOnce(&mut Vec<u8>)) {
        const HEAD: usize = 4 + 8 + 8;
        let start = self.buffer.len();
        self.buffer.extend([0u8; HEAD]);
        payload(&mut self.buffer);
        let len = (self.buffer.len() - start - HEAD) as u64;
        self.buffer[start + 4..start + 12].copy_from_slice(&epoch.to_le_bytes());
        self.buffer[start + 12..start + 20].copy_from_slice(&len.to_le_bytes());
        let crc = crc32fast::hash(&self.buffer[start + 4..]);
        self.buffer[start..start + 4].copy_from_slice(&crc.to_le_bytes());
        self.size += self.buffer.len() - start;
    }

    pub(super) fn flush(&mut self, write: bool, sync: bool) -> Result<()> {
        #[cfg(test)]
        if write && std::mem::take(&mut self.fail_next_write) {
            return Err(std::io::Error::other("injected write failure"))
                .ctx("writing WAL", &self.path);
        }
        if write && !self.buffer.is_empty() {
            self.file
                .write_all(&self.buffer)
                .ctx("writing WAL", &self.path)?;
            self.buffer.clear();
            // one large tx should not leave its size allocated for good
            if self.buffer.capacity() > WAL_BUFFER_RETAIN {
                self.buffer = Vec::new();
            }
        }
        if sync {
            self.file.sync_data().ctx("syncing WAL", &self.path)?;
        }
        Ok(())
    }
}
/// Append one payload section, `keyspace u64 | len u64 | body`, with the
/// body written by `body`.
pub(super) fn encode_section<R>(
    ks: u64,
    out: &mut Vec<u8>,
    body: impl FnOnce(&mut Vec<u8>) -> R,
) -> R {
    out.extend(ks.to_le_bytes());
    let len_at = out.len();
    out.extend(0u64.to_le_bytes());
    let r = body(out);
    let len = (out.len() - len_at - 8) as u64;
    out[len_at..len_at + 8].copy_from_slice(&len.to_le_bytes());
    r
}

impl Drop for Wal {
    /// A clean shutdown hands buffered frames to the OS. It does not fsync:
    /// that is the policy's call.
    fn drop(&mut self) {
        // nowhere to report a failure; the next open replays what landed
        let _ = self.flush(true, false);
    }
}

/// Create the next WAL segment and list it in `m` as active. The file is
/// created before the manifest names it; the manifest write syncs the
/// directory entry.
pub(super) fn new_segment(dir: &Path, m: &mut Manifest) -> Result<Wal> {
    let seg = Segment {
        number: m.next_file,
        sealed: None,
    };
    let path = dir.join(seg.file_name());
    let file = OpenOptions::new()
        .append(true)
        .create_new(true)
        .open(&path)
        .ctx("creating WAL", &path)?;
    m.next_file += 1;
    m.segments.push(seg);
    Ok(Wal {
        path,
        file,
        #[cfg(test)]
        fail_next_write: false,
        buffer: Vec::new(),
        size: 0,
    })
}

/// Split one frame off the front of `b`: `(epoch, payload, bytes used)`.
pub(super) fn parse_frame(b: &[u8]) -> Result<(u64, &[u8], usize), String> {
    const HEAD: usize = 4 + 8 + 8;
    if b.len() < HEAD {
        return Err("truncated frame header".into());
    }
    let crc = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let epoch = u64::from_le_bytes(b[4..12].try_into().unwrap());
    let len = u64::from_le_bytes(b[12..20].try_into().unwrap());
    let end = usize::try_from(len)
        .ok()
        .and_then(|len| HEAD.checked_add(len))
        .filter(|&end| end <= b.len())
        .ok_or("truncated frame payload")?;
    if crc32fast::hash(&b[4..end]) != crc {
        return Err("frame checksum mismatch".into());
    }
    Ok((epoch, &b[HEAD..end], end))
}

pub(super) enum FrameError {
    Unknown(u64),
    Decode(String),
}
