//! Opening an existing directory: loading tables and replaying the WAL.

use super::*;

/// Delete WAL segments and checkpoints the manifest does not name, and any
/// half-written manifest. A crash between creating a file and naming it,
/// or between dropping a file from the manifest and deleting it, leaves
/// these behind. Files with other names are left alone.
pub(super) fn sweep_orphans(dir: &Path, m: &Manifest) -> Result<()> {
    let live: HashSet<String> = m.live_files().collect();
    for entry in fs::read_dir(dir).ctx("listing", dir)? {
        let entry = entry.ctx("listing", dir)?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let numbered = name.split_once('.').is_some_and(|(num, ext)| {
            matches!(ext, "wal" | "ckpt")
                && !num.is_empty()
                && num.bytes().all(|b| b.is_ascii_digit())
        });
        if (numbered && !live.contains(name)) || name == MANIFEST_TMP {
            fs::remove_file(entry.path()).ctx("removing orphan", &entry.path())?;
        }
    }
    Ok(())
}

/// Everything replay recovered.
#[derive(Default)]
pub(super) struct Replayed {
    /// The newest epoch recovered, from the WAL or else the last checkpoint.
    pub(super) epoch: u64,
    /// Everything in the WAL, folded into one tree per keyspace.
    pub(super) wal: HashMap<u64, Box<dyn Any + Send + Sync>>,
    /// Valid length of the active segment, if it ends in a torn frame.
    pub(super) torn_active: Option<u64>,
}

pub(super) fn corrupt(path: &Path, offset: usize, why: impl Into<String>) -> Error {
    Error::corrupt(path, offset as u64, why)
}

pub(super) fn frame_error(path: &Path, offset: usize) -> impl Fn(FrameError) -> Error + '_ {
    move |e| match e {
        FrameError::Unknown(id) => Error::UnknownKeyspace(id),
        FrameError::Decode(why) => corrupt(path, offset, why),
    }
}

/// Open every checkpoint's table. Only footers and indexes are read; data
/// blocks are paged in through the cache as reads need them.
pub(super) fn open_tables(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
    dropped: &BTreeSet<u64>,
) -> Result<Vec<Arc<Table>>> {
    let mut tables = Vec::with_capacity(m.checkpoints.len());
    for ckpt in &m.checkpoints {
        let path = dir.join(ckpt.file_name());
        let table = Table::open(&path, ckpt.number)?;
        if table.epoch() != ckpt.epoch {
            return Err(corrupt(
                &path,
                0,
                format!(
                    "holds epoch {}, manifest says {}",
                    table.epoch(),
                    ckpt.epoch
                ),
            ));
        }
        let unknown = |ks: &u64| {
            !decls.contains_key(ks) && !dropped.contains(ks) && m.floor(*ks) < table.epoch()
        };
        if let Some(ks) = table.keyspaces().find(unknown) {
            return Err(Error::UnknownKeyspace(ks));
        }
        tables.push(Arc::new(table));
    }
    Ok(tables)
}

/// Replay every WAL segment, folding all frames into one tree per
/// keyspace. Nothing can hold a snapshot during open, so one layer is
/// equivalent to a layer per epoch, and reads stay cheap. The WAL holds
/// only epochs after the last checkpoint, so replay starts there.
pub(super) fn replay(
    dir: &Path,
    m: &Manifest,
    decls: &HashMap<u64, KeyspaceDecl>,
    dropped: &BTreeSet<u64>,
) -> Result<Replayed> {
    let mut out = Replayed {
        epoch: m.checkpointed(),
        ..Replayed::default()
    };
    for seg in &m.segments {
        let path = dir.join(seg.file_name());
        let bytes = fs::read(&path).ctx("reading WAL", &path)?;
        let limit = match seg.sealed {
            Some(len) if (len as usize) > bytes.len() => {
                return Err(corrupt(
                    &path,
                    bytes.len(),
                    format!("sealed at {len} bytes but shorter"),
                ));
            }
            Some(len) => len as usize,
            None => bytes.len(),
        };

        let mut at = 0;
        while at < limit {
            let (epoch, payload, used) = match parse_frame(&bytes[at..limit]) {
                Ok(frame) => frame,
                // a crash mid-append leaves a partial or unchecksummed frame
                // at the end of the active segment; everything before it is good
                Err(_) if seg.sealed.is_none() => {
                    out.torn_active = Some(at as u64);
                    break;
                }
                Err(why) => return Err(corrupt(&path, at, why)),
            };
            if epoch != out.epoch + 1 {
                return Err(corrupt(
                    &path,
                    at,
                    format!("epoch {epoch} follows epoch {}", out.epoch),
                ));
            }
            // truncated or dropped keyspaces' sections are skipped
            let skip = |ks: u64| dropped.contains(&ks) || m.floor(ks) >= epoch;
            apply_frame(payload, decls, &skip, &mut out.wal).map_err(frame_error(&path, at))?;
            out.epoch = epoch;
            at += used;
        }
    }
    Ok(out)
}

/// Apply one checksummed frame: repeated `keyspace u64 | len u64 | body`.
pub(super) fn apply_frame(
    mut payload: &[u8],
    decls: &HashMap<u64, KeyspaceDecl>,
    skip: &dyn Fn(u64) -> bool,
    trees: &mut HashMap<u64, Box<dyn Any + Send + Sync>>,
) -> Result<(), FrameError> {
    let bad = |why: &str| FrameError::Decode(why.to_string());
    while !payload.is_empty() {
        if payload.len() < 16 {
            return Err(bad("truncated section header"));
        }
        let ks = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let len = u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let end = usize::try_from(len)
            .ok()
            .and_then(|len| 16usize.checked_add(len))
            .filter(|&end| end <= payload.len())
            .ok_or_else(|| bad("truncated section body"))?;
        if skip(ks) {
            payload = &payload[end..];
            continue;
        }
        let decl = decls.get(&ks).ok_or(FrameError::Unknown(ks))?;
        let mut tree = trees.remove(&ks);
        let res = (decl.replay)(&mut tree, &payload[16..end]);
        if let Some(tree) = tree {
            trees.insert(ks, tree);
        }
        res.map_err(|e| FrameError::Decode(format!("keyspace {ks}: {e}")))?;
        payload = &payload[end..];
    }
    Ok(())
}
