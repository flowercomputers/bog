//! State only the writer touches, and the guard a write transaction holds on it.

use super::*;

/// State only the writer touches. Held locked for the life of a
/// [`WriteTx`], which is what makes the store single-writer.
pub(crate) struct StoreWriter {
    /// `None` for an in-memory store.
    pub(super) dir: Option<PathBuf>,
    /// The manifest as last written. Only open and checkpoint change it,
    /// and checkpoint does so under this lock.
    pub(super) manifest: Manifest,
    pub(super) wal: Option<Wal>,
    pub(super) dp: DurabilityPolicy,
    pub(super) last_write: Option<Instant>,
    pub(super) last_fsync: Option<Instant>,
    /// Set when a failure leaves what is on disk uncertain; every later
    /// write fails with it until the store is reopened.
    pub(super) poisoned: Option<Error>,
}
impl StoreWriter {
    pub(super) fn new(
        dir: Option<PathBuf>,
        manifest: Manifest,
        wal: Option<Wal>,
        dp: DurabilityPolicy,
    ) -> Self {
        StoreWriter {
            dir,
            manifest,
            wal,
            dp,
            last_write: None,
            last_fsync: None,
            poisoned: None,
        }
    }

    pub(super) fn check(&self) -> Result<()> {
        match &self.poisoned {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Refuse every later write: `cause` left what is on disk uncertain.
    ///
    /// Buffered WAL frames are discarded rather than written on drop: after
    /// a failed write the file may end in a partial frame, and replay stops
    /// there, so nothing appended after it would be seen anyway, and the tx
    /// that failed must not reappear on reopen.
    pub(super) fn poison(&mut self, cause: Error) -> Error {
        if let Some(wal) = self.wal.as_mut() {
            wal.buffer.clear();
        }

        self.poisoned
            .get_or_insert(Error::Poisoned(Arc::new(cause)))
            .clone()
    }

    /// Make `next` the live manifest. A failure that leaves the old one
    /// live is an ordinary error; one that leaves it unknown which survives
    /// a crash poisons the store.
    pub(super) fn install_manifest(&mut self, dir: &Path, next: Manifest) -> Result<()> {
        manifest::stage(dir, &next).ctx("writing manifest in", dir)?;
        match manifest::install(dir) {
            Ok(()) => {}
            Err(InstallError::Unchanged(e)) => {
                return Err(Error::io(
                    format!("installing manifest in {}", dir.display()),
                    e,
                ));
            }
            Err(InstallError::Uncertain(e)) => {
                let cause = Error::io(format!("syncing {}", dir.display()), e);
                return Err(self.poison(cause));
            }
        }
        self.manifest = next;
        Ok(())
    }

    /// Make `next` the manifest: installed on disk, or just kept for an
    /// in-memory store.
    pub(super) fn update_manifest(&mut self, next: Manifest) -> Result<()> {
        match self.dir.clone() {
            Some(dir) => self.install_manifest(&dir, next),
            None => {
                self.manifest = next;
                Ok(())
            }
        }
    }

    /// Log one transaction, its payload written by `payload` (given `None`
    /// when there is no WAL), then write/fsync as the policy says. Returns
    /// the bytes appended to the WAL.
    // todo: windowed policies only flush when the next tx commits; a quiet
    // store needs a clock source to flush on its own.
    ///
    /// A failed write or fsync poisons the store: the frame, and frames
    /// buffered before it, may or may not be on disk.
    pub(super) fn log_with(
        &mut self,
        epoch: u64,
        payload: impl FnOnce(Option<&mut Vec<u8>>),
    ) -> Result<u64> {
        let Some(wal) = self.wal.as_mut() else {
            payload(None);
            return Ok(0);
        };
        let before = wal.size;
        wal.append_with(epoch, |buf| payload(Some(buf)));
        let appended = (wal.size - before) as u64;
        let now = Instant::now();
        let (write, sync) = self.dp.should(now, self.last_write, self.last_fsync);
        if let Err(e) = wal.flush(write, sync) {
            return Err(self.poison(e));
        }
        if write {
            self.last_write = Some(now);
        }
        if sync {
            // to accurately reflect the time it actually took to fsync
            self.last_fsync = Some(Instant::now());
        }
        Ok(appended)
    }
}

/// A number unique to the calling thread, never 0.
pub(super) fn thread_tag() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local!(static TAG: u64 = NEXT.fetch_add(1, Relaxed));
    TAG.with(|t| *t)
}

/// The writer lock as a write tx holds it: records the holding thread
/// while it lives.
pub(crate) struct WriterGuard<'a> {
    pub(super) guard: MutexGuard<'a, StoreWriter>,
    pub(super) owner: &'a AtomicU64,
}

impl Deref for WriterGuard<'_> {
    type Target = StoreWriter;
    fn deref(&self) -> &StoreWriter {
        &self.guard
    }
}

impl std::ops::DerefMut for WriterGuard<'_> {
    fn deref_mut(&mut self) -> &mut StoreWriter {
        &mut self.guard
    }
}

impl Drop for WriterGuard<'_> {
    fn drop(&mut self) {
        self.owner.store(0, Relaxed);
    }
}
