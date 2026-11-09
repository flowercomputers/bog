//! The store's error type.

use std::{
    fmt, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::tx::OutOfMemory;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything that can go wrong in the store. Cheap to clone, so a failed
/// transaction can hand the same error to every later call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// An I/O operation failed. `context` says what was being done to what.
    Io {
        context: String,
        source: Arc<io::Error>,
    },
    /// A file's contents are damaged somewhere a crash cannot explain.
    Corrupt {
        file: PathBuf,
        offset: u64,
        why: String,
    },
    /// Another open store holds this directory.
    Locked(PathBuf),
    /// The durability policy is invalid, or needs a directory.
    Policy(&'static str),
    /// A size option is out of range.
    InvalidOption(&'static str),
    /// The manifest could not be parsed.
    Manifest(String),
    /// A keyspace is registered with different types than it was written with.
    KeyspaceType {
        id: u64,
        recorded: (String, String),
        registered: (&'static str, &'static str),
    },
    /// The store holds data for a keyspace that was not registered.
    UnknownKeyspace(u64),
    /// A write transaction outgrew its memory limit and was killed.
    OutOfMemory(OutOfMemory),
    /// The application rejected a write transaction, typically from an
    /// operator enforcing a constraint. Carries the application's error;
    /// downcast it to recover the original.
    Rejected(Arc<dyn std::error::Error + Send + Sync>),
    /// An earlier failure left what is on disk uncertain, so the store
    /// refuses writes until it is reopened, which recovers from disk.
    Poisoned(Arc<Error>),
    /// Derived state was computed by a different graph, or a derived
    /// keyspace was added or dropped, and must be rebuilt, but the store
    /// was opened without a replay to rebuild it with.
    RebuildRequired,
    /// The transaction committed as `epoch`, but maintenance that ran
    /// after it failed. The commit stands; the next commit retries.
    AfterCommit { epoch: u64, source: Arc<Error> },
}

impl Error {
    pub(crate) fn io(context: impl Into<String>, source: io::Error) -> Self {
        Error::Io {
            context: context.into(),
            source: Arc::new(source),
        }
    }

    pub(crate) fn corrupt(file: &Path, offset: u64, why: impl Into<String>) -> Self {
        Error::Corrupt {
            file: file.to_path_buf(),
            offset,
            why: why.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Corrupt { file, offset, why } => {
                write!(f, "{} is corrupt at byte {offset}: {why}", file.display())
            }
            Self::Locked(dir) => write!(f, "{} is locked by another open store", dir.display()),
            Self::Policy(why) => write!(f, "durability policy: {why}"),
            Self::InvalidOption(why) => write!(f, "invalid option: {why}"),
            Self::Manifest(why) => write!(f, "bad manifest: {why}"),
            Self::KeyspaceType {
                id,
                recorded,
                registered,
            } => write!(
                f,
                "keyspace {id} was written as {recorded:?} but is registered as {registered:?}"
            ),
            Self::UnknownKeyspace(id) => {
                write!(f, "the store holds keyspace {id}, which is not registered")
            }
            Self::OutOfMemory(oom) => oom.fmt(f),
            Self::Rejected(why) => write!(f, "transaction rejected: {why}"),
            Self::RebuildRequired => write!(
                f,
                "derived state must be rebuilt for a changed graph; open with a replay"
            ),
            Self::Poisoned(cause) => write!(f, "store refuses writes until reopened: {cause}"),
            Self::AfterCommit { epoch, source } => {
                write!(
                    f,
                    "committed epoch {epoch}, but maintenance after it failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(&**source),
            Self::Poisoned(cause) => Some(&**cause),
            Self::AfterCommit { source, .. } => Some(&**source),
            Self::OutOfMemory(oom) => Some(oom),
            Self::Rejected(why) => Some(&**why),
            _ => None,
        }
    }
}

/// Attach what was being done, and to which file, to an I/O error.
pub(crate) trait IoContext<T> {
    fn ctx(self, doing: &str, path: &Path) -> Result<T>;
}

impl<T> IoContext<T> for io::Result<T> {
    fn ctx(self, doing: &str, path: &Path) -> Result<T> {
        self.map_err(|e| Error::io(format!("{doing} {}", path.display()), e))
    }
}
