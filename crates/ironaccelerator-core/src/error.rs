//! Unified error type. Backend errors are erased to a small enum + opaque code
//! so the hot path never allocates.

use core::fmt;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Backend not compiled in or runtime library not found.
    BackendUnavailable(&'static str),
    /// Operation is not supported by the selected device / driver.
    Unsupported(&'static str),
    /// Out of device or pinned host memory.
    OutOfMemory,
    /// Kernel launch failed; opaque driver code stored verbatim.
    LaunchFailure { code: i32 },
    /// Misuse: violated an `_unchecked` precondition.
    InvalidArgument(&'static str),
    /// Backend-specific error wrapped as a code so we don't allocate on hot
    /// paths. Backends keep their own decode tables.
    Backend {
        backend: crate::BackendKind,
        code: i64,
    },
    /// Anything else.
    Other(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BackendUnavailable(b) => write!(f, "backend `{b}` unavailable"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::OutOfMemory => f.write_str("out of memory"),
            Error::LaunchFailure { code } => write!(f, "kernel launch failed (code {code})"),
            Error::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Error::Backend { backend, code } => write!(f, "{backend:?} error {code}"),
            Error::Other(s) => f.write_str(s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T, E = Error> = core::result::Result<T, E>;
