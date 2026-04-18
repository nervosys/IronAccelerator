//! Dynamic library loader shared by every `*_sys` module in this crate.
//!
//! Each library module calls [`try_load`] with a list of candidate filenames
//! (Linux `.so`, Windows `.dll`, macOS `.dylib`) and stashes the resulting
//! `Library` in a `LazyLock<Result<_, LoadError>>`. The first caller pays
//! the `dlopen` cost; everyone else gets a `&'static Fns` back.
//!
//! Resolution order is:
//! 1. Platform-default OS search path (mirrors `LoadLibrary` / `dlopen`).
//! 2. Environment override — `IRON_CUDA_LIBDIR` prepended if set.

use libloading::Library;
use std::path::PathBuf;

pub type LoaderResult<T> = Result<T, LoadError>;

#[derive(Debug)]
pub enum LoadError {
    /// No candidate filename successfully loaded.
    LibraryNotFound { tried: Vec<String>, last: String },
    /// Library loaded but a required symbol is missing — likely driver is
    /// older than the CUDA version this crate binds (13.2).
    SymbolMissing { lib: &'static str, symbol: &'static str, err: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryNotFound { tried, last } => {
                write!(f, "could not load any of {tried:?}: last error = {last}")
            }
            Self::SymbolMissing { lib, symbol, err } => {
                write!(f, "{lib}: missing symbol `{symbol}`: {err}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Try each candidate filename in order. On Linux we also try the unversioned
/// name with `.1` / `.0` suffix if the bare name fails, mirroring how ld
/// resolves `SONAME` links.
pub fn try_load(candidates: &[&str]) -> LoaderResult<Library> {
    let extra_dir = std::env::var_os("IRON_CUDA_LIBDIR").map(PathBuf::from);
    let mut last_err = String::new();

    for name in candidates {
        if let Some(dir) = &extra_dir {
            let path = dir.join(name);
            unsafe {
                match Library::new(&path) {
                    Ok(lib) => return Ok(lib),
                    Err(e) => last_err = format!("{}: {e}", path.display()),
                }
            }
        }
        unsafe {
            match Library::new(*name) {
                Ok(lib) => return Ok(lib),
                Err(e) => last_err = format!("{name}: {e}"),
            }
        }
    }

    Err(LoadError::LibraryNotFound {
        tried: candidates.iter().map(|s| s.to_string()).collect(),
        last: last_err,
    })
}

/// Helper that resolves a symbol or returns a typed `SymbolMissing` error.
///
/// # Safety
/// Caller asserts the symbol has the declared signature.
pub unsafe fn sym<T: Copy>(
    lib: &Library, library_name: &'static str, symbol: &'static str,
) -> LoaderResult<T> {
    // libloading::Library::get requires a NUL-terminated byte slice on both
    // POSIX (CStr) and Windows (GetProcAddress). `str::as_bytes` omits the
    // terminator, so build one.
    let mut c = Vec::with_capacity(symbol.len() + 1);
    c.extend_from_slice(symbol.as_bytes());
    c.push(0);
    unsafe {
        let r: Result<libloading::Symbol<T>, _> = lib.get(&c[..]);
        match r {
            Ok(s) => Ok(*s),
            Err(e) => Err(LoadError::SymbolMissing {
                lib: library_name, symbol,
                err: format!("{e}"),
            }),
        }
    }
}

/// Same as `sym` but returns `None` instead of erroring — for optional
/// symbols added in newer CUDA versions.
///
/// # Safety
/// Same as [`sym`].
pub unsafe fn sym_opt<T: Copy>(lib: &Library, symbol: &'static str) -> Option<T> {
    let mut c = Vec::with_capacity(symbol.len() + 1);
    c.extend_from_slice(symbol.as_bytes());
    c.push(0);
    unsafe {
        let r: Result<libloading::Symbol<T>, _> = lib.get(&c[..]);
        r.ok().map(|s| *s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_library_reports_all_attempts() {
        let err = try_load(&["definitely-not-a-real-lib-ironcuda.so", "ironcuda.dll.unreal"])
            .unwrap_err();
        match err {
            LoadError::LibraryNotFound { tried, .. } => assert_eq!(tried.len(), 2),
            _ => panic!("wrong error variant"),
        }
    }
}
