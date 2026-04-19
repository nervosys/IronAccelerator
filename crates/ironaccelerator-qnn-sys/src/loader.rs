//! Library-loading helpers — parallel to `iron_cuda_sys::loader`.

use libloading::Library;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum LoadError {
    LibraryNotFound { tried: Vec<String>, last: String },
    SymbolMissing { lib: &'static str, symbol: &'static str, err: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::LibraryNotFound { tried, last } =>
                write!(f, "none of {tried:?} could be loaded (last: {last})"),
            LoadError::SymbolMissing { lib, symbol, err } =>
                write!(f, "{lib}: symbol `{symbol}` missing: {err}"),
        }
    }
}
impl std::error::Error for LoadError {}

pub type LoaderResult<T> = Result<T, LoadError>;

#[allow(unused_assignments)]
pub fn try_load(candidates: &[&str]) -> LoaderResult<Library> {
    let extra_dir = std::env::var_os("IRON_QNN_LIBDIR").map(PathBuf::from);
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

pub unsafe fn sym<T: Copy>(
    lib: &Library, lib_name: &'static str, symbol: &'static str,
) -> LoaderResult<T> {
    let mut c = Vec::with_capacity(symbol.len() + 1);
    c.extend_from_slice(symbol.as_bytes());
    c.push(0);
    unsafe {
        let r: Result<libloading::Symbol<T>, _> = lib.get(&c[..]);
        match r {
            Ok(s) => Ok(*s),
            Err(e) => Err(LoadError::SymbolMissing { lib: lib_name, symbol, err: format!("{e}") }),
        }
    }
}
