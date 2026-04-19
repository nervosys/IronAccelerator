//! QNN SDK FFI. Dynamic-loading mirror of the CUDA/ROCm sys crates.
//!
//! QNN ships one `.so` / `.dll` per backend (HTP, GPU, CPU, DSP). Each
//! exports a single entry point — `QnnInterface_getProviders` — that returns
//! a list of versioned function tables. We load the target library, call
//! that entry, pick the first v2.x provider, and expose its function table
//! through safe wrappers.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(clippy::missing_safety_doc, clippy::type_complexity)]

pub mod loader;
pub mod qnn;

pub use loader::{LoadError, LoaderResult};
