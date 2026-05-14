//! `libnrt` loader. We need three symbols — `nrt_init`,
//! `nrt_get_total_nc_count`, and `nrt_get_version` — and a best-effort
//! read of the Neuron-specific env vars that AWS containers set.
//!
//! `drop(sym)` on a `libloading::Symbol` ends the borrow on its Library;
//! Symbol doesn't impl Drop but the lifetime parameter does the work.

#![allow(clippy::drop_non_drop)]

use core::ffi::c_void;

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

pub type NrtStatus = u32;
pub const NRT_SUCCESS: NrtStatus = 0;

pub type NrtModelHandle = *mut c_void;
pub type NrtTensorSetHandle = *mut c_void;
pub type NrtTensorHandle = *mut c_void;

pub type NrtLoadFn = unsafe extern "C" fn(
    neff: *const u8,
    neff_size: usize,
    start_nc: i32,
    nc_count: i32,
    model: *mut NrtModelHandle,
) -> NrtStatus;
pub type NrtUnloadFn = unsafe extern "C" fn(model: NrtModelHandle) -> NrtStatus;
pub type NrtExecuteFn = unsafe extern "C" fn(
    model: NrtModelHandle,
    input_set: NrtTensorSetHandle,
    output_set: NrtTensorSetHandle,
) -> NrtStatus;
pub type NrtAllocTensorSetFn = unsafe extern "C" fn(out: *mut NrtTensorSetHandle) -> NrtStatus;
pub type NrtFreeTensorSetFn = unsafe extern "C" fn(set: NrtTensorSetHandle) -> NrtStatus;
pub type NrtAddTensorToSetFn = unsafe extern "C" fn(
    set: NrtTensorSetHandle,
    name: *const core::ffi::c_char,
    tensor: NrtTensorHandle,
) -> NrtStatus;

type NrtInitFn =
    unsafe extern "C" fn(framework: u32, framework_version: *const core::ffi::c_char) -> NrtStatus;
type NrtGetTotalNcCountFn = unsafe extern "C" fn(count: *mut u32) -> NrtStatus;
type NrtGetVersionFn = unsafe extern "C" fn(out: *mut core::ffi::c_char, out_len: u32) -> NrtStatus;

static LIB: OnceCell<Option<Loaded>> = OnceCell::new();

pub struct Loaded {
    _lib: Library,
    total_cores: u32,
    version: String,
    pub nrt_load: Option<NrtLoadFn>,
    pub nrt_unload: Option<NrtUnloadFn>,
    pub nrt_execute: Option<NrtExecuteFn>,
    pub nrt_alloc_tensor_set: Option<NrtAllocTensorSetFn>,
    pub nrt_free_tensor_set: Option<NrtFreeTensorSetFn>,
    pub nrt_add_tensor_to_set: Option<NrtAddTensorToSetFn>,
}

unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

#[cfg(any(target_os = "linux", target_os = "android"))]
const LIB_CANDIDATES: &[&str] = &[
    "libnrt.so.1",
    "libnrt.so",
    "/opt/aws/neuron/lib/libnrt.so.1",
];
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const LIB_CANDIDATES: &[&str] = &[];

pub(crate) fn loaded() -> Option<&'static Loaded> {
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<Loaded> {
    for name in LIB_CANDIDATES {
        let lib = match unsafe { Library::new(*name) } {
            Ok(l) => l,
            Err(_) => continue,
        };
        unsafe {
            let init: Symbol<NrtInitFn> = match lib.get(b"nrt_init\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            // framework=0 (NRT_FRAMEWORK_TYPE_NONE) is accepted for
            // non-framework embedders like us.
            let tag = b"ironaccelerator\0" as *const _ as *const core::ffi::c_char;
            if init(0, tag) != NRT_SUCCESS {
                continue;
            }
            let count_fn: Symbol<NrtGetTotalNcCountFn> = match lib.get(b"nrt_get_total_nc_count\0")
            {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut count: u32 = 0;
            if count_fn(&mut count) != NRT_SUCCESS {
                continue;
            }
            let version = lib
                .get::<NrtGetVersionFn>(b"nrt_get_version\0")
                .ok()
                .and_then(|f| read_version(*f))
                .unwrap_or_default();
            drop(init);
            drop(count_fn);

            // Optional symbols — present on Neuron SDK 2.x but we don't
            // want to fail the whole backend if one is missing on an
            // older runtime.
            let nrt_load = lib.get::<NrtLoadFn>(b"nrt_load\0").ok().map(|s| *s);
            let nrt_unload = lib.get::<NrtUnloadFn>(b"nrt_unload\0").ok().map(|s| *s);
            let nrt_execute = lib.get::<NrtExecuteFn>(b"nrt_execute\0").ok().map(|s| *s);
            let nrt_alloc_tensor_set = lib
                .get::<NrtAllocTensorSetFn>(b"nrt_allocate_tensor_set\0")
                .ok()
                .map(|s| *s);
            let nrt_free_tensor_set = lib
                .get::<NrtFreeTensorSetFn>(b"nrt_free_tensor_set\0")
                .ok()
                .map(|s| *s);
            let nrt_add_tensor_to_set = lib
                .get::<NrtAddTensorToSetFn>(b"nrt_add_tensor_to_tensor_set\0")
                .ok()
                .map(|s| *s);

            return Some(Loaded {
                _lib: lib,
                total_cores: count,
                version,
                nrt_load,
                nrt_unload,
                nrt_execute,
                nrt_alloc_tensor_set,
                nrt_free_tensor_set,
                nrt_add_tensor_to_set,
            });
        }
    }
    None
}

unsafe fn read_version(f: NrtGetVersionFn) -> Option<String> {
    let mut buf = vec![0i8; 128];
    let status = f(buf.as_mut_ptr() as *mut _, buf.len() as u32);
    if status != NRT_SUCCESS {
        return None;
    }
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8(bytes).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronGen {
    /// inf1 — first-gen Inferentia.
    Inf1,
    /// trn1 / inf2 — second-gen Trainium / Inferentia.
    Trn1,
    /// trn2 — third-gen Trainium.
    Trn2,
    Unknown,
}

pub fn is_available() -> bool {
    loaded().is_some()
}

pub fn total_cores() -> u32 {
    loaded().map(|l| l.total_cores).unwrap_or(0)
}

pub fn version() -> Option<&'static str> {
    loaded().map(|l| l.version.as_str())
}

/// Best-effort generation detection. Prefers `NEURON_INSTANCE_TYPE` env
/// hint (commonly set by AWS Deep Learning AMIs + containers), falls
/// back to the Neuron runtime major version.
pub fn detect_generation() -> NeuronGen {
    if let Ok(t) = std::env::var("NEURON_INSTANCE_TYPE") {
        let t = t.to_ascii_lowercase();
        if t.starts_with("trn2") {
            return NeuronGen::Trn2;
        }
        if t.starts_with("trn1") || t.starts_with("inf2") {
            return NeuronGen::Trn1;
        }
        if t.starts_with("inf1") {
            return NeuronGen::Inf1;
        }
    }
    match version()
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse::<u32>().ok())
    {
        Some(n) if n >= 3 => NeuronGen::Trn2,
        Some(2) => NeuronGen::Trn1,
        Some(1) => NeuronGen::Inf1,
        _ => NeuronGen::Unknown,
    }
}
