//! QNN core FFI — provider interface discovery + the entry points
//! IronAccelerator actually drives.
//!
//! ## Provider pattern
//!
//! A QNN backend library (e.g. `libQnnHtp.so`) exports exactly one C symbol:
//! ```c
//! Qnn_ErrorHandle_t QnnInterface_getProviders(
//!     const QnnInterface_t*** providerList,
//!     uint32_t* numProviders);
//! ```
//! Each `QnnInterface_t` has a name, a 32-bit API version, and a large
//! **function-pointer table** (`QNN_INTERFACE_VER_TYPE`). We don't bind every
//! field — only the ones the safe wrapper calls. The rest of the struct is
//! stored opaquely as a byte blob whose size matches the published ABI.

use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_char, c_void};
use std::sync::{LazyLock, OnceLock};

// ── scalar typedefs ─────────────────────────────────────────────────────────

pub type Qnn_ErrorHandle_t = u64;
pub type Qnn_BackendHandle_t = *mut c_void;
pub type Qnn_ContextHandle_t = *mut c_void;
pub type Qnn_GraphHandle_t = *mut c_void;
pub type Qnn_DeviceHandle_t = *mut c_void;
pub type Qnn_ProfileHandle_t = *mut c_void;
pub type Qnn_LogHandle_t = *mut c_void;

pub const QNN_SUCCESS: Qnn_ErrorHandle_t = 0;

// ── versioned-interface header ──────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct QnnApiVersion {
    pub core_major: u32,
    pub core_minor: u32,
    pub core_patch: u32,
    pub backend_major: u32,
    pub backend_minor: u32,
    pub backend_patch: u32,
}

/// `QnnInterface_t`. Fields after the header are the function-pointer table,
/// which we access through typed accessors rather than transliterating the
/// whole (very large) union.
#[repr(C)]
pub struct QnnInterface_t {
    pub backend_id: u32,
    pub provider_name: *const c_char,
    pub api_version: QnnApiVersion,
    /// Opaque function table. Size depends on the backend's published ABI;
    /// we only read it through `QnnInterfaceV2::from_ptr`.
    pub core_api: *const QnnInterfaceV2,
}

/// Subset of `QNN_INTERFACE_VER_TYPE` (v2.x) that IronAccelerator drives.
/// Layout matches the first N function pointers of the SDK's published
/// struct; newer fields trail after and are ignored.
#[repr(C)]
pub struct QnnInterfaceV2 {
    pub property_hasCapability: unsafe extern "C" fn(u32) -> Qnn_ErrorHandle_t,

    // ── Backend ────────────────────────────────────────────────────────────
    pub backend_create: unsafe extern "C" fn(
        Qnn_LogHandle_t,
        *const *const QnnBackendConfig,
        *mut Qnn_BackendHandle_t,
    ) -> Qnn_ErrorHandle_t,
    pub backend_setConfig: unsafe extern "C" fn(
        Qnn_BackendHandle_t,
        *const *const QnnBackendConfig,
    ) -> Qnn_ErrorHandle_t,
    pub backend_getApiVersion: unsafe extern "C" fn(*mut QnnApiVersion) -> Qnn_ErrorHandle_t,
    pub backend_free: unsafe extern "C" fn(Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t,

    // ── Device ─────────────────────────────────────────────────────────────
    pub device_getInfrastructure: unsafe extern "C" fn(*mut *const c_void) -> Qnn_ErrorHandle_t,
    pub device_create: unsafe extern "C" fn(
        Qnn_LogHandle_t,
        *const *const c_void,
        *mut Qnn_DeviceHandle_t,
    ) -> Qnn_ErrorHandle_t,
    pub device_free: unsafe extern "C" fn(Qnn_DeviceHandle_t) -> Qnn_ErrorHandle_t,

    // ── Context ────────────────────────────────────────────────────────────
    pub context_create: unsafe extern "C" fn(
        Qnn_BackendHandle_t,
        Qnn_DeviceHandle_t,
        *const *const c_void,
        *mut Qnn_ContextHandle_t,
    ) -> Qnn_ErrorHandle_t,
    pub context_getBinarySize:
        unsafe extern "C" fn(Qnn_ContextHandle_t, *mut usize) -> Qnn_ErrorHandle_t,
    pub context_getBinary: unsafe extern "C" fn(
        Qnn_ContextHandle_t,
        *mut c_void,
        usize,
        *mut usize,
    ) -> Qnn_ErrorHandle_t,
    pub context_createFromBinary: unsafe extern "C" fn(
        Qnn_BackendHandle_t,
        Qnn_DeviceHandle_t,
        *const *const c_void,
        *const c_void,
        usize,
        *mut Qnn_ContextHandle_t,
        Qnn_ProfileHandle_t,
    ) -> Qnn_ErrorHandle_t,
    pub context_free:
        unsafe extern "C" fn(Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t,

    // ── Graph ──────────────────────────────────────────────────────────────
    pub graph_create: unsafe extern "C" fn(
        Qnn_ContextHandle_t,
        *const c_char,
        *const *const c_void,
        *mut Qnn_GraphHandle_t,
    ) -> Qnn_ErrorHandle_t,
    pub graph_addNode:
        unsafe extern "C" fn(Qnn_GraphHandle_t, *const QnnOpConfig) -> Qnn_ErrorHandle_t,
    pub graph_finalize: unsafe extern "C" fn(
        Qnn_GraphHandle_t,
        Qnn_ProfileHandle_t,
        *mut c_void,
    ) -> Qnn_ErrorHandle_t,
    pub graph_execute: unsafe extern "C" fn(
        Qnn_GraphHandle_t,
        *const QnnTensor,
        u32,
        *mut QnnTensor,
        u32,
        Qnn_ProfileHandle_t,
        *mut c_void,
    ) -> Qnn_ErrorHandle_t,
    pub graph_executeAsync: unsafe extern "C" fn(
        Qnn_GraphHandle_t,
        *const QnnTensor,
        u32,
        *mut QnnTensor,
        u32,
        Qnn_ProfileHandle_t,
        *mut c_void,
        *const c_void,
        unsafe extern "C" fn(*mut c_void, Qnn_ErrorHandle_t),
    ) -> Qnn_ErrorHandle_t,

    // ── Tensor ─────────────────────────────────────────────────────────────
    pub tensor_createContextTensor:
        unsafe extern "C" fn(Qnn_ContextHandle_t, *mut QnnTensor) -> Qnn_ErrorHandle_t,
    pub tensor_createGraphTensor:
        unsafe extern "C" fn(Qnn_GraphHandle_t, *mut QnnTensor) -> Qnn_ErrorHandle_t,
}

impl QnnInterfaceV2 {
    /// # Safety
    /// Caller asserts that `iface.core_api` points to a published `v2.x`
    /// interface struct whose first fields match this layout.
    #[inline]
    pub unsafe fn from_iface(iface: &QnnInterface_t) -> Option<&'static QnnInterfaceV2> {
        if iface.core_api.is_null() {
            None
        } else {
            unsafe { Some(&*iface.core_api) }
        }
    }
}

// ── opaque config / op structs ──────────────────────────────────────────────

/// Opaque pointer-to-config; the SDK exposes nested unions for each sub-area
/// (backend, context, graph). We pass through as `*const c_void` at the
/// safe-wrapper boundary and only bind the pieces IronAccelerator constructs.
#[repr(C)]
pub struct QnnBackendConfig {
    _private: [u8; 0],
}

/// `Qnn_OpConfig_t` header. Rust-side callers build the full struct as a
/// byte buffer and pass a pointer; we intentionally don't model the union.
#[repr(C)]
pub struct QnnOpConfig {
    _private: [u8; 0],
}

/// `Qnn_Tensor_t` — versioned tagged union. Same strategy: callers build
/// the buffer with the right layout for the SDK version on their machine.
#[repr(C)]
pub struct QnnTensor {
    _private: [u8; 0],
}

// ── entry point ─────────────────────────────────────────────────────────────

/// `QnnInterface_getProviders`. Sole exported symbol from every QNN backend
/// library; everything else comes from the returned function table.
pub type QnnInterface_getProviders_t =
    unsafe extern "C" fn(*mut *const *const QnnInterface_t, *mut u32) -> Qnn_ErrorHandle_t;

// ── per-target lazy loaders ─────────────────────────────────────────────────

/// Which QNN backend to load. Caller picks; IronAccelerator's safe wrapper
/// defaults to `Htp` with fallback to `Cpu` when no NPU is present.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Htp,
    Gpu,
    Cpu,
    Dsp,
    Saver,
}

impl Target {
    pub fn libs(self) -> &'static [&'static str] {
        match self {
            Target::Htp => &["libQnnHtp.so", "QnnHtp.dll"],
            Target::Gpu => &["libQnnGpu.so", "QnnGpu.dll"],
            Target::Cpu => &["libQnnCpu.so", "QnnCpu.dll"],
            Target::Dsp => &["libQnnDsp.so", "QnnDsp.dll"],
            Target::Saver => &["libQnnSaver.so", "QnnSaver.dll"],
        }
    }
}

pub struct QnnFns {
    pub get_providers: QnnInterface_getProviders_t,
    /// Cached first-provider interface; safe wrapper uses these pointers.
    pub iface: &'static QnnInterfaceV2,
    pub provider_name: String,
    pub api_version: QnnApiVersion,
}

unsafe impl Send for QnnFns {}
unsafe impl Sync for QnnFns {}

fn load_for(target: Target) -> LoaderResult<(Library, QnnFns)> {
    let lib = try_load(target.libs())?;
    let get_providers: QnnInterface_getProviders_t =
        unsafe { sym(&lib, "qnn", "QnnInterface_getProviders")? };

    let mut providers: *const *const QnnInterface_t = std::ptr::null();
    let mut n: u32 = 0;
    let err = unsafe { get_providers(&mut providers, &mut n) };
    if err != QNN_SUCCESS || providers.is_null() || n == 0 {
        return Err(LoadError::SymbolMissing {
            lib: "qnn",
            symbol: "QnnInterface_getProviders",
            err: format!("returned error {err} or empty provider list"),
        });
    }
    let iface_ptr = unsafe { *providers };
    if iface_ptr.is_null() {
        return Err(LoadError::SymbolMissing {
            lib: "qnn",
            symbol: "providers[0]",
            err: "null first provider".into(),
        });
    }
    let iface: &'static QnnInterface_t = unsafe { &*iface_ptr };
    let v2 = unsafe { QnnInterfaceV2::from_iface(iface) }.ok_or(LoadError::SymbolMissing {
        lib: "qnn",
        symbol: "core_api",
        err: "provider returned null function table".into(),
    })?;

    let name = if iface.provider_name.is_null() {
        String::from("unknown")
    } else {
        unsafe { std::ffi::CStr::from_ptr(iface.provider_name) }
            .to_string_lossy()
            .into_owned()
    };

    Ok((
        lib,
        QnnFns {
            get_providers,
            iface: v2,
            provider_name: name,
            api_version: iface.api_version,
        },
    ))
}

// Keep each target's Library alive for the process lifetime.
static HTP: LazyLock<LoaderResult<(Library, QnnFns)>> = LazyLock::new(|| load_for(Target::Htp));
static GPU: LazyLock<LoaderResult<(Library, QnnFns)>> = LazyLock::new(|| load_for(Target::Gpu));
static CPU: LazyLock<LoaderResult<(Library, QnnFns)>> = LazyLock::new(|| load_for(Target::Cpu));
static DSP: LazyLock<LoaderResult<(Library, QnnFns)>> = LazyLock::new(|| load_for(Target::Dsp));

static HTP_FNS: OnceLock<Result<&'static QnnFns, LoadError>> = OnceLock::new();
static GPU_FNS: OnceLock<Result<&'static QnnFns, LoadError>> = OnceLock::new();
static CPU_FNS: OnceLock<Result<&'static QnnFns, LoadError>> = OnceLock::new();
static DSP_FNS: OnceLock<Result<&'static QnnFns, LoadError>> = OnceLock::new();

fn pick(
    slot: &'static OnceLock<Result<&'static QnnFns, LoadError>>,
    lib: &'static LazyLock<LoaderResult<(Library, QnnFns)>>,
) -> Result<&'static QnnFns, &'static LoadError> {
    match slot.get_or_init(|| match lib.as_ref() {
        Ok((_, fns)) => Ok(fns),
        Err(e) => Err(e.clone()),
    }) {
        Ok(fns) => Ok(fns),
        Err(e) => Err(e),
    }
}

pub fn fns(target: Target) -> Result<&'static QnnFns, &'static LoadError> {
    match target {
        Target::Htp => pick(&HTP_FNS, &HTP),
        Target::Gpu => pick(&GPU_FNS, &GPU),
        Target::Cpu => pick(&CPU_FNS, &CPU),
        Target::Dsp => pick(&DSP_FNS, &DSP),
        Target::Saver => Err(saver_err()),
    }
}

fn saver_err() -> &'static LoadError {
    static E: OnceLock<LoadError> = OnceLock::new();
    E.get_or_init(|| LoadError::LibraryNotFound {
        tried: vec!["saver".into()],
        last: "Saver target is offline-only".into(),
    })
}

pub fn is_available(target: Target) -> bool {
    fns(target).is_ok()
}
