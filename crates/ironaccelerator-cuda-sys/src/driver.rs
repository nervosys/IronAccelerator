//! CUDA Driver API (`libcuda.so` / `nvcuda.dll`). Targets 13.2.
//!
//! Only the functions IronAccelerator's safe layer actually uses are bound.
//! If you need a new one, add it to [`DriverFns`] and `DriverFns::load`.

use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{LazyLock, OnceLock};

// ── opaque handles ──────────────────────────────────────────────────────────

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUcontext(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUstream(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUevent(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUmodule(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUfunction(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUgraph(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUgraphExec(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CUmemPool(pub *mut c_void);

/// Device ordinal (i32). Not a pointer — the driver actually passes this by value.
pub type CUdevice = c_int;

/// Device-side pointer. 64-bit on every supported platform.
pub type CUdeviceptr = u64;

unsafe impl Send for CUcontext {} unsafe impl Sync for CUcontext {}
unsafe impl Send for CUstream {}  unsafe impl Sync for CUstream {}
unsafe impl Send for CUevent {}   unsafe impl Sync for CUevent {}
unsafe impl Send for CUmodule {}  unsafe impl Sync for CUmodule {}
unsafe impl Send for CUfunction {} unsafe impl Sync for CUfunction {}
unsafe impl Send for CUgraph {}     unsafe impl Sync for CUgraph {}
unsafe impl Send for CUgraphExec {} unsafe impl Sync for CUgraphExec {}
unsafe impl Send for CUmemPool {}   unsafe impl Sync for CUmemPool {}

// ── result codes (subset, with `Ok` = 0) ───────────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUresult {
    Success                     = 0,
    InvalidValue                = 1,
    OutOfMemory                 = 2,
    NotInitialized              = 3,
    Deinitialized               = 4,
    NoDevice                    = 100,
    InvalidDevice               = 101,
    InvalidContext              = 201,
    MapFailed                   = 205,
    UnmapFailed                 = 206,
    ArrayIsMapped               = 207,
    AlreadyMapped               = 208,
    NoBinaryForGpu              = 209,
    AlreadyAcquired             = 210,
    NotMapped                   = 211,
    InvalidSource               = 300,
    FileNotFound                = 301,
    SharedObjectSymbolNotFound  = 302,
    SharedObjectInitFailed      = 303,
    OperatingSystem             = 304,
    InvalidHandle               = 400,
    NotFound                    = 500,
    NotReady                    = 600,
    LaunchFailed                = 700,
    LaunchOutOfResources        = 701,
    LaunchTimeout               = 702,
    PeerAccessAlreadyEnabled    = 704,
    ContextIsDestroyed          = 709,
    StreamCaptureUnsupported    = 900,
    Unknown                     = 999,
    /// Any code we don't model — hold on to the numeric value.
    Other                       = 0xFFFF_FFFF,
}

impl CUresult {
    #[inline]
    pub fn from_raw(r: u32) -> Self {
        // Map the ones we know; everything else → Other.
        match r {
            0 => Self::Success, 1 => Self::InvalidValue, 2 => Self::OutOfMemory,
            3 => Self::NotInitialized, 4 => Self::Deinitialized,
            100 => Self::NoDevice, 101 => Self::InvalidDevice,
            201 => Self::InvalidContext,
            205 => Self::MapFailed, 206 => Self::UnmapFailed,
            207 => Self::ArrayIsMapped, 208 => Self::AlreadyMapped,
            209 => Self::NoBinaryForGpu, 210 => Self::AlreadyAcquired, 211 => Self::NotMapped,
            300 => Self::InvalidSource, 301 => Self::FileNotFound,
            302 => Self::SharedObjectSymbolNotFound, 303 => Self::SharedObjectInitFailed,
            304 => Self::OperatingSystem,
            400 => Self::InvalidHandle,
            500 => Self::NotFound,
            600 => Self::NotReady,
            700 => Self::LaunchFailed, 701 => Self::LaunchOutOfResources,
            702 => Self::LaunchTimeout,
            704 => Self::PeerAccessAlreadyEnabled,
            709 => Self::ContextIsDestroyed,
            900 => Self::StreamCaptureUnsupported,
            999 => Self::Unknown,
            _ => Self::Other,
        }
    }

    #[inline] pub fn ok(self) -> Result<(), Self> { if self == Self::Success { Ok(()) } else { Err(self) } }
    #[inline] pub fn is_ok(self) -> bool { self == Self::Success }
}

// ── attribute / flag enums we need ─────────────────────────────────────────

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CUdevice_attribute {
    MaxThreadsPerBlock              = 1,
    MaxSharedMemoryPerBlock         = 8,
    TotalConstantMemory             = 9,
    WarpSize                        = 10,
    MaxRegistersPerBlock            = 12,
    ClockRate                       = 13,
    MultiprocessorCount             = 16,
    IntegrationType                 = 18,
    ComputeCapabilityMajor          = 75,
    ComputeCapabilityMinor          = 76,
    PciBusId                        = 33,
    PciDeviceId                     = 34,
    PciDomainId                     = 50,
    MemoryClockRate                 = 36,
    GlobalMemoryBusWidth            = 37,
    L2CacheSize                     = 38,
    MaxThreadsPerMultiProcessor     = 39,
    AsyncEngineCount                = 40,
    UnifiedAddressing               = 41,
    StreamPrioritiesSupported       = 78,
    CooperativeLaunch               = 95,
    ConcurrentManagedAccess         = 89,
    ComputePreemptionSupported      = 90,
    ComputeMode                     = 20,
    ManagedMemory                   = 83,
    MultiGpuBoard                   = 84,
    MemoryPoolsSupported            = 115,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUevent_flags {
    Default             = 0x0,
    BlockingSync        = 0x1,
    DisableTiming       = 0x2,
    Interprocess        = 0x4,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUstreamCaptureMode {
    Global      = 0,
    ThreadLocal = 1,
    Relaxed     = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUhostAllocFlags {
    Default         = 0x0,
    Portable        = 0x1,
    Mapped          = 0x2,
    WriteCombined   = 0x4,
}

pub const CU_STREAM_DEFAULT: u32 = 0x0;
pub const CU_STREAM_NON_BLOCKING: u32 = 0x1;

// ── function pointer table ─────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
pub struct DriverFns {
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUresult,

    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult,
    pub cuDeviceTotalMem_v2: unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult,
    pub cuDeviceCanAccessPeer: unsafe extern "C" fn(*mut c_int, CUdevice, CUdevice) -> CUresult,

    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(CUdevice) -> CUresult,

    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxPushCurrent_v2: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxPopCurrent_v2: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuCtxEnablePeerAccess: unsafe extern "C" fn(CUcontext, c_uint) -> CUresult,

    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamCreateWithPriority: unsafe extern "C" fn(*mut CUstream, c_uint, c_int) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    pub cuStreamGetPriorityRange: unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult,

    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,

    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemAllocAsync: unsafe extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemFreeAsync: unsafe extern "C" fn(CUdeviceptr, CUstream) -> CUresult,
    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    pub cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoDAsync_v2: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,

    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,

    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint, c_uint, c_uint,        // grid
        c_uint, c_uint, c_uint,        // block
        c_uint,                        // shared mem bytes
        CUstream,
        *mut *mut c_void,              // kernel params
        *mut *mut c_void,              // extra
    ) -> CUresult,

    pub cuStreamBeginCapture_v2: unsafe extern "C" fn(CUstream, CUstreamCaptureMode) -> CUresult,
    pub cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    pub cuGraphInstantiateWithFlags: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
}

fn load_library() -> LoaderResult<Library> {
    try_load(&[
        "libcuda.so.1",
        "libcuda.so",
        "nvcuda.dll",
        "/usr/lib/wsl/lib/libcuda.so.1",
    ])
}

fn load_fns(lib: &Library) -> LoaderResult<DriverFns> {
    macro_rules! g { ($sym:ident) => { sym(lib, "libcuda", stringify!($sym))? } }
    // Some driver versions (notably the Windows nvcuda.dll) export the
    // context-scoped alias `cuCtxGetStreamPriorityRange` instead of, or in
    // addition to, `cuStreamGetPriorityRange`. Signatures are identical. Fall
    // back to the alias so we work on both.
    let cu_stream_get_priority_range: unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult =
        match unsafe { crate::loader::sym_opt(lib, "cuStreamGetPriorityRange") } {
            Some(f) => f,
            None => unsafe { sym(lib, "libcuda", "cuCtxGetStreamPriorityRange")? },
        };
    unsafe {
        Ok(DriverFns {
            cuInit: g!(cuInit),
            cuDriverGetVersion: g!(cuDriverGetVersion),
            cuDeviceGetCount: g!(cuDeviceGetCount),
            cuDeviceGet: g!(cuDeviceGet),
            cuDeviceGetName: g!(cuDeviceGetName),
            cuDeviceGetAttribute: g!(cuDeviceGetAttribute),
            cuDeviceTotalMem_v2: g!(cuDeviceTotalMem_v2),
            cuDeviceCanAccessPeer: g!(cuDeviceCanAccessPeer),
            cuDevicePrimaryCtxRetain: g!(cuDevicePrimaryCtxRetain),
            cuDevicePrimaryCtxRelease_v2: g!(cuDevicePrimaryCtxRelease_v2),
            cuCtxSetCurrent: g!(cuCtxSetCurrent),
            cuCtxGetCurrent: g!(cuCtxGetCurrent),
            cuCtxPushCurrent_v2: g!(cuCtxPushCurrent_v2),
            cuCtxPopCurrent_v2: g!(cuCtxPopCurrent_v2),
            cuCtxSynchronize: g!(cuCtxSynchronize),
            cuCtxEnablePeerAccess: g!(cuCtxEnablePeerAccess),
            cuStreamCreate: g!(cuStreamCreate),
            cuStreamCreateWithPriority: g!(cuStreamCreateWithPriority),
            cuStreamDestroy_v2: g!(cuStreamDestroy_v2),
            cuStreamSynchronize: g!(cuStreamSynchronize),
            cuStreamWaitEvent: g!(cuStreamWaitEvent),
            cuStreamGetPriorityRange: cu_stream_get_priority_range,
            cuEventCreate: g!(cuEventCreate),
            cuEventDestroy_v2: g!(cuEventDestroy_v2),
            cuEventRecord: g!(cuEventRecord),
            cuEventSynchronize: g!(cuEventSynchronize),
            cuEventElapsedTime: g!(cuEventElapsedTime),
            cuMemAlloc_v2: g!(cuMemAlloc_v2),
            cuMemAllocAsync: g!(cuMemAllocAsync),
            cuMemFree_v2: g!(cuMemFree_v2),
            cuMemFreeAsync: g!(cuMemFreeAsync),
            cuMemsetD8Async: g!(cuMemsetD8Async),
            cuMemcpyHtoDAsync_v2: g!(cuMemcpyHtoDAsync_v2),
            cuMemcpyDtoHAsync_v2: g!(cuMemcpyDtoHAsync_v2),
            cuMemcpyDtoDAsync_v2: g!(cuMemcpyDtoDAsync_v2),
            cuMemHostAlloc: g!(cuMemHostAlloc),
            cuMemFreeHost: g!(cuMemFreeHost),
            cuModuleLoadData: g!(cuModuleLoadData),
            cuModuleUnload: g!(cuModuleUnload),
            cuModuleGetFunction: g!(cuModuleGetFunction),
            cuLaunchKernel: g!(cuLaunchKernel),
            cuStreamBeginCapture_v2: g!(cuStreamBeginCapture_v2),
            cuStreamEndCapture: g!(cuStreamEndCapture),
            cuGraphDestroy: g!(cuGraphDestroy),
            cuGraphInstantiateWithFlags: g!(cuGraphInstantiateWithFlags),
            cuGraphExecDestroy: g!(cuGraphExecDestroy),
            cuGraphLaunch: g!(cuGraphLaunch),
        })
    }
}

static LIB: OnceLock<Library> = OnceLock::new();

static FNS: LazyLock<Result<DriverFns, LoadError>> = LazyLock::new(|| {
    let lib = load_library()?;
    let stored = LIB.get_or_init(|| lib);
    load_fns(stored)
});

static INIT_DONE: OnceLock<CUresult> = OnceLock::new();

/// Resolve the driver function table. First call performs a `cuInit(0)`.
pub fn fns() -> Result<&'static DriverFns, &'static LoadError> {
    let f = FNS.as_ref()?;
    INIT_DONE.get_or_init(|| unsafe { (f.cuInit)(0) });
    Ok(f)
}

/// Library is present and `cuInit(0)` returned `Success`. Useful as a
/// `Backend::is_available` probe.
pub fn is_available() -> bool {
    match fns() {
        Ok(_) => INIT_DONE.get().is_some_and(|r| r.is_ok()),
        Err(_) => false,
    }
}

// ── convenience wrappers so call sites read cleanly ─────────────────────────

/// Assert `CUresult::Success`, returning the `CUresult` on failure.
#[inline]
pub fn check(r: CUresult) -> Result<(), CUresult> { r.ok() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_enum_roundtrips_known_codes() {
        assert_eq!(CUresult::from_raw(0),   CUresult::Success);
        assert_eq!(CUresult::from_raw(700), CUresult::LaunchFailed);
        assert_eq!(CUresult::from_raw(12345), CUresult::Other);
    }

    #[test]
    fn handles_are_zero_sized_newtypes() {
        assert_eq!(std::mem::size_of::<CUcontext>(), std::mem::size_of::<*mut ()>());
        assert_eq!(std::mem::size_of::<CUstream>(),  std::mem::size_of::<*mut ()>());
    }

    #[test]
    fn result_ok_is_false_for_errors() {
        assert!(CUresult::Success.is_ok());
        assert!(!CUresult::OutOfMemory.is_ok());
    }
}
