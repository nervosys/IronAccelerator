//! HIP runtime FFI. ROCm ships HIP as a single library (`libamdhip64`) that
//! unifies driver- and runtime-scoped entry points, unlike CUDA's split.
//!
//! We bind the minimum we need for IronAccelerator's safe layer: device
//! enumeration, stream-ordered memory, events, module load, kernel launch.

use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{LazyLock, OnceLock};

// ── opaque handles ──────────────────────────────────────────────────────────

pub type HipDevice = c_int;
pub type HipDeviceptr = u64;
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipStream(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipEvent(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipModule(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipFunction(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipCtx(pub *mut c_void);

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}
unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}
unsafe impl Send for HipCtx {}
unsafe impl Sync for HipCtx {}

// ── result codes ────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipResult {
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorDeinitialized = 4,
    ErrorNoDevice = 100,
    ErrorInvalidDevice = 101,
    ErrorInvalidContext = 201,
    ErrorInvalidHandle = 400,
    ErrorNotFound = 500,
    ErrorNotReady = 600,
    ErrorLaunchFailure = 719,
    ErrorUnknown = 999,
    Other = 0xFFFF_FFFF,
}

impl HipResult {
    #[inline]
    pub fn from_raw(r: u32) -> Self {
        match r {
            0 => Self::Success,
            1 => Self::ErrorInvalidValue,
            2 => Self::ErrorOutOfMemory,
            3 => Self::ErrorNotInitialized,
            4 => Self::ErrorDeinitialized,
            100 => Self::ErrorNoDevice,
            101 => Self::ErrorInvalidDevice,
            201 => Self::ErrorInvalidContext,
            400 => Self::ErrorInvalidHandle,
            500 => Self::ErrorNotFound,
            600 => Self::ErrorNotReady,
            719 => Self::ErrorLaunchFailure,
            999 => Self::ErrorUnknown,
            _ => Self::Other,
        }
    }
    #[inline]
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success {
            Ok(())
        } else {
            Err(self)
        }
    }
    #[inline]
    pub fn is_ok(self) -> bool {
        self == Self::Success
    }
}

// ── attribute / flag enums ──────────────────────────────────────────────────

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum HipDeviceAttribute {
    MaxThreadsPerBlock = 0,
    MaxBlockDimX = 1,
    MaxBlockDimY = 2,
    MaxBlockDimZ = 3,
    MaxGridDimX = 4,
    MaxGridDimY = 5,
    MaxGridDimZ = 6,
    MaxSharedMemoryPerBlock = 8,
    WarpSize = 10,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    MultiprocessorCount = 16,
    ComputeCapabilityMajor = 23,
    ComputeCapabilityMinor = 24,
    ConcurrentKernels = 31,
    PciBusId = 33,
    PciDeviceId = 34,
    MaxThreadsPerMultiProcessor = 39,
    MemoryClockRate = 36,
    MemoryBusWidth = 37,
    L2CacheSize = 38,
    ManagedMemory = 83,
    IntegratedDevice = 18,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

pub const HIP_STREAM_DEFAULT: u32 = 0;
pub const HIP_STREAM_NON_BLOCKING: u32 = 1;
pub const HIP_EVENT_DISABLE_TIMING: u32 = 2;

// ── function pointer table ──────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
pub struct HipFns {
    pub hipInit: unsafe extern "C" fn(c_uint) -> HipResult,
    pub hipDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> HipResult,
    pub hipRuntimeGetVersion: unsafe extern "C" fn(*mut c_int) -> HipResult,
    pub hipGetDeviceCount: unsafe extern "C" fn(*mut c_int) -> HipResult,
    pub hipDeviceGet: unsafe extern "C" fn(*mut HipDevice, c_int) -> HipResult,
    pub hipDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, HipDevice) -> HipResult,
    pub hipDeviceGetAttribute:
        unsafe extern "C" fn(*mut c_int, HipDeviceAttribute, HipDevice) -> HipResult,
    pub hipDeviceTotalMem: unsafe extern "C" fn(*mut usize, HipDevice) -> HipResult,
    pub hipDeviceCanAccessPeer: unsafe extern "C" fn(*mut c_int, HipDevice, HipDevice) -> HipResult,

    pub hipSetDevice: unsafe extern "C" fn(c_int) -> HipResult,
    pub hipGetDevice: unsafe extern "C" fn(*mut c_int) -> HipResult,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> HipResult,
    pub hipDeviceReset: unsafe extern "C" fn() -> HipResult,
    pub hipCtxEnablePeerAccess: unsafe extern "C" fn(HipCtx, c_uint) -> HipResult,

    pub hipStreamCreate: unsafe extern "C" fn(*mut HipStream) -> HipResult,
    pub hipStreamCreateWithPriority:
        unsafe extern "C" fn(*mut HipStream, c_uint, c_int) -> HipResult,
    pub hipStreamDestroy: unsafe extern "C" fn(HipStream) -> HipResult,
    pub hipStreamSynchronize: unsafe extern "C" fn(HipStream) -> HipResult,
    pub hipStreamWaitEvent: unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> HipResult,
    pub hipDeviceGetStreamPriorityRange: unsafe extern "C" fn(*mut c_int, *mut c_int) -> HipResult,

    pub hipEventCreate: unsafe extern "C" fn(*mut HipEvent) -> HipResult,
    pub hipEventCreateWithFlags: unsafe extern "C" fn(*mut HipEvent, c_uint) -> HipResult,
    pub hipEventDestroy: unsafe extern "C" fn(HipEvent) -> HipResult,
    pub hipEventRecord: unsafe extern "C" fn(HipEvent, HipStream) -> HipResult,
    pub hipEventSynchronize: unsafe extern "C" fn(HipEvent) -> HipResult,
    pub hipEventElapsedTime: unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipResult,

    pub hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult,
    pub hipMallocAsync: unsafe extern "C" fn(*mut *mut c_void, usize, HipStream) -> HipResult,
    pub hipFree: unsafe extern "C" fn(*mut c_void) -> HipResult,
    pub hipFreeAsync: unsafe extern "C" fn(*mut c_void, HipStream) -> HipResult,
    pub hipMemsetAsync: unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> HipResult,
    pub hipMemcpyAsync: unsafe extern "C" fn(
        *mut c_void,
        *const c_void,
        usize,
        HipMemcpyKind,
        HipStream,
    ) -> HipResult,
    pub hipMemcpyHtoDAsync:
        unsafe extern "C" fn(HipDeviceptr, *const c_void, usize, HipStream) -> HipResult,
    pub hipMemcpyDtoHAsync:
        unsafe extern "C" fn(*mut c_void, HipDeviceptr, usize, HipStream) -> HipResult,
    pub hipMemcpyDtoDAsync:
        unsafe extern "C" fn(HipDeviceptr, HipDeviceptr, usize, HipStream) -> HipResult,
    pub hipHostMalloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> HipResult,
    pub hipHostFree: unsafe extern "C" fn(*mut c_void) -> HipResult,

    pub hipModuleLoadData: unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipResult,
    pub hipModuleUnload: unsafe extern "C" fn(HipModule) -> HipResult,
    pub hipModuleGetFunction:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipResult,
    pub hipModuleLaunchKernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint, // grid
        c_uint,
        c_uint,
        c_uint, // block
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipResult,
}

// ── library handle + lazy init ──────────────────────────────────────────────

static LIB: LazyLock<LoaderResult<Library>> = LazyLock::new(load_lib);
static FNS: OnceLock<LoaderResult<HipFns>> = OnceLock::new();

fn load_lib() -> LoaderResult<Library> {
    try_load(&[
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        "amdhip64.dll",
        "amdhip64_6.dll",
    ])
}

fn load_fns(lib: &Library) -> LoaderResult<HipFns> {
    macro_rules! g {
        ($sym:ident) => {
            sym(lib, "libamdhip64", stringify!($sym))?
        };
    }
    unsafe {
        Ok(HipFns {
            hipInit: g!(hipInit),
            hipDriverGetVersion: g!(hipDriverGetVersion),
            hipRuntimeGetVersion: g!(hipRuntimeGetVersion),
            hipGetDeviceCount: g!(hipGetDeviceCount),
            hipDeviceGet: g!(hipDeviceGet),
            hipDeviceGetName: g!(hipDeviceGetName),
            hipDeviceGetAttribute: g!(hipDeviceGetAttribute),
            hipDeviceTotalMem: g!(hipDeviceTotalMem),
            hipDeviceCanAccessPeer: g!(hipDeviceCanAccessPeer),
            hipSetDevice: g!(hipSetDevice),
            hipGetDevice: g!(hipGetDevice),
            hipDeviceSynchronize: g!(hipDeviceSynchronize),
            hipDeviceReset: g!(hipDeviceReset),
            hipCtxEnablePeerAccess: g!(hipCtxEnablePeerAccess),
            hipStreamCreate: g!(hipStreamCreate),
            hipStreamCreateWithPriority: g!(hipStreamCreateWithPriority),
            hipStreamDestroy: g!(hipStreamDestroy),
            hipStreamSynchronize: g!(hipStreamSynchronize),
            hipStreamWaitEvent: g!(hipStreamWaitEvent),
            hipDeviceGetStreamPriorityRange: g!(hipDeviceGetStreamPriorityRange),
            hipEventCreate: g!(hipEventCreate),
            hipEventCreateWithFlags: g!(hipEventCreateWithFlags),
            hipEventDestroy: g!(hipEventDestroy),
            hipEventRecord: g!(hipEventRecord),
            hipEventSynchronize: g!(hipEventSynchronize),
            hipEventElapsedTime: g!(hipEventElapsedTime),
            hipMalloc: g!(hipMalloc),
            hipMallocAsync: g!(hipMallocAsync),
            hipFree: g!(hipFree),
            hipFreeAsync: g!(hipFreeAsync),
            hipMemsetAsync: g!(hipMemsetAsync),
            hipMemcpyAsync: g!(hipMemcpyAsync),
            hipMemcpyHtoDAsync: g!(hipMemcpyHtoDAsync),
            hipMemcpyDtoHAsync: g!(hipMemcpyDtoHAsync),
            hipMemcpyDtoDAsync: g!(hipMemcpyDtoDAsync),
            hipHostMalloc: g!(hipHostMalloc),
            hipHostFree: g!(hipHostFree),
            hipModuleLoadData: g!(hipModuleLoadData),
            hipModuleUnload: g!(hipModuleUnload),
            hipModuleGetFunction: g!(hipModuleGetFunction),
            hipModuleLaunchKernel: g!(hipModuleLaunchKernel),
        })
    }
}

/// Hot-path cache. After the first successful `fns()` call this holds a
/// non-null pointer to the function table; subsequent calls become a single
/// acquire atomic load + null check, avoiding the `OnceLock` walk on every
/// wrapped HIP op.
static FNS_HOT: AtomicPtr<HipFns> = AtomicPtr::new(std::ptr::null_mut());

#[inline]
pub fn fns() -> Result<&'static HipFns, &'static LoadError> {
    // Fast path: pointer cached, library loaded.
    let cached = FNS_HOT.load(Ordering::Acquire);
    if !cached.is_null() {
        // SAFETY: only ever set to a `&'static HipFns` reference below,
        // and never cleared, so the pointer is valid for 'static.
        return Ok(unsafe { &*cached });
    }
    fns_slow()
}

#[cold]
#[inline(never)]
fn fns_slow() -> Result<&'static HipFns, &'static LoadError> {
    let r = FNS.get_or_init(|| {
        let lib = LIB.as_ref().map_err(|e| match e {
            LoadError::LibraryNotFound { tried, last } => LoadError::LibraryNotFound {
                tried: tried.clone(),
                last: last.clone(),
            },
            LoadError::SymbolMissing { lib, symbol, err } => LoadError::SymbolMissing {
                lib,
                symbol,
                err: err.clone(),
            },
        })?;
        load_fns(lib)
    });
    let r = r.as_ref()?;
    // Publish the static pointer for the hot path. `Release` so that the
    // OnceLock writes happen-before the pointer becomes observable.
    FNS_HOT.store(r as *const _ as *mut _, Ordering::Release);
    Ok(r)
}

pub fn is_available() -> bool {
    fns().is_ok()
}
