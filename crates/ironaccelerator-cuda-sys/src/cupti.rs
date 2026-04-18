//! CUPTI — CUDA Profiling Tools Interface.
//!
//! We bind the Activity API (records flushed in batches) and the simpler
//! subscriber/callback API used to time individual driver/runtime calls.

use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_uint, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CuptiSubscriber(pub *mut c_void);
unsafe impl Send for CuptiSubscriber {}
unsafe impl Sync for CuptiSubscriber {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuptiResult {
    Success = 0, InvalidParameter = 1, InvalidDevice = 2, InvalidContext = 3,
    InvalidEventDomainId = 4, InvalidEventId = 5, InvalidEventName = 6,
    InvalidOperation = 7, OutOfMemory = 8, HardwareError = 9,
    NotCompatible = 10, NotInitialized = 11, NotReady = 12,
    NotSupported = 13, Other = 0xFFFF_FFFF,
}

impl CuptiResult {
    pub fn from_raw(r: u32) -> Self {
        if r <= 13 { unsafe { std::mem::transmute(r) } } else { Self::Other }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

/// `CUpti_ActivityKind` — subset we care about.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CuptiActivityKind {
    Invalid = 0,
    Memcpy = 1,
    Memset = 2,
    Kernel = 3,
    Driver = 4,
    Runtime = 5,
    Event = 6,
    Metric = 7,
    Device = 8,
    Context = 9,
    ConcurrentKernel = 10,
    Name = 11,
    Marker = 12,
    MarkerData = 13,
    SourceLocator = 14,
    GlobalAccess = 15,
    Branch = 16,
    Overhead = 17,
    CdpKernel = 18,
    PreemptionSource = 19,
}

/// Callback for buffer allocation requests (first arg receives allocation).
pub type CuptiBufferRequestedCb = unsafe extern "C" fn(
    *mut *mut u8, *mut usize, *mut usize,
);

/// Callback for buffer completion — hand the filled activity buffer back.
pub type CuptiBufferCompletedCb = unsafe extern "C" fn(
    *mut c_void,     // ctx
    u32,             // streamId
    *mut u8,         // buffer
    usize, usize,    // size, validSize
);

pub struct CuptiFns {
    pub cuptiActivityEnable: unsafe extern "C" fn(CuptiActivityKind) -> CuptiResult,
    pub cuptiActivityDisable: unsafe extern "C" fn(CuptiActivityKind) -> CuptiResult,
    pub cuptiActivityRegisterCallbacks: unsafe extern "C" fn(
        CuptiBufferRequestedCb, CuptiBufferCompletedCb,
    ) -> CuptiResult,
    pub cuptiActivityFlushAll: unsafe extern "C" fn(u32) -> CuptiResult,
    pub cuptiActivityGetNextRecord: unsafe extern "C" fn(
        *mut u8, usize, *mut *mut c_void,
    ) -> CuptiResult,
    pub cuptiGetVersion: unsafe extern "C" fn(*mut u32) -> CuptiResult,
    pub cuptiGetTimestamp: unsafe extern "C" fn(*mut u64) -> CuptiResult,
}

fn candidates() -> &'static [&'static str] {
    &["libcupti.so.13", "libcupti.so.12", "libcupti.so",
      "cupti64_2025.1.0.dll", "cupti64_2024.3.0.dll", "cupti.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CuptiFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CuptiFns {
            cuptiActivityEnable: sym(lib, "cupti", "cuptiActivityEnable")?,
            cuptiActivityDisable: sym(lib, "cupti", "cuptiActivityDisable")?,
            cuptiActivityRegisterCallbacks: sym(lib, "cupti", "cuptiActivityRegisterCallbacks")?,
            cuptiActivityFlushAll: sym(lib, "cupti", "cuptiActivityFlushAll")?,
            cuptiActivityGetNextRecord: sym(lib, "cupti", "cuptiActivityGetNextRecord")?,
            cuptiGetVersion: sym(lib, "cupti", "cuptiGetVersion")?,
            cuptiGetTimestamp: sym(lib, "cupti", "cuptiGetTimestamp")?,
        })
    }
});

pub fn fns() -> Result<&'static CuptiFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
