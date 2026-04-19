//! hipBLAS — dense Level 3 matmul. We bind the legacy handle API because
//! that's what's stable across rocm 6.x; hipBLASLt (the FP8-era front-end)
//! lives in [`crate::hipblaslt`].

use crate::hip::HipStream;
use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct HipblasHandle(pub *mut c_void);
unsafe impl Send for HipblasHandle {} unsafe impl Sync for HipblasHandle {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipblasStatus {
    Success                 = 0,
    NotInitialized          = 1,
    AllocFailed             = 3,
    InvalidValue            = 7,
    MappingError            = 11,
    ExecutionFailed         = 13,
    InternalError           = 14,
    NotSupported            = 15,
    ArchMismatch            = 16,
    HandleIsNullptr         = 17,
    InvalidEnum             = 18,
    Unknown                 = 19,
    Other                   = 0xFFFF_FFFF,
}
impl HipblasStatus {
    #[inline] pub fn ok(self) -> Result<(), Self> { if self == Self::Success { Ok(()) } else { Err(self) } }
    #[inline] pub fn is_ok(self) -> bool { self == Self::Success }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasOp { N = 111, T = 112, C = 113 }

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasDataType {
    R16F   = 2,   // hipR16F
    R32F   = 0,
    R64F   = 1,
    R16BF  = 14,
    R8I    = 3,
    R8F_E4M3 = 28,
    R8F_E5M2 = 29,
}

#[allow(clippy::type_complexity)]
pub struct HipblasFns {
    pub hipblasCreate: unsafe extern "C" fn(*mut HipblasHandle) -> HipblasStatus,
    pub hipblasDestroy: unsafe extern "C" fn(HipblasHandle) -> HipblasStatus,
    pub hipblasSetStream: unsafe extern "C" fn(HipblasHandle, HipStream) -> HipblasStatus,
    pub hipblasGetStream: unsafe extern "C" fn(HipblasHandle, *mut HipStream) -> HipblasStatus,
    pub hipblasGetVersion: unsafe extern "C" fn(HipblasHandle, *mut c_int) -> HipblasStatus,

    pub hipblasSgemm: unsafe extern "C" fn(
        HipblasHandle, HipblasOp, HipblasOp,
        c_int, c_int, c_int,
        *const f32,
        *const f32, c_int,
        *const f32, c_int,
        *const f32,
        *mut f32, c_int,
    ) -> HipblasStatus,

    pub hipblasDgemm: unsafe extern "C" fn(
        HipblasHandle, HipblasOp, HipblasOp,
        c_int, c_int, c_int,
        *const f64,
        *const f64, c_int,
        *const f64, c_int,
        *const f64,
        *mut f64, c_int,
    ) -> HipblasStatus,

    /// General ex path — supports mixed-precision matmul (bf16/fp16 in/out,
    /// fp32 compute), the path most modern workloads use.
    pub hipblasGemmEx: unsafe extern "C" fn(
        HipblasHandle, HipblasOp, HipblasOp,
        c_int, c_int, c_int,
        *const c_void,
        *const c_void, HipblasDataType, c_int,
        *const c_void, HipblasDataType, c_int,
        *const c_void,
        *mut c_void, HipblasDataType, c_int,
        HipblasDataType,  // compute type
        u32,              // algo
    ) -> HipblasStatus,
}

static LIB: LazyLock<LoaderResult<Library>> = LazyLock::new(|| {
    try_load(&["libhipblas.so", "libhipblas.so.2", "hipblas.dll"])
});
static FNS: OnceLock<LoaderResult<HipblasFns>> = OnceLock::new();

fn load_fns(lib: &Library) -> LoaderResult<HipblasFns> {
    macro_rules! g { ($s:ident) => { sym(lib, "hipblas", stringify!($s))? } }
    unsafe {
        Ok(HipblasFns {
            hipblasCreate: g!(hipblasCreate),
            hipblasDestroy: g!(hipblasDestroy),
            hipblasSetStream: g!(hipblasSetStream),
            hipblasGetStream: g!(hipblasGetStream),
            hipblasGetVersion: g!(hipblasGetVersion),
            hipblasSgemm: g!(hipblasSgemm),
            hipblasDgemm: g!(hipblasDgemm),
            hipblasGemmEx: g!(hipblasGemmEx),
        })
    }
}

pub fn fns() -> Result<&'static HipblasFns, &'static LoadError> {
    FNS.get_or_init(|| {
        let lib = LIB.as_ref().map_err(clone_err)?;
        load_fns(lib)
    }).as_ref()
}

pub fn is_available() -> bool { fns().is_ok() }

fn clone_err(e: &LoadError) -> LoadError {
    match e {
        LoadError::LibraryNotFound { tried, last } =>
            LoadError::LibraryNotFound { tried: tried.clone(), last: last.clone() },
        LoadError::SymbolMissing { lib, symbol, err } =>
            LoadError::SymbolMissing { lib, symbol, err: err.clone() },
    }
}
