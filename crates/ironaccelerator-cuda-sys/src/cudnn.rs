//! cuDNN — deep-learning primitives (MHA, convolution).
//!
//! We bind a minimal surface: handle create/destroy, stream binding,
//! version query, and the v9 frontend `cudnnGraphBackend` entry points
//! used by the safe MHA / conv wrappers.

use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CudnnHandle(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CudnnBackendDescriptor(pub *mut c_void);

unsafe impl Send for CudnnHandle {} unsafe impl Sync for CudnnHandle {}
unsafe impl Send for CudnnBackendDescriptor {} unsafe impl Sync for CudnnBackendDescriptor {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 2,
    BadParam = 3,
    InternalError = 4,
    InvalidValue = 5,
    ArchMismatch = 6,
    MappingError = 7,
    ExecutionFailed = 8,
    NotSupported = 9,
    LicenseError = 10,
    RuntimePrerequisiteMissing = 11,
    RuntimeInProgress = 12,
    RuntimeFpOverflow = 13,
    VersionMismatch = 14,
    Other = 0xFFFF_FFFF,
}

impl CudnnStatus {
    pub fn from_raw(r: u32) -> Self {
        if r <= 14 { unsafe { std::mem::transmute(r) } } else { Self::Other }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

/// Data type enum matching `cudnnDataType_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CudnnDataType {
    Float = 0, Double = 1, Half = 2, Int8 = 3, Int32 = 4,
    Int8x4 = 5, Uint8 = 6, Uint8x4 = 7, Int8x32 = 8,
    Bfloat16 = 9, Int64 = 10, Boolean = 11,
    Fp8E4M3 = 12, Fp8E5M2 = 13, FastFloat32 = 14,
}

pub struct CudnnFns {
    pub cudnnCreate: unsafe extern "C" fn(*mut CudnnHandle) -> CudnnStatus,
    pub cudnnDestroy: unsafe extern "C" fn(CudnnHandle) -> CudnnStatus,
    pub cudnnSetStream: unsafe extern "C" fn(CudnnHandle, CUstream) -> CudnnStatus,
    pub cudnnGetStream: unsafe extern "C" fn(CudnnHandle, *mut CUstream) -> CudnnStatus,
    pub cudnnGetVersion: unsafe extern "C" fn() -> usize,
    pub cudnnGetCudartVersion: unsafe extern "C" fn() -> usize,

    // v9 graph backend (the path forward for MHA/conv)
    pub cudnnBackendCreateDescriptor: unsafe extern "C" fn(
        u32, *mut CudnnBackendDescriptor,
    ) -> CudnnStatus,
    pub cudnnBackendDestroyDescriptor: unsafe extern "C" fn(
        CudnnBackendDescriptor,
    ) -> CudnnStatus,
    pub cudnnBackendSetAttribute: unsafe extern "C" fn(
        CudnnBackendDescriptor, u32, u32, i64, *const c_void,
    ) -> CudnnStatus,
    pub cudnnBackendGetAttribute: unsafe extern "C" fn(
        CudnnBackendDescriptor, u32, u32, i64, *mut i64, *mut c_void,
    ) -> CudnnStatus,
    pub cudnnBackendFinalize: unsafe extern "C" fn(CudnnBackendDescriptor) -> CudnnStatus,
    pub cudnnBackendExecute: unsafe extern "C" fn(
        CudnnHandle, CudnnBackendDescriptor, CudnnBackendDescriptor,
    ) -> CudnnStatus,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcudnn.so.9", "libcudnn.so.8", "libcudnn.so",
        "cudnn64_9.dll", "cudnn64_8.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CudnnFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CudnnFns {
            cudnnCreate: sym(lib, "cudnn", "cudnnCreate")?,
            cudnnDestroy: sym(lib, "cudnn", "cudnnDestroy")?,
            cudnnSetStream: sym(lib, "cudnn", "cudnnSetStream")?,
            cudnnGetStream: sym(lib, "cudnn", "cudnnGetStream")?,
            cudnnGetVersion: sym(lib, "cudnn", "cudnnGetVersion")?,
            cudnnGetCudartVersion: sym(lib, "cudnn", "cudnnGetCudartVersion")?,
            cudnnBackendCreateDescriptor: sym(lib, "cudnn", "cudnnBackendCreateDescriptor")?,
            cudnnBackendDestroyDescriptor: sym(lib, "cudnn", "cudnnBackendDestroyDescriptor")?,
            cudnnBackendSetAttribute: sym(lib, "cudnn", "cudnnBackendSetAttribute")?,
            cudnnBackendGetAttribute: sym(lib, "cudnn", "cudnnBackendGetAttribute")?,
            cudnnBackendFinalize: sym(lib, "cudnn", "cudnnBackendFinalize")?,
            cudnnBackendExecute: sym(lib, "cudnn", "cudnnBackendExecute")?,
        })
    }
});

pub fn fns() -> Result<&'static CudnnFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
