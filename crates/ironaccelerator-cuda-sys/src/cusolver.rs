//! cuSOLVER Dense — LU/QR/SVD/Cholesky on dense matrices.

use crate::cublas_lt::CublasOp;
use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CusolverDnHandle(pub *mut c_void);
unsafe impl Send for CusolverDnHandle {}
unsafe impl Sync for CusolverDnHandle {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CusolverStatus {
    Success = 0, NotInitialized = 1, AllocFailed = 2, InvalidValue = 3,
    ArchMismatch = 4, MappingError = 5, ExecutionFailed = 6, InternalError = 7,
    MatrixTypeNotSupported = 8, NotSupported = 9, ZeroPivot = 10, InvalidLicense = 11,
    Other = 0xFFFF_FFFF,
}

impl CusolverStatus {
    pub fn from_raw(r: u32) -> Self {
        if r <= 11 { unsafe { std::mem::transmute(r) } } else { Self::Other }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusolverFillMode { Lower = 0, Upper = 1 }
#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusolverEigMode { NoVector = 0, Vector = 1 }

pub struct CusolverDnFns {
    pub cusolverDnCreate: unsafe extern "C" fn(*mut CusolverDnHandle) -> CusolverStatus,
    pub cusolverDnDestroy: unsafe extern "C" fn(CusolverDnHandle) -> CusolverStatus,
    pub cusolverDnSetStream: unsafe extern "C" fn(CusolverDnHandle, CUstream) -> CusolverStatus,
    pub cusolverDnGetStream: unsafe extern "C" fn(CusolverDnHandle, *mut CUstream) -> CusolverStatus,

    // Sgetrf (LU, single-precision)
    pub cusolverDnSgetrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnSgetrf: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f32, c_int, *mut f32, *mut c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnSgetrs: unsafe extern "C" fn(
        CusolverDnHandle, CublasOp, c_int, c_int,
        *const f32, c_int, *const c_int, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,

    // Spotrf (Cholesky)
    pub cusolverDnSpotrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, CusolverFillMode, c_int, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnSpotrf: unsafe extern "C" fn(
        CusolverDnHandle, CusolverFillMode, c_int, *mut f32, c_int, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,

    // Sgeqrf (QR)
    pub cusolverDnSgeqrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnSgeqrf: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f32, c_int, *mut f32, *mut f32, c_int, *mut c_int,
    ) -> CusolverStatus,

    // Dgetrf / Dgetrs (LU, double-precision)
    pub cusolverDnDgetrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnDgetrf: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f64, c_int, *mut f64, *mut c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnDgetrs: unsafe extern "C" fn(
        CusolverDnHandle, CublasOp, c_int, c_int,
        *const f64, c_int, *const c_int, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,

    // Dpotrf (Cholesky)
    pub cusolverDnDpotrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, CusolverFillMode, c_int, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnDpotrf: unsafe extern "C" fn(
        CusolverDnHandle, CusolverFillMode, c_int, *mut f64, c_int, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,

    // Dgeqrf (QR)
    pub cusolverDnDgeqrf_bufferSize: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,
    pub cusolverDnDgeqrf: unsafe extern "C" fn(
        CusolverDnHandle, c_int, c_int, *mut f64, c_int, *mut f64, *mut f64, c_int, *mut c_int,
    ) -> CusolverStatus,
}

fn candidates() -> &'static [&'static str] {
    &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so",
      "cusolver64_12.dll", "cusolver64_11.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CusolverDnFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CusolverDnFns {
            cusolverDnCreate: sym(lib, "cusolver", "cusolverDnCreate")?,
            cusolverDnDestroy: sym(lib, "cusolver", "cusolverDnDestroy")?,
            cusolverDnSetStream: sym(lib, "cusolver", "cusolverDnSetStream")?,
            cusolverDnGetStream: sym(lib, "cusolver", "cusolverDnGetStream")?,
            cusolverDnSgetrf_bufferSize: sym(lib, "cusolver", "cusolverDnSgetrf_bufferSize")?,
            cusolverDnSgetrf: sym(lib, "cusolver", "cusolverDnSgetrf")?,
            cusolverDnSgetrs: sym(lib, "cusolver", "cusolverDnSgetrs")?,
            cusolverDnSpotrf_bufferSize: sym(lib, "cusolver", "cusolverDnSpotrf_bufferSize")?,
            cusolverDnSpotrf: sym(lib, "cusolver", "cusolverDnSpotrf")?,
            cusolverDnSgeqrf_bufferSize: sym(lib, "cusolver", "cusolverDnSgeqrf_bufferSize")?,
            cusolverDnSgeqrf: sym(lib, "cusolver", "cusolverDnSgeqrf")?,
            cusolverDnDgetrf_bufferSize: sym(lib, "cusolver", "cusolverDnDgetrf_bufferSize")?,
            cusolverDnDgetrf: sym(lib, "cusolver", "cusolverDnDgetrf")?,
            cusolverDnDgetrs: sym(lib, "cusolver", "cusolverDnDgetrs")?,
            cusolverDnDpotrf_bufferSize: sym(lib, "cusolver", "cusolverDnDpotrf_bufferSize")?,
            cusolverDnDpotrf: sym(lib, "cusolver", "cusolverDnDpotrf")?,
            cusolverDnDgeqrf_bufferSize: sym(lib, "cusolver", "cusolverDnDgeqrf_bufferSize")?,
            cusolverDnDgeqrf: sym(lib, "cusolver", "cusolverDnDgeqrf")?,
        })
    }
});

pub fn fns() -> Result<&'static CusolverDnFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
