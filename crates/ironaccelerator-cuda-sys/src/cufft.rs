//! cuFFT — batched 1D/2D/3D FFTs.

use crate::driver::{CUdeviceptr, CUstream};
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::c_int;
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CufftHandle(pub c_int);

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CufftResult {
    Success = 0,
    InvalidPlan = 1,
    AllocFailed = 2,
    InvalidType = 3,
    InvalidValue = 4,
    InternalError = 5,
    ExecFailed = 6,
    SetupFailed = 7,
    InvalidSize = 8,
    UnalignedData = 9,
    IncompleteParameterList = 10,
    InvalidDevice = 11,
    ParseError = 12,
    NoWorkspace = 13,
    NotImplemented = 14,
    LicenseError = 15,
    NotSupported = 16,
    Other = 0xFFFF_FFFF,
}

impl CufftResult {
    pub fn from_raw(r: u32) -> Self {
        if r <= 16 {
            unsafe { std::mem::transmute(r) }
        } else {
            Self::Other
        }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success {
            Ok(())
        } else {
            Err(self)
        }
    }
    pub fn is_ok(self) -> bool {
        self == Self::Success
    }
}

/// `cufftType` transform type.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CufftType {
    R2C = 0x2a,
    C2R = 0x2c,
    C2C = 0x29,
    D2Z = 0x6a,
    Z2D = 0x6c,
    Z2Z = 0x69,
}

pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

pub struct CufftFns {
    pub cufftCreate: unsafe extern "C" fn(*mut CufftHandle) -> CufftResult,
    pub cufftDestroy: unsafe extern "C" fn(CufftHandle) -> CufftResult,
    pub cufftSetStream: unsafe extern "C" fn(CufftHandle, CUstream) -> CufftResult,
    pub cufftGetVersion: unsafe extern "C" fn(*mut c_int) -> CufftResult,

    pub cufftPlan1d: unsafe extern "C" fn(*mut CufftHandle, c_int, CufftType, c_int) -> CufftResult,
    pub cufftPlanMany: unsafe extern "C" fn(
        *mut CufftHandle,
        c_int,
        *const c_int,
        *const c_int,
        c_int,
        c_int,
        *const c_int,
        c_int,
        c_int,
        CufftType,
        c_int,
    ) -> CufftResult,

    pub cufftExecR2C: unsafe extern "C" fn(CufftHandle, CUdeviceptr, CUdeviceptr) -> CufftResult,
    pub cufftExecC2R: unsafe extern "C" fn(CufftHandle, CUdeviceptr, CUdeviceptr) -> CufftResult,
    pub cufftExecC2C:
        unsafe extern "C" fn(CufftHandle, CUdeviceptr, CUdeviceptr, c_int) -> CufftResult,
    pub cufftExecZ2Z:
        unsafe extern "C" fn(CufftHandle, CUdeviceptr, CUdeviceptr, c_int) -> CufftResult,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcufft.so.11",
        "libcufft.so.10",
        "libcufft.so",
        "cufft64_11.dll",
        "cufft64_10.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CufftFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CufftFns {
            cufftCreate: sym(lib, "cufft", "cufftCreate")?,
            cufftDestroy: sym(lib, "cufft", "cufftDestroy")?,
            cufftSetStream: sym(lib, "cufft", "cufftSetStream")?,
            cufftGetVersion: sym(lib, "cufft", "cufftGetVersion")?,
            cufftPlan1d: sym(lib, "cufft", "cufftPlan1d")?,
            cufftPlanMany: sym(lib, "cufft", "cufftPlanMany")?,
            cufftExecR2C: sym(lib, "cufft", "cufftExecR2C")?,
            cufftExecC2R: sym(lib, "cufft", "cufftExecC2R")?,
            cufftExecC2C: sym(lib, "cufft", "cufftExecC2C")?,
            cufftExecZ2Z: sym(lib, "cufft", "cufftExecZ2Z")?,
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static CufftFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
