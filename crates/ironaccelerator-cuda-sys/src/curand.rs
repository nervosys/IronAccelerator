//! cuRAND — pseudo-random number generation.
//!
//! We bind just enough to back [`ironaccelerator-cuda::rng::Rng`]:
//! create/destroy a host generator, set seed/offset/stream, and the
//! four fill variants (uniform/normal × f32/f64).

use crate::driver::{CUdeviceptr, CUstream};
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::c_void;
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CurandGenerator(pub *mut c_void);
unsafe impl Send for CurandGenerator {}
unsafe impl Sync for CurandGenerator {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurandStatus {
    Success = 0,
    VersionMismatch = 100,
    NotInitialized = 101,
    AllocationFailed = 102,
    TypeError = 103,
    OutOfRange = 104,
    LengthNotMultiple = 105,
    DoublePrecisionRequired = 106,
    LaunchFailure = 201,
    PreexistingFailure = 202,
    InitializationFailed = 203,
    ArchMismatch = 204,
    InternalError = 999,
    Other = 0xFFFF_FFFF,
}

impl CurandStatus {
    pub fn from_raw(r: u32) -> Self {
        match r {
            0 => Self::Success,
            100 => Self::VersionMismatch,
            101 => Self::NotInitialized,
            102 => Self::AllocationFailed,
            103 => Self::TypeError,
            104 => Self::OutOfRange,
            105 => Self::LengthNotMultiple,
            106 => Self::DoublePrecisionRequired,
            201 => Self::LaunchFailure,
            202 => Self::PreexistingFailure,
            203 => Self::InitializationFailed,
            204 => Self::ArchMismatch,
            999 => Self::InternalError,
            _ => Self::Other,
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

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CurandRngType {
    PseudoXorwow = 101,
    PseudoMrg32k3a = 121,
    PseudoMtgp32 = 141,
    PseudoMt19937 = 142,
    PseudoPhilox4_32_10 = 161,
}

pub struct CurandFns {
    pub curandCreateGenerator:
        unsafe extern "C" fn(*mut CurandGenerator, CurandRngType) -> CurandStatus,
    pub curandDestroyGenerator: unsafe extern "C" fn(CurandGenerator) -> CurandStatus,
    pub curandSetStream: unsafe extern "C" fn(CurandGenerator, CUstream) -> CurandStatus,
    pub curandSetPseudoRandomGeneratorSeed:
        unsafe extern "C" fn(CurandGenerator, u64) -> CurandStatus,
    pub curandSetGeneratorOffset: unsafe extern "C" fn(CurandGenerator, u64) -> CurandStatus,

    pub curandGenerate: unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize) -> CurandStatus,
    pub curandGenerateLongLong:
        unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize) -> CurandStatus,
    pub curandGenerateUniform:
        unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize) -> CurandStatus,
    pub curandGenerateUniformDouble:
        unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize) -> CurandStatus,
    pub curandGenerateNormal:
        unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize, f32, f32) -> CurandStatus,
    pub curandGenerateNormalDouble:
        unsafe extern "C" fn(CurandGenerator, CUdeviceptr, usize, f64, f64) -> CurandStatus,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcurand.so.10",
        "libcurand.so",
        "curand64_10.dll",
        "curand64_13.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CurandFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CurandFns {
            curandCreateGenerator: sym(lib, "curand", "curandCreateGenerator")?,
            curandDestroyGenerator: sym(lib, "curand", "curandDestroyGenerator")?,
            curandSetStream: sym(lib, "curand", "curandSetStream")?,
            curandSetPseudoRandomGeneratorSeed: sym(
                lib,
                "curand",
                "curandSetPseudoRandomGeneratorSeed",
            )?,
            curandSetGeneratorOffset: sym(lib, "curand", "curandSetGeneratorOffset")?,
            curandGenerate: sym(lib, "curand", "curandGenerate")?,
            curandGenerateLongLong: sym(lib, "curand", "curandGenerateLongLong")?,
            curandGenerateUniform: sym(lib, "curand", "curandGenerateUniform")?,
            curandGenerateUniformDouble: sym(lib, "curand", "curandGenerateUniformDouble")?,
            curandGenerateNormal: sym(lib, "curand", "curandGenerateNormal")?,
            curandGenerateNormalDouble: sym(lib, "curand", "curandGenerateNormalDouble")?,
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static CurandFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
