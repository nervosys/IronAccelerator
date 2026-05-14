//! NVRTC — runtime compilation of CUDA C++ to PTX.
//!
//! Library: `libnvrtc.so` / `nvrtc64_*.dll`. Targets CUDA 13.2.

use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NvrtcProgram(pub *mut c_void);

unsafe impl Send for NvrtcProgram {}
unsafe impl Sync for NvrtcProgram {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvrtcResult {
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    Compilation = 6,
    BuiltinOperationFailure = 7,
    NoNameExpressionsAfterCompilation = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11,
    TimeFileWriteFailed = 12,
    Other = 0xFFFF_FFFF,
}

impl NvrtcResult {
    pub fn from_raw(r: u32) -> Self {
        match r {
            0 => Self::Success,
            1 => Self::OutOfMemory,
            2 => Self::ProgramCreationFailure,
            3 => Self::InvalidInput,
            4 => Self::InvalidProgram,
            5 => Self::InvalidOption,
            6 => Self::Compilation,
            7 => Self::BuiltinOperationFailure,
            8 => Self::NoNameExpressionsAfterCompilation,
            9 => Self::NoLoweredNamesBeforeCompilation,
            10 => Self::NameExpressionNotValid,
            11 => Self::InternalError,
            12 => Self::TimeFileWriteFailed,
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

pub struct NvrtcFns {
    pub nvrtcCreateProgram: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    pub nvrtcDestroyProgram: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    pub nvrtcCompileProgram:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    pub nvrtcGetPTXSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    pub nvrtcGetPTX: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    pub nvrtcGetCUBINSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    pub nvrtcGetCUBIN: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    pub nvrtcGetProgramLogSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    pub nvrtcGetProgramLog: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    pub nvrtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
    pub nvrtcGetErrorString: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so",
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<NvrtcFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(NvrtcFns {
            nvrtcCreateProgram: sym(lib, "nvrtc", "nvrtcCreateProgram")?,
            nvrtcDestroyProgram: sym(lib, "nvrtc", "nvrtcDestroyProgram")?,
            nvrtcCompileProgram: sym(lib, "nvrtc", "nvrtcCompileProgram")?,
            nvrtcGetPTXSize: sym(lib, "nvrtc", "nvrtcGetPTXSize")?,
            nvrtcGetPTX: sym(lib, "nvrtc", "nvrtcGetPTX")?,
            nvrtcGetCUBINSize: sym(lib, "nvrtc", "nvrtcGetCUBINSize")?,
            nvrtcGetCUBIN: sym(lib, "nvrtc", "nvrtcGetCUBIN")?,
            nvrtcGetProgramLogSize: sym(lib, "nvrtc", "nvrtcGetProgramLogSize")?,
            nvrtcGetProgramLog: sym(lib, "nvrtc", "nvrtcGetProgramLog")?,
            nvrtcVersion: sym(lib, "nvrtc", "nvrtcVersion")?,
            nvrtcGetErrorString: sym(lib, "nvrtc", "nvrtcGetErrorString")?,
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static NvrtcFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
