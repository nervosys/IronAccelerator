//! HIPRTC — runtime compilation of HIP C++ to a device code object.
//!
//! The AMD analogue of NVRTC, and deliberately API-compatible with it. Library:
//! `libhiprtc.so` / `hiprtc*.dll`. The output is a **code object** (not PTX)
//! that `hipModuleLoadData` accepts directly, so the compile → load → launch
//! pipeline mirrors CUDA's NVRTC path one-for-one.
//!
//! Dynamically loaded on first use like every other library here; a host
//! without ROCm simply reports the runtime as unavailable.

use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HiprtcProgram(pub *mut c_void);

unsafe impl Send for HiprtcProgram {}
unsafe impl Sync for HiprtcProgram {}

/// HIPRTC status codes. The numeric values match NVRTC's, which is intentional
/// on AMD's part — the two runtime compilers share an interface shape.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiprtcResult {
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
    Other = 0xFFFF_FFFF,
}

impl HiprtcResult {
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

pub struct HiprtcFns {
    pub hiprtcCreateProgram: unsafe extern "C" fn(
        *mut HiprtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> HiprtcResult,
    pub hiprtcDestroyProgram: unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult,
    pub hiprtcCompileProgram:
        unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HiprtcResult,
    pub hiprtcGetCodeSize: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    pub hiprtcGetCode: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    pub hiprtcGetProgramLogSize: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    pub hiprtcGetProgramLog: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    pub hiprtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> HiprtcResult,
    pub hiprtcGetErrorString: unsafe extern "C" fn(HiprtcResult) -> *const c_char,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libhiprtc.so",
        "libhiprtc.so.6",
        "libhiprtc.so.5",
        // Windows ships HIPRTC as a version-stamped DLL; the bare name is a
        // common install-time symlink/copy.
        "hiprtc.dll",
        "hiprtc0605.dll",
        "hiprtc0604.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<HiprtcFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(HiprtcFns {
            hiprtcCreateProgram: sym(lib, "hiprtc", "hiprtcCreateProgram")?,
            hiprtcDestroyProgram: sym(lib, "hiprtc", "hiprtcDestroyProgram")?,
            hiprtcCompileProgram: sym(lib, "hiprtc", "hiprtcCompileProgram")?,
            hiprtcGetCodeSize: sym(lib, "hiprtc", "hiprtcGetCodeSize")?,
            hiprtcGetCode: sym(lib, "hiprtc", "hiprtcGetCode")?,
            hiprtcGetProgramLogSize: sym(lib, "hiprtc", "hiprtcGetProgramLogSize")?,
            hiprtcGetProgramLog: sym(lib, "hiprtc", "hiprtcGetProgramLog")?,
            hiprtcVersion: sym(lib, "hiprtc", "hiprtcVersion")?,
            hiprtcGetErrorString: sym(lib, "hiprtc", "hiprtcGetErrorString")?,
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static HiprtcFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
