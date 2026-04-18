//! cuBLASLt — descriptor-based matmul API. The path to FP8 GEMM.
//!
//! Library: `libcublasLt.so` / `cublasLt64_*.dll`. Targets CUDA 13.2.

use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CublasLtHandle(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CublasLtMatmulDesc(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CublasLtMatrixLayout(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CublasLtMatmulPreference(pub *mut c_void);

unsafe impl Send for CublasLtHandle {} unsafe impl Sync for CublasLtHandle {}
unsafe impl Send for CublasLtMatmulDesc {} unsafe impl Sync for CublasLtMatmulDesc {}
unsafe impl Send for CublasLtMatrixLayout {} unsafe impl Sync for CublasLtMatrixLayout {}
unsafe impl Send for CublasLtMatmulPreference {} unsafe impl Sync for CublasLtMatmulPreference {}

/// cuBLAS status code. 0 = success.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16,
    Other = 0xFFFF_FFFF,
}

impl CublasStatus {
    pub fn from_raw(r: u32) -> Self {
        match r {
            0 => Self::Success, 1 => Self::NotInitialized, 3 => Self::AllocFailed,
            7 => Self::InvalidValue, 8 => Self::ArchMismatch, 11 => Self::MappingError,
            13 => Self::ExecutionFailed, 14 => Self::InternalError,
            15 => Self::NotSupported, 16 => Self::LicenseError, _ => Self::Other,
        }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

/// `cudaDataType_t` — superset covers all element types cuBLASLt accepts.
/// Subset matching CUDA 13.2. Values are taken from `library_types.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDataType {
    R16F = 2,  C16F = 6,
    R16BF = 14, C16BF = 15,
    R32F = 0,  C32F = 4,
    R64F = 1,  C64F = 5,
    R8I = 3,   R8U = 8,
    R32I = 10, R32U = 12,
    R8F_E4M3 = 28,
    R8F_E5M2 = 29,
}

/// `cublasComputeType_t` — what tensor-core path the GEMM uses.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasComputeType {
    F16 = 64,
    F16Pedantic = 65,
    F32 = 68,
    F32Pedantic = 69,
    F32FastF16 = 74,
    F32FastBf16 = 75,
    F32FastTf32 = 77,
    F64 = 70,
    F64Pedantic = 71,
    I32 = 72,
}

/// `cublasOperation_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasOp { N = 0, T = 1, C = 2 }

/// `cublasLtMatmulDescAttributes_t` — the attrs we actually set.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasLtMatmulDescAttr {
    TransA = 3,
    TransB = 4,
    Epilogue = 23,
    BiasPointer = 22,
    ScaleA = 17,
    ScaleB = 18,
    ScaleC = 19,
    ScaleD = 20,
    AmaxDPointer = 21,
}

/// `cublasLtMatrixLayoutAttributes_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasLtMatrixLayoutAttr {
    Type = 0, Order = 1, Rows = 2, Cols = 3, Ld = 4,
    BatchCount = 5, StridedBatchOffset = 6,
}

/// `cublasLtMatmulPreferenceAttributes_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasLtMatmulPrefAttr {
    MaxWorkspaceBytes = 0,
}

/// `cublasLtOrder_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasLtOrder { Col = 0, Row = 1, Col32 = 2 }

/// Heuristic result (truncated — only what we read).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CublasLtMatmulHeuristicResult {
    pub algo: [u64; 8],       // opaque cublasLtMatmulAlgo_t
    pub workspace_size: usize,
    pub state: u32,
    pub waves_count: f32,
    pub reserved: [u32; 4],
}

impl Default for CublasLtMatmulHeuristicResult {
    fn default() -> Self {
        Self { algo: [0; 8], workspace_size: 0, state: 0, waves_count: 0.0, reserved: [0; 4] }
    }
}

pub struct CublasLtFns {
    pub cublasLtCreate: unsafe extern "C" fn(*mut CublasLtHandle) -> CublasStatus,
    pub cublasLtDestroy: unsafe extern "C" fn(CublasLtHandle) -> CublasStatus,
    pub cublasLtGetVersion: unsafe extern "C" fn() -> usize,

    pub cublasLtMatmulDescCreate: unsafe extern "C" fn(
        *mut CublasLtMatmulDesc, CublasComputeType, CudaDataType,
    ) -> CublasStatus,
    pub cublasLtMatmulDescDestroy: unsafe extern "C" fn(CublasLtMatmulDesc) -> CublasStatus,
    pub cublasLtMatmulDescSetAttribute: unsafe extern "C" fn(
        CublasLtMatmulDesc, CublasLtMatmulDescAttr, *const c_void, usize,
    ) -> CublasStatus,

    pub cublasLtMatrixLayoutCreate: unsafe extern "C" fn(
        *mut CublasLtMatrixLayout, CudaDataType, u64, u64, i64,
    ) -> CublasStatus,
    pub cublasLtMatrixLayoutDestroy: unsafe extern "C" fn(CublasLtMatrixLayout) -> CublasStatus,
    pub cublasLtMatrixLayoutSetAttribute: unsafe extern "C" fn(
        CublasLtMatrixLayout, CublasLtMatrixLayoutAttr, *const c_void, usize,
    ) -> CublasStatus,

    pub cublasLtMatmulPreferenceCreate: unsafe extern "C" fn(
        *mut CublasLtMatmulPreference,
    ) -> CublasStatus,
    pub cublasLtMatmulPreferenceDestroy: unsafe extern "C" fn(
        CublasLtMatmulPreference,
    ) -> CublasStatus,
    pub cublasLtMatmulPreferenceSetAttribute: unsafe extern "C" fn(
        CublasLtMatmulPreference, CublasLtMatmulPrefAttr, *const c_void, usize,
    ) -> CublasStatus,

    pub cublasLtMatmulAlgoGetHeuristic: unsafe extern "C" fn(
        CublasLtHandle, CublasLtMatmulDesc,
        CublasLtMatrixLayout, CublasLtMatrixLayout, CublasLtMatrixLayout, CublasLtMatrixLayout,
        CublasLtMatmulPreference,
        c_int, *mut CublasLtMatmulHeuristicResult, *mut c_int,
    ) -> CublasStatus,

    pub cublasLtMatmul: unsafe extern "C" fn(
        CublasLtHandle, CublasLtMatmulDesc,
        *const c_void,                      // alpha
        *const c_void, CublasLtMatrixLayout, // A
        *const c_void, CublasLtMatrixLayout, // B
        *const c_void,                      // beta
        *const c_void, CublasLtMatrixLayout, // C
        *mut c_void,   CublasLtMatrixLayout, // D
        *const c_void,                      // algo (opaque)
        *mut c_void, usize,                 // workspace, bytes
        CUstream,
    ) -> CublasStatus,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so",
        "cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CublasLtFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CublasLtFns {
            cublasLtCreate: sym(lib, "cublasLt", "cublasLtCreate")?,
            cublasLtDestroy: sym(lib, "cublasLt", "cublasLtDestroy")?,
            cublasLtGetVersion: sym(lib, "cublasLt", "cublasLtGetVersion")?,
            cublasLtMatmulDescCreate: sym(lib, "cublasLt", "cublasLtMatmulDescCreate")?,
            cublasLtMatmulDescDestroy: sym(lib, "cublasLt", "cublasLtMatmulDescDestroy")?,
            cublasLtMatmulDescSetAttribute: sym(lib, "cublasLt", "cublasLtMatmulDescSetAttribute")?,
            cublasLtMatrixLayoutCreate: sym(lib, "cublasLt", "cublasLtMatrixLayoutCreate")?,
            cublasLtMatrixLayoutDestroy: sym(lib, "cublasLt", "cublasLtMatrixLayoutDestroy")?,
            cublasLtMatrixLayoutSetAttribute: sym(lib, "cublasLt", "cublasLtMatrixLayoutSetAttribute")?,
            cublasLtMatmulPreferenceCreate: sym(lib, "cublasLt", "cublasLtMatmulPreferenceCreate")?,
            cublasLtMatmulPreferenceDestroy: sym(lib, "cublasLt", "cublasLtMatmulPreferenceDestroy")?,
            cublasLtMatmulPreferenceSetAttribute: sym(lib, "cublasLt", "cublasLtMatmulPreferenceSetAttribute")?,
            cublasLtMatmulAlgoGetHeuristic: sym(lib, "cublasLt", "cublasLtMatmulAlgoGetHeuristic")?,
            cublasLtMatmul: sym(lib, "cublasLt", "cublasLtMatmul")?,
        })
    }
});

pub fn fns() -> Result<&'static CublasLtFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
