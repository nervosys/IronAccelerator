//! cuDNN — deep-learning primitives (MHA, convolution).
//!
//! We bind a minimal surface: handle create/destroy, stream binding,
//! version query, and the v9 frontend `cudnnGraphBackend` entry points
//! used by the safe MHA / conv wrappers.

use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::c_void;
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CudnnHandle(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CudnnBackendDescriptor(pub *mut c_void);

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}
unsafe impl Send for CudnnBackendDescriptor {}
unsafe impl Sync for CudnnBackendDescriptor {}

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
        if r <= 14 {
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

/// Data type enum matching `cudnnDataType_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CudnnDataType {
    Float = 0,
    Double = 1,
    Half = 2,
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    Uint8 = 6,
    Uint8x4 = 7,
    Int8x32 = 8,
    Bfloat16 = 9,
    Int64 = 10,
    Boolean = 11,
    Fp8E4M3 = 12,
    Fp8E5M2 = 13,
    FastFloat32 = 14,
}

// ── v9 backend descriptor kinds (`cudnnBackendDescriptorType_t`) ───────────

pub const CUDNN_BACKEND_POINTWISE_DESCRIPTOR: u32 = 0;
pub const CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: u32 = 1;
pub const CUDNN_BACKEND_ENGINE_DESCRIPTOR: u32 = 2;
pub const CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: u32 = 3;
pub const CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: u32 = 4;
pub const CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: u32 = 5;
pub const CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: u32 = 13;
pub const CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: u32 = 15;
pub const CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: u32 = 16;
pub const CUDNN_BACKEND_TENSOR_DESCRIPTOR: u32 = 17;
pub const CUDNN_BACKEND_MATMUL_DESCRIPTOR: u32 = 18;
pub const CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: u32 = 19;
pub const CUDNN_BACKEND_REDUCTION_DESCRIPTOR: u32 = 21;
pub const CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: u32 = 22;

// ── v9 attribute names (`cudnnBackendAttributeName_t`) ─────────────────────

pub const CUDNN_ATTR_POINTWISE_MODE: u32 = 0;
pub const CUDNN_ATTR_POINTWISE_MATH_PREC: u32 = 1;
pub const CUDNN_ATTR_POINTWISE_AXIS: u32 = 8;

pub const CUDNN_ATTR_ENGINEHEUR_MODE: u32 = 200;
pub const CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH: u32 = 201;
pub const CUDNN_ATTR_ENGINEHEUR_RESULTS: u32 = 202;

pub const CUDNN_ATTR_ENGINECFG_ENGINE: u32 = 300;

pub const CUDNN_ATTR_EXECUTION_PLAN_HANDLE: u32 = 400;
pub const CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: u32 = 401;
pub const CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: u32 = 402;

pub const CUDNN_ATTR_ENGINE_OPERATION_GRAPH: u32 = 600;
pub const CUDNN_ATTR_ENGINE_GLOBAL_INDEX: u32 = 601;

pub const CUDNN_ATTR_MATMUL_COMP_TYPE: u32 = 700;

pub const CUDNN_ATTR_OPERATION_MATMUL_ADESC: u32 = 1100;
pub const CUDNN_ATTR_OPERATION_MATMUL_BDESC: u32 = 1101;
pub const CUDNN_ATTR_OPERATION_MATMUL_CDESC: u32 = 1102;
pub const CUDNN_ATTR_OPERATION_MATMUL_DESC: u32 = 1104;

pub const CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR: u32 = 1200;
pub const CUDNN_ATTR_OPERATION_POINTWISE_XDESC: u32 = 1201;
pub const CUDNN_ATTR_OPERATION_POINTWISE_BDESC: u32 = 1202;
pub const CUDNN_ATTR_OPERATION_POINTWISE_YDESC: u32 = 1203;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1: u32 = 1204;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2: u32 = 1205;

pub const CUDNN_ATTR_OPERATIONGRAPH_HANDLE: u32 = 1500;
pub const CUDNN_ATTR_OPERATIONGRAPH_OPS: u32 = 1501;

pub const CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT: u32 = 1600;
pub const CUDNN_ATTR_TENSOR_DATA_TYPE: u32 = 1601;
pub const CUDNN_ATTR_TENSOR_DIMENSIONS: u32 = 1602;
pub const CUDNN_ATTR_TENSOR_STRIDES: u32 = 1603;
pub const CUDNN_ATTR_TENSOR_UNIQUE_ID: u32 = 1606;
pub const CUDNN_ATTR_TENSOR_IS_VIRTUAL: u32 = 1607;

pub const CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS: u32 = 1700;
pub const CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS: u32 = 1701;
pub const CUDNN_ATTR_VARIANT_PACK_WORKSPACE: u32 = 1703;

pub const CUDNN_ATTR_REDUCTION_OPERATOR: u32 = 2200;
pub const CUDNN_ATTR_REDUCTION_COMP_TYPE: u32 = 2201;

pub const CUDNN_ATTR_OPERATION_REDUCTION_XDESC: u32 = 2300;
pub const CUDNN_ATTR_OPERATION_REDUCTION_YDESC: u32 = 2301;
pub const CUDNN_ATTR_OPERATION_REDUCTION_DESC: u32 = 2302;

// ── pointwise modes (`cudnnPointwiseMode_t`) ────────────────────────────────

pub const CUDNN_POINTWISE_ADD: i64 = 0;
pub const CUDNN_POINTWISE_MUL: i64 = 1;
pub const CUDNN_POINTWISE_SUB: i64 = 22;
pub const CUDNN_POINTWISE_DIV: i64 = 19;
pub const CUDNN_POINTWISE_EXP: i64 = 21;
pub const CUDNN_POINTWISE_IDENTITY: i64 = 44;
pub const CUDNN_POINTWISE_BINARY_SELECT: i64 = 48;
pub const CUDNN_POINTWISE_CMP_GE: i64 = 26;

// ── reduction operators (`cudnnReduceTensorOp_t`) ──────────────────────────

pub const CUDNN_REDUCE_TENSOR_ADD: i64 = 0;
pub const CUDNN_REDUCE_TENSOR_MAX: i64 = 3;

// ── heuristic modes (`cudnnBackendHeurMode_t`) ─────────────────────────────

pub const CUDNN_HEUR_MODE_INSTANT: i64 = 0;
pub const CUDNN_HEUR_MODE_B: i64 = 1;
pub const CUDNN_HEUR_MODE_FALLBACK: i64 = 2;
pub const CUDNN_HEUR_MODE_A: i64 = 3;

pub struct CudnnFns {
    pub cudnnCreate: unsafe extern "C" fn(*mut CudnnHandle) -> CudnnStatus,
    pub cudnnDestroy: unsafe extern "C" fn(CudnnHandle) -> CudnnStatus,
    pub cudnnSetStream: unsafe extern "C" fn(CudnnHandle, CUstream) -> CudnnStatus,
    pub cudnnGetStream: unsafe extern "C" fn(CudnnHandle, *mut CUstream) -> CudnnStatus,
    pub cudnnGetVersion: unsafe extern "C" fn() -> usize,
    pub cudnnGetCudartVersion: unsafe extern "C" fn() -> usize,

    // v9 graph backend (the path forward for MHA/conv)
    pub cudnnBackendCreateDescriptor:
        unsafe extern "C" fn(u32, *mut CudnnBackendDescriptor) -> CudnnStatus,
    pub cudnnBackendDestroyDescriptor: unsafe extern "C" fn(CudnnBackendDescriptor) -> CudnnStatus,
    pub cudnnBackendSetAttribute:
        unsafe extern "C" fn(CudnnBackendDescriptor, u32, u32, i64, *const c_void) -> CudnnStatus,
    pub cudnnBackendGetAttribute: unsafe extern "C" fn(
        CudnnBackendDescriptor,
        u32,
        u32,
        i64,
        *mut i64,
        *mut c_void,
    ) -> CudnnStatus,
    pub cudnnBackendFinalize: unsafe extern "C" fn(CudnnBackendDescriptor) -> CudnnStatus,
    pub cudnnBackendExecute: unsafe extern "C" fn(
        CudnnHandle,
        CudnnBackendDescriptor,
        CudnnBackendDescriptor,
    ) -> CudnnStatus,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcudnn.so.9",
        "libcudnn.so.8",
        "libcudnn.so",
        "cudnn64_9.dll",
        "cudnn64_8.dll",
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

#[inline]
pub fn fns() -> Result<&'static CudnnFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
