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

/// `cudnnStatus_t`, cuDNN 9 numbering (hierarchical: 1xxx library state,
/// 2xxx bad parameter, 3xxx not supported, 4xxx internal, 5xxx execution).
/// Every code the cuDNN 9 header declares is a variant, because the FFI
/// signatures return this type directly and an undeclared value would be
/// undefined behaviour.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnStatus {
    /// `CUDNN_STATUS_SUCCESS`.
    Success = 0,
    /// `CUDNN_STATUS_NOT_INITIALIZED`.
    NotInitialized = 1001,
    /// `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
    VersionMismatch = 1002,
    /// `CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH`.
    SerializationVersionMismatch = 1003,
    /// `CUDNN_STATUS_DEPRECATED`.
    Deprecated = 1004,
    /// `CUDNN_STATUS_LICENSE_ERROR`.
    LicenseError = 1005,
    /// `CUDNN_STATUS_RUNTIME_IN_PROGRESS`.
    RuntimeInProgress = 1006,
    /// `CUDNN_STATUS_RUNTIME_FP_OVERFLOW`.
    RuntimeFpOverflow = 1007,
    /// `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`.
    SublibraryLoadingFailed = 1008,
    /// `CUDNN_STATUS_BAD_PARAM`.
    BadParam = 2000,
    /// `CUDNN_STATUS_BAD_PARAM_NULL_POINTER`.
    BadParamNullPointer = 2002,
    /// `CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER`.
    BadParamMisalignedPointer = 2003,
    /// `CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED`.
    BadParamNotFinalized = 2004,
    /// `CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND`.
    BadParamOutOfBound = 2005,
    /// `CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT`.
    BadParamSizeInsufficient = 2006,
    /// `CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH`.
    BadParamStreamMismatch = 2007,
    /// `CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH`.
    BadParamShapeMismatch = 2008,
    /// `CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES`.
    BadParamDuplicatedEntries = 2009,
    /// `CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE`.
    BadParamAttributeType = 2010,
    /// `CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH`.
    BadParamCudaGraphMismatch = 2011,
    /// `CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE`.
    BadParamDescriptorType = 2012,
    /// `CUDNN_STATUS_NOT_SUPPORTED`.
    NotSupported = 3000,
    /// `CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN`.
    NotSupportedGraphPattern = 3001,
    /// `CUDNN_STATUS_NOT_SUPPORTED_SHAPE`.
    NotSupportedShape = 3002,
    /// `CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE`.
    NotSupportedDataType = 3003,
    /// `CUDNN_STATUS_NOT_SUPPORTED_LAYOUT`.
    NotSupportedLayout = 3004,
    /// `CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER`.
    NotSupportedIncompatibleCudaDriver = 3005,
    /// `CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART`.
    NotSupportedIncompatibleCudart = 3006,
    /// `CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH`.
    ArchMismatch = 3007,
    /// `CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING`.
    RuntimePrerequisiteMissing = 3008,
    /// `CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE`.
    NotSupportedSublibraryUnavailable = 3009,
    /// `CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT`.
    NotSupportedSharedMemoryInsufficient = 3010,
    /// `CUDNN_STATUS_NOT_SUPPORTED_PADDING`.
    NotSupportedPadding = 3011,
    /// `CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM`.
    NotSupportedBadLaunchParam = 3012,
    /// `CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API`.
    NotSupportedCudaGraphNativeApi = 3013,
    /// `CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE`.
    NotSupportedInvalidDynamicShape = 3014,
    /// `CUDNN_STATUS_INTERNAL_ERROR`.
    InternalError = 4000,
    /// `CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED`.
    InternalErrorCompilationFailed = 4001,
    /// `CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE`.
    InternalErrorUnexpectedValue = 4002,
    /// `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED`.
    InternalErrorHostAllocationFailed = 4003,
    /// `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`.
    InternalErrorDeviceAllocationFailed = 4004,
    /// `CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM`.
    InternalErrorBadLaunchParam = 4005,
    /// `CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED`.
    InternalErrorTextureCreationFailed = 4006,
    /// `CUDNN_STATUS_EXECUTION_FAILED`.
    ExecutionFailed = 5000,
    /// `CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER`.
    ExecutionFailedCudaDriver = 5001,
    /// `CUDNN_STATUS_EXECUTION_FAILED_CUBLAS`.
    ExecutionFailedCublas = 5002,
    /// `CUDNN_STATUS_EXECUTION_FAILED_CUDART`.
    ExecutionFailedCudart = 5003,
    /// `CUDNN_STATUS_EXECUTION_FAILED_CURAND`.
    ExecutionFailedCurand = 5004,
    /// cuDNN 8 `CUDNN_STATUS_ALLOC_FAILED`; cuDNN 9 never returns it.
    #[deprecated(
        note = "cuDNN 8 status; cuDNN 9 reports allocation failures as InternalErrorHostAllocationFailed / InternalErrorDeviceAllocationFailed"
    )]
    AllocFailed = 2,
    /// cuDNN 8 `CUDNN_STATUS_INVALID_VALUE`; cuDNN 9 never returns it.
    #[deprecated(note = "cuDNN 8 status; cuDNN 9 has no equivalent")]
    InvalidValue = 5,
    /// cuDNN 8 `CUDNN_STATUS_MAPPING_ERROR`; cuDNN 9 never returns it.
    #[deprecated(note = "cuDNN 8 status; cuDNN 9 has no equivalent")]
    MappingError = 7,
    /// Any code not modelled above (see [`CudnnStatus::from_raw`]).
    Other = 0xFFFF_FFFF,
}

impl CudnnStatus {
    pub fn from_raw(r: u32) -> Self {
        match r {
            0 => Self::Success,
            1001 => Self::NotInitialized,
            1002 => Self::VersionMismatch,
            1003 => Self::SerializationVersionMismatch,
            1004 => Self::Deprecated,
            1005 => Self::LicenseError,
            1006 => Self::RuntimeInProgress,
            1007 => Self::RuntimeFpOverflow,
            1008 => Self::SublibraryLoadingFailed,
            2000 => Self::BadParam,
            2002 => Self::BadParamNullPointer,
            2003 => Self::BadParamMisalignedPointer,
            2004 => Self::BadParamNotFinalized,
            2005 => Self::BadParamOutOfBound,
            2006 => Self::BadParamSizeInsufficient,
            2007 => Self::BadParamStreamMismatch,
            2008 => Self::BadParamShapeMismatch,
            2009 => Self::BadParamDuplicatedEntries,
            2010 => Self::BadParamAttributeType,
            2011 => Self::BadParamCudaGraphMismatch,
            2012 => Self::BadParamDescriptorType,
            3000 => Self::NotSupported,
            3001 => Self::NotSupportedGraphPattern,
            3002 => Self::NotSupportedShape,
            3003 => Self::NotSupportedDataType,
            3004 => Self::NotSupportedLayout,
            3005 => Self::NotSupportedIncompatibleCudaDriver,
            3006 => Self::NotSupportedIncompatibleCudart,
            3007 => Self::ArchMismatch,
            3008 => Self::RuntimePrerequisiteMissing,
            3009 => Self::NotSupportedSublibraryUnavailable,
            3010 => Self::NotSupportedSharedMemoryInsufficient,
            3011 => Self::NotSupportedPadding,
            3012 => Self::NotSupportedBadLaunchParam,
            3013 => Self::NotSupportedCudaGraphNativeApi,
            3014 => Self::NotSupportedInvalidDynamicShape,
            4000 => Self::InternalError,
            4001 => Self::InternalErrorCompilationFailed,
            4002 => Self::InternalErrorUnexpectedValue,
            4003 => Self::InternalErrorHostAllocationFailed,
            4004 => Self::InternalErrorDeviceAllocationFailed,
            4005 => Self::InternalErrorBadLaunchParam,
            4006 => Self::InternalErrorTextureCreationFailed,
            5000 => Self::ExecutionFailed,
            5001 => Self::ExecutionFailedCudaDriver,
            5002 => Self::ExecutionFailedCublas,
            5003 => Self::ExecutionFailedCudart,
            5004 => Self::ExecutionFailedCurand,
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
pub const CUDNN_ATTR_POINTWISE_AXIS: u32 = 9;

pub const CUDNN_ATTR_ENGINEHEUR_MODE: u32 = 200;
pub const CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH: u32 = 201;
pub const CUDNN_ATTR_ENGINEHEUR_RESULTS: u32 = 202;

pub const CUDNN_ATTR_ENGINECFG_ENGINE: u32 = 300;

pub const CUDNN_ATTR_EXECUTION_PLAN_HANDLE: u32 = 400;
pub const CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: u32 = 401;
pub const CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: u32 = 402;

pub const CUDNN_ATTR_ENGINE_OPERATION_GRAPH: u32 = 1300;
pub const CUDNN_ATTR_ENGINE_GLOBAL_INDEX: u32 = 1301;

pub const CUDNN_ATTR_MATMUL_COMP_TYPE: u32 = 1500;

pub const CUDNN_ATTR_OPERATION_MATMUL_ADESC: u32 = 1520;
pub const CUDNN_ATTR_OPERATION_MATMUL_BDESC: u32 = 1521;
pub const CUDNN_ATTR_OPERATION_MATMUL_CDESC: u32 = 1522;
pub const CUDNN_ATTR_OPERATION_MATMUL_DESC: u32 = 1523;

pub const CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR: u32 = 750;
pub const CUDNN_ATTR_OPERATION_POINTWISE_XDESC: u32 = 751;
pub const CUDNN_ATTR_OPERATION_POINTWISE_BDESC: u32 = 752;
pub const CUDNN_ATTR_OPERATION_POINTWISE_YDESC: u32 = 753;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1: u32 = 754;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2: u32 = 755;

pub const CUDNN_ATTR_OPERATIONGRAPH_HANDLE: u32 = 800;
pub const CUDNN_ATTR_OPERATIONGRAPH_OPS: u32 = 801;

pub const CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT: u32 = 900;
pub const CUDNN_ATTR_TENSOR_DATA_TYPE: u32 = 901;
pub const CUDNN_ATTR_TENSOR_DIMENSIONS: u32 = 902;
pub const CUDNN_ATTR_TENSOR_STRIDES: u32 = 903;
pub const CUDNN_ATTR_TENSOR_UNIQUE_ID: u32 = 906;
pub const CUDNN_ATTR_TENSOR_IS_VIRTUAL: u32 = 907;

pub const CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS: u32 = 1000;
pub const CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS: u32 = 1001;
pub const CUDNN_ATTR_VARIANT_PACK_WORKSPACE: u32 = 1003;

pub const CUDNN_ATTR_REDUCTION_OPERATOR: u32 = 1600;
pub const CUDNN_ATTR_REDUCTION_COMP_TYPE: u32 = 1601;

pub const CUDNN_ATTR_OPERATION_REDUCTION_XDESC: u32 = 1610;
pub const CUDNN_ATTR_OPERATION_REDUCTION_YDESC: u32 = 1611;
pub const CUDNN_ATTR_OPERATION_REDUCTION_DESC: u32 = 1612;

// ── pointwise modes (`cudnnPointwiseMode_t`) ────────────────────────────────

pub const CUDNN_POINTWISE_ADD: i64 = 0;
pub const CUDNN_POINTWISE_MUL: i64 = 1;
pub const CUDNN_POINTWISE_SUB: i64 = 9;
pub const CUDNN_POINTWISE_DIV: i64 = 6;
pub const CUDNN_POINTWISE_EXP: i64 = 13;
pub const CUDNN_POINTWISE_IDENTITY: i64 = 21;
pub const CUDNN_POINTWISE_BINARY_SELECT: i64 = 601;
pub const CUDNN_POINTWISE_CMP_GE: i64 = 303;

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
