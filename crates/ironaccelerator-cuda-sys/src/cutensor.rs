//! cuTENSOR — N-dimensional tensor contractions, reductions, element-wise ops.
//!
//! Binds cuTENSOR v2 (CUDA 13.2 ships cutensor 2.x) descriptor-based API.

use crate::cublas_lt::CudaDataType;
use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::c_void;
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CutensorHandle(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CutensorTensorDescr(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CutensorOperationDescr(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CutensorPlan(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CutensorPlanPref(pub *mut c_void);

unsafe impl Send for CutensorHandle {}
unsafe impl Sync for CutensorHandle {}
unsafe impl Send for CutensorTensorDescr {}
unsafe impl Sync for CutensorTensorDescr {}
unsafe impl Send for CutensorOperationDescr {}
unsafe impl Sync for CutensorOperationDescr {}
unsafe impl Send for CutensorPlan {}
unsafe impl Sync for CutensorPlan {}
unsafe impl Send for CutensorPlanPref {}
unsafe impl Sync for CutensorPlanPref {}

ffi_code! {
    pub struct CutensorStatus {
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
        CublasError = 17,
        /// `CUTENSOR_STATUS_CUDA_ERROR`.
        CudartError = 18,
        /// `CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE`.
        InsufficientWorkspace = 19,
        InsufficientDriver = 20,
        IoError = 21,
    }
}

impl CutensorStatus {
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

/// Selects one of cuTENSOR 2's preset compute descriptors. The discriminants
/// are the cuTENSOR 1.x `cutensorComputeType_t` tags and are never passed to
/// the library: cuTENSOR 2 takes a `cutensorComputeDescriptor_t` *pointer*,
/// exported as an `extern const` global per preset. Call
/// [`resolve`](Self::resolve) to read it.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CutensorComputeDesc {
    Compute16F = 1,
    Compute16BF = 1024,
    Compute32F = 4,
    Compute64F = 16,
    ComputeTF32 = 4096,
}

/// `cutensorComputeDescriptor_t`: pointer to an opaque descriptor struct.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CutensorComputeDescriptor(pub *const c_void);

unsafe impl Send for CutensorComputeDescriptor {}
unsafe impl Sync for CutensorComputeDescriptor {}

impl CutensorComputeDesc {
    /// Name of the exported global holding this preset's descriptor.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Compute16F => "CUTENSOR_COMPUTE_DESC_16F",
            Self::Compute16BF => "CUTENSOR_COMPUTE_DESC_16BF",
            Self::Compute32F => "CUTENSOR_COMPUTE_DESC_32F",
            Self::Compute64F => "CUTENSOR_COMPUTE_DESC_64F",
            Self::ComputeTF32 => "CUTENSOR_COMPUTE_DESC_TF32",
        }
    }

    /// Read the descriptor pointer from the loaded cuTENSOR library.
    pub fn resolve(self) -> LoaderResult<CutensorComputeDescriptor> {
        FNS.as_ref().map_err(Clone::clone)?;
        let lib = LIB
            .get()
            .expect("cuTENSOR library is loaded once FNS is Ok");
        // A data symbol: `sym` yields the global's address, which holds the
        // descriptor pointer.
        let slot: *const CutensorComputeDescriptor =
            unsafe { sym(lib, "cutensor", self.symbol())? };
        Ok(unsafe { *slot })
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CutensorOperator {
    Identity = 1,
    Sqrt = 2,
    Relu = 8,
    Conj = 9,
    Rcp = 10,
    Add = 3,
    Mul = 5,
    Max = 6,
    Min = 7,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CutensorAlgo {
    Default = -1i32 as u32,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CutensorWorksizePref {
    /// `CUTENSOR_WORKSPACE_MIN`.
    MinWorkspace = 1,
    /// `CUTENSOR_WORKSPACE_DEFAULT`.
    Recommended = 2,
    /// `CUTENSOR_WORKSPACE_MAX`.
    Max = 3,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CutensorJitMode {
    None = 0,
    Default = 1,
}

pub struct CutensorFns {
    pub cutensorCreate: unsafe extern "C" fn(*mut CutensorHandle) -> CutensorStatus,
    pub cutensorDestroy: unsafe extern "C" fn(CutensorHandle) -> CutensorStatus,

    pub cutensorCreateTensorDescriptor: unsafe extern "C" fn(
        CutensorHandle,
        *mut CutensorTensorDescr,
        u32,
        *const i64,
        *const i64,
        CudaDataType,
        u32,
    ) -> CutensorStatus,
    pub cutensorDestroyTensorDescriptor:
        unsafe extern "C" fn(CutensorTensorDescr) -> CutensorStatus,

    pub cutensorCreateContraction: unsafe extern "C" fn(
        CutensorHandle,
        *mut CutensorOperationDescr,
        CutensorTensorDescr,
        *const i32,
        CutensorOperator,
        CutensorTensorDescr,
        *const i32,
        CutensorOperator,
        CutensorTensorDescr,
        *const i32,
        CutensorOperator,
        CutensorTensorDescr,
        *const i32,
        CutensorComputeDescriptor,
    ) -> CutensorStatus,
    pub cutensorDestroyOperationDescriptor:
        unsafe extern "C" fn(CutensorOperationDescr) -> CutensorStatus,

    pub cutensorCreatePlanPreference: unsafe extern "C" fn(
        CutensorHandle,
        *mut CutensorPlanPref,
        CutensorAlgo,
        CutensorJitMode,
    ) -> CutensorStatus,
    pub cutensorDestroyPlanPreference: unsafe extern "C" fn(CutensorPlanPref) -> CutensorStatus,

    pub cutensorEstimateWorkspaceSize: unsafe extern "C" fn(
        CutensorHandle,
        CutensorOperationDescr,
        CutensorPlanPref,
        CutensorWorksizePref,
        *mut u64,
    ) -> CutensorStatus,

    pub cutensorCreatePlan: unsafe extern "C" fn(
        CutensorHandle,
        *mut CutensorPlan,
        CutensorOperationDescr,
        CutensorPlanPref,
        u64,
    ) -> CutensorStatus,
    pub cutensorDestroyPlan: unsafe extern "C" fn(CutensorPlan) -> CutensorStatus,

    pub cutensorContract: unsafe extern "C" fn(
        CutensorHandle,
        CutensorPlan,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *mut c_void,
        u64,
        CUstream,
    ) -> CutensorStatus,
}

fn candidates() -> &'static [&'static str] {
    &["libcutensor.so.2", "libcutensor.so", "cutensor.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CutensorFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CutensorFns {
            cutensorCreate: sym(lib, "cutensor", "cutensorCreate")?,
            cutensorDestroy: sym(lib, "cutensor", "cutensorDestroy")?,
            cutensorCreateTensorDescriptor: sym(lib, "cutensor", "cutensorCreateTensorDescriptor")?,
            cutensorDestroyTensorDescriptor: sym(
                lib,
                "cutensor",
                "cutensorDestroyTensorDescriptor",
            )?,
            cutensorCreateContraction: sym(lib, "cutensor", "cutensorCreateContraction")?,
            cutensorDestroyOperationDescriptor: sym(
                lib,
                "cutensor",
                "cutensorDestroyOperationDescriptor",
            )?,
            cutensorCreatePlanPreference: sym(lib, "cutensor", "cutensorCreatePlanPreference")?,
            cutensorDestroyPlanPreference: sym(lib, "cutensor", "cutensorDestroyPlanPreference")?,
            cutensorEstimateWorkspaceSize: sym(lib, "cutensor", "cutensorEstimateWorkspaceSize")?,
            cutensorCreatePlan: sym(lib, "cutensor", "cutensorCreatePlan")?,
            cutensorDestroyPlan: sym(lib, "cutensor", "cutensorDestroyPlan")?,
            cutensorContract: sym(lib, "cutensor", "cutensorContract")?,
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static CutensorFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
