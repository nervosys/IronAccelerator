//! hipBLASLt — the Lt front-end used for FP8 matmul on CDNA3+ (`gfx942`).
//!
//! API shape mirrors cuBLASLt: create a MatmulDesc + MatrixLayouts, query
//! heuristics for an algorithm, then Matmul. We bind the minimum we need.

use crate::hip::HipStream;
use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct HipblasLtHandle(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct HipblasLtMatmulDesc(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct HipblasLtMatrixLayout(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct HipblasLtMatmulPreference(pub *mut c_void);

unsafe impl Send for HipblasLtHandle {} unsafe impl Sync for HipblasLtHandle {}
unsafe impl Send for HipblasLtMatmulDesc {} unsafe impl Sync for HipblasLtMatmulDesc {}
unsafe impl Send for HipblasLtMatrixLayout {} unsafe impl Sync for HipblasLtMatrixLayout {}
unsafe impl Send for HipblasLtMatmulPreference {} unsafe impl Sync for HipblasLtMatmulPreference {}

pub use crate::hipblas::{HipblasOp, HipblasStatus, HipblasDataType};

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasComputeType {
    F32          = 68,
    F32FastF16   = 0,
    F32FastBF16  = 1,
    F32FastTF32  = 2,
    F16          = 64,
    F64          = 70,
    I32          = 72,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasLtEpilogue {
    Default = 1,
    Relu    = 2,
    Bias    = 4,
    ReluBias = 6,
    Gelu    = 32,
    GeluBias = 36,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasLtMatmulDescAttr {
    TransA        = 0,
    TransB        = 1,
    Epilogue      = 2,
    BiasPointer   = 3,
    ScaleA        = 4,
    ScaleB        = 5,
    ScaleC        = 6,
    ScaleD        = 7,
    AmaxD         = 8,
    ComputeType   = 9,
    ScaleType     = 10,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasLtMatrixLayoutAttr {
    Type          = 0,
    Order         = 1,
    Rows          = 2,
    Cols          = 3,
    Ld            = 4,
    BatchCount    = 5,
    StridedBatchOffset = 6,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum HipblasLtMatmulPreferenceAttr {
    SearchMode        = 0,
    MaxWorkspaceBytes = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipblasLtMatmulAlgo { pub data: [u64; 8] }

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct HipblasLtMatmulHeuristicResult {
    pub algo: HipblasLtMatmulAlgo,
    pub workspace_size: usize,
    pub state: u32,
    pub waves_count: f32,
    pub reserved: [i32; 4],
}

#[allow(clippy::type_complexity)]
pub struct HipblasLtFns {
    pub hipblasLtCreate: unsafe extern "C" fn(*mut HipblasLtHandle) -> HipblasStatus,
    pub hipblasLtDestroy: unsafe extern "C" fn(HipblasLtHandle) -> HipblasStatus,

    pub hipblasLtMatmulDescCreate: unsafe extern "C" fn(*mut HipblasLtMatmulDesc, HipblasComputeType, HipblasDataType) -> HipblasStatus,
    pub hipblasLtMatmulDescDestroy: unsafe extern "C" fn(HipblasLtMatmulDesc) -> HipblasStatus,
    pub hipblasLtMatmulDescSetAttribute: unsafe extern "C" fn(HipblasLtMatmulDesc, HipblasLtMatmulDescAttr, *const c_void, usize) -> HipblasStatus,

    pub hipblasLtMatrixLayoutCreate: unsafe extern "C" fn(*mut HipblasLtMatrixLayout, HipblasDataType, u64, u64, i64) -> HipblasStatus,
    pub hipblasLtMatrixLayoutDestroy: unsafe extern "C" fn(HipblasLtMatrixLayout) -> HipblasStatus,
    pub hipblasLtMatrixLayoutSetAttribute: unsafe extern "C" fn(HipblasLtMatrixLayout, HipblasLtMatrixLayoutAttr, *const c_void, usize) -> HipblasStatus,

    pub hipblasLtMatmulPreferenceCreate: unsafe extern "C" fn(*mut HipblasLtMatmulPreference) -> HipblasStatus,
    pub hipblasLtMatmulPreferenceDestroy: unsafe extern "C" fn(HipblasLtMatmulPreference) -> HipblasStatus,
    pub hipblasLtMatmulPreferenceSetAttribute: unsafe extern "C" fn(HipblasLtMatmulPreference, HipblasLtMatmulPreferenceAttr, *const c_void, usize) -> HipblasStatus,

    pub hipblasLtMatmulAlgoGetHeuristic: unsafe extern "C" fn(
        HipblasLtHandle, HipblasLtMatmulDesc,
        HipblasLtMatrixLayout, HipblasLtMatrixLayout,
        HipblasLtMatrixLayout, HipblasLtMatrixLayout,
        HipblasLtMatmulPreference,
        c_int, *mut HipblasLtMatmulHeuristicResult, *mut c_int,
    ) -> HipblasStatus,

    pub hipblasLtMatmul: unsafe extern "C" fn(
        HipblasLtHandle, HipblasLtMatmulDesc,
        *const c_void,
        *const c_void, HipblasLtMatrixLayout,
        *const c_void, HipblasLtMatrixLayout,
        *const c_void,
        *const c_void, HipblasLtMatrixLayout,
        *mut c_void,   HipblasLtMatrixLayout,
        *const HipblasLtMatmulAlgo,
        *mut c_void, usize,
        HipStream,
    ) -> HipblasStatus,
}

static LIB: LazyLock<LoaderResult<Library>> = LazyLock::new(|| {
    try_load(&["libhipblaslt.so", "libhipblaslt.so.0", "hipblaslt.dll"])
});
static FNS: OnceLock<LoaderResult<HipblasLtFns>> = OnceLock::new();

fn load_fns(lib: &Library) -> LoaderResult<HipblasLtFns> {
    macro_rules! g { ($s:ident) => { sym(lib, "hipblaslt", stringify!($s))? } }
    unsafe {
        Ok(HipblasLtFns {
            hipblasLtCreate: g!(hipblasLtCreate),
            hipblasLtDestroy: g!(hipblasLtDestroy),
            hipblasLtMatmulDescCreate: g!(hipblasLtMatmulDescCreate),
            hipblasLtMatmulDescDestroy: g!(hipblasLtMatmulDescDestroy),
            hipblasLtMatmulDescSetAttribute: g!(hipblasLtMatmulDescSetAttribute),
            hipblasLtMatrixLayoutCreate: g!(hipblasLtMatrixLayoutCreate),
            hipblasLtMatrixLayoutDestroy: g!(hipblasLtMatrixLayoutDestroy),
            hipblasLtMatrixLayoutSetAttribute: g!(hipblasLtMatrixLayoutSetAttribute),
            hipblasLtMatmulPreferenceCreate: g!(hipblasLtMatmulPreferenceCreate),
            hipblasLtMatmulPreferenceDestroy: g!(hipblasLtMatmulPreferenceDestroy),
            hipblasLtMatmulPreferenceSetAttribute: g!(hipblasLtMatmulPreferenceSetAttribute),
            hipblasLtMatmulAlgoGetHeuristic: g!(hipblasLtMatmulAlgoGetHeuristic),
            hipblasLtMatmul: g!(hipblasLtMatmul),
        })
    }
}

pub fn fns() -> Result<&'static HipblasLtFns, &'static LoadError> {
    FNS.get_or_init(|| {
        let lib = LIB.as_ref().map_err(clone_err)?;
        load_fns(lib)
    }).as_ref()
}

pub fn is_available() -> bool { fns().is_ok() }

fn clone_err(e: &LoadError) -> LoadError {
    match e {
        LoadError::LibraryNotFound { tried, last } =>
            LoadError::LibraryNotFound { tried: tried.clone(), last: last.clone() },
        LoadError::SymbolMissing { lib, symbol, err } =>
            LoadError::SymbolMissing { lib, symbol, err: err.clone() },
    }
}
