//! Legacy cuBLAS — used as a fallback path for BF16/F16/F32 GEMM when
//! cuBLASLt's heuristic returns nothing (tiny matrices, odd strides).

use crate::cublas_lt::{CublasComputeType, CublasOp, CublasStatus, CudaDataType};
use crate::driver::CUstream;
use crate::loader::{sym, sym_opt, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CublasHandle(pub *mut c_void);
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasGemmAlgo {
    Default = -1i32 as u32,
    DefaultTensorOp = 99,
}

impl CublasGemmAlgo {
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_GEMM_DEFAULT: Self = Self::Default;
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: Self = Self::DefaultTensorOp;
}

impl CublasMathMode {
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_DEFAULT_MATH: Self = Self::Default;
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_TENSOR_OP_MATH: Self = Self::TensorOpMath;
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_PEDANTIC_MATH: Self = Self::PedanticMath;
    #[allow(non_upper_case_globals)]
    pub const CUBLAS_TF32_TENSOR_OP_MATH: Self = Self::Tf32TensorOpMath;
}

/// `cublasMath_t` — selects whether GEMM uses tensor cores.
/// Required to enable tensor-core HGEMM (FP16 prefill hot path).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasMathMode {
    Default = 0,
    TensorOpMath = 1,
    PedanticMath = 2,
    Tf32TensorOpMath = 3,
    DisallowReducedPrecisionReduction = 16,
}

pub struct CublasFns {
    pub cublasCreate_v2: unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus,
    pub cublasDestroy_v2: unsafe extern "C" fn(CublasHandle) -> CublasStatus,
    pub cublasSetStream_v2: unsafe extern "C" fn(CublasHandle, CUstream) -> CublasStatus,
    pub cublasGetVersion_v2: unsafe extern "C" fn(CublasHandle, *mut c_int) -> CublasStatus,

    /// Enable tensor-core math mode. **Required for tensor-core HGEMM**
    /// (FP16 prefill hot path); without this, GEMM falls back to CUDA cores.
    pub cublasSetMathMode: unsafe extern "C" fn(CublasHandle, CublasMathMode) -> CublasStatus,

    /// Bind a caller-managed workspace to the cuBLAS handle. Lets us avoid
    /// per-call internal allocations (cuBLAS picks better algorithms when
    /// it has a known workspace budget).
    pub cublasSetWorkspace_v2:
        unsafe extern "C" fn(CublasHandle, *mut c_void, usize) -> CublasStatus,

    pub cublasSgemm_v2: unsafe extern "C" fn(
        CublasHandle,
        CublasOp,
        CublasOp,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        c_int,
        *const f32,
        c_int,
        *const f32,
        *mut f32,
        c_int,
    ) -> CublasStatus,

    pub cublasGemmEx: unsafe extern "C" fn(
        CublasHandle,
        CublasOp,
        CublasOp,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        CudaDataType,
        c_int,
        *const c_void,
        CudaDataType,
        c_int,
        *const c_void,
        *mut c_void,
        CudaDataType,
        c_int,
        CublasComputeType,
        CublasGemmAlgo,
    ) -> CublasStatus,

    /// `cublasGemmGroupedBatchedEx` (CUDA 12.4+). Fused dispatch of N GEMMs
    /// with potentially different (M, N, K). Loaded optionally — older cuBLAS
    /// versions don't export this symbol.
    ///
    /// Parameter arrays are indexed by group; `group_size[g]` gives the number
    /// of matrices that share the parameters of group `g`.
    pub cublasGemmGroupedBatchedEx: Option<
        unsafe extern "C" fn(
            CublasHandle,
            *const CublasOp,      // transa_array[group_count]
            *const CublasOp,      // transb_array[group_count]
            *const c_int,         // m_array[group_count]
            *const c_int,         // n_array[group_count]
            *const c_int,         // k_array[group_count]
            *const *const c_void, // alpha_array[group_count]
            *const *const c_void, // Aarray[sum(group_size)]
            CudaDataType,
            *const c_int,         // lda_array[group_count]
            *const *const c_void, // Barray[sum(group_size)]
            CudaDataType,
            *const c_int,         // ldb_array[group_count]
            *const *const c_void, // beta_array[group_count]
            *const *mut c_void,   // Carray[sum(group_size)]
            CudaDataType,
            *const c_int, // ldc_array[group_count]
            c_int,        // group_count
            *const c_int, // group_size[group_count]
            CublasComputeType,
        ) -> CublasStatus,
    >,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so",
        "cublas64_13.dll",
        "cublas64_12.dll",
        "cublas64_11.dll",
    ]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CublasFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CublasFns {
            cublasCreate_v2: sym(lib, "cublas", "cublasCreate_v2")?,
            cublasDestroy_v2: sym(lib, "cublas", "cublasDestroy_v2")?,
            cublasSetStream_v2: sym(lib, "cublas", "cublasSetStream_v2")?,
            cublasGetVersion_v2: sym(lib, "cublas", "cublasGetVersion_v2")?,
            cublasSetMathMode: sym(lib, "cublas", "cublasSetMathMode")?,
            cublasSetWorkspace_v2: sym(lib, "cublas", "cublasSetWorkspace_v2")?,
            cublasSgemm_v2: sym(lib, "cublas", "cublasSgemm_v2")?,
            cublasGemmEx: sym(lib, "cublas", "cublasGemmEx")?,
            cublasGemmGroupedBatchedEx: sym_opt(lib, "cublasGemmGroupedBatchedEx"),
        })
    }
});

#[inline]
pub fn fns() -> Result<&'static CublasFns, &'static LoadError> {
    FNS.as_ref()
}
pub fn is_available() -> bool {
    FNS.is_ok()
}
