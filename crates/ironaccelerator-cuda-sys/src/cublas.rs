//! Legacy cuBLAS — used as a fallback path for BF16/F16/F32 GEMM when
//! cuBLASLt's heuristic returns nothing (tiny matrices, odd strides).

use crate::cublas_lt::{CublasOp, CublasStatus, CudaDataType, CublasComputeType};
use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
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

pub struct CublasFns {
    pub cublasCreate_v2: unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus,
    pub cublasDestroy_v2: unsafe extern "C" fn(CublasHandle) -> CublasStatus,
    pub cublasSetStream_v2: unsafe extern "C" fn(CublasHandle, CUstream) -> CublasStatus,
    pub cublasGetVersion_v2: unsafe extern "C" fn(CublasHandle, *mut c_int) -> CublasStatus,

    pub cublasSgemm_v2: unsafe extern "C" fn(
        CublasHandle, CublasOp, CublasOp, c_int, c_int, c_int,
        *const f32, *const f32, c_int, *const f32, c_int,
        *const f32, *mut f32, c_int,
    ) -> CublasStatus,

    pub cublasGemmEx: unsafe extern "C" fn(
        CublasHandle, CublasOp, CublasOp, c_int, c_int, c_int,
        *const c_void,
        *const c_void, CudaDataType, c_int,
        *const c_void, CudaDataType, c_int,
        *const c_void,
        *mut c_void, CudaDataType, c_int,
        CublasComputeType, CublasGemmAlgo,
    ) -> CublasStatus,
}

fn candidates() -> &'static [&'static str] {
    &[
        "libcublas.so.13", "libcublas.so.12", "libcublas.so",
        "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll",
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
            cublasSgemm_v2: sym(lib, "cublas", "cublasSgemm_v2")?,
            cublasGemmEx: sym(lib, "cublas", "cublasGemmEx")?,
        })
    }
});

pub fn fns() -> Result<&'static CublasFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
