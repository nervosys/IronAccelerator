//! cuSPARSE — sparse BLAS (SpMM, SDDMM, SpMV).
//!
//! Binds the generic `cusparseSpMM` / `cusparseSDDMM` entry points along
//! with descriptor create/destroy for dense and sparse matrices.

use crate::cublas_lt::{CudaDataType, CublasOp};
use crate::driver::{CUdeviceptr, CUstream};
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CusparseHandle(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CusparseDnMatDescr(pub *mut c_void);
#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct CusparseSpMatDescr(pub *mut c_void);

unsafe impl Send for CusparseHandle {} unsafe impl Sync for CusparseHandle {}
unsafe impl Send for CusparseDnMatDescr {} unsafe impl Sync for CusparseDnMatDescr {}
unsafe impl Send for CusparseSpMatDescr {} unsafe impl Sync for CusparseSpMatDescr {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CusparseStatus {
    Success = 0, NotInitialized = 1, AllocFailed = 2, InvalidValue = 3,
    ArchMismatch = 4, MappingError = 5, ExecutionFailed = 6, InternalError = 7,
    MatrixTypeNotSupported = 8, ZeroPivot = 9, NotSupported = 10,
    InsufficientResources = 11, Other = 0xFFFF_FFFF,
}

impl CusparseStatus {
    pub fn from_raw(r: u32) -> Self {
        if r <= 11 { unsafe { std::mem::transmute(r) } } else { Self::Other }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusparseOrder { Row = 0, Col = 1 }
#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusparseIndexType { U16 = 1, I32 = 2, I64 = 3 }
#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusparseIndexBase { Zero = 0, One = 1 }
#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusparseSpMMAlg {
    Default = 0, CsrAlg1 = 4, CsrAlg2 = 6, CsrAlg3 = 12,
    CooAlg1 = 1, CooAlg2 = 2, CooAlg3 = 3, CooAlg4 = 5,
}
#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum CusparseSDDMMAlg { Default = 0 }

pub struct CusparseFns {
    pub cusparseCreate: unsafe extern "C" fn(*mut CusparseHandle) -> CusparseStatus,
    pub cusparseDestroy: unsafe extern "C" fn(CusparseHandle) -> CusparseStatus,
    pub cusparseSetStream: unsafe extern "C" fn(CusparseHandle, CUstream) -> CusparseStatus,
    pub cusparseGetVersion: unsafe extern "C" fn(CusparseHandle, *mut c_int) -> CusparseStatus,

    pub cusparseCreateDnMat: unsafe extern "C" fn(
        *mut CusparseDnMatDescr, i64, i64, i64, *mut c_void, CudaDataType, CusparseOrder,
    ) -> CusparseStatus,
    pub cusparseDestroyDnMat: unsafe extern "C" fn(CusparseDnMatDescr) -> CusparseStatus,

    pub cusparseCreateCsr: unsafe extern "C" fn(
        *mut CusparseSpMatDescr, i64, i64, i64,
        *mut c_void, *mut c_void, *mut c_void,
        CusparseIndexType, CusparseIndexType, CusparseIndexBase, CudaDataType,
    ) -> CusparseStatus,
    pub cusparseDestroySpMat: unsafe extern "C" fn(CusparseSpMatDescr) -> CusparseStatus,

    pub cusparseSpMM_bufferSize: unsafe extern "C" fn(
        CusparseHandle, CublasOp, CublasOp,
        *const c_void, CusparseSpMatDescr, CusparseDnMatDescr,
        *const c_void, CusparseDnMatDescr,
        CudaDataType, CusparseSpMMAlg, *mut usize,
    ) -> CusparseStatus,
    pub cusparseSpMM: unsafe extern "C" fn(
        CusparseHandle, CublasOp, CublasOp,
        *const c_void, CusparseSpMatDescr, CusparseDnMatDescr,
        *const c_void, CusparseDnMatDescr,
        CudaDataType, CusparseSpMMAlg, *mut c_void,
    ) -> CusparseStatus,

    pub cusparseSDDMM_bufferSize: unsafe extern "C" fn(
        CusparseHandle, CublasOp, CublasOp,
        *const c_void, CusparseDnMatDescr, CusparseDnMatDescr,
        *const c_void, CusparseSpMatDescr,
        CudaDataType, CusparseSDDMMAlg, *mut usize,
    ) -> CusparseStatus,
    pub cusparseSDDMM: unsafe extern "C" fn(
        CusparseHandle, CublasOp, CublasOp,
        *const c_void, CusparseDnMatDescr, CusparseDnMatDescr,
        *const c_void, CusparseSpMatDescr,
        CudaDataType, CusparseSDDMMAlg, *mut c_void,
    ) -> CusparseStatus,
}

fn candidates() -> &'static [&'static str] {
    &["libcusparse.so.12", "libcusparse.so.11", "libcusparse.so",
      "cusparse64_12.dll", "cusparse64_11.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<CusparseFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(CusparseFns {
            cusparseCreate: sym(lib, "cusparse", "cusparseCreate")?,
            cusparseDestroy: sym(lib, "cusparse", "cusparseDestroy")?,
            cusparseSetStream: sym(lib, "cusparse", "cusparseSetStream")?,
            cusparseGetVersion: sym(lib, "cusparse", "cusparseGetVersion")?,
            cusparseCreateDnMat: sym(lib, "cusparse", "cusparseCreateDnMat")?,
            cusparseDestroyDnMat: sym(lib, "cusparse", "cusparseDestroyDnMat")?,
            cusparseCreateCsr: sym(lib, "cusparse", "cusparseCreateCsr")?,
            cusparseDestroySpMat: sym(lib, "cusparse", "cusparseDestroySpMat")?,
            cusparseSpMM_bufferSize: sym(lib, "cusparse", "cusparseSpMM_bufferSize")?,
            cusparseSpMM: sym(lib, "cusparse", "cusparseSpMM")?,
            cusparseSDDMM_bufferSize: sym(lib, "cusparse", "cusparseSDDMM_bufferSize")?,
            cusparseSDDMM: sym(lib, "cusparse", "cusparseSDDMM")?,
        })
    }
});

pub fn fns() -> Result<&'static CusparseFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
