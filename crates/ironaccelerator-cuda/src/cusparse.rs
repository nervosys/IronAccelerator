//! cuSPARSE handle + per-device cache.

use crate::drv::{self, DeviceBuf, Repr, Stream};
use iron_cuda_sys::cublas_lt::{CublasOp, CudaDataType};
use iron_cuda_sys::cusparse as sys;
use iron_cuda_sys::driver::CUdeviceptr;
use ironaccelerator_core::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;

pub use iron_cuda_sys::cublas_lt::CublasOp as Op;
pub use sys::{
    CusparseIndexBase as IndexBase, CusparseIndexType as IndexType, CusparseOrder as Order,
    CusparseSDDMMAlg as SDDMMAlg, CusparseSpMMAlg as SpMMAlg, CusparseStatus,
};

fn fns() -> Result<&'static sys::CusparseFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cusparse not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: CusparseStatus) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub struct CusparseHandle {
    handle: sys::CusparseHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for CusparseHandle {}
unsafe impl Sync for CusparseHandle {}

impl CusparseHandle {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CusparseHandle::default();
        unsafe {
            check("cusparseCreate", (f.cusparseCreate)(&mut h))?;
        }
        Ok(Arc::new(Self {
            handle: h,
            _device: device,
        }))
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "cusparseSetStream",
                (fns()?.cusparseSetStream)(self.handle, stream.raw()),
            )
        }
    }

    pub fn version(&self) -> Result<i32> {
        let mut v: c_int = 0;
        unsafe {
            check(
                "cusparseGetVersion",
                (fns()?.cusparseGetVersion)(self.handle, &mut v),
            )?;
        }
        Ok(v as i32)
    }

    #[inline]
    pub fn raw(&self) -> sys::CusparseHandle {
        self.handle
    }
}

impl Drop for CusparseHandle {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cusparseDestroy)(self.handle);
            }
        }
    }
}

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<CusparseHandle>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn handle_for(stream: &Arc<Stream>) -> Result<Arc<CusparseHandle>> {
    let device = stream.device();
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) {
            h.set_stream(stream)?;
            return Ok(h.clone());
        }
    }
    let h = CusparseHandle::new(device.clone())?;
    h.set_stream(stream)?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

pub fn is_available() -> bool {
    sys::is_available()
}

// ─── Dtype inference for T: Repr ────────────────────────────────────────────

fn dtype_of<T: Repr>() -> Result<CudaDataType> {
    use std::any::TypeId;
    let t = TypeId::of::<T>();
    Ok(if t == TypeId::of::<f32>() {
        CudaDataType::R32F
    } else if t == TypeId::of::<f64>() {
        CudaDataType::R64F
    } else if t == TypeId::of::<i8>() {
        CudaDataType::R8I
    } else if t == TypeId::of::<u8>() {
        CudaDataType::R8U
    } else if t == TypeId::of::<i32>() {
        CudaDataType::R32I
    } else if t == TypeId::of::<u32>() {
        CudaDataType::R32U
    } else {
        return Err(Error::Other("cusparse: unsupported element type"));
    })
}

// ─── Dense matrix descriptor ────────────────────────────────────────────────

pub struct DnMat {
    raw: sys::CusparseDnMatDescr,
}

unsafe impl Send for DnMat {}
unsafe impl Sync for DnMat {}

impl DnMat {
    pub fn new<T: Repr>(
        rows: i64,
        cols: i64,
        ld: i64,
        ptr: CUdeviceptr,
        order: Order,
    ) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CusparseDnMatDescr::default();
        unsafe {
            check(
                "cusparseCreateDnMat",
                (f.cusparseCreateDnMat)(
                    &mut raw,
                    rows,
                    cols,
                    ld,
                    ptr as *mut c_void,
                    dtype_of::<T>()?,
                    order,
                ),
            )?;
        }
        Ok(Self { raw })
    }

    #[inline]
    pub fn raw(&self) -> sys::CusparseDnMatDescr {
        self.raw
    }
}

impl Drop for DnMat {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cusparseDestroyDnMat)(self.raw);
            }
        }
    }
}

// ─── Sparse CSR matrix descriptor ───────────────────────────────────────────

pub struct SpMatCsr {
    raw: sys::CusparseSpMatDescr,
}

unsafe impl Send for SpMatCsr {}
unsafe impl Sync for SpMatCsr {}

impl SpMatCsr {
    pub fn new<T: Repr>(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: CUdeviceptr,
        col_indices: CUdeviceptr,
        values: CUdeviceptr,
        idx_ty: IndexType,
        base: IndexBase,
    ) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CusparseSpMatDescr::default();
        unsafe {
            check(
                "cusparseCreateCsr",
                (f.cusparseCreateCsr)(
                    &mut raw,
                    rows,
                    cols,
                    nnz,
                    row_offsets as *mut c_void,
                    col_indices as *mut c_void,
                    values as *mut c_void,
                    idx_ty,
                    idx_ty,
                    base,
                    dtype_of::<T>()?,
                ),
            )?;
        }
        Ok(Self { raw })
    }

    #[inline]
    pub fn raw(&self) -> sys::CusparseSpMatDescr {
        self.raw
    }
}

impl Drop for SpMatCsr {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cusparseDestroySpMat)(self.raw);
            }
        }
    }
}

// ─── SpMM / SDDMM ───────────────────────────────────────────────────────────

/// Query workspace bytes for `C = alpha · op(A_sparse) · op(B) + beta · C`.
pub fn spmm_buffer_size<T: Repr>(
    h: &CusparseHandle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMatCsr,
    b: &DnMat,
    beta: &T,
    c: &DnMat,
    alg: SpMMAlg,
) -> Result<usize> {
    let f = fns()?;
    let mut bytes: usize = 0;
    let op_a: CublasOp = op_a;
    let op_b: CublasOp = op_b;
    unsafe {
        check(
            "cusparseSpMM_bufferSize",
            (f.cusparseSpMM_bufferSize)(
                h.raw(),
                op_a,
                op_b,
                alpha as *const T as *const c_void,
                a.raw,
                b.raw,
                beta as *const T as *const c_void,
                c.raw,
                dtype_of::<T>()?,
                alg,
                &mut bytes,
            ),
        )?;
    }
    Ok(bytes)
}

/// Execute SpMM using the caller-supplied workspace (see [`spmm_buffer_size`]).
pub fn spmm<T: Repr>(
    h: &CusparseHandle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMatCsr,
    b: &DnMat,
    beta: &T,
    c: &DnMat,
    alg: SpMMAlg,
    workspace: &mut DeviceBuf<u8>,
) -> Result<()> {
    let f = fns()?;
    let op_a: CublasOp = op_a;
    let op_b: CublasOp = op_b;
    unsafe {
        check(
            "cusparseSpMM",
            (f.cusparseSpMM)(
                h.raw(),
                op_a,
                op_b,
                alpha as *const T as *const c_void,
                a.raw,
                b.raw,
                beta as *const T as *const c_void,
                c.raw,
                dtype_of::<T>()?,
                alg,
                workspace.device_ptr() as *mut c_void,
            ),
        )
    }
}

pub fn sddmm_buffer_size<T: Repr>(
    h: &CusparseHandle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &DnMat,
    b: &DnMat,
    beta: &T,
    c: &SpMatCsr,
    alg: SDDMMAlg,
) -> Result<usize> {
    let f = fns()?;
    let mut bytes: usize = 0;
    let op_a: CublasOp = op_a;
    let op_b: CublasOp = op_b;
    unsafe {
        check(
            "cusparseSDDMM_bufferSize",
            (f.cusparseSDDMM_bufferSize)(
                h.raw(),
                op_a,
                op_b,
                alpha as *const T as *const c_void,
                a.raw,
                b.raw,
                beta as *const T as *const c_void,
                c.raw,
                dtype_of::<T>()?,
                alg,
                &mut bytes,
            ),
        )?;
    }
    Ok(bytes)
}

pub fn sddmm<T: Repr>(
    h: &CusparseHandle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &DnMat,
    b: &DnMat,
    beta: &T,
    c: &SpMatCsr,
    alg: SDDMMAlg,
    workspace: &mut DeviceBuf<u8>,
) -> Result<()> {
    let f = fns()?;
    let op_a: CublasOp = op_a;
    let op_b: CublasOp = op_b;
    unsafe {
        check(
            "cusparseSDDMM",
            (f.cusparseSDDMM)(
                h.raw(),
                op_a,
                op_b,
                alpha as *const T as *const c_void,
                a.raw,
                b.raw,
                beta as *const T as *const c_void,
                c.raw,
                dtype_of::<T>()?,
                alg,
                workspace.device_ptr() as *mut c_void,
            ),
        )
    }
}
