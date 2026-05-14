//! cuSOLVER Dense handle + per-device cache.

use crate::drv::{self, DeviceBuf, Stream};
use iron_cuda_sys::cublas_lt::CublasOp;
use iron_cuda_sys::cusolver as sys;
use ironaccelerator_core::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::os::raw::c_int;
use std::sync::Arc;

pub use iron_cuda_sys::cublas_lt::CublasOp as Op;
pub use sys::{CusolverEigMode as EigMode, CusolverFillMode as FillMode, CusolverStatus};

fn fns() -> Result<&'static sys::CusolverDnFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cusolver not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: CusolverStatus) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub struct CusolverDnHandle {
    handle: sys::CusolverDnHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for CusolverDnHandle {}
unsafe impl Sync for CusolverDnHandle {}

impl CusolverDnHandle {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CusolverDnHandle::default();
        unsafe {
            check("cusolverDnCreate", (f.cusolverDnCreate)(&mut h))?;
        }
        Ok(Arc::new(Self {
            handle: h,
            _device: device,
        }))
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "cusolverDnSetStream",
                (fns()?.cusolverDnSetStream)(self.handle, stream.raw()),
            )
        }
    }

    #[inline]
    pub fn raw(&self) -> sys::CusolverDnHandle {
        self.handle
    }

    // ─── LU (Sgetrf) ────────────────────────────────────────────────────────

    /// Workspace elements (f32) required for `getrf_f32` on `[m×n]`.
    pub fn getrf_f32_buffer_size(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnSgetrf_bufferSize",
                (f.cusolverDnSgetrf_bufferSize)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    /// LU factorization with partial pivoting, f32.
    /// `ipiv` must be at least `min(m,n)` ints; `info` is a single `c_int`.
    pub fn getrf_f32(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
        workspace: &mut DeviceBuf<f32>,
        ipiv: &mut DeviceBuf<i32>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnSgetrf",
                (f.cusolverDnSgetrf)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    workspace.device_ptr() as *mut f32,
                    ipiv.device_ptr() as *mut c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    /// Solve `op(A) · X = B` after `getrf_f32`. `a` holds the LU factors.
    pub fn getrs_f32(
        &self,
        op: Op,
        n: i32,
        nrhs: i32,
        a: &DeviceBuf<f32>,
        lda: i32,
        ipiv: &DeviceBuf<i32>,
        b: &mut DeviceBuf<f32>,
        ldb: i32,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let op: CublasOp = op;
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnSgetrs",
                (f.cusolverDnSgetrs)(
                    self.handle,
                    op,
                    n,
                    nrhs,
                    a.device_ptr() as *const f32,
                    lda,
                    ipiv.device_ptr() as *const c_int,
                    b.device_ptr() as *mut f32,
                    ldb,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    // ─── Cholesky (Spotrf) ──────────────────────────────────────────────────

    pub fn potrf_f32_buffer_size(
        &self,
        fill: FillMode,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnSpotrf_bufferSize",
                (f.cusolverDnSpotrf_bufferSize)(
                    self.handle,
                    fill,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    pub fn potrf_f32(
        &self,
        fill: FillMode,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
        workspace: &mut DeviceBuf<f32>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnSpotrf",
                (f.cusolverDnSpotrf)(
                    self.handle,
                    fill,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    workspace.device_ptr() as *mut f32,
                    workspace.len() as c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    // ─── QR (Sgeqrf) ────────────────────────────────────────────────────────

    pub fn geqrf_f32_buffer_size(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnSgeqrf_bufferSize",
                (f.cusolverDnSgeqrf_bufferSize)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    /// QR factorization. `tau` holds the Householder reflectors (length
    /// `min(m,n)`); `a` is overwritten in-place.
    pub fn geqrf_f32(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f32>,
        lda: i32,
        tau: &mut DeviceBuf<f32>,
        workspace: &mut DeviceBuf<f32>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnSgeqrf",
                (f.cusolverDnSgeqrf)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f32,
                    lda,
                    tau.device_ptr() as *mut f32,
                    workspace.device_ptr() as *mut f32,
                    workspace.len() as c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    // ─── f64 variants (D*) ──────────────────────────────────────────────────

    pub fn getrf_f64_buffer_size(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnDgetrf_bufferSize",
                (f.cusolverDnDgetrf_bufferSize)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    pub fn getrf_f64(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
        workspace: &mut DeviceBuf<f64>,
        ipiv: &mut DeviceBuf<i32>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnDgetrf",
                (f.cusolverDnDgetrf)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    workspace.device_ptr() as *mut f64,
                    ipiv.device_ptr() as *mut c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    pub fn getrs_f64(
        &self,
        op: Op,
        n: i32,
        nrhs: i32,
        a: &DeviceBuf<f64>,
        lda: i32,
        ipiv: &DeviceBuf<i32>,
        b: &mut DeviceBuf<f64>,
        ldb: i32,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let op: CublasOp = op;
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnDgetrs",
                (f.cusolverDnDgetrs)(
                    self.handle,
                    op,
                    n,
                    nrhs,
                    a.device_ptr() as *const f64,
                    lda,
                    ipiv.device_ptr() as *const c_int,
                    b.device_ptr() as *mut f64,
                    ldb,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    pub fn potrf_f64_buffer_size(
        &self,
        fill: FillMode,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnDpotrf_bufferSize",
                (f.cusolverDnDpotrf_bufferSize)(
                    self.handle,
                    fill,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    pub fn potrf_f64(
        &self,
        fill: FillMode,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
        workspace: &mut DeviceBuf<f64>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnDpotrf",
                (f.cusolverDnDpotrf)(
                    self.handle,
                    fill,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    workspace.device_ptr() as *mut f64,
                    workspace.len() as c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }

    pub fn geqrf_f64_buffer_size(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
    ) -> Result<i32> {
        let f = fns()?;
        let mut lwork: c_int = 0;
        unsafe {
            check(
                "cusolverDnDgeqrf_bufferSize",
                (f.cusolverDnDgeqrf_bufferSize)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    &mut lwork,
                ),
            )?;
        }
        Ok(lwork as i32)
    }

    pub fn geqrf_f64(
        &self,
        m: i32,
        n: i32,
        a: &mut DeviceBuf<f64>,
        lda: i32,
        tau: &mut DeviceBuf<f64>,
        workspace: &mut DeviceBuf<f64>,
        info: &mut DeviceBuf<i32>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cusolverDnDgeqrf",
                (f.cusolverDnDgeqrf)(
                    self.handle,
                    m,
                    n,
                    a.device_ptr() as *mut f64,
                    lda,
                    tau.device_ptr() as *mut f64,
                    workspace.device_ptr() as *mut f64,
                    workspace.len() as c_int,
                    info.device_ptr() as *mut c_int,
                ),
            )
        }
    }
}

impl Drop for CusolverDnHandle {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cusolverDnDestroy)(self.handle);
            }
        }
    }
}

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<CusolverDnHandle>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn handle_for(stream: &Arc<Stream>) -> Result<Arc<CusolverDnHandle>> {
    let device = stream.device();
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) {
            h.set_stream(stream)?;
            return Ok(h.clone());
        }
    }
    let h = CusolverDnHandle::new(device.clone())?;
    h.set_stream(stream)?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

pub fn is_available() -> bool {
    sys::is_available()
}
