//! cuTENSOR handle + per-device cache.

use crate::drv::{self, DeviceBuf, Repr, Stream};
use iron_cuda_sys::cublas_lt::CudaDataType;
use iron_cuda_sys::cutensor as sys;
use ironaccelerator_core::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

pub use iron_cuda_sys::cublas_lt::CudaDataType as DType;
pub use sys::{
    CutensorComputeDesc as ComputeDesc, CutensorOperator as Operator,
    CutensorAlgo as Algo, CutensorWorksizePref as WorksizePref,
    CutensorJitMode as JitMode, CutensorStatus,
};

fn fns() -> Result<&'static sys::CutensorFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(
        format!("cutensor not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: CutensorStatus) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub struct CutensorHandle {
    handle: sys::CutensorHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for CutensorHandle {}
unsafe impl Sync for CutensorHandle {}

impl CutensorHandle {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CutensorHandle::default();
        unsafe { check("cutensorCreate", (f.cutensorCreate)(&mut h))?; }
        Ok(Arc::new(Self { handle: h, _device: device }))
    }

    #[inline] pub fn raw(&self) -> sys::CutensorHandle { self.handle }
}

impl Drop for CutensorHandle {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.cutensorDestroy)(self.handle); }
        }
    }
}

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<CutensorHandle>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn handle_for(device: Arc<drv::Device>) -> Result<Arc<CutensorHandle>> {
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) { return Ok(h.clone()); }
    }
    let h = CutensorHandle::new(device)?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

pub fn is_available() -> bool { sys::is_available() }

fn dtype_of<T: Repr>() -> Result<CudaDataType> {
    use std::any::TypeId;
    let t = TypeId::of::<T>();
    Ok(if t == TypeId::of::<f32>() { CudaDataType::R32F }
       else if t == TypeId::of::<f64>() { CudaDataType::R64F }
       else { return Err(Error::Other("cutensor: unsupported element type (use f32/f64)")); })
}

// ─── Tensor descriptor ─────────────────────────────────────────────────────

pub struct TensorDescr {
    raw: sys::CutensorTensorDescr,
    handle: Arc<CutensorHandle>,
}

unsafe impl Send for TensorDescr {} unsafe impl Sync for TensorDescr {}

impl TensorDescr {
    /// Create a descriptor for a tensor with `modes.len()` dimensions. `strides`
    /// may be empty for a packed layout (cuTENSOR derives them).
    pub fn new<T: Repr>(
        handle: Arc<CutensorHandle>, extents: &[i64], strides: &[i64], alignment: u32,
    ) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CutensorTensorDescr::default();
        let strides_ptr = if strides.is_empty() { std::ptr::null() } else { strides.as_ptr() };
        unsafe {
            check("cutensorCreateTensorDescriptor",
                  (f.cutensorCreateTensorDescriptor)(
                      handle.raw(), &mut raw,
                      extents.len() as u32,
                      extents.as_ptr(), strides_ptr,
                      dtype_of::<T>()?, alignment))?;
        }
        Ok(Self { raw, handle })
    }

    #[inline] pub fn raw(&self) -> sys::CutensorTensorDescr { self.raw }
}

impl Drop for TensorDescr {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.cutensorDestroyTensorDescriptor)(self.raw); }
        }
    }
}

// ─── Contraction plan ───────────────────────────────────────────────────────

/// Einsum-style contraction: `D = alpha · op(A) · op(B) + beta · C`, with
/// mode labels picking out which axes contract.
pub struct Contract {
    op_descr: sys::CutensorOperationDescr,
    plan: sys::CutensorPlan,
    pref: sys::CutensorPlanPref,
    workspace_bytes: u64,
    handle: Arc<CutensorHandle>,
}

unsafe impl Send for Contract {} unsafe impl Sync for Contract {}

impl Contract {
    pub fn build(
        handle: Arc<CutensorHandle>,
        a: &TensorDescr, modes_a: &[i32],
        b: &TensorDescr, modes_b: &[i32],
        c: &TensorDescr, modes_c: &[i32],
        d: &TensorDescr, modes_d: &[i32],
        compute: ComputeDesc,
        ws_pref: WorksizePref,
        jit: JitMode,
    ) -> Result<Self> {
        let f = fns()?;

        let mut op_descr = sys::CutensorOperationDescr::default();
        unsafe {
            check("cutensorCreateContraction", (f.cutensorCreateContraction)(
                handle.raw(), &mut op_descr,
                a.raw, modes_a.as_ptr(), Operator::Identity,
                b.raw, modes_b.as_ptr(), Operator::Identity,
                c.raw, modes_c.as_ptr(), Operator::Identity,
                d.raw, modes_d.as_ptr(),
                compute,
            ))?;
        }

        let mut pref = sys::CutensorPlanPref::default();
        unsafe {
            check("cutensorCreatePlanPreference",
                  (f.cutensorCreatePlanPreference)(
                      handle.raw(), &mut pref, Algo::Default, jit))?;
        }

        let mut bytes: u64 = 0;
        unsafe {
            check("cutensorEstimateWorkspaceSize",
                  (f.cutensorEstimateWorkspaceSize)(
                      handle.raw(), op_descr, pref, ws_pref, &mut bytes))?;
        }

        let mut plan = sys::CutensorPlan::default();
        unsafe {
            check("cutensorCreatePlan",
                  (f.cutensorCreatePlan)(handle.raw(), &mut plan, op_descr, pref, bytes))?;
        }

        Ok(Self { op_descr, plan, pref, workspace_bytes: bytes, handle })
    }

    #[inline] pub fn workspace_bytes(&self) -> u64 { self.workspace_bytes }

    /// Execute the contraction on `stream`.
    ///
    /// # Safety
    /// Device pointers and `workspace` must match the descriptors handed to
    /// [`Self::build`].
    pub unsafe fn run<T: Repr>(
        &self,
        alpha: &T, a_ptr: *const c_void, b_ptr: *const c_void,
        beta: &T,  c_ptr: *const c_void, d_ptr: *mut c_void,
        workspace: Option<&mut DeviceBuf<u8>>,
        stream: &Stream,
    ) -> Result<()> {
        let f = fns()?;
        let (ws_ptr, ws_bytes) = match workspace {
            Some(w) => (w.device_ptr() as *mut c_void, w.byte_len() as u64),
            None => (std::ptr::null_mut(), 0),
        };
        check("cutensorContract", (f.cutensorContract)(
            self.handle.raw(), self.plan,
            alpha as *const T as *const c_void, a_ptr, b_ptr,
            beta  as *const T as *const c_void, c_ptr, d_ptr,
            ws_ptr, ws_bytes, stream.raw(),
        ))
    }
}

impl Drop for Contract {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cutensorDestroyPlan)(self.plan);
                let _ = (f.cutensorDestroyPlanPreference)(self.pref);
                let _ = (f.cutensorDestroyOperationDescriptor)(self.op_descr);
            }
        }
    }
}
