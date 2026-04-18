//! cuDNN handle + per-device cache.
//!
//! This is the foundation the (coming) MHA / conv frontend builders sit on.
//! We expose create/destroy, stream binding, version query, and a
//! process-wide handle cache keyed by device ordinal.

use crate::drv::{self, Stream};
use iron_cuda_sys::cudnn as sys;
use ironaccelerator_core::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub use sys::{CudnnDataType as CudnnDType, CudnnStatus};

fn fns() -> Result<&'static sys::CudnnFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(
        format!("cudnn not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: CudnnStatus) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub struct CudnnHandle {
    handle: sys::CudnnHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

impl CudnnHandle {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CudnnHandle::default();
        unsafe { check("cudnnCreate", (f.cudnnCreate)(&mut h))?; }
        Ok(Arc::new(Self { handle: h, _device: device }))
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check("cudnnSetStream", (fns()?.cudnnSetStream)(self.handle, stream.raw())) }
    }

    #[inline] pub fn raw(&self) -> sys::CudnnHandle { self.handle }

    pub fn version() -> Result<usize> { Ok(unsafe { (fns()?.cudnnGetVersion)() }) }
    pub fn cudart_version() -> Result<usize> { Ok(unsafe { (fns()?.cudnnGetCudartVersion)() }) }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.cudnnDestroy)(self.handle); }
        }
    }
}

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<CudnnHandle>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn handle_for(stream: &Arc<Stream>) -> Result<Arc<CudnnHandle>> {
    let device = stream.device();
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) {
            h.set_stream(stream)?;
            return Ok(h.clone());
        }
    }
    let h = CudnnHandle::new(device.clone())?;
    h.set_stream(stream)?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

pub fn is_available() -> bool { sys::is_available() }

// ─── Generic v9 backend-descriptor wrapper ─────────────────────────────────
//
// The v9 graph API is: create a descriptor of a known kind, set a handful of
// attributes, call `Finalize`, then either chain it into another descriptor
// (as an attribute) or execute it with a VariantPack.

use std::ffi::c_void;

/// Raw attribute-type tags from `cudnnBackendAttributeType_t`.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum AttrType {
    Handle = 0,
    DataType = 1,
    Boolean = 2,
    Int64 = 3,
    Float = 4,
    Double = 5,
    VoidPtr = 6,
    ConvMode = 7,
    Heur = 8,
    KnobType = 9,
    NanPropagation = 10,
    NumericalNote = 11,
    LayoutType = 12,
    AttribName = 13,
    PointwiseMode = 14,
    BackendDescriptor = 15,
    GenStats = 16,
    BnFinalizeStatsMode = 17,
    ReductionOperatorType = 18,
    BehaviorNote = 19,
    TensorReorderingMode = 20,
    ResampleMode = 21,
    PaddingMode = 22,
    IntArray = 23,
    NormMode = 24,
    NormFwdPhase = 25,
    RngDistribution = 26,
}

pub struct BackendDescr {
    raw: sys::CudnnBackendDescriptor,
    kind: u32,
    finalized: bool,
}

unsafe impl Send for BackendDescr {}
unsafe impl Sync for BackendDescr {}

impl BackendDescr {
    /// Create a descriptor of the given `cudnnBackendDescriptorType_t` kind.
    pub fn new(kind: u32) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CudnnBackendDescriptor::default();
        unsafe {
            check("cudnnBackendCreateDescriptor",
                  (f.cudnnBackendCreateDescriptor)(kind, &mut raw))?;
        }
        Ok(Self { raw, kind, finalized: false })
    }

    #[inline] pub fn raw(&self) -> sys::CudnnBackendDescriptor { self.raw }
    #[inline] pub fn kind(&self) -> u32 { self.kind }
    #[inline] pub fn is_finalized(&self) -> bool { self.finalized }

    /// Set `count` elements of `ty` at the given attribute slot. Generic over
    /// the element type — caller is responsible for matching `ty` to `T`.
    ///
    /// # Safety
    /// `ty` must be a valid `cudnnBackendAttributeType_t` and `T` must match
    /// its expected representation.
    pub unsafe fn set_attribute<T: Copy>(
        &mut self, attr: u32, ty: AttrType, elements: &[T],
    ) -> Result<()> {
        let f = fns()?;
        check("cudnnBackendSetAttribute",
              (f.cudnnBackendSetAttribute)(
                  self.raw, attr, ty as u32, elements.len() as i64,
                  elements.as_ptr() as *const c_void))
    }

    pub fn set_i64(&mut self, attr: u32, vals: &[i64]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Int64, vals) }
    }
    pub fn set_f64(&mut self, attr: u32, vals: &[f64]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Double, vals) }
    }
    pub fn set_bool(&mut self, attr: u32, vals: &[u8]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Boolean, vals) }
    }
    pub fn set_descriptors(&mut self, attr: u32, descs: &[sys::CudnnBackendDescriptor]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::BackendDescriptor, descs) }
    }
    pub fn set_handle(&mut self, attr: u32, handles: &[sys::CudnnHandle]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Handle, handles) }
    }
    pub fn set_data_type(&mut self, attr: u32, dtypes: &[CudnnDType]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::DataType, dtypes) }
    }

    pub fn finalize(&mut self) -> Result<()> {
        let f = fns()?;
        unsafe { check("cudnnBackendFinalize", (f.cudnnBackendFinalize)(self.raw))?; }
        self.finalized = true;
        Ok(())
    }

    /// Execute an already-finalized engine/plan descriptor against a
    /// finalized variant-pack descriptor.
    pub fn execute(
        handle: &CudnnHandle,
        plan: &BackendDescr,
        variant_pack: &BackendDescr,
    ) -> Result<()> {
        if !plan.finalized || !variant_pack.finalized {
            return Err(Error::Other("cudnn::execute: plan and variant pack must be finalized"));
        }
        let f = fns()?;
        unsafe {
            check("cudnnBackendExecute", (f.cudnnBackendExecute)(
                handle.raw(), plan.raw, variant_pack.raw))
        }
    }
}

impl Drop for BackendDescr {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.cudnnBackendDestroyDescriptor)(self.raw); }
        }
    }
}
