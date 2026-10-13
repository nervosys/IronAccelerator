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
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cudnn not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: CudnnStatus) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

/// Owned cuDNN handle (`cudnnHandle_t`) bound to one device; destroyed on drop.
pub struct CudnnHandle {
    handle: sys::CudnnHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

impl CudnnHandle {
    /// Creates a cuDNN handle on `device` (`cudnnCreate`). Binds the device first.
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CudnnHandle::default();
        unsafe {
            check("cudnnCreate", (f.cudnnCreate)(&mut h))?;
        }
        Ok(Arc::new(Self {
            handle: h,
            _device: device,
        }))
    }

    /// Binds `stream` to this handle (`cudnnSetStream`).
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "cudnnSetStream",
                (fns()?.cudnnSetStream)(self.handle, stream.raw()),
            )
        }
    }

    /// Returns the underlying `cudnnHandle_t`.
    #[inline]
    pub fn raw(&self) -> sys::CudnnHandle {
        self.handle
    }

    /// Returns the cuDNN library version (`cudnnGetVersion`).
    pub fn version() -> Result<usize> {
        Ok(unsafe { (fns()?.cudnnGetVersion)() })
    }
    /// Returns the CUDA runtime version cuDNN was built against (`cudnnGetCudartVersion`).
    pub fn cudart_version() -> Result<usize> {
        Ok(unsafe { (fns()?.cudnnGetCudartVersion)() })
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cudnnDestroy)(self.handle);
            }
        }
    }
}

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<CudnnHandle>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Returns the cached per-device handle for `stream`'s device, creating it on first use, and binds `stream` to it.
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
    // Double-checked insert: a parallel caller may have cached a handle for this
    // ordinal while we were creating ours. Adopt whichever landed first (dropping
    // our redundant one) so callers share a single handle, then bind the caller's
    // stream to it — matching the rebind-on-each-borrow contract the hit path
    // above also honours.
    let created = CudnnHandle::new(device.clone())?;
    let h = HANDLES.lock().entry(ord).or_insert(created).clone();
    h.set_stream(stream)?;
    Ok(h)
}

/// Returns `true` if the cuDNN library loads at runtime.
pub fn is_available() -> bool {
    sys::is_available()
}

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
    /// Mirrors `CUDNN_TYPE_HANDLE`.
    Handle = 0,
    /// Mirrors `CUDNN_TYPE_DATA_TYPE`.
    DataType = 1,
    /// Mirrors `CUDNN_TYPE_BOOLEAN`.
    Boolean = 2,
    /// Mirrors `CUDNN_TYPE_INT64`.
    Int64 = 3,
    /// Mirrors `CUDNN_TYPE_FLOAT`.
    Float = 4,
    /// Mirrors `CUDNN_TYPE_DOUBLE`.
    Double = 5,
    /// Mirrors `CUDNN_TYPE_VOID_PTR`.
    VoidPtr = 6,
    /// Mirrors `CUDNN_TYPE_CONVOLUTION_MODE`.
    ConvMode = 7,
    /// Mirrors `CUDNN_TYPE_HEUR_MODE`.
    Heur = 8,
    /// Mirrors `CUDNN_TYPE_KNOB_TYPE`.
    KnobType = 9,
    /// Mirrors `CUDNN_TYPE_NAN_PROPOGATION`.
    NanPropagation = 10,
    /// Mirrors `CUDNN_TYPE_NUMERICAL_NOTE`.
    NumericalNote = 11,
    /// Mirrors `CUDNN_TYPE_LAYOUT_TYPE`.
    LayoutType = 12,
    /// Mirrors `CUDNN_TYPE_ATTRIB_NAME`.
    AttribName = 13,
    /// Mirrors `CUDNN_TYPE_POINTWISE_MODE`.
    PointwiseMode = 14,
    /// Mirrors `CUDNN_TYPE_BACKEND_DESCRIPTOR`.
    BackendDescriptor = 15,
    /// Mirrors `CUDNN_TYPE_GENSTATS_MODE`.
    GenStats = 16,
    /// Mirrors `CUDNN_TYPE_BN_FINALIZE_STATS_MODE`.
    BnFinalizeStatsMode = 17,
    /// Mirrors `CUDNN_TYPE_REDUCTION_OPERATOR_TYPE`.
    ReductionOperatorType = 18,
    /// Mirrors `CUDNN_TYPE_BEHAVIOR_NOTE`.
    BehaviorNote = 19,
    /// Mirrors `CUDNN_TYPE_TENSOR_REORDERING_MODE`.
    TensorReorderingMode = 20,
    /// Mirrors `CUDNN_TYPE_RESAMPLE_MODE`.
    ResampleMode = 21,
    /// Mirrors `CUDNN_TYPE_PADDING_MODE`.
    PaddingMode = 22,
    /// Mirrors `CUDNN_TYPE_INT32` — `int32_t` values, one or an array.
    IntArray = 23,
    /// Mirrors `CUDNN_TYPE_CHAR`.
    Char = 24,
    /// Mirrors `CUDNN_TYPE_SIGNAL_MODE`.
    SignalMode = 25,
    /// Mirrors `CUDNN_TYPE_FRACTION`.
    Fraction = 26,
    /// Mirrors `CUDNN_TYPE_NORM_MODE`.
    NormMode = 27,
    /// Mirrors `CUDNN_TYPE_NORM_FWD_PHASE`.
    NormFwdPhase = 28,
    /// Mirrors `CUDNN_TYPE_RNG_DISTRIBUTION`.
    RngDistribution = 29,
}

/// RAII wrapper around a `cudnnBackendDescriptor_t`; destroyed on drop.
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
            check(
                "cudnnBackendCreateDescriptor",
                (f.cudnnBackendCreateDescriptor)(kind, &mut raw),
            )?;
        }
        Ok(Self {
            raw,
            kind,
            finalized: false,
        })
    }

    /// Returns the underlying `cudnnBackendDescriptor_t`.
    #[inline]
    pub fn raw(&self) -> sys::CudnnBackendDescriptor {
        self.raw
    }
    /// Returns the `cudnnBackendDescriptorType_t` this descriptor was created with.
    #[inline]
    pub fn kind(&self) -> u32 {
        self.kind
    }
    /// Returns `true` once `finalize` has succeeded.
    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Set `count` elements of `ty` at the given attribute slot. Generic over
    /// the element type — caller is responsible for matching `ty` to `T`.
    ///
    /// # Safety
    /// `ty` must be a valid `cudnnBackendAttributeType_t` and `T` must match
    /// its expected representation.
    pub unsafe fn set_attribute<T: Copy>(
        &mut self,
        attr: u32,
        ty: AttrType,
        elements: &[T],
    ) -> Result<()> {
        let f = fns()?;
        check(
            "cudnnBackendSetAttribute",
            (f.cudnnBackendSetAttribute)(
                self.raw,
                attr,
                ty as u32,
                elements.len() as i64,
                elements.as_ptr() as *const c_void,
            ),
        )
    }

    /// Sets an `i64` attribute slot (`CUDNN_TYPE_INT64`).
    pub fn set_i64(&mut self, attr: u32, vals: &[i64]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Int64, vals) }
    }
    /// Sets an `f64` attribute slot (`CUDNN_TYPE_DOUBLE`).
    pub fn set_f64(&mut self, attr: u32, vals: &[f64]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Double, vals) }
    }
    /// Sets a boolean attribute slot (`CUDNN_TYPE_BOOLEAN`) from `u8` values.
    pub fn set_bool(&mut self, attr: u32, vals: &[u8]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Boolean, vals) }
    }
    /// Sets a backend-descriptor attribute slot (`CUDNN_TYPE_BACKEND_DESCRIPTOR`).
    pub fn set_descriptors(
        &mut self,
        attr: u32,
        descs: &[sys::CudnnBackendDescriptor],
    ) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::BackendDescriptor, descs) }
    }
    /// Sets a `cudnnHandle_t` attribute slot (`CUDNN_TYPE_HANDLE`).
    pub fn set_handle(&mut self, attr: u32, handles: &[sys::CudnnHandle]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::Handle, handles) }
    }
    /// Sets a data-type attribute slot (`CUDNN_TYPE_DATA_TYPE`).
    pub fn set_data_type(&mut self, attr: u32, dtypes: &[CudnnDType]) -> Result<()> {
        unsafe { self.set_attribute(attr, AttrType::DataType, dtypes) }
    }

    /// Finalizes the descriptor (`cudnnBackendFinalize`) and marks it finalized.
    pub fn finalize(&mut self) -> Result<()> {
        let f = fns()?;
        unsafe {
            check("cudnnBackendFinalize", (f.cudnnBackendFinalize)(self.raw))?;
        }
        self.finalized = true;
        Ok(())
    }

    /// Read `cap` elements of `ty` from the given attribute slot. Returns
    /// the actual count reported by cuDNN.
    ///
    /// # Safety
    /// `ty` must match the attribute's declared type, and `out` must have
    /// room for `cap` elements of `T`.
    pub unsafe fn get_attribute<T>(
        &self,
        attr: u32,
        ty: AttrType,
        cap: i64,
        out: *mut T,
    ) -> Result<i64> {
        let f = fns()?;
        let mut n: i64 = 0;
        unsafe {
            check(
                "cudnnBackendGetAttribute",
                (f.cudnnBackendGetAttribute)(
                    self.raw,
                    attr,
                    ty as u32,
                    cap,
                    &mut n,
                    out as *mut c_void,
                ),
            )?;
        }
        Ok(n)
    }

    /// Reads a single `i64` attribute slot (`cudnnBackendGetAttribute`).
    pub fn get_i64(&self, attr: u32) -> Result<i64> {
        let mut v: i64 = 0;
        unsafe {
            self.get_attribute(attr, AttrType::Int64, 1, &mut v)?;
        }
        Ok(v)
    }
    /// Reads a single backend-descriptor attribute slot. Errors if cuDNN returns no element.
    pub fn get_descriptor(&self, attr: u32) -> Result<sys::CudnnBackendDescriptor> {
        let mut v = sys::CudnnBackendDescriptor::default();
        let n = unsafe { self.get_attribute(attr, AttrType::BackendDescriptor, 1, &mut v)? };
        if n < 1 {
            return Err(Error::Other("cudnn::get_descriptor: no result"));
        }
        Ok(v)
    }

    /// Execute an already-finalized engine/plan descriptor against a
    /// finalized variant-pack descriptor.
    pub fn execute(
        handle: &CudnnHandle,
        plan: &BackendDescr,
        variant_pack: &BackendDescr,
    ) -> Result<()> {
        if !plan.finalized || !variant_pack.finalized {
            return Err(Error::Other(
                "cudnn::execute: plan and variant pack must be finalized",
            ));
        }
        let f = fns()?;
        unsafe {
            check(
                "cudnnBackendExecute",
                (f.cudnnBackendExecute)(handle.raw(), plan.raw, variant_pack.raw),
            )
        }
    }
}

/// Adopt a raw descriptor that was created externally (e.g. by
/// `cudnnBackendGetAttribute`) so it participates in RAII destruction.
pub fn adopt_descriptor(
    raw: sys::CudnnBackendDescriptor,
    kind: u32,
    finalized: bool,
) -> BackendDescr {
    BackendDescr {
        raw,
        kind,
        finalized,
    }
}

impl Drop for BackendDescr {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cudnnBackendDestroyDescriptor)(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod attr_type_tests {
    use super::AttrType;

    /// Declaration order in `cudnnBackendAttributeType_t` (cuDNN 9).
    #[test]
    fn tags_match_cudnn_header() {
        assert_eq!(AttrType::PaddingMode as u32, 22);
        assert_eq!(AttrType::IntArray as u32, 23);
        assert_eq!(AttrType::Char as u32, 24);
        assert_eq!(AttrType::SignalMode as u32, 25);
        assert_eq!(AttrType::Fraction as u32, 26);
        assert_eq!(AttrType::NormMode as u32, 27);
        assert_eq!(AttrType::NormFwdPhase as u32, 28);
        assert_eq!(AttrType::RngDistribution as u32, 29);
    }
}
