//! hipBLASLt safe wrapper — mirrors `ironaccelerator_cuda::blas`.
//!
//! Narrow surface: handle, descriptor, layout, preference, heuristic, matmul.
//! Higher-level GEMM recipes compose these primitives.

use crate::drv::{self, DeviceBuf, Stream};
use iron_rocm_sys::hip::HipDeviceptr;
use iron_rocm_sys::hipblaslt as sys;
use ironaccelerator_core::{BackendKind, Error, Result, Strategy};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

pub use sys::{
    HipblasComputeType as ComputeType, HipblasDataType as DType, HipblasOp as Op,
    HipblasLtEpilogue as Epilogue,
};

fn fns() -> Result<&'static sys::HipblasLtFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(format!("hipblasLt not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: sys::HipblasStatus) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend { backend: BackendKind::Rocm, code: (s as u32) as i64 })
    }
}

// ─── Handle ─────────────────────────────────────────────────────────────────

pub struct BlasLt {
    handle: sys::HipblasLtHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for BlasLt {}
unsafe impl Sync for BlasLt {}

impl BlasLt {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::HipblasLtHandle::default();
        unsafe { check("hipblasLtCreate", (f.hipblasLtCreate)(&mut h))?; }
        Ok(Arc::new(Self { handle: h, _device: device }))
    }

    #[inline] pub fn raw(&self) -> sys::HipblasLtHandle { self.handle }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.hipblasLtDestroy)(self.handle); }
        }
    }
}

// ─── Process-wide handle cache, keyed by device ordinal ─────────────────────

static HANDLES: once_cell::sync::Lazy<Mutex<HashMap<u32, Arc<BlasLt>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn handle_for(stream: &Arc<Stream>) -> Result<Arc<BlasLt>> {
    let device = stream.device();
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) { return Ok(h.clone()); }
    }
    let h = BlasLt::new(device.clone())?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

// ─── Matmul descriptor ──────────────────────────────────────────────────────

pub struct MatmulDesc { raw: sys::HipblasLtMatmulDesc }
unsafe impl Send for MatmulDesc {}
unsafe impl Sync for MatmulDesc {}

impl MatmulDesc {
    pub fn new(compute: ComputeType, scale: DType) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::HipblasLtMatmulDesc::default();
        unsafe { check("hipblasLtMatmulDescCreate",
                       (f.hipblasLtMatmulDescCreate)(&mut raw, compute, scale))?; }
        Ok(Self { raw })
    }

    pub fn set_transpose(&mut self, trans_a: Op, trans_b: Op) -> Result<()> {
        let f = fns()?;
        let ta = trans_a as u32;
        let tb = trans_b as u32;
        unsafe {
            check("hipblasLtMatmulDescSetAttribute(TransA)",
                  (f.hipblasLtMatmulDescSetAttribute)(
                      self.raw, sys::HipblasLtMatmulDescAttr::TransA,
                      &ta as *const u32 as *const c_void, std::mem::size_of::<u32>()))?;
            check("hipblasLtMatmulDescSetAttribute(TransB)",
                  (f.hipblasLtMatmulDescSetAttribute)(
                      self.raw, sys::HipblasLtMatmulDescAttr::TransB,
                      &tb as *const u32 as *const c_void, std::mem::size_of::<u32>()))?;
        }
        Ok(())
    }

    pub fn set_epilogue(&mut self, epi: Epilogue) -> Result<()> {
        let f = fns()?;
        let v = epi as u32;
        unsafe {
            check("hipblasLtMatmulDescSetAttribute(Epilogue)",
                  (f.hipblasLtMatmulDescSetAttribute)(
                      self.raw, sys::HipblasLtMatmulDescAttr::Epilogue,
                      &v as *const u32 as *const c_void, std::mem::size_of::<u32>()))
        }
    }

    pub fn set_bias_pointer(&mut self, ptr: HipDeviceptr) -> Result<()> {
        let f = fns()?;
        unsafe {
            check("hipblasLtMatmulDescSetAttribute(BiasPointer)",
                  (f.hipblasLtMatmulDescSetAttribute)(
                      self.raw, sys::HipblasLtMatmulDescAttr::BiasPointer,
                      &ptr as *const _ as *const c_void, std::mem::size_of::<u64>()))
        }
    }

    pub fn set_scale_pointer(&mut self, which: ScaleTensor, ptr: HipDeviceptr) -> Result<()> {
        let f = fns()?;
        let attr = match which {
            ScaleTensor::A => sys::HipblasLtMatmulDescAttr::ScaleA,
            ScaleTensor::B => sys::HipblasLtMatmulDescAttr::ScaleB,
            ScaleTensor::C => sys::HipblasLtMatmulDescAttr::ScaleC,
            ScaleTensor::D => sys::HipblasLtMatmulDescAttr::ScaleD,
        };
        unsafe {
            check("hipblasLtMatmulDescSetAttribute(Scale)",
                  (f.hipblasLtMatmulDescSetAttribute)(
                      self.raw, attr,
                      &ptr as *const _ as *const c_void, std::mem::size_of::<u64>()))
        }
    }

    #[inline] pub fn raw(&self) -> sys::HipblasLtMatmulDesc { self.raw }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.hipblasLtMatmulDescDestroy)(self.raw); }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ScaleTensor { A, B, C, D }

// ─── Matrix layout ──────────────────────────────────────────────────────────

pub struct MatrixLayout { raw: sys::HipblasLtMatrixLayout }
unsafe impl Send for MatrixLayout {}
unsafe impl Sync for MatrixLayout {}

impl MatrixLayout {
    pub fn new(dtype: DType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::HipblasLtMatrixLayout::default();
        unsafe {
            check("hipblasLtMatrixLayoutCreate",
                  (f.hipblasLtMatrixLayoutCreate)(&mut raw, dtype, rows, cols, ld))?;
        }
        Ok(Self { raw })
    }

    pub fn set_batch(&mut self, count: i32, stride: i64) -> Result<()> {
        let f = fns()?;
        unsafe {
            check("hipblasLtMatrixLayoutSetAttribute(BatchCount)",
                  (f.hipblasLtMatrixLayoutSetAttribute)(
                      self.raw, sys::HipblasLtMatrixLayoutAttr::BatchCount,
                      &count as *const _ as *const c_void, std::mem::size_of::<i32>()))?;
            check("hipblasLtMatrixLayoutSetAttribute(StridedBatchOffset)",
                  (f.hipblasLtMatrixLayoutSetAttribute)(
                      self.raw, sys::HipblasLtMatrixLayoutAttr::StridedBatchOffset,
                      &stride as *const _ as *const c_void, std::mem::size_of::<i64>()))?;
        }
        Ok(())
    }

    #[inline] pub fn raw(&self) -> sys::HipblasLtMatrixLayout { self.raw }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.hipblasLtMatrixLayoutDestroy)(self.raw); }
        }
    }
}

// ─── Preference ─────────────────────────────────────────────────────────────

pub struct Preference { raw: sys::HipblasLtMatmulPreference }
unsafe impl Send for Preference {}
unsafe impl Sync for Preference {}

impl Preference {
    pub fn new() -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::HipblasLtMatmulPreference::default();
        unsafe { check("hipblasLtMatmulPreferenceCreate",
                       (f.hipblasLtMatmulPreferenceCreate)(&mut raw))?; }
        Ok(Self { raw })
    }

    pub fn set_max_workspace(&mut self, bytes: usize) -> Result<()> {
        let f = fns()?;
        unsafe {
            check("hipblasLtMatmulPreferenceSetAttribute(MaxWorkspaceBytes)",
                  (f.hipblasLtMatmulPreferenceSetAttribute)(
                      self.raw, sys::HipblasLtMatmulPreferenceAttr::MaxWorkspaceBytes,
                      &bytes as *const _ as *const c_void, std::mem::size_of::<usize>()))
        }
    }

    #[inline] pub fn raw(&self) -> sys::HipblasLtMatmulPreference { self.raw }
}

impl Drop for Preference {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.hipblasLtMatmulPreferenceDestroy)(self.raw); }
        }
    }
}

// ─── Heuristic + launch ─────────────────────────────────────────────────────

pub fn heuristic(
    blaslt: &BlasLt,
    desc: &MatmulDesc,
    a: &MatrixLayout, b: &MatrixLayout, c: &MatrixLayout, d: &MatrixLayout,
    pref: &Preference,
) -> Result<sys::HipblasLtMatmulHeuristicResult> {
    let f = fns()?;
    let mut out = sys::HipblasLtMatmulHeuristicResult::default();
    let mut returned: i32 = 0;
    unsafe {
        check("hipblasLtMatmulAlgoGetHeuristic",
              (f.hipblasLtMatmulAlgoGetHeuristic)(
                  blaslt.handle, desc.raw,
                  a.raw, b.raw, c.raw, d.raw,
                  pref.raw,
                  1, &mut out, &mut returned))?;
    }
    if returned == 0 {
        return Err(Error::Other("hipblasLtMatmulAlgoGetHeuristic: no algorithms returned"));
    }
    Ok(out)
}

/// Execute `D = alpha · op(A) · op(B) + beta · C`.
///
/// # Safety
/// All descriptor / layout dimensions must be consistent with the underlying
/// device pointers and the host scalars.
pub unsafe fn matmul(
    blaslt: &BlasLt,
    desc: &MatmulDesc,
    alpha: &[u8], beta: &[u8],
    a_ptr: HipDeviceptr, a_layout: &MatrixLayout,
    b_ptr: HipDeviceptr, b_layout: &MatrixLayout,
    c_ptr: HipDeviceptr, c_layout: &MatrixLayout,
    d_ptr: HipDeviceptr, d_layout: &MatrixLayout,
    algo: Option<&sys::HipblasLtMatmulHeuristicResult>,
    workspace: Option<&mut DeviceBuf<u8>>,
    stream: &Stream,
) -> Result<()> {
    let f = fns()?;
    let algo_ptr = algo.map(|a| &a.algo as *const _).unwrap_or(std::ptr::null());
    let (ws_ptr, ws_bytes) = match workspace {
        Some(w) => (w.device_ptr() as *mut c_void, w.byte_len()),
        None => (std::ptr::null_mut(), 0),
    };
    check("hipblasLtMatmul",
          (f.hipblasLtMatmul)(
              blaslt.handle, desc.raw,
              alpha.as_ptr() as *const c_void,
              a_ptr as *const c_void, a_layout.raw,
              b_ptr as *const c_void, b_layout.raw,
              beta.as_ptr() as *const c_void,
              c_ptr as *const c_void, c_layout.raw,
              d_ptr as *mut c_void,   d_layout.raw,
              algo_ptr,
              ws_ptr, ws_bytes,
              stream.raw(),
          ))
}

// ─── Convenience: planner epilogue tag ──────────────────────────────────────

pub fn epilogue_for(strategy: &Strategy) -> &'static str {
    match strategy {
        Strategy::BlasLt { epilogue } => epilogue,
        _ => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn epilogue_lookup() {
        assert_eq!(epilogue_for(&Strategy::BlasLt { epilogue: "bias-gelu" }), "bias-gelu");
        assert_eq!(epilogue_for(&Strategy::Reference), "none");
    }
}
