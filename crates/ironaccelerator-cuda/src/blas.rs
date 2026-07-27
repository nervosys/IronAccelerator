//! cuBLASLt safe wrapper.
//!
//! The layer is intentionally narrow: a handle, a descriptor/layout RAII pair,
//! a preference knob, and one `matmul` entry point. Higher-level GEMM recipes
//! (FP8 delayed-scaling, epilogue fusion) compose these primitives — they do
//! not live here.

use crate::drv::{self, DeviceBuf, Stream};
use iron_cuda_sys::cublas_lt as sys;
use ironaccelerator_core::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

pub use sys::{
    CublasComputeType as ComputeType, CublasLtOrder as Order, CublasOp as Op, CudaDataType as DType,
};

fn fns() -> Result<&'static sys::CublasLtFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cublasLt not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: sys::CublasStatus) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

// ─── Handle ─────────────────────────────────────────────────────────────────

pub struct BlasLt {
    handle: sys::CublasLtHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for BlasLt {}
unsafe impl Sync for BlasLt {}

impl BlasLt {
    pub fn new(device: Arc<drv::Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = fns()?;
        let mut h = sys::CublasLtHandle::default();
        unsafe {
            check("cublasLtCreate", (f.cublasLtCreate)(&mut h))?;
        }
        Ok(Arc::new(Self {
            handle: h,
            _device: device,
        }))
    }

    #[inline]
    pub fn raw(&self) -> sys::CublasLtHandle {
        self.handle
    }

    pub fn version() -> Result<usize> {
        Ok(unsafe { (fns()?.cublasLtGetVersion)() })
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cublasLtDestroy)(self.handle);
            }
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
        if let Some(h) = g.get(&ord) {
            return Ok(h.clone());
        }
    }
    let h = BlasLt::new(device.clone())?;
    HANDLES.lock().insert(ord, h.clone());
    Ok(h)
}

// ─── Matmul descriptor ──────────────────────────────────────────────────────

pub struct MatmulDesc {
    raw: sys::CublasLtMatmulDesc,
}

unsafe impl Send for MatmulDesc {}
unsafe impl Sync for MatmulDesc {}

impl MatmulDesc {
    pub fn new(compute: ComputeType, scale: DType) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CublasLtMatmulDesc::default();
        unsafe {
            check(
                "cublasLtMatmulDescCreate",
                (f.cublasLtMatmulDescCreate)(&mut raw, compute, scale),
            )?;
        }
        Ok(Self { raw })
    }

    pub fn set_transpose(&mut self, trans_a: Op, trans_b: Op) -> Result<()> {
        let f = fns()?;
        let ta = trans_a as u32;
        let tb = trans_b as u32;
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(TransA)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::TransA,
                    &ta as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
            )?;
            check(
                "cublasLtMatmulDescSetAttribute(TransB)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::TransB,
                    &tb as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
            )?;
        }
        Ok(())
    }

    /// Set a raw `cublasLtEpilogue_t` value. See NVIDIA's cuBLASLt docs.
    pub fn set_epilogue_raw(&mut self, epi: u32) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(Epilogue)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::Epilogue,
                    &epi as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
            )
        }
    }

    /// Set a matmul-desc attribute identified by its raw enum value.
    /// `CUBLASLT_MATMUL_DESC_FAST_ACCUM = 11` is the only one we need beyond
    /// the typed setters, so this escape hatch is kept deliberately narrow.
    pub unsafe fn set_attr_raw_u32(&mut self, attr_raw: u32, value: u32) -> Result<()> {
        let f = fns()?;
        // SAFETY: `CublasLtMatmulDescAttr` is `#[repr(u32)]`; any in-range
        // discriminant is valid. The caller vouches for the raw attribute.
        let attr: sys::CublasLtMatmulDescAttr = std::mem::transmute(attr_raw);
        check(
            "cublasLtMatmulDescSetAttribute(raw)",
            (f.cublasLtMatmulDescSetAttribute)(
                self.raw,
                attr,
                &value as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            ),
        )
    }

    pub fn set_amax_d_pointer(&mut self, ptr: iron_cuda_sys::driver::CUdeviceptr) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(AmaxDPointer)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::AmaxDPointer,
                    &ptr as *const _ as *const c_void,
                    std::mem::size_of::<u64>(),
                ),
            )
        }
    }

    pub fn set_bias_pointer(&mut self, ptr: iron_cuda_sys::driver::CUdeviceptr) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(BiasPointer)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::BiasPointer,
                    &ptr as *const _ as *const c_void,
                    std::mem::size_of::<u64>(),
                ),
            )
        }
    }

    /// Set a device scale pointer (for FP8 delayed-scaling). `which` picks
    /// whether this is A, B, C, or D's scale.
    pub fn set_scale_pointer(
        &mut self,
        which: ScaleTensor,
        ptr: iron_cuda_sys::driver::CUdeviceptr,
    ) -> Result<()> {
        let f = fns()?;
        let attr = match which {
            ScaleTensor::A => sys::CublasLtMatmulDescAttr::ScaleA,
            ScaleTensor::B => sys::CublasLtMatmulDescAttr::ScaleB,
            ScaleTensor::C => sys::CublasLtMatmulDescAttr::ScaleC,
            ScaleTensor::D => sys::CublasLtMatmulDescAttr::ScaleD,
        };
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(Scale)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    attr,
                    &ptr as *const _ as *const c_void,
                    std::mem::size_of::<u64>(),
                ),
            )
        }
    }

    #[inline]
    pub fn raw(&self) -> sys::CublasLtMatmulDesc {
        self.raw
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cublasLtMatmulDescDestroy)(self.raw);
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ScaleTensor {
    A,
    B,
    C,
    D,
}

// ─── Matrix layout ──────────────────────────────────────────────────────────

pub struct MatrixLayout {
    raw: sys::CublasLtMatrixLayout,
}

unsafe impl Send for MatrixLayout {}
unsafe impl Sync for MatrixLayout {}

impl MatrixLayout {
    pub fn new(dtype: DType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CublasLtMatrixLayout::default();
        unsafe {
            check(
                "cublasLtMatrixLayoutCreate",
                (f.cublasLtMatrixLayoutCreate)(&mut raw, dtype, rows, cols, ld),
            )?;
        }
        Ok(Self { raw })
    }

    pub fn set_order(&mut self, order: Order) -> Result<()> {
        let f = fns()?;
        let v = order as u32;
        unsafe {
            check(
                "cublasLtMatrixLayoutSetAttribute(Order)",
                (f.cublasLtMatrixLayoutSetAttribute)(
                    self.raw,
                    sys::CublasLtMatrixLayoutAttr::Order,
                    &v as *const _ as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
            )
        }
    }

    pub fn set_batch(&mut self, count: i32, stride: i64) -> Result<()> {
        let f = fns()?;
        unsafe {
            check(
                "cublasLtMatrixLayoutSetAttribute(BatchCount)",
                (f.cublasLtMatrixLayoutSetAttribute)(
                    self.raw,
                    sys::CublasLtMatrixLayoutAttr::BatchCount,
                    &count as *const _ as *const c_void,
                    std::mem::size_of::<i32>(),
                ),
            )?;
            check(
                "cublasLtMatrixLayoutSetAttribute(StridedBatchOffset)",
                (f.cublasLtMatrixLayoutSetAttribute)(
                    self.raw,
                    sys::CublasLtMatrixLayoutAttr::StridedBatchOffset,
                    &stride as *const _ as *const c_void,
                    std::mem::size_of::<i64>(),
                ),
            )?;
        }
        Ok(())
    }

    #[inline]
    pub fn raw(&self) -> sys::CublasLtMatrixLayout {
        self.raw
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cublasLtMatrixLayoutDestroy)(self.raw);
            }
        }
    }
}

// ─── Preference ─────────────────────────────────────────────────────────────

pub struct Preference {
    raw: sys::CublasLtMatmulPreference,
}

unsafe impl Send for Preference {}
unsafe impl Sync for Preference {}

impl Preference {
    pub fn new() -> Result<Self> {
        let f = fns()?;
        let mut raw = sys::CublasLtMatmulPreference::default();
        unsafe {
            check(
                "cublasLtMatmulPreferenceCreate",
                (f.cublasLtMatmulPreferenceCreate)(&mut raw),
            )?;
        }
        Ok(Self { raw })
    }

    pub fn set_max_workspace(&mut self, bytes: usize) -> Result<()> {
        let f = fns()?;
        // cuBLASLt documents MaxWorkspaceBytes as u64 (size_t). On 64-bit
        // platforms `usize` matches; on 32-bit we still need a 8-byte write.
        let v: u64 = bytes as u64;
        unsafe {
            check(
                "cublasLtMatmulPreferenceSetAttribute(MaxWorkspaceBytes)",
                (f.cublasLtMatmulPreferenceSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulPrefAttr::MaxWorkspaceBytes,
                    &v as *const u64 as *const c_void,
                    std::mem::size_of::<u64>(),
                ),
            )
        }
    }

    #[inline]
    pub fn raw(&self) -> sys::CublasLtMatmulPreference {
        self.raw
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cublasLtMatmulPreferenceDestroy)(self.raw);
            }
        }
    }
}

// ─── Heuristic + launch ─────────────────────────────────────────────────────

/// Pick the best algorithm for a given configuration.
pub fn heuristic(
    blaslt: &BlasLt,
    desc: &MatmulDesc,
    a: &MatrixLayout,
    b: &MatrixLayout,
    c: &MatrixLayout,
    d: &MatrixLayout,
    pref: &Preference,
) -> Result<sys::CublasLtMatmulHeuristicResult> {
    let f = fns()?;
    let mut out = sys::CublasLtMatmulHeuristicResult::default();
    let mut returned: i32 = 0;
    unsafe {
        check(
            "cublasLtMatmulAlgoGetHeuristic",
            (f.cublasLtMatmulAlgoGetHeuristic)(
                blaslt.handle,
                desc.raw,
                a.raw,
                b.raw,
                c.raw,
                d.raw,
                pref.raw,
                1,
                &mut out,
                &mut returned,
            ),
        )?;
    }
    if returned == 0 {
        return Err(Error::Other(
            "cublasLtMatmulAlgoGetHeuristic: no algorithms returned",
        ));
    }
    Ok(out)
}

/// Execute `D = alpha · op(A) · op(B) + beta · C`.
///
/// `alpha` / `beta` are host scalars whose type must match the descriptor's
/// scale dtype. `workspace` must be at least as large as the heuristic's
/// reported `workspace_size`; pass `None` to use zero workspace.
///
/// # Safety
/// All descriptor / layout dimensions must be consistent with the underlying
/// device pointers and the host scalars.
pub unsafe fn matmul(
    blaslt: &BlasLt,
    desc: &MatmulDesc,
    alpha: &[u8],
    beta: &[u8],
    a_ptr: iron_cuda_sys::driver::CUdeviceptr,
    a_layout: &MatrixLayout,
    b_ptr: iron_cuda_sys::driver::CUdeviceptr,
    b_layout: &MatrixLayout,
    c_ptr: iron_cuda_sys::driver::CUdeviceptr,
    c_layout: &MatrixLayout,
    d_ptr: iron_cuda_sys::driver::CUdeviceptr,
    d_layout: &MatrixLayout,
    algo: Option<&sys::CublasLtMatmulHeuristicResult>,
    workspace: Option<&mut DeviceBuf<u8>>,
    stream: &Stream,
) -> Result<()> {
    let f = fns()?;
    let algo_ptr = algo
        .map(|a| a.algo.as_ptr() as *const c_void)
        .unwrap_or(std::ptr::null());
    let (ws_ptr, ws_bytes) = match workspace {
        Some(w) => (w.device_ptr() as *mut c_void, w.byte_len()),
        None => (std::ptr::null_mut(), 0),
    };
    check(
        "cublasLtMatmul",
        (f.cublasLtMatmul)(
            blaslt.handle,
            desc.raw,
            alpha.as_ptr() as *const c_void,
            a_ptr as *const c_void,
            a_layout.raw,
            b_ptr as *const c_void,
            b_layout.raw,
            beta.as_ptr() as *const c_void,
            c_ptr as *const c_void,
            c_layout.raw,
            d_ptr as *mut c_void,
            d_layout.raw,
            algo_ptr,
            ws_ptr,
            ws_bytes,
            stream.raw(),
        ),
    )
}
