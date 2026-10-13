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
            code: s.raw() as i64,
        })
    }
}

// ─── Handle ─────────────────────────────────────────────────────────────────

/// Owned cuBLASLt handle bound to one device. Destroyed on drop.
pub struct BlasLt {
    handle: sys::CublasLtHandle,
    _device: Arc<drv::Device>,
}

unsafe impl Send for BlasLt {}
unsafe impl Sync for BlasLt {}

impl BlasLt {
    /// Create a cuBLASLt handle on `device`. Wraps `cublasLtCreate`.
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

    /// Raw `cublasLtHandle_t`.
    #[inline]
    pub fn raw(&self) -> sys::CublasLtHandle {
        self.handle
    }

    /// cuBLASLt library version number. Wraps `cublasLtGetVersion`.
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

/// Return the process-wide cuBLASLt handle for `stream`'s device, creating it on first use.
pub fn handle_for(stream: &Arc<Stream>) -> Result<Arc<BlasLt>> {
    let device = stream.device();
    let ord = device.ordinal();
    {
        let g = HANDLES.lock();
        if let Some(h) = g.get(&ord) {
            return Ok(h.clone());
        }
    }
    // Create outside the lock (FFI), then double-check on insert: a parallel
    // caller may have cached a handle for this ordinal meanwhile. Keep whichever
    // landed first so every caller shares one handle instead of an unconditional
    // insert silently replacing (and orphaning) it.
    let h = BlasLt::new(device.clone())?;
    Ok(HANDLES.lock().entry(ord).or_insert(h).clone())
}

// ─── Matmul descriptor ──────────────────────────────────────────────────────

/// Owned `cublasLtMatmulDesc_t`. Destroyed on drop.
pub struct MatmulDesc {
    raw: sys::CublasLtMatmulDesc,
}

unsafe impl Send for MatmulDesc {}
unsafe impl Sync for MatmulDesc {}

impl MatmulDesc {
    /// Create a matmul descriptor with the given compute and scale types. Wraps `cublasLtMatmulDescCreate`.
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

    /// Set the transpose flags for operands A and B. Wraps `cublasLtMatmulDescSetAttribute`.
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

    /// Enable or disable FP8 fast accumulation
    /// (`CUBLASLT_MATMUL_DESC_FAST_ACCUM`, an `int8_t` flag).
    pub fn set_fast_accum(&mut self, enabled: bool) -> Result<()> {
        let f = fns()?;
        let v: i8 = enabled as i8;
        unsafe {
            check(
                "cublasLtMatmulDescSetAttribute(FAST_ACCUM)",
                (f.cublasLtMatmulDescSetAttribute)(
                    self.raw,
                    sys::CublasLtMatmulDescAttr::FastAccum,
                    &v as *const i8 as *const c_void,
                    std::mem::size_of::<i8>(),
                ),
            )
        }
    }

    /// Set a 4-byte matmul-desc attribute by its raw
    /// `cublasLtMatmulDescAttributes_t` value, for attributes without a typed
    /// setter.
    ///
    /// # Safety
    /// `attr_raw` must name an attribute whose value is exactly 4 bytes and
    /// `value` must be valid for it; cuBLASLt reads it as such.
    pub unsafe fn set_attr_raw_u32(&mut self, attr_raw: u32, value: u32) -> Result<()> {
        let f = fns()?;
        // Call through a `u32` view of the function pointer rather than
        // transmuting `attr_raw` into the enum: an undeclared discriminant
        // would be undefined behaviour, while `#[repr(u32)]` enum and `u32`
        // parameters share one C ABI.
        type RawSetAttr = unsafe extern "C" fn(
            sys::CublasLtMatmulDesc,
            u32,
            *const c_void,
            usize,
        ) -> sys::CublasStatus;
        let set: RawSetAttr = std::mem::transmute(f.cublasLtMatmulDescSetAttribute);
        check(
            "cublasLtMatmulDescSetAttribute(raw)",
            set(
                self.raw,
                attr_raw,
                &value as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            ),
        )
    }

    /// Set the device pointer that receives the amax of D. Wraps `cublasLtMatmulDescSetAttribute`.
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

    /// Set the device pointer to the epilogue bias vector. Wraps `cublasLtMatmulDescSetAttribute`.
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

    /// Raw `cublasLtMatmulDesc_t`.
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

/// Operand whose FP8 scale pointer is set by [`MatmulDesc::set_scale_pointer`].
#[derive(Copy, Clone, Debug)]
pub enum ScaleTensor {
    /// Scale for operand A.
    A,
    /// Scale for operand B.
    B,
    /// Scale for operand C.
    C,
    /// Scale for the output D.
    D,
}

// ─── Matrix layout ──────────────────────────────────────────────────────────

/// Owned `cublasLtMatrixLayout_t`. Destroyed on drop.
pub struct MatrixLayout {
    raw: sys::CublasLtMatrixLayout,
}

unsafe impl Send for MatrixLayout {}
unsafe impl Sync for MatrixLayout {}

impl MatrixLayout {
    /// Create a `rows` x `cols` layout of `dtype` with leading dimension `ld`. Wraps `cublasLtMatrixLayoutCreate`.
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

    /// Set the element order (row- or column-major). Wraps `cublasLtMatrixLayoutSetAttribute`.
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

    /// Set the batch count and the stride between batches. Wraps `cublasLtMatrixLayoutSetAttribute` for `BatchCount` and `StridedBatchOffset`.
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

    /// Raw `cublasLtMatrixLayout_t`.
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

/// Owned `cublasLtMatmulPreference_t` that constrains heuristic queries. Destroyed on drop.
pub struct Preference {
    raw: sys::CublasLtMatmulPreference,
}

unsafe impl Send for Preference {}
unsafe impl Sync for Preference {}

impl Preference {
    /// Create an empty matmul preference. Wraps `cublasLtMatmulPreferenceCreate`.
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

    /// Cap the workspace size in bytes that heuristics may request. Wraps `cublasLtMatmulPreferenceSetAttribute`.
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

    /// Raw `cublasLtMatmulPreference_t`.
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
