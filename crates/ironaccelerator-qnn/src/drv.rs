//! Safe QNN driver layer. Opens a backend library, creates a context, and
//! exposes graph construction + execute in an API that mirrors the CUDA/ROCm
//! `drv` modules.
//!
//! The layer is intentionally narrow. QNN's full op set is enormous; rather
//! than wrapping every operator as a typed Rust builder, we treat `OpConfig`
//! and `Tensor` as opaque byte buffers that the caller constructs from the
//! SDK headers. IronAccelerator's planner builds them via a small set of
//! helper builders (GEMM, softmax, layernorm, MHA) that layer on top.

use iron_qnn_sys::qnn as sys;
use iron_qnn_sys::qnn::{
    QnnInterfaceV2, Qnn_BackendHandle_t, Qnn_ContextHandle_t, Qnn_DeviceHandle_t,
    Qnn_ErrorHandle_t, Qnn_GraphHandle_t, Target, QNN_SUCCESS,
};
use iron_qnn_sys::LoadError;
use std::ffi::{c_void, CString};
use std::sync::Arc;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Error {
    NotAvailable {
        target: Target,
        detail: String,
    },
    Call {
        op: &'static str,
        code: Qnn_ErrorHandle_t,
    },
    Precondition {
        op: &'static str,
        msg: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable { target, detail } => {
                write!(f, "QNN {target:?} not available: {detail}")
            }
            Self::Call { op, code } => write!(f, "{op}: QNN error 0x{code:x}"),
            Self::Precondition { op, msg } => write!(f, "{op}: {msg}"),
        }
    }
}
impl std::error::Error for Error {}

impl Error {
    pub fn numeric(&self) -> i64 {
        match self {
            Self::NotAvailable { .. } => -1,
            Self::Call { code, .. } => *code as i64,
            Self::Precondition { .. } => -2,
        }
    }
}

impl From<&LoadError> for Error {
    fn from(e: &LoadError) -> Self {
        Error::NotAvailable {
            target: Target::Htp,
            detail: format!("{e}"),
        }
    }
}

impl From<Error> for ironaccelerator_core::Error {
    fn from(e: Error) -> Self {
        ironaccelerator_core::Error::Backend {
            backend: ironaccelerator_core::BackendKind::QualcommNpu,
            code: e.numeric(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

fn iface(target: Target) -> Result<&'static QnnInterfaceV2> {
    sys::fns(target)
        .map(|f| f.iface)
        .map_err(|e| Error::NotAvailable {
            target,
            detail: format!("{e}"),
        })
}

#[inline]
fn check(op: &'static str, code: Qnn_ErrorHandle_t) -> Result<()> {
    if code == QNN_SUCCESS {
        Ok(())
    } else {
        Err(Error::Call { op, code })
    }
}

// ── Backend ─────────────────────────────────────────────────────────────────

/// Handle to a loaded QNN backend (HTP, GPU, CPU, …). One per target per
/// process is enough; the safe wrapper caches them.
pub struct Backend {
    target: Target,
    handle: Qnn_BackendHandle_t,
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

impl Backend {
    pub fn new(target: Target) -> Result<Arc<Self>> {
        let i = iface(target)?;
        let mut h: Qnn_BackendHandle_t = std::ptr::null_mut();
        unsafe {
            check(
                "QnnBackend_create",
                (i.backend_create)(std::ptr::null_mut(), std::ptr::null(), &mut h),
            )?;
        }
        Ok(Arc::new(Self { target, handle: h }))
    }

    #[inline]
    pub fn target(&self) -> Target {
        self.target
    }
    #[inline]
    pub fn raw(&self) -> Qnn_BackendHandle_t {
        self.handle
    }

    pub fn api_version(&self) -> Result<sys::QnnApiVersion> {
        let i = iface(self.target)?;
        let mut v = sys::QnnApiVersion {
            core_major: 0,
            core_minor: 0,
            core_patch: 0,
            backend_major: 0,
            backend_minor: 0,
            backend_patch: 0,
        };
        unsafe {
            check(
                "QnnBackend_getApiVersion",
                (i.backend_getApiVersion)(&mut v),
            )?;
        }
        Ok(v)
    }

    pub fn provider_name(&self) -> Result<String> {
        Ok(sys::fns(self.target)
            .map_err(|e| Error::NotAvailable {
                target: self.target,
                detail: format!("{e}"),
            })?
            .provider_name
            .clone())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Ok(i) = iface(self.target) {
            unsafe {
                let _ = (i.backend_free)(self.handle);
            }
        }
    }
}

// ── Device ──────────────────────────────────────────────────────────────────

pub struct Device {
    backend: Arc<Backend>,
    handle: Qnn_DeviceHandle_t,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn new(backend: Arc<Backend>) -> Result<Arc<Self>> {
        let i = iface(backend.target)?;
        let mut h: Qnn_DeviceHandle_t = std::ptr::null_mut();
        unsafe {
            check(
                "QnnDevice_create",
                (i.device_create)(std::ptr::null_mut(), std::ptr::null(), &mut h),
            )?;
        }
        Ok(Arc::new(Self { backend, handle: h }))
    }

    #[inline]
    pub fn backend(&self) -> &Arc<Backend> {
        &self.backend
    }
    #[inline]
    pub fn raw(&self) -> Qnn_DeviceHandle_t {
        self.handle
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Ok(i) = iface(self.backend.target) {
            unsafe {
                let _ = (i.device_free)(self.handle);
            }
        }
    }
}

// ── Context ─────────────────────────────────────────────────────────────────

pub struct Context {
    device: Arc<Device>,
    handle: Qnn_ContextHandle_t,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>> {
        let i = iface(device.backend.target)?;
        let mut h: Qnn_ContextHandle_t = std::ptr::null_mut();
        unsafe {
            check(
                "QnnContext_create",
                (i.context_create)(
                    device.backend.handle,
                    device.handle,
                    std::ptr::null(),
                    &mut h,
                ),
            )?;
        }
        Ok(Arc::new(Self { device, handle: h }))
    }

    /// Rehydrate a context from a serialized binary blob (`.bin` produced by
    /// a prior `context_getBinary`).
    pub fn from_binary(device: Arc<Device>, blob: &[u8]) -> Result<Arc<Self>> {
        let i = iface(device.backend.target)?;
        let mut h: Qnn_ContextHandle_t = std::ptr::null_mut();
        unsafe {
            check(
                "QnnContext_createFromBinary",
                (i.context_createFromBinary)(
                    device.backend.handle,
                    device.handle,
                    std::ptr::null(),
                    blob.as_ptr() as *const c_void,
                    blob.len(),
                    &mut h,
                    std::ptr::null_mut(),
                ),
            )?;
        }
        Ok(Arc::new(Self { device, handle: h }))
    }

    /// Serialize this context to bytes for later rehydration.
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        let i = iface(self.device.backend.target)?;
        let mut size: usize = 0;
        unsafe {
            check(
                "QnnContext_getBinarySize",
                (i.context_getBinarySize)(self.handle, &mut size),
            )?;
        }
        let mut buf = vec![0u8; size];
        let mut written: usize = 0;
        unsafe {
            check(
                "QnnContext_getBinary",
                (i.context_getBinary)(
                    self.handle,
                    buf.as_mut_ptr() as *mut c_void,
                    size,
                    &mut written,
                ),
            )?;
        }
        buf.truncate(written);
        Ok(buf)
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    #[inline]
    pub fn raw(&self) -> Qnn_ContextHandle_t {
        self.handle
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Ok(i) = iface(self.device.backend.target) {
            unsafe {
                let _ = (i.context_free)(self.handle, std::ptr::null_mut());
            }
        }
    }
}

// ── Graph ───────────────────────────────────────────────────────────────────

pub struct Graph {
    context: Arc<Context>,
    handle: Qnn_GraphHandle_t,
    finalized: bool,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    pub fn new(context: Arc<Context>, name: &str) -> Result<Self> {
        let i = iface(context.device.backend.target)?;
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Graph::new",
            msg: "name contains NUL".into(),
        })?;
        let mut h: Qnn_GraphHandle_t = std::ptr::null_mut();
        unsafe {
            check(
                "QnnGraph_create",
                (i.graph_create)(context.handle, cname.as_ptr(), std::ptr::null(), &mut h),
            )?;
        }
        Ok(Self {
            context,
            handle: h,
            finalized: false,
        })
    }

    /// Add a node built by the caller. The `op_config` pointer must reference
    /// a fully-populated `Qnn_OpConfig_t` for the SDK version in use.
    ///
    /// # Safety
    /// The caller vouches for the layout and lifetimes of the underlying
    /// `OpConfig` and any tensors it references.
    pub unsafe fn add_node_raw(&mut self, op_config: *const sys::QnnOpConfig) -> Result<()> {
        if self.finalized {
            return Err(Error::Precondition {
                op: "Graph::add_node_raw",
                msg: "graph already finalized".into(),
            });
        }
        let i = iface(self.context.device.backend.target)?;
        unsafe {
            check(
                "QnnGraph_addNode",
                (i.graph_addNode)(self.handle, op_config),
            )
        }
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        let i = iface(self.context.device.backend.target)?;
        unsafe {
            check(
                "QnnGraph_finalize",
                (i.graph_finalize)(self.handle, std::ptr::null_mut(), std::ptr::null_mut()),
            )?;
        }
        self.finalized = true;
        Ok(())
    }

    /// Execute with caller-built tensor arrays. See `add_node_raw` safety
    /// notes — the tensor pointers must be correctly laid out.
    ///
    /// # Safety
    /// Inputs/outputs must point to properly constructed `Qnn_Tensor_t`
    /// buffers with host/device memory that outlives the call.
    pub unsafe fn execute_raw(
        &self,
        inputs: *const sys::QnnTensor,
        n_in: u32,
        outputs: *mut sys::QnnTensor,
        n_out: u32,
    ) -> Result<()> {
        if !self.finalized {
            return Err(Error::Precondition {
                op: "Graph::execute_raw",
                msg: "graph not finalized".into(),
            });
        }
        let i = iface(self.context.device.backend.target)?;
        unsafe {
            check(
                "QnnGraph_execute",
                (i.graph_execute)(
                    self.handle,
                    inputs,
                    n_in,
                    outputs,
                    n_out,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }

    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
    #[inline]
    pub fn raw(&self) -> Qnn_GraphHandle_t {
        self.handle
    }
    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

// ── availability ────────────────────────────────────────────────────────────

pub fn is_available(target: Target) -> bool {
    sys::is_available(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn availability_does_not_panic() {
        let _ = is_available(Target::Htp);
        let _ = is_available(Target::Cpu);
    }
}
