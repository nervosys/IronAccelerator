//! Safe HIP driver layer. Parity with `ironaccelerator_cuda::drv`, minus the
//! primary-context dance: HIP manages a per-thread current device implicitly.

use iron_rocm_sys::hip as sys;
use iron_rocm_sys::hip::{HipDevice, HipDeviceptr, HipEvent, HipFunction, HipModule, HipStream};
use iron_rocm_sys::loader::LoadError;
use std::ffi::{c_char, c_void, CString};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

// ── error helpers ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Error {
    NotAvailable { lib: &'static str, detail: String },
    Driver { op: &'static str, code: sys::HipResult },
    Precondition { op: &'static str, msg: String },
}

impl Error {
    #[inline]
    pub fn op(&self) -> &'static str {
        match self {
            Self::NotAvailable { .. } => "load",
            Self::Driver { op, .. } => op,
            Self::Precondition { op, .. } => op,
        }
    }
    #[inline]
    pub fn numeric(&self) -> i64 {
        match self {
            Self::NotAvailable { .. } => -1,
            Self::Driver { code, .. } => (*code as u32) as i64,
            Self::Precondition { .. } => -2,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable { lib, detail } => write!(f, "{lib} not available: {detail}"),
            Self::Driver { op, code } => write!(f, "{op}: HIP error {code:?}"),
            Self::Precondition { op, msg } => write!(f, "{op}: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<&LoadError> for Error {
    fn from(e: &LoadError) -> Self {
        Error::NotAvailable { lib: "hip", detail: format!("{e}") }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

fn hip() -> Result<&'static sys::HipFns> {
    sys::fns().map_err(|e| Error::NotAvailable { lib: "hip", detail: format!("{e}") })
}

#[inline]
fn check(op: &'static str, r: sys::HipResult) -> Result<()> {
    if r.is_ok() { Ok(()) } else { Err(Error::Driver { op, code: r }) }
}

static INIT: OnceLock<Result<()>> = OnceLock::new();
fn ensure_init() -> Result<()> {
    INIT.get_or_init(|| unsafe {
        let f = hip()?;
        check("hipInit", (f.hipInit)(0))
    }).clone()
}

// ── Device ─────────────────────────────────────────────────────────────────

pub struct Device {
    ordinal: i32,
    device: HipDevice,
}

impl Device {
    pub fn open(ordinal: u32) -> Result<Arc<Self>> {
        ensure_init()?;
        let f = hip()?;
        let mut d: HipDevice = 0;
        unsafe { check("hipDeviceGet", (f.hipDeviceGet)(&mut d, ordinal as i32))?; }
        unsafe { check("hipSetDevice", (f.hipSetDevice)(ordinal as i32))?; }
        Ok(Arc::new(Self { ordinal: ordinal as i32, device: d }))
    }

    pub fn count() -> Result<u32> {
        ensure_init()?;
        let f = hip()?;
        let mut n: i32 = 0;
        unsafe { check("hipGetDeviceCount", (f.hipGetDeviceCount)(&mut n))?; }
        Ok(n.max(0) as u32)
    }

    #[inline] pub fn ordinal(&self) -> u32 { self.ordinal as u32 }
    #[inline] pub fn raw(&self) -> HipDevice { self.device }

    pub fn bind(&self) -> Result<()> {
        let f = hip()?;
        unsafe { check("hipSetDevice", (f.hipSetDevice)(self.ordinal)) }
    }

    pub fn name(&self) -> Result<String> {
        let f = hip()?;
        let mut buf = vec![0 as c_char; 256];
        unsafe {
            check("hipDeviceGetName",
                  (f.hipDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.device))?;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn attribute(&self, attr: sys::HipDeviceAttribute) -> Result<i32> {
        let f = hip()?;
        let mut v: i32 = 0;
        unsafe { check("hipDeviceGetAttribute",
                       (f.hipDeviceGetAttribute)(&mut v, attr, self.device))?; }
        Ok(v)
    }

    pub fn total_mem(&self) -> Result<usize> {
        let f = hip()?;
        let mut b: usize = 0;
        unsafe { check("hipDeviceTotalMem", (f.hipDeviceTotalMem)(&mut b, self.device))?; }
        Ok(b)
    }

    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        let maj = self.attribute(sys::HipDeviceAttribute::ComputeCapabilityMajor)? as u32;
        let min = self.attribute(sys::HipDeviceAttribute::ComputeCapabilityMinor)? as u32;
        Ok((maj, min))
    }

    pub fn can_access_peer(&self, other: &Device) -> Result<bool> {
        let f = hip()?;
        let mut v: i32 = 0;
        unsafe { check("hipDeviceCanAccessPeer",
                       (f.hipDeviceCanAccessPeer)(&mut v, self.device, other.device))?; }
        Ok(v != 0)
    }
}

unsafe impl Send for Device {} unsafe impl Sync for Device {}

// ── Stream ─────────────────────────────────────────────────────────────────

pub struct Stream {
    device: Arc<Device>,
    handle: HipStream,
}

impl Stream {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = hip()?;
        let mut h = HipStream::default();
        unsafe {
            check("hipStreamCreateWithPriority",
                  (f.hipStreamCreateWithPriority)(&mut h, sys::HIP_STREAM_NON_BLOCKING, 0))?;
        }
        Ok(Arc::new(Self { device, handle: h }))
    }

    #[inline] pub fn device(&self) -> &Arc<Device> { &self.device }
    #[inline] pub fn raw(&self) -> HipStream { self.handle }

    pub fn synchronize(&self) -> Result<()> {
        let f = hip()?;
        unsafe { check("hipStreamSynchronize", (f.hipStreamSynchronize)(self.handle)) }
    }

    pub fn wait_for(&self, event: &Event) -> Result<()> {
        let f = hip()?;
        unsafe { check("hipStreamWaitEvent", (f.hipStreamWaitEvent)(self.handle, event.handle, 0)) }
    }
}

unsafe impl Send for Stream {} unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(f) = hip() {
            unsafe { let _ = (f.hipStreamDestroy)(self.handle); }
        }
    }
}

// ── Event ──────────────────────────────────────────────────────────────────

pub struct Event {
    _device: Arc<Device>,
    handle: HipEvent,
    timing: bool,
}

impl Event {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::new_flags(device, sys::HIP_EVENT_DISABLE_TIMING, false)
    }

    pub fn new_timing(device: Arc<Device>) -> Result<Self> {
        Self::new_flags(device, 0, true)
    }

    fn new_flags(device: Arc<Device>, flags: u32, timing: bool) -> Result<Self> {
        device.bind()?;
        let f = hip()?;
        let mut h = HipEvent::default();
        unsafe { check("hipEventCreateWithFlags",
                       (f.hipEventCreateWithFlags)(&mut h, flags))?; }
        Ok(Self { _device: device, handle: h, timing })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        let f = hip()?;
        unsafe { check("hipEventRecord", (f.hipEventRecord)(self.handle, stream.handle)) }
    }

    pub fn synchronize(&self) -> Result<()> {
        let f = hip()?;
        unsafe { check("hipEventSynchronize", (f.hipEventSynchronize)(self.handle)) }
    }

    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        if !start.timing || !end.timing {
            return Err(Error::Precondition {
                op: "Event::elapsed_ms",
                msg: "both events must be created with timing enabled".into(),
            });
        }
        let f = hip()?;
        let mut ms: f32 = 0.0;
        unsafe { check("hipEventElapsedTime",
                       (f.hipEventElapsedTime)(&mut ms, start.handle, end.handle))?; }
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(f) = hip() {
            unsafe { let _ = (f.hipEventDestroy)(self.handle); }
        }
    }
}

unsafe impl Send for Event {} unsafe impl Sync for Event {}

// ── DeviceBuf ──────────────────────────────────────────────────────────────

pub trait Repr: Copy + Send + Sync + 'static {}
impl Repr for u8 {} impl Repr for u16 {} impl Repr for u32 {} impl Repr for u64 {}
impl Repr for i8 {} impl Repr for i16 {} impl Repr for i32 {} impl Repr for i64 {}
impl Repr for f32 {} impl Repr for f64 {}

pub struct DeviceBuf<T: Repr> {
    stream: Arc<Stream>,
    ptr: HipDeviceptr,
    len: usize,
    _m: PhantomData<T>,
}

impl<T: Repr> DeviceBuf<T> {
    pub fn alloc(stream: Arc<Stream>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self { stream, ptr: 0, len: 0, _m: PhantomData });
        }
        let f = hip()?;
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or(Error::Precondition {
            op: "DeviceBuf::alloc", msg: "size overflow".into(),
        })?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        unsafe { check("hipMallocAsync",
                       (f.hipMallocAsync)(&mut raw, bytes, stream.handle))?; }
        Ok(Self { stream, ptr: raw as HipDeviceptr, len, _m: PhantomData })
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_host",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len()),
            });
        }
        if self.len == 0 { return Ok(()); }
        let f = hip()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check("hipMemcpyHtoDAsync",
                  (f.hipMemcpyHtoDAsync)(self.ptr, src.as_ptr() as *const c_void,
                                         bytes, self.stream.handle))
        }
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_to_host",
                msg: format!("length mismatch: src={} dst={}", self.len, dst.len()),
            });
        }
        if self.len == 0 { return Ok(()); }
        let f = hip()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check("hipMemcpyDtoHAsync",
                  (f.hipMemcpyDtoHAsync)(dst.as_mut_ptr() as *mut c_void,
                                         self.ptr, bytes, self.stream.handle))
        }
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }
    #[inline] pub fn device_ptr(&self) -> HipDeviceptr { self.ptr }
    #[inline] pub fn stream(&self) -> &Arc<Stream> { &self.stream }
}

impl<T: Repr> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if self.len == 0 { return; }
        if let Ok(f) = hip() {
            unsafe { let _ = (f.hipFreeAsync)(self.ptr as *mut c_void, self.stream.handle); }
        }
    }
}

unsafe impl<T: Repr> Send for DeviceBuf<T> {} unsafe impl<T: Repr> Sync for DeviceBuf<T> {}

// ── Module + kernel launch ─────────────────────────────────────────────────

pub struct Module {
    _device: Arc<Device>,
    handle: HipModule,
}

impl Module {
    pub fn load(device: Arc<Device>, image: &[u8]) -> Result<Arc<Self>> {
        device.bind()?;
        let f = hip()?;
        let mut h = HipModule::default();
        unsafe { check("hipModuleLoadData",
                       (f.hipModuleLoadData)(&mut h, image.as_ptr() as *const c_void))?; }
        Ok(Arc::new(Self { _device: device, handle: h }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<Function> {
        let f = hip()?;
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Module::function", msg: "name contains NUL".into(),
        })?;
        let mut fun = HipFunction::default();
        unsafe { check("hipModuleGetFunction",
                       (f.hipModuleGetFunction)(&mut fun, self.handle, cname.as_ptr()))?; }
        Ok(Function { _module: self.clone(), handle: fun })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(f) = hip() {
            unsafe { let _ = (f.hipModuleUnload)(self.handle); }
        }
    }
}

unsafe impl Send for Module {} unsafe impl Sync for Module {}

pub struct Function {
    _module: Arc<Module>,
    handle: HipFunction,
}

pub struct LaunchCfg {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}
impl LaunchCfg {
    pub fn linear(grid: u32, block: u32) -> Self {
        Self { grid: (grid, 1, 1), block: (block, 1, 1), shared_mem: 0 }
    }
}

impl Function {
    /// Launch with raw pointer-to-pointer argv. Caller is responsible for
    /// building the correct argv layout (ROCm uses the same extra-params
    /// scheme as CUDA's `cuLaunchKernel`).
    ///
    /// # Safety
    /// `argv` must point to valid argument slots for the kernel.
    pub unsafe fn launch_raw(
        &self, cfg: LaunchCfg, stream: &Stream, argv: *mut *mut c_void,
    ) -> Result<()> {
        let f = hip()?;
        unsafe {
            check("hipModuleLaunchKernel", (f.hipModuleLaunchKernel)(
                self.handle,
                cfg.grid.0, cfg.grid.1, cfg.grid.2,
                cfg.block.0, cfg.block.1, cfg.block.2,
                cfg.shared_mem, stream.raw(),
                argv, std::ptr::null_mut(),
            ))
        }
    }
}

unsafe impl Send for Function {} unsafe impl Sync for Function {}

// ── availability probe ────────────────────────────────────────────────────

pub fn is_available() -> bool { sys::is_available() }

impl From<Error> for ironaccelerator_core::Error {
    fn from(e: Error) -> Self {
        ironaccelerator_core::Error::Backend {
            backend: ironaccelerator_core::BackendKind::Rocm,
            code: e.numeric(),
        }
    }
}
