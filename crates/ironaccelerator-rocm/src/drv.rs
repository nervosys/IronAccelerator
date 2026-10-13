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
/// Errors from the safe HIP driver layer.
pub enum Error {
    /// The HIP runtime library could not be loaded.
    NotAvailable {
        /// Name of the library that failed to load.
        lib: &'static str,
        /// Loader error text.
        detail: String,
    },
    /// A HIP call returned a non-success code.
    Driver {
        /// Name of the HIP call that failed.
        op: &'static str,
        /// Raw `hipError_t` returned by the call.
        code: sys::HipResult,
    },
    /// A caller-side precondition was violated before any HIP call.
    Precondition {
        /// Name of the operation that was rejected.
        op: &'static str,
        /// Human-readable reason.
        msg: String,
    },
}

impl Error {
    #[inline]
    /// Name of the operation that produced this error.
    pub fn op(&self) -> &'static str {
        match self {
            Self::NotAvailable { .. } => "load",
            Self::Driver { op, .. } => op,
            Self::Precondition { op, .. } => op,
        }
    }
    #[inline]
    /// Numeric code: -1 unavailable, -2 precondition, otherwise the `hipError_t` value.
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
        Error::NotAvailable {
            lib: "hip",
            detail: format!("{e}"),
        }
    }
}

/// Result alias for HIP driver operations.
pub type Result<T> = std::result::Result<T, Error>;

#[inline(always)]
fn hip() -> Result<&'static sys::HipFns> {
    sys::fns().map_err(hip_load_err)
}

#[cold]
#[inline(never)]
fn hip_load_err(e: &'static LoadError) -> Error {
    Error::NotAvailable {
        lib: "hip",
        detail: format!("{e}"),
    }
}

#[inline(always)]
fn check(op: &'static str, r: sys::HipResult) -> Result<()> {
    if r.is_ok() {
        Ok(())
    } else {
        check_err(op, r)
    }
}

#[cold]
#[inline(never)]
fn check_err(op: &'static str, r: sys::HipResult) -> Result<()> {
    Err(Error::Driver { op, code: r })
}

#[cold]
#[inline(never)]
fn alloc_overflow() -> Error {
    Error::Precondition {
        op: "DeviceBuf::alloc",
        msg: "size overflow".into(),
    }
}

static INIT: OnceLock<Result<()>> = OnceLock::new();
fn ensure_init() -> Result<()> {
    INIT.get_or_init(|| unsafe {
        let f = hip()?;
        check("hipInit", (f.hipInit)(0))
    })
    .clone()
}

// ── Device ─────────────────────────────────────────────────────────────────

/// An opened HIP device. Device-scoped calls rely on [`Device::bind`].
pub struct Device {
    ordinal: i32,
    device: HipDevice,
    /// Cached driver function table — set at construction so every method on
    /// `Device` / `Stream` / `Event` skips the AtomicPtr load.
    drv: &'static sys::HipFns,
}

impl Device {
    /// Initialises HIP and opens the device at `ordinal` via `hipDeviceGet`.
    pub fn open(ordinal: u32) -> Result<Arc<Self>> {
        ensure_init()?;
        let f = hip()?;
        let mut d: HipDevice = 0;
        unsafe {
            check("hipDeviceGet", (f.hipDeviceGet)(&mut d, ordinal as i32))?;
        }
        unsafe {
            check("hipSetDevice", (f.hipSetDevice)(ordinal as i32))?;
        }
        Ok(Arc::new(Self {
            ordinal: ordinal as i32,
            device: d,
            drv: f,
        }))
    }

    /// Returns the number of HIP devices via `hipGetDeviceCount`.
    pub fn count() -> Result<u32> {
        ensure_init()?;
        let f = hip()?;
        let mut n: i32 = 0;
        unsafe {
            check("hipGetDeviceCount", (f.hipGetDeviceCount)(&mut n))?;
        }
        Ok(n.max(0) as u32)
    }

    #[inline]
    /// Returns the device ordinal.
    pub fn ordinal(&self) -> u32 {
        self.ordinal as u32
    }
    #[inline]
    /// Returns the raw `hipDevice_t` handle.
    pub fn raw(&self) -> HipDevice {
        self.device
    }

    #[inline]
    /// Makes this device current on the calling thread via `hipSetDevice`.
    pub fn bind(&self) -> Result<()> {
        unsafe { check("hipSetDevice", (self.drv.hipSetDevice)(self.ordinal)) }
    }

    /// Returns the device name via `hipDeviceGetName`.
    pub fn name(&self) -> Result<String> {
        let mut buf = vec![0 as c_char; 256];
        unsafe {
            check(
                "hipDeviceGetName",
                (self.drv.hipDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.device),
            )?;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    #[inline]
    /// Queries a device attribute via `hipDeviceGetAttribute`.
    pub fn attribute(&self, attr: sys::HipDeviceAttribute) -> Result<i32> {
        let mut v: i32 = 0;
        unsafe {
            check(
                "hipDeviceGetAttribute",
                (self.drv.hipDeviceGetAttribute)(&mut v, attr, self.device),
            )?;
        }
        Ok(v)
    }

    /// Returns total device memory in bytes via `hipDeviceTotalMem`.
    pub fn total_mem(&self) -> Result<usize> {
        let mut b: usize = 0;
        unsafe {
            check(
                "hipDeviceTotalMem",
                (self.drv.hipDeviceTotalMem)(&mut b, self.device),
            )?;
        }
        Ok(b)
    }

    /// Returns the `(major, minor)` compute capability from the two `ComputeCapability` attributes.
    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        let maj = self.attribute(sys::HipDeviceAttribute::ComputeCapabilityMajor)? as u32;
        let min = self.attribute(sys::HipDeviceAttribute::ComputeCapabilityMinor)? as u32;
        Ok((maj, min))
    }

    /// Returns whether this device can access `other` directly via `hipDeviceCanAccessPeer`.
    pub fn can_access_peer(&self, other: &Device) -> Result<bool> {
        let mut v: i32 = 0;
        unsafe {
            check(
                "hipDeviceCanAccessPeer",
                (self.drv.hipDeviceCanAccessPeer)(&mut v, self.device, other.device),
            )?;
        }
        Ok(v != 0)
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

// ── Stream ─────────────────────────────────────────────────────────────────

/// A non-blocking HIP stream (`hipStreamCreateWithPriority`) bound to a device.
pub struct Stream {
    device: Arc<Device>,
    handle: HipStream,
    drv: &'static sys::HipFns,
}

impl Stream {
    /// Creates a non-blocking stream on `device`.
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>> {
        device.bind()?;
        let f = device.drv;
        let mut h = HipStream::default();
        unsafe {
            check(
                "hipStreamCreateWithPriority",
                (f.hipStreamCreateWithPriority)(&mut h, sys::HIP_STREAM_NON_BLOCKING, 0),
            )?;
        }
        Ok(Arc::new(Self {
            device,
            handle: h,
            drv: f,
        }))
    }

    #[inline]
    /// Returns the device this stream runs on.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    #[inline]
    /// Returns the raw `hipStream_t` handle.
    pub fn raw(&self) -> HipStream {
        self.handle
    }

    #[inline]
    /// Blocks until queued work on this stream completes via `hipStreamSynchronize`.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check(
                "hipStreamSynchronize",
                (self.drv.hipStreamSynchronize)(self.handle),
            )
        }
    }

    #[inline]
    /// Makes subsequent work on this stream wait for `event` via `hipStreamWaitEvent`.
    pub fn wait_for(&self, event: &Event) -> Result<()> {
        unsafe {
            check(
                "hipStreamWaitEvent",
                (self.drv.hipStreamWaitEvent)(self.handle, event.handle, 0),
            )
        }
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.hipStreamDestroy)(self.handle);
        }
    }
}

// ── Event ──────────────────────────────────────────────────────────────────

/// A HIP event (`hipEventCreateWithFlags`) bound to a device.
pub struct Event {
    _device: Arc<Device>,
    handle: HipEvent,
    timing: bool,
    drv: &'static sys::HipFns,
}

impl Event {
    /// Creates an event with timing disabled.
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::new_flags(device, sys::HIP_EVENT_DISABLE_TIMING, false)
    }

    /// Creates an event with timing enabled, for use with [`Event::elapsed_ms`].
    pub fn new_timing(device: Arc<Device>) -> Result<Self> {
        Self::new_flags(device, 0, true)
    }

    fn new_flags(device: Arc<Device>, flags: u32, timing: bool) -> Result<Self> {
        device.bind()?;
        let f = device.drv;
        let mut h = HipEvent::default();
        unsafe {
            check(
                "hipEventCreateWithFlags",
                (f.hipEventCreateWithFlags)(&mut h, flags),
            )?;
        }
        Ok(Self {
            _device: device,
            handle: h,
            timing,
            drv: f,
        })
    }

    #[inline]
    /// Records this event on `stream` via `hipEventRecord`.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "hipEventRecord",
                (self.drv.hipEventRecord)(self.handle, stream.handle),
            )
        }
    }

    #[inline]
    /// Blocks until this event completes via `hipEventSynchronize`.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check(
                "hipEventSynchronize",
                (self.drv.hipEventSynchronize)(self.handle),
            )
        }
    }

    /// Returns the milliseconds between two timing events via `hipEventElapsedTime`. Errors if either lacks timing.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        if !start.timing || !end.timing {
            return Err(Error::Precondition {
                op: "Event::elapsed_ms",
                msg: "both events must be created with timing enabled".into(),
            });
        }
        let mut ms: f32 = 0.0;
        unsafe {
            check(
                "hipEventElapsedTime",
                (start.drv.hipEventElapsedTime)(&mut ms, start.handle, end.handle),
            )?;
        }
        Ok(ms)
    }
}

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.hipEventDestroy)(self.handle);
        }
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

// ── DeviceBuf ──────────────────────────────────────────────────────────────

/// Marker for plain element types that may live in device memory.
pub trait Repr: Copy + Send + Sync + 'static {}
impl Repr for u8 {}
impl Repr for u16 {}
impl Repr for u32 {}
impl Repr for u64 {}
impl Repr for i8 {}
impl Repr for i16 {}
impl Repr for i32 {}
impl Repr for i64 {}
impl Repr for f32 {}
impl Repr for f64 {}

/// Typed device allocation backed by `hipMallocAsync` on a stream.
pub struct DeviceBuf<T: Repr> {
    stream: Arc<Stream>,
    ptr: HipDeviceptr,
    len: usize,
    _m: PhantomData<T>,
}

impl<T: Repr> DeviceBuf<T> {
    #[inline]
    /// Allocates `len` elements via `hipMallocAsync`. A zero length allocates nothing.
    pub fn alloc(stream: Arc<Stream>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                stream,
                ptr: 0,
                len: 0,
                _m: PhantomData,
            });
        }
        // Cold-path the overflow error so the success path stays alloc-free.
        let Some(bytes) = len.checked_mul(std::mem::size_of::<T>()) else {
            return Err(alloc_overflow());
        };
        let mut raw: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(
                "hipMallocAsync",
                (stream.drv.hipMallocAsync)(&mut raw, bytes, stream.handle),
            )?;
        }
        Ok(Self {
            stream,
            ptr: raw as HipDeviceptr,
            len,
            _m: PhantomData,
        })
    }

    #[inline]
    /// Copies `src` to the device via `hipMemcpyHtoDAsync`. `src.len()` must equal the buffer length.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_host",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "hipMemcpyHtoDAsync",
                (self.stream.drv.hipMemcpyHtoDAsync)(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    bytes,
                    self.stream.handle,
                ),
            )
        }
    }

    #[inline]
    /// Copies the device buffer into `dst` via `hipMemcpyDtoHAsync`. `dst.len()` must equal the buffer length.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_to_host",
                msg: format!("length mismatch: src={} dst={}", self.len, dst.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "hipMemcpyDtoHAsync",
                (self.stream.drv.hipMemcpyDtoHAsync)(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    self.stream.handle,
                ),
            )
        }
    }

    #[inline]
    /// Returns the element count.
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    /// Returns `true` if the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    /// Returns the size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    #[inline]
    /// Returns the raw device pointer.
    pub fn device_ptr(&self) -> HipDeviceptr {
        self.ptr
    }
    #[inline]
    /// Returns the stream this buffer is allocated on.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

impl<T: Repr> Drop for DeviceBuf<T> {
    #[inline]
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        unsafe {
            let _ = (self.stream.drv.hipFreeAsync)(self.ptr as *mut c_void, self.stream.handle);
        }
    }
}

unsafe impl<T: Repr> Send for DeviceBuf<T> {}
unsafe impl<T: Repr> Sync for DeviceBuf<T> {}

// ── Module + kernel launch ─────────────────────────────────────────────────

/// A loaded HIP module (`hipModuleLoadData`). Unloaded on drop.
pub struct Module {
    _device: Arc<Device>,
    handle: HipModule,
    drv: &'static sys::HipFns,
}

impl Module {
    /// Loads a code object from `image` via `hipModuleLoadData`.
    pub fn load(device: Arc<Device>, image: &[u8]) -> Result<Arc<Self>> {
        device.bind()?;
        let f = device.drv;
        let mut h = HipModule::default();
        unsafe {
            check(
                "hipModuleLoadData",
                (f.hipModuleLoadData)(&mut h, image.as_ptr() as *const c_void),
            )?;
        }
        Ok(Arc::new(Self {
            _device: device,
            handle: h,
            drv: f,
        }))
    }

    /// Looks up the kernel `name` via `hipModuleGetFunction`.
    pub fn function(self: &Arc<Self>, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Module::function",
            msg: "name contains NUL".into(),
        })?;
        let mut fun = HipFunction::default();
        unsafe {
            check(
                "hipModuleGetFunction",
                (self.drv.hipModuleGetFunction)(&mut fun, self.handle, cname.as_ptr()),
            )?;
        }
        Ok(Function {
            _module: self.clone(),
            handle: fun,
            drv: self.drv,
        })
    }
}

impl Drop for Module {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.hipModuleUnload)(self.handle);
        }
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// A kernel handle from a [`Module`], launchable via `hipModuleLaunchKernel`.
pub struct Function {
    _module: Arc<Module>,
    handle: HipFunction,
    drv: &'static sys::HipFns,
}

/// Grid, block and dynamic shared-memory configuration for a launch.
pub struct LaunchCfg {
    /// Grid dimensions in blocks (x, y, z).
    pub grid: (u32, u32, u32),
    /// Block dimensions in threads (x, y, z).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory per block, in bytes.
    pub shared_mem: u32,
}
impl LaunchCfg {
    /// Builds a 1-D launch with `grid` blocks of `block` threads and no shared memory.
    pub fn linear(grid: u32, block: u32) -> Self {
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem: 0,
        }
    }
}

impl Function {
    /// Launch with raw pointer-to-pointer argv. Caller is responsible for
    /// building the correct argv layout (ROCm uses the same extra-params
    /// scheme as CUDA's `cuLaunchKernel`).
    ///
    /// # Safety
    /// `argv` must point to valid argument slots for the kernel.
    #[inline]
    pub unsafe fn launch_raw(
        &self,
        cfg: LaunchCfg,
        stream: &Stream,
        argv: *mut *mut c_void,
    ) -> Result<()> {
        unsafe {
            check(
                "hipModuleLaunchKernel",
                (self.drv.hipModuleLaunchKernel)(
                    self.handle,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_mem,
                    stream.raw(),
                    argv,
                    std::ptr::null_mut(),
                ),
            )
        }
    }
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

// ── availability probe ────────────────────────────────────────────────────

/// Returns `true` if the HIP runtime library loads.
pub fn is_available() -> bool {
    sys::is_available()
}

impl From<Error> for ironaccelerator_core::Error {
    fn from(e: Error) -> Self {
        ironaccelerator_core::Error::Backend {
            backend: ironaccelerator_core::BackendKind::Rocm,
            code: e.numeric(),
        }
    }
}
