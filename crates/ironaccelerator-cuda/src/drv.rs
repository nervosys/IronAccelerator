//! Safe CUDA driver layer built directly on `iron_cuda_sys`.
//!
//! This is the foundation for every higher-level module in this crate. It
//! replaces the cudarc surface we used to consume with a purpose-built API
//! aimed at three things IronAccelerator cares about:
//!
//! 1. **No hidden refcount traffic on the hot path.** Tensors borrow `&Stream`
//!    directly. An `Arc<Stream>` is available when you need cross-thread sharing,
//!    but kernel launches and memcpys don't touch it.
//! 2. **Compile-time launch-arg packing.** `LaunchArgs` is a trait implemented
//!    for tuples up to 16; no `Vec<Box<dyn>>` per launch.
//! 3. **Explicit timing.** `Event` defaults to timing-disabled (the fast path).
//!    Opt-in to timing with `TimingEvent`. Two-phase resolution: record at
//!    launch time, resolve at a later sync point.
//!
//! The layer is intentionally narrow — just the primitives the rest of the
//! crate needs. Library-specific handles (cuBLASLt, cuDNN, cuFFT, …) live in
//! their own modules on top of this.

use iron_cuda_sys::driver as sys;
use iron_cuda_sys::driver::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUevent_flags, CUfunction, CUgraph, CUgraphExec,
    CUmodule, CUresult, CUstream, CUstreamCaptureMode,
};
use iron_cuda_sys::loader::LoadError;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

// ════════════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════════════

/// Structured CUDA error with op-level context.
#[derive(Debug, Clone)]
pub enum Error {
    /// Driver/runtime library was not loaded (missing .so / .dll).
    NotAvailable { lib: &'static str, detail: String },
    /// A driver API call returned a non-Success status.
    Driver { op: &'static str, code: CUresult },
    /// A host-side precondition was violated (wrong length, bad UTF-8, …).
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
    /// Numeric code suitable for `ironaccelerator_core::Error::Backend.code`.
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
            Self::Driver { op, code } => write!(f, "{op}: CUDA error {code:?}"),
            Self::Precondition { op, msg } => write!(f, "{op}: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<&LoadError> for Error {
    fn from(e: &LoadError) -> Self {
        Error::NotAvailable { lib: "cuda-driver", detail: format!("{e}") }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[inline]
fn driver() -> Result<&'static sys::DriverFns> {
    sys::fns().map_err(|e| Error::NotAvailable { lib: "cuda-driver", detail: format!("{e}") })
}

#[inline]
fn check(op: &'static str, code: CUresult) -> Result<()> {
    if code.is_ok() { Ok(()) } else { Err(Error::Driver { op, code }) }
}

// ════════════════════════════════════════════════════════════════════════════
// One-shot driver init
// ════════════════════════════════════════════════════════════════════════════

static INIT: OnceLock<Result<()>> = OnceLock::new();

fn ensure_init() -> Result<()> {
    INIT.get_or_init(|| unsafe {
        let d = driver()?;
        check("cuInit", (d.cuInit)(0))
    }).clone()
}

// ════════════════════════════════════════════════════════════════════════════
// Device — primary context retained for the ordinal
// ════════════════════════════════════════════════════════════════════════════

/// A retained primary context. Dropping releases exactly one reference.
///
/// Process-wide the driver reference-counts the primary context, so obtaining
/// two `Arc<Device>` for the same ordinal is safe and cheap.
pub struct Device {
    ordinal: i32,
    device: CUdevice,
    ctx: CUcontext,
}

impl Device {
    /// Retain the primary context for `ordinal`. Lazily initialises the driver.
    pub fn open(ordinal: u32) -> Result<Arc<Self>> {
        ensure_init()?;
        let d = driver()?;
        let mut device: CUdevice = 0;
        unsafe { check("cuDeviceGet", (d.cuDeviceGet)(&mut device, ordinal as i32))?; }
        let mut ctx = CUcontext::default();
        unsafe { check("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, device))?; }
        Ok(Arc::new(Self { ordinal: ordinal as i32, device, ctx }))
    }

    pub fn count() -> Result<u32> {
        ensure_init()?;
        let d = driver()?;
        let mut n: i32 = 0;
        unsafe { check("cuDeviceGetCount", (d.cuDeviceGetCount)(&mut n))?; }
        Ok(n.max(0) as u32)
    }

    #[inline] pub fn ordinal(&self) -> u32 { self.ordinal as u32 }
    #[inline] pub fn raw_ctx(&self) -> CUcontext { self.ctx }
    #[inline] pub fn raw_device(&self) -> CUdevice { self.device }

    /// Bind this primary context to the calling thread. Required before any
    /// call that reads the "current context" (most driver functions do).
    pub fn bind(&self) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuCtxSetCurrent", (d.cuCtxSetCurrent)(self.ctx)) }
    }

    pub fn name(&self) -> Result<String> {
        let d = driver()?;
        let mut buf = vec![0i8; 256];
        unsafe {
            check("cuDeviceGetName",
                  (d.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.device))?;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn attribute(&self, attr: sys::CUdevice_attribute) -> Result<i32> {
        let d = driver()?;
        let mut v: i32 = 0;
        unsafe { check("cuDeviceGetAttribute",
                       (d.cuDeviceGetAttribute)(&mut v, attr, self.device))?; }
        Ok(v)
    }

    pub fn total_mem(&self) -> Result<usize> {
        let d = driver()?;
        let mut bytes: usize = 0;
        unsafe { check("cuDeviceTotalMem_v2",
                       (d.cuDeviceTotalMem_v2)(&mut bytes, self.device))?; }
        Ok(bytes)
    }

    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        let maj = self.attribute(sys::CUdevice_attribute::ComputeCapabilityMajor)? as u32;
        let min = self.attribute(sys::CUdevice_attribute::ComputeCapabilityMinor)? as u32;
        Ok((maj, min))
    }

    pub fn can_access_peer(&self, other: &Device) -> Result<bool> {
        let d = driver()?;
        let mut v: i32 = 0;
        unsafe { check("cuDeviceCanAccessPeer",
                       (d.cuDeviceCanAccessPeer)(&mut v, self.device, other.device))?; }
        Ok(v != 0)
    }

    pub fn enable_peer_access(&self, other: &Device) -> Result<()> {
        self.bind()?;
        let d = driver()?;
        let code = unsafe { (d.cuCtxEnablePeerAccess)(other.ctx, 0) };
        // Already-enabled is not an error for us.
        if code == CUresult::Success || code == CUresult::PeerAccessAlreadyEnabled {
            Ok(())
        } else {
            Err(Error::Driver { op: "cuCtxEnablePeerAccess", code })
        }
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuDevicePrimaryCtxRelease_v2)(self.device); }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Stream
// ════════════════════════════════════════════════════════════════════════════

/// Priority hint for stream creation. Mapped to the driver's priority range.
#[derive(Copy, Clone, Debug)]
pub enum Priority {
    Default,
    /// Most-negative priority (highest). Useful for latency-sensitive work.
    High,
    /// Least-negative priority (lowest). Background work.
    Low,
    /// Raw driver priority. Clamped to the device's legal range.
    Raw(i32),
}

pub struct Stream {
    device: Arc<Device>,
    handle: CUstream,
    priority: i32,
}

impl Stream {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>> {
        Self::with_priority(device, Priority::Default)
    }

    pub fn with_priority(device: Arc<Device>, pri: Priority) -> Result<Arc<Self>> {
        device.bind()?;
        let d = driver()?;
        let (lo, hi) = {
            let mut lo = 0i32; let mut hi = 0i32;
            unsafe { check("cuStreamGetPriorityRange",
                           (d.cuStreamGetPriorityRange)(&mut lo, &mut hi))?; }
            (lo, hi)
        };
        let priority = match pri {
            Priority::Default => 0,
            Priority::High => lo,
            Priority::Low => hi,
            Priority::Raw(v) => v.clamp(lo, hi),
        };
        let mut handle = CUstream::default();
        unsafe {
            check("cuStreamCreateWithPriority",
                  (d.cuStreamCreateWithPriority)(&mut handle, sys::CU_STREAM_NON_BLOCKING, priority))?;
        }
        Ok(Arc::new(Self { device, handle, priority }))
    }

    #[inline] pub fn device(&self) -> &Arc<Device> { &self.device }
    #[inline] pub fn raw(&self) -> CUstream { self.handle }
    #[inline] pub fn priority(&self) -> i32 { self.priority }

    pub fn synchronize(&self) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuStreamSynchronize", (d.cuStreamSynchronize)(self.handle)) }
    }

    /// Make this stream block until `event` completes on whatever stream
    /// recorded it.
    pub fn wait_for(&self, event: &Event) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuStreamWaitEvent",
                       (d.cuStreamWaitEvent)(self.handle, event.handle, 0)) }
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuStreamDestroy_v2)(self.handle); }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Events
// ════════════════════════════════════════════════════════════════════════════

/// Fast event — timing disabled, suitable for stream-ordered fences.
pub struct Event {
    device: Arc<Device>,
    handle: CUevent,
    timing: bool,
}

impl Event {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::new_impl(device, CUevent_flags::DisableTiming, false)
    }
    fn new_impl(device: Arc<Device>, flags: CUevent_flags, timing: bool) -> Result<Self> {
        device.bind()?;
        let d = driver()?;
        let mut handle = CUevent::default();
        unsafe { check("cuEventCreate", (d.cuEventCreate)(&mut handle, flags as u32))?; }
        Ok(Self { device, handle, timing })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuEventRecord", (d.cuEventRecord)(self.handle, stream.handle)) }
    }

    pub fn synchronize(&self) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuEventSynchronize", (d.cuEventSynchronize)(self.handle)) }
    }

    #[inline] pub fn raw(&self) -> CUevent { self.handle }
    #[inline] pub fn supports_timing(&self) -> bool { self.timing }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuEventDestroy_v2)(self.handle); }
        }
    }
}

/// Timing-enabled event. Call [`TimingEvent::elapsed_ms`] with a `start` and
/// `end` pair; both must be synchronised before reading the elapsed time.
pub struct TimingEvent(Event);

impl TimingEvent {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Event::new_impl(device, CUevent_flags::Default, true).map(Self)
    }
    #[inline] pub fn record(&self, stream: &Stream) -> Result<()> { self.0.record(stream) }
    #[inline] pub fn synchronize(&self) -> Result<()> { self.0.synchronize() }
    #[inline] pub fn as_event(&self) -> &Event { &self.0 }

    /// Milliseconds elapsed between `start` and `end`. Both events must have
    /// completed (call `.synchronize()` on the later one first).
    pub fn elapsed_ms(start: &TimingEvent, end: &TimingEvent) -> Result<f32> {
        let d = driver()?;
        let mut ms: f32 = 0.0;
        unsafe { check("cuEventElapsedTime",
                       (d.cuEventElapsedTime)(&mut ms, start.0.handle, end.0.handle))?; }
        Ok(ms)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Device memory buffers
// ════════════════════════════════════════════════════════════════════════════

/// Marker: `T` is safe to bit-copy to and from device memory. Implement for
/// POD types only; auto-impl covers the primitives we need.
///
/// # Safety
/// `T` must not contain references, pointers, `Box`, `Arc`, etc.
pub unsafe trait Repr: Copy + Send + Sync + 'static {}

macro_rules! impl_repr {
    ($($t:ty),*) => { $(unsafe impl Repr for $t {})* };
}
impl_repr!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);

/// Marker: the all-zero bit pattern is a valid value of `T`. Required for
/// `alloc_zeros`.
///
/// # Safety
/// `std::mem::zeroed::<T>()` must be sound.
pub unsafe trait ZeroBits: Repr {}
macro_rules! impl_zb {
    ($($t:ty),*) => { $(unsafe impl ZeroBits for $t {})* };
}
impl_zb!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);

/// Owned device buffer. Allocated on a stream's async-alloc pool; freed on the
/// same stream when dropped.
pub struct DeviceBuf<T: Repr> {
    stream: Arc<Stream>,
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Repr> DeviceBuf<T> {
    /// Allocate `len` elements, leaving the memory uninitialised.
    pub fn alloc(stream: Arc<Stream>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self { stream, ptr: 0, len: 0, _marker: PhantomData });
        }
        let d = driver()?;
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or(Error::Precondition {
            op: "DeviceBuf::alloc", msg: "size overflow".into(),
        })?;
        let mut ptr: CUdeviceptr = 0;
        unsafe { check("cuMemAllocAsync",
                       (d.cuMemAllocAsync)(&mut ptr, bytes, stream.handle))?; }
        Ok(Self { stream, ptr, len, _marker: PhantomData })
    }

    pub fn alloc_zeros(stream: Arc<Stream>, len: usize) -> Result<Self> where T: ZeroBits {
        let buf = Self::alloc(stream, len)?;
        if buf.len > 0 {
            let d = driver()?;
            let bytes = buf.len * std::mem::size_of::<T>();
            unsafe { check("cuMemsetD8Async",
                           (d.cuMemsetD8Async)(buf.ptr, 0, bytes, buf.stream.handle))?; }
        }
        Ok(buf)
    }

    /// Allocate and copy a host slice in one stream-ordered operation.
    pub fn from_host(stream: Arc<Stream>, src: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(stream, src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_host",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len()),
            });
        }
        if self.len == 0 { return Ok(()); }
        let d = driver()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe { check("cuMemcpyHtoDAsync_v2",
                       (d.cuMemcpyHtoDAsync_v2)(self.ptr, src.as_ptr() as *const c_void,
                                                bytes, self.stream.handle))?; }
        Ok(())
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_to_host",
                msg: format!("length mismatch: src={} dst={}", self.len, dst.len()),
            });
        }
        if self.len == 0 { return Ok(()); }
        let d = driver()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe { check("cuMemcpyDtoHAsync_v2",
                       (d.cuMemcpyDtoHAsync_v2)(dst.as_mut_ptr() as *mut c_void, self.ptr,
                                                bytes, self.stream.handle))?; }
        Ok(())
    }

    /// Device-to-device copy. Both buffers must be on the same device; we pick
    /// `self`'s stream for ordering.
    pub fn copy_from_device(&mut self, src: &DeviceBuf<T>) -> Result<()> {
        if src.len != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_device",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len),
            });
        }
        if self.len == 0 { return Ok(()); }
        let d = driver()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe { check("cuMemcpyDtoDAsync_v2",
                       (d.cuMemcpyDtoDAsync_v2)(self.ptr, src.ptr, bytes, self.stream.handle))?; }
        Ok(())
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }
    #[inline] pub fn device_ptr(&self) -> CUdeviceptr { self.ptr }
    #[inline] pub fn stream(&self) -> &Arc<Stream> { &self.stream }

    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }

    /// Reinterpret the byte payload as a different POD element type. The caller
    /// asserts the byte pattern is valid for `U`.
    ///
    /// # Safety
    /// `byte_len` must be a multiple of `size_of::<U>()`; the bytes must be a
    /// valid representation of `[U; _]`.
    pub unsafe fn transmute<U: Repr>(self) -> Result<DeviceBuf<U>> {
        let bytes = self.byte_len();
        let us = std::mem::size_of::<U>();
        if us == 0 || bytes % us != 0 {
            return Err(Error::Precondition {
                op: "DeviceBuf::transmute",
                msg: format!("{} bytes not divisible by sizeof<U>={}", bytes, us),
            });
        }
        let out = DeviceBuf::<U> {
            stream: self.stream.clone(),
            ptr: self.ptr,
            len: bytes / us,
            _marker: PhantomData,
        };
        std::mem::forget(self);
        Ok(out)
    }
}

impl<T: Repr> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Ok(d) = driver() {
                unsafe { let _ = (d.cuMemFreeAsync)(self.ptr, self.stream.handle); }
            }
        }
    }
}

unsafe impl<T: Repr> Send for DeviceBuf<T> {}
unsafe impl<T: Repr> Sync for DeviceBuf<T> {}

/// Non-owning immutable view. Lifetime-bound to the owning buffer.
#[derive(Copy, Clone)]
pub struct DeviceView<'a, T: Repr> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: Repr> DeviceView<'a, T> {
    #[inline] pub fn device_ptr(&self) -> CUdeviceptr { self.ptr }
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }

    /// Narrow to `[offset .. offset + len]`. Panics on out-of-bounds.
    pub fn slice(&self, offset: usize, len: usize) -> DeviceView<'a, T> {
        assert!(offset.saturating_add(len) <= self.len, "DeviceView::slice out of bounds");
        DeviceView {
            ptr: self.ptr + (offset * std::mem::size_of::<T>()) as u64,
            len,
            _marker: PhantomData,
        }
    }
}

/// Non-owning mutable view. Lifetime-bound to the owning buffer.
pub struct DeviceViewMut<'a, T: Repr> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: Repr> DeviceViewMut<'a, T> {
    #[inline] pub fn device_ptr(&self) -> CUdeviceptr { self.ptr }
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }
    #[inline] pub fn as_view(&self) -> DeviceView<'_, T> {
        DeviceView { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Pinned host memory
// ════════════════════════════════════════════════════════════════════════════

/// Page-locked host buffer. Enables true async H↔D transfers.
pub struct PinnedBuf<T: Repr> {
    ptr: *mut T,
    len: usize,
    _keep: Arc<Device>,
}

impl<T: Repr> PinnedBuf<T> {
    pub fn alloc(device: Arc<Device>, len: usize) -> Result<Self> {
        device.bind()?;
        let d = driver()?;
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or(Error::Precondition {
            op: "PinnedBuf::alloc", msg: "size overflow".into(),
        })?;
        let mut raw: *mut c_void = ptr::null_mut();
        unsafe { check("cuMemHostAlloc",
                       (d.cuMemHostAlloc)(&mut raw, bytes,
                                          sys::CUhostAllocFlags::Portable as u32))?; }
        Ok(Self { ptr: raw as *mut T, len, _keep: device })
    }
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[inline] pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Repr> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(d) = driver() {
                unsafe { let _ = (d.cuMemFreeHost)(self.ptr as *mut c_void); }
            }
        }
    }
}

unsafe impl<T: Repr> Send for PinnedBuf<T> {}

// ════════════════════════════════════════════════════════════════════════════
// Modules and functions
// ════════════════════════════════════════════════════════════════════════════

pub struct Module {
    device: Arc<Device>,
    handle: CUmodule,
}

impl Module {
    /// Load a PTX (null-terminated inside) or CUBIN image.
    pub fn load(device: Arc<Device>, image: &[u8]) -> Result<Arc<Self>> {
        device.bind()?;
        let d = driver()?;
        // cuModuleLoadData takes a raw pointer; the image must be NUL-terminated
        // for PTX strings. Cubins carry their own length.
        let mut handle = CUmodule::default();
        unsafe { check("cuModuleLoadData",
                       (d.cuModuleLoadData)(&mut handle, image.as_ptr() as *const c_void))?; }
        Ok(Arc::new(Self { device, handle }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Module::function", msg: "function name contains NUL".into(),
        })?;
        let d = driver()?;
        let mut f = CUfunction::default();
        unsafe { check("cuModuleGetFunction",
                       (d.cuModuleGetFunction)(&mut f, self.handle, cname.as_ptr()))?; }
        Ok(Function { module: self.clone(), handle: f, _name: cname })
    }

    #[inline] pub fn device(&self) -> &Arc<Device> { &self.device }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuModuleUnload)(self.handle); }
        }
    }
}

pub struct Function {
    module: Arc<Module>,
    handle: CUfunction,
    _name: CString,
}

impl Function {
    #[inline] pub fn module(&self) -> &Arc<Module> { &self.module }
    #[inline] pub fn raw(&self) -> CUfunction { self.handle }

    /// Launch with the given config, stream, and argument tuple.
    pub fn launch<A: LaunchArgs>(&self, cfg: LaunchCfg, stream: &Stream, args: A) -> Result<()> {
        let d = driver()?;
        let mut slots = A::slots();
        let ptrs = args.pack(&mut slots);
        unsafe {
            check("cuLaunchKernel", (d.cuLaunchKernel)(
                self.handle,
                cfg.grid.0, cfg.grid.1, cfg.grid.2,
                cfg.block.0, cfg.block.1, cfg.block.2,
                cfg.shared_bytes, stream.handle,
                ptrs.as_ptr() as *mut *mut c_void,
                ptr::null_mut(),
            ))
        }
    }
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// Launch geometry.
#[derive(Copy, Clone, Debug)]
pub struct LaunchCfg {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_bytes: u32,
}

impl LaunchCfg {
    pub fn linear(grid: u32, block: u32) -> Self {
        Self { grid: (grid, 1, 1), block: (block, 1, 1), shared_bytes: 0 }
    }
    pub fn for_elements(n: u32, block: u32) -> Self {
        let grid = (n + block - 1) / block.max(1);
        Self::linear(grid.max(1), block)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LaunchArgs — compile-time typed arg packing
// ────────────────────────────────────────────────────────────────────────────

/// Trait implemented for tuples of launch arguments. `slots` returns a scratch
/// array of pointers; `pack` fills it with pointers to each packed arg.
pub trait LaunchArgs {
    type Slots: AsMut<[*const c_void]>;
    fn slots() -> Self::Slots;
    fn pack(self, slots: &mut Self::Slots) -> &[*const c_void];
}

/// Anything that can be passed to a kernel as a scalar argument. Blanket impls
/// below cover primitive POD + `CUdeviceptr` + `DeviceView`/`DeviceViewMut`
/// (pointer-valued).
pub trait KernelArg {
    /// Write the pointer representing this arg into `slot`. Returns the pointer
    /// stored (same as `slot` but typed).
    fn bind(&self, slot: &mut *const c_void);
}

macro_rules! kernel_arg_pod {
    ($($t:ty),*) => {
        $(impl KernelArg for $t {
            #[inline]
            fn bind(&self, slot: &mut *const c_void) {
                *slot = self as *const $t as *const c_void;
            }
        })*
    };
}
kernel_arg_pod!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);
// CUdeviceptr is a type alias for u64 so the u64 impl covers it.

impl<'a, T: Repr> KernelArg for DeviceView<'a, T> {
    #[inline]
    fn bind(&self, slot: &mut *const c_void) {
        *slot = (&self.ptr) as *const CUdeviceptr as *const c_void;
    }
}

impl<'a, T: Repr> KernelArg for DeviceViewMut<'a, T> {
    #[inline]
    fn bind(&self, slot: &mut *const c_void) {
        *slot = (&self.ptr) as *const CUdeviceptr as *const c_void;
    }
}

macro_rules! launch_args_tuple {
    ($n:literal; $($i:tt => $ty:ident),*) => {
        impl<$($ty: KernelArg),*> LaunchArgs for ($($ty,)*) {
            type Slots = [*const c_void; $n];
            #[inline] fn slots() -> Self::Slots { [ptr::null(); $n] }
            #[inline]
            fn pack(self, slots: &mut Self::Slots) -> &[*const c_void] {
                $( self.$i.bind(&mut slots[$i]); )*
                &slots[..]
            }
        }
    };
}

impl LaunchArgs for () {
    type Slots = [*const c_void; 0];
    #[inline] fn slots() -> Self::Slots { [] }
    #[inline] fn pack(self, _s: &mut Self::Slots) -> &[*const c_void] { &[] }
}
launch_args_tuple!(1; 0 => A);
launch_args_tuple!(2; 0 => A, 1 => B);
launch_args_tuple!(3; 0 => A, 1 => B, 2 => C);
launch_args_tuple!(4; 0 => A, 1 => B, 2 => C, 3 => D);
launch_args_tuple!(5; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E);
launch_args_tuple!(6; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E, 5 => F);
launch_args_tuple!(7; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E, 5 => F, 6 => G);
launch_args_tuple!(8; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E, 5 => F, 6 => G, 7 => H);
launch_args_tuple!(9; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I);
launch_args_tuple!(10; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J);
launch_args_tuple!(11; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K);
launch_args_tuple!(12; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L);

// ════════════════════════════════════════════════════════════════════════════
// Graph capture/replay
// ════════════════════════════════════════════════════════════════════════════

pub struct CapturedGraph { handle: CUgraph }

pub struct GraphExec { handle: CUgraphExec, device: Arc<Device> }

impl Stream {
    /// Begin capturing all subsequent work on this stream into a graph.
    pub fn begin_capture(&self) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuStreamBeginCapture_v2",
                       (d.cuStreamBeginCapture_v2)(self.handle, CUstreamCaptureMode::ThreadLocal)) }
    }

    /// End capture and return the resulting graph. Call [`GraphExec::new`] to
    /// instantiate an executable.
    pub fn end_capture(&self) -> Result<CapturedGraph> {
        let d = driver()?;
        let mut g = CUgraph::default();
        unsafe { check("cuStreamEndCapture",
                       (d.cuStreamEndCapture)(self.handle, &mut g))?; }
        Ok(CapturedGraph { handle: g })
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuGraphDestroy)(self.handle); }
        }
    }
}

impl GraphExec {
    pub fn new(graph: CapturedGraph, device: Arc<Device>) -> Result<Self> {
        device.bind()?;
        let d = driver()?;
        let mut exec = CUgraphExec::default();
        unsafe { check("cuGraphInstantiateWithFlags",
                       (d.cuGraphInstantiateWithFlags)(&mut exec, graph.handle, 0))?; }
        // graph is consumed: destroy it now.
        unsafe { let _ = (d.cuGraphDestroy)(graph.handle); }
        std::mem::forget(graph);
        Ok(Self { handle: exec, device })
    }

    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        unsafe { check("cuGraphLaunch", (d.cuGraphLaunch)(self.handle, stream.handle)) }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            unsafe { let _ = (d.cuGraphExecDestroy)(self.handle); }
        }
    }
}

unsafe impl Send for CapturedGraph {}
unsafe impl Sync for CapturedGraph {}
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

// ════════════════════════════════════════════════════════════════════════════
// Bridge: drv::Error → ironaccelerator_core::Error
// ════════════════════════════════════════════════════════════════════════════

impl From<Error> for ironaccelerator_core::Error {
    fn from(e: Error) -> Self {
        ironaccelerator_core::Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: e.numeric(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_open_without_cuda_reports_not_available() {
        // On a machine without CUDA this returns NotAvailable; on a machine
        // with CUDA it succeeds. Either outcome is fine — we just ensure the
        // call doesn't panic.
        let _ = Device::open(0);
    }

    #[test]
    fn launch_cfg_for_elements_rounds_up() {
        let c = LaunchCfg::for_elements(1024, 256);
        assert_eq!(c.grid.0, 4);
        assert_eq!(c.block.0, 256);
        let c = LaunchCfg::for_elements(1000, 256);
        assert_eq!(c.grid.0, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn launch_args_tuple_lengths() {
        fn check<A: LaunchArgs>() -> usize { std::mem::size_of::<A::Slots>() }
        assert_eq!(check::<()>(), 0);
        assert_eq!(check::<(u32,)>(), std::mem::size_of::<*const c_void>());
        assert_eq!(check::<(u32, u32, u32)>(), 3 * std::mem::size_of::<*const c_void>());
    }
}
