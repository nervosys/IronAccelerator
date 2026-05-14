//! `cudarc`-shaped compatibility surface — **drop-in replacement** for
//! `cudarc::driver`.
//!
//! ## Migration (TL;DR)
//!
//! ```text
//! // before
//! use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync};
//! use cudarc::nvrtc::compile_ptx;
//!
//! // after
//! use ironaccelerator_cuda::cudarc_compat::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, compile_ptx};
//! ```
//!
//! Idioms work unchanged:
//!
//! ```no_run
//! use ironaccelerator_cuda::cudarc_compat::*;
//! let dev    = CudaDevice::new(0).unwrap();
//! let stream = dev.default_stream();
//! let xs: CudaSlice<f32> = stream.htod_copy(vec![1.0f32, 2.0, 3.0]).unwrap();
//! let out: Vec<f32> = stream.dtoh_sync_copy(&xs).unwrap();
//! assert_eq!(out, [1.0, 2.0, 3.0]);
//! ```
//!
//! ## API coverage map
//!
//! | cudarc 0.19                            | ironaccelerator_cuda::cudarc_compat       |
//! |----------------------------------------|--------------------------------------------|
//! | `CudaDevice::new(ordinal)`             | [`CudaDevice::new`]                        |
//! | `CudaContext::device_count()`          | [`CudaDevice::device_count`] / `::count`   |
//! | `CudaContext::compute_capability()`    | [`CudaDevice::compute_capability`]         |
//! | `CudaContext::mem_get_info()`          | [`CudaDevice::mem_get_info`]               |
//! | `dev.default_stream()`                 | [`CudaDevice::default_stream`]             |
//! | `dev.new_stream()`                     | [`CudaDevice::new_stream`]                 |
//! | `dev.new_stream_with_priority(p)`      | [`CudaDevice::new_stream_with_priority`]   |
//! | `dev.synchronize()`                    | [`CudaDevice::synchronize`]                |
//! | `stream.alloc::<T>(len)`               | `stream.alloc::<T>(len)` via [`CudaStreamExt`] |
//! | `stream.alloc_zeros::<T>(len)`         | `stream.alloc_zeros::<T>(len)`             |
//! | `stream.clone_htod(&host)`             | `stream.htod_sync_copy(&host)`             |
//! | `stream.memcpy_stod(&host)` / `clone_htod` | `stream.htod_copy(vec)` / `htod_sync_copy(&[T])` |
//! | `stream.memcpy_dtov(&buf)` / `clone_dtoh` | `stream.dtoh_sync_copy(&buf)`           |
//! | `stream.memcpy_dtoh_into(&buf, dst)`   | `stream.dtoh_sync_copy_into(&buf, dst)`    |
//! | `stream.synchronize()`                 | `stream.synchronize()`                     |
//! | `stream.record_event(None)`            | `stream.record_event()`                    |
//! | `stream.wait(&event)`                  | `stream.wait(&event)`                      |
//! | `stream.join(&other)`                  | `stream.join(&other)`                      |
//! | `CudaSlice::{len, num_bytes, is_empty}`| same (from [`DeviceBuf`](crate::drv::DeviceBuf)) |
//! | `CudaSlice::ordinal()`                 | same (on [`DeviceBuf`](crate::drv::DeviceBuf))   |
//! | `CudaSlice::try_clone()`               | same (D2D copy on the same stream)               |
//! | `CudaSlice::as_view{,_mut}`            | `buf.view() / buf.view_mut()`                    |
//! | `nvrtc::compile_ptx(src)`              | [`compile_ptx`]                            |
//! | `nvrtc::compile_ptx_with_opts(src, _)` | [`compile_ptx_with_opts`]                  |
//! | `LaunchAsync::launch_async(cfg, args)` | [`LaunchAsync::launch_async`] (`&stream` arg) |
//! | `DevicePtr::device_ptr`                | [`DevicePtr::device_ptr`]                  |
//!
//! ## Differences worth knowing
//!
//! - **Faster.** Wrapper overhead is ~half of cudarc's; alloc/free is ~2×
//!   faster, stream sync ~1.4× faster. See the workspace README for numbers.
//! - **Stream ordering.** Allocation goes through `cuMemAllocAsync`
//!   (stream-ordered) — same as cudarc's default. Call
//!   [`CudaStream::synchronize`] before reading host buffers.
//! - **Launch signature.** [`LaunchAsync::launch_async`] takes
//!   `(cfg, &stream, args)` whereas cudarc takes `(cfg, args)` and reads the
//!   stream from `self`. The extra `&stream` argument makes the binding
//!   explicit; tuple-arg ergonomics are otherwise identical.
//! - **No per-call context rebind.** cudarc verifies thread-context binding
//!   on every driver call via `cuCtxGetCurrent`. We bind once at
//!   [`CudaDevice::new`] and trust the binding to persist. If you switch
//!   contexts manually, call `dev.raw().bind()` before resuming on this device.
//! - **Drop semantics.** [`CudaSlice`] / [`CudaEvent`] / [`CudaStream`] drop is
//!   a single async FFI (`cuMemFreeAsync` / `cuEventDestroy_v2` /
//!   `cuStreamDestroy_v2`). No per-buffer event-fence tracking.

use std::sync::Arc;

pub use crate::drv::{
    DeviceBuf as CudaSlice, DeviceView as CudaView, DeviceViewMut as CudaViewMut,
    Error as DriverError, Event as CudaEvent, Function as CudaFunction, LaunchArgs, LaunchCfg,
    Module as CudaModule, Priority, Repr as DeviceRepr, Result as DriverResult,
    Stream as CudaStream, TimingEvent as CudaTimingEvent, ZeroBits,
};

/// Wrapper around [`crate::drv::Device`] that also owns a lazily-created
/// "default stream" — mirroring cudarc's `CudaDevice`, where most callers
/// reach for `dev.stream()` or treat the device itself as stream-like.
#[derive(Clone)]
pub struct CudaDevice {
    device: Arc<crate::drv::Device>,
    default_stream: Arc<CudaStream>,
}

impl CudaDevice {
    /// Retain the primary context for `ordinal` and create a default stream.
    /// Matches `cudarc::driver::CudaDevice::new`.
    pub fn new(ordinal: usize) -> DriverResult<Arc<Self>> {
        let device = crate::drv::Device::open(ordinal as u32)?;
        device.bind()?;
        let default_stream = CudaStream::new(device.clone())?;
        Ok(Arc::new(Self {
            device,
            default_stream,
        }))
    }

    /// Number of visible CUDA devices.
    pub fn count() -> DriverResult<u32> {
        crate::drv::Device::count()
    }

    #[inline]
    pub fn ordinal(&self) -> u32 {
        self.device.ordinal()
    }
    #[inline]
    pub fn name(&self) -> DriverResult<String> {
        self.device.name()
    }
    #[inline]
    pub fn total_mem(&self) -> DriverResult<usize> {
        self.device.total_mem()
    }

    /// Clone the default stream. cudarc users often write `dev.clone()` and
    /// treat the device as a stream; here the explicit call is cheap and
    /// makes the lifetime obvious.
    #[inline]
    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.default_stream.clone()
    }

    /// Create a fresh non-blocking stream on this device.
    pub fn new_stream(&self) -> DriverResult<Arc<CudaStream>> {
        CudaStream::new(self.device.clone())
    }

    pub fn new_stream_with_priority(&self, priority: Priority) -> DriverResult<Arc<CudaStream>> {
        CudaStream::with_priority(self.device.clone(), priority)
    }

    /// Borrow the underlying IronAccelerator [`crate::drv::Device`] handle.
    #[inline]
    pub fn raw(&self) -> &Arc<crate::drv::Device> {
        &self.device
    }

    /// Block the calling thread until all prior work on the default stream
    /// completes. Matches `cudarc::driver::CudaDevice::synchronize`.
    pub fn synchronize(&self) -> DriverResult<()> {
        self.default_stream.synchronize()
    }

    /// `(free_bytes, total_bytes)` of device memory, queried via
    /// `cuMemGetInfo_v2`. Matches `cudarc::driver::CudaContext::mem_get_info`.
    pub fn mem_get_info(&self) -> DriverResult<(usize, usize)> {
        self.device.bind()?;
        let f = iron_cuda_sys::driver::fns().map_err(|_| DriverError::NotAvailable {
            lib: "cuda-driver",
            detail: String::from("cuda-driver not loaded"),
        })?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        let code = unsafe { (f.cuMemGetInfo_v2)(&mut free, &mut total) };
        if code.is_ok() {
            Ok((free, total))
        } else {
            Err(DriverError::Driver {
                op: "cuMemGetInfo_v2",
                code,
            })
        }
    }

    /// Compute capability `(major, minor)` of this device.
    /// Matches `cudarc::driver::CudaContext::compute_capability`.
    pub fn compute_capability(&self) -> DriverResult<(u32, u32)> {
        self.device.compute_capability()
    }

    /// Number of CUDA devices visible to the driver.
    /// Matches `cudarc::driver::CudaContext::device_count`.
    #[inline]
    pub fn device_count() -> DriverResult<u32> {
        Self::count()
    }
}

/// Compile CUDA C++ source to PTX with NVRTC. Matches
/// `cudarc::nvrtc::compile_ptx`. Defaults to `compute_80`; override with
/// [`compile_ptx_with_opts`] for a specific arch.
///
/// The returned bytes are a NUL-terminated PTX image suitable for
/// [`CudaModule::load`].
pub fn compile_ptx(src: &str) -> ironaccelerator_core::Result<Vec<u8>> {
    compile_ptx_with_opts(src, crate::kernel::CompileOptions::default())
}

/// Compile CUDA C++ source with explicit NVRTC options. Matches
/// `cudarc::nvrtc::compile_ptx_with_opts`.
pub fn compile_ptx_with_opts(
    src: &str,
    opts: crate::kernel::CompileOptions,
) -> ironaccelerator_core::Result<Vec<u8>> {
    let arch = opts.arch.clone().unwrap_or_else(|| "compute_80".into());
    crate::kernel::compile(src, &arch, &opts)
}

/// Marker trait for anything exposing a raw `CUdeviceptr`. Mirrors
/// `cudarc::driver::DevicePtr`. Implemented for our slice / view types.
pub trait DevicePtr {
    fn device_ptr(&self) -> iron_cuda_sys::driver::CUdeviceptr;
}

impl<T: DeviceRepr> DevicePtr for CudaSlice<T> {
    #[inline]
    fn device_ptr(&self) -> iron_cuda_sys::driver::CUdeviceptr {
        self.device_ptr()
    }
}
impl<'a, T: DeviceRepr> DevicePtr for CudaView<'a, T> {
    #[inline]
    fn device_ptr(&self) -> iron_cuda_sys::driver::CUdeviceptr {
        self.device_ptr()
    }
}
impl<'a, T: DeviceRepr> DevicePtr for CudaViewMut<'a, T> {
    #[inline]
    fn device_ptr(&self) -> iron_cuda_sys::driver::CUdeviceptr {
        self.device_ptr()
    }
}

/// cudarc-shaped convenience methods on [`CudaStream`].
///
/// We implement these as a trait rather than inherent methods so that the
/// core `drv::Stream` type stays minimal.
pub trait CudaStreamExt {
    /// Copy a host `Vec` onto the device on this stream. Equivalent to
    /// `cudarc::driver::CudaStream::memcpy_stod`.
    fn htod_copy<T: DeviceRepr>(&self, src: Vec<T>) -> DriverResult<CudaSlice<T>>;

    /// Copy a host slice onto the device on this stream.
    fn htod_sync_copy<T: DeviceRepr>(&self, src: &[T]) -> DriverResult<CudaSlice<T>>;

    /// Allocate an uninitialised device buffer.
    fn alloc<T: DeviceRepr>(&self, len: usize) -> DriverResult<CudaSlice<T>>;

    /// Allocate a zero-initialised device buffer.
    fn alloc_zeros<T: DeviceRepr + ZeroBits>(&self, len: usize) -> DriverResult<CudaSlice<T>>;

    /// Copy `src` to a freshly allocated `Vec` and synchronise the stream.
    fn dtoh_sync_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> DriverResult<Vec<T>>;

    /// Copy `src` into a pre-allocated host buffer and synchronise the stream.
    fn dtoh_sync_copy_into<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> DriverResult<()>;

    /// Create a fresh event and record it on this stream. Matches
    /// `cudarc::driver::CudaStream::record_event(None)`.
    fn record_event(&self) -> DriverResult<CudaEvent>;

    /// Make this stream wait for the given event before launching subsequent
    /// work. Matches `cudarc::driver::CudaStream::wait`.
    fn wait(&self, event: &CudaEvent) -> DriverResult<()>;

    /// Make this stream wait for any pending work on `other`. Implemented by
    /// recording an event on `other` and waiting for it on `self`. Matches
    /// `cudarc::driver::CudaStream::join`.
    fn join(&self, other: &Arc<CudaStream>) -> DriverResult<()>;
}

impl CudaStreamExt for Arc<CudaStream> {
    #[inline]
    fn htod_copy<T: DeviceRepr>(&self, src: Vec<T>) -> DriverResult<CudaSlice<T>> {
        CudaSlice::from_host(self.clone(), &src)
    }

    #[inline]
    fn htod_sync_copy<T: DeviceRepr>(&self, src: &[T]) -> DriverResult<CudaSlice<T>> {
        let buf = CudaSlice::from_host(self.clone(), src)?;
        self.synchronize()?;
        Ok(buf)
    }

    #[inline]
    fn alloc<T: DeviceRepr>(&self, len: usize) -> DriverResult<CudaSlice<T>> {
        CudaSlice::alloc(self.clone(), len)
    }

    #[inline]
    fn alloc_zeros<T: DeviceRepr + ZeroBits>(&self, len: usize) -> DriverResult<CudaSlice<T>> {
        CudaSlice::alloc_zeros(self.clone(), len)
    }

    #[inline]
    fn dtoh_sync_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> DriverResult<Vec<T>> {
        // SAFETY for the uninit window: `DeviceRepr` bounds T to plain-old-data
        // and we synchronise the stream before exposing `out` to the caller, so
        // every byte has been written by `copy_to_host` before any read.
        let len = src.len();
        let mut out: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(len);
        unsafe {
            out.set_len(len);
        }
        // Reinterpret as &mut [T] — copy_to_host treats it as a bag of bytes.
        let dst: &mut [T] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut T, len) };
        src.copy_to_host(dst)?;
        self.synchronize()?;
        let mut out = std::mem::ManuallyDrop::new(out);
        let (ptr, len, cap) = (out.as_mut_ptr() as *mut T, out.len(), out.capacity());
        Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    }

    #[inline]
    fn dtoh_sync_copy_into<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> DriverResult<()> {
        src.copy_to_host(dst)?;
        self.synchronize()
    }

    #[inline]
    fn record_event(&self) -> DriverResult<CudaEvent> {
        let e = CudaEvent::new(self.device().clone())?;
        e.record(self)?;
        Ok(e)
    }

    #[inline]
    fn wait(&self, event: &CudaEvent) -> DriverResult<()> {
        self.wait_for(event)
    }

    #[inline]
    fn join(&self, other: &Arc<CudaStream>) -> DriverResult<()> {
        let e = other.record_event()?;
        self.wait(&e)
    }
}

/// cudarc's `LaunchAsync` trait: kernels expose `launch_async(cfg, args)`.
/// We adapt our [`CudaFunction::launch`] which takes an explicit stream.
pub trait LaunchAsync {
    fn launch_async<A: LaunchArgs>(
        &self,
        cfg: LaunchCfg,
        stream: &CudaStream,
        args: A,
    ) -> DriverResult<()>;
}

impl LaunchAsync for CudaFunction {
    #[inline]
    fn launch_async<A: LaunchArgs>(
        &self,
        cfg: LaunchCfg,
        stream: &CudaStream,
        args: A,
    ) -> DriverResult<()> {
        self.launch(cfg, stream, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_aliases_resolve() {
        // Purely a compile-test: if any of these re-exports drifts, the
        // crate stops building long before a live-GPU test sees the change.
        fn _uses_aliases<T: DeviceRepr + ZeroBits>() {
            let _: Option<Arc<CudaStream>> = None;
            let _: Option<CudaSlice<T>> = None;
            let _: Option<CudaView<'_, T>> = None;
            let _: Option<CudaFunction> = None;
            let _: Option<CudaModule> = None;
        }
        _uses_aliases::<f32>();
    }
}
