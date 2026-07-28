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
        Error::NotAvailable {
            lib: "cuda-driver",
            detail: format!("{e}"),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[inline(always)]
fn driver() -> Result<&'static sys::DriverFns> {
    sys::fns().map_err(driver_load_err)
}

// Cold error path kept out of line so the hot driver() path stays tiny.
#[cold]
#[inline(never)]
fn driver_load_err(e: &'static iron_cuda_sys::LoadError) -> Error {
    Error::NotAvailable {
        lib: "cuda-driver",
        detail: format!("{e}"),
    }
}

#[inline(always)]
fn check(op: &'static str, code: CUresult) -> Result<()> {
    if code.is_ok() {
        Ok(())
    } else {
        check_err(op, code)
    }
}

// Out-of-line error construction so the hot branch is just `test/jne/ret`.
#[cold]
#[inline(never)]
fn check_err(op: &'static str, code: CUresult) -> Result<()> {
    Err(Error::Driver { op, code })
}

#[cold]
#[inline(never)]
fn alloc_overflow() -> Error {
    Error::Precondition {
        op: "DeviceBuf::alloc",
        msg: "size overflow".into(),
    }
}

#[cold]
#[inline(never)]
fn pinned_alloc_overflow() -> Error {
    Error::Precondition {
        op: "PinnedBuf::alloc",
        msg: "size overflow".into(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// One-shot driver init
// ════════════════════════════════════════════════════════════════════════════

static INIT: OnceLock<Result<()>> = OnceLock::new();

fn ensure_init() -> Result<()> {
    INIT.get_or_init(|| unsafe {
        let d = driver()?;
        check("cuInit", (d.cuInit)(0))
    })
    .clone()
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
    /// Cached driver function table — set at construction so every method on
    /// `Device` skips the AtomicPtr load (`bind`, `attribute`, `total_mem`,
    /// `name`, etc., plus every `Stream::with_priority` that reuses it).
    drv: &'static sys::DriverFns,
    /// Cached `(min, max)` stream priority range, populated lazily on first
    /// use. Saves a `cuStreamGetPriorityRange` FFI per `Stream::new`.
    priority_range: once_cell::sync::OnceCell<(i32, i32)>,
}

impl Device {
    /// Retain the primary context for `ordinal`. Lazily initialises the driver.
    pub fn open(ordinal: u32) -> Result<Arc<Self>> {
        ensure_init()?;
        let d = driver()?;
        let mut device: CUdevice = 0;
        unsafe {
            check("cuDeviceGet", (d.cuDeviceGet)(&mut device, ordinal as i32))?;
        }
        let mut ctx = CUcontext::default();
        unsafe {
            check(
                "cuDevicePrimaryCtxRetain",
                (d.cuDevicePrimaryCtxRetain)(&mut ctx, device),
            )?;
        }

        // Configure the default stream-ordered memory pool to RETAIN freed
        // memory across alloc/free cycles. Without this attribute, every
        // `cuMemFreeAsync` returns memory to the OS at next sync, costing
        // page-remap latency on the next `cuMemAllocAsync`. Setting the
        // release threshold to MAX is the canonical CUDA caching-allocator
        // pattern (PyTorch, cudarc, NVIDIA samples). Required for fast
        // tight allocate/free loops (decode-step intermediate tensors).
        //
        // Best-effort: bind the context first, then query+set the pool. If
        // any step fails the pool just stays at the default (release-every-
        // free) — IA still works, just slower.
        unsafe {
            if (d.cuCtxSetCurrent)(ctx) == sys::CUresult::Success {
                let mut pool = sys::CUmemPool::default();
                if (d.cuDeviceGetDefaultMemPool)(&mut pool, device) == sys::CUresult::Success {
                    let mut threshold: u64 = u64::MAX;
                    let _ = (d.cuMemPoolSetAttribute)(
                        pool,
                        sys::CUmemPool_attribute::ReleaseThreshold,
                        &mut threshold as *mut u64 as *mut std::ffi::c_void,
                    );
                }
            }
        }

        Ok(Arc::new(Self {
            ordinal: ordinal as i32,
            device,
            ctx,
            drv: d,
            priority_range: once_cell::sync::OnceCell::new(),
        }))
    }

    pub fn count() -> Result<u32> {
        ensure_init()?;
        let d = driver()?;
        let mut n: i32 = 0;
        unsafe {
            check("cuDeviceGetCount", (d.cuDeviceGetCount)(&mut n))?;
        }
        Ok(n.max(0) as u32)
    }

    #[inline]
    pub fn ordinal(&self) -> u32 {
        self.ordinal as u32
    }
    #[inline]
    pub fn raw_ctx(&self) -> CUcontext {
        self.ctx
    }
    #[inline]
    pub fn raw_device(&self) -> CUdevice {
        self.device
    }
    /// Crate-internal accessor for the cached driver function table — lets
    /// modules like `cudarc_compat` skip the AtomicPtr load when they already
    /// hold an `Arc<Device>`.
    #[inline]
    pub(crate) fn drv(&self) -> &'static sys::DriverFns {
        self.drv
    }
    /// Bind this primary context to the calling thread. Required before any
    /// call that reads the "current context" (most driver functions do).
    #[inline]
    pub fn bind(&self) -> Result<()> {
        unsafe { check("cuCtxSetCurrent", (self.drv.cuCtxSetCurrent)(self.ctx)) }
    }

    pub fn name(&self) -> Result<String> {
        let mut buf = vec![0i8; 256];
        unsafe {
            check(
                "cuDeviceGetName",
                (self.drv.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.device),
            )?;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    #[inline]
    pub fn attribute(&self, attr: sys::CUdevice_attribute) -> Result<i32> {
        let mut v: i32 = 0;
        unsafe {
            check(
                "cuDeviceGetAttribute",
                (self.drv.cuDeviceGetAttribute)(&mut v, attr, self.device),
            )?;
        }
        Ok(v)
    }

    pub fn total_mem(&self) -> Result<usize> {
        let mut bytes: usize = 0;
        unsafe {
            check(
                "cuDeviceTotalMem_v2",
                (self.drv.cuDeviceTotalMem_v2)(&mut bytes, self.device),
            )?;
        }
        Ok(bytes)
    }

    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        let maj = self.attribute(sys::CUdevice_attribute::ComputeCapabilityMajor)? as u32;
        let min = self.attribute(sys::CUdevice_attribute::ComputeCapabilityMinor)? as u32;
        Ok((maj, min))
    }

    /// 16-byte device UUID. Stable across reboots; useful for identifying a
    /// physical GPU across enumerations and across MIG slices. Matches
    /// `cudarc::driver::CudaContext::uuid`.
    pub fn uuid(&self) -> Result<sys::CUuuid> {
        let mut u = sys::CUuuid::default();
        unsafe {
            check(
                "cuDeviceGetUuid_v2",
                (self.drv.cuDeviceGetUuid_v2)(&mut u, self.device),
            )?;
        }
        Ok(u)
    }

    pub fn can_access_peer(&self, other: &Device) -> Result<bool> {
        let mut v: i32 = 0;
        unsafe {
            check(
                "cuDeviceCanAccessPeer",
                (self.drv.cuDeviceCanAccessPeer)(&mut v, self.device, other.device),
            )?;
        }
        Ok(v != 0)
    }

    pub fn enable_peer_access(&self, other: &Device) -> Result<()> {
        self.bind()?;
        let code = unsafe { (self.drv.cuCtxEnablePeerAccess)(other.ctx, 0) };
        // Already-enabled is not an error for us.
        if code == CUresult::Success || code == CUresult::PeerAccessAlreadyEnabled {
            Ok(())
        } else {
            Err(Error::Driver {
                op: "cuCtxEnablePeerAccess",
                code,
            })
        }
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.cuDevicePrimaryCtxRelease_v2)(self.device);
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

/// Chunk size for staged host→device copies. 4 MiB is large enough that the
/// per-chunk event round-trip is amortised, small enough that two of them is a
/// modest pinned footprint.
const STAGE_CHUNK: usize = 2 << 20;
/// Two slots is what buys the overlap: the host fills one while the DMA drains
/// the other. More slots stop helping once the copy is bandwidth-bound.
const STAGE_SLOTS: usize = 4;
/// Staging only earns its host-side `memcpy` once there is a second chunk for
/// the transfer of the first to overlap with. At a single chunk the copy is
/// strictly serial — measured 300 µs for 1 MiB against 211 µs for a plain
/// blocking copy — so below two chunks the direct paths win.
const STAGE_THRESHOLD: usize = 2 * STAGE_CHUNK;
// Note: there is deliberately no staging on the device→host path. It was
// implemented and measured, and it lost: on H2D the host `memcpy` into pinned
// memory happens *before* the DMA, so staging converts a slow driver-staged
// pageable transfer into a fast one. On D2H the order is forced the other way —
// DMA first, then `memcpy` out — so each chunk serialises against its own
// transfer. Interleaved A/B measurement put staged D2H at 5.22 ms for 16 MiB
// versus 3.47 ms unstaged. The driver's own pageable D2H path is simply better
// than anything we can assemble on top of it.

/// Below this a single thread saturates the copy and the hand-off costs more
/// than it saves.
const PARALLEL_MEMCPY_MIN: usize = 512 << 10;

/// A tiny persistent pool used only to widen the staging `memcpy`.
///
/// Once the transfer pipeline is deep enough, host→pinned `memcpy` — not the
/// DMA — is what bounds a large H2D copy: a single core moves roughly 6 GB/s
/// while the link takes over 11 GiB/s. Splitting that copy across a few threads
/// puts the DMA back in charge.
///
/// Deliberately small and process-wide: a driver library should not size its
/// own thread budget against an application that has its own.
struct CopyPool {
    work: Vec<std::sync::mpsc::Sender<CopyJob>>,
}

/// Completion counter for one `wide_copy`. Per-call rather than shared, so two
/// concurrent staged copies cannot consume each other's completions.
type Pending = std::sync::Arc<(parking_lot::Mutex<usize>, parking_lot::Condvar)>;

struct CopyJob {
    src: *const u8,
    dst: *mut u8,
    len: usize,
    done: Pending,
}
// SAFETY: each job describes a disjoint half-open range of a live allocation,
// and the submitting thread blocks on `done` until every worker has finished,
// so neither pointer outlives the call.
unsafe impl Send for CopyJob {}

static COPY_POOL: OnceLock<Option<CopyPool>> = OnceLock::new();

fn copy_pool() -> Option<&'static CopyPool> {
    COPY_POOL
        .get_or_init(|| {
            let extra = std::thread::available_parallelism()
                .map(|n| (n.get() - 1).min(3))
                .unwrap_or(0);
            if extra == 0 {
                return None;
            }
            let mut work = Vec::with_capacity(extra);
            for _ in 0..extra {
                let (tx, rx) = std::sync::mpsc::channel::<CopyJob>();
                std::thread::Builder::new()
                    .name("ia-stage-copy".into())
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            unsafe { core::ptr::copy_nonoverlapping(job.src, job.dst, job.len) };
                            let (lock, cv) = &*job.done;
                            *lock.lock() -= 1;
                            cv.notify_all();
                        }
                    })
                    .ok()?;
                work.push(tx);
            }
            Some(CopyPool { work })
        })
        .as_ref()
}

/// `memcpy` `len` bytes, split across the pool when it is worth doing.
///
/// # Safety
/// `src` and `dst` must be valid for `len` bytes and must not overlap.
unsafe fn wide_copy(src: *const u8, dst: *mut u8, len: usize) {
    let Some(pool) = copy_pool().filter(|_| len >= PARALLEL_MEMCPY_MIN) else {
        core::ptr::copy_nonoverlapping(src, dst, len);
        return;
    };
    let parts = pool.work.len() + 1;
    // Keep slices page-aligned so no two threads write the same page.
    let step = (len / parts + 4095) & !4095;
    let done: Pending =
        std::sync::Arc::new((parking_lot::Mutex::new(0), parking_lot::Condvar::new()));
    let mut off = 0;
    for tx in pool.work.iter() {
        if off + step >= len {
            break;
        }
        *done.0.lock() += 1;
        if tx
            .send(CopyJob {
                src: src.add(off),
                dst: dst.add(off),
                len: step,
                done: done.clone(),
            })
            .is_err()
        {
            // Worker gone: take the slice back rather than leaving a hole.
            *done.0.lock() -= 1;
            break;
        }
        off += step;
    }
    // The submitting thread takes the remainder rather than idling.
    core::ptr::copy_nonoverlapping(src.add(off), dst.add(off), len - off);

    let (lock, cv) = &*done;
    let mut n = lock.lock();
    while *n > 0 {
        cv.wait(&mut n);
    }
}

struct StageSlot {
    ptr: *mut c_void,
    event: CUevent,
    /// Whether `event` has been recorded and not yet waited on.
    armed: bool,
}

/// A per-stream ring of pinned staging chunks.
///
/// Pageable `cuMemcpyHtoDAsync` on a non-null stream cannot DMA directly: the
/// driver stages it internally, and on large transfers that path is roughly an
/// order of magnitude slower than a DMA out of pinned memory. Staging through
/// our own pinned ring turns those into real DMAs and lets the host-side
/// `memcpy` of one chunk overlap the transfer of the previous one.
///
/// The ring is owned by the `Stream`, so the pinned memory outlives any DMA
/// still in flight when a staged copy returns — which is what keeps the
/// asynchronous contract intact.
struct StageRing {
    slots: [StageSlot; STAGE_SLOTS],
    cursor: usize,
}

// SAFETY: the raw pointers are pinned host allocations owned solely by this
// ring, and every access goes through the owning `Mutex`.
unsafe impl Send for StageRing {}

pub struct Stream {
    device: Arc<Device>,
    handle: CUstream,
    priority: i32,
    /// Cached driver function table — avoids the AtomicPtr load on every
    /// `synchronize` / `cuMemAllocAsync` / `cuMemFreeAsync` etc. through this
    /// stream. Safe because the driver is fully initialised before any
    /// `Stream` exists, and `&'static DriverFns` outlives every stream.
    drv: &'static sys::DriverFns,
    /// Created on the first staged copy, so streams that never move large
    /// pageable buffers pay nothing for it.
    stage: once_cell::sync::OnceCell<parking_lot::Mutex<StageRing>>,
}

impl Stream {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>> {
        Self::with_priority(device, Priority::Default)
    }

    pub fn with_priority(device: Arc<Device>, pri: Priority) -> Result<Arc<Self>> {
        device.bind()?;
        let d = device.drv;
        // Cached on `Device`: the priority range is a device-static property,
        // so we ask the driver at most once per Device instead of per Stream.
        let (lo, hi) = *device
            .priority_range
            .get_or_try_init(|| -> Result<(i32, i32)> {
                let mut lo = 0i32;
                let mut hi = 0i32;
                unsafe {
                    check(
                        "cuStreamGetPriorityRange",
                        (d.cuStreamGetPriorityRange)(&mut lo, &mut hi),
                    )?;
                }
                Ok((lo, hi))
            })?;
        let priority = match pri {
            Priority::Default => 0,
            Priority::High => lo,
            Priority::Low => hi,
            Priority::Raw(v) => v.clamp(lo, hi),
        };
        let mut handle = CUstream::default();
        unsafe {
            check(
                "cuStreamCreateWithPriority",
                (d.cuStreamCreateWithPriority)(&mut handle, sys::CU_STREAM_NON_BLOCKING, priority),
            )?;
        }
        Ok(Arc::new(Self {
            device,
            handle,
            priority,
            drv: d,
            stage: once_cell::sync::OnceCell::new(),
        }))
    }

    /// The staging ring, created on first use.
    fn stage_ring(&self) -> Result<&parking_lot::Mutex<StageRing>> {
        self.stage.get_or_try_init(|| {
            let mut slots: [Option<StageSlot>; STAGE_SLOTS] = Default::default();
            for slot in slots.iter_mut() {
                let mut ptr: *mut c_void = core::ptr::null_mut();
                let mut event = CUevent::default();
                unsafe {
                    check(
                        "cuMemHostAlloc",
                        (self.drv.cuMemHostAlloc)(&mut ptr, STAGE_CHUNK, 0),
                    )?;
                    // CU_EVENT_DISABLE_TIMING: we only ever wait on these.
                    if let Err(e) = check("cuEventCreate", (self.drv.cuEventCreate)(&mut event, 2))
                    {
                        let _ = (self.drv.cuMemFreeHost)(ptr);
                        return Err(e);
                    }
                }
                *slot = Some(StageSlot {
                    ptr,
                    event,
                    armed: false,
                });
            }
            Ok(parking_lot::Mutex::new(StageRing {
                slots: slots.map(|s| s.expect("every slot initialised above")),
                cursor: 0,
            }))
        })
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    #[inline]
    pub fn raw(&self) -> CUstream {
        self.handle
    }
    #[inline]
    pub fn priority(&self) -> i32 {
        self.priority
    }

    #[inline]
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check(
                "cuStreamSynchronize",
                (self.drv.cuStreamSynchronize)(self.handle),
            )
        }
    }

    /// Make this stream block until `event` completes on whatever stream
    /// recorded it.
    #[inline]
    pub fn wait_for(&self, event: &Event) -> Result<()> {
        unsafe {
            check(
                "cuStreamWaitEvent",
                (self.drv.cuStreamWaitEvent)(self.handle, event.handle, 0),
            )
        }
    }

    /// Stream-ordered `cuMemsetD8Async` — set `bytes` bytes at `ptr` to
    /// `value` on THIS stream. Capture-safe: when this stream is being
    /// captured the memset is recorded into the graph. (cudarc's
    /// `memset_zeros` issues on the buffer's owning stream — the legacy NULL
    /// stream for pool buffers — which invalidates an in-progress capture of
    /// another stream. Use this to keep the zeroing on the captured stream.)
    #[inline]
    pub fn memset_d8_async(&self, ptr: CUdeviceptr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(
                "cuMemsetD8Async",
                (self.drv.cuMemsetD8Async)(ptr, value, bytes, self.handle),
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
            // Any staged copy may have left a DMA in flight reading from the
            // pinned ring, so drain the stream before the ring is freed.
            // `cuStreamDestroy` alone does not guarantee that ordering.
            if self.stage.get().is_some() {
                let _ = (self.drv.cuStreamSynchronize)(self.handle);
            }
            let _ = (self.drv.cuStreamDestroy_v2)(self.handle);
            if let Some(ring) = self.stage.get() {
                let ring = ring.lock();
                for slot in ring.slots.iter() {
                    let _ = (self.drv.cuEventDestroy_v2)(slot.event);
                    let _ = (self.drv.cuMemFreeHost)(slot.ptr);
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Events
// ════════════════════════════════════════════════════════════════════════════

/// Fast event — timing disabled, suitable for stream-ordered fences.
pub struct Event {
    _device: Arc<Device>,
    handle: CUevent,
    timing: bool,
    drv: &'static sys::DriverFns,
}

impl Event {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::new_impl(device, CUevent_flags::DisableTiming, false)
    }
    fn new_impl(device: Arc<Device>, flags: CUevent_flags, timing: bool) -> Result<Self> {
        // Context is already bound from `Device::open`. cuEventCreate
        // reads the current context; no per-event `cuCtxSetCurrent` needed.
        // This saves ~5 µs (the 14 % event-lifecycle gap vs cudarc).
        let d = device.drv;
        let mut handle = CUevent::default();
        unsafe {
            check(
                "cuEventCreate",
                (d.cuEventCreate)(&mut handle, flags as u32),
            )?;
        }
        Ok(Self {
            _device: device,
            handle,
            timing,
            drv: d,
        })
    }

    #[inline]
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "cuEventRecord",
                (self.drv.cuEventRecord)(self.handle, stream.handle),
            )
        }
    }

    #[inline]
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check(
                "cuEventSynchronize",
                (self.drv.cuEventSynchronize)(self.handle),
            )
        }
    }

    #[inline]
    pub fn raw(&self) -> CUevent {
        self.handle
    }
    #[inline]
    pub fn supports_timing(&self) -> bool {
        self.timing
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.cuEventDestroy_v2)(self.handle);
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
    #[inline]
    pub fn record(&self, stream: &Stream) -> Result<()> {
        self.0.record(stream)
    }
    #[inline]
    pub fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
    #[inline]
    pub fn as_event(&self) -> &Event {
        &self.0
    }

    /// Milliseconds elapsed between `start` and `end`. Both events must have
    /// completed (call `.synchronize()` on the later one first).
    pub fn elapsed_ms(start: &TimingEvent, end: &TimingEvent) -> Result<f32> {
        let mut ms: f32 = 0.0;
        unsafe {
            check(
                "cuEventElapsedTime",
                (start.0.drv.cuEventElapsedTime)(&mut ms, start.0.handle, end.0.handle),
            )?;
        }
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

#[cfg(feature = "f16")]
unsafe impl Repr for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl Repr for half::bf16 {}

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

#[cfg(feature = "f16")]
unsafe impl ZeroBits for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl ZeroBits for half::bf16 {}

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
    #[inline]
    pub fn alloc(stream: Arc<Stream>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                stream,
                ptr: 0,
                len: 0,
                _marker: PhantomData,
            });
        }
        // checked_mul + cold-path error construction. Using `ok_or(Error { msg: _.into() })`
        // here would heap-allocate the message on every successful alloc.
        let Some(bytes) = len.checked_mul(std::mem::size_of::<T>()) else {
            return Err(alloc_overflow());
        };
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check(
                "cuMemAllocAsync",
                (stream.drv.cuMemAllocAsync)(&mut ptr, bytes, stream.handle),
            )?;
        }
        Ok(Self {
            stream,
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn alloc_zeros(stream: Arc<Stream>, len: usize) -> Result<Self>
    where
        T: ZeroBits,
    {
        let buf = Self::alloc(stream, len)?;
        if buf.len > 0 {
            let bytes = buf.len * std::mem::size_of::<T>();
            unsafe {
                check(
                    "cuMemsetD8Async",
                    (buf.stream.drv.cuMemsetD8Async)(buf.ptr, 0, bytes, buf.stream.handle),
                )?;
            }
        }
        Ok(buf)
    }

    /// Allocate and copy a host slice in one stream-ordered operation.
    #[inline]
    pub fn from_host(stream: Arc<Stream>, src: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(stream, src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    #[inline]
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
        // Multi-chunk pageable sources go through the pinned staging ring: the
        // driver cannot DMA out of pageable memory on a non-null stream, and its
        // internal staging is roughly an order of magnitude slower than doing it
        // ourselves once there is enough work to pipeline.
        if bytes >= STAGE_THRESHOLD {
            return self.copy_from_host_staged(src.as_ptr() as *const u8, bytes);
        }
        unsafe {
            check(
                "cuMemcpyHtoDAsync_v2",
                (self.stream.drv.cuMemcpyHtoDAsync_v2)(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    bytes,
                    self.stream.handle,
                ),
            )?;
        }
        Ok(())
    }

    /// Chunked host→pinned→device copy. Stream-ordered and asynchronous on
    /// return, exactly like the direct path: the final chunk's DMA may still be
    /// in flight, and the pinned chunks it reads from are owned by the `Stream`,
    /// so they outlive it.
    fn copy_from_host_staged(&mut self, src: *const u8, bytes: usize) -> Result<()> {
        let s = &self.stream;
        let drv = s.drv;
        let ring = s.stage_ring()?;
        let mut ring = ring.lock();

        let mut off = 0usize;
        while off < bytes {
            let n = (bytes - off).min(STAGE_CHUNK);
            let i = ring.cursor % STAGE_SLOTS;

            unsafe {
                // Do not overwrite a chunk the GPU is still reading. This is
                // also what throttles the pipeline to the DMA's pace.
                if ring.slots[i].armed {
                    check(
                        "cuEventSynchronize",
                        (drv.cuEventSynchronize)(ring.slots[i].event),
                    )?;
                    ring.slots[i].armed = false;
                }
                wide_copy(src.add(off), ring.slots[i].ptr as *mut u8, n);
                check(
                    "cuMemcpyHtoDAsync_v2",
                    (drv.cuMemcpyHtoDAsync_v2)(
                        self.ptr + off as CUdeviceptr,
                        ring.slots[i].ptr,
                        n,
                        s.handle,
                    ),
                )?;
                check(
                    "cuEventRecord",
                    (drv.cuEventRecord)(ring.slots[i].event, s.handle),
                )?;
                ring.slots[i].armed = true;
            }

            ring.cursor += 1;
            off += n;
        }
        Ok(())
    }

    /// Blocking host→device copy: returns once `src` has been consumed.
    ///
    /// The right primitive behind a *synchronous* API below the staging
    /// threshold. `cuMemcpyHtoDAsync` on a non-null stream has to stage a
    /// pageable source internally; `cuMemcpyHtoD_v2` takes the driver's
    /// optimised blocking path, which measured 211 µs for 1 MiB against 477 µs
    /// for the async call and 300 µs for single-chunk staging. Above the
    /// threshold the staged path wins by far more, so callers should prefer
    /// [`DeviceBuf::copy_from_host`] there.
    pub fn copy_from_host_blocking(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_host_blocking",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        // `cuMemcpyHtoD_v2` is not ordered against this stream, and the buffer
        // itself was handed out by the stream-ordered allocator, so the stream
        // has to be drained before the copy may touch it. This is a
        // correctness requirement, not a precaution.
        self.stream.synchronize()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "cuMemcpyHtoD_v2",
                (self.stream.drv.cuMemcpyHtoD_v2)(self.ptr, src.as_ptr() as *const c_void, bytes),
            )?;
        }
        Ok(())
    }

    /// Host→device copy for a caller that is about to block anyway.
    ///
    /// Picks whichever path is actually faster at this size — the blocking
    /// driver copy below the staging threshold, the pipelined staged path above
    /// it — and leaves the stream idle either way. This keeps the choice, and
    /// the threshold, out of callers.
    pub fn copy_from_host_sync(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_host_sync",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len()),
            });
        }
        let bytes = self.len * std::mem::size_of::<T>();
        if bytes >= STAGE_THRESHOLD {
            self.copy_from_host(src)?;
            self.stream.synchronize()
        } else {
            self.copy_from_host_blocking(src)
        }
    }

    /// Blocking device→host copy: returns only once `dst` is populated.
    ///
    /// This is the right primitive behind a *synchronous* API. The async entry
    /// point plus an explicit `synchronize` makes the driver stage a pageable
    /// destination on a non-null stream; `cuMemcpyDtoH_v2` takes its optimised
    /// blocking path instead, which is what a caller who is about to block
    /// wants anyway. Any pending staged host→device chunks are drained first so
    /// the copy observes them.
    pub fn copy_to_host_blocking(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_to_host_blocking",
                msg: format!("length mismatch: src={} dst={}", self.len, dst.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        self.stream.synchronize()?;
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "cuMemcpyDtoH_v2",
                (self.stream.drv.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes),
            )?;
        }
        Ok(())
    }

    /// Device→host copy for a caller that is about to block anyway.
    ///
    /// Mirrors [`DeviceBuf::copy_from_host_sync`]: the pipelined staged path
    /// once there are enough chunks to keep the ring busy, the blocking driver
    /// copy below that. Leaves the stream idle either way.
    pub fn copy_to_host_sync(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_to_host_sync",
                msg: format!("length mismatch: src={} dst={}", self.len, dst.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        // Always the blocking driver copy. Staging was implemented, pipelined
        // four deep, and measured against it with the paired harness: 5.46 ms
        // for 16 MiB staged against 3.81 ms blocking. The D2H order is forced
        // DMA-then-memcpy, so the host copy cannot hide behind the transfer the
        // way it does on H2D, and the driver's own pageable path is better than
        // anything assembled on top of it. Measured twice with the paired
        // harness, the second time with the host leg widened over the copy
        // pool — the same change that took H2D at 16 MiB from parity to 1.57×.
        // It still lost: 0.89× [0.83, 0.93]. The asymmetry is that the driver's
        // pageable *D2H* path is not pathological the way its H2D path on a
        // non-null stream is, so staging only adds a hop.
        self.copy_to_host_blocking(dst)
    }

    #[inline]
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
                "cuMemcpyDtoHAsync_v2",
                (self.stream.drv.cuMemcpyDtoHAsync_v2)(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    self.stream.handle,
                ),
            )?;
        }
        Ok(())
    }

    /// Device-to-device copy. Both buffers must be on the same device; we pick
    /// `self`'s stream for ordering.
    #[inline]
    pub fn copy_from_device(&mut self, src: &DeviceBuf<T>) -> Result<()> {
        if src.len != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_device",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "cuMemcpyDtoDAsync_v2",
                (self.stream.drv.cuMemcpyDtoDAsync_v2)(
                    self.ptr,
                    src.ptr,
                    bytes,
                    self.stream.handle,
                ),
            )?;
        }
        Ok(())
    }

    /// Async copy from a buffer on another device. Both contexts must
    /// allow peer access (see [`Device::enable_peer_access`]); the call
    /// enqueues on `self`'s stream which the driver treats as belonging to
    /// the *destination* (source) ordering. Length must match `src`.
    #[inline]
    pub fn copy_from_peer_async(&mut self, src: &DeviceBuf<T>) -> Result<()> {
        if src.len != self.len {
            return Err(Error::Precondition {
                op: "DeviceBuf::copy_from_peer_async",
                msg: format!("length mismatch: dst={} src={}", self.len, src.len),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "cuMemcpyPeerAsync",
                (self.stream.drv.cuMemcpyPeerAsync)(
                    self.ptr,
                    self.stream.device.ctx,
                    src.ptr,
                    src.stream.device.ctx,
                    bytes,
                    self.stream.handle,
                ),
            )?;
        }
        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    /// Alias of [`Self::byte_len`] for `cudarc::driver::CudaSlice` API parity.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.byte_len()
    }
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
    /// Ordinal of the device this buffer is allocated on. Matches
    /// `cudarc::driver::CudaSlice::ordinal`.
    #[inline]
    pub fn ordinal(&self) -> u32 {
        self.stream.device().ordinal()
    }

    /// Allocate a sibling buffer on the same stream and enqueue a
    /// device-to-device copy. The clone is independent — drop it whenever.
    /// Matches `cudarc::driver::CudaSlice::try_clone`.
    /// cudarc-compatible: `slice.slice(range)` produces a `DeviceView`
    /// over `[range.start, range.end)`. Used by code ported from cudarc.
    pub fn slice(&self, range: std::ops::Range<usize>) -> DeviceView<'_, T> {
        let start = range.start.min(self.len);
        let end = range.end.min(self.len).max(start);
        self.view().slice(start, end - start)
    }

    pub fn try_clone(&self) -> Result<Self> {
        let mut out = Self::alloc(self.stream.clone(), self.len)?;
        out.copy_from_device(self)?;
        Ok(out)
    }

    /// Shrink the buffer's logical length. The underlying device allocation
    /// is unchanged; only [`Self::len`] / [`Self::byte_len`] reports the new
    /// value. Panics if `new_len > self.len`. Used by [`crate::pool`] to hand
    /// out partial views of bucket-rounded allocations.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "truncate: new_len > len");
        self.len = new_len;
    }

    /// Zero every byte of the buffer on its stream. Async; the write is
    /// stream-ordered against subsequent reads on the same stream.
    #[inline]
    pub fn zero_in_place(&mut self) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            check(
                "cuMemsetD8Async",
                (self.stream.drv.cuMemsetD8Async)(self.ptr, 0, bytes, self.stream.handle),
            )
        }
    }

    /// Build a `DeviceBuf` from an already-allocated `ptr` on `stream`. The
    /// `capacity_bytes` argument is the byte size of the underlying
    /// allocation; the buffer's logical length is `len` elements. When the
    /// resulting `DeviceBuf` drops, it calls `cuMemFreeAsync(ptr)`.
    ///
    /// # Safety
    /// `ptr` must be a live `cuMemAllocAsync`-issued pointer on `stream`'s
    /// allocator that this `DeviceBuf` is now responsible for freeing.
    /// `capacity_bytes >= len * size_of::<T>()`. Used by [`crate::pool`] to
    /// reuse cached allocations.
    #[inline]
    pub unsafe fn from_raw_parts(
        stream: Arc<Stream>,
        ptr: CUdeviceptr,
        len: usize,
        capacity_bytes: usize,
    ) -> Self {
        debug_assert!(capacity_bytes >= len * std::mem::size_of::<T>());
        // capacity_bytes is intentionally unused at the type level — the
        // pool tracks it externally so it knows which bucket to return to.
        let _ = capacity_bytes;
        Self {
            stream,
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Detach the device pointer without freeing it and zero this buffer's
    /// state so [`Drop`] becomes a no-op for the pointer. The returned ptr
    /// is now the caller's responsibility to free via the appropriate driver
    /// call.
    ///
    /// Used by [`crate::pool`] to recycle a buffer's underlying allocation
    /// back into the pool's cache. The `Arc<Stream>` inside the buffer still
    /// drops normally — only `cuMemFreeAsync` is suppressed.
    ///
    /// # Safety
    /// After this call the buffer's logical state is `(ptr = 0, len = 0)`;
    /// no further reads of the storage are valid.
    #[inline]
    pub unsafe fn detach_ptr(&mut self) -> CUdeviceptr {
        let p = self.ptr;
        self.ptr = 0;
        self.len = 0;
        p
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
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
    #[inline]
    fn drop(&mut self) {
        if self.ptr != 0 {
            // Driver was guaranteed loaded when `self.stream` was created.
            unsafe {
                let _ = (self.stream.drv.cuMemFreeAsync)(self.ptr, self.stream.handle);
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
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Narrow to `[offset .. offset + len]`. Panics on out-of-bounds.
    pub fn slice(&self, offset: usize, len: usize) -> DeviceView<'a, T> {
        assert!(
            offset.saturating_add(len) <= self.len,
            "DeviceView::slice out of bounds"
        );
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
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    #[inline]
    pub fn as_view(&self) -> DeviceView<'_, T> {
        DeviceView {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
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
        let Some(bytes) = len.checked_mul(std::mem::size_of::<T>()) else {
            return Err(pinned_alloc_overflow());
        };
        let mut raw: *mut c_void = ptr::null_mut();
        unsafe {
            check(
                "cuMemHostAlloc",
                (device.drv.cuMemHostAlloc)(
                    &mut raw,
                    bytes,
                    sys::CUhostAllocFlags::Portable as u32,
                ),
            )?;
        }
        Ok(Self {
            ptr: raw as *mut T,
            len,
            _keep: device,
        })
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Repr> Drop for PinnedBuf<T> {
    #[inline]
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self._keep.drv.cuMemFreeHost)(self.ptr as *mut c_void);
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
    drv: &'static sys::DriverFns,
}

impl Module {
    /// Load a PTX (null-terminated inside) or CUBIN image.
    pub fn load(device: Arc<Device>, image: &[u8]) -> Result<Arc<Self>> {
        device.bind()?;
        let d = device.drv;
        // cuModuleLoadData takes a raw pointer; the image must be NUL-terminated
        // for PTX strings. Cubins carry their own length.
        let mut handle = CUmodule::default();
        unsafe {
            check(
                "cuModuleLoadData",
                (d.cuModuleLoadData)(&mut handle, image.as_ptr() as *const c_void),
            )?;
        }
        Ok(Arc::new(Self {
            device,
            handle,
            drv: d,
        }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Module::function",
            msg: "function name contains NUL".into(),
        })?;
        let mut f = CUfunction::default();
        unsafe {
            check(
                "cuModuleGetFunction",
                (self.drv.cuModuleGetFunction)(&mut f, self.handle, cname.as_ptr()),
            )?;
        }
        Ok(Function {
            module: self.clone(),
            handle: f,
            _name: cname,
            drv: self.drv,
        })
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Look up a device-side `__constant__` or `__device__` symbol by
    /// name. Returns `(device_ptr, size_in_bytes)`. Use the pointer to
    /// `cuMemcpyHtoDAsync` constants into the kernel's address space.
    pub fn global(&self, name: &str) -> Result<(CUdeviceptr, usize)> {
        let cname = CString::new(name).map_err(|_| Error::Precondition {
            op: "Module::global",
            msg: "symbol name contains NUL".into(),
        })?;
        let mut p: CUdeviceptr = 0;
        let mut n: usize = 0;
        unsafe {
            check(
                "cuModuleGetGlobal_v2",
                (self.drv.cuModuleGetGlobal_v2)(&mut p, &mut n, self.handle, cname.as_ptr()),
            )?;
        }
        Ok((p, n))
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.cuModuleUnload)(self.handle);
        }
    }
}

#[derive(Clone)]
pub struct Function {
    module: Arc<Module>,
    handle: CUfunction,
    _name: CString,
    drv: &'static sys::DriverFns,
}

impl Function {
    #[inline]
    pub fn module(&self) -> &Arc<Module> {
        &self.module
    }
    #[inline]
    pub fn raw(&self) -> CUfunction {
        self.handle
    }

    /// Launch with the given config, stream, and argument tuple.
    #[inline]
    pub fn launch<A: LaunchArgs>(&self, cfg: LaunchCfg, stream: &Stream, args: A) -> Result<()> {
        let mut storage = A::storage();
        let mut ptrs = A::ptrs_init();
        args.pack(&mut storage, &mut ptrs);
        unsafe {
            check(
                "cuLaunchKernel",
                (self.drv.cuLaunchKernel)(
                    self.handle,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_bytes,
                    stream.handle,
                    ptrs.as_mut().as_mut_ptr() as *mut *mut c_void,
                    ptr::null_mut(),
                ),
            )
        }
    }

    /// Cooperative-groups launch. Same shape as [`Self::launch`] but routes
    /// through `cuLaunchCooperativeKernel` so the kernel can use
    /// `grid.sync()` / `cooperative_groups::this_grid()`. Requires every
    /// block to fit on the device concurrently; use
    /// [`Self::occupancy_max_active_blocks_per_sm`] to size the grid.
    #[inline]
    pub fn launch_cooperative<A: LaunchArgs>(
        &self,
        cfg: LaunchCfg,
        stream: &Stream,
        args: A,
    ) -> Result<()> {
        let mut storage = A::storage();
        let mut ptrs = A::ptrs_init();
        args.pack(&mut storage, &mut ptrs);
        unsafe {
            check(
                "cuLaunchCooperativeKernel",
                (self.drv.cuLaunchCooperativeKernel)(
                    self.handle,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_bytes,
                    stream.handle,
                    ptrs.as_mut().as_mut_ptr() as *mut *mut c_void,
                ),
            )
        }
    }

    /// Read a per-function attribute. Common ones:
    /// [`sys::CUfunction_attribute::MaxThreadsPerBlock`],
    /// [`sys::CUfunction_attribute::NumRegs`],
    /// [`sys::CUfunction_attribute::MaxDynamicSharedSizeBytes`].
    #[inline]
    pub fn attribute(&self, attr: sys::CUfunction_attribute) -> Result<i32> {
        let mut v: i32 = 0;
        unsafe {
            check(
                "cuFuncGetAttribute",
                (self.drv.cuFuncGetAttribute)(&mut v, attr, self.handle),
            )?;
        }
        Ok(v)
    }

    /// Set a per-function attribute. Most commonly used to opt in to
    /// >48 KiB of dynamic shared memory by setting
    /// [`sys::CUfunction_attribute::MaxDynamicSharedSizeBytes`] before
    /// the first launch.
    #[inline]
    pub fn set_attribute(&self, attr: sys::CUfunction_attribute, value: i32) -> Result<()> {
        unsafe {
            check(
                "cuFuncSetAttribute",
                (self.drv.cuFuncSetAttribute)(self.handle, attr, value),
            )
        }
    }

    /// Maximum number of concurrent thread blocks per SM for this function
    /// at the given `block_size` and `dynamic_shmem_bytes`. Multiply by
    /// the device's `MultiprocessorCount` attribute to size a grid that
    /// keeps the GPU fully resident — required input for
    /// [`Self::launch_cooperative`].
    #[inline]
    pub fn occupancy_max_active_blocks_per_sm(
        &self,
        block_size: u32,
        dynamic_shmem_bytes: usize,
    ) -> Result<i32> {
        let mut blocks: i32 = 0;
        unsafe {
            check(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                (self.drv.cuOccupancyMaxActiveBlocksPerMultiprocessor)(
                    &mut blocks,
                    self.handle,
                    block_size as i32,
                    dynamic_shmem_bytes,
                ),
            )?;
        }
        Ok(blocks)
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
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_bytes: 0,
        }
    }
    pub fn for_elements(n: u32, block: u32) -> Self {
        let grid = (n + block - 1) / block.max(1);
        Self::linear(grid.max(1), block)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LaunchArgs — compile-time typed arg packing
// ────────────────────────────────────────────────────────────────────────────

/// Trait implemented for tuples of launch arguments.
///
/// The launch call owns both an 8-byte-per-arg `storage` buffer and a parallel
/// `ptrs` array of `void*`. `pack` copies each arg's bytes into `storage` and
/// writes a pointer into `ptrs[i]` aimed at that slot. Keeping the arg values
/// in caller-owned storage (rather than pointing into the moved-in tuple)
/// means correctness does not depend on `pack` being inlined.
pub trait LaunchArgs {
    type Storage: AsMut<[u64]>;
    type Ptrs: AsMut<[*const c_void]>;
    fn storage() -> Self::Storage;
    fn ptrs_init() -> Self::Ptrs;
    fn pack(self, storage: &mut Self::Storage, ptrs: &mut Self::Ptrs);
}

/// Anything that can be passed to a kernel as a scalar argument. Blanket impls
/// below cover primitive POD + `CUdeviceptr` + `DeviceView`/`DeviceViewMut`
/// (pointer-valued).
pub trait KernelArg {
    /// Write this arg's bit pattern into the 8-byte `slot`. The caller retains
    /// ownership of `slot`; `cuLaunchKernel` will read `sizeof(argtype)` bytes
    /// from the address of `slot`.
    fn write_into(self, slot: &mut u64);
}

macro_rules! kernel_arg_pod {
    ($($t:ty),*) => {
        $(impl KernelArg for $t {
            #[inline]
            fn write_into(self, slot: &mut u64) {
                // Zero the slot first so any unused high bytes are defined.
                *slot = 0;
                unsafe {
                    (slot as *mut u64 as *mut $t).write_unaligned(self);
                }
            }
        })*
    };
}
kernel_arg_pod!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);
// CUdeviceptr is a type alias for u64 so the u64 impl covers it.

impl<'a, T: Repr> KernelArg for DeviceView<'a, T> {
    #[inline]
    fn write_into(self, slot: &mut u64) {
        *slot = self.ptr;
    }
}

impl<'a, T: Repr> KernelArg for DeviceViewMut<'a, T> {
    #[inline]
    fn write_into(self, slot: &mut u64) {
        *slot = self.ptr;
    }
}

// cudarc-compatible: `&CudaSlice<T>` accepted directly as a kernel arg.
// Forwards through `view()` so the call site can keep its `(buf1, buf2, ...)`
// shape unchanged after migration from cudarc.
impl<T: Repr> KernelArg for &DeviceBuf<T> {
    #[inline]
    fn write_into(self, slot: &mut u64) {
        *slot = self.device_ptr();
    }
}

impl<T: Repr> KernelArg for &mut DeviceBuf<T> {
    #[inline]
    fn write_into(self, slot: &mut u64) {
        *slot = self.device_ptr();
    }
}

macro_rules! launch_args_tuple {
    ($n:literal; $($i:tt => $ty:ident),*) => {
        impl<$($ty: KernelArg),*> LaunchArgs for ($($ty,)*) {
            type Storage = [u64; $n];
            type Ptrs = [*const c_void; $n];
            #[inline] fn storage() -> Self::Storage { [0u64; $n] }
            #[inline] fn ptrs_init() -> Self::Ptrs { [ptr::null(); $n] }
            #[inline]
            fn pack(self, storage: &mut Self::Storage, ptrs: &mut Self::Ptrs) {
                $(
                    self.$i.write_into(&mut storage[$i]);
                    ptrs[$i] = &storage[$i] as *const u64 as *const c_void;
                )*
            }
        }
    };
}

impl LaunchArgs for () {
    type Storage = [u64; 0];
    type Ptrs = [*const c_void; 0];
    #[inline]
    fn storage() -> Self::Storage {
        []
    }
    #[inline]
    fn ptrs_init() -> Self::Ptrs {
        []
    }
    #[inline]
    fn pack(self, _s: &mut Self::Storage, _p: &mut Self::Ptrs) {}
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
launch_args_tuple!(13; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M);
launch_args_tuple!(14; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M,13=>N);
launch_args_tuple!(15; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M,13=>N,14=>O);
launch_args_tuple!(16; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M,13=>N,14=>O,15=>P);
launch_args_tuple!(17; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M,13=>N,14=>O,15=>P,16=>Q);
launch_args_tuple!(18; 0=>A,1=>B,2=>C,3=>D,4=>E,5=>F,6=>G,7=>H,8=>I,9=>J,10=>K,11=>L,12=>M,13=>N,14=>O,15=>P,16=>Q,17=>R);

// ════════════════════════════════════════════════════════════════════════════
// Graph capture/replay
// ════════════════════════════════════════════════════════════════════════════

pub struct CapturedGraph {
    handle: CUgraph,
    drv: &'static sys::DriverFns,
}

pub struct GraphExec {
    handle: CUgraphExec,
    _device: Arc<Device>,
    drv: &'static sys::DriverFns,
}

impl Stream {
    /// Begin capturing all subsequent work on this stream into a graph.
    /// Uses `ThreadLocal` mode: in-thread stream-ordered ops on *any* stream
    /// implicitly join the capture. For independent cross-stream work during
    /// capture (e.g. allocations on a sibling stream), use
    /// [`Stream::begin_capture_mode`] with
    /// [`CUstreamCaptureMode::Relaxed`].
    pub fn begin_capture(&self) -> Result<()> {
        self.begin_capture_mode(CUstreamCaptureMode::ThreadLocal)
    }

    /// Begin capture with an explicit capture mode. `Relaxed` is required
    /// when other streams in the same thread need to perform non-captured
    /// stream-ordered operations (typically allocations or frees) concurrently.
    pub fn begin_capture_mode(&self, mode: CUstreamCaptureMode) -> Result<()> {
        unsafe {
            check(
                "cuStreamBeginCapture_v2",
                (self.drv.cuStreamBeginCapture_v2)(self.handle, mode),
            )
        }
    }

    /// End capture and return the resulting graph. Call [`GraphExec::new`] to
    /// instantiate an executable.
    pub fn end_capture(&self) -> Result<CapturedGraph> {
        let mut g = CUgraph::default();
        unsafe {
            check(
                "cuStreamEndCapture",
                (self.drv.cuStreamEndCapture)(self.handle, &mut g),
            )?;
        }
        Ok(CapturedGraph {
            handle: g,
            drv: self.drv,
        })
    }
}

impl Drop for CapturedGraph {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.cuGraphDestroy)(self.handle);
        }
    }
}

impl GraphExec {
    pub fn new(graph: CapturedGraph, device: Arc<Device>) -> Result<Self> {
        device.bind()?;
        let d = device.drv;
        let mut exec = CUgraphExec::default();
        unsafe {
            check(
                "cuGraphInstantiateWithFlags",
                (d.cuGraphInstantiateWithFlags)(&mut exec, graph.handle, 0),
            )?;
        }
        // graph is consumed: destroy it now.
        unsafe {
            let _ = (d.cuGraphDestroy)(graph.handle);
        }
        std::mem::forget(graph);
        Ok(Self {
            handle: exec,
            _device: device,
            drv: d,
        })
    }

    #[inline]
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check(
                "cuGraphLaunch",
                (self.drv.cuGraphLaunch)(self.handle, stream.handle),
            )
        }
    }
}

impl Drop for GraphExec {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = (self.drv.cuGraphExecDestroy)(self.handle);
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
        fn check<A: LaunchArgs>() -> usize {
            std::mem::size_of::<A::Storage>()
        }
        assert_eq!(check::<()>(), 0);
        assert_eq!(check::<(u32,)>(), std::mem::size_of::<u64>());
        assert_eq!(check::<(u32, u32, u32)>(), 3 * std::mem::size_of::<u64>());
    }
}
