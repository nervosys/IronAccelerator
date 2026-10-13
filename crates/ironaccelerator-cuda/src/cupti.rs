//! CUPTI — safe entry points for the Activity API.
//!
//! CUPTI is process-global (no per-device handle). We expose enable/disable
//! for the common activity kinds, callback registration, buffer flushing, and
//! the monotonic timestamp helpers used by [`crate::profile`].

use iron_cuda_sys::cupti as sys;
use ironaccelerator_core::{Error, Result};

pub use sys::{
    CuptiActivityKind as ActivityKind, CuptiBufferCompletedCb, CuptiBufferRequestedCb, CuptiResult,
    CuptiSubscriber,
};

fn fns() -> Result<&'static sys::CuptiFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cupti not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: CuptiResult) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

/// Enables activity recording for `kind` (`cuptiActivityEnable`).
pub fn enable(kind: ActivityKind) -> Result<()> {
    unsafe { check("cuptiActivityEnable", (fns()?.cuptiActivityEnable)(kind)) }
}

/// Disables activity recording for `kind` (`cuptiActivityDisable`).
pub fn disable(kind: ActivityKind) -> Result<()> {
    unsafe { check("cuptiActivityDisable", (fns()?.cuptiActivityDisable)(kind)) }
}

/// Registers the buffer-requested and buffer-completed callbacks (`cuptiActivityRegisterCallbacks`).
pub fn register_callbacks(
    requested: CuptiBufferRequestedCb,
    completed: CuptiBufferCompletedCb,
) -> Result<()> {
    unsafe {
        check(
            "cuptiActivityRegisterCallbacks",
            (fns()?.cuptiActivityRegisterCallbacks)(requested, completed),
        )
    }
}

/// Flush all pending activity records. `flag=0` flushes everything.
pub fn flush_all(flag: u32) -> Result<()> {
    unsafe {
        check(
            "cuptiActivityFlushAll",
            (fns()?.cuptiActivityFlushAll)(flag),
        )
    }
}

/// Returns the CUPTI version (`cuptiGetVersion`).
pub fn version() -> Result<u32> {
    let mut v: u32 = 0;
    unsafe {
        check("cuptiGetVersion", (fns()?.cuptiGetVersion)(&mut v))?;
    }
    Ok(v)
}

/// CUPTI monotonic timestamp (nanoseconds since an arbitrary CUPTI epoch).
/// Pairs with [`crate::drv::TimingEvent`] when you need device+host correlation.
pub fn timestamp_ns() -> Result<u64> {
    let mut t: u64 = 0;
    unsafe {
        check("cuptiGetTimestamp", (fns()?.cuptiGetTimestamp)(&mut t))?;
    }
    Ok(t)
}

/// Returns `true` if the CUPTI library loads at runtime.
pub fn is_available() -> bool {
    sys::is_available()
}

// ─── Activity record decoder ────────────────────────────────────────────────
//
// Activity records share a common prefix (`kind: u32`), but the per-kind
// payloads differ. We bind the two kinds used for kernel/memcpy tracing,
// which covers ~all GPU timeline data. Unknown kinds are returned opaquely
// so callers can either skip or decode them themselves.

use std::ffi::c_void;
use std::ptr;

/// Common prefix read directly from any record to dispatch by kind.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ActivityHeader {
    /// Activity kind tag (`CUpti_ActivityKind`) used to dispatch the record layout.
    pub kind: u32,
    /// Padding that keeps the header 8 bytes wide; not read.
    pub _pad: u32,
}

/// `CUpti_ActivityKernel4` — Hopper-era kernel record. Layout mirrors the
/// CUPTI header (`cupti_activity.h`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ActivityKernel {
    /// Activity kind tag (`CUpti_ActivityKind`); kernel records use 3 or 10.
    pub kind: u32,
    /// Cache preference (`CUpti_FuncCachePreference`) requested by the launch, as the raw value.
    pub cache_config: u32,
    /// Shared-memory bank configuration (`CUpti_SharedMemoryConfig`), as the raw value.
    pub shared_mem_config: u32,
    /// Registers used per thread.
    pub registers_per_thread: u16,
    /// Requested partitioned global cache configuration, as the raw CUPTI value.
    pub partitioned_global_cache_requested: u8,
    /// Partitioned global cache configuration the kernel ran with, as the raw CUPTI value.
    pub partitioned_global_cache_executed: u8,
    /// Kernel start timestamp, in nanoseconds on the CUPTI timebase.
    pub start_ns: u64,
    /// Kernel end timestamp, in nanoseconds on the CUPTI timebase.
    pub end_ns: u64,
    /// Timestamp, in nanoseconds, at which CUPTI recorded completion.
    pub completed_ns: u64,
    /// Device ordinal the kernel ran on.
    pub device_id: u32,
    /// CUDA context ID the kernel was launched in.
    pub context_id: u32,
    /// Stream ID the kernel was launched on.
    pub stream_id: u32,
    /// Grid dimension X, in blocks.
    pub grid_x: i32,
    /// Grid dimension Y, in blocks.
    pub grid_y: i32,
    /// Grid dimension Z, in blocks.
    pub grid_z: i32,
    /// Block dimension X, in threads.
    pub block_x: i32,
    /// Block dimension Y, in threads.
    pub block_y: i32,
    /// Block dimension Z, in threads.
    pub block_z: i32,
    /// Statically allocated shared memory per block, in bytes.
    pub static_shared_memory: i32,
    /// Dynamically allocated shared memory per block, in bytes.
    pub dynamic_shared_memory: i32,
    /// Local memory per thread, in bytes.
    pub local_memory_per_thread: u32,
    /// Total local memory for the launch, in bytes.
    pub local_memory_total: u32,
    /// Correlation ID linking this record to its runtime or driver API call.
    pub correlation_id: u32,
    /// Unique ID of this grid launch.
    pub grid_id: i64,
    /// Pointer to the NUL-terminated kernel name; owned by CUPTI and valid only while the buffer is.
    pub name: *const i8,
    /// Reserved by CUPTI; not interpreted.
    pub reserved0: *mut c_void,
    /// Timestamp, in nanoseconds, at which the launch was queued.
    pub queued_ns: u64,
    /// Timestamp, in nanoseconds, at which the launch was submitted to the device.
    pub submitted_ns: u64,
    /// Launch type, as the raw CUPTI value.
    pub launch_type: u8,
    /// Non-zero if a shared-memory carveout was requested.
    pub is_shared_mem_carveout_requested: u8,
    /// Requested shared-memory carveout, as the raw CUPTI value.
    pub shared_mem_carveout_requested: u8,
    /// Padding byte; not read.
    pub _padding: u8,
    /// Shared-memory limit configuration, as the raw CUPTI value.
    pub shared_mem_limit_config: u32,
}

/// `CUpti_ActivityMemcpy5` — memcpy event.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ActivityMemcpy {
    /// Activity kind tag (`CUpti_ActivityKind`); memcpy records use 1.
    pub kind: u32,
    /// Copy direction, as the raw `CUpti_ActivityMemcpyKind` value.
    pub copy_kind: u8,
    /// Source memory kind, as the raw `CUpti_ActivityMemoryKind` value.
    pub src_kind: u8,
    /// Destination memory kind, as the raw `CUpti_ActivityMemoryKind` value.
    pub dst_kind: u8,
    /// Raw copy flags.
    pub flags: u8,
    /// Number of bytes copied.
    pub bytes: u64,
    /// Copy start timestamp, in nanoseconds on the CUPTI timebase.
    pub start_ns: u64,
    /// Copy end timestamp, in nanoseconds on the CUPTI timebase.
    pub end_ns: u64,
    /// Device ordinal involved in the copy.
    pub device_id: u32,
    /// CUDA context ID the copy was issued in.
    pub context_id: u32,
    /// Stream ID the copy was issued on.
    pub stream_id: u32,
    /// Correlation ID linking this record to its runtime or driver API call.
    pub correlation_id: u32,
    /// Correlation ID of the runtime API call that issued the copy.
    pub runtime_correlation_id: u32,
    /// CUDA graph node ID associated with the copy, if any.
    pub graph_node_id: u64,
    /// CUDA graph ID associated with the copy, if any.
    pub graph_id: u32,
    /// Channel ID the copy used.
    pub channel_id: u32,
    /// Channel type, as the raw CUPTI value.
    pub channel_type: u32,
    /// Copy count reported by CUPTI for this record.
    pub copy_count: u32,
}

/// One decoded activity record.
#[derive(Copy, Clone, Debug)]
pub enum Activity {
    /// Kernel or concurrent-kernel record (kinds 3 and 10).
    Kernel(ActivityKernel),
    /// Memcpy record (kind 1).
    Memcpy(ActivityMemcpy),
    /// Unknown kind. Callers can fall back to their own decoder.
    Other {
        /// Raw activity kind.
        kind: u32,
        /// Pointer to the unparsed record inside the caller's buffer.
        ptr: *const c_void,
    },
}

/// Pull-style decoder over a completed activity buffer.
///
/// # Safety
/// `buffer` and `valid_size` must be the exact arguments received in the
/// `CuptiBufferCompletedCb`; the buffer must outlive the iterator.
pub struct ActivityDecoder {
    buffer: *mut u8,
    valid_size: usize,
    cursor: *mut c_void,
}

impl ActivityDecoder {
    /// # Safety
    /// See [`ActivityDecoder`].
    pub unsafe fn new(buffer: *mut u8, valid_size: usize) -> Self {
        Self {
            buffer,
            valid_size,
            cursor: ptr::null_mut(),
        }
    }
}

impl Iterator for ActivityDecoder {
    type Item = Activity;

    fn next(&mut self) -> Option<Activity> {
        let f = sys::fns().ok()?;
        let status = unsafe {
            (f.cuptiActivityGetNextRecord)(self.buffer, self.valid_size, &mut self.cursor)
        };
        if !status.is_ok() {
            return None;
        }
        // SAFETY: cursor now points into the caller-owned buffer; read the
        // kind discriminant to pick a layout.
        let hdr = unsafe { &*(self.cursor as *const ActivityHeader) };
        Some(match hdr.kind {
            // CUpti_ActivityKind::Kernel (3) | ConcurrentKernel (10)
            3 | 10 => Activity::Kernel(unsafe { *(self.cursor as *const ActivityKernel) }),
            // CUpti_ActivityKind::Memcpy (1)
            1 => Activity::Memcpy(unsafe { *(self.cursor as *const ActivityMemcpy) }),
            k => Activity::Other {
                kind: k,
                ptr: self.cursor,
            },
        })
    }
}
