//! CUPTI — safe entry points for the Activity API.
//!
//! CUPTI is process-global (no per-device handle). We expose enable/disable
//! for the common activity kinds, callback registration, buffer flushing, and
//! the monotonic timestamp helpers used by [`crate::profile`].

use iron_cuda_sys::cupti as sys;
use ironaccelerator_core::{Error, Result};

pub use sys::{
    CuptiActivityKind as ActivityKind, CuptiBufferCompletedCb, CuptiBufferRequestedCb,
    CuptiResult, CuptiSubscriber,
};

fn fns() -> Result<&'static sys::CuptiFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(
        format!("cupti not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: CuptiResult) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub fn enable(kind: ActivityKind) -> Result<()> {
    unsafe { check("cuptiActivityEnable", (fns()?.cuptiActivityEnable)(kind)) }
}

pub fn disable(kind: ActivityKind) -> Result<()> {
    unsafe { check("cuptiActivityDisable", (fns()?.cuptiActivityDisable)(kind)) }
}

pub fn register_callbacks(
    requested: CuptiBufferRequestedCb, completed: CuptiBufferCompletedCb,
) -> Result<()> {
    unsafe {
        check("cuptiActivityRegisterCallbacks",
              (fns()?.cuptiActivityRegisterCallbacks)(requested, completed))
    }
}

/// Flush all pending activity records. `flag=0` flushes everything.
pub fn flush_all(flag: u32) -> Result<()> {
    unsafe { check("cuptiActivityFlushAll", (fns()?.cuptiActivityFlushAll)(flag)) }
}

pub fn version() -> Result<u32> {
    let mut v: u32 = 0;
    unsafe { check("cuptiGetVersion", (fns()?.cuptiGetVersion)(&mut v))?; }
    Ok(v)
}

/// CUPTI monotonic timestamp (nanoseconds since an arbitrary CUPTI epoch).
/// Pairs with [`crate::drv::TimingEvent`] when you need device+host correlation.
pub fn timestamp_ns() -> Result<u64> {
    let mut t: u64 = 0;
    unsafe { check("cuptiGetTimestamp", (fns()?.cuptiGetTimestamp)(&mut t))?; }
    Ok(t)
}

pub fn is_available() -> bool { sys::is_available() }

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
    pub kind: u32,
    pub _pad: u32,
}

/// `CUpti_ActivityKernel4` — Hopper-era kernel record. Layout mirrors the
/// CUPTI header (`cupti_activity.h`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ActivityKernel {
    pub kind: u32,
    pub cache_config: u32,
    pub shared_mem_config: u32,
    pub registers_per_thread: u16,
    pub partitioned_global_cache_requested: u8,
    pub partitioned_global_cache_executed: u8,
    pub start_ns: u64,
    pub end_ns: u64,
    pub completed_ns: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub grid_x: i32,
    pub grid_y: i32,
    pub grid_z: i32,
    pub block_x: i32,
    pub block_y: i32,
    pub block_z: i32,
    pub static_shared_memory: i32,
    pub dynamic_shared_memory: i32,
    pub local_memory_per_thread: u32,
    pub local_memory_total: u32,
    pub correlation_id: u32,
    pub grid_id: i64,
    pub name: *const i8,
    pub reserved0: *mut c_void,
    pub queued_ns: u64,
    pub submitted_ns: u64,
    pub launch_type: u8,
    pub is_shared_mem_carveout_requested: u8,
    pub shared_mem_carveout_requested: u8,
    pub _padding: u8,
    pub shared_mem_limit_config: u32,
}

/// `CUpti_ActivityMemcpy5` — memcpy event.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ActivityMemcpy {
    pub kind: u32,
    pub copy_kind: u8,
    pub src_kind: u8,
    pub dst_kind: u8,
    pub flags: u8,
    pub bytes: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u32,
    pub runtime_correlation_id: u32,
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub channel_id: u32,
    pub channel_type: u32,
    pub copy_count: u32,
}

/// One decoded activity record.
#[derive(Copy, Clone, Debug)]
pub enum Activity {
    Kernel(ActivityKernel),
    Memcpy(ActivityMemcpy),
    /// Unknown kind. Callers can fall back to their own decoder.
    Other { kind: u32, ptr: *const c_void },
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
        Self { buffer, valid_size, cursor: ptr::null_mut() }
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
            k => Activity::Other { kind: k, ptr: self.cursor },
        })
    }
}
