//! Per-stream small-buffer pool for ROCm — the AMD analogue of
//! `ironaccelerator_cuda::pool::MemPool`.
//!
//! A dispatch loop that allocates and frees thousands of small buffers per
//! second spends real time in `hipMallocAsync` / `hipFreeAsync`. [`MemPool`]
//! recycles freed blocks into power-of-two size-class buckets and hands a
//! cached block straight back on the next same-size request, skipping the
//! driver round-trip.
//!
//! ## Scope vs. the CUDA pool, and status
//!
//! This implements the **shared-freelist tier**: each bucket is a
//! `Mutex<Vec<ptr>>` recycling `hipMallocAsync` blocks, with over-cap blocks
//! returned to the driver. The CUDA pool additionally has a lock-free
//! *per-thread front cache* (its ~70×, ~10 ns tier) built on the `thread_local`
//! crate's per-instance storage; that tier is the remaining optimisation here,
//! deferred until there is AMD hardware to tune the bucket sizing against.
//!
//! **Not live-tested** — no AMD GPU in CI. The bucket math is unit-tested; the
//! device path compiles clean and mirrors the validated CUDA design, but has
//! not run on real hardware.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use iron_rocm_sys::hip::{self as sys, HipDeviceptr};

use crate::drv::{Error, Repr, Result, Stream};

fn fns() -> Result<&'static sys::HipFns> {
    sys::fns().map_err(|e| Error::NotAvailable {
        lib: "libamdhip64",
        detail: format!("{e}"),
    })
}

#[inline]
fn check(op: &'static str, r: sys::HipResult) -> Result<()> {
    if r.is_ok() {
        Ok(())
    } else {
        Err(Error::Driver { op, code: r })
    }
}

/// Smallest power-of-two byte size we bucket (1 KiB).
const MIN_BUCKET_LOG2: u32 = 10;
/// Largest power-of-two byte size we bucket (256 MiB).
const MAX_BUCKET_LOG2: u32 = 28;
const NUM_BUCKETS: usize = (MAX_BUCKET_LOG2 - MIN_BUCKET_LOG2 + 1) as usize;
/// Default cap on cached blocks per bucket; overflow returns to the driver.
const DEFAULT_MAX_PER_BUCKET: usize = 32;

/// `ceil(log2(n))` for `n >= 1`.
#[inline]
fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

/// Bucket index for a byte request, or `None` if it is larger than the biggest
/// bucket (those allocations bypass the pool and go straight to the driver).
fn bucket_for_bytes(bytes: usize) -> Option<usize> {
    if bytes == 0 {
        return None;
    }
    let log2 = ceil_log2(bytes).max(MIN_BUCKET_LOG2);
    if log2 > MAX_BUCKET_LOG2 {
        None
    } else {
        Some((log2 - MIN_BUCKET_LOG2) as usize)
    }
}

/// Byte capacity of a bucket.
#[inline]
fn bucket_bytes(bucket: usize) -> usize {
    1usize << (bucket as u32 + MIN_BUCKET_LOG2)
}

struct PoolInner {
    stream: Arc<Stream>,
    buckets: Vec<Mutex<Vec<*mut c_void>>>,
    max_per_bucket: usize,
}

// Raw device pointers are just integers to the host; access is serialised by
// each bucket's mutex and the stream ordering, same as `DeviceBuf`.
unsafe impl Send for PoolInner {}
unsafe impl Sync for PoolInner {}

impl Drop for PoolInner {
    fn drop(&mut self) {
        let Ok(f) = fns() else {
            return;
        };
        let stream = self.stream.raw();
        for bucket in &self.buckets {
            let mut g = bucket.lock().unwrap_or_else(|e| e.into_inner());
            for ptr in g.drain(..) {
                unsafe {
                    let _ = (f.hipFreeAsync)(ptr, stream);
                }
            }
        }
    }
}

/// Per-stream pool of recycled allocations. Cheap to clone (an `Arc` bump);
/// clones share the same buckets.
#[derive(Clone)]
pub struct MemPool {
    inner: Arc<PoolInner>,
}

impl MemPool {
    /// Create a pool bound to `stream` with the default per-bucket cap.
    pub fn new(stream: Arc<Stream>) -> Self {
        Self::with_max_per_bucket(stream, DEFAULT_MAX_PER_BUCKET)
    }

    /// Create a pool with an explicit cap on cached blocks per size bucket.
    pub fn with_max_per_bucket(stream: Arc<Stream>, max_per_bucket: usize) -> Self {
        let buckets = (0..NUM_BUCKETS).map(|_| Mutex::new(Vec::new())).collect();
        Self {
            inner: Arc::new(PoolInner {
                stream,
                buckets,
                max_per_bucket,
            }),
        }
    }

    /// The stream this pool allocates on.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.inner.stream
    }

    /// Allocate a buffer of `len` elements, serving it from a recycled block
    /// when one of the right size class is cached.
    pub fn alloc<T: Repr>(&self, len: usize) -> Result<PooledBuf<T>> {
        let bytes = len * core::mem::size_of::<T>();
        self.inner.stream.device().bind()?;
        let f = fns()?;
        let stream = self.inner.stream.raw();

        let bucket = bucket_for_bytes(bytes);
        let alloc_bytes = match bucket {
            Some(b) => bucket_bytes(b),
            None => bytes.max(1),
        };

        // Try the freelist first for a pooled size class.
        if let Some(b) = bucket {
            if let Some(ptr) = self.inner.buckets[b]
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop()
            {
                return Ok(PooledBuf {
                    inner: self.inner.clone(),
                    ptr,
                    len,
                    bucket,
                    _t: PhantomData,
                });
            }
        }

        let mut ptr: *mut c_void = core::ptr::null_mut();
        check("hipMallocAsync", unsafe {
            (f.hipMallocAsync)(&mut ptr, alloc_bytes, stream)
        })?;
        Ok(PooledBuf {
            inner: self.inner.clone(),
            ptr,
            len,
            bucket,
            _t: PhantomData,
        })
    }
}

/// A pooled device allocation. Returns its block to the pool on drop (or to the
/// driver if the bucket is over cap or the allocation bypassed the pool).
pub struct PooledBuf<T: Repr> {
    inner: Arc<PoolInner>,
    ptr: *mut c_void,
    len: usize,
    bucket: Option<usize>,
    _t: PhantomData<T>,
}

unsafe impl<T: Repr> Send for PooledBuf<T> {}
unsafe impl<T: Repr> Sync for PooledBuf<T> {}

impl<T: Repr> PooledBuf<T> {
    /// Raw device pointer.
    #[inline]
    pub fn device_ptr(&self) -> HipDeviceptr {
        self.ptr as usize as HipDeviceptr
    }

    /// Element count requested at allocation.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Requested size in bytes (not the rounded-up bucket capacity).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }

    /// Copy `src` into this buffer and wait for the transfer.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() > self.len {
            return Err(Error::Precondition {
                op: "PooledBuf::copy_from_host",
                msg: "source longer than buffer".into(),
            });
        }
        let f = fns()?;
        self.inner.stream.device().bind()?;
        check("hipMemcpyHtoDAsync", unsafe {
            (f.hipMemcpyHtoDAsync)(
                self.device_ptr(),
                src.as_ptr() as *const c_void,
                core::mem::size_of_val(src),
                self.inner.stream.raw(),
            )
        })?;
        self.inner.stream.synchronize()
    }

    /// Copy this buffer back into `dst` and wait for the transfer.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() > self.len {
            return Err(Error::Precondition {
                op: "PooledBuf::copy_to_host",
                msg: "destination longer than buffer".into(),
            });
        }
        let f = fns()?;
        self.inner.stream.device().bind()?;
        check("hipMemcpyDtoHAsync", unsafe {
            (f.hipMemcpyDtoHAsync)(
                dst.as_mut_ptr() as *mut c_void,
                self.device_ptr(),
                core::mem::size_of_val(dst),
                self.inner.stream.raw(),
            )
        })?;
        self.inner.stream.synchronize()
    }
}

impl<T: Repr> Drop for PooledBuf<T> {
    fn drop(&mut self) {
        if let Some(b) = self.bucket {
            let mut g = self.inner.buckets[b]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if g.len() < self.inner.max_per_bucket {
                g.push(self.ptr);
                return;
            }
        }
        // Over-cap, or a bypass allocation: return it to the driver.
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.hipFreeAsync)(self.ptr, self.inner.stream.raw());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_definition() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1024), 10);
        assert_eq!(ceil_log2(1025), 11);
    }

    #[test]
    fn buckets_round_up_and_cap_out() {
        // Anything <= 1 KiB lands in bucket 0 (1 KiB).
        assert_eq!(bucket_for_bytes(1), Some(0));
        assert_eq!(bucket_for_bytes(1024), Some(0));
        assert_eq!(bucket_bytes(0), 1024);
        // 1 KiB + 1 rounds up to the 2 KiB bucket.
        assert_eq!(bucket_for_bytes(1025), Some(1));
        assert_eq!(bucket_bytes(1), 2048);
        // Exactly the biggest bucket is still pooled.
        assert_eq!(bucket_for_bytes(1 << 28), Some(NUM_BUCKETS - 1));
        // Larger than the biggest bucket bypasses the pool.
        assert_eq!(bucket_for_bytes((1 << 28) + 1), None);
        // Zero never buckets.
        assert_eq!(bucket_for_bytes(0), None);
    }
}
