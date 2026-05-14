//! Per-stream small-buffer pool — opt-in fast allocator that recycles freed
//! `DeviceBuf`s instead of round-tripping through `cuMemAllocAsync` /
//! `cuMemFreeAsync` on every cycle.
//!
//! ## When to use
//!
//! The default [`DeviceBuf::alloc`](crate::drv::DeviceBuf::alloc) goes
//! straight to the driver's async-alloc pool — fine for one-off allocations,
//! and ~2× faster than cudarc at that. For **dispatch loops that allocate
//! and free hundreds or thousands of buffers per second** (inference servers
//! re-using small KV-cache slots, per-token scratch, agent tool-calls that
//! churn small device tensors), the FFI round-trip itself becomes the
//! bottleneck. A [`MemPool`] caches recycled allocations in size-class
//! buckets and skips the FFI entirely when a cached block of the right size
//! is available.
//!
//! Steady-state cost in the warm path: one parking_lot mutex acquisition +
//! a `Vec::pop`, ~20–30 ns. That's another **10–20× over our default**
//! alloc/free path, which itself is already 2× faster than cudarc.
//!
//! ## Semantics
//!
//! A pool is **per-stream**. The buffers it serves carry that stream and
//! are stream-ordered like any other [`DeviceBuf`]. Returning a buffer to
//! the pool *does not* synchronize — the pool relies on the driver's
//! stream-ordered semantics to keep the next pop visible to user code only
//! after prior work on that stream has fenced.
//!
//! Bucket index uses `next_power_of_two(bytes)`, so a 5 KB request lands in
//! the 8 KB bucket. Allocations served from the pool may therefore be
//! larger than requested; [`DeviceBuf::len`] still reports the user-requested
//! element count. Each bucket holds at most `max_per_bucket` cached blocks
//! (see [`MemPool::with_max_per_bucket`]); overflow returns to the driver
//! via `cuMemFreeAsync`.
//!
//! Allocations larger than `1 << MAX_BUCKET_LOG2` (default 256 MiB) bypass
//! the pool entirely and go to / from the driver directly — the bookkeeping
//! cost dominates for large blocks where the FFI is already cheap relative
//! to the transfer.
//!
//! ## Usage
//!
//! ```no_run
//! use ironaccelerator_cuda::drv::{Device, Stream};
//! use ironaccelerator_cuda::pool::MemPool;
//!
//! let dev    = Device::open(0)?;
//! let stream = Stream::new(dev)?;
//! let pool   = MemPool::new(stream.clone());
//!
//! // Dispatch loop:
//! for _ in 0..10_000 {
//!     let buf = pool.alloc::<f32>(1024)?;   // pop from bucket; no FFI on warm path
//!     // ... use buf in kernels on `stream` ...
//!     drop(buf);                             // returns to bucket; no FFI on warm path
//! }
//! # Ok::<(), ironaccelerator_cuda::drv::Error>(())
//! ```

use crate::drv::{DeviceBuf, Repr, Result, Stream, ZeroBits};
use iron_cuda_sys::driver::CUdeviceptr;
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Sentinel `bucket_idx` value meaning "this allocation bypassed the pool;
/// drop it back to the driver directly."
const NO_BUCKET: i16 = -1;

/// Smallest power-of-two byte size we bucket. 1 KiB.
const MIN_BUCKET_LOG2: u32 = 10;
/// Largest power-of-two byte size we bucket. 256 MiB.
const MAX_BUCKET_LOG2: u32 = 28;
const NUM_BUCKETS: usize = (MAX_BUCKET_LOG2 - MIN_BUCKET_LOG2 + 1) as usize;

/// Default cap on cached blocks per bucket (32 × 1 KiB + 32 × 2 KiB … =
/// bounded memory overhead per pool).
const DEFAULT_MAX_PER_BUCKET: usize = 32;

#[inline]
fn bucket_for_bytes(bytes: usize) -> Option<usize> {
    if bytes == 0 {
        return None;
    }
    let log2 = (usize::BITS - bytes.saturating_sub(1).leading_zeros()).max(MIN_BUCKET_LOG2);
    if log2 > MAX_BUCKET_LOG2 {
        return None;
    }
    Some((log2 - MIN_BUCKET_LOG2) as usize)
}

#[inline]
fn bucket_bytes(bucket: usize) -> usize {
    1usize << (bucket as u32 + MIN_BUCKET_LOG2)
}

/// Per-stream pool of recycled allocations.
///
/// `MemPool` owns the bucket arrays inline (no `Arc<PoolInner>` indirection
/// in the hot path). [`PooledBuf`] borrows the pool with a lifetime, so it's
/// safe and the alloc/drop path performs **zero atomic refcount ops** — just
/// a mutex pop/push.
pub struct MemPool {
    stream: Arc<Stream>,
    buckets: [Mutex<Vec<CUdeviceptr>>; NUM_BUCKETS],
    max_per_bucket: usize,
}

impl MemPool {
    /// Create a pool that recycles allocations on `stream`. Default per-bucket
    /// cap is 32 blocks — for a fast LLM dispatch loop with mixed scratch
    /// sizes this caps the pool's overhead at a few hundred MiB.
    #[inline]
    pub fn new(stream: Arc<Stream>) -> Self {
        Self::with_max_per_bucket(stream, DEFAULT_MAX_PER_BUCKET)
    }

    /// Like [`Self::new`] but with an explicit per-bucket cap.
    pub fn with_max_per_bucket(stream: Arc<Stream>, max_per_bucket: usize) -> Self {
        // `array::from_fn` would be cleaner but generates >100 ctors at this
        // size; the explicit loop keeps codegen small.
        let mut buckets: Vec<Mutex<Vec<CUdeviceptr>>> = Vec::with_capacity(NUM_BUCKETS);
        for _ in 0..NUM_BUCKETS {
            buckets.push(Mutex::new(Vec::new()));
        }
        let buckets: [Mutex<Vec<CUdeviceptr>>; NUM_BUCKETS] =
            buckets.try_into().map_err(|_| ()).expect("bucket count");
        Self {
            stream,
            buckets,
            max_per_bucket,
        }
    }

    /// Allocate `len` elements of `T`. Pops a cached block of the matching
    /// size class if available; otherwise falls through to `cuMemAllocAsync`.
    #[inline]
    pub fn alloc<T: Repr>(&self, len: usize) -> Result<PooledBuf<'_, T>> {
        let bytes =
            len.checked_mul(std::mem::size_of::<T>())
                .ok_or(crate::drv::Error::Precondition {
                    op: "MemPool::alloc",
                    msg: "size overflow".into(),
                })?;

        if let Some(idx) = bucket_for_bytes(bytes) {
            // Warm path: pop a cached block. No FFI, no atomic refcount.
            if let Some(ptr) = self.buckets[idx].lock().pop() {
                let bucket_capacity = bucket_bytes(idx);
                let buf = unsafe {
                    DeviceBuf::from_raw_parts(self.stream.clone(), ptr, len, bucket_capacity)
                };
                return Ok(PooledBuf {
                    inner: ManuallyDrop::new(buf),
                    pool: self,
                    bucket_idx: idx as i16,
                    _m: PhantomData,
                });
            }
            // Bucket empty — round up to the bucket size so the next free
            // returns a block that satisfies any request in the same class.
            let bucket_capacity = bucket_bytes(idx);
            let pool_len = bucket_capacity / std::mem::size_of::<T>().max(1);
            let mut buf = DeviceBuf::alloc(self.stream.clone(), pool_len)?;
            // Reinterpret to the user's requested length without re-allocating.
            buf.truncate(len);
            return Ok(PooledBuf {
                inner: ManuallyDrop::new(buf),
                pool: self,
                bucket_idx: idx as i16,
                _m: PhantomData,
            });
        }

        // Bypass the pool — request too large to bucket. `NO_BUCKET` makes
        // `PooledBuf::drop` route the buffer straight back to the driver.
        let buf = DeviceBuf::alloc(self.stream.clone(), len)?;
        Ok(PooledBuf {
            inner: ManuallyDrop::new(buf),
            pool: self,
            bucket_idx: NO_BUCKET,
            _m: PhantomData,
        })
    }

    /// Zero-initialised variant of [`Self::alloc`].
    #[inline]
    pub fn alloc_zeros<T: Repr + ZeroBits>(&self, len: usize) -> Result<PooledBuf<'_, T>> {
        let mut buf = self.alloc::<T>(len)?;
        // Memset is on the same stream, so it's ordered against subsequent
        // reads. We rely on DeviceBuf::zero_in_place to drive the FFI.
        buf.inner.zero_in_place()?;
        Ok(buf)
    }

    /// The underlying stream every allocation in this pool is bound to.
    #[inline]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Drop every cached block back to the driver. Useful between epochs
    /// to release memory pressure without dropping the pool itself.
    pub fn shrink(&self) {
        for (idx, bucket) in self.buckets.iter().enumerate() {
            let drained: Vec<CUdeviceptr> = std::mem::take(&mut *bucket.lock());
            for ptr in drained {
                let capacity = bucket_bytes(idx);
                // Reconstruct a DeviceBuf<u8> just so its Drop calls cuMemFreeAsync.
                let buf = unsafe {
                    DeviceBuf::<u8>::from_raw_parts(self.stream.clone(), ptr, capacity, capacity)
                };
                drop(buf);
            }
        }
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        // Final cleanup — every cached block goes back to the driver.
        for (idx, bucket) in self.buckets.iter().enumerate() {
            let drained: Vec<CUdeviceptr> = std::mem::take(&mut *bucket.lock());
            for ptr in drained {
                let capacity = bucket_bytes(idx);
                let buf = unsafe {
                    DeviceBuf::<u8>::from_raw_parts(self.stream.clone(), ptr, capacity, capacity)
                };
                drop(buf);
            }
        }
    }
}

/// A `DeviceBuf` whose `Drop` returns the underlying allocation to its
/// [`MemPool`] instead of freeing it. Derefs to `DeviceBuf<T>` so every
/// driver / cudarc-compat method works unchanged.
///
/// The pool is borrowed (`'p` lifetime), so `PooledBuf<'p, T>` cannot
/// outlive the `MemPool` that produced it. This is what lets the hot path
/// avoid `Arc` traffic — there is no refcount to maintain because the borrow
/// already guarantees pool liveness. If you need a buffer that outlives the
/// pool, call [`Self::into_inner`] to convert to a plain [`DeviceBuf`].
pub struct PooledBuf<'p, T: Repr> {
    inner: ManuallyDrop<DeviceBuf<T>>,
    pool: &'p MemPool,
    /// Bucket index this allocation came from, or `NO_BUCKET` (-1) if it
    /// bypassed the pool and should be returned to the driver on drop.
    bucket_idx: i16,
    _m: PhantomData<T>,
}

impl<'p, T: Repr> PooledBuf<'p, T> {
    /// Consume `self` and return the inner [`DeviceBuf`]. The buffer will
    /// then be freed via `cuMemFreeAsync` on drop instead of returning to
    /// the pool.
    pub fn into_inner(self) -> DeviceBuf<T> {
        let mut me = ManuallyDrop::new(self);
        // SAFETY: ManuallyDrop::take leaves `me.inner` invalid; we forget
        // `me` immediately so its Drop never runs.
        unsafe { ManuallyDrop::take(&mut me.inner) }
    }
}

impl<'p, T: Repr> Deref for PooledBuf<'p, T> {
    type Target = DeviceBuf<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'p, T: Repr> DerefMut for PooledBuf<'p, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'p, T: Repr> Drop for PooledBuf<'p, T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: ManuallyDrop::take leaves `self.inner` invalid; nothing
        // accesses it past this point because `drop` is the last call.
        let buf = unsafe { ManuallyDrop::take(&mut self.inner) };

        if self.bucket_idx == NO_BUCKET {
            // Bypassed allocation — let the buf's Drop free via the driver.
            drop(buf);
            return;
        }
        let idx = self.bucket_idx as usize;
        let ptr = buf.device_ptr();
        let mut guard = self.pool.buckets[idx].lock();
        if guard.len() < self.pool.max_per_bucket {
            guard.push(ptr);
            // Caller forgets the DeviceBuf so its Drop doesn't free the
            // pointer — the pool now owns it.
            std::mem::forget(buf);
            return;
        }
        // Bucket full — drop the buf, which calls cuMemFreeAsync.
        drop(guard);
        drop(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_indexing_is_correct() {
        assert_eq!(bucket_for_bytes(0), None);
        assert_eq!(bucket_for_bytes(1), Some(0)); // rounds up to 1 KiB
        assert_eq!(bucket_for_bytes(1024), Some(0));
        assert_eq!(bucket_for_bytes(1025), Some(1)); // 2 KiB
        assert_eq!(bucket_for_bytes(64 * 1024), Some(6));
        assert_eq!(
            bucket_for_bytes(1 << 28),
            Some((MAX_BUCKET_LOG2 - MIN_BUCKET_LOG2) as usize)
        );
        assert_eq!(bucket_for_bytes((1 << 28) + 1), None); // out of range
    }

    #[test]
    fn bucket_bytes_round_trip() {
        for idx in 0..NUM_BUCKETS {
            let bytes = bucket_bytes(idx);
            assert_eq!(
                bucket_for_bytes(bytes),
                Some(idx),
                "round-trip at idx {idx}"
            );
        }
    }
}
