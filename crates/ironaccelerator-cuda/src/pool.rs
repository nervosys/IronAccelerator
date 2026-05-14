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
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use thread_local::ThreadLocal;

/// Per-thread storage cell. `UnsafeCell` skips `RefCell`'s borrow-counter
/// check on the hot path; the `thread_local::ThreadLocal` outside guarantees
/// only one thread ever reaches a given `ThreadLocalCell`, so the unchecked
/// access is sound.
#[repr(transparent)]
struct ThreadLocalCell<T>(UnsafeCell<T>);

// SAFETY: `ThreadLocal<ThreadLocalCell<T>>` hands out one cell per thread;
// no two threads share the same cell. Adding `Sync` to the cell type itself
// is the contract we owe `thread_local`'s `get_or`.
unsafe impl<T: Send> Sync for ThreadLocalCell<T> {}

impl<T> ThreadLocalCell<T> {
    #[inline]
    fn new(v: T) -> Self {
        Self(UnsafeCell::new(v))
    }
    /// # Safety
    /// Caller must guarantee no other reference to the inner T exists. The
    /// `thread_local::ThreadLocal` containing this cell already guarantees
    /// single-thread access; not aliasing within the thread is the caller's
    /// duty (we only ever do one short-lived `&mut` per alloc/free).
    #[inline]
    #[allow(clippy::mut_from_ref)] // this is the entire point of UnsafeCell.
    unsafe fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}

/// Sentinel `bucket_idx` value meaning "this allocation bypassed the pool;
/// drop it back to the driver directly."
const NO_BUCKET: i16 = -1;

/// Capacity of the per-thread front cache for each bucket. Small so the whole
/// front cache fits in a cache line per bucket.
const FRONT_CAP: usize = 4;

/// Per-thread, per-pool front-cache: a small fixed-size stack of cached
/// pointers for each bucket, accessed without any locking. Misses fall
/// through to the pool's shared `Mutex<Vec<...>>` back cache.
struct FrontCache {
    /// Per-bucket `(len, [CUdeviceptr; FRONT_CAP])` stack. We use an
    /// explicit `len` rather than `Vec` to avoid touching the allocator on
    /// the hot path and to keep each bucket's state in one cache line.
    buckets: [(u8, [CUdeviceptr; FRONT_CAP]); NUM_BUCKETS],
}

impl FrontCache {
    fn new() -> Self {
        Self {
            buckets: [(0, [0; FRONT_CAP]); NUM_BUCKETS],
        }
    }
}

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
    /// Per-thread, per-bucket lock-free front caches. The first alloc/free
    /// from a thread for this pool initialises the thread's `FrontCache`.
    /// Hits skip the mutex entirely; misses fall through to the shared
    /// back cache below.
    front: ThreadLocal<ThreadLocalCell<FrontCache>>,
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
        // Infallible: `array::from_fn` populates the array element-by-element
        // with no panics, no allocation churn.
        Self {
            stream,
            front: ThreadLocal::new(),
            buckets: std::array::from_fn(|_| Mutex::new(Vec::new())),
            max_per_bucket,
        }
    }

    /// Get or initialise the calling thread's front cache for this pool.
    /// Returns a mutable reference; safe because the `ThreadLocal` storage
    /// hands out one cell per thread, so this `&mut` cannot alias across
    /// threads, and within a thread we only hold it for the body of one
    /// alloc/free.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn front_cache(&self) -> &mut FrontCache {
        let cell = self
            .front
            .get_or(|| ThreadLocalCell::new(FrontCache::new()));
        // SAFETY: this cell is unique to the calling thread (thread_local
        // invariant). We hold the returned `&mut` for the duration of the
        // alloc/free body only, no aliasing.
        unsafe { cell.get_mut() }
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
            let bucket_capacity = bucket_bytes(idx);

            // Tier 1 — thread-local front cache. No lock, no borrow check.
            {
                let front = self.front_cache();
                let (flen, slots) = &mut front.buckets[idx];
                if *flen > 0 {
                    *flen -= 1;
                    let ptr = slots[*flen as usize];
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
            }

            // Tier 2 — shared mutex'd back cache.
            if let Some(ptr) = self.buckets[idx].lock().pop() {
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

            // Tier 3 — bucket empty everywhere; ask the driver.
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

    /// Drop every cached block (both per-thread front caches and the
    /// shared back cache) back to the driver. Useful between epochs to
    /// release memory pressure without dropping the pool itself.
    ///
    /// Takes `&mut self` because draining the per-thread front caches
    /// requires exclusive access to the `ThreadLocal` storage.
    pub fn shrink(&mut self) {
        // Drain every thread's front cache. We hold `&mut self` so no
        // other thread can be mid-access through `front_cache`.
        for front_cell in self.front.iter_mut() {
            let front: &mut FrontCache = front_cell.0.get_mut();
            for (idx, (flen, slots)) in front.buckets.iter_mut().enumerate() {
                let capacity = bucket_bytes(idx);
                for ptr in &slots[..*flen as usize] {
                    let buf = unsafe {
                        DeviceBuf::<u8>::from_raw_parts(
                            self.stream.clone(),
                            *ptr,
                            capacity,
                            capacity,
                        )
                    };
                    drop(buf);
                }
                *flen = 0;
            }
        }
        // Drain the shared back cache.
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

impl Drop for MemPool {
    fn drop(&mut self) {
        // Final cleanup — drain per-thread front caches and the shared back
        // cache; every cached block goes back to the driver.
        self.shrink();
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
        let mut buf = unsafe { ManuallyDrop::take(&mut self.inner) };

        if self.bucket_idx == NO_BUCKET {
            // Bypassed allocation — let the buf's Drop free via the driver.
            drop(buf);
            return;
        }
        let idx = self.bucket_idx as usize;

        // Tier 1 — push to the thread-local front cache. No lock, no borrow check.
        //
        // We `detach_ptr` instead of `mem::forget(buf)` so the `Arc<Stream>`
        // inside the buffer still gets its refcount decrement — otherwise
        // every alloc/free cycle would leak one `Arc::clone` increment.
        {
            let front = self.pool.front_cache();
            let (flen, slots) = &mut front.buckets[idx];
            if (*flen as usize) < FRONT_CAP {
                slots[*flen as usize] = unsafe { buf.detach_ptr() };
                *flen += 1;
                drop(buf); // Arc<Stream> dec, no FFI (ptr is now 0)
                return;
            }
        }

        // Tier 2 — front full; push to the shared back cache.
        let mut guard = self.pool.buckets[idx].lock();
        if guard.len() < self.pool.max_per_bucket {
            guard.push(unsafe { buf.detach_ptr() });
            drop(guard);
            drop(buf); // Arc<Stream> dec, no FFI
            return;
        }

        // Tier 3 — back full too; let the driver reclaim.
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
