//! Pinned (page-locked) host memory pool, built on [`crate::drv::PinnedBuf`].

use crate::drv::{Device, PinnedBuf};
use ironaccelerator_core::Result;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const MIN_BUCKET_LOG2: u32 = 12;
const MAX_BUCKET_LOG2: u32 = 34;
const NUM_BUCKETS: usize = (MAX_BUCKET_LOG2 - MIN_BUCKET_LOG2 + 1) as usize;

/// Size-bucketed pool of pinned host allocations, one bucket per power of two.
pub struct PinnedPool {
    device: Arc<Device>,
    buckets: Vec<Mutex<Vec<PinnedBuf<u8>>>>,
    allocated_bytes: AtomicU64,
    live_slabs: AtomicU64,
}

impl PinnedPool {
    /// Create an empty pool that allocates pinned memory on `device`.
    pub fn new(device: Arc<Device>) -> Self {
        let buckets = (0..NUM_BUCKETS).map(|_| Mutex::new(Vec::new())).collect();
        Self {
            device,
            buckets,
            allocated_bytes: AtomicU64::new(0),
            live_slabs: AtomicU64::new(0),
        }
    }

    /// Cumulative bytes this pool has allocated from the driver. Never decremented.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
    /// Number of slabs this pool has allocated from the driver. Never decremented.
    pub fn live_slabs(&self) -> u64 {
        self.live_slabs.load(Ordering::Relaxed)
    }

    /// Take a slab with capacity for at least `bytes`, reusing a pooled slab
    /// when one is available. Returns the driver error if a new pinned
    /// allocation fails. The slab returns to the pool when dropped.
    pub fn acquire(self: &Arc<Self>, bytes: usize) -> Result<PinnedSlab> {
        let (bucket_idx, cap) = bucket_of(bytes);
        let slab = self.buckets[bucket_idx].lock().pop();
        let slab = match slab {
            Some(s) => s,
            None => {
                let s = PinnedBuf::<u8>::alloc(self.device.clone(), cap)?;
                self.allocated_bytes
                    .fetch_add(cap as u64, Ordering::Relaxed);
                self.live_slabs.fetch_add(1, Ordering::Relaxed);
                s
            }
        };
        Ok(PinnedSlab {
            pool: Some(self.clone()),
            bucket: bucket_idx,
            slab: Some(slab),
            requested: bytes,
        })
    }
}

/// Pinned host buffer checked out of a [`PinnedPool`]. Returned to the pool on drop.
pub struct PinnedSlab {
    pool: Option<Arc<PinnedPool>>,
    bucket: usize,
    slab: Option<PinnedBuf<u8>>,
    requested: usize,
}

impl PinnedSlab {
    /// Size in bytes originally requested from [`PinnedPool::acquire`].
    #[inline]
    pub fn requested(&self) -> usize {
        self.requested
    }
    /// Allocated capacity in bytes (the rounded bucket size).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slab.as_ref().map(|s| s.len()).unwrap_or(0)
    }
    /// View the slab's bytes. Panics if the slab was already released.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.slab.as_ref().expect("slab not released").as_slice()
    }
    /// Mutable view of the slab's bytes. Panics if the slab was already released.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.slab
            .as_mut()
            .expect("slab not released")
            .as_mut_slice()
    }
}

impl Drop for PinnedSlab {
    fn drop(&mut self) {
        if let (Some(pool), Some(slab)) = (self.pool.take(), self.slab.take()) {
            pool.buckets[self.bucket].lock().push(slab);
        }
    }
}

fn bucket_of(bytes: usize) -> (usize, usize) {
    let min_cap = 1usize << MIN_BUCKET_LOG2;
    let cap = bytes.max(min_cap).next_power_of_two();
    let log2 = cap.trailing_zeros();
    let idx = (log2.saturating_sub(MIN_BUCKET_LOG2)) as usize;
    let idx = idx.min(NUM_BUCKETS - 1);
    (idx, cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bucket_sizes_are_power_of_two() {
        let (_, c0) = bucket_of(1);
        let (_, c1) = bucket_of(4097);
        let (_, c2) = bucket_of(1 << 20);
        assert_eq!(c0, 1 << MIN_BUCKET_LOG2);
        assert_eq!(c1, 8192);
        assert_eq!(c2, 1 << 20);
    }
    #[test]
    fn buckets_grow_monotonically() {
        let (i_small, _) = bucket_of(4096);
        let (i_big, _) = bucket_of(1 << 24);
        assert!(i_big > i_small);
    }
    #[test]
    fn bucket_index_clamps_at_max() {
        let (idx, _) = bucket_of(1 << 40);
        assert_eq!(idx, NUM_BUCKETS - 1);
    }
}
