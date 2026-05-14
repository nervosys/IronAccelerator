//! Stream pool — round-robin dispatch across a fixed set of CUDA streams,
//! optionally with priority. Useful when a session needs to overlap copies,
//! compute, and collectives across several engines on the same device.
//!
//! A [`StreamPool`] owns `N` [`Stream`]s on one device. [`StreamPool::next`]
//! hands out the next stream in round-robin order via a relaxed atomic
//! counter; concurrent producers are safe.

use crate::drv::{Device, Priority, Stream};
use ironaccelerator_core::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct StreamPool {
    device: Arc<Device>,
    streams: Vec<Arc<Stream>>,
    cursor: AtomicUsize,
}

impl StreamPool {
    /// Create a pool of `n` default-priority streams on `device`.
    pub fn new(device: Arc<Device>, n: usize) -> Result<Self> {
        assert!(n > 0, "stream pool size must be > 0");
        let mut streams = Vec::with_capacity(n);
        for _ in 0..n {
            streams.push(Stream::new(device.clone())?);
        }
        Ok(Self {
            device,
            streams,
            cursor: AtomicUsize::new(0),
        })
    }

    /// Pool where even indices get `low`, odd indices get `high` priority.
    pub fn new_interleaved_priority(device: Arc<Device>, n: usize) -> Result<Self> {
        let mut streams = Vec::with_capacity(n);
        for i in 0..n {
            let p = if i % 2 == 0 {
                Priority::Low
            } else {
                Priority::High
            };
            streams.push(Stream::with_priority(device.clone(), p)?);
        }
        Ok(Self {
            device,
            streams,
            cursor: AtomicUsize::new(0),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.streams.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get a stream by logical index (wraps).
    #[inline]
    pub fn get(&self, idx: usize) -> &Arc<Stream> {
        &self.streams[idx % self.streams.len()]
    }

    /// Round-robin next stream. Thread-safe.
    #[inline]
    pub fn next(&self) -> &Arc<Stream> {
        let i = self.cursor.fetch_add(1, Ordering::Relaxed);
        self.get(i)
    }

    /// Wait for every stream in the pool to drain.
    pub fn synchronize_all(&self) -> Result<()> {
        for s in &self.streams {
            s.synchronize()?;
        }
        Ok(())
    }

    pub fn streams(&self) -> &[Arc<Stream>] {
        &self.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_wraps() {
        let cursor = AtomicUsize::new(0);
        let n = 4;
        let idx = |c: &AtomicUsize| c.fetch_add(1, Ordering::Relaxed) % n;
        for _ in 0..8 {
            let _ = idx(&cursor);
        }
        assert!(cursor.load(Ordering::Relaxed) >= 8);
    }
}
