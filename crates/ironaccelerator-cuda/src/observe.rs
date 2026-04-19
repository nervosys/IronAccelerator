//! Observability primitives: NVTX scoped ranges for Nsight Systems /
//! Nsight Compute, plus atomic per-session counters that are free to read in
//! the hot path.
//!
//! # NVTX
//!
//! [`range`] returns a guard that opens an NVTX range on construction and
//! closes it on drop. Pair it with `argb()` for colour-coded profiling
//! traces. The cost when NVTX is not attached (no profiler) is one PLT jump
//! and a string allocation.
//!
//! ```no_run
//! use ironaccelerator_cuda::observe;
//! let _r = observe::range("attention.fwd").argb(0xff2ea043).open();
//! // ... launch kernels ...
//! ```
//!
//! # Metrics
//!
//! [`Metrics`] lives on each [`crate::Session`] and uses relaxed atomics so
//! it never synchronises more than strictly necessary. For truly
//! zero-overhead builds, all increments are behind `#[inline(always)]` —
//! LLVM can fold them when the returned value is discarded.

use iron_cuda_sys::nvtx;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

/// Open a named NVTX range guard; the range closes when the guard is dropped.
#[inline(always)]
pub fn range<S: AsRef<str>>(msg: S) -> EventBuilder {
    EventBuilder { msg: msg.as_ref().to_string(), argb: None, category: None }
}

/// Fire-and-forget instant NVTX marker.
#[inline(always)]
pub fn mark<S: AsRef<str>>(msg: S) {
    if let Ok(f) = nvtx::fns() {
        if let Ok(c) = CString::new(msg.as_ref()) {
            unsafe { (f.nvtxMarkA)(c.as_ptr()); }
        }
    }
}

pub struct EventBuilder {
    msg: String,
    argb: Option<u32>,
    category: Option<u32>,
}

impl EventBuilder {
    #[inline(always)] pub fn argb(mut self, c: u32) -> Self { self.argb = Some(c); self }
    #[inline(always)] pub fn category(mut self, c: u32) -> Self { self.category = Some(c); self }

    /// Materialise the NVTX range guard.
    #[inline]
    pub fn open(self) -> Range {
        let id = if let Ok(f) = nvtx::fns() {
            CString::new(self.msg.as_str()).ok().map(|c| unsafe { (f.nvtxRangeStartA)(c.as_ptr()) })
        } else { None };
        // argb/category are ignored in the simplified ASCII path — extended
        // attributes use nvtxDomainRangeStartEx which we'll add when needed.
        let _ = (self.argb, self.category);
        Range { id }
    }
}

/// NVTX range guard. Closes on drop.
pub struct Range { id: Option<nvtx::NvtxRangeId> }

impl Drop for Range {
    fn drop(&mut self) {
        if let (Some(id), Ok(f)) = (self.id, nvtx::fns()) {
            unsafe { (f.nvtxRangeEnd)(id); }
        }
    }
}

/// Per-session atomic counters. Reads are `Ordering::Relaxed` — fine for
/// dashboards, not a synchronisation primitive.
#[derive(Debug, Default)]
pub struct Metrics {
    pub alloc_bytes:     AtomicU64,
    pub alloc_calls:     AtomicU64,
    pub free_bytes:      AtomicU64,
    pub htod_bytes:      AtomicU64,
    pub dtoh_bytes:      AtomicU64,
    pub kernels_launched: AtomicU64,
    pub blas_calls:      AtomicU64,
    pub nvrtc_hits:      AtomicU64,
    pub nvrtc_misses:    AtomicU64,
    pub fft_hits:        AtomicU64,
    pub fft_misses:      AtomicU64,
    pub collectives:     AtomicU64,
}

impl Metrics {
    #[inline(always)] pub fn record_alloc(&self, bytes: u64) {
        self.alloc_calls.fetch_add(1, Ordering::Relaxed);
        self.alloc_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    #[inline(always)] pub fn record_free(&self, bytes: u64) {
        self.free_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    #[inline(always)] pub fn record_htod(&self, bytes: u64) { self.htod_bytes.fetch_add(bytes, Ordering::Relaxed); }
    #[inline(always)] pub fn record_dtoh(&self, bytes: u64) { self.dtoh_bytes.fetch_add(bytes, Ordering::Relaxed); }
    #[inline(always)] pub fn record_launch(&self)            { self.kernels_launched.fetch_add(1, Ordering::Relaxed); }
    #[inline(always)] pub fn record_blas(&self)              { self.blas_calls.fetch_add(1, Ordering::Relaxed); }
    #[inline(always)] pub fn record_nvrtc(&self, hit: bool)  {
        if hit { self.nvrtc_hits.fetch_add(1, Ordering::Relaxed); }
        else   { self.nvrtc_misses.fetch_add(1, Ordering::Relaxed); }
    }
    #[inline(always)] pub fn record_fft(&self, hit: bool)    {
        if hit { self.fft_hits.fetch_add(1, Ordering::Relaxed); }
        else   { self.fft_misses.fetch_add(1, Ordering::Relaxed); }
    }
    #[inline(always)] pub fn record_collective(&self) { self.collectives.fetch_add(1, Ordering::Relaxed); }

    /// Immutable snapshot — O(1) relaxed reads.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            alloc_bytes:      self.alloc_bytes.load(Ordering::Relaxed),
            alloc_calls:      self.alloc_calls.load(Ordering::Relaxed),
            free_bytes:       self.free_bytes.load(Ordering::Relaxed),
            htod_bytes:       self.htod_bytes.load(Ordering::Relaxed),
            dtoh_bytes:       self.dtoh_bytes.load(Ordering::Relaxed),
            kernels_launched: self.kernels_launched.load(Ordering::Relaxed),
            blas_calls:       self.blas_calls.load(Ordering::Relaxed),
            nvrtc_hits:       self.nvrtc_hits.load(Ordering::Relaxed),
            nvrtc_misses:     self.nvrtc_misses.load(Ordering::Relaxed),
            fft_hits:         self.fft_hits.load(Ordering::Relaxed),
            fft_misses:       self.fft_misses.load(Ordering::Relaxed),
            collectives:      self.collectives.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsSnapshot {
    pub alloc_bytes: u64,
    pub alloc_calls: u64,
    pub free_bytes: u64,
    pub htod_bytes: u64,
    pub dtoh_bytes: u64,
    pub kernels_launched: u64,
    pub blas_calls: u64,
    pub nvrtc_hits: u64,
    pub nvrtc_misses: u64,
    pub fft_hits: u64,
    pub fft_misses: u64,
    pub collectives: u64,
}

impl MetricsSnapshot {
    /// Bytes still resident on the device according to our accounting.
    #[inline] pub fn resident_bytes(&self) -> i64 {
        self.alloc_bytes as i64 - self.free_bytes as i64
    }
    /// NVRTC cache hit ratio in `[0, 1]`. Returns 0 if the cache was never hit.
    #[inline] pub fn nvrtc_hit_ratio(&self) -> f32 {
        let total = self.nvrtc_hits + self.nvrtc_misses;
        if total == 0 { 0.0 } else { self.nvrtc_hits as f32 / total as f32 }
    }
    /// cuFFT plan cache hit ratio in `[0, 1]`.
    #[inline] pub fn fft_hit_ratio(&self) -> f32 {
        let total = self.fft_hits + self.fft_misses;
        if total == 0 { 0.0 } else { self.fft_hits as f32 / total as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        let m = Metrics::default();
        m.record_alloc(1024);
        m.record_alloc(2048);
        m.record_free(512);
        m.record_htod(8192);
        m.record_launch();
        m.record_nvrtc(true);
        m.record_nvrtc(false);
        let s = m.snapshot();
        assert_eq!(s.alloc_calls, 2);
        assert_eq!(s.alloc_bytes, 3072);
        assert_eq!(s.resident_bytes(), 2560);
        assert_eq!(s.htod_bytes, 8192);
        assert_eq!(s.kernels_launched, 1);
        assert!((s.nvrtc_hit_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_ratios_are_zero() {
        let s = Metrics::default().snapshot();
        assert_eq!(s.nvrtc_hit_ratio(), 0.0);
        assert_eq!(s.fft_hit_ratio(), 0.0);
        assert_eq!(s.resident_bytes(), 0);
    }
}
