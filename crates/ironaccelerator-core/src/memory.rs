//! Vendor-neutral memory abstractions. Backends provide concrete `Allocation`
//! and `Stream` types; this module defines the trait surface and the
//! categories used by strategy selection.

use crate::error::Result;
use core::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum MemoryKind {
    /// Plain device-local memory.
    Device,
    /// Host-allocated, page-locked, zero-copy mappable.
    Pinned,
    /// Unified / managed memory.
    Unified,
    /// Constant memory (CUDA `__constant__`, equivalent on others).
    Constant,
    /// Shared / scratchpad (per-block / per-workgroup).
    Shared,
    /// Texture / surface memory.
    Texture,
}

/// A handle to backend-allocated memory. Concrete backends provide a
/// non-trait struct that derefs to a typed slice for the hot path.
pub trait Allocation: Send + Sync {
    fn kind(&self) -> MemoryKind;
    fn len_bytes(&self) -> usize;
    /// Raw device pointer. Backends that virtualise pointers (Metal, QNN)
    /// return their handle cast to `*mut u8`.
    fn as_ptr(&self) -> NonNull<u8>;
}

/// Pluggable allocator. Concrete backends register one default pool per
/// device (e.g. CUDA stream-ordered async allocator) plus optional pools for
/// pinned / unified memory.
pub trait MemoryPool: Send + Sync {
    fn kind(&self) -> MemoryKind;
    /// Allocate `bytes` aligned to `align`. Speed-over-safety: the returned
    /// allocation is *uninitialised*.
    fn alloc(&self, bytes: usize, align: usize) -> Result<Box<dyn Allocation>>;
    /// Optional: hint a deallocation watermark. No-op for backends without it.
    fn trim(&self) {}
}
