//! Stream-ordered allocation helpers.
//!
//! Thin sugar over [`crate::drv::DeviceBuf`] for the common case of allocating
//! a typed buffer on a session's stream. The underlying driver call is
//! `cuMemAllocAsync` — freeing happens on the same stream at drop.

use crate::drv::{DeviceBuf, Repr, Stream, ZeroBits};
use ironaccelerator_core::Result;
use std::sync::Arc;

/// Allocate `len` elements of `T` on `stream`'s allocation pool. Memory is
/// **uninitialised** — caller must populate it before reading.
#[inline(always)]
pub fn alloc<T: Repr>(stream: &Arc<Stream>, len: usize) -> Result<DeviceBuf<T>> {
    DeviceBuf::alloc(stream.clone(), len).map_err(Into::into)
}

/// Allocate `len` elements of `T` initialised to zero.
#[inline(always)]
pub fn alloc_zeros<T: ZeroBits>(stream: &Arc<Stream>, len: usize) -> Result<DeviceBuf<T>> {
    DeviceBuf::alloc_zeros(stream.clone(), len).map_err(Into::into)
}

/// Convenience: allocate and copy from a host slice in one step.
#[inline(always)]
pub fn from_host<T: Repr>(stream: &Arc<Stream>, host: &[T]) -> Result<DeviceBuf<T>> {
    DeviceBuf::from_host(stream.clone(), host).map_err(Into::into)
}
