//! Async memcpy helpers that go through [`Session`] so every byte moved is
//! reflected in the per-session [`Metrics`](crate::observe::Metrics).
//!
//! All directions are stream-ordered: they enqueue the DMA on the session's
//! stream and return immediately. Callers that need completion must
//! `session.synchronize()`.

use crate::drv::{DeviceBuf, Repr};
use crate::{CudaTensor, Session};
use ironaccelerator_core::{Error, Result};

/// Host → device.
#[inline]
pub fn htod<T: Repr>(session: &Session, src: &[T], dst: &mut DeviceBuf<T>) -> Result<()> {
    dst.copy_from_host(src)?;
    session.metrics().record_htod(std::mem::size_of_val(src) as u64);
    Ok(())
}

/// Device → host.
#[inline]
pub fn dtoh<T: Repr>(session: &Session, src: &DeviceBuf<T>, dst: &mut [T]) -> Result<()> {
    src.copy_to_host(dst)?;
    session.metrics().record_dtoh(std::mem::size_of_val(dst) as u64);
    Ok(())
}

/// Device → device (same context).
#[inline]
pub fn dtod<T: Repr>(_session: &Session, src: &DeviceBuf<T>, dst: &mut DeviceBuf<T>) -> Result<()> {
    dst.copy_from_device(src)?;
    Ok(())
}

/// Byte-level tensor → tensor copy (same shape).
pub fn copy_tensor(_session: &Session, src: &CudaTensor, dst: &mut CudaTensor) -> Result<()> {
    if src.bytes() != dst.bytes() {
        return Err(Error::InvalidArgument("copy_tensor: size mismatch"));
    }
    dst.raw_mut().copy_from_device(src.raw())?;
    Ok(())
}

/// Read a tensor back to a host `Vec<u8>`. Blocking — calls `synchronize`.
pub fn to_host_bytes(session: &Session, src: &CudaTensor) -> Result<Vec<u8>> {
    let mut out = vec![0u8; src.bytes() as usize];
    src.raw().copy_to_host(&mut out)?;
    session.synchronize()?;
    session.metrics().record_dtoh(src.bytes());
    Ok(out)
}
