//! Bridge between [`ironaccelerator_core::TensorDesc`] and a live
//! `DeviceBuf<u8>` allocated on a session's stream.
//!
//! `CudaTensor` is byte-typed at the storage level — dtypes are tracked in
//! the descriptor, and views into typed slices are obtained lazily via
//! [`CudaTensor::as_view`]. This avoids forcing a generic `T` through every
//! planner / launch / cache path.

use crate::Session;
use crate::drv::{DeviceBuf, DeviceView, DeviceViewMut, Repr};
use iron_cuda_sys::driver::CUdeviceptr;
use ironaccelerator_core::{DType, Error, Layout, Result, TensorDesc};

pub struct CudaTensor {
    desc: TensorDesc,
    storage: DeviceBuf<u8>,
}

impl CudaTensor {
    /// Allocate a dense, uninitialised tensor on `session`'s stream.
    pub fn new(session: &Session, desc: TensorDesc) -> Result<Self> {
        let bytes = desc.bytes() as usize;
        let storage = DeviceBuf::<u8>::alloc(session.stream().clone(), bytes)?;
        session.metrics().record_alloc(bytes as u64);
        Ok(Self { desc, storage })
    }

    /// Allocate and zero-initialise.
    pub fn zeros(session: &Session, desc: TensorDesc) -> Result<Self> {
        let bytes = desc.bytes() as usize;
        let storage = DeviceBuf::<u8>::alloc_zeros(session.stream().clone(), bytes)?;
        session.metrics().record_alloc(bytes as u64);
        Ok(Self { desc, storage })
    }

    /// Allocate and copy from a host slice. Caller must ensure the host
    /// buffer matches the descriptor's byte size.
    pub fn from_host_bytes(session: &Session, desc: TensorDesc, host: &[u8]) -> Result<Self> {
        if host.len() as u64 != desc.bytes() {
            return Err(Error::InvalidArgument("from_host_bytes: size mismatch"));
        }
        let storage = DeviceBuf::<u8>::from_host(session.stream().clone(), host)?;
        let b = desc.bytes();
        session.metrics().record_alloc(b);
        session.metrics().record_htod(b);
        Ok(Self { desc, storage })
    }

    /// Convenience: allocate a 2-D matrix `(rows × cols)` of `dtype`.
    pub fn matrix(session: &Session, rows: u32, cols: u32, dtype: DType) -> Result<Self> {
        Self::new(session, TensorDesc {
            dtype, shape: vec![rows, cols], strides: None, layout: Layout::RowMajor,
        })
    }

    #[inline] pub fn desc(&self) -> &TensorDesc { &self.desc }
    #[inline] pub fn dtype(&self) -> DType { self.desc.dtype }
    #[inline] pub fn shape(&self) -> &[u32] { &self.desc.shape }
    #[inline] pub fn numel(&self) -> u64 { self.desc.numel() }
    #[inline] pub fn bytes(&self) -> u64 { self.desc.bytes() }
    #[inline] pub fn raw(&self) -> &DeviceBuf<u8> { &self.storage }
    #[inline] pub fn raw_mut(&mut self) -> &mut DeviceBuf<u8> { &mut self.storage }
    #[inline] pub fn device_ptr(&self) -> CUdeviceptr { self.storage.device_ptr() }

    /// View the raw bytes as a typed slice. Caller asserts that `T` matches
    /// the descriptor's dtype.
    ///
    /// # Safety
    /// `T`'s in-memory representation must match `self.dtype()`. Length is
    /// derived from the descriptor's element count.
    #[inline]
    pub unsafe fn as_view<T: Repr>(&self) -> Result<DeviceView<'_, T>> {
        let want = self.desc.numel() as usize * std::mem::size_of::<T>();
        if want != self.storage.byte_len() {
            return Err(Error::InvalidArgument("CudaTensor::as_view size mismatch"));
        }
        // SAFETY: lifetime & byte count validated above; T:Repr is POD.
        Ok(unsafe { std::mem::transmute::<DeviceView<'_, u8>, DeviceView<'_, T>>(self.storage.view()) })
    }

    /// Mutable typed view.
    ///
    /// # Safety
    /// As [`CudaTensor::as_view`].
    #[inline]
    pub unsafe fn as_view_mut<T: Repr>(&mut self) -> Result<DeviceViewMut<'_, T>> {
        let want = self.desc.numel() as usize * std::mem::size_of::<T>();
        if want != self.storage.byte_len() {
            return Err(Error::InvalidArgument("CudaTensor::as_view_mut size mismatch"));
        }
        Ok(unsafe {
            std::mem::transmute::<DeviceViewMut<'_, u8>, DeviceViewMut<'_, T>>(self.storage.view_mut())
        })
    }
}
