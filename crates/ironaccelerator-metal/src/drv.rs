//! Safe Metal driver layer. Apple-only — the module is empty on other hosts
//! so the non-Apple scaffold keeps building.

use ironaccelerator_core::{Error, Result};
use metal::{Buffer as MtlBuffer, CommandQueue, Device as MtlDevice, MTLResourceOptions};
use std::sync::Arc;

/// Thin wrapper around an `MTLDevice`.
pub struct Device {
    inner: MtlDevice,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Enumerate all Metal devices on the host.
    pub fn all() -> Vec<Arc<Self>> {
        MtlDevice::all()
            .into_iter()
            .map(|inner| Arc::new(Self { inner }))
            .collect()
    }

    pub fn system_default() -> Option<Arc<Self>> {
        MtlDevice::system_default().map(|inner| Arc::new(Self { inner }))
    }

    #[inline]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
    #[inline]
    pub fn registry_id(&self) -> u64 {
        self.inner.registry_id()
    }

    /// `MTLGPUFamily::Apple{N}` → `N` if the device is Apple Silicon, else 0.
    pub fn apple_family(&self) -> u32 {
        use metal::MTLGPUFamily::*;
        for (family, n) in [
            (Apple9, 9),
            (Apple8, 8),
            (Apple7, 7),
            (Apple6, 6),
            (Apple5, 5),
            (Apple4, 4),
            (Apple3, 3),
        ] {
            if self.inner.supports_family(family) {
                return n;
            }
        }
        0
    }

    #[inline]
    pub fn has_unified_memory(&self) -> bool {
        self.inner.has_unified_memory()
    }
    #[inline]
    pub fn max_buffer_length(&self) -> u64 {
        self.inner.max_buffer_length()
    }
    #[inline]
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.inner.recommended_max_working_set_size()
    }

    #[inline]
    pub fn raw(&self) -> &MtlDevice {
        &self.inner
    }

    pub fn new_queue(self: &Arc<Self>) -> Arc<Queue> {
        let q = self.inner.new_command_queue();
        Arc::new(Queue {
            device: self.clone(),
            inner: q,
        })
    }

    pub fn new_buffer(self: &Arc<Self>, bytes: usize, shared: bool) -> Result<Arc<Buffer>> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("new_buffer: size is zero"));
        }
        let opts = if shared {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let buf = self.inner.new_buffer(bytes as u64, opts);
        Ok(Arc::new(Buffer {
            _device: self.clone(),
            inner: buf,
            bytes,
        }))
    }
}

pub struct Queue {
    device: Arc<Device>,
    inner: CommandQueue,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl Queue {
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    #[inline]
    pub fn raw(&self) -> &CommandQueue {
        &self.inner
    }

    /// Create and commit a one-shot command buffer. Caller populates it inside
    /// `f`; the buffer is committed and awaited on return.
    pub fn scope<F: FnOnce(&metal::CommandBufferRef)>(&self, f: F) {
        let cb = self.inner.new_command_buffer();
        f(cb);
        cb.commit();
        cb.wait_until_completed();
    }
}

pub struct Buffer {
    _device: Arc<Device>,
    inner: MtlBuffer,
    bytes: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    #[inline]
    pub fn raw(&self) -> &MtlBuffer {
        &self.inner
    }
    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Host-visible pointer for `StorageModeShared` buffers. `None` on
    /// private-storage buffers.
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.inner.contents()
    }
}
